import os
import ffsim
import numpy as np
from pyscf import cc, gto, mcscf, mp, scf
from pyscf.data import elements
import pickle
import scipy.linalg

DATA_ROOT = "/disk1/kevinsung@ibm.com/lucj-ffsim"

DATA_DIR = os.path.join(DATA_ROOT, "molecular_data")
BASIS = "sto-6g"
NE, NORB = 6, 6
BASE_NAME = f"nitrogen_dissociation_{BASIS}_{NE}e{NORB}o"


def find_twop_shell():
    ATOM = gto.Mole()
    ATOM.build(
        verbose=0,
        atom=[["N", (0, 0, 0)]],
        basis=BASIS,
        spin=3,
        charge=0,
        symmetry="c1",
    )
    mf_ATOM = scf.ROHF(ATOM)
    mf_ATOM = scf.newton(mf_ATOM)
    mf_ATOM.kernel()
    DL = mf_ATOM.mo_coeff[:, [2, 3, 4]]
    return DL


def get_MP2_natural_orbitals(mf):
    mp2 = mp.MP2(mf)
    mp2.kernel()
    rho = mp2.make_rdm1()
    n_mp, UU = np.linalg.eigh(rho)
    idx = n_mp.argsort()[::-1]
    n_mp = n_mp[idx]
    UU = UU[:, idx]
    C_mp = np.dot(mf.mo_coeff, UU)
    return C_mp[:, :]


def get_projections(mf, C, D):
    # projections of C on D
    from scipy import linalg as LA

    Omega = np.einsum("pi,pq,qj->ij", D, mf.get_ovlp(), D)
    s, U = LA.eigh(Omega)
    D = np.einsum("pi,ij,j->pj", D, U, 1.0 / np.sqrt(s))
    Omega = np.einsum("pi,pq,qj->ij", D, mf.get_ovlp(), D)
    INVS = LA.inv(Omega)
    P = np.einsum("pi,ij,qj,qr->pr", D, INVS, D, mf.get_ovlp())
    PROJ = np.einsum("pi,pr,rq,qi->i", C, mf.get_ovlp(), P, C)
    IDX = sorted(range(len(PROJ)), key=lambda x: PROJ[x])[::-1][: D.shape[1]]
    IDX = sorted(IDX)
    return PROJ, IDX


# ----------------------------------------------------------------------------

C = find_twop_shell()
Req = 1.0975135

dm_scf = None
t1 = None
t2 = None
t1_as = None
t2_as = None

d_range = np.arange(0.80, 3.01, 0.10)
# d_range = np.arange(0.80, 1.11, 0.10)
for j, d in enumerate(d_range):
    filename = os.path.join(DATA_DIR, f"{BASE_NAME}_d-{d:.2f}.pickle")

    mol = gto.Mole()
    mol.atom = [["N", (0, 0, 0)], ["N", (d * Req, 0, 0)]]
    mol.basis = BASIS
    mol.symmetry = "c1"
    mol.verbose = 4
    mol.build()

    # Hartree Fock
    mf = scf.RHF(mol)
    mf = scf.newton(mf)
    mf.kernel()
    mf.analyze()
    assert mf.converged

    frz = elements.chemcore(mol)
    mmp = mp.MP2(mf)
    mmp.frozen = frz
    dEmp = mmp.kernel()[0]
    mcc = cc.CCSD(mf)
    mcc.frozen = frz

    if t2 is not None:
        Cv, Co = (
            scipy.linalg.inv(mf.mo_coeff)[mol.nelectron // 2 :, :],
            scipy.linalg.inv(mf.mo_coeff)[frz : mol.nelectron // 2, :],
        )
        t1 = np.einsum("pr,cp,kr->kc", t1, Cv, Co, optimize=True)
        t2 = np.einsum("prqs,cp,dq,ls,kr->klcd", t2, Cv, Cv, Co, Co, optimize=True)
    dEcc = mcc.kernel(t1, t2)[0]
    Co, Cv = (
        mf.mo_coeff[:, frz : mol.nelectron // 2],
        mf.mo_coeff[:, mol.nelectron // 2 :],
    )
    t1 = np.einsum("ia,pa,ri->pr", mcc.t1, Cv, Co, optimize=True)
    t2 = np.einsum("ijab,pa,qb,sj,ri->prqs", mcc.t2, Cv, Cv, Co, Co, optimize=True)

    nao = mol.nao_nr()
    D = np.zeros((nao, 6))
    D[: nao // 2, :3] = C
    D[nao // 2 :, 3:] = C

    PROJ, IDX = get_projections(mf, get_MP2_natural_orbitals(mf), D)
    print("bondlength, projections ", d, PROJ[IDX], IDX)

    norb = 6
    nelec = 3, 3
    mycas = mcscf.CASCI(mf, norb, nelec)
    mycas.mo_coeff = get_MP2_natural_orbitals(mf)
    # mycas.mo_coeff = mycas.sort_mo(IDX, base=0)
    if d > 2.00:
        mo = mycas.sort_mo(IDX, base=0)
    else:
        mo = None
    mycas.fix_spin_(ss=0)
    Ecas = mycas.kernel(mo)[0]

    h1e_cas, ecore = mycas.get_h1eff()
    h2e_cas = mycas.get_h2eff()

    ########

    mol = gto.M(verbose=0)
    mol.nelectron = sum(nelec)
    mol.spin = nelec[0] - nelec[1]
    mol.incore_anyway = True
    mol.nao_nr = lambda *args: norb
    mol.energy_nuc = lambda *args: ecore
    mf_as = scf.RHF(mol)
    mf_as.get_hcore = lambda *args: h1e_cas
    mf_as.get_ovlp = lambda *args: np.eye(norb)
    mf_as._eri = h2e_cas
    dm0 = np.zeros((mol.nao_nr(), mol.nao_nr()))
    na = (mol.nelectron + mol.spin) // 2
    nb = (mol.nelectron - mol.spin) // 2
    no = mf_as.get_ovlp().shape[0]
    for i in range(na):
        dm0[i, i] = 2
    mf_as = scf.newton(mf_as)
    EHF = mf_as.kernel(dm0=dm0)
    for _ in range(10):
        moi, moe, si, se = mf_as.stability(return_status=True)
        E = mf_as.kernel(mf_as.make_rdm1(moi, mf_as.mo_occ))

    cc_as = cc.CCSD(mf_as)
    cc_as.kernel(t1_as, t2_as)
    t1_as = cc_as.t1
    t2_as = cc_as.t2

    mycas = mcscf.CASCI(mf_as, norb, nelec)
    # mycas.fix_spin_(ss=0)
    # Ecas = mycas.kernel(mo)[0]

    h1e_cas, ecore = mycas.get_h1eff()
    h2e_cas = mycas.get_h2eff()

    ########

    mol_data = ffsim.MolecularData(
        atom=mol.atom,
        basis=mol.basis,
        spin=mol.spin,
        symmetry=mol.symmetry,
        norb=norb,
        nelec=nelec,
        active_space=IDX,
        core_energy=ecore,
        one_body_integrals=h1e_cas,
        two_body_integrals=h2e_cas,
        hf_energy=mf_as.e_tot,
        mo_coeff=mf_as.mo_coeff,
        mo_occ=mf_as.mo_occ,
        mp2_energy=mmp.e_tot,
        ccsd_energy=cc_as.e_tot,
        ccsd_t1=t1_as,
        ccsd_t2=t2_as,
        fci_energy=Ecas,
    )

    with open(filename, "wb") as f:
        pickle.dump(mol_data, f)
