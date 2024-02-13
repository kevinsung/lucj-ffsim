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
NE, NORB = 10, 8
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
    DL = mf_ATOM.mo_coeff[:, [1, 2, 3, 4]]
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

for j, d in enumerate(list(np.arange(0.80, 3.01, 0.10))):
    filename = os.path.join(DATA_DIR, f"{BASE_NAME}_d-{d:.2f}.pickle")

    mol = gto.Mole()
    mol.atom = [["N", (0, 0, 0)], ["N", (d * Req, 0, 0)]]
    mol.basis = BASIS
    mol.symmetry = "Dooh"
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

    # if j in [15, 16, 17, 18, 19]:
    #     this_filename = os.path.join(DATA_DIR, f"{BASE_NAME}_d_2.70.pickle")
    #     with open(this_filename, "rb") as f:
    #         this_mol_data = pickle.load(f)
    #         t1 = this_mol_data.ccsd_t1
    #         t2 = this_mol_data.ccsd_t2

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
    D = np.zeros((nao, 8))
    D[: nao // 2, :4] = C
    D[nao // 2 :, 4:] = C

    PROJ, IDX = get_projections(mf, get_MP2_natural_orbitals(mf), D)
    print("bondlength, projections ", d, PROJ[IDX], IDX)

    norb = 8
    nelec = 5, 5
    mycas = mcscf.CASCI(mf, norb, nelec)
    mycas.mo_coeff = get_MP2_natural_orbitals(mf)
    mo = mycas.sort_mo(IDX, base=0)
    mycas.fix_spin_(ss=0)
    Ecas = mycas.kernel(mo)[0]

    h1e_cas, ecore = mycas.get_h1eff()
    h2e_cas = mycas.get_h2eff()

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
        hf_energy=mf.e_tot,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        mp2_energy=mmp.e_tot,
        ccsd_energy=mcc.e_tot,
        ccsd_t1=t1,
        ccsd_t2=t2,
        fci_energy=Ecas,
    )

    with open(filename, "wb") as f:
        pickle.dump(mol_data, f)
