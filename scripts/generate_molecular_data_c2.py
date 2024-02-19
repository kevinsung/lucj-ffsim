import os
import pickle

import ffsim
import numpy as np
import pyscf
from pyscf import gto, mcscf, scf, symm

DATA_ROOT = "/disk1/kevinsung@ibm.com/lucj-ffsim"

DATA_DIR = os.path.join(DATA_ROOT, "molecular_data")
BASIS = "sto-6g"
NE, NORB = 8, 8
BASE_NAME = f"c2_dissociation_{BASIS}_{NE}e{NORB}o"


def transport(t1, t2, mf_old, mf_new, mo_new=None):
    from scipy import linalg as LA

    occ_old = mf_old.mo_coeff[:, mf_old.mo_occ > 0]
    vir_old = mf_old.mo_coeff[:, mf_old.mo_occ == 0]
    t1 = np.einsum("ia,pa,ri->pr", t1, vir_old, occ_old, optimize=True)
    t2 = np.einsum(
        "ijab,pa,qb,sj,ri->prqs", t2, vir_old, vir_old, occ_old, occ_old, optimize=True
    )
    if mo_new is None:
        occ_new = LA.inv(mf_new.mo_coeff)[mf_new.mo_occ > 0, :]
        vir_new = LA.inv(mf_new.mo_coeff)[mf_new.mo_occ == 0, :]
    else:
        occ_new = LA.inv(mo_new)[mf_old.mo_occ > 0, :]
        vir_new = LA.inv(mo_new)[mf_old.mo_occ == 0, :]
    t1 = np.einsum("pr,cp,kr->kc", t1, vir_new, occ_new, optimize=True)
    t2 = np.einsum(
        "prqs,cp,dq,ls,kr->klcd", t2, vir_new, vir_new, occ_new, occ_new, optimize=True
    )
    return t1, t2


def enforce_smoothness(orb_old, orb_new, mf_old, mf_new):
    s12 = gto.mole.intor_cross("int1e_ovlp_sph", mf_old.mol, mf_new.mol)
    print(s12.shape, orb_old.shape, orb_new.shape)
    w12 = np.einsum("pX,pq,qY->XY", orb_old, s12, orb_new, optimize=True)
    print("PERMUTATION ", np.diag(w12))
    w12 = np.abs(w12)
    idx = [np.argmax(w12[i, :]) for i in range(w12.shape[0])]
    print("PERMUTATION ", idx)
    orb_new = orb_new[:, idx]
    w12 = np.einsum("pX,pq,qY->XY", orb_old, s12, orb_new, optimize=True)
    for i in range(w12.shape[0]):
        if w12[i, i] < 0:
            orb_new[:, i] *= -1
    w12 = np.einsum("pX,pq,qY->XY", orb_old, s12, orb_new, optimize=True)
    print("PERMUTATION ", np.diag(w12))
    return orb_new


def get_active_space_info(mymf, mycas, occ_inactive=0, vir_active=0, calc={}):
    h1e_cas, ecore = mycas.get_h1eff()
    h2e_cas = mycas.get_h2eff()
    mol_as = gto.M(verbose=0)
    mol_as.nelectron = sum(mycas.nelecas)
    mol_as.spin = mycas.nelecas[0] - mycas.nelecas[1]
    mol_as.incore_anyway = True
    mol_as.nao_nr = lambda *args: mycas.ncas
    mol_as.energy_nuc = lambda *args: ecore
    mf_as = scf.RHF(mol_as)
    mf_as.get_hcore = lambda *args: h1e_cas
    mf_as.get_ovlp = lambda *args: np.eye(mycas.ncas)
    mf_as._eri = h2e_cas
    dm0 = np.zeros((mycas.ncas, mycas.ncas))
    na = (mol_as.nelectron + mol_as.spin) // 2
    mf_as.get_ovlp().shape[0]
    for i in range(na):
        dm0[i, i] = 2
    mf_as = scf.newton(mf_as)
    mf_as.kernel(dm0=dm0)
    mf_as.mo_coeff = np.eye(mycas.ncas)
    cc_as = pyscf.cc.RCCSD(mf_as)
    t1_as, t2_as = transport(t1, t2, mymf, mf_new=None, mo_new=mycas.mo_coeff)
    cc_as.kernel(
        t1_as[occ_inactive:, :vir_active],
        t2_as[occ_inactive:, occ_inactive:, :vir_active, :vir_active],
    )
    calc["e_as_ccsd"] = cc_as.e_tot
    calc["s_as_ccsd"] = 0
    calc["t1_as"] = cc_as.t1
    calc["t2_as"] = cc_as.t2
    calc["ecore"] = ecore
    calc["h1e_cas"] = h1e_cas
    calc["h2e_cas"] = h2e_cas


rho = None
t1 = None
t2 = None
orb = None
ci0 = None
dx = 0.05

dl = [0.90 + dx * x for x in range(1000) if 0.80 + dx * x < 3.0 + 1e-4]
dl = dl + dl[::-1] + dl

mf_list = []
for idx_calc, d in enumerate(dl):
    calc = {"R": d}

    mol = pyscf.gto.Mole()
    mol.atom = [["C", (-d / 2.0, 0, 0)], ["C", (d / 2.0, 0, 0)]]
    mol.basis = BASIS
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 4
    mol.symmetry = "Dooh"
    mol.build()
    n = mol.nao_nr()

    # Hartree Fock
    mf = pyscf.scf.RHF(mol)
    mf.irrep_nelec = {"A1g": 4, "A1u": 4, "E1ux": 2, "E1uy": 2}
    mf.kernel(rho)
    for f in range(3):
        mo1 = mf.stability()[0]
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mf.stability()
    mf.analyze()
    rho = mf.make_rdm1()
    e_scf = mf.e_tot if mf.converged else np.nan
    s_scf = 0.0 if mf.converged else np.nan
    calc["E_scf"] = e_scf
    calc["S_scf"] = s_scf
    calc["C_scf"] = mf.mo_coeff.copy()
    calc["Irr_scf"] = symm.label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff
    )
    calc["Eps_scf"] = mf.mo_energy
    calc["Occ_scf"] = mf.mo_occ
    mf_list.append(mf)
    print("IRREPS ", calc["Irr_scf"])

    # CCSD
    cc = pyscf.cc.RCCSD(mf)
    if t1 is not None:
        t1, t2 = transport(t1, t2, mf_list[len(mf_list) - 1], mf_list[len(mf_list) - 2])
    cc.kernel(t1, t2)
    t1, t2 = cc.t1, cc.t2
    e_ccsd = cc.e_tot if cc.converged else np.nan
    s_ccsd = 0.0 if cc.converged else np.nan
    calc["E_ccsd"] = e_ccsd
    calc["S_ccsd"] = s_ccsd
    calc["t1_ccsd"] = t1.copy()
    calc["t2_ccsd"] = t2.copy()

    # ----- CASCI initial orbitals
    norb = n - 2
    nelecas = mol.nelectron - 4
    nela, nelb = nelecas // 2, nelecas // 2

    cas = mcscf.RCASCI(mf, ncas=norb, nelecas=(nela, nela))
    cas.mo_coeff = mf.mo_coeff.copy()
    if orb is not None:
        print("PERMUTATION ", d)
        orb = enforce_smoothness(
            orb, cas.mo_coeff, mf_list[len(mf_list) - 2], mf_list[len(mf_list) - 1]
        )
        cas.mo_coeff = orb
    else:
        orb = cas.mo_coeff
    cas.fcisolver.wfnsym = "A1g"
    cas.fix_spin_(ss=0)
    cas.kernel(mo_coeff=orb, ci0=ci0)
    cas.mo_coeff = orb
    # ci0   = cas.ci
    e_cas = cas.e_tot
    s_cas = cas.fcisolver.spin_square(cas.ci, norb=cas.ncas, nelec=cas.nelecas)[0]
    calc["E_casci"] = e_cas
    calc["S_casci"] = s_cas
    calc["C_casci"] = cas.mo_coeff.copy()
    calc["V_casci"] = cas.ci.copy()
    calc["Irr_casci"] = symm.label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, cas.mo_coeff.copy()
    )

    # active-space SCF and CCSD
    get_active_space_info(mf, cas, occ_inactive=2, vir_active=norb - nela, calc=calc)
    e_as_ccsd = calc["e_as_ccsd"]
    s_as_ccsd = calc["s_as_ccsd"]

    norb = cas.ncas
    nelec = cas.nelecas
    nocc = NE // 2
    nvrt = norb - nocc
    assert norb == NORB
    assert sum(nelec) == NE
    assert nelec == (NE // 2, NE // 2)
    assert calc["h1e_cas"].shape == (norb, norb)
    print(calc["t1_as"].shape)
    print(calc["t2_as"].shape)
    assert calc["t1_as"].shape == (nocc, nvrt)
    assert calc["t2_as"].shape == (nocc, nocc, nvrt, nvrt)

    mol_data = ffsim.MolecularData(
        atom=mol.atom,
        basis=mol.basis,
        spin=mol.spin,
        symmetry=mol.symmetry,
        norb=norb,
        nelec=nelec,
        active_space=list(
            range(
                mol.nelectron // 2 - nelec[0],
                mol.nelectron // 2 - nelec[0] + norb,
            )
        ),
        core_energy=calc["ecore"],
        one_body_integrals=calc["h1e_cas"],
        two_body_integrals=calc["h2e_cas"],
        hf_energy=calc["E_scf"],
        mo_coeff=cas.mo_coeff,
        mo_occ=mf.mo_occ,
        ccsd_energy=calc["e_as_ccsd"],
        ccsd_t1=calc["t1_as"],
        ccsd_t2=calc["t2_as"],
        fci_energy=cas.e_tot,
    )

    filename = os.path.join(DATA_DIR, f"{BASE_NAME}_d-{d:.2f}.pickle")
    with open(filename, "wb") as f:
        pickle.dump(mol_data, f)
