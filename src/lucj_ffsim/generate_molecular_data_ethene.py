import itertools
import os
import pickle

import ffsim
import numpy as np
import pyscf

DATA_DIR = "data/molecular_data"
BASIS = "sto-6g"
NE, NORB = 4, 4
BASE_NAME = f"ethene_dissociation_{BASIS}_{NE}e{NORB}o"

os.makedirs(DATA_DIR, exist_ok=True)

reference_curves_bond_distance_range = np.linspace(1.3, 4.0, 50)
other_points_of_interest = [1.339]
experiment_bond_distance_range_0 = np.linspace(1.3, 4.0, 6)
experiment_bond_distance_range_1 = np.linspace(1.3, 4.0, 20)

for bond_distance in itertools.chain(
    reference_curves_bond_distance_range,
    other_points_of_interest,
    experiment_bond_distance_range_0,
    experiment_bond_distance_range_1,
):
    filename = os.path.join(
        DATA_DIR, f"{BASE_NAME}_bond_distance_{bond_distance:.5f}.pickle"
    )
    if os.path.exists(filename):
        continue

    a = 0.5 * bond_distance
    b = a + 0.5626
    c = 0.9289
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[
            ["C", (0, 0, a)],
            ["C", (0, 0, -a)],
            ["H", (0, c, b)],
            ["H", (0, -c, b)],
            ["H", (0, c, -b)],
            ["H", (0, -c, -b)],
        ],
        basis="sto-6g",
        symmetry="d2h",
    )

    # Define active space
    active_space = range(mol.nelectron // 2 - 2, mol.nelectron // 2 + 2)

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    mol_data = ffsim.MolecularData.from_mole(
        mol, active_space=active_space, mp2=True, ccsd=True, fci=True
    )

    with open(filename, "wb") as f:
        pickle.dump(mol_data, f)
