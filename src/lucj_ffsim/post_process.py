import itertools
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import ffsim
import numpy as np
from pyscf.fci import spin_square

from lucj_ffsim.lucj import LUCJTask

MOL_DATA_DIR = "data/molecular_data"
DATA_DIR = "data"
MAX_PROCESSES = 24


def process_result(task: LUCJTask, overwrite: bool = False):
    out_filename = os.path.join(DATA_DIR, task.dirname, "data.pickle")
    if (not overwrite) and os.path.exists(out_filename):
        print(f"{out_filename} already exists. Skipping...\n")
        return

    result_filename = os.path.join(DATA_DIR, task.dirname, "result.pickle")
    if not os.path.exists(result_filename):
        print(f"{result_filename} does not exist. Skipping...\n")
        return
    with open(result_filename, "rb") as f:
        result = pickle.load(f)

    mol_filename = os.path.join(MOL_DATA_DIR, f"{task.molecule_basename}.pickle")
    with open(mol_filename, "rb") as f:
        mol_data = pickle.load(f)

    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)

    if task.connectivity == "all-to-all":
        alpha_alpha_indices = None
        alpha_beta_indices = None
    elif task.connectivity == "square":
        alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
        alpha_beta_indices = [(p, p) for p in range(norb)]
    elif task.connectivity == "hex":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid connectivity: {task.connectivity}")

    operator = ffsim.UCJOperator.from_parameters(
        result.x,
        norb=norb,
        n_reps=task.n_reps,
        alpha_alpha_indices=alpha_alpha_indices,
        alpha_beta_indices=alpha_beta_indices,
        with_final_orbital_rotation=task.with_final_orbital_rotation,
    )
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    energy = np.vdot(final_state, hamiltonian @ final_state).real
    np.testing.assert_allclose(energy, result.fun)

    error = energy - mol_data.fci_energy

    spin_squared, multiplicity = spin_square(
        final_state, norb=mol_data.norb, nelec=mol_data.nelec
    )
    np.testing.assert_allclose(
        multiplicity, 2 * (math.sqrt(spin_squared + 0.25) - 0.5) + 1
    )

    data = {
        "energy": energy,
        "error": error,
        "spin_squared": spin_squared,
        "nit": result.nit,
        "nfev": result.nfev,
    }
    if task.optimization_method == "linear-method":
        data["nlinop"] = result.nlinop
    else:
        data["nlinop"] = None
    with open(out_filename, "wb") as f:
        pickle.dump(data, f)


def main():
    basis = "sto-6g"
    ne, norb = 4, 4

    bond_distance_range = np.linspace(1.3, 4.0, 6)
    connectivities = ["all-to-all", "square"]
    n_reps_range = [2, 4, 6]
    with_final_orbital_rotation_choices = [False, True]
    optimization_methods = [
        "L-BFGS-B",
        "linear-method",
    ]
    ansatz_settings = [
        # connectivity, n_reps, with_final_orbital_rotation, param_initialization, optimization_method
        (connectivity, n_reps, with_final_orbital_rotation, "ccsd", optimization_method)
        for connectivity, n_reps, with_final_orbital_rotation, optimization_method in itertools.product(
            connectivities,
            n_reps_range,
            with_final_orbital_rotation_choices,
            optimization_methods,
        )
    ]

    molecule_basename = f"ethene_dissociation_{basis}_{ne}e{norb}o"
    with ProcessPoolExecutor(MAX_PROCESSES) as executor:
        for (
            connectivity,
            n_reps,
            with_final_orbital_rotation,
            param_initialization,
            optimization_method,
        ) in ansatz_settings:
            for bond_distance in bond_distance_range:
                task = LUCJTask(
                    molecule_basename=f"{molecule_basename}_bond_distance_{bond_distance:.5f}",
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=with_final_orbital_rotation,
                    param_initialization=param_initialization,
                    optimization_method=optimization_method,
                )
                executor.submit(process_result, task, overwrite=True)


if __name__ == "__main__":
    main()
