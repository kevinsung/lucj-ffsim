import itertools
import os
import pickle
import timeit
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import ffsim
import numpy as np
import scipy.optimize

MOL_DATA_DIR = "data/molecular_data"
DATA_DIR = "data"
MAX_PROCESSES = 44


@dataclass(frozen=True)
class LUCJTask:
    molecule_basename: str
    connectivity: str  # options: all-to-all, square, hex
    n_reps: int
    with_final_orbital_rotation: bool
    param_initialization: str  # options: ccsd, bootstrap
    optimization_method: str

    @property
    def dirname(self) -> str:
        return os.path.join(
            self.molecule_basename,
            f"{self.connectivity}",
            f"n_reps-{self.n_reps}",
            f"with_final_orbital_rotation-{self.with_final_orbital_rotation}",
            f"param_initialization-{self.param_initialization}",
            f"optimization_method-{self.optimization_method}",
        )


def run_lucj_task(task: LUCJTask):
    print(f"{task} Starting...\n")
    os.makedirs(os.path.join(DATA_DIR, task.dirname), exist_ok=True)

    result_filename = os.path.join(DATA_DIR, task.dirname, "result.pickle")
    if os.path.exists(result_filename):
        print(f"{result_filename} already exists. Skipping...\n")
        return

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    molecule_filename = os.path.join(MOL_DATA_DIR, f"{task.molecule_basename}.pickle")
    with open(molecule_filename, "rb") as f:
        mol_data: ffsim.MolecularData = pickle.load(f)
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

    def fun(x: np.ndarray) -> float:
        # Initialize the ansatz operator from the parameter vector
        operator = ffsim.UCJOperator.from_parameters(
            x,
            norb=norb,
            n_reps=task.n_reps,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
        )
        # Compute energy
        final_state = ffsim.apply_unitary(
            reference_state, operator, norb=norb, nelec=nelec
        )
        return np.vdot(final_state, hamiltonian @ final_state).real

    def params_to_vec(x: np.ndarray) -> np.ndarray:
        operator = ffsim.UCJOperator.from_parameters(
            x,
            norb=norb,
            n_reps=task.n_reps,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
        )
        return ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    # generate initial parameters
    if task.param_initialization == "ccsd":
        params = ffsim.UCJOperator.from_t_amplitudes(
            mol_data.ccsd_t2, n_reps=task.n_reps
        ).to_parameters(
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
        )
    elif task.param_initialization == "bootstrap":
        raise NotImplementedError
    else:
        raise ValueError(
            f"Invalid parameter initialization strategy: {task.param_initialization}"
        )

    # optimize ansatz
    print(f"{task} Optimizing ansatz...\n")
    t0 = timeit.default_timer()
    if task.optimization_method == "L-BFGS-B":
        result = scipy.optimize.minimize(
            fun,
            x0=params,
            method="L-BFGS-B",
            options=dict(
                maxiter=100000,
                # eps=1e-12
            ),
        )
    elif task.optimization_method == "linear-method":
        result = ffsim.optimize.minimize_linear_method(
            params_to_vec,
            hamiltonian,
            x0=params,
            maxiter=100000,
        )
    elif task.optimization_method == "basinhopping":
        result = scipy.optimize.basinhopping(
            fun,
            x0=params,
            minimizer_kwargs=dict(
                method="L-BFGS-B",
                options=dict(
                    maxiter=100000,
                    # eps=1e-12
                ),
            ),
        )
    else:
        raise NotImplementedError
    t1 = timeit.default_timer()
    print(f"{task} Done in {t1 - t0} seconds.\n")

    with open(result_filename, "wb") as f:
        pickle.dump(result, f)


def main():
    basis = "sto-6g"
    ne, norb = 4, 4

    bond_distance_range = np.linspace(1.3, 4.0, 6)
    connectivities = ["all-to-all", "square"]
    n_reps_range = [2, 4]
    optimization_methods = ["L-BFGS-B", "linear-method"]
    ansatz_settings = [
        # connectivity, n_reps, with_final_orbital_rotation, param_initialization, optimization_method
        (connectivity, n_reps, False, "ccsd", optimization_method)
        for connectivity, n_reps, optimization_method in itertools.product(
            connectivities, n_reps_range, optimization_methods
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
                executor.submit(run_lucj_task, task).result()


if __name__ == "__main__":
    main()
