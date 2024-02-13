from __future__ import annotations

import logging
import math
import os
import pickle
import timeit
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass

import ffsim
import numpy as np
import scipy.optimize
from pyscf.fci import spin_square


@dataclass(frozen=True)
class LUCJTask:
    molecule_basename: str
    connectivity: str  # options: all-to-all, square, hex, heavy-hex
    n_reps: int
    with_final_orbital_rotation: bool
    optimization_method: str
    maxiter: int
    bootstrap_task: LUCJTask | None

    @property
    def dirname(self) -> str:
        return os.path.join(
            self.molecule_basename,
            f"{self.connectivity}",
            f"n_reps-{self.n_reps}",
            f"with_final_orbital_rotation-{self.with_final_orbital_rotation}",
            f"optimization_method-{self.optimization_method}",
            f"maxiter-{self.maxiter}",
        )


def _get_lucj_indices(
    connectivity: str, norb: int
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]:
    if connectivity == "all-to-all":
        alpha_alpha_indices = None
        alpha_beta_indices = None
    elif connectivity == "square":
        alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
        alpha_beta_indices = [(p, p) for p in range(norb)]
    elif connectivity == "hex":
        alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
        alpha_beta_indices = [(p, p) for p in range(norb) if p % 2 == 0]
    elif connectivity == "heavy-hex":
        alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
        alpha_beta_indices = [(p, p) for p in range(norb) if p % 4 == 0]
    else:
        raise ValueError(f"Invalid connectivity: {connectivity}")
    return alpha_alpha_indices, alpha_beta_indices


def run_lucj_task(
    task: LUCJTask,
    data_dir: str,
    mol_data_dir: str,
    overwrite: bool = True,
) -> LUCJTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(os.path.join(data_dir, task.dirname), exist_ok=True)

    result_filename = os.path.join(data_dir, task.dirname, "result.pickle")
    info_filename = os.path.join(data_dir, task.dirname, "info.pickle")
    if (
        (not overwrite)
        and os.path.exists(result_filename)
        and os.path.exists(info_filename)
    ):
        logging.info(f"Data for {task} already exists. Skipping...\n")
        return task

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    molecule_filename = os.path.join(mol_data_dir, f"{task.molecule_basename}.pickle")
    with open(molecule_filename, "rb") as f:
        mol_data: ffsim.MolecularData = pickle.load(f)
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)

    alpha_alpha_indices, alpha_beta_indices = _get_lucj_indices(task.connectivity, norb)

    def fun(x: np.ndarray) -> float:
        # Initialize the ansatz operator from the parameter vector
        operator = ffsim.UCJOperator.from_parameters(
            x,
            norb=norb,
            n_reps=task.n_reps,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
            with_final_orbital_rotation=task.with_final_orbital_rotation,
        )
        # Compute energy
        logging.debug("Computing statevector...")
        t0 = timeit.default_timer()
        final_state = ffsim.apply_unitary(
            reference_state, operator, norb=norb, nelec=nelec
        )
        t1 = timeit.default_timer()
        logging.debug(f"Computing statevector done in {t1 - t0} seconds.")
        logging.debug("Computing energy...")
        t0 = timeit.default_timer()
        energy = np.vdot(final_state, hamiltonian @ final_state).real
        t1 = timeit.default_timer()
        logging.debug(f"Computing energy done in {t1 - t0} seconds.")
        return energy

    def params_to_vec(x: np.ndarray) -> np.ndarray:
        operator = ffsim.UCJOperator.from_parameters(
            x,
            norb=norb,
            n_reps=task.n_reps,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
            with_final_orbital_rotation=task.with_final_orbital_rotation,
        )
        return ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    # generate initial parameters
    if task.bootstrap_task is None:
        # use CCSD to initialize parameters
        params = ffsim.UCJOperator.from_t_amplitudes(
            mol_data.ccsd_t2,
            n_reps=task.n_reps,
            t1_amplitudes=mol_data.ccsd_t1
            if task.with_final_orbital_rotation
            else None,
        ).to_parameters(
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
        )
    else:
        bootstrap_result_filename = os.path.join(
            data_dir, task.bootstrap_task.dirname, "result.pickle"
        )
        with open(bootstrap_result_filename, "rb") as f:
            result = pickle.load(f)
            params = result.x

    # optimize ansatz
    logging.info(f"{task} Optimizing ansatz...\n")
    info = defaultdict(list)
    info["nit"] = 0
    t0 = timeit.default_timer()
    if task.optimization_method == "L-BFGS-B":

        def callback(intermediate_result: scipy.optimize.OptimizeResult):
            info["x"].append(intermediate_result.x)
            info["fun"].append(intermediate_result.fun)
            info["nit"] += 1

        result = scipy.optimize.minimize(
            fun,
            x0=params,
            method="L-BFGS-B",
            options=dict(
                maxiter=task.maxiter,
                # eps=1e-12
            ),
            callback=callback,
        )
    elif task.optimization_method == "linear-method":

        def callback(intermediate_result: scipy.optimize.OptimizeResult):
            logging.info(f"Task {task} is on iteration {info['nit']}.")
            info["x"].append(intermediate_result.x)
            info["fun"].append(intermediate_result.fun)
            if hasattr(intermediate_result, "jac"):
                info["jac"].append(intermediate_result.jac)
            if hasattr(intermediate_result, "regularization"):
                info["regularization"].append(intermediate_result.regularization)
            if hasattr(intermediate_result, "variation"):
                info["variation"].append(intermediate_result.variation)
            nit = info["nit"]
            if nit < 100 or nit % 20 == 0:
                if hasattr(intermediate_result, "energy_mat"):
                    info["energy_mat"].append((nit, intermediate_result.energy_mat))
                if hasattr(intermediate_result, "overlap_mat"):
                    info["overlap_mat"].append((nit, intermediate_result.overlap_mat))
            info["nit"] += 1

        result = ffsim.optimize.minimize_linear_method(
            params_to_vec,
            hamiltonian,
            x0=params,
            maxiter=task.maxiter,
            callback=callback,
        )
    elif task.optimization_method == "basinhopping":
        result = scipy.optimize.basinhopping(
            fun,
            x0=params,
            minimizer_kwargs=dict(
                method="L-BFGS-B",
                options=dict(
                    maxiter=1000,
                    # eps=1e-12
                ),
            ),
        )
    else:
        raise NotImplementedError
    t1 = timeit.default_timer()
    logging.info(f"{task} Done in {t1 - t0} seconds.\n")

    with open(result_filename, "wb") as f:
        pickle.dump(result, f)

    with open(info_filename, "wb") as f:
        pickle.dump(info, f)

    process_result(
        task, data_dir=data_dir, mol_data_dir=mol_data_dir, overwrite=overwrite
    )

    return task


def process_result(
    future: Future | LUCJTask,
    data_dir: str,
    mol_data_dir: str,
    overwrite: bool = True,
):
    if isinstance(future, Future):
        task: LUCJTask = future.result()
    else:
        task: LUCJTask = future

    out_filename = os.path.join(data_dir, task.dirname, "data.pickle")
    if (not overwrite) and os.path.exists(out_filename):
        logging.info(f"{out_filename} already exists. Skipping...\n")
        return

    result_filename = os.path.join(data_dir, task.dirname, "result.pickle")
    if not os.path.exists(result_filename):
        logging.info(f"{result_filename} does not exist. Skipping...\n")
        return
    with open(result_filename, "rb") as f:
        result = pickle.load(f)

    mol_filename = os.path.join(mol_data_dir, f"{task.molecule_basename}.pickle")
    with open(mol_filename, "rb") as f:
        mol_data: ffsim.MolecularData = pickle.load(f)
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)

    alpha_alpha_indices, alpha_beta_indices = _get_lucj_indices(task.connectivity, norb)

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
