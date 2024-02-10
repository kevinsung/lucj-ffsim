from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from lucj_ffsim.lucj import LUCJTask, run_lucj_task

import numpy as np

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
    filename="logs/run_ethene_bootstrap.log",
)

DATA_ROOT = "/disk1/kevinsung@ibm.com/lucj-ffsim"

MOL_DATA_DIR = os.path.join(DATA_ROOT, "molecular_data")
DATA_DIR = os.path.join(DATA_ROOT, "lucj")
MAX_PROCESSES = 144

basis = "sto-6g"
ne, norb = 4, 4
molecule_basename = f"ethene_dissociation_{basis}_{ne}e{norb}o"
overwrite = True

bond_distance_range = np.linspace(1.3, 4.0, 20)
connectivities = [
    "all-to-all",
    "square",
    "hex",
    "heavy-hex",
]
n_reps_range = [
    2,
    4,
    6,
]
with_final_orbital_rotation_choices = [False]
optimization_methods = [
    "L-BFGS-B",
    "linear-method",
]
maxiter = 10000


def run_bootstrap_tasks(
    molecule_basename: str,
    bond_distance_range: np.ndarray,
    connectivity: str,
    n_reps: int,
    with_final_orbital_rotation: bool,
    optimization_method: str,
    maxiter: int,
    overwrite: bool,
):
    tasks = [
        LUCJTask(
            molecule_basename=f"{molecule_basename}_bond_distance_{bond_distance_range[0]:.5f}",
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            bootstrap_task=None,
            optimization_method=optimization_method,
            maxiter=maxiter,
        )
    ]
    for i in range(1, len(bond_distance_range)):
        bootstrap_task = tasks[-1]
        bond_distance = bond_distance_range[i]
        tasks.append(
            LUCJTask(
                molecule_basename=f"{molecule_basename}_bond_distance_{bond_distance:.5f}",
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=with_final_orbital_rotation,
                bootstrap_task=bootstrap_task,
                optimization_method=optimization_method,
                maxiter=maxiter,
            )
        )
    for task in tasks:
        run_lucj_task(task, overwrite=overwrite)


with ProcessPoolExecutor(MAX_PROCESSES) as executor:
    for (
        connectivity,
        n_reps,
        with_final_orbital_rotation,
        optimization_method,
    ) in itertools.product(
        connectivities,
        n_reps_range,
        with_final_orbital_rotation_choices,
        optimization_methods,
    ):
        _ = executor.submit(
            run_bootstrap_tasks,
            molecule_basename=molecule_basename,
            bond_distance_range=bond_distance_range,
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            optimization_method=optimization_method,
            maxiter=maxiter,
            overwrite=overwrite,
        )
