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
    filename="logs/run_ethene.log",
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
    "square",
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

tasks = [
    LUCJTask(
        molecule_basename=f"{molecule_basename}_bond_distance_{bond_distance:.5f}",
        connectivity=connectivity,
        n_reps=n_reps,
        with_final_orbital_rotation=with_final_orbital_rotation,
        optimization_method=optimization_method,
        maxiter=maxiter,
        bootstrap_task=None,
    )
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
    )
    for bond_distance in bond_distance_range
]

with ProcessPoolExecutor(MAX_PROCESSES) as executor:
    for task in tasks:
        future = executor.submit(
            run_lucj_task,
            task,
            data_dir=DATA_DIR,
            mol_data_dir=MOL_DATA_DIR,
            overwrite=overwrite,
        )
