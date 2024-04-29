from __future__ import annotations

import itertools
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from lucj_ffsim.lucj import LUCJTask, run_lucj_task

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
    filename=f"logs/{os.path.splitext(os.path.basename(__file__))[0]}.log",
)

DATA_ROOT = "/disk1/kevinsung@ibm.com/lucj-ffsim"

MOL_DATA_DIR = os.path.join(DATA_ROOT, "molecular_data")
DATA_DIR = os.path.join(DATA_ROOT, "lucj-orb-opt")
BOOTSTRAP_DATA_DIRS = [
    os.path.join(DATA_ROOT, dirname)
    for dirname in [
        "lucj-bootstrap",
        "lucj-bootstrap-repeat",
        "lucj-bootstrap-repeat-1",
    ]
]
MAX_PROCESSES = 88


def get_bootstrap_task(
    molecule_basename: str,
    connectivity: str,
    n_reps: int,
    optimization_method: str,
    maxiter: int,
):
    task = LUCJTask(
        molecule_basename=molecule_basename,
        connectivity=connectivity,
        n_reps=n_reps,
        with_final_orbital_rotation=False,
        optimization_method=optimization_method,
        maxiter=maxiter,
    )
    final_bootstrap_dir = None
    min_energy = float("inf")
    for bootstrap_dir in BOOTSTRAP_DATA_DIRS:
        filename = os.path.join(bootstrap_dir, task.dirname, "data.pickle")
        with open(filename, "rb") as f:
            data = pickle.load(f)
        if data["energy"] < min_energy:
            min_energy = data["energy"]
            final_bootstrap_dir = bootstrap_dir
    return task, final_bootstrap_dir


basis = "sto-6g"
ne, norb = 8, 8
molecule_basename = f"c2_dissociation_{basis}_{ne}e{norb}o"
overwrite = False

d_range = np.arange(0.90, 3.01, 0.10)
connectivities = [
    "square",
    # "all-to-all",
]
n_reps_range = [
    # None,
    2,
    4,
    6,
    8,
    10,
    12,
]
optimization_methods = [
    "L-BFGS-B",
    "linear-method",
]
maxiter = 1000


def generate_tasks():
    for connectivity, n_reps, optimization_method in itertools.product(
        connectivities, n_reps_range, optimization_methods
    ):
        for d in d_range:
            bootstrap_task, bootstrap_dir = get_bootstrap_task(
                molecule_basename=f"{molecule_basename}_d-{d:.2f}",
                connectivity=connectivity,
                n_reps=n_reps,
                optimization_method=optimization_method,
                maxiter=maxiter,
            )
            yield (
                LUCJTask(
                    molecule_basename=f"{molecule_basename}_d-{d:.2f}",
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                    optimization_method=optimization_method,
                    maxiter=maxiter,
                    bootstrap_task=bootstrap_task,
                ),
                bootstrap_dir,
            )


with ProcessPoolExecutor(MAX_PROCESSES) as executor:
    for task, bootstrap_data_dir in generate_tasks():
        future = executor.submit(
            run_lucj_task,
            task,
            data_dir=DATA_DIR,
            mol_data_dir=MOL_DATA_DIR,
            bootstrap_data_dir=bootstrap_data_dir,
            overwrite=overwrite,
        )
