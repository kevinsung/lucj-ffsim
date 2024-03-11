from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from lucj_ffsim.lucj import LUCJTask, copy_data, run_lucj_task

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
    filename=f"logs/{os.path.splitext(os.path.basename(__file__))[0]}.log",
)

DATA_ROOT = "/disk1/kevinsung@ibm.com/lucj-ffsim"

MOL_DATA_DIR = os.path.join(DATA_ROOT, "molecular_data")
DATA_DIR = os.path.join(DATA_ROOT, "lucj-bootstrap-repeat-1")
BOOTSTRAP_DATA_DIR = os.path.join(DATA_ROOT, "lucj-bootstrap-repeat")
MAX_PROCESSES = 96


def generate_lucj_tasks_bootstrap_forward(
    d_range: np.ndarray,
    molecule_basename: str,
    connectivity: str,
    n_reps: int,
    with_final_orbital_rotation: bool,
    optimization_method: str,
    maxiter: int,
):
    current_task = LUCJTask(
        molecule_basename=f"{molecule_basename}_d-{d_range[0]:.2f}",
        connectivity=connectivity,
        n_reps=n_reps,
        with_final_orbital_rotation=with_final_orbital_rotation,
        optimization_method=optimization_method,
        maxiter=maxiter,
    )
    copy_data(
        current_task,
        src_data_dir=BOOTSTRAP_DATA_DIR,
        dst_data_dir=DATA_DIR,
        dirs_exist_ok=True,
    )
    for d in d_range[1:]:
        task = LUCJTask(
            molecule_basename=f"{molecule_basename}_d-{d:.2f}",
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            optimization_method=optimization_method,
            maxiter=maxiter,
            bootstrap_task=current_task,
        )
        yield task
        current_task = task


basis = "sto-6g"
ne, norb = 10, 8
molecule_basename = f"nitrogen_dissociation_{basis}_{ne}e{norb}o"
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
]
optimization_methods = [
    "none",
    "L-BFGS-B",
    "linear-method",
]
with_final_orbital_rotation_choices = [False]
maxiter = 1000

task_lists = [
    list(
        generate_lucj_tasks_bootstrap_forward(
            d_range=d_range,
            molecule_basename=molecule_basename,
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            optimization_method=optimization_method,
            maxiter=maxiter,
        )
    )
    for (
        connectivity,
        n_reps,
        optimization_method,
        with_final_orbital_rotation,
    ) in itertools.product(
        connectivities,
        n_reps_range,
        optimization_methods,
        with_final_orbital_rotation_choices,
    )
]


def run_lucj_tasks(tasks: list[LUCJTask], overwrite: bool):
    for task in tasks:
        run_lucj_task(
            task,
            data_dir=DATA_DIR,
            mol_data_dir=MOL_DATA_DIR,
            overwrite=overwrite,
        )


with ProcessPoolExecutor(MAX_PROCESSES) as executor:
    for task_list in task_lists:
        future = executor.submit(
            run_lucj_tasks,
            task_list,
            overwrite=overwrite,
        )
