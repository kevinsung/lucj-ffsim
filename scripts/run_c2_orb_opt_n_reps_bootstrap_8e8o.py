from __future__ import annotations

import itertools
import logging
import os
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
DATA_DIR = os.path.join(DATA_ROOT, "lucj-orb-opt-n_reps-bootstrap")
BOOTSTRAP_DATA_DIR = os.path.join(DATA_ROOT, "lucj-orb-opt")
MAX_PROCESSES = 88


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
    # 2,
    4,
    6,
    8,
    # 10,
]
optimization_methods = [
    "L-BFGS-B",
    "linear-method",
]
maxiter = 1000


# for connectivity, optimization_method in itertools.product(
#     connectivities, optimization_methods
# ):
#     for d in d_range:
#         task = LUCJTask(
#             molecule_basename=f"{molecule_basename}_d-{d:.2f}",
#             connectivity=connectivity,
#             n_reps=2,
#             with_final_orbital_rotation=True,
#             optimization_method=optimization_method,
#             maxiter=maxiter,
#         )
#         copy_data(
#             task,
#             src_data_dir=BOOTSTRAP_DATA_DIR,
#             dst_data_dir=DATA_DIR,
#             dirs_exist_ok=True,
#         )


def generate_tasks(n_reps: int):
    for connectivity, optimization_method in itertools.product(
        connectivities, optimization_methods
    ):
        for d in d_range:
            bootstrap_task = LUCJTask(
                molecule_basename=f"{molecule_basename}_d-{d:.2f}",
                connectivity=connectivity,
                n_reps=n_reps - 2,
                with_final_orbital_rotation=True,
                optimization_method=optimization_method,
                maxiter=maxiter,
            )
            yield LUCJTask(
                molecule_basename=f"{molecule_basename}_d-{d:.2f}",
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
                optimization_method=optimization_method,
                maxiter=maxiter,
                bootstrap_task=bootstrap_task,
            )


for n_reps in n_reps_range:
    logging.info(f"Running n_reps={n_reps}...")
    with ProcessPoolExecutor(MAX_PROCESSES) as executor:
        for task in generate_tasks(n_reps):
            future = executor.submit(
                run_lucj_task,
                task,
                data_dir=DATA_DIR,
                mol_data_dir=MOL_DATA_DIR,
                overwrite=overwrite,
            )
