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
DATA_DIR = os.path.join(DATA_ROOT, "lucj")
MAX_PROCESSES = 96

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
    # 2,
    # 4,
    6,
    # None,
]
optimization_methods = [
    # "none",
    # "L-BFGS-B",
    "linear-method",
]
linear_method_regularizations = [
    None,
    0.0,
    1.0,
    10.0,
]
linear_method_variations = [
    None,
    0.0,
    0.5,
    1.0,
]
with_final_orbital_rotation_choices = [False]
maxiter = 1000

tasks = [
    LUCJTask(
        molecule_basename=f"{molecule_basename}_d-{d:.2f}",
        connectivity=connectivity,
        n_reps=n_reps,
        with_final_orbital_rotation=with_final_orbital_rotation,
        optimization_method=optimization_method,
        maxiter=maxiter,
        linear_method_regularization=linear_method_regularization,
        linear_method_variation=linear_method_variation,
        bootstrap_task=None,
    )
    for (
        connectivity,
        n_reps,
        optimization_method,
        linear_method_regularization,
        linear_method_variation,
        with_final_orbital_rotation,
    ) in itertools.product(
        connectivities,
        n_reps_range,
        optimization_methods,
        linear_method_regularizations,
        linear_method_variations,
        with_final_orbital_rotation_choices,
    )
    for d in d_range
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
        # future = executor.submit(
        #     process_result,
        #     task,
        #     data_dir=DATA_DIR,
        #     mol_data_dir=MOL_DATA_DIR,
        #     overwrite=overwrite,
        # ).result()
