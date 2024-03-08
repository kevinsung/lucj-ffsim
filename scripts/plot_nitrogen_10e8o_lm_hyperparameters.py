import itertools
import os
import pickle

import ffsim
import numpy as np
import pandas as pd
from lucj_ffsim.lucj import LUCJTask
from lucj_ffsim.plot import (
    # plot_n_reps,
    # plot_optimization_method,
    # plot_overlap_mats,
    # plot_reference_curves,
    # plot_optimization_iterations,
    # plot_linear_method_hyperparameters,
    # plot_parameters_distance,
    # plot_initial_parameters_distance,
    plot_lm_hyperparameter,
)

DATA_ROOT = "/disk1/kevinsung@ibm.com/lucj-ffsim"

MOL_DATA_DIR = os.path.join(DATA_ROOT, "molecular_data")
DATA_DIR = os.path.join(DATA_ROOT, "lucj")
PLOTS_DIR = "plots/lucj"
os.makedirs(PLOTS_DIR, exist_ok=True)


basis = "sto-6g"
ne, norb = 10, 8
molecule_basename = f"nitrogen_dissociation_{basis}_{ne}e{norb}o"

reference_curves_d_range = np.arange(0.90, 3.01, 0.05)
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
    # None,
    0.0,
    # 0.1,
    1.0,
    10.0,
]
linear_method_variations = [
    # None,
    0.0,
    0.5,
    1.0,
]
with_final_orbital_rotation_choices = [False]
maxiter = 1000

ansatz_settings = list(
    itertools.product(
        connectivities,
        n_reps_range,
        optimization_methods,
        linear_method_regularizations,
        linear_method_variations,
        with_final_orbital_rotation_choices,
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
):
    ansatz_settings.append(
        (
            connectivity,
            n_reps,
            optimization_method,
            None,
            None,
            with_final_orbital_rotation,
        )
    )
n_pts = len(d_range)

mol_datas_reference: dict[float, ffsim.MolecularData] = {}
mol_datas_experiment: dict[float, ffsim.MolecularData] = {}

for d in reference_curves_d_range:
    filename = os.path.join(MOL_DATA_DIR, f"{molecule_basename}_d-{d:.2f}.pickle")
    with open(filename, "rb") as f:
        mol_data = pickle.load(f)
        mol_datas_reference[d] = mol_data

for d in d_range:
    filename = os.path.join(MOL_DATA_DIR, f"{molecule_basename}_d-{d:.2f}.pickle")
    with open(filename, "rb") as f:
        mol_data = pickle.load(f)
        mol_datas_experiment[d] = mol_data

hf_energies_reference = np.array(
    [mol_data.hf_energy for mol_data in mol_datas_reference.values()]
)
fci_energies_reference = np.array(
    [mol_data.fci_energy for mol_data in mol_datas_reference.values()]
)
ccsd_energies_reference = np.array(
    [mol_data.ccsd_energy for mol_data in mol_datas_reference.values()]
)
hf_energies_experiment = np.array(
    [mol_data.hf_energy for mol_data in mol_datas_experiment.values()]
)
fci_energies_experiment = np.array(
    [mol_data.fci_energy for mol_data in mol_datas_experiment.values()]
)

data = {}
infos = {}
results = {}
for (
    connectivity,
    n_reps,
    optimization_method,
    linear_method_regularization,
    linear_method_variation,
    with_final_orbital_rotation,
) in ansatz_settings:
    for d in d_range:
        task = LUCJTask(
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
        filename = os.path.join(DATA_DIR, task.dirname, "data.pickle")
        with open(filename, "rb") as f:
            data[
                (
                    d,
                    connectivity,
                    n_reps,
                    optimization_method,
                    linear_method_regularization,
                    linear_method_variation,
                    with_final_orbital_rotation,
                )
            ] = pickle.load(f)
        filename = os.path.join(DATA_DIR, task.dirname, "info.pickle")
        with open(filename, "rb") as f:
            infos[
                (
                    d,
                    connectivity,
                    n_reps,
                    optimization_method,
                    linear_method_regularization,
                    linear_method_variation,
                    with_final_orbital_rotation,
                )
            ] = pickle.load(f)
        filename = os.path.join(DATA_DIR, task.dirname, "result.pickle")
        with open(filename, "rb") as f:
            results[
                (
                    d,
                    connectivity,
                    n_reps,
                    optimization_method,
                    linear_method_regularization,
                    linear_method_variation,
                    with_final_orbital_rotation,
                )
            ] = pickle.load(f)


keys = ["energy", "error", "spin_squared", "nit", "nfev", "nlinop"]
data = pd.DataFrame(
    list(
        zip(
            data.keys(),
            *zip(*[[d[k] for k in keys] for d in data.values()]),
        )
    ),
    columns=["key"] + keys,
)
data.set_index(
    pd.MultiIndex.from_tuples(
        data["key"],
        names=[
            "bond_distance",
            "connectivity",
            "n_reps",
            "optimization_method",
            "linear_method_regularization",
            "linear_method_variation",
            "with_final_orbital_rotation",
        ],
    ),
    inplace=True,
)
data.drop(columns="key", inplace=True)  # Drop the original 'Key' column


for connectivity in connectivities:
    plots_dir = os.path.join(
        PLOTS_DIR,
        molecule_basename,
        f"npts-{n_pts}",
        connectivity,
    )
    os.makedirs(plots_dir, exist_ok=True)
    for n_reps in n_reps_range:
        plot_lm_hyperparameter(
            filename=os.path.join(plots_dir, f"lm_hyperparameter_n_reps-{n_reps}.svg"),
            title="Nitrogen dissociation STO-6g (10e, 8o) overlap matrix"
            + f", L={n_reps}",
            data=data,
            bond_distance_range=d_range,
            linear_method_regularizations=linear_method_regularizations,
            linear_method_variations=linear_method_variations,
            connectivity=connectivity,
            n_reps=n_reps,
        )
