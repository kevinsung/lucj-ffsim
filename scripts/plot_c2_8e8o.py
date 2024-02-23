import itertools
import os
import pickle

import ffsim
import numpy as np
import pandas as pd
from lucj_ffsim.lucj import LUCJTask
from lucj_ffsim.plot import (
    plot_n_reps,
    plot_optimization_method,
    plot_overlap_mats,
    plot_reference_curves,
)

DATA_ROOT = "/disk1/kevinsung@ibm.com/lucj-ffsim"

MOL_DATA_DIR = os.path.join(DATA_ROOT, "molecular_data")
DATA_DIR = os.path.join(DATA_ROOT, "lucj")
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


basis = "sto-6g"
ne, norb = 8, 8
molecule_basename = f"c2_dissociation_{basis}_{ne}e{norb}o"

reference_curves_d_range = np.arange(0.90, 3.01, 0.05)
d_range = np.arange(0.90, 3.01, 0.10)
connectivities = [
    "square",
    "all-to-all",
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

ansatz_settings = list(
    itertools.product(
        connectivities,
        n_reps_range,
        with_final_orbital_rotation_choices,
        optimization_methods,
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
for (
    connectivity,
    n_reps,
    with_final_orbital_rotation,
    # param_initialization,
    optimization_method,
) in ansatz_settings:
    for d in d_range:
        task = LUCJTask(
            molecule_basename=f"{molecule_basename}_d-{d:.2f}",
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            # param_initialization=param_initialization,
            optimization_method=optimization_method,
            maxiter=maxiter,
            bootstrap_task=None,
        )
        filename = os.path.join(DATA_DIR, task.dirname, "data.pickle")
        with open(filename, "rb") as f:
            data[
                (
                    d,
                    connectivity,
                    n_reps,
                    with_final_orbital_rotation,
                    # param_initialization,
                    optimization_method,
                )
            ] = pickle.load(f)
        filename = os.path.join(DATA_DIR, task.dirname, "info.pickle")
        with open(filename, "rb") as f:
            infos[
                (
                    d,
                    connectivity,
                    n_reps,
                    with_final_orbital_rotation,
                    # param_initialization,
                    optimization_method,
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
            "with_final_orbital_rotation",
            "optimization_method",
        ],
    ),
    inplace=True,
)
data.drop(columns="key", inplace=True)  # Drop the original 'Key' column


for connectivity, n_reps in itertools.product(connectivities, n_reps_range):
    # plot_optimization_iterations(connectivity=connectivity, n_reps=n_reps)
    plot_reference_curves(
        plots_dir=PLOTS_DIR,
        title="Carbon dimer dissociation STO-6g (8e, 8o)",
        molecule_basename=molecule_basename,
        reference_curves_bond_distance_range=reference_curves_d_range,
        hf_energies_reference=hf_energies_reference,
        ccsd_energies_reference=ccsd_energies_reference,
        fci_energies_reference=fci_energies_reference,
    )
    plot_optimization_method(
        plots_dir=PLOTS_DIR,
        title="Carbon dimer dissociation STO-6g (8e, 8o)" + f", {connectivity}",
        data=data,
        molecule_basename=molecule_basename,
        reference_curves_bond_distance_range=reference_curves_d_range,
        hf_energies_reference=hf_energies_reference,
        fci_energies_reference=fci_energies_reference,
        bond_distance_range=d_range,
        n_pts=n_pts,
        optimization_methods=optimization_methods,
        connectivity=connectivity,
        n_reps=n_reps,
    )
    plot_overlap_mats(
        plots_dir=PLOTS_DIR,
        title="Carbon dimer dissociation STO-6g (8e, 8o)" + f", L={n_reps}",
        infos=infos,
        molecule_basename=molecule_basename,
        bond_distance_range=d_range,
        n_pts=n_pts,
        connectivity=connectivity,
        n_reps=n_reps,
    )

for connectivity, optimization_method in itertools.product(
    connectivities, optimization_methods
):
    plot_n_reps(
        plots_dir=PLOTS_DIR,
        title="Carbon dimer dissociation STO-6g (8e, 8o)" + f", {connectivity}",
        data=data,
        molecule_basename=molecule_basename,
        reference_curves_bond_distance_range=reference_curves_d_range,
        hf_energies_reference=hf_energies_reference,
        fci_energies_reference=fci_energies_reference,
        bond_distance_range=d_range,
        n_pts=n_pts,
        optimization_method=optimization_method,
        connectivity=connectivity,
        n_reps_range=n_reps_range,
    )
