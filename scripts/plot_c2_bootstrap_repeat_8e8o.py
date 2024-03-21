import itertools
import os
import pickle

import ffsim
import numpy as np
import pandas as pd
from lucj_ffsim.lucj import LUCJTask
from lucj_ffsim.plot import plot_reference_curves, plot_bootstrap_iteration, plot_error

DATA_ROOT = "/disk1/kevinsung@ibm.com/lucj-ffsim"

MOL_DATA_DIR = os.path.join(DATA_ROOT, "molecular_data")
DATA_DIR = os.path.join(DATA_ROOT, "lucj-bootstrap")
DATA_DIR_0 = os.path.join(DATA_ROOT, "lucj-bootstrap-repeat")
# DATA_DIR_1 = os.path.join(DATA_ROOT, "lucj-bootstrap-repeat-1")
PLOTS_DIR = "plots/lucj-bootstrap-repeat"
os.makedirs(PLOTS_DIR, exist_ok=True)


basis = "sto-6g"
ne, norb = 8, 8
molecule_basename = f"c2_dissociation_{basis}_{ne}e{norb}o"

reference_curves_d_range = np.arange(0.90, 3.01, 0.05)
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
]
optimization_methods = [
    # "none",
    "L-BFGS-B",
    "linear-method",
]
with_final_orbital_rotation_choices = [False, True]
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
results = {}
for (
    connectivity,
    n_reps,
    with_final_orbital_rotation,
    optimization_method,
) in ansatz_settings:
    for d in d_range:
        task = LUCJTask(
            molecule_basename=f"{molecule_basename}_d-{d:.2f}",
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            optimization_method=optimization_method,
            maxiter=maxiter,
            bootstrap_task=None,
        )
        for i, data_dir in enumerate([DATA_DIR, DATA_DIR_0]):
            filename = os.path.join(data_dir, task.dirname, "data.pickle")
            with open(filename, "rb") as f:
                data[
                    (
                        d,
                        connectivity,
                        n_reps,
                        with_final_orbital_rotation,
                        optimization_method,
                        i,
                    )
                ] = pickle.load(f)
            filename = os.path.join(data_dir, task.dirname, "info.pickle")
            with open(filename, "rb") as f:
                infos[
                    (
                        d,
                        connectivity,
                        n_reps,
                        with_final_orbital_rotation,
                        optimization_method,
                        i,
                    )
                ] = pickle.load(f)
            filename = os.path.join(data_dir, task.dirname, "result.pickle")
            with open(filename, "rb") as f:
                results[
                    (
                        d,
                        connectivity,
                        n_reps,
                        with_final_orbital_rotation,
                        optimization_method,
                        i,
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
            "bootstrap_iteration",
        ],
    ),
    inplace=True,
)
data.drop(columns="key", inplace=True)  # Drop the original 'Key' column

plots_dir = os.path.join(PLOTS_DIR, molecule_basename)
os.makedirs(plots_dir, exist_ok=True)
plot_reference_curves(
    filename=os.path.join(plots_dir, "reference_curves.svg"),
    title="C2 dissociation STO-6g (8e, 8o)",
    reference_curves_bond_distance_range=reference_curves_d_range,
    hf_energies_reference=hf_energies_reference,
    ccsd_energies_reference=ccsd_energies_reference,
    fci_energies_reference=fci_energies_reference,
)

for connectivity in connectivities:
    plots_dir = os.path.join(
        PLOTS_DIR,
        molecule_basename,
        f"npts-{n_pts}",
        connectivity,
    )
    os.makedirs(plots_dir, exist_ok=True)
    for n_reps, optimization_method, with_final_orbital_rotation in itertools.product(
        n_reps_range, optimization_methods, with_final_orbital_rotation_choices
    ):
        plot_error(
            filename=os.path.join(
                plots_dir,
                f"error_n_reps-{n_reps}_{optimization_method}_orb_rot-{with_final_orbital_rotation}.svg",
            ),
            title=f"C2 dissociation STO-6g (8e, 8o), {connectivity}, L={n_reps}, {optimization_method}",
            data=data,
            reference_curves_bond_distance_range=reference_curves_d_range,
            hf_energies_reference=hf_energies_reference,
            fci_energies_reference=fci_energies_reference,
            bond_distance_range=d_range,
            optimization_methods=optimization_methods,
            with_final_orbital_rotation=with_final_orbital_rotation,
            bootstrap_iterations=[0, 1, 2],
            connectivity=connectivity,
            n_reps_range=n_reps_range,
        )
        plot_bootstrap_iteration(
            filename=os.path.join(
                plots_dir,
                f"bootstrap_n_reps-{n_reps}_{optimization_method}_orb_rot-{with_final_orbital_rotation}.svg",
            ),
            title=f"C2 dissociation STO-6g (8e, 8o), {connectivity}, L={n_reps}, {optimization_method}",
            data=data,
            reference_curves_bond_distance_range=reference_curves_d_range,
            hf_energies_reference=hf_energies_reference,
            fci_energies_reference=fci_energies_reference,
            bond_distance_range=d_range,
            optimization_method=optimization_method,
            with_final_orbital_rotation=with_final_orbital_rotation,
            bootstrap_iterations=[0, 1],
            connectivity=connectivity,
            n_reps=n_reps,
        )
