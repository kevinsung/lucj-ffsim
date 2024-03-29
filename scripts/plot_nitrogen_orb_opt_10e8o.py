import itertools
import os
import pickle

import matplotlib.pyplot as plt
import ffsim
import numpy as np
import pandas as pd
from lucj_ffsim.lucj import LUCJTask
from lucj_ffsim.plot import (
    plot_reference_curves,
    plot_energy,
    plot_error,
    plot_optimization_method,
)

DATA_ROOT = "/disk1/kevinsung@ibm.com/lucj-ffsim"

MOL_DATA_DIR = os.path.join(DATA_ROOT, "molecular_data")
DATA_DIR = os.path.join(DATA_ROOT, "lucj-orb-opt")
DATA_DIR_0 = os.path.join(DATA_ROOT, "lucj-bootstrap")
DATA_DIR_1 = os.path.join(DATA_ROOT, "lucj-bootstrap-repeat")
DATA_DIR_2 = os.path.join(DATA_ROOT, "lucj-bootstrap-repeat-1")
PLOTS_DIR = "plots/lucj-orb-rot"
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
    # None,
    2,
    4,
    6,
]
optimization_methods = [
    # "none",
    "L-BFGS-B",
    "linear-method",
]
maxiter = 1000

ansatz_settings = list(
    itertools.product(
        connectivities,
        n_reps_range,
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
    optimization_method,
) in ansatz_settings:
    for d in d_range:
        task = LUCJTask(
            molecule_basename=f"{molecule_basename}_d-{d:.2f}",
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=False,
            optimization_method=optimization_method,
            maxiter=maxiter,
            bootstrap_task=None,
        )
        min_energy = float("inf")
        result_data = None
        info = None
        result = None
        for data_dir in [DATA_DIR_0, DATA_DIR_1, DATA_DIR_2]:
            filename = os.path.join(data_dir, task.dirname, "data.pickle")
            with open(filename, "rb") as f:
                this_data = pickle.load(f)
            if this_data["energy"] < min_energy:
                min_energy = this_data["energy"]
                result_data = this_data
                filename = os.path.join(data_dir, task.dirname, "info.pickle")
                with open(filename, "rb") as f:
                    info = pickle.load(f)
                filename = os.path.join(data_dir, task.dirname, "result.pickle")
                with open(filename, "rb") as f:
                    result = pickle.load(f)
        data[
            (
                d,
                connectivity,
                n_reps,
                False,
                optimization_method,
            )
        ] = result_data
        infos[
            (
                d,
                connectivity,
                n_reps,
                False,
                optimization_method,
            )
        ] = info
        results[
            (
                d,
                connectivity,
                n_reps,
                False,
                optimization_method,
            )
        ] = result

        task = LUCJTask(
            molecule_basename=f"{molecule_basename}_d-{d:.2f}",
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
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
                    True,
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
                    True,
                    optimization_method,
                )
            ] = pickle.load(f)
        filename = os.path.join(DATA_DIR, task.dirname, "result.pickle")
        with open(filename, "rb") as f:
            results[
                (
                    d,
                    connectivity,
                    n_reps,
                    True,
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

plots_dir = os.path.join(PLOTS_DIR, molecule_basename)
os.makedirs(plots_dir, exist_ok=True)
plot_reference_curves(
    filename=os.path.join(plots_dir, "reference_curves.svg"),
    title="N2 dissociation STO-6g (10e, 8o)",
    reference_curves_bond_distance_range=reference_curves_d_range,
    hf_energies_reference=hf_energies_reference,
    ccsd_energies_reference=ccsd_energies_reference,
    fci_energies_reference=fci_energies_reference,
)

markers = ["o", "s", "v", "D", "p", "*"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
for connectivity in connectivities:
    plots_dir = os.path.join(
        PLOTS_DIR,
        molecule_basename,
        f"npts-{n_pts}",
        connectivity,
    )
    os.makedirs(plots_dir, exist_ok=True)
    for n_reps in n_reps_range:
        plot_optimization_method(
            filename=os.path.join(plots_dir, f"n_reps-{n_reps}.svg"),
            title=f"N2 dissociation STO-6g (10e, 8o), {connectivity}, L={n_reps}",
            data=data,
            reference_curves_bond_distance_range=reference_curves_d_range,
            hf_energies_reference=hf_energies_reference,
            fci_energies_reference=fci_energies_reference,
            bond_distance_range=d_range,
            optimization_methods=optimization_methods,
            with_final_orbital_rotation_choices=[False, True],
            connectivity=connectivity,
            n_reps=n_reps,
        )
    for with_final_orbital_rotation in [False, True]:
        plot_energy(
            filename=os.path.join(
                plots_dir,
                f"energy_orb_rot-{with_final_orbital_rotation}.svg",
            ),
            data=data,
            reference_curves_bond_distance_range=reference_curves_d_range,
            hf_energies_reference=hf_energies_reference,
            fci_energies_reference=fci_energies_reference,
            bond_distance_range=d_range,
            optimization_method="linear-method",
            with_final_orbital_rotation=with_final_orbital_rotation,
            connectivity=connectivity,
            n_reps_range=[2, 4, 6],
            markers=markers[2::-1],
            colors=colors[2::-1],
        )
        plot_error(
            filename=os.path.join(
                plots_dir,
                f"error_orb_rot-{with_final_orbital_rotation}.svg",
            ),
            data=data,
            bond_distance_range=d_range,
            optimization_methods=optimization_methods,
            with_final_orbital_rotation=with_final_orbital_rotation,
            connectivity=connectivity,
            n_reps_range=[4, 6],
            ymin=1e-5,
            ymax=1e-2,
            markers=markers[1::-1],
            colors=colors[1::-1],
        )
