import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_optimization_iterations(
    plots_dir: str,
    title: str,
    data: pd.DataFrame,
    molecule_basename: str,
    bond_distance_range: np.ndarray,
    n_pts: int,
    optimization_methods: list[str],
    connectivity: str,
    n_reps: int,
):
    markers = ["o", "s", "v", "D", "p", "*"]

    this_data = {}
    for optimization_method in optimization_methods:
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "with_final_orbital_rotation": False,
            "optimization_method": optimization_method,
        }
        this_data[optimization_method] = data.xs(
            tuple(settings.values()), level=tuple(settings.keys())
        )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(wspace=0.25)

    for optimization_method, marker in zip(optimization_methods, markers):
        ax1.plot(
            bond_distance_range,
            this_data[optimization_method]["error"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax1.set_yscale("log")
    ax1.axhline(1e-3, linestyle="--", color="gray")
    ax1.legend()
    ax1.set_ylabel("Energy error (Hartree)")

    ax2.plot(
        bond_distance_range,
        this_data["L-BFGS-B"]["nfev"].values,
        f"{marker}--",
        label=f"LUCJ, L={n_reps}, L-BFGS-B",
    )
    ax2.plot(
        bond_distance_range,
        this_data["linear-method"]["nfev"].values,
        f"{marker}--",
        label=f"LUCJ, L={n_reps}, LM, vec",
    )
    ax2.plot(
        bond_distance_range,
        this_data["linear-method"]["nlinop"].values,
        f"{marker}--",
        label=f"LUCJ, L={n_reps}, LM, op",
    )
    ax2.legend()
    ax2.set_ylabel("Number of evaluations")

    for optimization_method, marker in zip(optimization_methods, markers):
        ax3.plot(
            bond_distance_range,
            this_data[optimization_method]["spin_squared"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax3.axhline(0, linestyle="--", color="gray")
    ax3.legend()
    ax3.set_ylabel("Spin squared")
    ax3.set_xlabel("Bond length (Å)")

    for optimization_method, marker in zip(optimization_methods, markers):
        ax4.plot(
            bond_distance_range,
            this_data[optimization_method]["nit"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax4.legend()
    ax4.set_ylabel("Number of iterations")

    fig.suptitle(title)

    dirname = os.path.join(
        plots_dir,
        molecule_basename,
        f"npts-{n_pts}",
        connectivity,
    )
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, f"n_reps-{n_reps}-iterations.svg")
    plt.savefig(filename)

    plt.close()


def plot_reference_curves(
    plots_dir: str,
    title: str,
    molecule_basename: str,
    reference_curves_bond_distance_range: np.ndarray,
    hf_energies_reference: np.ndarray,
    ccsd_energies_reference: np.ndarray,
    fci_energies_reference: np.ndarray,
):
    fig, ax = plt.subplots()

    ax.plot(
        reference_curves_bond_distance_range,
        hf_energies_reference,
        "--",
        label="HF",
        color="blue",
    )
    ax.plot(
        reference_curves_bond_distance_range,
        ccsd_energies_reference,
        "--",
        label="CCSD",
        color="orange",
    )
    ax.plot(
        reference_curves_bond_distance_range,
        fci_energies_reference,
        "-",
        label="FCI",
        color="black",
    )
    ax.legend()
    ax.set_title(title)

    dirname = os.path.join(
        plots_dir,
        molecule_basename,
    )
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, "reference_curves.svg")
    plt.savefig(filename)
    plt.close()


def plot_optimization_method(
    plots_dir: str,
    title: str,
    data: pd.DataFrame,
    molecule_basename: str,
    reference_curves_bond_distance_range: np.ndarray,
    hf_energies_reference: np.ndarray,
    fci_energies_reference: np.ndarray,
    bond_distance_range: np.ndarray,
    n_pts: int,
    optimization_methods: list[str],
    connectivity: str,
    n_reps: int,
):
    # effect of optimization method
    markers = ["o", "s", "v", "D", "p", "*"]

    this_data = {}
    for optimization_method in optimization_methods:
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "with_final_orbital_rotation": False,
            "optimization_method": optimization_method,
        }
        this_data[optimization_method] = data.xs(
            tuple(settings.values()), level=tuple(settings.keys())
        )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(wspace=0.25)

    ax1.plot(
        reference_curves_bond_distance_range,
        hf_energies_reference,
        "--",
        label="HF",
        color="blue",
    )
    ax1.plot(
        reference_curves_bond_distance_range,
        fci_energies_reference,
        "-",
        label="FCI",
        color="black",
    )
    for optimization_method, marker in zip(optimization_methods, markers):
        ax1.plot(
            bond_distance_range,
            this_data[optimization_method]["energy"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax1.legend()
    ax1.set_ylabel("Energy (Hartree)")

    for optimization_method, marker in zip(optimization_methods, markers):
        ax2.plot(
            bond_distance_range,
            this_data[optimization_method]["error"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax2.set_yscale("log")
    ax2.axhline(1e-3, linestyle="--", color="gray")
    ax2.legend()
    ax2.set_ylabel("Energy error (Hartree)")

    for optimization_method, marker in zip(optimization_methods, markers):
        ax3.plot(
            bond_distance_range,
            this_data[optimization_method]["spin_squared"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax3.axhline(0, linestyle="--", color="gray")
    ax3.legend()
    ax3.set_ylabel("Spin squared")
    ax3.set_xlabel("Bond length (Å)")

    for optimization_method, marker in zip(optimization_methods, markers):
        ax4.plot(
            bond_distance_range,
            this_data[optimization_method]["nit"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax4.legend()
    ax4.set_ylabel("Number of iterations")

    fig.suptitle(title)

    dirname = os.path.join(
        plots_dir,
        molecule_basename,
        f"npts-{n_pts}",
        connectivity,
    )
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, f"n_reps-{n_reps}.svg")
    plt.savefig(filename)

    plt.close()


def plot_n_reps(
    plots_dir: str,
    title: str,
    data: pd.DataFrame,
    molecule_basename: str,
    reference_curves_bond_distance_range: np.ndarray,
    hf_energies_reference: np.ndarray,
    fci_energies_reference: np.ndarray,
    bond_distance_range: np.ndarray,
    n_pts: int,
    optimization_method: str,
    connectivity: str,
    n_reps_range: list[int],
):
    # effect of number of circuit repetitions
    markers = ["o", "s", "v", "D", "p", "*"]

    this_data = {}
    for n_reps in n_reps_range:
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "with_final_orbital_rotation": False,
            "optimization_method": optimization_method,
        }
        this_data[n_reps] = data.xs(
            tuple(settings.values()), level=tuple(settings.keys())
        )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(wspace=0.25)

    ax1.plot(
        reference_curves_bond_distance_range,
        hf_energies_reference,
        "--",
        label="HF",
        color="blue",
    )
    ax1.plot(
        reference_curves_bond_distance_range,
        fci_energies_reference,
        "-",
        label="FCI",
        color="black",
    )
    for n_reps, marker in zip(n_reps_range, markers):
        ax1.plot(
            bond_distance_range,
            this_data[n_reps]["energy"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax1.legend()
    ax1.set_ylabel("Energy (Hartree)")

    for n_reps, marker in zip(n_reps_range, markers):
        ax2.plot(
            bond_distance_range,
            this_data[n_reps]["error"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {n_reps}",
        )
    ax2.set_yscale("log")
    ax2.axhline(1e-3, linestyle="--", color="gray")
    ax2.legend()
    ax2.set_ylabel("Energy error (Hartree)")

    for n_reps, marker in zip(n_reps_range, markers):
        ax3.plot(
            bond_distance_range,
            this_data[n_reps]["spin_squared"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax3.axhline(0, linestyle="--", color="gray")
    ax3.legend()
    ax3.set_ylabel("Spin squared")
    ax3.set_xlabel("Bond length (Å)")

    for n_reps, marker in zip(n_reps_range, markers):
        ax4.plot(
            bond_distance_range,
            this_data[n_reps]["nit"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax4.legend()
    ax4.set_ylabel("Number of iterations")

    fig.suptitle(title)

    dirname = os.path.join(
        plots_dir,
        molecule_basename,
        f"npts-{n_pts}",
        connectivity,
    )
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, f"{optimization_method}.svg")
    plt.savefig(filename)

    plt.close()


def plot_overlap_mats(
    plots_dir: str,
    title: str,
    infos: dict,
    molecule_basename: str,
    bond_distance_range: np.ndarray,
    n_pts: int,
    connectivity: str,
    n_reps: int,
):
    n_mat = 5
    bond_distances = bond_distance_range[::6]

    fig, axes = plt.subplots(
        len(bond_distances), n_mat, figsize=(6 * n_mat, 6 * len(bond_distances))
    )

    for these_axes, bond_distance in zip(axes, bond_distances):
        info = infos[
            bond_distance,
            connectivity,
            n_reps,
            False,
            "linear-method",
        ]

        iterations, overlap_mats = zip(*info["overlap_mat"])
        nit = len(overlap_mats)
        step = nit // (n_mat - 1)
        indices = list(range(0, nit, step))
        while len(indices) > n_mat - 1:
            indices.pop()
        if nit - 1 not in indices:
            indices.append(nit - 1)
        mats = [overlap_mats[i] for i in indices]
        iteration_nums = [iterations[i] for i in indices]

        for ax, mat, i in zip(these_axes, mats, iteration_nums):
            max_val = np.max(np.abs(mat))
            im = ax.matshow(
                mat,
                cmap="bwr",
                vmin=-max_val,
                vmax=max_val,
            )
            ax.set_title(f"Iteration {i}")
            if i == 0:
                ax.set_ylabel(f"d = {bond_distance}")
            fig.colorbar(im)

        fig.suptitle(title)

    dirname = os.path.join(
        plots_dir,
        molecule_basename,
        f"npts-{n_pts}",
        connectivity,
    )
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, f"overlap_mat_n_reps-{n_reps}.svg")
    plt.savefig(filename)


def plot_linear_method_hyperparameters(
    plots_dir: str,
    title: str,
    infos: dict,
    molecule_basename: str,
    bond_distances: np.ndarray,
    n_pts: int,
    connectivity: str,
    n_reps: int,
):
    fig, axes = plt.subplots(
        2, len(bond_distances), figsize=(6 * len(bond_distances), 12)
    )

    for ax, bond_distance in zip(axes[0], bond_distances):
        info = infos[
            bond_distance,
            connectivity,
            n_reps,
            False,
            "linear-method",
        ]
        vals = info["regularization"]
        ax.set_title(f"regularization, d={bond_distance:0.2f}")
        ax.plot(range(len(vals)), vals)
    for ax, bond_distance in zip(axes[1], bond_distances):
        info = infos[
            bond_distance,
            connectivity,
            n_reps,
            False,
            "linear-method",
        ]
        vals = info["variation"]
        ax.set_ylim(0, 1)
        ax.set_title(f"variation, d={bond_distance:0.2f}")
        ax.plot(range(len(vals)), vals)

    fig.suptitle(title)

    dirname = os.path.join(
        plots_dir,
        molecule_basename,
        f"npts-{n_pts}",
        connectivity,
    )
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, f"hyperparameters_n_reps-{n_reps}.svg")
    plt.savefig(filename)


def plot_parameters(
    plots_dir: str,
    title: str,
    results: dict,
    molecule_basename: str,
    bond_distance_range: np.ndarray,
    n_pts: int,
    optimization_method: str,
    connectivity: str,
    n_reps: int,
):
    fig, ax = plt.subplots(1, 1)

    result = results[
        bond_distance_range[0], connectivity, n_reps, False, optimization_method
    ]
    current_params = result.x

    vals = []
    for bond_distance in bond_distance_range[1:]:
        result = results[
            bond_distance, connectivity, n_reps, False, optimization_method
        ]
        params = result.x
        vals.append(np.linalg.norm(params - current_params))
        current_params = params

    ax.plot(bond_distance_range[1:], vals, "o--")
    ax.set_yscale("log")
    ax.set_title("|x_t - x_{t-1}|")
    fig.suptitle(title)

    dirname = os.path.join(
        plots_dir,
        molecule_basename,
        f"npts-{n_pts}",
        connectivity,
    )
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, f"parameters_n_reps-{n_reps}.svg")
    plt.savefig(filename)
