import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_optimization_iterations(
    plots_dir: str,
    data: pd.DataFrame,
    molecule_basename: str,
    bond_distance_range: np.ndarray,
    n_pts: int,
    optimization_methods: list[str],
    connectivity: str,
    n_reps: int,
):
    # compare cost of optimization methods
    markers = ["o", "s", "v", "D", "p", "*"]

    filename = (
        f"{molecule_basename}_npts-{n_pts}_nit"
        + f"_{connectivity}"
        + f"_n_reps-{n_reps}"
    )

    this_data = {}
    for optimization_method in optimization_methods:
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "with_final_orbital_rotation": False,
            # "param_initialization": "ccsd",
            "optimization_method": optimization_method,
        }
        this_data[optimization_method] = data.xs(
            tuple(settings.values()), level=tuple(settings.keys())
        )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    fig.subplots_adjust(wspace=0.25)

    for optimization_method, marker in zip(optimization_methods, markers):
        ax1.plot(
            bond_distance_range,
            this_data[optimization_method]["nit"].values,
            f"{marker}--",
            label=f"LUCJ, L={n_reps}, {optimization_method}",
        )
    ax1.legend()
    # ax.set_yscale("log")
    ax1.set_ylabel("Number of iterations")

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

    ax1.set_title(r"Ethene dissociation STO-6g (4e, 4o)" + f", {connectivity}")

    plt.show()
    plt.savefig(f"{plots_dir}/{filename}.svg")
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

    ax.set_title(title)

    dirname = os.path.join(
        plots_dir,
        molecule_basename,
    )
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, "reference_curves.svg")
    plt.savefig(filename)


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
            # "param_initialization": "ccsd",
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

    # dirname = os.path.join(
    #     plots_dir,
    #     molecule_basename,
    #     f"npts-{n_pts}",
    #     f"n_reps-{n_reps}",
    # )
    # os.makedirs(dirname, exist_ok=True)
    # filename = os.path.join(dirname, f"{connectivity}.svg")
    # plt.savefig(filename)
    # plt.close()


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

        overlap_mats = np.stack(info["overlap_mat"])
        nit = len(overlap_mats)
        step = nit // (n_mat - 1)
        mats = list(overlap_mats[::step])
        iteration_nums = list(range(0, nit, step))
        assert len(mats) in (n_mat, n_mat - 1)
        # if len(mats) == n_mat - 1:
        #     mats.append(overlap_mats[-1])
        #     iteration_nums.append(nit - 1)

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
