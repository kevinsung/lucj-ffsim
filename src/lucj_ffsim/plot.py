import itertools
import os

import ffsim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lucj_ffsim.lucj import get_lucj_indices


def plot_optimization_iterations(
    filename: str,
    title: str,
    data: pd.DataFrame,
    bond_distance_range: np.ndarray,
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
    ax1.axhline(1.6e-3, linestyle="--", color="gray")
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

    plt.savefig(filename)

    plt.close()


def plot_reference_curves(
    filename: str,
    title: str,
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

    plt.savefig(filename)
    plt.close()


def plot_optimization_method(
    filename: str,
    title: str,
    data: pd.DataFrame,
    reference_curves_bond_distance_range: np.ndarray,
    hf_energies_reference: np.ndarray,
    fci_energies_reference: np.ndarray,
    bond_distance_range: np.ndarray,
    optimization_methods: list[str],
    with_final_orbital_rotation_choices: list[bool],
    connectivity: str,
    n_reps: int,
):
    # effect of optimization method
    markers = ["o", "s", "v", "D", "p", "*"]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    alphas = [1.0, 0.5]
    linestyles = ["--", ":"]

    this_data = {}
    for optimization_method, with_final_orbital_rotation in itertools.product(
        optimization_methods, with_final_orbital_rotation_choices
    ):
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "with_final_orbital_rotation": with_final_orbital_rotation,
            "optimization_method": optimization_method,
        }
        this_data[optimization_method, with_final_orbital_rotation] = data.xs(
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
    for optimization_method, marker, color in zip(
        optimization_methods, markers, colors
    ):
        for with_final_orbital_rotation, alpha, linestyle in zip(
            with_final_orbital_rotation_choices, alphas, linestyles
        ):
            ax1.plot(
                bond_distance_range,
                this_data[optimization_method, with_final_orbital_rotation][
                    "energy"
                ].values,
                f"{marker}{linestyle}",
                color=color,
                alpha=alpha,
                label=f"LUCJ + orb opt, {optimization_method}"
                if with_final_orbital_rotation
                else f"LUCJ, {optimization_method}",
            )
    ax1.legend()
    ax1.set_ylabel("Energy (Hartree)")

    for optimization_method, marker, color in zip(
        optimization_methods, markers, colors
    ):
        for with_final_orbital_rotation, alpha, linestyle in zip(
            with_final_orbital_rotation_choices, alphas, linestyles
        ):
            ax2.plot(
                bond_distance_range,
                this_data[optimization_method, with_final_orbital_rotation][
                    "error"
                ].values,
                f"{marker}{linestyle}",
                color=color,
                alpha=alpha,
                label=f"LUCJ + orb opt, {optimization_method}"
                if with_final_orbital_rotation
                else f"LUCJ, {optimization_method}",
            )
    ax2.set_yscale("log")
    ax2.axhline(1.6e-3, linestyle="--", color="gray")
    ax2.legend()
    ax2.set_ylabel("Energy error (Hartree)")

    for optimization_method, marker, color in zip(
        optimization_methods, markers, colors
    ):
        for with_final_orbital_rotation, alpha, linestyle in zip(
            with_final_orbital_rotation_choices, alphas, linestyles
        ):
            ax3.plot(
                bond_distance_range,
                this_data[optimization_method, with_final_orbital_rotation][
                    "spin_squared"
                ].values,
                f"{marker}{linestyle}",
                color=color,
                alpha=alpha,
                label=f"LUCJ + orb opt, {optimization_method}"
                if with_final_orbital_rotation
                else f"LUCJ, {optimization_method}",
            )
    ax3.axhline(0, linestyle="--", color="gray")
    ax3.legend()
    ax3.set_ylabel("Spin squared")
    ax3.set_xlabel("Bond length (Å)")

    for optimization_method, marker, color in zip(
        optimization_methods, markers, colors
    ):
        for with_final_orbital_rotation, alpha, linestyle in zip(
            with_final_orbital_rotation_choices, alphas, linestyles
        ):
            ax4.plot(
                bond_distance_range,
                this_data[optimization_method, with_final_orbital_rotation][
                    "nit"
                ].values,
                f"{marker}{linestyle}",
                color=color,
                alpha=alpha,
                label=f"LUCJ + orb opt, {optimization_method}"
                if with_final_orbital_rotation
                else f"LUCJ, {optimization_method}",
            )
    ax4.legend()
    ax4.set_ylabel("Number of iterations")

    fig.suptitle(title)

    plt.savefig(filename)

    plt.close()


def plot_info(
    filename: str,
    data: pd.DataFrame,
    reference_curves_bond_distance_range: np.ndarray,
    hf_energies_reference: np.ndarray,
    fci_energies_reference: np.ndarray,
    bond_distance_range: np.ndarray,
    optimization_methods: list[str],
    with_final_orbital_rotation: bool,
    connectivity: str,
    n_reps_range: list[int],
    markers: list[str] | None = None,
    colors: list[str] | None = None,
):
    if markers is None:
        markers = ["o", "s", "v", "D", "p", "*"]
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
    alphas = [0.5, 1.0]
    linestyles = ["--", ":"]

    this_data = {}
    for n_reps, optimization_method in itertools.product(
        n_reps_range, optimization_methods
    ):
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "with_final_orbital_rotation": with_final_orbital_rotation,
            "optimization_method": optimization_method,
        }
        df = data.xs(tuple(settings.values()), level=tuple(settings.keys()))
        min_energy_indices = df.groupby(level="bond_distance")["energy"].idxmin()
        this_data[n_reps, optimization_method] = df.loc[min_energy_indices]

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        layout="constrained",
    )

    ax1.plot(
        reference_curves_bond_distance_range,
        fci_energies_reference,
        "-",
        label="FCI",
        color="black",
    )

    legend = {"linear-method": "linear method"}

    for n_reps, marker, color in zip(n_reps_range, markers, colors):
        for optimization_method, alpha in zip(optimization_methods, alphas):
            ax1.plot(
                bond_distance_range,
                this_data[n_reps, optimization_method]["energy"].values,
                f"{marker}{linestyles[0]}",
                color=color,
                alpha=alpha,
                label=f"LUCJ, {legend.get(optimization_method, optimization_method)}, L={n_reps}",
            )
    ax1.legend()
    ax1.set_ylabel("Energy (Hartree)")
    ax1.set_xlabel("Bond length (Å)")

    for n_reps, marker, color in zip(n_reps_range, markers, colors):
        for optimization_method, alpha in zip(optimization_methods, alphas):
            ax2.plot(
                bond_distance_range,
                this_data[n_reps, optimization_method]["error"].values,
                f"{marker}{linestyles[0]}",
                color=color,
                alpha=alpha,
                label=f"LUCJ, {legend.get(optimization_method, optimization_method)}, L={n_reps}",
            )
    ax2.set_yscale("log")
    ax2.set_ylim(1e-5, 1e-2)
    ax2.axhline(1.6e-3, linestyle="--", color="gray")
    ax2.legend()
    ax2.set_ylabel("Energy error (Hartree)")
    ax2.set_xlabel("Bond length (Å)")

    for n_reps, marker, color in zip(n_reps_range, markers, colors):
        for optimization_method, alpha in zip(optimization_methods, alphas):
            ax3.plot(
                bond_distance_range,
                this_data[n_reps, optimization_method]["spin_squared"].values,
                f"{marker}{linestyles[0]}",
                color=color,
                alpha=alpha,
                label=f"LUCJ, {legend.get(optimization_method, optimization_method)}, L={n_reps}",
            )
    ax3.set_ylim(0, 1e-1)
    ax3.legend()
    ax3.set_ylabel("Spin squared")
    ax3.set_xlabel("Bond length (Å)")

    for n_reps, marker, color in zip(n_reps_range, markers, colors):
        for optimization_method, alpha in zip(optimization_methods, alphas):
            ax4.plot(
                bond_distance_range,
                this_data[n_reps, optimization_method]["nit"].values,
                f"{marker}{linestyles[0]}",
                color=color,
                alpha=alpha,
                label=f"LUCJ, {legend.get(optimization_method, optimization_method)}, L={n_reps}",
            )
    # ax4.set_ylim(0, 1000)
    ax4.legend()
    ax4.set_ylabel("Number of iterations")
    ax4.set_xlabel("Bond length (Å)")

    plt.savefig(filename)

    plt.close()


def plot_bootstrap_iteration(
    filename: str,
    title: str,
    data: pd.DataFrame,
    reference_curves_bond_distance_range: np.ndarray,
    hf_energies_reference: np.ndarray,
    fci_energies_reference: np.ndarray,
    bond_distance_range: np.ndarray,
    optimization_method: str,
    with_final_orbital_rotation: bool,
    bootstrap_iterations: list[int],
    connectivity: str,
    n_reps: int,
):
    # effect of optimization method
    markers = ["o", "s", "v", "D", "p", "*"]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    linestyles = ["--", ":"]

    this_data = {}
    for bootstrap_iteration in bootstrap_iterations:
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "with_final_orbital_rotation": with_final_orbital_rotation,
            "optimization_method": optimization_method,
            "bootstrap_iteration": bootstrap_iteration,
        }
        this_data[optimization_method, bootstrap_iteration] = data.xs(
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
    for bootstrap_iteration, marker, color in zip(
        bootstrap_iterations, markers, colors
    ):
        ax1.plot(
            bond_distance_range,
            this_data[optimization_method, bootstrap_iteration]["energy"].values,
            f"{marker}{linestyles[0]}",
            color=color,
            label=f"LUCJ + orb opt, iter {bootstrap_iteration}"
            if with_final_orbital_rotation
            else f"LUCJ, iter {bootstrap_iteration}",
        )
    ax1.legend()
    ax1.set_ylabel("Energy (Hartree)")

    for bootstrap_iteration, marker, color in zip(
        bootstrap_iterations, markers, colors
    ):
        ax2.plot(
            bond_distance_range,
            this_data[optimization_method, bootstrap_iteration]["error"].values,
            f"{marker}{linestyles[0]}",
            color=color,
            label=f"LUCJ + orb opt, iter {bootstrap_iteration}"
            if with_final_orbital_rotation
            else f"LUCJ, iter {bootstrap_iteration}",
        )
    ax2.set_yscale("log")
    ax2.axhline(1.6e-3, linestyle="--", color="gray")
    ax2.legend()
    ax2.set_ylabel("Energy error (Hartree)")

    for bootstrap_iteration, marker, color in zip(
        bootstrap_iterations, markers, colors
    ):
        ax3.plot(
            bond_distance_range,
            this_data[optimization_method, bootstrap_iteration]["spin_squared"].values,
            f"{marker}{linestyles[0]}",
            color=color,
            label=f"LUCJ + orb opt, iter {bootstrap_iteration}"
            if with_final_orbital_rotation
            else f"LUCJ, iter {bootstrap_iteration}",
        )
    ax3.axhline(0, linestyle="--", color="gray")
    ax3.legend()
    ax3.set_ylabel("Spin squared")
    ax3.set_xlabel("Bond length (Å)")

    for bootstrap_iteration, marker, color in zip(
        bootstrap_iterations, markers, colors
    ):
        ax4.plot(
            bond_distance_range,
            this_data[optimization_method, bootstrap_iteration]["nit"].values,
            f"{marker}{linestyles[0]}",
            color=color,
            label=f"LUCJ + orb opt, iter {bootstrap_iteration}"
            if with_final_orbital_rotation
            else f"LUCJ, iter {bootstrap_iteration}",
        )
    ax4.legend()
    ax4.set_ylabel("Number of iterations")

    fig.suptitle(title)

    plt.savefig(filename)

    plt.close()


def plot_error(
    filename: str,
    data: pd.DataFrame,
    bond_distance_range: np.ndarray,
    optimization_methods: list[str],
    with_final_orbital_rotation: bool,
    connectivity: str,
    n_reps_range: list[int],
    ymin: float | None = None,
    ymax: float | None = None,
    markers: list[str] | None = None,
    colors: list[str] | None = None,
):
    if markers is None:
        markers = ["o", "s", "v", "D", "p", "*"]
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
    alphas = [0.5, 1.0]
    linestyles = ["--", ":"]

    this_data = {}
    for n_reps, optimization_method in itertools.product(
        n_reps_range, optimization_methods
    ):
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "with_final_orbital_rotation": with_final_orbital_rotation,
            "optimization_method": optimization_method,
        }
        df = data.xs(tuple(settings.values()), level=tuple(settings.keys()))
        min_energy_indices = df.groupby(level="bond_distance")["energy"].idxmin()
        this_data[n_reps, optimization_method] = df.loc[min_energy_indices]

    _, ax = plt.subplots(layout="constrained")

    legend = {"linear-method": "linear method"}
    for n_reps, marker, color in zip(n_reps_range, markers, colors):
        for optimization_method, alpha in zip(optimization_methods, alphas):
            ax.plot(
                bond_distance_range,
                this_data[n_reps, optimization_method]["error"].values,
                f"{marker}{linestyles[0]}",
                color=color,
                alpha=alpha,
                label=f"{legend.get(optimization_method, optimization_method)}, L={n_reps}",
            )
    ax.set_yscale("log")
    if ymin and ymax:
        ax.set_ylim(ymin, ymax)
    ax.axhline(1.6e-3, linestyle="--", color="gray")
    ax.legend()
    ax.set_ylabel("Energy error (Hartree)")
    ax.set_xlabel("Bond length (Å)")

    plt.savefig(filename)

    plt.close()


def plot_energy(
    filename: str,
    data: pd.DataFrame,
    reference_curves_bond_distance_range: np.ndarray,
    hf_energies_reference: np.ndarray | None,
    fci_energies_reference: np.ndarray,
    bond_distance_range: np.ndarray,
    optimization_method: str,
    with_final_orbital_rotation: bool,
    connectivity: str,
    n_reps_range: list[int],
    ymin: float | None = None,
    ymax: float | None = None,
    markers: list[str] | None = None,
    colors: list[str] | None = None,
):
    if markers is None:
        markers = ["o", "s", "v", "D", "p", "*"]
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

    this_data = {}
    for n_reps in n_reps_range:
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "with_final_orbital_rotation": with_final_orbital_rotation,
            "optimization_method": optimization_method,
        }
        df = data.xs(tuple(settings.values()), level=tuple(settings.keys()))
        min_energy_indices = df.groupby(level="bond_distance")["energy"].idxmin()
        this_data[n_reps, optimization_method] = df.loc[min_energy_indices]

    _, ax = plt.subplots(layout="constrained")

    if hf_energies_reference is not None:
        ax.plot(
            reference_curves_bond_distance_range,
            hf_energies_reference,
            "--",
            label="HF",
            color="blue",
        )
    ax.plot(
        reference_curves_bond_distance_range,
        fci_energies_reference,
        "-",
        label="FCI",
        color="black",
    )
    for n_reps, marker, color in zip(n_reps_range, markers, colors):
        ax.plot(
            bond_distance_range,
            this_data[n_reps, optimization_method]["energy"].values,
            f"{marker}--",
            color=color,
            label=f"LUCJ, L={n_reps}",
        )
    if ymin and ymax:
        ax.set_ylim(ymin, ymax)
    ax.legend(loc="upper right")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_xlabel("Bond length (Å)")

    plt.savefig(filename)

    plt.close()


def plot_error_and_iterations(
    filename: str,
    title: str,
    data: pd.DataFrame,
    bond_distance_range: np.ndarray,
    optimization_methods: list[str],
    with_final_orbital_rotation: bool,
    connectivity: str,
    n_reps_range: list[int],
):
    # effect of optimization method
    markers = ["o", "s", "v", "D", "p", "*"]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    alphas = [1.0, 0.5]
    linestyles = ["--", ":"]

    this_data = {}
    for n_reps, optimization_method in itertools.product(
        n_reps_range, optimization_methods
    ):
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "with_final_orbital_rotation": with_final_orbital_rotation,
            "optimization_method": optimization_method,
        }
        df = data.xs(tuple(settings.values()), level=tuple(settings.keys()))
        min_energy_indices = df.groupby(level="bond_distance")["energy"].idxmin()
        this_data[n_reps, optimization_method] = df.loc[min_energy_indices]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    # fig.subplots_adjust(wspace=0.25)

    for n_reps, color in zip(n_reps_range, colors):
        for optimization_method, marker, alpha in zip(
            optimization_methods, markers, alphas
        ):
            ax1.plot(
                bond_distance_range,
                this_data[n_reps, optimization_method]["error"].values,
                f"{marker}{linestyles[0]}",
                color=color,
                alpha=alpha,
                label=f"{optimization_method}, L={n_reps}",
            )
    ax1.set_yscale("log")
    ax1.axhline(1.6e-3, linestyle="--", color="gray")
    ax1.legend()
    ax1.set_ylabel("Energy error (Hartree)")

    for n_reps, color in zip(n_reps_range, colors):
        for optimization_method, marker, alpha in zip(
            optimization_methods, markers, alphas
        ):
            ax2.plot(
                bond_distance_range,
                this_data[n_reps, optimization_method]["nit"].values,
                f"{marker}{linestyles[0]}",
                color=color,
                alpha=alpha,
                label=f"{optimization_method}, L={n_reps}",
            )
    ax2.legend()
    ax2.set_ylabel("Number of iterations")

    fig.suptitle(title)

    plt.savefig(filename)

    plt.close()


def plot_lm_hyperparameter(
    filename: str,
    title: str,
    data: pd.DataFrame,
    bond_distance_range: np.ndarray,
    linear_method_regularizations: list[float],
    linear_method_variations: list[float],
    connectivity: str,
    n_reps: int,
):
    # effect of optimization method
    markers = ["o", "s", "v", "D", "p", "*"]

    this_data = {}
    settings = {
        "connectivity": connectivity,
        "n_reps": n_reps,
        "optimization_method": "linear-method",
        "linear_method_regularization": None,
        "linear_method_variation": None,
        "with_final_orbital_rotation": False,
    }
    this_data[None, None] = data.xs(
        tuple(settings.values()), level=tuple(settings.keys())
    )
    for regularization, variation in itertools.product(
        linear_method_regularizations, linear_method_variations
    ):
        settings = {
            "connectivity": connectivity,
            "n_reps": n_reps,
            "optimization_method": "linear-method",
            "linear_method_regularization": regularization,
            "linear_method_variation": variation,
            "with_final_orbital_rotation": False,
        }
        this_data[regularization, variation] = data.xs(
            tuple(settings.values()), level=tuple(settings.keys())
        )

    fig, axes = plt.subplots(
        2,
        len(linear_method_regularizations),
        figsize=(6 * len(linear_method_regularizations), 12),
    )

    for ax, regularization in zip(axes[0], linear_method_regularizations):
        ax.plot(
            bond_distance_range,
            this_data[None, None]["error"].values,
            f"{markers[0]}--",
            color="darkgray",
            label="opt",
        )
        for variation, marker in zip(linear_method_variations, markers[1:]):
            ax.plot(
                bond_distance_range,
                this_data[regularization, variation]["error"].values,
                f"{marker}--",
                label=f"variation = {variation}",
            )
        ax.set_yscale("log")
        ax.axhline(1.6e-3, linestyle="--", color="gray")
        ax.legend()
        ax.set_ylabel("Energy error (Hartree)")
        ax.set_title(f"regularization = {regularization}")

    for ax, regularization in zip(axes[1], linear_method_regularizations):
        ax.plot(
            bond_distance_range,
            this_data[None, None]["spin_squared"].values,
            f"{markers[0]}--",
            color="darkgray",
            label="opt",
        )
        for variation, marker in zip(linear_method_variations, markers[1:]):
            ax.plot(
                bond_distance_range,
                this_data[regularization, variation]["spin_squared"].values,
                f"{marker}--",
                label=f"variation = {variation}",
            )
        ax.axhline(0, linestyle="--", color="gray")
        ax.legend()
        ax.set_ylabel("Spin squared")
        ax.set_xlabel("Bond length (Å)")
        ax.set_title(f"regularization = {regularization}")

    fig.suptitle(title)

    plt.savefig(filename)

    plt.close()


def plot_n_reps(
    filename: str,
    title: str,
    data: pd.DataFrame,
    reference_curves_bond_distance_range: np.ndarray,
    hf_energies_reference: np.ndarray,
    fci_energies_reference: np.ndarray,
    bond_distance_range: np.ndarray,
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
    ax2.axhline(1.6e-3, linestyle="--", color="gray")
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

    plt.savefig(filename)

    plt.close()


def plot_overlap_mats(
    filename: str,
    title: str,
    infos: dict,
    bond_distance_range: np.ndarray,
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

    plt.savefig(filename)

    plt.close()


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


def plot_parameters_distance(
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
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    result = results[
        bond_distance_range[0], connectivity, n_reps, False, optimization_method
    ]
    initial_params = result.x
    current_params = result.x

    distance = []
    change = []
    for bond_distance in bond_distance_range[1:]:
        result = results[
            bond_distance, connectivity, n_reps, False, optimization_method
        ]
        params = result.x
        distance.append(np.linalg.norm(params - initial_params))
        change.append(np.linalg.norm(params - current_params))
        current_params = params

    ax0, ax1 = axes
    ax0.plot(bond_distance_range[1:], change, "o--")
    ax0.set_xticks(bond_distance_range[1:])
    ax0.set_yscale("log")
    ax0.set_title(r"$|x_t - x_{t-1}|$")
    ax1.plot(bond_distance_range[1:], distance, "o--")
    ax1.set_xticks(bond_distance_range[1:])
    ax1.set_yscale("log")
    ax1.set_title(r"$|x_t - x_0|$")
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


def plot_initial_parameters_distance(
    plots_dir: str,
    title: str,
    molecule_basename: str,
    mol_datas: dict[float, ffsim.MolecularData],
    bond_distance_range: np.ndarray,
    n_pts: int,
    connectivity: str,
    n_reps: int,
    with_final_orbital_rotation: bool,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    alpha_alpha_indices, alpha_beta_indices = get_lucj_indices(
        connectivity, next(iter(mol_datas.values())).norb
    )

    all_params = []
    for bond_distance in bond_distance_range:
        mol_data = mol_datas[bond_distance]
        op = ffsim.UCJOperator.from_t_amplitudes(
            mol_data.ccsd_t2,
            n_reps=n_reps,
            t1_amplitudes=mol_data.ccsd_t1 if with_final_orbital_rotation else None,
        )
        params = op.to_parameters(
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
        )
        all_params.append(params)

    current_params = all_params[0]

    change = []
    for params in all_params[1:]:
        change.append(np.linalg.norm(params - current_params))
        current_params = params

    ax.plot(bond_distance_range[1:], change, "o--")
    ax.set_xticks(bond_distance_range[1:])
    ax.set_yscale("log")
    ax.set_title(r"$|x_t - x_{t-1}|$")
    fig.suptitle(title)

    dirname = os.path.join(
        plots_dir,
        molecule_basename,
        f"npts-{n_pts}",
        connectivity,
    )
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, f"initial_parameters_n_reps-{n_reps}.svg")
    plt.savefig(filename)


# def plot_parameters(
#     plots_dir: str,
#     title: str,
#     infos: dict,
#     molecule_basename: str,
#     bond_distances: np.ndarray,
#     n_pts: int,
#     connectivity: str,
#     n_reps: int,
# ):
#     n_plots = 5

#     fig, axes = plt.subplots(
#         len(bond_distances), n_plots, figsize=(6 * n_plots, 6 * len(bond_distances))
#     )

#     for these_axes, bond_distance in zip(axes, bond_distances):
#         info = infos[
#             bond_distance,
#             connectivity,
#             n_reps,
#             False,
#             "linear-method",
#         ]

#         xs = info["x"]
#         nit = len(xs)
#         step = nit // (n_plots - 1)
#         indices = list(range(0, nit, step))
#         while len(indices) > n_plots - 1:
#             indices.pop()
#         if nit - 1 not in indices:
#             indices.append(nit - 1)
#         these_xs = [xs[i] for i in indices]

#         for ax, mat, i in zip(these_axes, these_xs, indices):
#             max_val = np.max(np.abs(mat))
#             im = ax.matshow(
#                 mat,
#                 cmap="bwr",
#                 vmin=-max_val,
#                 vmax=max_val,
#             )
#             ax.set_title(f"Iteration {i}")
#             if i == 0:
#                 ax.set_ylabel(f"d = {bond_distance}")
#             fig.colorbar(im)

#         fig.suptitle(title)

#     dirname = os.path.join(
#         plots_dir,
#         molecule_basename,
#         f"npts-{n_pts}",
#         connectivity,
#     )
#     os.makedirs(dirname, exist_ok=True)
#     filename = os.path.join(dirname, f"overlap_mat_n_reps-{n_reps}.svg")
#     plt.savefig(filename)
