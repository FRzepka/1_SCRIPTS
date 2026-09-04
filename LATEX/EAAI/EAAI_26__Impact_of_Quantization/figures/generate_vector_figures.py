#!/usr/bin/env python3
"""Generate the baseline paper figures as vector PDFs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


FIGURES_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURES_DIR.parents[3]
STATEFUL_RESULTS = (
    REPOSITORY_ROOT
    / "DL_Models"
    / "LFP_SOH_Optimization_Study"
    / "5_benchmark"
    / "batmm"
    / "LFP_SOH_Optimization_Study"
    / "5_benchmark"
    / "Stateful_Base_Comparison"
    / "results"
)
CAPACITY_RESULTS = (
    REPOSITORY_ROOT
    / "DL_Models"
    / "LFP_SOH_Optimization_Study"
    / "5_benchmark"
    / "SOH_Comparison_Base"
    / "results"
    / "FINAL_RESULTS"
)

MODEL_ORDER = ["cnn", "gru", "lstm", "tcn"]
MODEL_LABELS = {model: model.upper() for model in MODEL_ORDER}
COLORS = {
    "cnn": "#59C7C2",
    "gru": "#59E83A",
    "lstm": "#E76B91",
    "tcn": "#294862",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 18,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 24,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.2,
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def blend_with_white(color: str, white_fraction: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb * (1.0 - white_fraction) + white_fraction)


def save_pdf(fig: plt.Figure, name: str) -> None:
    fig.savefig(
        FIGURES_DIR / f"{name}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def render_model_comparison() -> None:
    metrics = pd.read_csv(STATEFUL_RESULTS / "metrics_summary.csv")
    metrics = metrics[metrics["aggregation"] == "cell_macro"].set_index("model")
    metrics = metrics.loc[MODEL_ORDER]
    inventory = pd.read_csv(STATEFUL_RESULTS / "model_inventory.csv").set_index("model")
    inventory = inventory.loc[MODEL_ORDER]

    fig, (ax_error, ax_size) = plt.subplots(1, 2, figsize=(20, 9.2))
    fig.subplots_adjust(left=0.073, right=0.985, top=0.91, bottom=0.18, wspace=0.23)
    positions = np.arange(len(MODEL_ORDER))
    width = 0.25

    for index, model in enumerate(MODEL_ORDER):
        color = COLORS[model]
        mae = float(metrics.loc[model, "mae"])
        rmse = float(metrics.loc[model, "rmse"])
        ax_error.bar(
            index - width / 1.6,
            mae,
            width,
            color=blend_with_white(color, 0.52),
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        ax_error.bar(
            index + width / 1.6,
            rmse,
            width,
            color=blend_with_white(color, 0.04),
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        ax_error.text(index - width / 1.6, mae + 0.00065, f"{mae:.4f}", ha="center", fontsize=24)
        ax_error.text(index + width / 1.6, rmse + 0.00065, f"{rmse:.4f}", ha="center", fontsize=24)

        parameters = float(inventory.loc[model, "parameters"])
        size_mib = parameters * 4.0 / (1024.0**2)
        ax_size.bar(
            index,
            size_mib,
            0.52,
            color=blend_with_white(color, 0.28),
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        ax_size.text(index, size_mib + 0.11, f"{size_mib:.3f}", ha="center", fontsize=24)

    max_error = max(float(metrics["mae"].max()), float(metrics["rmse"].max()))
    error_limit = np.ceil((max_error + 0.002) / 0.005) * 0.005
    ax_error.set_ylim(0.0, error_limit)
    ax_error.set_yticks(np.arange(0.0, error_limit + 0.0001, 0.005))
    ax_error.set_ylabel("SOH error [0-1]", fontsize=30)
    ax_error.set_xlabel("Architecture", fontsize=30, labelpad=24)
    ax_error.set_xticks(positions, [MODEL_LABELS[name] for name in MODEL_ORDER], fontweight="bold", fontsize=25)
    ax_error.tick_params(axis="y", labelsize=25)
    ax_error.legend(
        handles=[
            Patch(facecolor="#C6C6C6", edgecolor="#777777", label="MAE"),
            Patch(facecolor="#777777", edgecolor="#777777", label="RMSE"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.56, 1.13),
        ncol=2,
        frameon=False,
        fontsize=31,
    )

    ax_size.set_ylim(0.0, 4.0)
    ax_size.set_yticks(np.arange(0, 4.1, 1.0))
    ax_size.set_ylabel("FP32 weights [MiB]", fontsize=30)
    ax_size.set_xlabel("Architecture", fontsize=30, labelpad=24)
    ax_size.set_xticks(positions, [MODEL_LABELS[name] for name in MODEL_ORDER], fontweight="bold", fontsize=25)
    ax_size.tick_params(axis="y", labelsize=25)

    for label, axis in zip(("(a)", "(b)"), (ax_error, ax_size)):
        axis.text(-0.055, 1.045, label, transform=axis.transAxes, fontsize=17)
        axis.grid(axis="y", color="#D5D8DC", linestyle=(0, (4, 3)), linewidth=1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    save_pdf(fig, "baseline_model_comparison")


def render_selected_trajectory() -> None:
    trajectory = pd.read_csv(STATEFUL_RESULTS / "soh_trajectory_C11.csv")
    fig, axis = plt.subplots(figsize=(20, 10))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.955, bottom=0.14)

    for model in MODEL_ORDER:
        prediction = trajectory[f"soh_{model}"].rolling(window=7, center=True, min_periods=1).median()
        axis.plot(
            trajectory["time_h"].iloc[::3],
            prediction.iloc[::3],
            color=COLORS[model],
            linewidth=2.2,
            label=MODEL_LABELS[model],
            zorder=3,
        )

    axis.plot(
        trajectory["time_h"],
        trajectory["soh_reference"],
        color="#191919",
        linewidth=3.0,
        label="Reference SOH",
        zorder=4,
    )
    handles, labels = axis.get_legend_handles_labels()
    order = [labels.index("Reference SOH")] + [labels.index(MODEL_LABELS[name]) for name in MODEL_ORDER]
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#AFAFAF",
        fontsize=31,
    )
    axis.set_xlim(0, float(trajectory["time_h"].max()))
    axis.set_ylim(0.62, 1.00)
    axis.set_xlabel("Time [h]", fontsize=30, labelpad=17)
    axis.set_ylabel("SOH [0-1]", fontsize=30)
    axis.tick_params(axis="both", labelsize=26)
    axis.grid(color="#DADDE0", linewidth=1.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    save_pdf(fig, "selected_baseline_soh_trajectory")


def render_capacity_sensitivity() -> None:
    capacity = pd.read_csv(CAPACITY_RESULTS / "baseline_capacity_sensitivity_data.csv")
    for model in MODEL_ORDER:
        family = capacity[capacity["architecture"] == MODEL_LABELS[model]].sort_values("parameters")
        fig, axis = plt.subplots(figsize=(7.2, 5.2))
        fig.subplots_adjust(left=0.16, right=0.975, top=0.95, bottom=0.18)
        axis.plot(
            family["parameters"],
            family["mae"],
            color=COLORS[model],
            marker="o",
            markersize=12,
            linewidth=2.8,
            zorder=3,
        )
        for row in family.itertuples(index=False):
            axis.annotate(
                row.variant,
                (row.parameters, row.mae),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=17,
            )
        axis.set_xlabel("Parameters", labelpad=12)
        axis.set_ylabel("MAE", labelpad=10)
        axis.grid(color="#C8C8C8", alpha=0.45, linewidth=1.1)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
        axis.margins(x=0.09)
        axis.set_ylim(0.010, 0.036)
        axis.set_yticks(np.arange(0.010, 0.0351, 0.005))
        save_pdf(fig, f"baseline_capacity_sensitivity_{model}")


def main() -> None:
    configure_style()
    render_model_comparison()
    render_selected_trajectory()
    render_capacity_sensitivity()
    print(f"Generated vector PDF figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
