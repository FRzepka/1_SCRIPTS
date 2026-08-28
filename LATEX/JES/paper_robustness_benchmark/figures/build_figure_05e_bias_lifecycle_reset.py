from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

RESULTS_CODE = (
    Path(__file__).resolve().parents[4]
    / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results"
)
sys.path.insert(0, str(RESULTS_CODE))

from jes2_plot_style import MODEL_COLORS, clean_axes, save_figure, setup_style


MODELS = ("DM", "HDM", "HECM", "DD")


def minmax_error(values: pd.Series, mean: float) -> np.ndarray:
    lower = min(float(values.min()), mean)
    upper = max(float(values.max()), mean)
    return np.array([[mean - lower], [upper - mean]])


def draw_model_bar(ax, x: float, height: float, model: str, alpha: float, width: float) -> None:
    ax.bar(
        x,
        height,
        width=width,
        facecolor=to_rgba(MODEL_COLORS[model], alpha),
        edgecolor=MODEL_COLORS[model],
        linewidth=1.5,
        zorder=2,
    )


def main() -> None:
    paper = Path(__file__).resolve().parents[1]
    default_results = paper / "JES_2.0/results"
    parser = argparse.ArgumentParser(
        description="Build the time-resolved lifecycle current-bias and recovery figure."
    )
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            paper
            / "figures/Results/All Cells/Figure_06_Current_Bias_Lifecycle_Reset.png"
        ),
    )
    args = parser.parse_args()

    normalized = pd.read_csv(args.results_dir / "c29_bias_normalized_cycle.csv")
    lifecycle = pd.read_csv(args.results_dir / "c29_bias_lifecycle_24h.csv")
    events = pd.read_csv(args.results_dir / "c29_bias_reset_event_metrics.csv")
    cycles = pd.read_csv(args.results_dir / "c29_bias_inter_reset_cycles.csv")
    summary = pd.read_csv(args.results_dir / "c29_bias_temporal_model_summary.csv")

    setup_style()
    fig = plt.figure(figsize=(13.2, 9.2))
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
    ax_life = fig.add_subplot(grid[0, 0])
    ax_cycle = fig.add_subplot(grid[0, 1])
    ax_reset = fig.add_subplot(grid[1, 0])
    ax_rank = fig.add_subplot(grid[1, 1])

    for model in MODELS:
        part = lifecycle[lifecycle.model == model].sort_values("elapsed_h")
        ax_life.plot(
            part.elapsed_h,
            part.delta_mae_24h,
            color=MODEL_COLORS[model],
            linewidth=1.8,
            label=model,
        )
    ax_life.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax_life.set_xlabel("Elapsed lifecycle time [h]")
    ax_life.set_ylabel(r"Bias contribution $\Delta$MAE [SOC]")
    ax_life.set_title("(a) Bias Penalty over Full Life")
    ax_life.legend(ncol=2, frameon=False, loc="best")
    ax_life.text(
        0.99,
        0.03,
        "24-h rolling mean; exact 5-min MAE bins",
        transform=ax_life.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="#666666",
    )
    clean_axes(ax_life)

    error_positions = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    for model in MODELS:
        part = normalized[normalized.model == model].sort_values("progress_pct")
        ax_cycle.plot(
            part.progress_pct,
            part.delta_mae_mean,
            color=MODEL_COLORS[model],
            linewidth=2.0,
            label=model,
            zorder=3,
        )
        error = part[part.progress_pct.isin(error_positions)]
        yerr = np.vstack(
            (
                error.delta_mae_mean - error.delta_mae_min,
                error.delta_mae_max - error.delta_mae_mean,
            )
        )
        ax_cycle.errorbar(
            error.progress_pct,
            error.delta_mae_mean,
            yerr=yerr,
            fmt="o",
            markersize=4.2,
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.0,
            capsize=3.0,
            capthick=1.0,
            zorder=4,
        )
    ax_cycle.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax_cycle.set_xlim(-2.0, 102.0)
    ax_cycle.set_xticks(error_positions)
    ax_cycle.set_xlabel("Progress between consecutive full-charge events [%]")
    ax_cycle.set_ylabel(r"Bias contribution $\Delta$MAE [SOC]")
    ax_cycle.set_title("(b) Bias Accumulation between Full Charges")
    ax_cycle.text(
        0.99,
        0.96,
        "Mean and min-max range over 44 intervals",
        transform=ax_cycle.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
        color="#666666",
    )
    clean_axes(ax_cycle)

    x = np.arange(len(MODELS), dtype=float)
    width = 0.31
    for model_index, model in enumerate(MODELS):
        part = events[events.model == model]
        for offset, phase, alpha in ((-width / 2, "before", 0.14), (width / 2, "after", 0.38)):
            values = part[part.phase == phase].delta_mae
            mean = float(values.mean())
            position = x[model_index] + offset
            draw_model_bar(ax_reset, position, mean, model, alpha, width)
            ax_reset.errorbar(
                position,
                mean,
                yerr=minmax_error(values, mean),
                fmt="o",
                color=MODEL_COLORS[model],
                ecolor=MODEL_COLORS[model],
                markersize=3.8,
                elinewidth=1.0,
                capsize=3.0,
                zorder=4,
            )
    ax_reset.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax_reset.set_xticks(x, MODELS)
    ax_reset.set_ylabel(r"Bias contribution $\Delta$MAE [SOC]")
    ax_reset.set_title("(c) Bias before and after Full Charge")
    ax_reset.legend(
        handles=[
            Patch(facecolor=to_rgba("#666666", 0.14), edgecolor="#666666", label="Before"),
            Patch(facecolor=to_rgba("#666666", 0.38), edgecolor="#666666", label="After"),
        ],
        frameon=False,
        loc="best",
    )
    clean_axes(ax_reset)

    summary_by_model = summary.set_index("model")
    for model_index, model in enumerate(MODELS):
        model_cycles = cycles[cycles.model == model]
        for offset, column, label, alpha in (
            (-width / 2, "baseline_mae", "Baseline", 0.14),
            (width / 2, "biased_mae", "Worst ±3% bias", 0.38),
        ):
            values = model_cycles[column]
            summary_column = (
                "baseline_mae_full_life" if column == "baseline_mae" else "biased_mae_full_life"
            )
            mean = float(summary_by_model.loc[model, summary_column])
            position = x[model_index] + offset
            draw_model_bar(ax_rank, position, mean, model, alpha, width)
            ax_rank.errorbar(
                position,
                mean,
                yerr=minmax_error(values, mean),
                fmt="o",
                color=MODEL_COLORS[model],
                ecolor=MODEL_COLORS[model],
                markersize=3.8,
                elinewidth=1.0,
                capsize=3.0,
                zorder=4,
            )
        bias = int(summary_by_model.loc[model, "bias_percent"])
        ax_rank.text(
            x[model_index] + width / 2,
            0.002,
            f"{bias:+d}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=MODEL_COLORS[model],
        )
    ax_rank.set_xticks(x, MODELS)
    ax_rank.set_ylabel("Absolute MAE [SOC]")
    ax_rank.set_title("(d) Full-Life Accuracy by Model")
    ax_rank.legend(
        handles=[
            Patch(facecolor=to_rgba("#666666", 0.14), edgecolor="#666666", label="Baseline"),
            Patch(
                facecolor=to_rgba("#666666", 0.38),
                edgecolor="#666666",
                label="Worst ±3% bias",
            ),
        ],
        frameon=False,
        loc="best",
    )
    clean_axes(ax_rank)

    fig.suptitle("Current-bias accumulation, full-charge response, and lifetime accuracy", y=0.995)
    save_figure(fig, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
