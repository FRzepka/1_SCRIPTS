from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[4]
STYLE_DIR = (
    WORKSPACE / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results"
)
sys.path.insert(0, str(STYLE_DIR))
from jes2_plot_style import MODEL_COLORS, clean_axes, save_figure, setup_style


MODELS = ("DM", "HDM", "HECM", "DD")
CONTROLLED_MODELS = ("HDM", "HECM", "DD")
STATES = ("fresh", "mid_life", "aged")
STATE_LABELS = {"fresh": "Fresh", "mid_life": "Mid-life", "aged": "Aged"}


def main() -> None:
    paper = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the current-bias SOH mechanism figure.")
    parser.add_argument("--results-dir", type=Path, default=paper / "JES_2.0/results")
    parser.add_argument(
        "--out", type=Path,
        default=paper / "figures/Results/All Cells/Figure_05c_Current_Bias_SOH_Mechanism.png",
    )
    args = parser.parse_args()

    natural = pd.read_csv(args.results_dir / "jes2_bias_aging_natural_summary.csv")
    controlled = pd.read_csv(args.results_dir / "controlled_soh_bias_adverse.csv")
    charge = pd.read_csv(args.results_dir / "jes2_bias_charge_history.csv")
    charge = charge[charge.complete_three_state_cell.astype(bool)].copy()

    setup_style()
    fig = plt.figure(figsize=(13.4, 8.1))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.08, 1.0), hspace=0.38, wspace=0.27)
    ax_natural = fig.add_subplot(grid[0, :])
    ax_controlled = fig.add_subplot(grid[1, 0])
    ax_charge = fig.add_subplot(grid[1, 1])

    x = np.arange(len(STATES), dtype=float)
    offsets = np.linspace(-0.12, 0.12, len(MODELS))
    for offset, model in zip(offsets, MODELS):
        part = natural[natural.model == model].set_index("window_soh_state").reindex(STATES)
        mean = part.mean_delta_mae.to_numpy(float)
        low = part.min_delta_mae.to_numpy(float)
        high = part.max_delta_mae.to_numpy(float)
        ax_natural.errorbar(
            x + offset, mean, yerr=np.vstack([mean - low, high - mean]),
            color=MODEL_COLORS[model], marker="o", markersize=6.2, linewidth=1.8,
            elinewidth=1.4, capsize=3.5, label=model,
        )
    ax_natural.axhline(0.0, color="#555555", linewidth=0.85)
    ax_natural.set_xticks(x, [STATE_LABELS[state] for state in STATES])
    ax_natural.set_ylabel(r"Adverse-direction $Delta$MAE [SOC]")
    ax_natural.set_title("(a) Observed aging-state sensitivity across five common cells")
    ax_natural.legend(ncol=4, frameon=False, loc="upper left")
    ax_natural.text(
        0.995, 0.04, "Markers: cell mean; error bars: cell min-max",
        transform=ax_natural.transAxes, ha="right", va="bottom", color="#666666", fontsize=9,
    )
    clean_axes(ax_natural)

    for model in CONTROLLED_MODELS:
        part = controlled[controlled.model == model].set_index("soh_level").reindex(STATES)
        ax_controlled.plot(
            x, part.delta_mae, color=MODEL_COLORS[model], marker="o",
            linewidth=1.9, markersize=6.2, label=model,
        )
    ax_controlled.axhline(0.0, color="#555555", linewidth=0.85)
    ax_controlled.set_xticks(
        x,
        [f"{STATE_LABELS[state]}\nSOH {controlled[controlled.soh_level == state].controlled_soh.iloc[0]:.3f}"
         for state in STATES],
    )
    ax_controlled.set_ylabel(r"Adverse-direction $Delta$MAE [SOC]")
    ax_controlled.set_title("(b) Controlled SOH replay\nidentical C29 load and voltage trace")
    ax_controlled.legend(ncol=3, frameon=False, loc="best")
    clean_axes(ax_controlled)

    descriptor = "mean_abs_cumulative_charge_ah"
    for model in MODELS:
        part = charge[charge.model == model].sort_values(descriptor)
        xv = part[descriptor].to_numpy(float)
        yv = part.delta_mae.to_numpy(float)
        correlation = float(np.corrcoef(xv, yv)[0, 1])
        ax_charge.scatter(
            xv, yv, color=MODEL_COLORS[model], s=24, alpha=0.58, edgecolors="none",
        )
        fit_x = np.linspace(xv.min(), xv.max(), 100)
        fit = np.polyfit(xv, yv, 1)
        ax_charge.plot(
            fit_x, np.polyval(fit, fit_x), color=MODEL_COLORS[model], linewidth=1.8,
            label=f"{model}  r={correlation:.2f}",
        )
    ax_charge.set_xlabel(r"Mean $|$cumulative signed charge$|$ [Ah]")
    ax_charge.set_ylabel(r"Adverse-direction $Delta$MAE [SOC]")
    ax_charge.set_title("(c) Association with 24-h signed-charge history\n15 cell-state windows")
    ax_charge.legend(ncol=2, frameon=False, loc="best")
    clean_axes(ax_charge)

    fig.suptitle(
        "Current-gain bias: observed aging pattern versus controlled mechanism tests",
        fontsize=13, y=0.985,
    )
    fig.subplots_adjust(top=0.91, left=0.08, right=0.985, bottom=0.095)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
