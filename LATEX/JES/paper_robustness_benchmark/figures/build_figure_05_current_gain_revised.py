#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "JES_2.0" / "results"
TRAJECTORIES = (
    ROOT / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns"
    / "jes2_representative_trajectories_20260826"
)
OUTPUT = PAPER / "figures/Results/All Cells/Figure_05_Current_Bias_REVISED.png"
MODEL_ORDER = ["DM", "HDM", "HECM", "DD"]
MODEL_COLORS = {"DM": "#2ca02c", "HDM": "#9467bd", "HECM": "#1f77b4", "DD": "#d62728"}
FILES = {
    "DM": ("soc_cc_fullcell_C29.csv", "soc_cc"),
    "HDM": ("soc_cc_soh_fullcell_C29.csv", "soc_cc"),
    "HECM": ("ecm_soc_fullcell_C29.csv", "soc_ecm"),
    "DD": ("soc_pred_fullcell_C29.csv", "soc_pred"),
}


def load_trajectory(condition: str, model: str) -> pd.DataFrame:
    filename, prediction = FILES[model]
    frame = pd.read_csv(TRAJECTORIES / condition / model / filename)
    return frame.rename(columns={prediction: "soc_pred"})


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#d9d9d9",
        "grid.alpha": 0.65,
        "grid.linewidth": 0.7,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    })


def main() -> None:
    setup_style()
    statistics = pd.read_csv(RESULTS / "jes2_signed_current_bias_statistics.csv")
    baseline = {model: load_trajectory("baseline", model) for model in MODEL_ORDER}
    gain = {model: load_trajectory("current_bias_3p0pct", model) for model in MODEL_ORDER}

    figure = plt.figure(figsize=(13.8, 8.4))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.12, 1.0), hspace=0.31, wspace=0.24)
    sensitivity_axis = figure.add_subplot(grid[0, :])
    current_axis = figure.add_subplot(grid[1, 0])
    trajectory_axis = figure.add_subplot(grid[1, 1])

    for model in MODEL_ORDER:
        part = statistics[statistics["model"] == model].sort_values("bias_magnitude_pct")
        sensitivity_axis.errorbar(
            part["bias_magnitude_pct"],
            part["mean"],
            yerr=np.vstack([part["mean"] - part["ci_low"], part["ci_high"] - part["mean"]]),
            color=MODEL_COLORS[model], marker="o", linewidth=2.0, capsize=3, label=model,
        )
    sensitivity_axis.axhline(0.0, color="#444444", linewidth=0.8)
    sensitivity_axis.set_xticks([0.0, 0.5, 1.5, 3.0])
    sensitivity_axis.set_xlabel("Current-gain error magnitude [%]")
    sensitivity_axis.set_ylabel(r"Adverse-pair $\Delta$MAE [SOC]")
    sensitivity_axis.set_title("(a) Six-cell adverse-direction sensitivity")
    sensitivity_axis.legend(ncol=4, frameon=False, loc="upper left")

    base_current = baseline["HECM"]
    gain_current = gain["HECM"]
    current = base_current["I"].to_numpy(float)
    transitions = np.flatnonzero((current[1:] < -0.5) & (current[:-1] > -0.1)) + 1
    center = int(transitions[0]) if len(transitions) else len(base_current) // 2
    start = max(0, center - 14 * 60)
    stop = min(len(base_current), center + 16 * 60)
    x_min = (base_current["time_s"].iloc[start:stop] - base_current["time_s"].iloc[start]) / 60.0
    current_axis.plot(x_min, base_current["I"].iloc[start:stop],
                      color="#444444", linewidth=1.5, label="Baseline")
    current_axis.plot(x_min, gain_current["I"].iloc[start:stop],
                      color=MODEL_COLORS["DD"], linewidth=1.5, linestyle="--", label="+3% gain error")
    current_axis.set_xlabel("Time [min]")
    current_axis.set_ylabel("Measured current [A]")
    current_axis.set_title("(b) Applied gain error")
    current_axis.legend(frameon=False)

    common_origin = max(float(frame["time_s"].iloc[0]) for frame in baseline.values())
    excerpt_start_h = 1.5
    excerpt_end_h = 4.5
    truth_drawn = False
    for model in MODEL_ORDER:
        clean = baseline[model]
        disturbed = gain[model]
        clean_mask = (clean["time_s"] >= common_origin + excerpt_start_h * 3600.0) & (
            clean["time_s"] <= common_origin + excerpt_end_h * 3600.0
        )
        disturbed_mask = (disturbed["time_s"] >= common_origin + excerpt_start_h * 3600.0) & (
            disturbed["time_s"] <= common_origin + excerpt_end_h * 3600.0
        )
        clean_part = clean.loc[clean_mask].iloc[::10]
        disturbed_part = disturbed.loc[disturbed_mask].iloc[::10]
        panel_origin = common_origin + excerpt_start_h * 3600.0
        x_clean = (clean_part["time_s"] - panel_origin) / 3600.0
        x_disturbed = (disturbed_part["time_s"] - panel_origin) / 3600.0
        if not truth_drawn:
            trajectory_axis.plot(x_clean, clean_part["soc_true"], color="#111111", linewidth=1.3,
                                 label="SOC true", zorder=2)
            truth_drawn = True
        trajectory_axis.plot(x_clean, clean_part["soc_pred"], color=MODEL_COLORS[model],
                             linewidth=1.0, linestyle="--", alpha=0.56)
        trajectory_axis.plot(x_disturbed, disturbed_part["soc_pred"], color=MODEL_COLORS[model],
                             linewidth=1.45, label=model)
    trajectory_axis.set_xlabel("Time in excerpt [h]")
    trajectory_axis.set_ylabel("SOC [-]")
    trajectory_axis.set_title("(c) C29 SOC response at +3% gain error")
    trajectory_axis.legend(ncol=2, frameon=False, loc="upper left")
    trajectory_axis.text(
        0.02, 0.04, "Dashed: baseline prediction\nSolid: gain-error prediction",
        transform=trajectory_axis.transAxes, va="bottom", fontsize=8.2,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )

    for axis in (sensitivity_axis, current_axis, trajectory_axis):
        axis.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.95)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
