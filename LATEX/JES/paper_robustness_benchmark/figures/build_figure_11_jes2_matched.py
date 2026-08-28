#!/usr/bin/env python3
"""Recreate the calm legacy Figure 11 layout with current JES2 C29 data."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RESULTS_CODE = ROOT / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results"
sys.path.insert(0, str(RESULTS_CODE))

from build_jes2_trajectory_figures import load_run  # noqa: E402


TRAJECTORIES = (
    ROOT / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns"
    / "jes2_spike_C29_fresh_trajectories_20260828"
)
OUTPUT = Path(__file__).resolve().parent / "Results/Figure_11c_Voltage_Spike_Response_JES2.png"
MODEL_ORDER = ["DM", "HDM", "HECM", "DD"]
# Match the established dissertation/JES robustness palette used throughout
# the All Cells collection: green DM, purple HDM, blue HECM, and red DD.
COLORS = {"DM": "#2ca02c", "HDM": "#9467bd", "HECM": "#1f77b4", "DD": "#d62728"}
REPRESENTATIVE_HECM_INDEX = 22000
RELATIVE_SECONDS = np.arange(-60, 181)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.34,
        "grid.color": "#c8c8c8",
    })


def aligned_excess_errors(baseline, disturbed, event_times: np.ndarray) -> np.ndarray:
    rows = []
    time = disturbed.time_s.to_numpy()
    for event_time in event_times:
        index = int(np.searchsorted(time, event_time))
        if index < 60 or index + 180 >= len(disturbed) or abs(time[index] - event_time) > 1.0:
            continue
        selection = slice(index - 60, index + 181)
        excess = (
            disturbed.abs_err.iloc[selection].to_numpy()
            - baseline.abs_err.iloc[selection].to_numpy()
        )
        excess -= float(np.mean(excess[50:60]))
        rows.append(excess)
    return np.vstack(rows)


def main() -> None:
    setup_style()
    baseline = {model: load_run(TRAJECTORIES, "baseline", model)[0] for model in MODEL_ORDER}
    disturbed = {model: load_run(TRAJECTORIES, "voltage_spikes", model)[0] for model in MODEL_ORDER}

    hecm_base = baseline["HECM"]
    hecm_spike = disturbed["HECM"]
    voltage_delta = hecm_spike.U.to_numpy() - hecm_base.U.to_numpy()
    event_indices = np.flatnonzero(np.abs(voltage_delta) > 0.1)
    event_times = hecm_spike.time_s.iloc[event_indices].to_numpy()
    center = float(hecm_spike.time_s.iloc[REPRESENTATIVE_HECM_INDEX])

    fig = plt.figure(figsize=(13.6, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.26)
    ax_soc = fig.add_subplot(grid[0, :])
    ax_bar = fig.add_subplot(grid[1, 0])
    ax_local = fig.add_subplot(grid[1, 1])

    reference = disturbed["DM"]
    mask = (reference.time_s >= center - 60) & (reference.time_s <= center + 180)
    ax_soc.plot(reference.loc[mask, "time_s"] - center, reference.loc[mask, "soc_true"],
                color="black", linewidth=2.0, linestyle="--", label="Ground truth")

    peak_rows = []
    for model in MODEL_ORDER:
        run = disturbed[model]
        base = baseline[model]
        mask = (run.time_s >= center - 60) & (run.time_s <= center + 180)
        part = run.loc[mask]
        indices = part.index
        ax_soc.plot(part.time_s - center, base.loc[indices, "soc_pred"],
                    color=COLORS[model], linewidth=1.7, linestyle="--", alpha=0.72)
        ax_soc.plot(part.time_s - center, part.soc_pred, color=COLORS[model],
                    linewidth=2.1, label=model)

        aligned = aligned_excess_errors(base, run, event_times)
        median = np.nanmedian(aligned, axis=0)
        low, high = np.nanpercentile(aligned, [25, 75], axis=0)
        ax_local.plot(RELATIVE_SECONDS, median, color=COLORS[model], linewidth=2.1, label=model)
        ax_local.fill_between(RELATIVE_SECONDS, low, high, color=COLORS[model], alpha=0.20)
        peaks = np.nanmax(np.maximum(aligned[:, 60:], 0.0), axis=1)
        peak_rows.append(float(np.nanpercentile(peaks, 95)))

    positions = np.arange(len(MODEL_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        ax_bar.bar(index, peak_rows[index], width=0.72, color=COLORS[model], alpha=0.38,
                   edgecolor=COLORS[model], linewidth=2.0)
    ax_bar.set_xticks(positions, MODEL_ORDER)
    ax_bar.set_ylabel("p95 peak excess error")
    ax_bar.set_title("(b)", loc="left", fontsize=14, fontweight="bold", pad=6)

    ax_soc.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_soc.set_xlabel("Seconds relative to a representative voltage spike")
    ax_soc.set_ylabel("SOC")
    ax_soc.set_title("(a)", loc="left", fontsize=14, fontweight="bold", pad=6)
    ax_soc.legend(frameon=True, ncol=3, loc="upper right")

    ax_local.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_local.set_xlabel("Seconds relative to voltage spike")
    ax_local.set_ylabel("Excess absolute error")
    ax_local.set_title("(c)", loc="left", fontsize=14, fontweight="bold", pad=6)
    ax_local.legend(frameon=True, ncol=2, loc="upper right")
    fig.subplots_adjust(top=0.90)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
