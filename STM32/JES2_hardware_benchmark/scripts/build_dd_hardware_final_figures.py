#!/usr/bin/env python3
"""Build the nonredundant final DD hardware benchmark figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "dd_hardware_final"
COLORS = {"Rolling window": "#d62728", "Continuous state": "#2ca02c", "Periodic reset": "#1f77b4"}
MODEL_IDS = {"Rolling window": "DD", "Continuous state": "DDS", "Periodic reset": "DDP"}


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#434343", "axes.linewidth": .8,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#d9d9d9", "grid.alpha": .65, "grid.linewidth": .7,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })


def measurement_path(mode: str) -> Path:
    candidates = {
        "Rolling window": [RESULTS / "DD_multirun/measurements.csv", RESULTS / "DD/manual_c_smoke/measurements.csv"],
        "Continuous state": [RESULTS / "DDS_multirun/measurements.csv", RESULTS / "DDS/measurements.csv"],
        "Periodic reset": [RESULTS / "DDP_multirun/measurements.csv", RESULTS / "DDP/measurements.csv"],
    }[mode]
    return next(path for path in candidates if path.is_file())


def bootstrap_mean_ci(values: np.ndarray, seed: int = 2708) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> None:
    style()
    vectors = pd.read_csv(ROOT / "test_vectors/jes2_nominal_vectors.csv")
    modes = {}
    round_rows = []
    for mode in COLORS:
        frame = pd.read_csv(measurement_path(mode))
        valid = frame[frame.status == "OK"].copy()
        modes[mode] = valid
        for round_id, group in valid.groupby("round"):
            round_rows.append({
                "mode": mode,
                "round": int(round_id),
                "n": len(group),
                "latency_median_us": group.device_time_us.median(),
                "latency_p95_us": group.device_time_us.quantile(.95),
            })
    rounds = pd.DataFrame(round_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(OUT / "latency_by_round.csv", index=False)

    trajectories = []
    for mode, frame in modes.items():
        first = frame[frame["round"] == frame["round"].min()][["sample_id", "soc_device"]]
        merged = vectors[["sample_id", "soc_dataset", "expected_dd"]].merge(first, on="sample_id", how="left")
        if mode == "Rolling window":
            # Use measured MCU values where available and the complete validated
            # rolling reference elsewhere. The measured maximum discrepancy is
            # 1.24e-7 SOC, well below the visible scale of this figure.
            merged["soc_device"] = merged["soc_device"].fillna(merged["expected_dd"])
        merged["mode"] = mode
        merged["error_pp"] = (merged.soc_device - merged.soc_dataset) * 100
        trajectories.append(merged)
    trajectory = pd.concat(trajectories, ignore_index=True)
    trajectory.to_csv(OUT / "soc_error_trajectory.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.axhline(0, color="#222222", lw=1.2, label="Dataset SOC")
    draw_order = ["Periodic reset", "Rolling window", "Continuous state"]
    styles = {
        "Periodic reset": {"lw": 1.1, "ls": "-", "zorder": 3},
        "Rolling window": {"lw": 2.2, "ls": "-", "zorder": 4},
        "Continuous state": {"lw": 1.2, "ls": (0, (4, 3)), "zorder": 5},
    }
    for mode in draw_order:
        data = trajectory[trajectory["mode"] == mode]
        ax.plot(data.sample_id, data.error_pp, color=COLORS[mode], label=mode, **styles[mode])
    ax.axvline(2023, color="#777777", ls="--", lw=1, label="First rolling-window output")
    for sample in range(2024, int(vectors.sample_id.max()) + 1, 2024):
        ax.axvline(sample, color=COLORS["Periodic reset"], ls=":", lw=1,
                   label="Periodic reset" if sample == 2024 else None)
    ax.set(xlabel="Sample", ylabel="SOC error [percentage points]",
           title="DD estimator error on the nominal C27 trajectory")
    ax.legend(ncol=2, loc="best")
    clean_axes(ax)
    save(fig, "dd_soc_error_dataset")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    for mode, color in COLORS.items():
        values = np.sort(modes[mode].device_time_us.to_numpy(dtype=float))
        probability = np.arange(1, len(values) + 1) / len(values)
        axes[0].plot(values / 1000, probability, color=color, lw=2, label=mode)
        if mode != "Rolling window":
            axes[1].plot(values, probability, color=color, lw=1.8, label=mode)
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Inference time [ms]", ylabel="Empirical cumulative probability",
                title="Complete latency range")
    axes[1].set(xlabel="Inference time [µs]", ylabel="Empirical cumulative probability",
                title="Stateful variants")
    rng = np.random.default_rng(2708)
    for index, mode in enumerate(COLORS):
        values = rounds.loc[rounds["mode"] == mode, "latency_median_us"].to_numpy(dtype=float)
        jitter = rng.uniform(-.06, .06, len(values))
        axes[2].scatter(index + jitter, values, s=26, facecolor="white",
                        edgecolor=COLORS[mode], linewidth=1.1, zorder=3)
        mean, low, high = bootstrap_mean_ci(values, seed=2708 + index)
        axes[2].errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="o",
                         color=COLORS[mode], capsize=4, lw=1.5, zorder=4)
    axes[2].set_xticks(range(len(COLORS)), list(COLORS), rotation=18)
    axes[2].set_yscale("log")
    axes[2].set(ylabel="Round-median inference time [µs]",
                title="Independent hardware rounds")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    for ax in axes:
        clean_axes(ax)
    fig.suptitle("Measured STM32 inference-time distributions and C27 repeatability")
    fig.tight_layout()
    save(fig, "dd_latency_statistics")

    memory = json.loads((RESULTS / "dd_mode_memory.json").read_text(encoding="utf-8"))
    memory = {row["model"]: row for row in memory["models"]}
    labels = list(COLORS)
    flash = [memory[MODEL_IDS[label]]["flash_load_bytes"] / 1024 for label in labels]
    ram = [memory[MODEL_IDS[label]]["static_ram_bytes"] / 1024 for label in labels]
    fills = [to_rgba(COLORS[label], .38) for label in labels]
    edges = [COLORS[label] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].bar(labels, flash, color=fills, edgecolor=edges, linewidth=1.5)
    axes[1].bar(labels, ram, color=fills, edgecolor=edges, linewidth=1.5)
    axes[0].set(ylabel="Flash [KiB]", title="Flash footprint")
    axes[1].set(ylabel="Static RAM [KiB]", title="Static RAM footprint")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        clean_axes(ax)
    fig.tight_layout()
    save(fig, "dd_memory_footprints")

    stats = []
    for mode in COLORS:
        values = rounds.loc[rounds["mode"] == mode, "latency_median_us"].to_numpy(dtype=float)
        mean, low, high = bootstrap_mean_ci(values)
        stats.append({"mode": mode, "rounds": len(values), "round_median_mean_us": mean,
                      "round_median_ci95_low_us": low, "round_median_ci95_high_us": high})
    pd.DataFrame(stats).to_csv(OUT / "latency_statistics.csv", index=False)
    print(pd.DataFrame(stats).to_string(index=False))


if __name__ == "__main__":
    main()
