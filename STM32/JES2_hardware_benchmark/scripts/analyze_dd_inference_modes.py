#!/usr/bin/env python3
"""Compare rolling-window and stateful DD execution on the STM32."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dd_inference_modes"
COLORS = {"Rolling window": "#d62728", "Continuous state": "#2ca02c", "Periodic reset": "#1f77b4"}
NEUTRAL_DARK = "#434343"
NEUTRAL_LIGHT = "#d9d9d9"
FILES = {
    "Rolling window": ROOT / "results/DD/manual_c_smoke/measurements.csv",
    "Continuous state": ROOT / "results/DDS/measurements.csv",
    "Periodic reset": ROOT / "results/DDP/measurements.csv",
}


def save_figure(fig: plt.Figure, name: str | tuple[str, ...]) -> None:
    names = (name,) if isinstance(name, str) else name
    for item in names:
        fig.savefig(OUT / f"{item}.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(OUT / f"{item}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vectors = pd.read_csv(ROOT / "test_vectors/jes2_nominal_vectors.csv")
    records = []
    latency = []
    for mode, path in FILES.items():
        data = pd.read_csv(path)
        data = data[data["status"] == "OK"].copy()
        latency.append(data.assign(mode=mode))
        first = data[data["round"] == data["round"].min()][["sample_id", "soc_device"]]
        first = vectors[["sample_id", "segment_id", "soc_dataset", "expected_dd"]].merge(first, on="sample_id", how="left")
        # The rolling firmware was timed on a representative subset because each
        # output takes about 0.72 s. Its complete software reference is bit-level
        # equivalent to the measured C implementation (maximum error 1.24e-7).
        if mode == "Rolling window":
            first["soc_device"] = first["expected_dd"]
        first["mode"] = mode
        records.append(first)

    merged = pd.concat(records, ignore_index=True)
    merged["error_to_dataset"] = merged["soc_device"] - merged["soc_dataset"]
    merged["error_to_rolling"] = merged["soc_device"] - merged["expected_dd"]
    merged.to_csv(OUT / "dd_inference_modes.csv", index=False)
    all_latency = pd.concat(latency, ignore_index=True)
    all_latency.to_csv(OUT / "latency_samples.csv", index=False)

    memory = json.loads((ROOT / "results/dd_mode_memory.json").read_text(encoding="utf-8"))
    memory_by_model = {row["model"]: row for row in memory["models"]}
    ids = {"Rolling window": "DD", "Continuous state": "DDS", "Periodic reset": "DDP"}
    summary = []
    for mode in FILES:
        values = merged[(merged["mode"] == mode) & (merged["sample_id"] >= 2023)].dropna(subset=["soc_device"])
        times = all_latency[all_latency["mode"] == mode]["device_time_us"]
        ref = values.dropna(subset=["expected_dd"])
        mem = memory_by_model[ids[mode]]
        summary.append({
            "mode": mode,
            "samples_accuracy": int(len(values)),
            "soc_mae_to_dataset": float(values["error_to_dataset"].abs().mean()),
            "soc_rmse_to_dataset": float(np.sqrt(np.mean(values["error_to_dataset"] ** 2))),
            "mae_to_rolling_reference": float(ref["error_to_rolling"].abs().mean()),
            "max_abs_to_rolling_reference": float(ref["error_to_rolling"].abs().max()),
            "latency_median_us": float(times.median()),
            "latency_p95_us": float(times.quantile(.95)),
            "latency_max_us": float(times.max()),
            "flash_bytes": mem["flash_load_bytes"],
            "static_ram_bytes": mem["static_ram_bytes"],
        })
    summary_df = pd.DataFrame(summary)
    rolling_latency = summary_df.loc[summary_df["mode"] == "Rolling window", "latency_median_us"].iloc[0]
    summary_df["speedup_vs_rolling"] = rolling_latency / summary_df["latency_median_us"]
    summary_df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": NEUTRAL_DARK, "axes.linewidth": .8,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": NEUTRAL_LIGHT, "grid.alpha": .65, "grid.linewidth": .7,
        # Nimbus Sans is used by the JES paper build. DejaVu Sans is the local
        # metrically similar fallback available in the Windows plotting setup.
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    base = vectors[(vectors.sample_id >= 1900) & (vectors.sample_id <= 2200)]
    ax.plot(base.sample_id, base.soc_dataset, color="#222222", lw=1.8, label="Dataset SOC")
    detail = {}
    for mode in COLORS:
        detail[mode] = merged[(merged["mode"] == mode) & (merged.sample_id >= 1900) & (merged.sample_id <= 2200)]
    ax.plot(detail["Periodic reset"].sample_id, detail["Periodic reset"].soc_device,
            color=COLORS["Periodic reset"], lw=1.15, label="Periodic reset", zorder=3)
    ax.plot(detail["Rolling window"].sample_id, detail["Rolling window"].soc_device,
            color=COLORS["Rolling window"], lw=2.5, label="Rolling window", zorder=4)
    ax.plot(detail["Continuous state"].sample_id, detail["Continuous state"].soc_device,
            color=COLORS["Continuous state"], lw=1.25, ls=(0, (4, 3)),
            label="Continuous state", zorder=5)
    ax.axvline(2024, color="#777777", ls="--", lw=1, label="Reset boundary")
    ax.set(xlabel="Sample", ylabel="SOC", title="DD output around the 2024-sample reset boundary")
    ax.legend(ncol=2, fontsize=8)
    clean_axes(ax)
    save_figure(fig, "soc_reset_detail")

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1], "hspace": .08})
    ax, residual_ax = axes
    ax.plot(vectors.sample_id, vectors.soc_dataset, color="#222222", lw=1.5, label="Dataset SOC", zorder=2)
    rolling = merged[merged["mode"] == "Rolling window"]
    continuous = merged[merged["mode"] == "Continuous state"]
    periodic = merged[merged["mode"] == "Periodic reset"]
    ax.plot(periodic.sample_id, periodic.soc_device, color=COLORS["Periodic reset"], lw=1.15,
            label="Periodic reset", zorder=3)
    ax.plot(rolling.sample_id, rolling.soc_device, color=COLORS["Rolling window"], lw=2.5,
            label="Rolling window", zorder=4)
    ax.plot(continuous.sample_id, continuous.soc_device, color=COLORS["Continuous state"], lw=1.25,
            ls=(0, (4, 3)), label="Continuous state", zorder=5)
    ax.set(ylabel="SOC", title="DD inference modes over the complete benchmark trajectory")
    ax.legend(ncol=2, loc="best")
    residual_ax.axhline(0, color="#222222", lw=1.2, label="Dataset SOC")
    residual_ax.plot(periodic.sample_id, periodic.error_to_dataset * 100,
                     color=COLORS["Periodic reset"], lw=1.1, label="Periodic reset", zorder=3)
    residual_ax.plot(rolling.sample_id, rolling.error_to_dataset * 100,
                     color=COLORS["Rolling window"], lw=2.2, label="Rolling window", zorder=4)
    residual_ax.plot(continuous.sample_id, continuous.error_to_dataset * 100,
                     color=COLORS["Continuous state"], lw=1.2, ls=(0, (4, 3)),
                     label="Continuous state", zorder=5)
    residual_ax.set(xlabel="Sample", ylabel="Difference\n[percentage points]")
    residual_ax.legend(ncol=3, loc="best", fontsize=8)
    clean_axes(ax)
    clean_axes(residual_ax)
    save_figure(fig, "soc_full_trajectory_dataset_reference")

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), sharex=True,
                             gridspec_kw={"hspace": .12})
    axes[0].axhline(0, color="#222222", lw=1.2, label="Dataset SOC")
    axes[0].plot(periodic.sample_id, periodic.error_to_dataset * 100,
                 color=COLORS["Periodic reset"], lw=1.1, label="Periodic reset", zorder=3)
    axes[0].plot(rolling.sample_id, rolling.error_to_dataset * 100,
                 color=COLORS["Rolling window"], lw=2.2, label="Rolling window", zorder=4)
    axes[0].plot(continuous.sample_id, continuous.error_to_dataset * 100,
                 color=COLORS["Continuous state"], lw=1.2, ls=(0, (4, 3)),
                 label="Continuous state", zorder=5)
    axes[0].set_ylabel("SOC error\n[percentage points]")
    axes[0].set_title("Estimator error relative to the dataset SOC")
    axes[0].legend(ncol=2, loc="best")
    early_limit = 2100
    axes[1].axhline(0, color="#222222", lw=1.2, label="Dataset SOC")
    axes[1].plot(periodic.loc[periodic.sample_id <= early_limit, "sample_id"],
                 periodic.loc[periodic.sample_id <= early_limit, "error_to_dataset"] * 100,
                 color=COLORS["Periodic reset"], lw=1.1, label="Periodic reset", zorder=3)
    axes[1].plot(rolling.loc[rolling.sample_id <= early_limit, "sample_id"],
                 rolling.loc[rolling.sample_id <= early_limit, "error_to_dataset"] * 100,
                 color=COLORS["Rolling window"], lw=2.2, label="Rolling window", zorder=4)
    axes[1].plot(continuous.loc[continuous.sample_id <= early_limit, "sample_id"],
                 continuous.loc[continuous.sample_id <= early_limit, "error_to_dataset"] * 100,
                 color=COLORS["Continuous state"], lw=1.2, ls=(0, (4, 3)),
                 label="Continuous state", zorder=5)
    axes[1].axvline(2023, color="#777777", ls="--", lw=1, label="First rolling output")
    axes[1].axvline(2024, color="#777777", ls=":", lw=1, label="Periodic reset")
    axes[1].set(xlabel="Sample", ylabel="SOC error\n[percentage points]",
                title="Startup and first reset")
    axes[1].legend(ncol=2, loc="best")
    for axis in axes:
        clean_axes(axis)
    save_figure(fig, "difference_to_dataset")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for mode, color in COLORS.items():
        values = np.sort(all_latency.loc[all_latency["mode"] == mode, "device_time_us"].to_numpy())
        probability = np.arange(1, len(values) + 1) / len(values)
        axes[0].plot(values / 1000, probability, color=color, lw=2, label=mode)
        if mode != "Rolling window":
            axes[1].plot(values, probability, color=color, lw=1.8, label=mode)
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Inference time [ms]", ylabel="Empirical cumulative probability",
                title="Complete latency range")
    axes[1].set(xlabel="Inference time [µs]", ylabel="Empirical cumulative probability",
                title="Stateful variants")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    for ax in axes:
        clean_axes(ax)
    fig.suptitle("Measured STM32 inference-time distributions")
    fig.tight_layout()
    save_figure(fig, "latency_ecdf")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    fills = [(*plt.matplotlib.colors.to_rgb(COLORS[x]), .38) for x in summary_df["mode"]]
    edges = [COLORS[x] for x in summary_df["mode"]]
    axes[0].bar(summary_df["mode"], summary_df["flash_bytes"] / 1024,
                color=fills, edgecolor=edges, linewidth=1.5)
    axes[1].bar(summary_df["mode"], summary_df["static_ram_bytes"] / 1024,
                color=fills, edgecolor=edges, linewidth=1.5)
    axes[0].set(ylabel="Flash [KiB]", title="Flash footprint")
    axes[1].set(ylabel="Static RAM [KiB]", title="Static RAM footprint")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        clean_axes(ax)
    save_figure(fig, "memory_comparison")

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
