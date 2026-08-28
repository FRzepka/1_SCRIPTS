#!/usr/bin/env python3
"""Build the four consolidated JES2 STM32 hardware benchmark figures."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


CELLS = ("C09", "C13", "C15", "C25", "C27", "C29")
MODELS = ("DM", "HDM", "HECM", "DD")
VARIANTS = ("DD", "DDS", "DDP")
MODEL_LABELS = {"DM": "DM", "HDM": "HDM", "HECM": "HECM", "DD": "DD"}
VARIANT_LABELS = {"DD": "Rolling window", "DDS": "Continuous state", "DDP": "Periodic reset"}
COLORS = {"DM": "#2ca02c", "HDM": "#9467bd", "HECM": "#1f77b4", "DD": "#d62728"}
VARIANT_COLORS = {"DD": "#d62728", "DDS": "#2ca02c", "DDP": "#1f77b4"}
LOAD_CLASSES = {"Low": ("C25", "C27"), "Medium": ("C09", "C13", "C15"), "High": ("C29",)}


def read_csv(path: Path) -> list[dict[str, str]]:
    # Synology occasionally inserts sparse NUL runs between complete CSV lines.
    # Removing only those bytes preserves both adjacent protocol records.
    text = path.read_bytes().replace(b"\x00", b"").decode("utf-8")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    delta = prediction - target
    return float(np.mean(np.abs(delta))), float(np.sqrt(np.mean(delta**2))), float(np.max(np.abs(delta)))


def measurements(root: Path, cell: str, model: str, first_round: bool = False) -> list[dict[str, str]]:
    path = root / "results" / cell / model / "measurements.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing hardware measurements: {path}")
    rows = [row for row in read_csv(path) if row.get("status") == "OK" and row.get("round", "").isdigit()]
    if first_round:
        rows = [row for row in rows if int(row["round"]) == 1]
    return rows


def load_memory(root: Path) -> dict[str, dict]:
    data: dict[str, dict] = {}
    for name in ("memory.json", "dd_mode_memory.json"):
        payload = json.loads((root / "results" / name).read_text(encoding="utf-8"))
        data.update({row["model"]: row for row in payload["models"]})
    return data


def load_summary(root: Path, cell: str, model: str) -> dict:
    path = root / "results" / cell / model / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing hardware summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_vectors(root: Path) -> dict[str, list[dict[str, str]]]:
    directory = root / "test_vectors" / "multicell"
    manifest = json.loads((directory / "jes2_multicell_manifest.json").read_text(encoding="utf-8"))
    return {item["cell"]: read_csv(directory / item["csv"]) for item in manifest["vectors"]}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 8, "axes.grid": True,
        "grid.alpha": 0.24, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "figure.dpi": 120, "savefig.dpi": 300,
    })


def collect_accuracy(root: Path, vectors: dict[str, list[dict[str, str]]]) -> list[dict]:
    rows: list[dict] = []
    for cell in CELLS:
        for model in MODELS:
            summary = load_summary(root, cell, model)
            dataset_metric = summary["dataset_difference"]
            agreement = summary["reference_difference"]
            expected_column = f"expected_{model.lower()}"
            valid = [row for row in vectors[cell] if row.get(expected_column, "")]
            software = np.array([float(row[expected_column]) for row in valid])
            dataset = np.array([float(row["soc_dataset"]) for row in valid])
            s_mae, s_rmse, s_max = metric(software, dataset)
            rows.append({
                "cell": cell, "model": model, "n_samples": len(valid),
                "hardware_dataset_mae": dataset_metric["mae"],
                "hardware_dataset_rmse": dataset_metric["rmse"],
                "hardware_dataset_max_abs_error": dataset_metric["maximum_absolute_error"],
                "software_dataset_mae": s_mae,
                "software_dataset_rmse": s_rmse, "software_dataset_max_abs_error": s_max,
                "hardware_software_mae": agreement["mae"],
                "hardware_software_rmse": agreement["rmse"],
                "hardware_software_max_abs_error": agreement["maximum_absolute_error"],
            })
    return rows


def collect_latency(root: Path) -> tuple[list[dict], list[dict]]:
    cell_rows: list[dict] = []
    round_rows: list[dict] = []
    for cell in CELLS:
        for model in MODELS:
            summary = load_summary(root, cell, model)
            timing = summary["device_time_us"]
            cell_rows.append({
                "cell": cell, "model": model, "n_inferences": timing["n"],
                "median_us": timing["median"], "p95_us": timing["p95"],
                "maximum_us": timing["maximum"],
            })
            if model == "DD":
                continue
            samples = measurements(root, cell, model)
            grouped: dict[int, list[float]] = defaultdict(list)
            for sample in samples:
                grouped[int(sample["round"])].append(float(sample["device_time_us"]))
            for round_id, timings in grouped.items():
                round_rows.append({"cell": cell, "model": model, "round": round_id,
                                   "round_median_us": float(np.median(timings))})
    # The multicell rolling-window campaign has one replay per cell. Use the
    # dedicated ten-round C27 run to quantify DD replay repeatability.
    round_rows = [row for row in round_rows if row["model"] != "DD"]
    for row in read_csv(root / "results" / "dd_hardware_final" / "latency_by_round.csv"):
        if row["mode"] == "Rolling window":
            round_rows.append({"cell": "C27", "model": "DD", "round": int(row["round"]),
                               "round_median_us": float(row["latency_median_us"])})
    return cell_rows, round_rows


def collect_variants(root: Path, memory: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for cell in CELLS:
        for variant in VARIANTS:
            summary = load_summary(root, cell, variant)
            if variant == "DD":
                dataset_metric = summary["dataset_difference"]
                reference_metric = summary["reference_difference"]
                mae, rmse, maximum = (dataset_metric["mae"], dataset_metric["rmse"],
                                      dataset_metric["maximum_absolute_error"])
                ref_mae, ref_max = reference_metric["mae"], reference_metric["maximum_absolute_error"]
                n_samples = dataset_metric["n"]
            else:
                samples = measurements(root, cell, variant, first_round=True)
                samples = [row for row in samples if int(row["sample_id"]) >= 2023]
                hardware = np.array([float(row["soc_device"]) for row in samples])
                dataset = np.array([float(row["soc_dataset"]) for row in samples])
                reference = np.array([float(row["soc_reference"]) for row in samples])
                mae, rmse, maximum = metric(hardware, dataset)
                ref_mae, _, ref_max = metric(hardware, reference)
                n_samples = len(samples)
            timing = summary["device_time_us"]
            resource = memory[variant]
            rows.append({
                "cell": cell, "variant": VARIANT_LABELS[variant],
                "native_first_output_sample": 2023 if variant == "DD" else 0,
                "common_horizon_start": 2023, "n_common_samples": n_samples,
                "dataset_mae": mae, "dataset_rmse": rmse, "dataset_max_abs_error": maximum,
                "rolling_reference_mae": ref_mae, "rolling_reference_max_abs_error": ref_max,
                "latency_median_us": timing["median"], "latency_p95_us": timing["p95"],
                "flash_kib": resource["flash_load_bytes"] / 1024,
                "static_ram_kib": resource["static_ram_bytes"] / 1024,
            })
    return rows


def aggregate_accuracy(rows: list[dict]) -> list[dict]:
    result = []
    for load_class, cells in LOAD_CLASSES.items():
        for model in MODELS:
            values = np.array([row["hardware_dataset_mae"] for row in rows
                               if row["cell"] in cells and row["model"] == model])
            result.append({"load_class": load_class, "model": model, "n_cells": len(values),
                           "mean_mae": values.mean(), "minimum_mae": values.min(),
                           "maximum_mae": values.max()})
    return result


def aggregate_equivalence(rows: list[dict]) -> list[dict]:
    result = []
    for model in MODELS:
        values = np.array([row["hardware_software_mae"] for row in rows if row["model"] == model])
        result.append({
            "model": model, "n_cells": len(values), "mean_mae": values.mean(),
            "minimum_mae": values.min(), "maximum_mae": values.max(),
        })
    return result


def latency_distributions(
    root: Path, timing_column: str, scale: float = 1.0
) -> tuple[dict[str, np.ndarray], list[dict]]:
    distributions: dict[str, np.ndarray] = {}
    stats: list[dict] = []
    for model in MODELS:
        source_model = "DDS" if model == "DD" else model
        rows = []
        for cell in CELLS:
            rows.extend(measurements(root, cell, source_model))
        source = ("DD continuous-state execution over six cells with three replay rounds each"
                  if model == "DD" else "six cells with three replay rounds each")
        values = scale * np.array([float(row[timing_column]) for row in rows])
        distributions[model] = values
        stats.append({"model": model, "source": source, "n_inferences": len(values),
                      "mean_ms": values.mean(), "median_ms": np.median(values),
                      "p05_ms": np.percentile(values, 5), "p95_ms": np.percentile(values, 95),
                      "minimum_ms": values.min(), "maximum_ms": values.max()})
    return distributions, stats


def plot_accuracy(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(MODELS))
    for index, model in enumerate(MODELS):
        values = np.array([100 * row["hardware_software_mae"] for row in rows if row["model"] == model])
        mean = values.mean()
        bar = ax.bar(index, mean, width=0.62, facecolor=to_rgba(COLORS[model], 0.28),
                     edgecolor=COLORS[model], linewidth=1.4, zorder=2)
        ax.errorbar(index, mean, yerr=[[mean - values.min()], [values.max() - mean]], fmt="none",
                    ecolor=COLORS[model], elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
        jitter = np.linspace(-0.07, 0.07, len(values))
        ax.scatter(index + jitter, values, s=22, color=COLORS[model], edgecolor="white",
                   linewidth=0.4, zorder=5)
        ax.bar_label(bar, labels=[f"{mean:.2e}"], padding=4, fontsize=8)
    ax.set_xticks(x, MODELS)
    ax.set_yscale("log")
    ax.set_ylabel("Hardware-software MAE [percentage points]")
    ax.set_title("Numerical Equivalence of Hardware and Software Execution")
    fig.tight_layout()
    fig.savefig(out / "figure_01_soc_accuracy_by_load_class.png", bbox_inches="tight")
    plt.close(fig)


def plot_latency(
    distributions: dict[str, np.ndarray], out: Path, *, filename: str,
    title: str, xlabel: str, logarithmic: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    pooled = np.concatenate([distributions[model] for model in MODELS])
    lo, hi = np.percentile(pooled, [0.1, 99.9])
    bins = (np.logspace(np.log10(lo), np.log10(hi), 52)
            if logarithmic else np.linspace(lo, hi, 32))
    for model in MODELS:
        values = distributions[model]
        ax.hist(values, bins=bins, facecolor=to_rgba(COLORS[model], 0.16),
                edgecolor=COLORS[model], linewidth=1.15, label=model)
        ax.axvline(np.median(values), color=COLORS[model], linewidth=1.0, linestyle="--")
    if logarithmic:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel("Count")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)


def plot_latency_detail(distributions: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    for ax, model in zip(axes.flat, MODELS):
        values_us = 1000.0 * distributions[model]
        lo, hi = np.percentile(values_us, [0.05, 99.95])
        margin = max(0.04 * (hi - lo), 0.002)
        bins = np.linspace(lo - margin, hi + margin, 42)
        ax.hist(values_us, bins=bins, facecolor=to_rgba(COLORS[model], 0.24),
                edgecolor=COLORS[model], linewidth=1.0)
        median = np.median(values_us)
        p05, p95 = np.percentile(values_us, [5, 95])
        ax.axvline(median, color=COLORS[model], linewidth=1.2, linestyle="--",
                   label=f"Median: {median:.4g} us")
        ax.axvspan(p05, p95, color=COLORS[model], alpha=0.08,
                   label=f"P5-P95: {p05:.4g}-{p95:.4g} us")
        ax.set_title(f"{model} On-Device Latency")
        ax.set_xlabel("Inference latency [us]")
        ax.set_ylabel("Count")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figure_02_latency_distributions_on_device_detailed.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_latency_combined_detail(distributions: dict[str, np.ndarray], out: Path) -> None:
    fig = plt.figure(figsize=(10.2, 10.4))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.82), hspace=0.38, wspace=0.28)
    detail_axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    for panel, (ax, model) in enumerate(zip(detail_axes, MODELS)):
        values_us = 1000.0 * distributions[model]
        lo, hi = np.percentile(values_us, [0.05, 99.95])
        margin = max(0.04 * (hi - lo), 0.002)
        bins = np.linspace(lo - margin, hi + margin, 42)
        color = COLORS[model]
        ax.hist(values_us, bins=bins, facecolor=to_rgba(color, 0.24),
                edgecolor=color, linewidth=1.0)
        median = np.median(values_us)
        p05, p95 = np.percentile(values_us, [5, 95])
        ax.axvline(median, color=color, linewidth=1.2, linestyle="--",
                   label=f"Median: {median:.4g} us")
        ax.axvspan(p05, p95, color=color, alpha=0.08,
                   label=f"P5-P95: {p05:.4g}-{p95:.4g} us")
        ax.set_title(f"({chr(97 + panel)}) {model}")
        ax.set_xlabel("Inference latency [us]")
        ax.set_ylabel("Inference count")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.legend(frameon=False, fontsize=6.8)

    overview = fig.add_subplot(grid[2, :])
    y_positions = np.arange(len(MODELS))[::-1]
    for y, model in zip(y_positions, MODELS):
        values = distributions[model]
        minimum, p05, median, p95, maximum = np.percentile(values, [0, 5, 50, 95, 100])
        color = COLORS[model]
        overview.hlines(y, minimum, maximum, color=to_rgba(color, 0.48), linewidth=1.2,
                        zorder=1)
        overview.barh(y, p95 - p05, left=p05, height=0.46,
                      facecolor=to_rgba(color, 0.22), edgecolor=color,
                      linewidth=1.2, zorder=2)
        overview.vlines(median, y - 0.29, y + 0.29, color=color, linewidth=3.0,
                        zorder=3)
        overview.text(maximum * 1.08, y, f"{median * 1000:.4g} us", va="center",
                      fontsize=8, color=color)
    overview.set_xscale("log")
    overview.set_yticks(y_positions, MODELS)
    overview.set_xlabel("On-device inference latency [ms]")
    overview.set_title("(e) Absolute latency comparison")
    overview.grid(axis="y", visible=False)
    overview.legend(handles=[
        Line2D([0], [0], color="#777777", linewidth=1.2, label="Min-Max"),
        Patch(facecolor=to_rgba("#777777", 0.22), edgecolor="#777777", label="P5-P95"),
        Line2D([0], [0], color="#777777", linewidth=3.0, label="Median"),
    ], frameon=False, ncol=3, loc="upper center")
    fig.suptitle("On-Device Inference-Latency Distributions Across Six Cells", y=0.985)
    fig.savefig(out / "figure_02_latency_distributions_on_device.png", bbox_inches="tight")
    plt.close(fig)


def plot_latency_range_overview(distributions: dict[str, np.ndarray], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    y_positions = np.arange(len(MODELS))[::-1]
    for y, model in zip(y_positions, MODELS):
        values = distributions[model]
        minimum, p05, median, p95, maximum = np.percentile(values, [0, 5, 50, 95, 100])
        color = COLORS[model]
        ax.hlines(y, minimum, maximum, color=to_rgba(color, 0.42), linewidth=1.2)
        ax.hlines(y, p05, p95, color=color, linewidth=6.0)
        ax.scatter(median, y, s=64, color=color, edgecolor="white", linewidth=0.7,
                   zorder=3, label=model)
        ax.text(maximum * 1.08, y, f"{median * 1000:.4g} us", va="center",
                fontsize=8, color=color)
    ax.set_xscale("log")
    ax.set_yticks(y_positions, MODELS)
    ax.set_xlabel("On-device inference latency [ms]")
    ax.set_title("On-Device Inference Latency: Median, P5-P95, and Min-Max")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "figure_02_latency_distributions_on_device_log_overview.png",
                bbox_inches="tight")
    plt.close(fig)


def dd_variant_latency_distributions(root: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    distributions: dict[str, np.ndarray] = {}
    statistics: list[dict] = []
    for variant in VARIANTS:
        values = []
        for cell in CELLS:
            values.extend(float(row["device_time_us"]) / 1000.0
                          for row in measurements(root, cell, variant))
        data = np.asarray(values)
        distributions[variant] = data
        statistics.append({
            "variant": VARIANT_LABELS[variant], "n_inferences": len(data),
            "n_unique_timings": len(np.unique(data)), "mean_ms": data.mean(),
            "median_ms": np.median(data), "p05_ms": np.percentile(data, 5),
            "p95_ms": np.percentile(data, 95), "minimum_ms": data.min(),
            "maximum_ms": data.max(),
        })
    return distributions, statistics


def plot_dd_variant_latency_distributions(
    distributions: dict[str, np.ndarray], out: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))
    for ax, variant in zip(axes, VARIANTS):
        values = distributions[variant]
        lo, hi = np.percentile(values, [0.05, 99.95])
        margin = max(0.04 * (hi - lo), 0.00002)
        bins = np.linspace(lo - margin, hi + margin, 42)
        color = VARIANT_COLORS[variant]
        ax.hist(values, bins=bins, facecolor=to_rgba(color, 0.24),
                edgecolor=color, linewidth=1.0)
        median = np.median(values)
        p05, p95 = np.percentile(values, [5, 95])
        ax.axvline(median, color=color, linewidth=1.2, linestyle="--",
                   label=f"Median: {median:.4g} ms")
        ax.axvspan(p05, p95, color=color, alpha=0.08,
                   label=f"P5-P95: {p05:.4g}-{p95:.4g} ms")
        ax.set_title(VARIANT_LABELS[variant])
        ax.set_xlabel("Inference latency [ms]")
        ax.set_ylabel("Inference count")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.legend(frameon=False, fontsize=7, title=f"n = {len(values):,}")
    fig.suptitle("DD Inference-Latency Distributions Across Six Cells", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "figure_04_dd_inference_mode_latency_distributions.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_memory(memory: dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    x = np.arange(len(MODELS)); colors = [COLORS[model] for model in MODELS]
    flash = [memory[model]["flash_load_bytes"] / 1024 for model in MODELS]
    ram = [memory[model]["static_ram_bytes"] / 1024 for model in MODELS]
    for ax, values, title, ylabel in ((axes[0], flash, "(a) Flash footprint", "Flash [KiB]"),
                                      (axes[1], ram, "(b) Static RAM footprint", "Static RAM [KiB]")):
        bars = ax.bar(x, values, facecolor=[to_rgba(color, 0.28) for color in colors],
                      edgecolor=colors, linewidth=1.35, width=0.62)
        ax.set_xticks(x, MODELS); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure_03_flash_and_ram.png", bbox_inches="tight")
    plt.close(fig)


def plot_variants(rows: list[dict], memory: dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1))
    x = np.arange(len(VARIANTS))
    labels = [VARIANT_LABELS[variant] for variant in VARIANTS]
    metric_specs = (
        ("dataset_max_abs_error", 100.0, "Maximum SOC error [percentage points]"),
        ("latency_median_us", 0.001, "Median inference time [ms]"),
    )
    for axis, (metric_name, scale, ylabel) in zip(axes[:2], metric_specs):
        for index, variant in enumerate(VARIANTS):
            label = VARIANT_LABELS[variant]
            values = scale * np.array([row[metric_name] for row in rows if row["variant"] == label])
            mean = values.mean()
            axis.bar(index, mean, width=0.62, facecolor=to_rgba(VARIANT_COLORS[variant], 0.28),
                     edgecolor=VARIANT_COLORS[variant], linewidth=1.3, zorder=2)
            axis.errorbar(index, mean, yerr=[[mean - values.min()], [values.max() - mean]], fmt="none",
                          ecolor=VARIANT_COLORS[variant], elinewidth=1.15, capsize=3.5, zorder=4)
            jitter = np.linspace(-0.06, 0.06, len(values))
            axis.scatter(index + jitter, values, s=17, color=VARIANT_COLORS[variant],
                         edgecolor="white", linewidth=0.35, zorder=5)
        axis.set_xticks(x, [label.replace(" ", "\n") for label in labels])
        axis.set_ylabel(ylabel)
    axes[0].set_title("(a) Maximum error")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Runtime")

    width = 0.35; xv = np.arange(len(VARIANTS))
    flash = [memory[variant]["flash_load_bytes"] / 1024 for variant in VARIANTS]
    ram = [memory[variant]["static_ram_bytes"] / 1024 for variant in VARIANTS]
    variant_edges = [VARIANT_COLORS[variant] for variant in VARIANTS]
    flash_bars = axes[2].bar(xv - width / 2, flash, width,
                             facecolor=[to_rgba(color, 0.20) for color in variant_edges],
                             edgecolor=variant_edges,
                             linewidth=1.25, hatch="..", label="Flash")
    ram_bars = axes[2].bar(xv + width / 2, ram, width,
                           facecolor=[to_rgba(color, 0.20) for color in variant_edges],
                           edgecolor=variant_edges,
                           linewidth=1.25, hatch="//", label="Static RAM")
    axes[2].set_xticks(xv, ["Rolling\nwindow", "Continuous\nstate", "Periodic\nreset"])
    axes[2].set_ylabel("Memory [KiB]"); axes[2].set_title("(c) Memory footprint")
    axes[2].set_ylim(0, 1.18 * max(flash + ram))
    axes[2].bar_label(flash_bars, labels=[f"{value:.1f}" for value in flash], fontsize=7, padding=2)
    axes[2].bar_label(ram_bars, labels=[f"{value:.1f}" for value in ram], fontsize=7, padding=2)
    mode_handles = [
        Patch(facecolor=to_rgba(VARIANT_COLORS[variant], 0.28),
              edgecolor=VARIANT_COLORS[variant], linewidth=1.3,
              label=VARIANT_LABELS[variant])
        for variant in VARIANTS
    ]
    resource_handles = [
        Patch(facecolor="#e5e5e5", edgecolor="#555555", hatch="..", label="Flash"),
        Patch(facecolor="#e5e5e5", edgecolor="#555555", hatch="//", label="Static RAM"),
    ]
    fig.legend(handles=mode_handles + resource_handles, frameon=False, ncol=5,
               fontsize=9.5, handlelength=2.0, columnspacing=1.8,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "figure_04_dd_inference_modes.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--out-dir", type=Path, default=root / "results" / "four_figure_summary")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    memory = load_memory(args.root)
    vectors = load_vectors(args.root)
    accuracy = collect_accuracy(args.root, vectors)
    accuracy_classes = aggregate_accuracy(accuracy)
    equivalence = aggregate_equivalence(accuracy)
    latency, repeatability = collect_latency(args.root)
    host_distributions, host_distribution_stats = latency_distributions(
        args.root, "host_rtt_ms"
    )
    device_distributions, device_distribution_stats = latency_distributions(
        args.root, "device_time_us", scale=0.001
    )
    variants = collect_variants(args.root, memory)
    dd_latency_distributions, dd_latency_statistics = dd_variant_latency_distributions(args.root)
    write_csv(args.out_dir / "soc_accuracy_and_equivalence.csv", accuracy)
    write_csv(args.out_dir / "soc_accuracy_by_load_class.csv", accuracy_classes)
    write_csv(args.out_dir / "hardware_software_equivalence_by_model.csv", equivalence)
    write_csv(args.out_dir / "latency_by_cell.csv", latency)
    write_csv(args.out_dir / "latency_by_replay_round.csv", repeatability)
    write_csv(args.out_dir / "latency_distribution_statistics.csv", host_distribution_stats)
    write_csv(args.out_dir / "latency_distribution_statistics_on_device.csv", device_distribution_stats)
    write_csv(args.out_dir / "dd_inference_modes.csv", variants)
    write_csv(args.out_dir / "dd_inference_mode_latency_distribution_statistics.csv",
              dd_latency_statistics)
    write_csv(args.out_dir / "memory_footprints.csv", [
        {"model": model, "flash_kib": memory[model]["flash_load_bytes"] / 1024,
         "static_ram_kib": memory[model]["static_ram_bytes"] / 1024}
        for model in ("DM", "HDM", "HECM", "DD", "DDS", "DDP")
    ])
    plot_accuracy(accuracy, args.out_dir)
    plot_latency_combined_detail(device_distributions, args.out_dir)
    plot_memory(memory, args.out_dir)
    plot_variants(variants, memory, args.out_dir)
    plot_dd_variant_latency_distributions(dd_latency_distributions, args.out_dir)
    print(f"Created five PNG figures and ten source tables in {args.out_dir}")


if __name__ == "__main__":
    main()
