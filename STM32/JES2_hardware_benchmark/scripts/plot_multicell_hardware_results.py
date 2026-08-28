#!/usr/bin/env python3
"""Create multicell STM32 benchmark tables and publication plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"DM": "#2ca02c", "HDM": "#9467bd", "HECM": "#1f77b4", "DD": "#d62728"}
DD_COLORS = {"Rolling window": "#d62728", "Continuous state": "#2ca02c", "Periodic reset": "#1f77b4"}
LOAD_ORDER = ("low", "medium", "high")
MODEL_ORDER = ("DM", "HDM", "HECM", "DD")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_tables(out_dir: Path, cell_rows: list[dict], load_rows: list[dict]) -> None:
    cells = list(dict.fromkeys(row["cell"] for row in cell_rows))
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Cell & Load class & DM & HDM & HECM & DD \\", r"\midrule"]
    for cell in cells:
        group = [row for row in cell_rows if row["cell"] == cell]
        values = {row["model"]: 100 * float(row["dataset_mae"]) for row in group}
        lines.append(f'{cell} & {group[0]["load_class"]} & ' + " & ".join(f'{values[model]:.3f}' for model in MODEL_ORDER) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "hardware_accuracy_by_cell.tex").write_text("\n".join(lines) + "\n", encoding="ascii")

    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Load class & Model & $n$ & Mean & Minimum & Maximum \\", r"\midrule"]
    for row in load_rows:
        lines.append(f'{row["load_class"]} & {row["model"]} & {row["n_cells"]} & '
                     f'{100 * float(row["dataset_mae_mean"]):.3f} & '
                     f'{100 * float(row["dataset_mae_min"]):.3f} & '
                     f'{100 * float(row["dataset_mae_max"]):.3f}' + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "hardware_accuracy_by_load_class.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


def errors(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    delta = prediction - target
    return float(np.mean(np.abs(delta))), float(np.sqrt(np.mean(delta**2))), float(np.max(np.abs(delta)))


def first_round(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    return [row for row in rows if int(row["round"]) == 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--out-dir", type=Path, default=root / "results" / "multicell_summary")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.root / "test_vectors" / "multicell" / "jes2_multicell_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = manifest["vectors"]
    cell_rows: list[dict] = []
    dd_rows: list[dict] = []

    for item in cells:
        cell = item["cell"]
        vector_rows = read_csv(manifest_path.parent / item["csv"])
        dataset = np.array([float(row["soc_dataset"]) for row in vector_rows])
        first_valid = int(item["dd_first_valid_sample_id"])

        for model in MODEL_ORDER[:-1]:
            summary_path = args.root / "results" / cell / model / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metric = summary["dataset_difference"]
            latency = summary["device_time_us"]
            cell_rows.append({
                "cell": cell, "load_class": item["load_class"], "model": model,
                "source": "STM32, three replay rounds", "n_accuracy_samples": metric["n"] // 3,
                "dataset_mae": metric["mae"], "dataset_rmse": metric["rmse"],
                "dataset_max_abs_error": metric["maximum_absolute_error"],
                "latency_median_us": latency["median"], "latency_p95_us": latency["p95"],
            })

        valid_vectors = [row for row in vector_rows if int(row["sample_id"]) >= first_valid]
        dd_prediction = np.array([float(row["expected_dd"]) for row in valid_vectors])
        dd_target = np.array([float(row["soc_dataset"]) for row in valid_vectors])
        mae, rmse, maximum = errors(dd_prediction, dd_target)
        cell_rows.append({
            "cell": cell, "load_class": item["load_class"], "model": "DD",
            "source": "rolling-window software reference; C kernel hardware-validated",
            "n_accuracy_samples": len(valid_vectors), "dataset_mae": mae, "dataset_rmse": rmse,
            "dataset_max_abs_error": maximum, "latency_median_us": 723960.43,
            "latency_p95_us": "",
        })

        variants = (("Rolling window", None), ("Continuous state", "DDS"), ("Periodic reset", "DDP"))
        for label, firmware in variants:
            if firmware is None:
                pred = dd_prediction
                latency = 723960.43
            else:
                measured = first_round(args.root / "results" / cell / firmware / "measurements.csv")
                measured = [row for row in measured if int(row["sample_id"]) >= first_valid]
                pred = np.array([float(row["soc_device"]) for row in measured])
                summary = json.loads((args.root / "results" / cell / firmware / "summary.json").read_text(encoding="utf-8"))
                latency = float(summary["device_time_us"]["median"])
            mae, rmse, maximum = errors(pred, dd_target)
            ref_mae, ref_rmse, ref_max = errors(pred, dd_prediction)
            dd_rows.append({
                "cell": cell, "load_class": item["load_class"], "variant": label,
                "common_horizon_start": first_valid, "n_samples": len(dd_target),
                "dataset_mae": mae, "dataset_rmse": rmse, "dataset_max_abs_error": maximum,
                "rolling_reference_mae": ref_mae, "rolling_reference_rmse": ref_rmse,
                "rolling_reference_max_abs_error": ref_max, "latency_median_us": latency,
            })

    write_csv(args.out_dir / "hardware_accuracy_by_cell.csv", cell_rows)
    write_csv(args.out_dir / "dd_variants_common_horizon_by_cell.csv", dd_rows)

    load_rows: list[dict] = []
    for load in LOAD_ORDER:
        for model in MODEL_ORDER:
            group = [row for row in cell_rows if row["load_class"] == load and row["model"] == model]
            values = np.array([float(row["dataset_mae"]) for row in group])
            load_rows.append({"load_class": load, "model": model, "n_cells": len(group),
                              "dataset_mae_mean": values.mean(), "dataset_mae_min": values.min(),
                              "dataset_mae_max": values.max()})
    write_csv(args.out_dir / "hardware_accuracy_by_load_class.csv", load_rows)
    write_latex_tables(args.out_dir, cell_rows, load_rows)

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(cells))
    offsets = np.linspace(-0.27, 0.27, len(MODEL_ORDER))
    for offset, model in zip(offsets, MODEL_ORDER):
        values = [100 * float(next(row["dataset_mae"] for row in cell_rows if row["cell"] == item["cell"] and row["model"] == model)) for item in cells]
        ax.scatter(x + offset, values, s=38, color=COLORS[model], label=model, zorder=3)
    ax.set_xticks(x, [f'{item["cell"]}\n{item["load_class"]}' for item in cells])
    ax.set_ylabel("SOC MAE [percentage points]")
    ax.legend(ncol=4, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(args.out_dir / "hardware_accuracy_by_cell.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "hardware_accuracy_by_cell.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    latency = []
    for model in MODEL_ORDER:
        values = [float(row["latency_median_us"]) for row in cell_rows if row["model"] == model]
        latency.append(values[0] if model == "DD" else float(np.median(values)))
    bars = ax.bar(MODEL_ORDER, latency, color=[COLORS[model] for model in MODEL_ORDER], width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("Median inference time [us, logarithmic scale]")
    ax.bar_label(bars, labels=[f"{value:.2f}" if value < 100 else f"{value / 1000:.1f} ms" for value in latency], padding=3)
    ax.text(3, latency[3] / 2.3, "C27 timing", ha="center", va="top", color="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "hardware_latency_by_model.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "hardware_latency_by_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(LOAD_ORDER))
    width = 0.18
    for index, model in enumerate(MODEL_ORDER):
        group = [row for row in load_rows if row["model"] == model]
        means = 100 * np.array([float(row["dataset_mae_mean"]) for row in group])
        lower = means - 100 * np.array([float(row["dataset_mae_min"]) for row in group])
        upper = 100 * np.array([float(row["dataset_mae_max"]) for row in group]) - means
        ax.errorbar(x + (index - 1.5) * width, means, yerr=np.vstack([lower, upper]), fmt="o",
                    capsize=4, color=COLORS[model], label=model)
    ax.set_xticks(x, ["Low (n=2)", "Medium (n=3)", "High (n=1)"])
    ax.set_ylabel("Mean SOC MAE [percentage points]")
    ax.legend(ncol=4, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(args.out_dir / "hardware_accuracy_by_load_class.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "hardware_accuracy_by_load_class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for variant, color in DD_COLORS.items():
        rows = [row for row in dd_rows if row["variant"] == variant]
        ax.plot([row["cell"] for row in rows], [100 * float(row["dataset_mae"]) for row in rows],
                marker="o", linewidth=1.6, color=color, label=variant)
    ax.set_ylabel("SOC MAE on common horizon [percentage points]")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_dir / "dd_variants_accuracy_by_cell.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "dd_variants_accuracy_by_cell.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {len(cell_rows)} model rows and {len(dd_rows)} DD-variant rows to {args.out_dir}")


if __name__ == "__main__":
    main()
