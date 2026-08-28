#!/usr/bin/env python3
"""Create per-cell and load-class JES2 STM32 result tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("DM", "HDM", "HECM", "DD")
METRICS = (
    "dataset_mae", "dataset_rmse", "dataset_max_abs_error",
    "latency_median_us", "latency_p95_us", "latency_max_us",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--vectors-manifest", required=True, type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = load_json(args.vectors_manifest)
    load_class = {item["cell"]: item["load_class"] for item in manifest["vectors"]}
    memory_rows = {}
    if args.memory and args.memory.is_file():
        memory_rows = {row["model"]: row for row in load_json(args.memory).get("models", [])}
    rows = []
    for cell in load_class:
        for model in MODELS:
            path = args.results_root / cell / model / "summary.json"
            if not path.is_file():
                continue
            summary = load_json(path)
            dataset = summary.get("dataset_difference", {})
            reference = summary.get("reference_difference", {})
            latency = summary.get("device_time_us", {})
            memory = memory_rows.get(model, {})
            rows.append({
                "cell": cell, "load_class": load_class[cell], "model": model,
                "n_inferences": summary.get("rows_ok"),
                "dataset_mae": dataset.get("mae"),
                "dataset_rmse": dataset.get("rmse"),
                "dataset_max_abs_error": dataset.get("maximum_absolute_error"),
                "software_reference_mae": reference.get("mae"),
                "software_reference_rmse": reference.get("rmse"),
                "software_reference_max_abs_error": reference.get("maximum_absolute_error"),
                "latency_median_us": latency.get("median"),
                "latency_p95_us": latency.get("p95"),
                "latency_max_us": latency.get("maximum"),
                "flash_load_bytes": memory.get("flash_load_bytes"),
                "static_ram_bytes": memory.get("static_ram_bytes"),
            })
    if not rows:
        raise SystemExit(f"No cell/model summary.json files found below {args.results_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "hardware_results_by_cell.csv", rows)

    aggregate = []
    for load in ("low", "medium", "high"):
        for model in MODELS:
            group = [row for row in rows if row["load_class"] == load and row["model"] == model]
            if not group:
                continue
            item = {"load_class": load, "model": model, "n_cells": len(group)}
            for metric in METRICS:
                values = [float(row[metric]) for row in group if row[metric] is not None]
                item[f"{metric}_mean"] = sum(values) / len(values) if values else None
                item[f"{metric}_min"] = min(values) if values else None
                item[f"{metric}_max"] = max(values) if values else None
            aggregate.append(item)
    write_csv(args.out_dir / "hardware_results_by_load_class.csv", aggregate)
    (args.out_dir / "hardware_multicell_summary.json").write_text(
        json.dumps({"per_cell": rows, "by_load_class": aggregate, "high_class_limitation": manifest.get("high_class_limitation")}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} per-cell rows and {len(aggregate)} load-class rows")


if __name__ == "__main__":
    main()
