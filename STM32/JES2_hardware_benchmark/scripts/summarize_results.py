#!/usr/bin/env python3
"""Combine JES2 STM32 timing and memory summaries into one paper-facing table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("DM", "HDM", "HECM", "DD")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--memory", type=Path, help="Optional memory.json from extract_memory_report.py")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    memory_path = args.memory or args.results_root / "memory.json"
    memory_rows = {}
    if memory_path.is_file():
        memory_rows = {row["model"]: row for row in load_json(memory_path).get("models", [])}

    rows = []
    for model in MODELS:
        summary_path = args.results_root / model / "summary.json"
        if not summary_path.is_file():
            continue
        timing = load_json(summary_path)
        memory = memory_rows.get(model, {})
        rows.append({
            "model": model,
            "n_inferences": timing.get("rows_ok"),
            "device_time_median_us": timing.get("device_time_us", {}).get("median"),
            "device_time_p95_us": timing.get("device_time_us", {}).get("p95"),
            "device_time_max_us": timing.get("device_time_us", {}).get("maximum"),
            "reference_mae": timing.get("reference_difference", {}).get("mae"),
            "reference_rmse": timing.get("reference_difference", {}).get("rmse"),
            "reference_max_abs_error": timing.get("reference_difference", {}).get("maximum_absolute_error"),
            "flash_load_bytes": memory.get("flash_load_bytes"),
            "static_ram_bytes": memory.get("static_ram_bytes"),
            "peak_stack_bytes": None,
            "activation_buffer_bytes": None,
            "firmware_revision": timing.get("device", {}).get("firmware_revision"),
            "clock_hz": timing.get("device", {}).get("clock_hz"),
            "vectors_sha256": timing.get("vectors_sha256"),
        })
    if not rows:
        raise SystemExit(f"No model summary.json files found below {args.results_root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".json").write_text(json.dumps({"models": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {len(rows)} model rows")


if __name__ == "__main__":
    main()
