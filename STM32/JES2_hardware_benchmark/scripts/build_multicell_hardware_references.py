#!/usr/bin/env python3
"""Rebuild six-cell software references with the frozen hardware-input SOH traces."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


CELLS = ("C09", "C13", "C15", "C25", "C27", "C29")
# DD is exported deterministically through the validated CPU ONNX fixed-window path.
MODELS = ("DM", "HDM", "HECM")


def replace_arg(command: list[str], name: str, value: Path) -> None:
    index = command.index(name)
    command[index + 1] = str(value.resolve())


def main() -> None:
    workspace = Path(__file__).resolve().parents[3]
    simulation = workspace / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-manifest", type=Path,
        default=simulation / "campaigns/jes2_initial_state_paired_sixcell_20260827_cuda/jes2_manifest.json",
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=simulation / "campaigns/jes2_hardware_reference_multicell_20260828/runs",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.template_manifest.read_text(encoding="utf-8"))["runs"]
    selected = {}
    for record in records:
        primary = (
            record["alias"] == "baseline"
            and record["window_id"] == f"{record['cell']}_fresh"
            and int(record["seed"]) == 42
            and ((record["model"] == "DM" and record.get("soh_mode") == "none")
                 or (record["model"] != "DM" and record.get("soh_condition") == "lstm_h1"))
        )
        if primary:
            selected[(record["cell"], record["model"])] = record

    for cell in CELLS:
        frozen_trace = simulation / f"campaigns/jes2_full_{cell}_20260825/traces/{cell}/{cell}_fresh/baseline/seed_42/lstm_h1.npz"
        if not frozen_trace.is_file():
            raise FileNotFoundError(frozen_trace)
        for model in MODELS:
            record = selected[(cell, model)]
            mode = "no_soh" if model == "DM" else "lstm_h1"
            out_dir = args.out_root / cell / f"{cell}_fresh/baseline/seed_42" / mode / model
            summary = out_dir / "summary.json"
            if summary.is_file() and not args.force:
                print(f"skip {cell} {model}: {summary}", flush=True)
                continue
            command = list(record["command"])
            replace_arg(command, "--out_dir", out_dir)
            if "--soh_trace" in command:
                replace_arg(command, "--soh_trace", frozen_trace)
            print(f"run  {cell} {model}", flush=True)
            subprocess.run(command, check=True)
    print(f"Hardware software references are complete below {args.out_root}")


if __name__ == "__main__":
    main()
