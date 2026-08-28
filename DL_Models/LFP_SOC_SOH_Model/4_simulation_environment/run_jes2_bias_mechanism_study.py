from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
MODELS = ("DM", "HDM", "HECM", "DD")
STATES = ("fresh", "mid_life", "aged")
ALIASES = ("baseline", "positive_3pct", "negative_3pct")
RESET_MODES = ("natural", "no_full_charge_reset")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def replace_option(command: list[str], name: str, value: str) -> None:
    if name in command:
        command[command.index(name) + 1] = value
    else:
        command.extend([name, value])


def remove_flag(command: list[str], name: str) -> None:
    while name in command:
        command.remove(name)


def write_manifest(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def primary(record: dict) -> bool:
    if int(record.get("seed", -1)) != 42:
        return False
    if record["model"] == "DM":
        return record.get("soh_mode") == "none"
    return record.get("soh_condition") == "lstm_h1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the focused JES2 bias accumulation/reset study.")
    parser.add_argument(
        "--source-manifest", type=Path,
        default=ROOT / "campaigns/jes2_full_holdout_merged_20260825.json",
    )
    parser.add_argument(
        "--signed-manifest", type=Path,
        default=ROOT / "campaigns/jes2_signed_bias_20260826/jes2_manifest.json",
    )
    parser.add_argument("--tag", default="jes2_bias_mechanism_C29_20260828")
    parser.add_argument("--cells", nargs="+", default=["C29"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))["runs"]
    signed = json.loads(args.signed_manifest.read_text(encoding="utf-8"))["runs"]
    source_index = {}
    for record in source:
        if record.get("cell") not in args.cells or record.get("model") not in MODELS or not primary(record):
            continue
        if record.get("alias") not in {"baseline", "current_bias_3p0pct"}:
            continue
        source_index[(record["cell"], record["window_id"], record["alias"], record["model"])] = record
    signed_index = {
        (record["cell"], record["window_id"], record["model"]): record
        for record in signed
        if record.get("cell") in args.cells
        and record.get("alias") == "current_bias_neg_3p0pct"
        and record.get("model") in MODELS
        and primary(record)
    }

    campaign = ROOT / "campaigns" / args.tag
    manifest_path = campaign / "jes2_manifest.json"
    expected = len(args.cells) * len(STATES) * len(RESET_MODES) * len(ALIASES) * len(MODELS)
    manifest = {
        "tag": args.tag,
        "started_utc": utc_now(),
        "purpose": "causal separation of time accumulation, aging state, and full-charge reset under +/-3% current gain error",
        "cells": args.cells,
        "states": list(STATES),
        "models": list(MODELS),
        "aliases": list(ALIASES),
        "reset_modes": list(RESET_MODES),
        "expected_runs": expected,
        "runs": [],
    }
    write_manifest(manifest_path, manifest)

    completed = 0
    for cell in args.cells:
        for state in STATES:
            window_id = f"{cell}_{state}"
            for reset_mode in RESET_MODES:
                for alias in ALIASES:
                    template_alias = "baseline" if alias == "baseline" else "current_bias_3p0pct"
                    for model in MODELS:
                        template = source_index.get((cell, window_id, template_alias, model))
                        if template is None:
                            raise KeyError(f"Missing source command: {cell} {window_id} {template_alias} {model}")
                        out_dir = campaign / "runs" / cell / window_id / reset_mode / alias / model
                        summary = out_dir / "summary.json"
                        command = list(template["command"])
                        remove_flag(command, "--summary_only")
                        replace_option(command, "--out_dir", str(out_dir))
                        if alias == "positive_3pct":
                            replace_option(command, "--current_offset_pct", "0.03")
                        elif alias == "negative_3pct":
                            replace_option(command, "--current_offset_pct", "-0.03")
                            if model != "DM":
                                signed_record = signed_index[(cell, window_id, model)]
                                trace = ROOT / "campaigns/jes2_signed_bias_20260826/traces" / cell / window_id / "current_bias_neg_3p0pct_lstm_h1.npz"
                                if not trace.is_file():
                                    raise FileNotFoundError(trace)
                                replace_option(command, "--soh_trace", str(trace))
                        if reset_mode == "no_full_charge_reset":
                            replace_option(command, "--v_max", "99.0")

                        row = {
                            "cell": cell,
                            "window_id": window_id,
                            "soh_state": state,
                            "reset_mode": reset_mode,
                            "alias": alias,
                            "model": model,
                            "out_dir": str(out_dir),
                            "command": command,
                            "status": "running",
                        }
                        manifest["runs"].append(row)
                        write_manifest(manifest_path, manifest)
                        if args.skip_existing and summary.is_file():
                            row["status"] = "skipped_existing"
                        else:
                            out_dir.mkdir(parents=True, exist_ok=True)
                            subprocess.run(command, cwd=WORKSPACE, check=True)
                            row["status"] = "completed"
                        completed += 1
                        write_manifest(manifest_path, manifest)
                        print(f"BIAS_MECHANISM_PROGRESS={completed}/{expected}", flush=True)

    manifest["finished_utc"] = utc_now()
    write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "runs": completed}, indent=2))


if __name__ == "__main__":
    main()
