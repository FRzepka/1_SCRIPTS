from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jes2_protocol import (
    DEFAULT_REFERENCE_ALIASES,
    INITIAL_STATE_APPLICABLE_MODELS,
    MODEL_ORDER,
    PRIMARY_STOCHASTIC_ALIASES,
    SCENARIOS,
    STOCHASTIC_ALIASES,
)
from robustness_common import STRATIFICATION_PROTOCOL


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
DEFAULT_SOC_ROOT = WORKSPACE / "DL_Models" / "LFP_SOC_SOH_Model" / "2_models" / "SOC_1.7.0.0" / "PrunedFT_1.7.0.0_s30_struct"
DEFAULT_SOH_ROOT = WORKSPACE / "DL_Models" / "LFP_SOC_SOH_Model" / "2_models" / "SOH_JES2_0.1.0"

RUNNERS = {
    "DM": ROOT / "CC_1.0.0" / "run_cc_scenario.py",
    "HDM": ROOT / "CC_SOH_1.0.0" / "run_cc_soh_scenario.py",
    "HECM": ROOT / "ECM_0.0.3" / "run_ecm_scenario.py",
    "DD": ROOT / "SOC_SOH_1.7.0.0_0.1.2.3" / "run_soc_soh_scenario.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_cell(cell: str) -> str:
    return cell.rsplit("_", 1)[-1]


def validate_split(soc_config: Path, soh_config: Path, cells: list[str], allow_non_holdout: bool) -> dict[str, Any]:
    soc_cells = read_yaml(soc_config)["cells"]
    soh_cells = read_yaml(soh_config)["cells"]
    if soc_cells["train"] != soh_cells["train"] or soc_cells["val"] != soh_cells["val"]:
        raise ValueError("SOC and SOH train/validation splits are not identical")

    holdout = {canonical_cell(cell) for cell in soh_cells.get("test", [])}
    requested = {canonical_cell(cell) for cell in cells}
    invalid = sorted(requested - holdout)
    if invalid and not allow_non_holdout:
        raise ValueError(
            f"Non-holdout cells requested: {invalid}. Use --allow_non_holdout only for diagnostics."
        )
    return {
        "train": soc_cells["train"],
        "validation": soc_cells["val"],
        "holdout": soh_cells.get("test", []),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def execute(cmd: list[str], dry_run: bool) -> None:
    print("RUN", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=WORKSPACE, check=True)


def completed_summary(out_dir: Path) -> bool:
    return (out_dir / "summary.json").is_file()


def load_evaluation_windows(path: Path, cells: list[str]) -> dict[str, list[dict[str, Any]]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("windows", payload)
    else:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    requested = {canonical_cell(cell): cell for cell in cells}
    windows: dict[str, list[dict[str, Any]]] = {cell: [] for cell in cells}
    for source in rows:
        canonical = canonical_cell(str(source["cell"]))
        if canonical not in requested:
            continue
        row = dict(source)
        for key in ["start_row", "primary_rows", "event_rows"]:
            row[key] = int(float(row[key]))
        row["cell"] = requested[canonical]
        windows[row["cell"]].append(row)
    missing = [cell for cell, selected in windows.items() if not selected]
    if missing:
        raise ValueError(f"Window manifest has no rows for requested cells: {missing}")
    return windows


def scenario_repeat_count(alias: str, primary: int, secondary: int) -> int:
    if alias in PRIMARY_STOCHASTIC_ALIASES:
        return primary
    if alias in STOCHASTIC_ALIASES:
        return secondary
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JES2 multi-cell benchmark with one shared causal SOH trace per scenario."
    )
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER,
                        help="Estimator subset to execute; intended for shard/backfill runs.")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--aliases", nargs="*", default=None)
    parser.add_argument("--soh_modes", nargs="+", choices=["lstm", "reference"], default=["lstm", "reference"])
    parser.add_argument("--reference_aliases", nargs="*", default=sorted(DEFAULT_REFERENCE_ALIASES),
                        help="Scenario aliases evaluated with ideal reference SOH.")
    parser.add_argument("--lstm_publish_intervals", nargs="+", type=int, default=[1],
                        help="SOH publication cadence in multiples of the trained one-hour interval.")
    parser.add_argument("--reference_publish_intervals", nargs="+", type=int, default=[1])
    parser.add_argument("--cadence_aliases", nargs="+", default=["baseline"],
                        help="Aliases receiving all LSTM cadences; other aliases use the primary 1 h cadence.")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--stochastic_repeats", type=int, default=10,
                        help="Repeat count for Gaussian sensor-noise scenarios.")
    parser.add_argument("--secondary_stochastic_repeats", type=int, default=5,
                        help="Repeat count for random missing, jitter, and spike scenarios.")
    parser.add_argument("--trace_device", default=None)
    parser.add_argument("--model_device", default="cpu")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--keep_run_artifacts", action="store_true",
                        help="Keep per-sample CSV/plots for every model run (large campaigns should omit this).")
    parser.add_argument("--allow_non_holdout", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--window_manifest", type=Path, default=None,
                        help="Frozen JES2 CSV/JSON windows; replaces the single start/max row window.")
    parser.add_argument("--soh_context_rows", type=int, default=691200,
                        help="Undisturbed 192 h SOH initialization context at nominal 1 Hz.")
    parser.add_argument("--data_root", default="/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
    parser.add_argument("--soc_config", type=Path, default=DEFAULT_SOC_ROOT / "config" / "train_soc.yaml")
    parser.add_argument("--soc_ckpt", type=Path, default=DEFAULT_SOC_ROOT / "checkpoints" / "best_model_finetuned.pt")
    parser.add_argument("--soc_scaler", type=Path, default=DEFAULT_SOC_ROOT / "scaler_robust.joblib")
    parser.add_argument("--soh_config", type=Path, default=DEFAULT_SOH_ROOT / "config" / "train_soh.yaml")
    parser.add_argument("--soh_ckpt", type=Path, default=DEFAULT_SOH_ROOT / "checkpoints" / "best_model.pt")
    parser.add_argument("--soh_scaler", type=Path, default=DEFAULT_SOH_ROOT / "scaler_robust.joblib")
    args = parser.parse_args()

    cadence_values = args.lstm_publish_intervals + args.reference_publish_intervals
    if any(value < 1 for value in cadence_values):
        parser.error("SOH publication intervals must be positive integers")
    args.lstm_publish_intervals = sorted(set(args.lstm_publish_intervals))
    args.reference_publish_intervals = sorted(set(args.reference_publish_intervals))
    if 1 not in args.lstm_publish_intervals or 1 not in args.reference_publish_intervals:
        parser.error("The primary 1 h SOH publication interval must be included for LSTM and reference modes")
    if args.stochastic_repeats < 1 or args.secondary_stochastic_repeats < 1:
        parser.error("Stochastic repeat counts must be positive")
    if args.soh_context_rows < 0:
        parser.error("soh_context_rows must not be negative")

    artifact_paths = [args.soc_config, args.soc_ckpt, args.soc_scaler, args.soh_config, args.soh_ckpt, args.soh_scaler]
    missing = [str(path) for path in artifact_paths if not path.is_file()]
    if missing:
        parser.error(f"Missing model artifacts: {missing}")

    split = validate_split(args.soc_config, args.soh_config, args.cells, args.allow_non_holdout)
    if args.window_manifest:
        if not args.window_manifest.is_file():
            parser.error(f"Window manifest not found: {args.window_manifest}")
        windows_by_cell = load_evaluation_windows(args.window_manifest, args.cells)
    else:
        windows_by_cell = {
            cell: [{
                "window_id": "single_window",
                "cell": cell,
                "soh_state": "all",
                "cell_load_class": "unassigned",
                "start_row": args.start_row,
                "primary_rows": args.max_rows,
                "event_rows": args.max_rows,
            }]
            for cell in args.cells
        }
    selected_aliases = set(args.aliases) if args.aliases else None
    scenarios = [row for row in SCENARIOS if selected_aliases is None or row[0] in selected_aliases]
    if selected_aliases:
        unknown = selected_aliases - {row[0] for row in scenarios}
        if unknown:
            parser.error(f"Unknown scenario aliases: {sorted(unknown)}")
    unknown_reference = set(args.reference_aliases) - {row[0] for row in SCENARIOS}
    if unknown_reference:
        parser.error(f"Unknown reference-SOH aliases: {sorted(unknown_reference)}")
    unknown_cadence = set(args.cadence_aliases) - {row[0] for row in SCENARIOS}
    if unknown_cadence:
        parser.error(f"Unknown cadence aliases: {sorted(unknown_cadence)}")

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M_jes2")
    campaign_dir = ROOT / "campaigns" / tag
    manifest_path = campaign_dir / "jes2_manifest.json"
    manifest: dict[str, Any] = {
        "benchmark_version": "JES2.0",
        "tag": tag,
        "started_utc": utc_now(),
        "python": sys.executable,
        "cells": args.cells,
        "models": args.models,
        "split": split,
        "soh_modes": args.soh_modes,
        "reference_aliases": args.reference_aliases,
        "lstm_publish_intervals": args.lstm_publish_intervals,
        "reference_publish_intervals": args.reference_publish_intervals,
        "cadence_aliases": args.cadence_aliases,
        "base_seed": args.base_seed,
        "stochastic_repeats": args.stochastic_repeats,
        "secondary_stochastic_repeats": args.secondary_stochastic_repeats,
        "window": {
            "start_row": args.start_row,
            "max_rows": args.max_rows,
            "manifest": str(args.window_manifest.resolve()) if args.window_manifest else None,
            "soh_context_rows": args.soh_context_rows,
            "selection_uses_model_outputs": False if args.window_manifest else None,
            "definitions": [
                {
                    "cell": cell,
                    "window_id": str(window["window_id"]),
                    "soh_state": window.get("soh_state", "all"),
                    "start_row": int(window["start_row"]),
                    "primary_rows": int(window["primary_rows"]),
                    "event_rows": int(window["event_rows"]),
                }
                for cell, cell_windows in windows_by_cell.items()
                for window in cell_windows
            ],
        },
        "output_policy": "full_run_artifacts" if args.keep_run_artifacts else "summary_only",
        "artifacts": {
            key: {
                "path": str(getattr(args, key).resolve()),
                "sha256": sha256(getattr(args, key)),
            }
            for key in ["soc_config", "soc_ckpt", "soc_scaler", "soh_config", "soh_ckpt", "soh_scaler"]
        },
        "protocol": {
            "statistical_unit": "cell (windows averaged within cell before inference)",
            "initial_state_comparison_models": sorted(INITIAL_STATE_APPLICABLE_MODELS),
            "initial_state_dd_policy": "equivalent_soc_offset_applied_to_q_c_report_realized_output_error",
            "initial_state_realization": {
                "DM": "coulomb_counting_soc_state",
                "HDM": "soh_corrected_coulomb_counting_soc_state",
                "HECM": "ekf_soc_state",
                "DD": "q_c_feature_offset_equal_to_soc_delta_times_nominal_capacity",
            },
            "common_recovery_abs_error_threshold": 0.02,
            "common_recovery_sustain_seconds": 300.0,
            "stratification": STRATIFICATION_PROTOCOL,
            "scenarios": [
                {"alias": alias, "scenario": scenario, "arguments": scenario_args}
                for alias, scenario, scenario_args in scenarios
            ],
        },
        "runs": [],
    }
    write_manifest(manifest_path, manifest)

    for cell in args.cells:
        for window in windows_by_cell[cell]:
            window_id = str(window["window_id"])
            for alias, scenario, scenario_args in scenarios:
                repeat_count = scenario_repeat_count(
                    alias, args.stochastic_repeats, args.secondary_stochastic_repeats
                )
                for repeat in range(repeat_count):
                    seed = args.base_seed + repeat
                    max_rows = int(window["event_rows"] if alias == "missing_gap_1h" else window["primary_rows"])
                    window_args = ["--start_row", str(int(window["start_row"]))]
                    if max_rows > 0:
                        window_args.extend(["--max_rows", str(max_rows)])
                    common = [
                        "--cell", cell, "--scenario", scenario, "--seed", str(seed),
                        *window_args, *scenario_args,
                    ]
                    artifact_args = [] if args.keep_run_artifacts else ["--summary_only"]
                    record_window = {
                        "window_id": window_id,
                        "soh_state": window.get("soh_state", "all"),
                        "cell_load_class": window.get("cell_load_class", "unassigned"),
                        "start_row": int(window["start_row"]),
                        "max_rows": max_rows,
                    }

                    if "DM" in args.models:
                        dm_out = (
                            campaign_dir / "runs" / cell / window_id / alias /
                            f"seed_{seed}" / "no_soh" / "DM"
                        )
                        dm_record = {
                            "cell": cell,
                            "alias": alias,
                            "scenario": scenario,
                            "seed": seed,
                            "soh_mode": "none",
                            "model": "DM",
                            "out_dir": str(dm_out),
                            "status": "running",
                            **record_window,
                        }
                        if args.skip_existing and completed_summary(dm_out):
                            dm_record["status"] = "skipped_existing"
                        else:
                            dm_out.mkdir(parents=True, exist_ok=True)
                            dm_cmd = [
                                sys.executable, str(RUNNERS["DM"]), "--out_dir", str(dm_out),
                                "--data_root", args.data_root, *artifact_args, *common,
                            ]
                            dm_record["command"] = dm_cmd
                            execute(dm_cmd, args.dry_run)
                            dm_record["status"] = "dry_run" if args.dry_run else "completed"
                        manifest["runs"].append(dm_record)
                        write_manifest(manifest_path, manifest)

                    soh_models = [model for model in args.models if model != "DM"]
                    if not soh_models:
                        continue

                    for mode in args.soh_modes:
                        if mode == "reference" and alias not in set(args.reference_aliases):
                            continue
                        publish_values = (
                            args.lstm_publish_intervals if mode == "lstm" else args.reference_publish_intervals
                        )
                        if mode == "lstm" and alias not in set(args.cadence_aliases):
                            publish_values = [1]
                        for publish_every in publish_values:
                            condition = f"{mode}_h{publish_every}"
                            trace_path = (
                                campaign_dir / "traces" / cell / window_id / alias /
                                f"seed_{seed}" / f"{condition}.npz"
                            )
                            trace_context_rows = args.soh_context_rows if args.window_manifest else 0
                            trace_cmd = [
                                sys.executable,
                                str(ROOT / "shared_soh_trace.py"),
                                "--mode", mode,
                                "--cell", cell,
                                "--out", str(trace_path),
                                "--data_root", args.data_root,
                                "--soh_config", str(args.soh_config),
                                "--soh_ckpt", str(args.soh_ckpt),
                                "--soh_scaler", str(args.soh_scaler),
                                "--publish_every_intervals", str(publish_every),
                                "--context_rows", str(trace_context_rows),
                                *common[2:],
                            ]
                            if args.trace_device:
                                trace_cmd.extend(["--device", args.trace_device])
                            if not (args.skip_existing and trace_path.is_file()):
                                execute(trace_cmd, args.dry_run)

                            shared_args = ["--soh_trace", str(trace_path)]
                            model_commands = {
                                "HDM": [
                                    sys.executable, str(RUNNERS["HDM"]), "--device", args.model_device,
                                    "--data_root", args.data_root,
                                    "--soh_config", str(args.soh_config), "--soh_ckpt", str(args.soh_ckpt),
                                    "--soh_scaler", str(args.soh_scaler),
                                ],
                                "HECM": [
                                    sys.executable, str(RUNNERS["HECM"]), "--device", args.model_device,
                                    "--data_root", args.data_root,
                                ],
                                "DD": [
                                    sys.executable, str(RUNNERS["DD"]), "--device", args.model_device,
                                    "--data_root", args.data_root,
                                    "--soc_config", str(args.soc_config), "--soc_ckpt", str(args.soc_ckpt),
                                    "--soc_scaler", str(args.soc_scaler), "--soh_config", str(args.soh_config),
                                    "--soh_ckpt", str(args.soh_ckpt), "--soh_scaler", str(args.soh_scaler),
                                ],
                            }
                            model_commands = {
                                model: command for model, command in model_commands.items() if model in soh_models
                            }
                            for model_name, model_cmd in model_commands.items():
                                out_dir = (
                                    campaign_dir / "runs" / cell / window_id / alias /
                                    f"seed_{seed}" / condition / model_name
                                )
                                record = {
                                    "cell": cell,
                                    "alias": alias,
                                    "scenario": scenario,
                                    "seed": seed,
                                    "soh_mode": mode,
                                    "soh_condition": condition,
                                    "soh_publish_intervals": publish_every,
                                    "soh_trace": str(trace_path),
                                    "model": model_name,
                                    "out_dir": str(out_dir),
                                    "status": "running",
                                    **record_window,
                                }
                                if args.skip_existing and completed_summary(out_dir):
                                    record["status"] = "skipped_existing"
                                else:
                                    out_dir.mkdir(parents=True, exist_ok=True)
                                    model_run_cmd = [
                                        *model_cmd, "--out_dir", str(out_dir), *shared_args,
                                        *artifact_args, *common,
                                    ]
                                    record["command"] = model_run_cmd
                                    execute(model_run_cmd, args.dry_run)
                                    record["status"] = "dry_run" if args.dry_run else "completed"
                                manifest["runs"].append(record)
                                write_manifest(manifest_path, manifest)

    manifest["finished_utc"] = utc_now()
    write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "runs": len(manifest["runs"])}, indent=2))


if __name__ == "__main__":
    main()
