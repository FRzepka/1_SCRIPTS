from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
MODELS = ("DM", "HDM", "HECM", "DD")
ALIASES = {"baseline": 0.0, "positive_3pct": 0.03, "negative_3pct": -0.03}


def replace_option(command: list[str], name: str, value: str) -> None:
    if name in command:
        command[command.index(name) + 1] = value
    else:
        command.extend([name, value])


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a continuous C29 lifecycle current-bias mechanism test.")
    parser.add_argument(
        "--source-manifest", type=Path,
        default=ROOT / "campaigns/jes2_full_holdout_merged_20260825.json",
    )
    parser.add_argument(
        "--soh-trace", type=Path,
        default=ROOT / "campaigns/jes2_six_cell_pilot_C29_20260825/traces/C29/baseline/seed_42/lstm_h1.npz",
    )
    parser.add_argument(
        "--baseline-manifest", type=Path,
        default=ROOT / "campaigns/jes2_six_cell_pilot_C29_20260825/jes2_manifest.json",
    )
    parser.add_argument("--tag", default="jes2_c29_lifecycle_bias_20260828")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--aliases", nargs="+", choices=tuple(ALIASES), default=tuple(ALIASES))
    parser.add_argument("--force-rerun-baseline", action="store_true")
    parser.add_argument("--temporal-metrics-seconds", type=float, default=0.0)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))["runs"]
    templates = {}
    for model in MODELS:
        matches = [
            row for row in source
            if row.get("cell") == "C29" and row.get("window_id") == "C29_mid_life"
            and row.get("alias") == "baseline" and row.get("model") == model
            and (model == "DM" or row.get("soh_condition") == "lstm_h1")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one source template for {model}, found {len(matches)}")
        templates[model] = matches[0]
    baseline_records = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))["runs"]
    baseline_summaries = {}
    for model in MODELS:
        match = next(
            row for row in baseline_records
            if row.get("cell") == "C29" and row.get("alias") == "baseline"
            and row.get("model") == model and (model == "DM" or row.get("soh_condition") == "lstm_h1")
        )
        baseline_summaries[model] = Path(match["out_dir"]) / "summary.json"

    campaign = ROOT / "campaigns" / args.tag
    manifest_path = campaign / "jes2_manifest.json"
    manifest = {
        "tag": args.tag,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Continuous full-C29 lifecycle bias accumulation with active CV resets",
        "protocol": {
            "cell": "C29", "start_row": 0, "max_rows": 0,
            "bias_directions": ALIASES, "soh_trace": str(args.soh_trace),
            "soh_role": "shared frozen LSTM trace; isolates SOC-estimator response",
        },
        "expected_runs": len(MODELS) * len(args.aliases),
        "runs": [],
    }
    write_json(manifest_path, manifest)

    tasks = []
    for alias in args.aliases:
        offset = ALIASES[alias]
        for model in MODELS:
            out_dir = campaign / "runs" / alias / model
            summary = out_dir / "summary.json"
            command = list(templates[model]["command"])
            replace_option(command, "--out_dir", str(out_dir))
            replace_option(command, "--start_row", "0")
            replace_option(command, "--max_rows", "0")
            if args.temporal_metrics_seconds > 0:
                replace_option(
                    command, "--temporal_metrics_seconds", str(args.temporal_metrics_seconds)
                )
            if model != "DM":
                replace_option(command, "--soh_trace", str(args.soh_trace))
            if alias == "baseline":
                replace_option(command, "--scenario", "baseline")
            else:
                replace_option(command, "--scenario", "current_offset")
                replace_option(command, "--current_offset_pct", str(offset))

            row = {
                "cell": "C29", "alias": alias, "current_offset_pct": offset,
                "model": model, "out_dir": str(out_dir), "command": command, "status": "running",
            }
            if alias == "baseline":
                row["reused_summary"] = str(baseline_summaries[model])
            manifest["runs"].append(row)
            tasks.append((row, summary, out_dir, command, alias, model))
    write_json(manifest_path, manifest)

    def execute(task):
        row, summary, out_dir, command, alias, model = task
        if args.skip_existing and summary.is_file():
            return row, "skipped_existing"
        if alias == "baseline" and not args.force_rerun_baseline:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_summaries[model], summary)
            return row, "reused_pilot_baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=WORKSPACE, check=True)
        return row, "completed"

    progress = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            row, status = future.result()
            row["status"] = status
            progress += 1
            write_json(manifest_path, manifest)
            print(f"LIFECYCLE_BIAS_PROGRESS={progress}/{manifest['expected_runs']}", flush=True)

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
