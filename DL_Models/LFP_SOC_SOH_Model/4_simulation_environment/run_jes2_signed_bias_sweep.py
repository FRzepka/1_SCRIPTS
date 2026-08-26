from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
LEVELS = {
    "current_bias_0p5pct": ("current_bias_neg_0p5pct", -0.005),
    "current_bias_1p5pct": ("current_bias_neg_1p5pct", -0.015),
    "current_bias_3p0pct": ("current_bias_neg_3p0pct", -0.030),
}
MODEL_ORDER = ("DM", "HDM", "HECM", "DD")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def replace_option(command: list[str], name: str, value: str) -> None:
    command[command.index(name) + 1] = value


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def execute(command: list[str]) -> None:
    subprocess.run(command, cwd=WORKSPACE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supplemental symmetric JES2 current-bias sweep.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--tag", default="jes2_signed_bias_20260826")
    parser.add_argument("--trace-device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    campaign = ROOT / "campaigns" / args.tag
    manifest_path = campaign / "jes2_manifest.json"
    source_records = []
    seen = set()
    for record in source["runs"]:
        if record["alias"] not in LEVELS or record["model"] not in MODEL_ORDER:
            continue
        if record["model"] != "DM" and record.get("soh_condition") != "lstm_h1":
            continue
        key = (record["cell"], record["window_id"], record["alias"], record["model"])
        if key in seen:
            continue
        seen.add(key)
        source_records.append(record)

    source_records.sort(key=lambda row: (
        row["cell"], row["window_id"], list(LEVELS).index(row["alias"]), MODEL_ORDER.index(row["model"])
    ))
    manifest = {
        "tag": args.tag,
        "started_utc": utc_now(),
        "source_manifest": str(args.source_manifest.resolve()),
        "purpose": "supplemental symmetric signed current-bias analysis",
        "protocol": {
            "aliases": [value[0] for value in LEVELS.values()],
            "offset_fractions": [value[1] for value in LEVELS.values()],
            "primary_soh_condition": "lstm_h1",
        },
        "expected_runs": len(source_records),
        "runs": [],
    }
    write_manifest(manifest_path, manifest)

    traces: dict[tuple[str, str, str], Path] = {}
    for index, record in enumerate(source_records, start=1):
        negative_alias, offset = LEVELS[record["alias"]]
        target = campaign / "runs" / record["cell"] / record["window_id"] / negative_alias / record["model"]
        summary = target / "summary.json"
        trace_path = campaign / "traces" / record["cell"] / record["window_id"] / f"{negative_alias}_lstm_h1.npz"

        if record["model"] != "DM" and not trace_path.is_file():
            command = record["command"]
            trace_command = [
                sys.executable, str(ROOT / "shared_soh_trace.py"),
                "--mode", "lstm",
                "--cell", record["cell"],
                "--out", str(trace_path),
                "--data_root", option(command, "--data_root"),
                "--soh_config", option(command, "--soh_config"),
                "--soh_ckpt", option(command, "--soh_ckpt"),
                "--soh_scaler", option(command, "--soh_scaler"),
                "--publish_every_intervals", "1",
                "--context_rows", "691200",
                "--device", args.trace_device,
                "--scenario", "current_offset",
                "--seed", str(record["seed"]),
                "--start_row", str(record["start_row"]),
                "--max_rows", str(record["max_rows"]),
                "--current_offset_pct", str(offset),
            ]
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            execute(trace_command)

        new_record = {
            key: record.get(key) for key in
            ["cell", "window_id", "soh_state", "cell_load_class", "start_row", "max_rows", "seed", "model"]
        }
        new_record.update({
            "alias": negative_alias,
            "scenario": "current_offset",
            "current_offset_pct": offset,
            "soh_mode": "none" if record["model"] == "DM" else "lstm",
            "soh_condition": "none" if record["model"] == "DM" else "lstm_h1",
            "out_dir": str(target),
            "status": "running",
        })
        manifest["runs"].append(new_record)
        write_manifest(manifest_path, manifest)

        if args.skip_existing and summary.is_file():
            new_record["status"] = "skipped_existing"
        else:
            command = list(record["command"])
            replace_option(command, "--out_dir", str(target))
            replace_option(command, "--current_offset_pct", str(offset))
            if record["model"] != "DM":
                replace_option(command, "--soh_trace", str(trace_path))
            target.mkdir(parents=True, exist_ok=True)
            execute(command)
            new_record["status"] = "completed"
        write_manifest(manifest_path, manifest)
        print(f"SIGNED_BIAS_PROGRESS={index}/{len(source_records)}", flush=True)

    manifest["finished_utc"] = utc_now()
    write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "runs": len(manifest["runs"])}, indent=2))


if __name__ == "__main__":
    main()
