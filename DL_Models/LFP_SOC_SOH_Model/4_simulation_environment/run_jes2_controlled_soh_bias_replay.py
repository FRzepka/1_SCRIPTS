from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
MODELS = ("HDM", "HECM", "DD")
LEVELS = {"fresh": 0.962, "mid_life": 0.897, "aged": 0.831}
ALIASES = ("baseline", "positive_3pct", "negative_3pct")


def replace_option(command: list[str], name: str, value: str) -> None:
    if name in command:
        command[command.index(name) + 1] = value
    else:
        command.extend([name, value])


def remove_flag(command: list[str], name: str) -> None:
    while name in command:
        command.remove(name)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one C29 trace at controlled constant SOH levels.")
    parser.add_argument(
        "--source-manifest", type=Path,
        default=ROOT / "campaigns/jes2_full_holdout_merged_20260825.json",
    )
    parser.add_argument("--tag", default="jes2_controlled_soh_bias_C29_20260828")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.source_manifest.read_text(encoding="utf-8"))["runs"]
    index = {}
    for record in records:
        if record.get("cell") != "C29" or record.get("window_id") != "C29_mid_life":
            continue
        if record.get("model") not in MODELS or record.get("soh_condition") != "lstm_h1" or int(record.get("seed", -1)) != 42:
            continue
        if record.get("alias") in {"baseline", "current_bias_3p0pct"}:
            index[(record["alias"], record["model"])] = record

    campaign = ROOT / "campaigns" / args.tag
    manifest_path = campaign / "jes2_manifest.json"
    manifest = {
        "tag": args.tag,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "controlled SOH factorial replay with fixed C29 mid-life measurements",
        "soh_levels": LEVELS,
        "expected_runs": len(LEVELS) * len(ALIASES) * len(MODELS),
        "runs": [],
    }
    write_json(manifest_path, manifest)
    progress = 0
    for level, soh_value in LEVELS.items():
        for alias in ALIASES:
            source_alias = "baseline" if alias == "baseline" else "current_bias_3p0pct"
            for model in MODELS:
                record = index[(source_alias, model)]
                out_dir = campaign / "runs" / level / alias / model
                summary = out_dir / "summary.json"
                source_trace = Path(record["soh_trace"])
                controlled_trace = campaign / "traces" / level / f"{alias}.npz"
                if not controlled_trace.is_file():
                    source = np.load(source_trace, allow_pickle=False)
                    metadata = json.loads(str(source["metadata_json"]))
                    metadata.update({
                        "source": "controlled_constant_soh_replay",
                        "controlled_soh": soh_value,
                        "measurement_window": "C29_mid_life",
                        "measurement_alias": alias,
                    })
                    controlled_trace.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        controlled_trace,
                        time_s=source["time_s"],
                        soh=np.full(source["soh"].shape, soh_value, dtype=np.float32),
                        metadata_json=json.dumps(metadata),
                    )
                command = list(record["command"])
                remove_flag(command, "--summary_only")
                replace_option(command, "--out_dir", str(out_dir))
                replace_option(command, "--soh_trace", str(controlled_trace))
                if alias == "negative_3pct":
                    replace_option(command, "--current_offset_pct", "-0.03")
                row = {
                    "soh_level": level, "soh_value": soh_value, "alias": alias,
                    "model": model, "out_dir": str(out_dir), "command": command, "status": "running",
                }
                manifest["runs"].append(row)
                write_json(manifest_path, manifest)
                if args.skip_existing and summary.is_file():
                    row["status"] = "skipped_existing"
                else:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    subprocess.run(command, cwd=WORKSPACE, check=True)
                    row["status"] = "completed"
                progress += 1
                write_json(manifest_path, manifest)
                print(f"CONTROLLED_SOH_PROGRESS={progress}/{manifest['expected_runs']}", flush=True)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
