from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from robustness_common import COMMON_EVALUATION_START_SAMPLE


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
LEVELS = {
    "current_bias_0p5pct": ("current_bias_neg_0p5pct", -0.005),
    "current_bias_1p5pct": ("current_bias_neg_1p5pct", -0.015),
    "current_bias_3p0pct": ("current_bias_neg_3p0pct", -0.030),
}


def replace_option(command: list[str], option: str, value: str) -> None:
    command[command.index(option) + 1] = value


def run(command: list[str], out_dir: Path) -> tuple[str, str | None]:
    if (out_dir / "summary.json").is_file():
        return "skipped_existing", None
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "rerun.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=WORKSPACE, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode or not (out_dir / "summary.json").is_file():
        return "failed", str(log)
    return "completed", None


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct the supplemental signed current-gain sweep mask.")
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--signed_manifest", type=Path, required=True)
    parser.add_argument("--tag", default="jes2_signed_gain_common_mask_20260830")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--evaluation_start_sample", type=int, default=COMMON_EVALUATION_START_SAMPLE)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    signed = json.loads(args.signed_manifest.read_text(encoding="utf-8"))
    signed_by_key = {
        (row["cell"], row["window_id"], row["alias"], row["model"]): row
        for row in signed["runs"]
    }
    selected = []
    seen = set()
    for record in source["runs"]:
        if record["alias"] not in LEVELS:
            continue
        condition = record.get("soh_condition", "none")
        if condition != ("none" if record["model"] == "DM" else "lstm_h1"):
            continue
        key = (record["cell"], record["window_id"], record["alias"], record["model"])
        if key not in seen:
            selected.append(record)
            seen.add(key)

    campaign = ROOT / "campaigns" / args.tag
    campaign.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign / "jes2_manifest.json"
    output = {
        "tag": args.tag,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_signed_manifest": str(args.signed_manifest.resolve()),
        "protocol": {
            "quantity": "multiplicative current-gain error",
            "common_evaluation_start_sample": int(args.evaluation_start_sample),
        },
        "runs": [],
    }
    jobs = []
    for record in selected:
        negative_alias, offset = LEVELS[record["alias"]]
        old = signed_by_key[(record["cell"], record["window_id"], negative_alias, record["model"])]
        if record["model"] == "DD":
            new_record = dict(old)
            new_record["status"] = "reused_common_interval"
            output["runs"].append(new_record)
            continue
        out_dir = campaign / "runs" / record["cell"] / record["window_id"] / negative_alias / record["model"]
        command = list(record["command"])
        replace_option(command, "--out_dir", str(out_dir))
        replace_option(command, "--current_offset_pct", str(offset))
        if record["model"] != "DM":
            old_trace = (
                args.signed_manifest.parent / "traces" / record["cell"] / record["window_id"]
                / f"{negative_alias}_lstm_h1.npz"
            )
            replace_option(command, "--soh_trace", str(old_trace))
        command.extend(["--evaluation_start_sample", str(args.evaluation_start_sample)])
        new_record = dict(old)
        new_record.update({"out_dir": str(out_dir), "command": command, "status": "pending"})
        output["runs"].append(new_record)
        jobs.append((len(output["runs"]) - 1, command, out_dir))

    manifest_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run, command, out_dir): index for index, command, out_dir in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            status, error = future.result()
            output["runs"][index]["status"] = status
            if error:
                output["runs"][index]["error_log"] = error
                failures.append(error)
            if completed % 12 == 0 or completed == len(jobs):
                output["progress"] = {"completed": completed, "total": len(jobs)}
                manifest_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
                print(f"SIGNED_GAIN_COMMON_MASK_PROGRESS={completed}/{len(jobs)}", flush=True)

    output["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output["failures"] = failures
    manifest_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} signed-gain correction runs failed")


if __name__ == "__main__":
    main()
