from __future__ import annotations

import argparse
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from robustness_common import COMMON_EVALUATION_START_SAMPLE


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
WRITE_LOCK = threading.Lock()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def corrected_command(record: dict, out_dir: Path, evaluation_start_sample: int) -> list[str]:
    command = list(record["command"])
    out_index = command.index("--out_dir") + 1
    command[out_index] = str(out_dir)
    if "--evaluation_start_sample" in command:
        value_index = command.index("--evaluation_start_sample") + 1
        command[value_index] = str(evaluation_start_sample)
    else:
        command.extend(["--evaluation_start_sample", str(evaluation_start_sample)])
    return command


def relative_run_path(record: dict) -> Path:
    return Path(
        "runs",
        str(record["cell"]),
        str(record.get("window_id", "single_window")),
        str(record["alias"]),
        f"seed_{int(record['seed'])}",
        str(record.get("soh_condition", "none")),
        str(record["model"]),
    )


def run_record(index: int, record: dict, campaign_dir: Path, evaluation_start_sample: int) -> tuple[int, str, str | None]:
    out_dir = campaign_dir / relative_run_path(record)
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        return index, "skipped_existing", None
    out_dir.mkdir(parents=True, exist_ok=True)
    command = corrected_command(record, out_dir, evaluation_start_sample)
    log_path = out_dir / "rerun.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not summary_path.is_file():
        return index, "failed", str(log_path)
    return index, "completed", None


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute JES2 non-DD summaries on the DD-valid common mask.")
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--tag", default="jes2_common_mask_20260830")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--evaluation_start_sample", type=int, default=COMMON_EVALUATION_START_SAMPLE)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    campaign_dir = ROOT / "campaigns" / args.tag
    manifest_path = campaign_dir / "jes2_manifest.json"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    manifest = deepcopy(source)
    manifest["tag"] = args.tag
    manifest["source_manifest"] = str(args.source_manifest.resolve())
    manifest["correction"] = {
        "reason": "common DD-valid evaluation interval",
        "evaluation_start_sample": int(args.evaluation_start_sample),
        "dd_policy": "reuse original rolling-window summaries already starting at source sample 2023",
        "non_dd_policy": "rerun from stored commands and shared SOH traces",
    }
    manifest.setdefault("protocol", {})["common_evaluation_start_sample"] = int(args.evaluation_start_sample)
    manifest["started_utc"] = datetime.now(timezone.utc).isoformat()

    jobs: list[tuple[int, dict]] = []
    for index, source_record in enumerate(source["runs"]):
        record = manifest["runs"][index]
        if source_record["model"] == "DD":
            record["status"] = "reused_common_interval"
            record["source_summary"] = str(Path(source_record["out_dir"]) / "summary.json")
            continue
        out_dir = campaign_dir / relative_run_path(source_record)
        record["out_dir"] = str(out_dir)
        record["command"] = corrected_command(source_record, out_dir, args.evaluation_start_sample)
        record["status"] = "pending"
        jobs.append((index, source_record))
    write_json(manifest_path, manifest)

    completed = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_record, index, record, campaign_dir, args.evaluation_start_sample): index
            for index, record in jobs
        }
        for future in as_completed(futures):
            index, status, error = future.result()
            with WRITE_LOCK:
                manifest["runs"][index]["status"] = status
                if error:
                    manifest["runs"][index]["error_log"] = error
                    failures.append(error)
                completed += 1
                if completed % 25 == 0 or completed == len(jobs):
                    manifest["progress"] = {"completed": completed, "total": len(jobs)}
                    write_json(manifest_path, manifest)
                    print(f"COMMON_MASK_PROGRESS={completed}/{len(jobs)}", flush=True)

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failures"] = failures
    write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"{len(failures)} corrected runs failed")
    print(json.dumps({"manifest": str(manifest_path), "corrected_runs": len(jobs)}, indent=2))


if __name__ == "__main__":
    main()
