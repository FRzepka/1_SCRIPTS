from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_jes2_hecm_full_lookup_sensitivity import (
    LOOKUP_CONDITIONS,
    LOOKUP_LABELS,
    build_command,
    cell_bootstrap,
    execute_record,
)


ROOT = Path(__file__).resolve().parent
ALIASES = {"current_offset_neg_50mA", "current_offset_pos_50mA"}
EXPECTED_CELLS = {"C09", "C13", "C15", "C25", "C27", "C29"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_sources(manifests: list[Path]) -> list[dict]:
    sources = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.extend(
            row
            for row in payload["runs"]
            if row.get("model") == "HECM"
            and row.get("soh_condition") == "lstm_h1"
            and row.get("alias") in ALIASES
        )
    keys = [(row["cell"], row["window_id"], row["alias"]) for row in sources]
    if len(sources) != 32 or len(keys) != len(set(keys)):
        raise ValueError(f"Expected 32 unique HECM current-offset sources, found {len(sources)}")
    if {row["cell"] for row in sources} != EXPECTED_CELLS:
        raise ValueError("HECM current-offset source cells are incomplete")
    return sources


def build_records(
    sources: list[dict], out_root: Path, evaluation_start_sample: int, device: str
) -> list[dict]:
    records = []
    for lookup_condition, lookup_parameters in LOOKUP_CONDITIONS.items():
        for source in sources:
            out_dir = (
                out_root
                / "runs"
                / lookup_condition
                / source["cell"]
                / source["window_id"]
                / source["alias"]
            )
            records.append(
                {
                    "cell": source["cell"],
                    "window_id": source["window_id"],
                    "soh_state": source["soh_state"],
                    "cell_load_class": source["cell_load_class"],
                    "alias": source["alias"],
                    "lookup_condition": lookup_condition,
                    "lookup_parameters": lookup_parameters,
                    "out_dir": str(out_dir),
                    "command": build_command(
                        source,
                        out_dir,
                        lookup_parameters,
                        evaluation_start_sample,
                        device,
                    ),
                    "status": "pending",
                }
            )
    if len(records) != 224:
        raise ValueError(f"Expected 224 HECM lookup-offset runs, built {len(records)}")
    return records


def load_analysis_rows(manifest: dict, baseline_root: Path) -> pd.DataFrame:
    rows = []
    for record in manifest["runs"]:
        summary_path = Path(record["out_dir"]) / "summary.json"
        baseline_path = (
            baseline_root
            / "runs"
            / record["lookup_condition"]
            / record["cell"]
            / record["window_id"]
            / "baseline"
            / "seed_42"
            / "summary.json"
        )
        if not summary_path.is_file() or not baseline_path.is_file():
            raise FileNotFoundError(f"Missing offset or matched baseline: {summary_path}, {baseline_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        offset_a = float(summary["scenario_meta"]["current_offset_a"])
        rows.append(
            {
                "lookup_condition": record["lookup_condition"],
                "lookup_label": LOOKUP_LABELS[record["lookup_condition"]],
                "cell": record["cell"],
                "window_id": record["window_id"],
                "alias": record["alias"],
                "current_offset_a": offset_a,
                "mae": float(summary["mae"]),
                "baseline_mae": float(baseline["mae"]),
                "delta_mae": float(summary["mae"] - baseline["mae"]),
                "evaluation_start_sample": int(summary["evaluation_start_sample"]),
                "evaluation_samples": int(summary["evaluation_samples"]),
                "summary_path": str(summary_path.resolve()),
                "baseline_summary_path": str(baseline_path.resolve()),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 224 or set(frame["evaluation_start_sample"]) != {2023}:
        raise ValueError("HECM lookup-offset analysis coverage or common mask is invalid")
    if set(frame["evaluation_samples"]) != {84377}:
        raise ValueError("HECM lookup-offset sample counts are not matched")
    return frame


def analyze(manifest_path: Path, baseline_root: Path, out_dir: Path, samples: int) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = load_analysis_rows(manifest, baseline_root)
    cells = (
        runs.groupby(
            ["lookup_condition", "lookup_label", "cell", "alias", "current_offset_a"],
            as_index=False,
        )[["mae", "baseline_mae", "delta_mae"]]
        .mean()
    )
    nominal = cells[cells["lookup_condition"].eq("nominal_lookup")][
        ["cell", "alias", "delta_mae"]
    ].rename(columns={"delta_mae": "nominal_delta_mae"})
    cells = cells.merge(nominal, on=["cell", "alias"], validate="many_to_one")
    cells["interaction_mae"] = cells["delta_mae"] - cells["nominal_delta_mae"]

    statistics = []
    for index, ((lookup, alias), group) in enumerate(
        cells.groupby(["lookup_condition", "alias"], sort=False)
    ):
        interaction = cell_bootstrap(group["interaction_mae"], samples, 27100 + index)
        delta = cell_bootstrap(group["delta_mae"], samples, 27200 + index)
        statistics.append(
            {
                "lookup_condition": lookup,
                "lookup_label": LOOKUP_LABELS[lookup],
                "alias": alias,
                "current_offset_a": float(group["current_offset_a"].iloc[0]),
                "n_cells": len(group),
                "delta_mae": delta[0],
                "delta_mae_ci_low": delta[1],
                "delta_mae_ci_high": delta[2],
                "interaction_mae": interaction[0],
                "interaction_ci_low": interaction[1],
                "interaction_ci_high": interaction[2],
                "interaction_ci_includes_zero": interaction[1] <= 0.0 <= interaction[2],
            }
        )
    statistics_frame = pd.DataFrame(statistics)
    perturbed = statistics_frame[~statistics_frame["lookup_condition"].eq("nominal_lookup")]
    max_interaction = float(perturbed["interaction_mae"].abs().max())

    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "hecm_lookup_current_offset_runs.csv", index=False)
    cells.to_csv(out_dir / "hecm_lookup_current_offset_cells.csv", index=False)
    statistics_frame.to_csv(out_dir / "hecm_lookup_current_offset_statistics.csv", index=False)
    protocol = {
        "analysis": "HECM lookup sensitivity for the signed additive current-offset extension",
        "runs": len(runs),
        "lookup_conditions": LOOKUP_CONDITIONS,
        "current_offset_a": [-0.05, 0.05],
        "cells": sorted(runs["cell"].unique()),
        "windows": sorted(runs["window_id"].unique()),
        "evaluation_start_sample": 2023,
        "aggregation": f"windows averaged within cell; equal-weight cell macro; {samples}-draw cell bootstrap",
        "interaction_definition": "lookup-specific current-offset delta MAE minus nominal-lookup current-offset delta MAE",
        "maximum_absolute_interaction_mae": max_interaction,
    }
    (out_dir / "hecm_lookup_current_offset_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    return {"runs": len(runs), "maximum_absolute_interaction_mae": max_interaction}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend the HECM lookup sensitivity with signed additive current offset."
    )
    parser.add_argument("--offset_manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline_lookup_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--evaluation_start_sample", type=int, default=2023)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    args.offset_manifests = [path.resolve() for path in args.offset_manifests]
    args.baseline_lookup_root = args.baseline_lookup_root.resolve()
    args.out_root = args.out_root.resolve()
    args.result_dir = args.result_dir.resolve()

    manifest_path = args.out_root / "manifest.json"
    if not args.analyze_only:
        sources = load_sources(args.offset_manifests)
        records = build_records(sources, args.out_root, args.evaluation_start_sample, args.device)
        payload = {
            "analysis": "HECM lookup sensitivity for signed additive current offset",
            "started_utc": utc_now(),
            "source_manifests": [str(path.resolve()) for path in args.offset_manifests],
            "baseline_lookup_root": str(args.baseline_lookup_root.resolve()),
            "evaluation_start_sample": args.evaluation_start_sample,
            "lookup_conditions": LOOKUP_CONDITIONS,
            "runs": records,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(execute_record, record, args.skip_existing): record
                for record in records
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                record = futures[future]
                status, error = future.result()
                record["status"] = status
                if error:
                    record["error"] = error
                if completed % 16 == 0 or completed == len(records) or error:
                    payload["progress"] = {"completed": completed, "total": len(records)}
                    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    print(f"HECM_LOOKUP_CURRENT_OFFSET_PROGRESS={completed}/{len(records)}", flush=True)
        failures = [row for row in records if row["status"] == "failed"]
        payload["finished_utc"] = utc_now()
        payload["failures"] = len(failures)
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if failures:
            raise SystemExit(f"{len(failures)} HECM lookup current-offset runs failed")

    result = analyze(
        manifest_path,
        args.baseline_lookup_root,
        args.result_dir,
        args.bootstrap_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
