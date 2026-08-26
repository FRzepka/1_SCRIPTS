from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPATIBILITY_FIELDS = [
    "benchmark_version",
    "split",
    "artifacts",
    "base_seed",
    "stochastic_repeats",
    "secondary_stochastic_repeats",
    "models",
    "soh_modes",
    "lstm_publish_intervals",
    "reference_publish_intervals",
    "output_policy",
]
STATUS_RANK = {
    "running": 0,
    "dry_run": 1,
    "not_applicable": 2,
    "completed": 3,
    "skipped_existing": 3,
}


def canonical_cell(cell: str) -> str:
    return str(cell).rsplit("_", 1)[-1]


def run_key(record: dict[str, Any]) -> tuple:
    return (
        canonical_cell(record["cell"]),
        record.get("window_id", "single_window"),
        record["alias"],
        int(record["seed"]),
        record.get("soh_condition", "none"),
        record["model"],
    )


def merge_manifests(paths: list[Path], tag: str) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one JES2 manifest is required")
    resolved = [path.resolve() for path in paths]
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in resolved]
    reference = manifests[0]
    for path, manifest in zip(resolved[1:], manifests[1:]):
        for field in COMPATIBILITY_FIELDS:
            if manifest.get(field) != reference.get(field):
                raise ValueError(f"Incompatible field {field!r} in {path}")
        reference_window = {
            key: value for key, value in reference.get("window", {}).items()
            if key != "definitions"
        }
        candidate_window = {
            key: value for key, value in manifest.get("window", {}).items()
            if key != "definitions"
        }
        if candidate_window != reference_window:
            raise ValueError(f"Incompatible field 'window' in {path}")

    merged = copy.deepcopy(reference)
    merged["tag"] = tag
    merged["merged_utc"] = datetime.now(timezone.utc).isoformat()
    merged["source_manifests"] = [str(path) for path in resolved]
    merged["cells"] = sorted(
        {canonical_cell(cell) for manifest in manifests for cell in manifest.get("cells", [])}
    )
    merged["reference_aliases"] = sorted(
        {alias for manifest in manifests for alias in manifest.get("reference_aliases", [])}
    )
    merged["cadence_aliases"] = sorted(
        {alias for manifest in manifests for alias in manifest.get("cadence_aliases", ["baseline"])}
    )
    window_definitions = {}
    for manifest in manifests:
        for row in manifest.get("window", {}).get("definitions", []):
            key = (canonical_cell(row["cell"]), str(row["window_id"]))
            normalized = copy.deepcopy(row)
            normalized["cell"] = key[0]
            if key in window_definitions and window_definitions[key] != normalized:
                raise ValueError(f"Conflicting window definition {key}")
            window_definitions[key] = normalized
    merged.setdefault("window", {})["definitions"] = [
        window_definitions[key] for key in sorted(window_definitions)
    ]

    scenario_by_alias = {}
    for manifest in manifests:
        for scenario in manifest.get("protocol", {}).get("scenarios", []):
            alias = scenario["alias"]
            if alias in scenario_by_alias and scenario_by_alias[alias] != scenario:
                raise ValueError(f"Conflicting protocol definition for alias {alias!r}")
            scenario_by_alias[alias] = scenario
    merged["protocol"]["scenarios"] = list(scenario_by_alias.values())

    records = {}
    for manifest in manifests:
        for record in manifest.get("runs", []):
            key = run_key(record)
            current = records.get(key)
            if current is None or STATUS_RANK.get(record.get("status"), -1) > STATUS_RANK.get(current.get("status"), -1):
                records[key] = record
            elif (
                STATUS_RANK.get(record.get("status"), -1) == STATUS_RANK.get(current.get("status"), -1)
                and record.get("out_dir") != current.get("out_dir")
            ):
                raise ValueError(f"Conflicting duplicate run {key}")
    merged["runs"] = [records[key] for key in sorted(records)]
    merged["finished_utc"] = max(
        (manifest.get("finished_utc", manifest.get("started_utc", "")) for manifest in manifests),
        default="",
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge compatible JES2 cell/alias campaign manifests.")
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default="jes2_merged")
    args = parser.parse_args()

    merged = merge_manifests(args.manifests, args.tag)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.out.resolve()), "cells": merged["cells"], "runs": len(merged["runs"])}, indent=2))


if __name__ == "__main__":
    main()
