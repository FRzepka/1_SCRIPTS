import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from merge_jes2_manifests import merge_manifests


def _manifest(cell: str, window_id: str, out_dir: str) -> dict:
    return {
        "benchmark_version": "JES2.0",
        "split": {"holdout": ["C09", "C13"]},
        "artifacts": {},
        "base_seed": 42,
        "stochastic_repeats": 10,
        "secondary_stochastic_repeats": 5,
        "models": ["DD"],
        "soh_modes": ["lstm"],
        "lstm_publish_intervals": [1],
        "reference_publish_intervals": [1],
        "output_policy": "summary_only",
        "cells": [cell],
        "window": {
            "manifest": "/frozen/windows.csv",
            "soh_context_rows": 691200,
            "definitions": [{"cell": cell, "window_id": window_id, "start_row": 10}],
        },
        "protocol": {"scenarios": [{"alias": "baseline"}]},
        "runs": [{
            "cell": cell,
            "window_id": window_id,
            "alias": "baseline",
            "seed": 42,
            "soh_condition": "lstm_h1",
            "model": "DD",
            "out_dir": out_dir,
            "status": "completed",
        }],
    }


def test_merge_preserves_distinct_window_runs_and_definitions(tmp_path):
    paths = []
    for index, manifest in enumerate([
        _manifest("C09", "C09_fresh", "/runs/fresh"),
        _manifest("C13", "C13_aged", "/runs/aged"),
    ]):
        path = tmp_path / f"manifest_{index}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)

    merged = merge_manifests(paths, "merged")

    assert len(merged["runs"]) == 2
    assert {row["window_id"] for row in merged["runs"]} == {"C09_fresh", "C13_aged"}
    assert len(merged["window"]["definitions"]) == 2
