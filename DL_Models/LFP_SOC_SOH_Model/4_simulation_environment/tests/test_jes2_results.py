from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))

from build_jes2_paper_results import (
    add_baseline_deltas,
    build_paired_statistical_tests,
    exact_sign_flip_pvalue,
    hierarchical_stats,
    validate_coverage,
)
from jes2_protocol import INITIAL_STATE_APPLICABLE_MODELS


def _manifest():
    return {
        "cells": ["C29"],
        "split": {"holdout": ["MGFarm_18650_C29"]},
        "base_seed": 42,
        "stochastic_repeats": 20,
        "soh_modes": ["lstm"],
        "lstm_publish_intervals": [1],
        "reference_publish_intervals": [1],
        "cadence_aliases": ["baseline"],
        "protocol": {"scenarios": [{"alias": "baseline"}]},
    }


def _complete_frame():
    rows = [
        {"cell": "C29", "alias": "baseline", "seed": 42, "soh_condition": "none", "model": "DM"},
    ]
    rows.extend(
        {"cell": "C29", "alias": "baseline", "seed": 42, "soh_condition": "lstm_h1", "model": model}
        for model in ["HDM", "HECM", "DD"]
    )
    return pd.DataFrame(rows)


def test_coverage_accepts_normalized_complete_holdout_campaign():
    validate_coverage(_manifest(), _complete_frame(), allow_incomplete=False)


def test_coverage_rejects_missing_model_run():
    frame = _complete_frame()
    frame = frame[frame["model"] != "DD"]
    with pytest.raises(ValueError, match="expected runs are missing"):
        validate_coverage(_manifest(), frame, allow_incomplete=False)


def test_coverage_allows_incomplete_diagnostic_campaign():
    validate_coverage(_manifest(), _complete_frame().head(1), allow_incomplete=True)


def test_dd_is_required_for_new_initialization_campaigns():
    manifest = _manifest()
    manifest["protocol"] = {
        "initial_state_comparison_models": sorted(INITIAL_STATE_APPLICABLE_MODELS),
        "scenarios": [{"alias": "initial_soc_error"}],
    }
    frame = pd.DataFrame([
        {"cell": "C29", "alias": "initial_soc_error", "seed": 42,
         "soh_condition": "none", "model": "DM"},
        *[
            {"cell": "C29", "alias": "initial_soc_error", "seed": 42,
             "soh_condition": "lstm_h1", "model": model}
            for model in ["HDM", "HECM", "DD"]
        ],
    ])
    validate_coverage(manifest, frame, allow_incomplete=False)
    with pytest.raises(ValueError, match="expected runs are missing"):
        validate_coverage(manifest, frame[frame["model"] != "DD"], allow_incomplete=False)


def test_initialization_protocol_includes_dd_qc_mapping():
    assert INITIAL_STATE_APPLICABLE_MODELS == {"DM", "HDM", "HECM", "DD"}


def test_exact_sign_flip_test_uses_cells_not_samples():
    assert exact_sign_flip_pvalue(pd.Series([1.0] * 6).to_numpy()) == 2.0 / 64.0


def test_paired_statistics_report_effects_and_holm_adjustment():
    rows = []
    for cell_index in range(6):
        cell = f"C{cell_index:02d}"
        for model_index, model in enumerate(["DM", "HDM", "HECM", "DD"]):
            condition = "none" if model == "DM" else "lstm_h1"
            mode = "none" if model == "DM" else "lstm"
            baseline = 0.01 + model_index * 0.002 + cell_index * 0.0001
            rows.append({"cell": cell, "alias": "baseline", "seed": 42, "model": model,
                         "soh_condition": condition, "soh_mode": mode, "mae": baseline, "rmse": baseline})
            for seed in [42, 43, 44]:
                rows.append({"cell": cell, "alias": "current_noise_high", "seed": seed, "model": model,
                             "soh_condition": condition, "soh_mode": mode,
                             "mae": baseline + 0.003 + seed * 1e-7, "rmse": baseline + 0.004})
    frame = add_baseline_deltas(pd.DataFrame(rows))
    scenario, pairs = build_paired_statistical_tests(frame, bootstrap_samples=100)

    assert len(scenario) == 4
    assert len(pairs) == 12
    assert scenario["n_cells"].eq(6).all()
    assert scenario["p_holm"].notna().all()


def test_windows_are_averaged_within_cell_seed_before_bootstrap():
    frame = pd.DataFrame([
        {"cell": "C09", "seed": 42, "window_id": "fresh", "mae": 0.01},
        {"cell": "C09", "seed": 42, "window_id": "aged", "mae": 0.05},
        {"cell": "C13", "seed": 42, "window_id": "fresh", "mae": 0.02},
    ])

    stats = hierarchical_stats(frame, "mae", bootstrap_samples=0, seed=42)

    assert stats["n_cells"] == 2
    assert stats["n_runs"] == 2
    assert stats["mean"] == pytest.approx((0.03 + 0.02) / 2.0)
