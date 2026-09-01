from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_jes2_hecm_parameter_sensitivity import (
    TABLE_CONDITIONS,
    build_compact_statistics,
    write_compact_latex_table,
)


def _statistics() -> pd.DataFrame:
    rows = []
    for index, table_condition in enumerate(TABLE_CONDITIONS):
        values = {
            "baseline_mae": (0.03 + index * 0.001, 0.02, 0.04),
            "adverse_gain_delta_mae": (0.006, 0.004, 0.008),
            "gain_interaction_delta_delta_mae": (
                0.0 if index == 0 else index * 0.00005,
                0.0 if index == 0 else -0.0004,
                0.0 if index == 0 else 0.0005,
            ),
        }
        for metric, (mean, ci_low, ci_high) in values.items():
            rows.append(
                {
                    "table_condition": table_condition,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def test_compact_lookup_statistics_retain_scope_and_interval_result(tmp_path):
    compact = build_compact_statistics(_statistics())

    assert compact["table_condition"].tolist() == list(TABLE_CONDITIONS)
    assert compact["interaction_ci_includes_zero"].all()

    table_path = tmp_path / "lookup_table.tex"
    write_compact_latex_table(compact, table_path)
    table = table_path.read_text(encoding="utf-8")

    assert "Baseline MAE" in table
    assert "Gain $\\Delta$MAE" in table
    assert "All perturbed-lookup 95\\% intervals include zero" not in table
    assert "all 95\\% intervals include zero" in table
