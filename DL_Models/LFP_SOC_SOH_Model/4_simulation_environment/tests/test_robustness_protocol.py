from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robustness_common import (
    build_online_aux_features,
    compute_common_recovery_metrics,
    compute_protocol_event_metrics,
    compute_stratified_error_metrics,
    load_cell_dataframe,
)


def test_common_recovery_uses_fixed_threshold_and_sustain_time():
    time_s = np.arange(0.0, 1801.0, 60.0)
    y_true = np.zeros(len(time_s))
    y_pred = np.where(time_s < 300.0, 0.10, 0.01)

    metrics = compute_common_recovery_metrics(
        time_s,
        y_true,
        y_pred,
        start_index=0,
        threshold=0.02,
        sustain_seconds=300.0,
        horizon_seconds=1800.0,
    )

    assert metrics["common_recovery_time_s"] == 300.0
    assert metrics["common_recovery_or_censor_time_h"] == 300.0 / 3600.0
    assert metrics["common_recovery_censored"] is False
    assert metrics["common_recovery_threshold_abs_err"] == 0.02


def test_missing_gap_metrics_explain_unobserved_charge():
    time_s = np.arange(0.0, 1201.0, 60.0)
    current = np.ones(len(time_s))
    y_true = np.linspace(1.0, 0.8, len(time_s))
    y_pred = y_true.copy()
    freeze = (time_s >= 300.0) & (time_s <= 600.0)

    metrics = compute_protocol_event_metrics(
        scenario="missing_gap",
        time_s=time_s,
        y_true=y_true,
        y_pred=y_pred,
        current_a=current,
        freeze_mask=freeze,
        threshold=0.02,
        sustain_seconds=120.0,
        horizon_seconds=600.0,
    )

    assert np.isclose(metrics["gap_net_charge_ah"], 6.0 / 60.0)
    assert np.isclose(metrics["gap_throughput_ah"], 6.0 / 60.0)
    assert metrics["gap_reference_soc_change"] < 0.0
    assert metrics["common_recovery_time_s"] == 0.0


def test_censored_recovery_reports_observed_censor_time():
    time_s = np.arange(0.0, 601.0, 60.0)
    metrics = compute_common_recovery_metrics(
        time_s,
        np.zeros(len(time_s)),
        np.full(len(time_s), 0.1),
        start_index=0,
        threshold=0.02,
        sustain_seconds=120.0,
        horizon_seconds=1800.0,
    )

    assert metrics["common_recovery_time_h"] is None
    assert metrics["common_recovery_censored"] is True
    assert metrics["common_recovery_or_censor_time_h"] == 600.0 / 3600.0


def test_online_features_preserve_physical_gap_current():
    frame = pd.DataFrame({
        "Testtime[s]": [0.0, 1.0, 2.0, 3.0],
        "Current[A]": [1.0, 2.0, -3.0, 4.0],
        "Voltage[V]": [3.2, 3.2, 3.2, 3.2],
    })
    freeze = np.array([False, True, True, False])

    result = build_online_aux_features(frame, freeze, 1.0, 3.65, 0.02, 300.0, 1.8)

    assert result["Current[A]"].tolist() == [1.0, 1.0, 1.0, 4.0]
    assert result["_protocol_current_a"].tolist() == [1.0, 2.0, -3.0, 4.0]


def test_stratified_metrics_cover_fixed_physical_states():
    y_true = np.array([0.1, 0.5, 0.9, 0.6])
    y_pred = y_true + np.array([0.01, -0.02, 0.03, -0.04])
    rows = compute_stratified_error_metrics(
        y_true,
        y_pred,
        reference_soh=np.array([0.95, 0.85, 0.75, 0.75]),
        reference_temperature_c=np.array([28.0, 32.0, 38.0, 38.0]),
        reference_c_rate=np.array([0.2, 1.0, 2.0, 2.0]),
    )
    indexed = {(row["dimension"], row["stratum"]): row for row in rows}

    assert indexed[("soh_state", "aged")]["n_samples"] == 2
    assert indexed[("temperature_state", "hot")]["n_samples"] == 2
    assert indexed[("instantaneous_load", "high")]["n_samples"] == 2
    assert np.isclose(indexed[("soh_state", "aged")]["mae"], 0.035)


def test_parquet_window_loader_matches_full_dataframe_slice(tmp_path):
    source = pd.DataFrame({"row": np.arange(25), "value": np.linspace(0.0, 1.0, 25)})
    source.to_parquet(tmp_path / "df_FE_C09.parquet", row_group_size=7)

    loaded = load_cell_dataframe(str(tmp_path), "C09", start_row=6, max_rows=13)

    pd.testing.assert_frame_equal(loaded, source.iloc[6:19].reset_index(drop=True))
