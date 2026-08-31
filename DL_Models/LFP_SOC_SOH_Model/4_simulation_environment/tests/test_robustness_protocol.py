from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robustness_common import (
    build_common_evaluation_mask,
    build_online_aux_features,
    compute_max_abs_net_charge_window_mask,
    compute_common_recovery_metrics,
    compute_protocol_event_metrics,
    compute_stratified_error_metrics,
    load_cell_dataframe,
    write_temporal_error_metrics,
)


def test_common_evaluation_mask_aligns_full_and_windowed_outputs():
    full = build_common_evaluation_mask(86_400, 2023)
    rolling = build_common_evaluation_mask(84_377, 2023, source_start_sample=2023)

    assert int(full.sum()) == 84_377
    assert int(rolling.sum()) == 84_377
    assert np.flatnonzero(full)[0] == 2023
    assert np.flatnonzero(rolling)[0] == 0


def test_temporal_metrics_preserve_binned_mae_and_full_charge_events(tmp_path):
    time_s = np.arange(0.0, 901.0, 60.0)
    y_true = np.zeros(len(time_s))
    y_pred = time_s / 10000.0
    voltage = np.full(len(time_s), 3.3)
    voltage[(time_s >= 300.0) & (time_s <= 660.0)] = 3.65

    metrics_path, events_path = write_temporal_error_metrics(
        out_dir=str(tmp_path), cell="C00", time_s=time_s, y_true=y_true,
        y_pred=y_pred, voltage_v=voltage, interval_seconds=300.0,
        reset_voltage_v=3.63, reset_sustain_seconds=300.0,
    )

    metrics = pd.read_csv(metrics_path)
    events = pd.read_csv(events_path)
    assert np.isclose(metrics.loc[0, "mae"], np.mean(y_pred[:5]))
    assert len(events) == 1
    assert events.loc[0, "time_s"] == 600.0


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
    assert metrics["common_recovery_relapsed"] is False
    assert metrics["common_stable_recovery_time_h"] == 300.0 / 3600.0
    assert metrics["common_recovery_threshold_abs_err"] == 0.02


def test_recovery_reports_later_relapse_and_stable_return():
    time_s = np.arange(0.0, 1801.0, 60.0)
    error = np.full(len(time_s), 0.01)
    error[(time_s >= 600.0) & (time_s < 900.0)] = 0.05

    metrics = compute_common_recovery_metrics(
        time_s,
        np.zeros(len(time_s)),
        error,
        start_index=0,
        threshold=0.02,
        sustain_seconds=300.0,
        horizon_seconds=1800.0,
    )

    assert metrics["common_recovery_time_h"] == 0.0
    assert metrics["common_recovery_relapsed"] is True
    assert metrics["common_recovery_first_relapse_time_h"] == 600.0 / 3600.0
    assert metrics["common_stable_recovery_time_h"] == 900.0 / 3600.0


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
    assert not any(key.startswith("common_recovery") for key in metrics)


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
    assert metrics["common_stable_recovery_censored"] is True
    assert metrics["common_stable_recovery_or_censor_time_h"] == 600.0 / 3600.0


def test_recovery_clock_can_precede_the_common_scored_interval():
    time_s = np.arange(2023.0, 2601.0)
    metrics = compute_common_recovery_metrics(
        time_s,
        np.zeros(len(time_s)),
        np.zeros(len(time_s)),
        start_index=0,
        threshold=0.02,
        sustain_seconds=300.0,
        horizon_seconds=86400.0,
        event_time_s=0.0,
    )

    assert metrics["common_recovery_time_s"] == 2023.0
    assert metrics["common_recovery_observed_horizon_h"] == 2600.0 / 3600.0


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
    assert result["_dt_s_online"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert result["Q_c"].iloc[1] == result["Q_c"].iloc[2]


def test_online_q_c_matches_deployed_reset_and_clamp_semantics():
    frame = pd.DataFrame({
        "Testtime[s]": np.arange(7.0),
        "Current[A]": [-2.0, -2.0, -2.0, 1.0, 1.0, -10.0, -10.0],
        "Voltage[V]": [3.2, 3.2, 3.2, 3.5, 3.6002, 3.0, 3.0],
    })

    result = build_online_aux_features(
        frame,
        np.zeros(len(frame), dtype=bool),
        current_sign=1.0,
        v_max=3.65,
        v_tol=0.02,
        cv_seconds=300.0,
        nominal_capacity_ah=0.003,
        q_c_reset_voltage_v=3.6002,
        q_c_reset_current_a=0.1,
    )

    assert result["Q_c"].iloc[4] == 0.0
    assert result["Q_c"].iloc[-1] == -0.003
    assert np.all(result["Q_c"].to_numpy() <= 0.0)


def test_charge_severity_gap_uses_measurements_and_respects_context():
    time_s = np.arange(0.0, 10_001.0, 1.0)
    current = np.zeros(len(time_s))
    current[3000:4001] = 1.0
    current[5000:6001] = -3.0

    mask = compute_max_abs_net_charge_window_mask(
        time_s,
        current,
        gap_seconds=1000.0,
        min_pre_seconds=2000.0,
        min_post_seconds=3000.0,
    )

    indices = np.flatnonzero(mask)
    assert time_s[indices[0]] == 5000.0
    assert time_s[indices[-1]] == 6000.0


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
