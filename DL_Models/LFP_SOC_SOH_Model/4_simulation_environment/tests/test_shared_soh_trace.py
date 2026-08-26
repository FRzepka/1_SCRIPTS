from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared_soh_trace import (
    build_reference_soh_trace,
    load_soh_trace,
    prepare_trace_frame,
    save_soh_trace,
)


class ScenarioArgs:
    seed = 42
    current_offset_a = None
    current_offset_pct = None
    voltage_offset_v = None
    temp_offset_c = None
    current_noise_std = 0.1


def test_reference_trace_uses_only_completed_intervals(tmp_path):
    time_s = np.arange(0.0, 3.0 * 3600.0, 600.0)
    soh = np.where(time_s < 3600.0, 1.0, np.where(time_s < 7200.0, 0.9, 0.8))
    frame = pd.DataFrame({"Testtime[s]": time_s, "SOH": soh})
    trace, metadata = build_reference_soh_trace(
        frame,
        freeze_mask=np.zeros(len(frame), dtype=bool),
        interval_seconds=3600,
    )

    assert np.allclose(trace[time_s < 3600.0], 1.0)
    assert np.allclose(trace[(time_s >= 3600.0) & (time_s < 7200.0)], 1.0)
    assert np.allclose(trace[time_s >= 7200.0], 0.9)
    assert metadata["completed_intervals"] == 3

    path = tmp_path / "trace.npz"
    save_soh_trace(path, time_s, trace, metadata)
    aligned, loaded_metadata = load_soh_trace(path, time_s[::2])
    assert np.allclose(aligned, trace[::2])
    assert loaded_metadata == metadata


def test_fully_missing_interval_is_not_used_for_reference_update():
    time_s = np.arange(0.0, 3.0 * 3600.0, 600.0)
    frame = pd.DataFrame(
        {
            "Testtime[s]": time_s,
            "SOH": np.where(time_s < 3600.0, 1.0, np.where(time_s < 7200.0, 0.7, 0.8)),
        }
    )
    freeze = (time_s >= 3600.0) & (time_s < 7200.0)
    trace, _ = build_reference_soh_trace(frame, freeze_mask=freeze, interval_seconds=3600)

    assert np.allclose(trace[time_s >= 7200.0], 1.0)


def test_missing_rows_are_excluded_from_reference_aggregation():
    time_s = np.arange(0.0, 2.0 * 3600.0, 600.0)
    frame = pd.DataFrame({"Testtime[s]": time_s, "SOH": np.ones(len(time_s))})
    freeze = np.zeros(len(frame), dtype=bool)
    freeze[5] = True
    frame.loc[5, "SOH"] = 0.1

    trace, _ = build_reference_soh_trace(frame, freeze_mask=freeze, interval_seconds=3600)

    assert np.allclose(trace, 1.0)


def test_reference_publication_cadence_holds_completed_updates():
    time_s = np.arange(0.0, 5.0 * 3600.0, 600.0)
    soh = 1.0 - 0.1 * np.floor_divide(time_s.astype(np.int64), 3600)
    frame = pd.DataFrame({"Testtime[s]": time_s, "SOH": soh})

    trace, metadata = build_reference_soh_trace(
        frame,
        freeze_mask=np.zeros(len(frame), dtype=bool),
        interval_seconds=3600,
        publish_every_intervals=2,
    )

    assert np.allclose(trace[time_s < 7200.0], 1.0)
    assert np.allclose(trace[(time_s >= 7200.0) & (time_s < 14400.0)], 0.9)
    assert np.allclose(trace[time_s >= 14400.0], 0.7)
    assert metadata["completed_intervals"] == 5
    assert metadata["published_intervals"] == 2


def test_trace_context_is_undisturbed_and_excluded_from_evaluation_start():
    frame = pd.DataFrame(
        {
            "Testtime[s]": np.arange(20, dtype=float),
            "Current[A]": np.zeros(20),
            "Voltage[V]": np.full(20, 3.5),
            "Temperature[°C]": np.full(20, 25.0),
            "SOC": np.ones(20),
            "SOH": np.ones(20),
            "C_Rate": np.zeros(20),
        }
    )
    prepared, freeze, _, metadata = prepare_trace_frame(
        frame, "current_noise", ScenarioArgs(), start_row=10, max_rows=5, context_rows=4
    )

    assert len(prepared) == 9
    assert np.allclose(prepared.iloc[:4]["Current[A]"], 0.0)
    assert not np.allclose(prepared.iloc[4:]["Current[A]"], 0.0)
    assert not freeze.any()
    assert metadata == {
        "context_start_row": 6,
        "context_rows_actual": 4,
        "evaluation_start_index": 4,
    }
