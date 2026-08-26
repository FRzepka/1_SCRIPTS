#!/usr/bin/env python3
"""Build compact nominal STM32 vectors from verified JES2 reference runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_by_index(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=["index", column])
    return frame.set_index("index")[column]


def main() -> None:
    workspace = Path(__file__).resolve().parents[3]
    simulation = workspace / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment"
    default_reference = simulation / "campaigns/jes2_hardware_reference_C27_20260826"
    default_trace = simulation / "campaigns/jes2_full_all_scenario_smoke_20260825/traces/C27/C27_fresh/baseline/seed_42/lstm_h1.npz"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"))
    parser.add_argument("--reference-root", type=Path, default=default_reference)
    parser.add_argument("--soh-trace", type=Path, default=default_trace)
    parser.add_argument("--cell", default="C27")
    parser.add_argument("--start-row", type=int, default=5427475)
    parser.add_argument("--source-rows", type=int, default=86400)
    parser.add_argument("--vector-rows", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "test_vectors/jes2_nominal_vectors.csv")
    args = parser.parse_args()

    sys.path.insert(0, str(simulation))
    from robustness_common import build_online_aux_features, load_cell_dataframe
    from shared_soh_trace import load_soh_trace

    frame = load_cell_dataframe(args.data_root, args.cell, args.start_row, args.source_rows)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["Testtime[s]", "Current[A]", "Voltage[V]", "Temperature[°C]", "SOC"]).reset_index(drop=True)
    frame = build_online_aux_features(
        df=frame,
        freeze_mask=np.zeros(len(frame), dtype=bool),
        current_sign=1.0,
        v_max=3.65,
        v_tol=0.02,
        cv_seconds=300.0,
        nominal_capacity_ah=1.8,
    )
    time_float32 = frame["Testtime[s]"].to_numpy(dtype=np.float32)
    dt_s = np.empty_like(time_float32)
    dt_s[0] = max(float(time_float32[1] - time_float32[0]), 1e-6) if len(frame) > 1 else 1.0
    dt_s[1:] = np.diff(time_float32)
    soh, trace_metadata = load_soh_trace(args.soh_trace, frame["Testtime[s]"].to_numpy(dtype=np.float64))

    count = min(args.vector_rows, len(frame))
    vectors = pd.DataFrame({
        "sample_id": np.arange(count, dtype=np.int64),
        "segment_id": "C27_fresh_baseline",
        "reset": np.r_[1, np.zeros(max(count - 1, 0), dtype=np.int64)],
        "voltage_v": frame["Voltage[V]"].to_numpy(dtype=np.float32)[:count],
        "current_a": frame["Current[A]"].to_numpy(dtype=np.float32)[:count],
        "temperature_c": frame["Temperature[°C]"].to_numpy(dtype=np.float32)[:count],
        "soh": soh[:count],
        "q_c_ah": frame["Q_c"].to_numpy(dtype=np.float32)[:count],
        "dv_dt_v_s": frame["dU_dt[V/s]"].to_numpy(dtype=np.float32)[:count],
        "di_dt_a_s": frame["dI_dt[A/s]"].to_numpy(dtype=np.float32)[:count],
        "dt_s": dt_s[:count],
        "soc_dataset": frame["SOC"].to_numpy(dtype=np.float32)[:count],
    })
    references = {
        "expected_dm": prediction_by_index(args.reference_root / "DM/soc_cc_fullcell_C27.csv", "soc_cc"),
        "expected_hdm": prediction_by_index(args.reference_root / "HDM/soc_cc_soh_fullcell_C27.csv", "soc_cc"),
        "expected_hecm": prediction_by_index(args.reference_root / "HECM/ecm_soc_fullcell_C27.csv", "soc_ecm"),
        "expected_dd": prediction_by_index(args.reference_root / "DD/soc_pred_fullcell_C27.csv", "soc_pred"),
    }
    for name, values in references.items():
        vectors[name] = vectors["sample_id"].map(values)

    if vectors["expected_dd"].iloc[:2023].notna().any() or pd.isna(vectors["expected_dd"].iloc[2023]):
        raise ValueError("DD reference does not have the expected 2024-sample warm-up alignment")
    for name in ["expected_dm", "expected_hdm", "expected_hecm"]:
        if vectors[name].isna().any():
            raise ValueError(f"Reference column {name} is incomplete")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vectors.to_csv(args.out, index=False, float_format="%.9g")
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "nominal isolated STM32 SOC hardware benchmark; no measurement disturbances",
        "cell": args.cell,
        "window": "C27_fresh",
        "scenario": "baseline",
        "start_row": args.start_row,
        "source_rows": args.source_rows,
        "vector_rows": len(vectors),
        "dd_sequence_length": 2024,
        "dd_first_valid_sample_id": 2023,
        "soh_source": "shared causal JES2 LSTM trace",
        "soh_trace_metadata": trace_metadata,
        "csv": args.out.name,
        "csv_sha256": sha256(args.out),
        "reference_root": str(args.reference_root.resolve()),
    }
    metadata_path = args.out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
