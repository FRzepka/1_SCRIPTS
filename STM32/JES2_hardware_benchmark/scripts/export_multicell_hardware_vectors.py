#!/usr/bin/env python3
"""Export frozen nominal JES2 STM32 vectors for all six holdout cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CELLS = ("C09", "C13", "C15", "C25", "C27", "C29")
LOAD_CLASSES = {
    "C09": "medium", "C13": "medium", "C15": "medium",
    "C25": "low", "C27": "low", "C29": "high",
}
MODEL_CONFIG = {
    "DM": ("no_soh", "soc_cc_fullcell_*.csv", "soc_cc"),
    "HDM": ("lstm_h1", "soc_cc_soh_fullcell_*.csv", "soc_cc"),
    "HECM": ("lstm_h1", "ecm_soc_fullcell_*.csv", "soc_ecm"),
    "DD": ("lstm_h1", "soc_pred_fullcell_*.csv", "soc_pred"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=["index", column])
    return frame.set_index("index")[column]


def dd_onnx_predictions(vectors: pd.DataFrame, onnx_path: Path, scaler_path: Path) -> np.ndarray:
    import joblib
    import onnxruntime as ort

    features = [
        "voltage_v", "current_a", "temperature_c", "soh",
        "q_c_ah", "dv_dt_v_s", "di_dt_a_s", "dt_s",
    ]
    scaler = joblib.load(scaler_path)
    scaled = scaler.transform(vectors[features].to_numpy()).astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = np.full(len(vectors), np.nan, dtype=np.float32)
    for end in range(2023, len(vectors)):
        window = scaled[end - 2023:end + 1][None, :, :]
        output[end] = session.run(None, {input_name: window})[0][0]
    return output


def primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        ((frame.model == "DM") & (frame.soh_condition == "none"))
        | ((frame.model != "DM") & (frame.soh_condition == "lstm_h1"))
    ].copy()


def main() -> None:
    workspace = Path(__file__).resolve().parents[3]
    simulation = workspace / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"))
    parser.add_argument(
        "--run-metrics", type=Path,
        default=workspace / "LATEX/JES/paper_robustness_benchmark/JES_2.0/results/jes2_run_metrics.csv",
    )
    parser.add_argument("--vector-rows", type=int, default=4096)
    parser.add_argument(
        "--dd-onnx", type=Path,
        default=workspace / "STM32/JES2_hardware_benchmark/exports/DD/jes2_dd_window2024.onnx",
    )
    parser.add_argument(
        "--dd-scaler", type=Path,
        default=simulation.parents[0] / "2_models/SOC_1.7.0.0/PrunedFT_1.7.0.0_s30_struct/scaler_robust.joblib",
    )
    parser.add_argument(
        "--reference-runs-root", type=Path,
        default=simulation / "campaigns/jes2_hardware_reference_multicell_20260828/runs",
        help="Six-cell hardware-reference campaign built with the frozen input SOH traces.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "test_vectors/multicell")
    args = parser.parse_args()

    sys.path.insert(0, str(simulation))
    from robustness_common import build_online_aux_features, load_cell_dataframe
    from shared_soh_trace import load_soh_trace

    metrics = primary_rows(pd.read_csv(args.run_metrics))
    metrics = metrics[
        (metrics.alias == "baseline")
        & (metrics.window_soh_state == "fresh")
        & (metrics.seed == 42)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for cell in CELLS:
        selected = metrics[metrics.cell == cell]
        if set(selected.model) != set(MODEL_CONFIG):
            raise ValueError(f"Incomplete primary baseline runs for {cell}: {sorted(selected.model.unique())}")
        first = selected.iloc[0]
        start_row, source_rows = int(first.start_row), int(first.max_rows)
        frame = load_cell_dataframe(args.data_root, cell, start_row, source_rows)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(
            subset=["Testtime[s]", "Current[A]", "Voltage[V]", "Temperature[°C]", "SOC"]
        ).reset_index(drop=True)
        frame = build_online_aux_features(
            df=frame,
            freeze_mask=np.zeros(len(frame), dtype=bool),
            current_sign=1.0,
            v_max=3.65,
            v_tol=0.02,
            cv_seconds=300.0,
            nominal_capacity_ah=1.8,
        )
        campaign = simulation / f"campaigns/jes2_full_{cell}_20260825"
        trace = campaign / f"traces/{cell}/{cell}_fresh/baseline/seed_42/lstm_h1.npz"
        soh, trace_metadata = load_soh_trace(trace, frame["Testtime[s]"].to_numpy(dtype=np.float64))
        count = min(args.vector_rows, len(frame))
        time_float32 = frame["Testtime[s]"].to_numpy(dtype=np.float32)
        dt_s = np.empty_like(time_float32)
        dt_s[0] = max(float(time_float32[1] - time_float32[0]), 1e-6) if len(frame) > 1 else 1.0
        dt_s[1:] = np.diff(time_float32)
        vectors = pd.DataFrame({
            "sample_id": np.arange(count, dtype=np.int64),
            "segment_id": f"{cell}_fresh_baseline",
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
        reference_paths = {}
        for model, (soh_mode, pattern, column) in MODEL_CONFIG.items():
            if model == "DD":
                vectors["expected_dd"] = dd_onnx_predictions(vectors, args.dd_onnx, args.dd_scaler)
                reference_paths[model] = str(args.dd_onnx.relative_to(workspace))
                continue
            run_dir = args.reference_runs_root / cell / f"{cell}_fresh/baseline/seed_42" / soh_mode / model
            paths = list(run_dir.glob(pattern))
            if not paths:
                raise FileNotFoundError(
                    f"Missing {model} reference for {cell}. Run build_multicell_hardware_references.py first."
                )
            path = paths[0]
            vectors[f"expected_{model.lower()}"] = vectors.sample_id.map(prediction(path, column))
            reference_paths[model] = str(path.relative_to(workspace))
        if vectors.expected_dd.iloc[:2023].notna().any() or pd.isna(vectors.expected_dd.iloc[2023]):
            raise ValueError(f"Unexpected DD warm-up alignment for {cell}")
        if vectors[["expected_dm", "expected_hdm", "expected_hecm"]].isna().any().any():
            raise ValueError(f"Incomplete software references for {cell}")

        csv_path = args.out_dir / f"jes2_nominal_{cell}_vectors.csv"
        vectors.to_csv(csv_path, index=False, float_format="%.9g")
        metadata = {
            "schema_version": 2,
            "cell": cell,
            "load_class": LOAD_CLASSES[cell],
            "window": f"{cell}_fresh",
            "scenario": "baseline",
            "start_row": start_row,
            "source_rows": source_rows,
            "vector_rows": len(vectors),
            "dd_sequence_length": 2024,
            "dd_first_valid_sample_id": 2023,
            "soh_source": "shared causal JES2 LSTM trace",
            "soh_trace_metadata": trace_metadata,
            "csv": csv_path.name,
            "csv_sha256": sha256(csv_path),
            "software_references": reference_paths,
        }
        csv_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest_rows.append(metadata)

    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "nominal six-cell STM32 SOC hardware benchmark",
        "load_classes": {"low": ["C25", "C27"], "medium": ["C09", "C13", "C15"], "high": ["C29"]},
        "range_definition": "minimum and maximum across cells within each load class",
        "high_class_limitation": "C29 is the only high-load cell; no between-cell spread is estimable",
        "vectors": manifest_rows,
    }
    path = args.out_dir / "jes2_multicell_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest_rows)} cell vector sets to {args.out_dir}")


if __name__ == "__main__":
    main()
