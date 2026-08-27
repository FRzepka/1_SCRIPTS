from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
CC_SOH_DIR = ROOT.parent / "2_models" / "CC_SOH_1.0.0"
sys.path.append(str(CC_SOH_DIR))

from cc_soh_model import (  # noqa: E402
    CCSOHConfig,
    CCSOHModel,
    aggregate_hourly,
    expand_features_for_sampling,
)
from robustness_common import (  # noqa: E402
    add_common_scenario_args,
    apply_measurement_scenario,
    build_online_aux_features,
    load_cell_dataframe,
)


def _sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_after_completed_interval(
    time_s: np.ndarray,
    completed_bins: np.ndarray,
    completed_values: np.ndarray,
    interval_seconds: int,
    initial_value: float,
) -> np.ndarray:
    """Apply each hourly value only after its source interval has completed."""
    time_s = np.asarray(time_s, dtype=np.float64)
    completed_bins = np.asarray(completed_bins, dtype=np.int64)
    completed_values = np.asarray(completed_values, dtype=np.float64)
    if len(completed_bins) != len(completed_values):
        raise ValueError("SOH bin/value length mismatch")
    if len(completed_bins) == 0:
        return np.full(len(time_s), float(initial_value), dtype=np.float32)

    order = np.argsort(completed_bins)
    available_bins = completed_bins[order] + 1
    values = completed_values[order]
    row_bins = np.floor_divide(time_s.astype(np.int64), int(interval_seconds))
    source_idx = np.searchsorted(available_bins, row_bins, side="right") - 1

    out = np.full(len(time_s), float(initial_value), dtype=np.float64)
    valid = source_idx >= 0
    out[valid] = values[source_idx[valid]]
    return out.astype(np.float32)


def _select_publications(
    completed_bins: np.ndarray,
    completed_values: np.ndarray,
    publish_every_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    every = int(publish_every_intervals)
    if every < 1:
        raise ValueError("publish_every_intervals must be at least 1")
    bins = np.asarray(completed_bins, dtype=np.int64)
    values = np.asarray(completed_values)
    if every == 1 or len(bins) == 0:
        return bins, values
    publish_mask = np.mod(bins + 1, every) == 0
    return bins[publish_mask], values[publish_mask]


def build_reference_soh_trace(
    df: pd.DataFrame,
    freeze_mask: np.ndarray,
    interval_seconds: int,
    publish_every_intervals: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    if "SOH" not in df.columns:
        raise ValueError("Reference-SOH requested, but the SOH column is missing")

    reference = df[["Testtime[s]", "SOH"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if reference.empty:
        raise ValueError("No finite reference-SOH values available")
    observed_mask = ~np.asarray(freeze_mask, dtype=bool)
    work = df.loc[observed_mask, ["Testtime[s]", "SOH"]]
    work = work.replace([np.inf, -np.inf], np.nan).dropna().copy()
    work["_bin"] = np.floor_divide(work["Testtime[s]"].astype(np.int64), int(interval_seconds))
    hourly = work.groupby("_bin", sort=True)["SOH"].last()
    published_bins, published_values = _select_publications(
        hourly.index.to_numpy(dtype=np.int64),
        hourly.to_numpy(dtype=np.float64),
        publish_every_intervals,
    )

    initial_value = float(reference.iloc[0]["SOH"])
    values = _expand_after_completed_interval(
        time_s=df["Testtime[s]"].to_numpy(dtype=np.float64),
        completed_bins=published_bins,
        completed_values=published_values,
        interval_seconds=interval_seconds,
        initial_value=initial_value,
    )
    return values, {
        "source": "reference_soh",
        "initial_value": initial_value,
        "completed_intervals": int(len(hourly)),
        "published_intervals": int(len(published_bins)),
        "publish_every_intervals": int(publish_every_intervals),
        "interval_seconds": int(interval_seconds),
    }


def build_lstm_soh_trace(
    df: pd.DataFrame,
    freeze_mask: np.ndarray,
    interval_seconds: int,
    soh_config: str,
    soh_checkpoint: str,
    soh_scaler: str,
    soh_init: float,
    device: str | None,
    nominal_capacity_ah: float,
    current_sign: float,
    v_max: float,
    v_tol: float,
    cv_seconds: float,
    publish_every_intervals: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = CCSOHConfig(
        soh_config=soh_config,
        soh_checkpoint=soh_checkpoint,
        soh_scaler=soh_scaler,
        nominal_capacity_ah=nominal_capacity_ah,
        soh_interval_seconds=interval_seconds,
        soh_init=soh_init,
        device=device,
        current_sign=current_sign,
        v_max=v_max,
        v_tol=v_tol,
        cv_seconds=cv_seconds,
    )
    model = CCSOHModel(cfg)
    interval_seconds = int(model.soh_interval_seconds)
    observed_df = df.loc[~np.asarray(freeze_mask, dtype=bool)].copy()
    hourly = aggregate_hourly(
        observed_df,
        model.soh_base_features,
        interval_seconds,
        model.soh_feature_aggs,
    )
    if hourly.empty:
        values = np.full(len(df), float(soh_init), dtype=np.float32)
        return values, {
            "source": "lstm_soh",
            "model_type": model.soh_model_type,
            "initial_value": float(soh_init),
            "completed_intervals": 0,
            "published_intervals": 0,
            "publish_every_intervals": int(publish_every_intervals),
            "interval_seconds": int(interval_seconds),
        }

    feature_cols = expand_features_for_sampling(model.soh_base_features, model.soh_feature_aggs)
    scaled = model.soh_scaler.transform(hourly[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    predictions: list[float] = []
    state = None
    with torch.no_grad():
        for row in scaled:
            x_step = torch.from_numpy(row[None, None, :]).to(model.device)
            y_seq, state = model.soh_model(x_step, state=state, return_state=True)
            predictions.append(float(y_seq.squeeze().detach().cpu().numpy()))

    published_bins, published_values = _select_publications(
        hourly["bin"].to_numpy(dtype=np.int64),
        np.asarray(predictions, dtype=np.float64),
        publish_every_intervals,
    )

    values = _expand_after_completed_interval(
        time_s=df["Testtime[s]"].to_numpy(dtype=np.float64),
        completed_bins=published_bins,
        completed_values=published_values,
        interval_seconds=interval_seconds,
        initial_value=soh_init,
    )
    return values, {
        "source": "lstm_soh",
        "model_type": model.soh_model_type,
        "initial_value": float(soh_init),
        "completed_intervals": int(len(hourly)),
        "published_intervals": int(len(published_bins)),
        "publish_every_intervals": int(publish_every_intervals),
        "interval_seconds": int(interval_seconds),
        "soh_config": str(Path(soh_config).resolve()),
        "soh_checkpoint": str(Path(soh_checkpoint).resolve()),
        "soh_scaler": str(Path(soh_scaler).resolve()),
        "soh_config_sha256": _sha256(soh_config),
        "soh_checkpoint_sha256": _sha256(soh_checkpoint),
        "soh_scaler_sha256": _sha256(soh_scaler),
    }


def save_soh_trace(path: str | Path, time_s: np.ndarray, soh: np.ndarray, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_s=np.asarray(time_s, dtype=np.float64),
        soh=np.asarray(soh, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_soh_trace(path: str | Path, target_time_s: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        trace_time = payload["time_s"].astype(np.float64)
        trace_soh = payload["soh"].astype(np.float32)
        raw_metadata = payload["metadata_json"].item()
    metadata = json.loads(str(raw_metadata))

    if len(trace_time) == 0 or len(trace_time) != len(trace_soh):
        raise ValueError(f"Invalid SOH trace: {path}")
    if np.any(np.diff(trace_time) < 0):
        raise ValueError(f"SOH trace timestamps are not monotonic: {path}")

    target = np.asarray(target_time_s, dtype=np.float64)
    tolerance = 1e-6
    if len(target) and (target[0] < trace_time[0] - tolerance or target[-1] > trace_time[-1] + tolerance):
        raise ValueError("SOH trace does not cover the requested scenario time range")
    source_idx = np.searchsorted(trace_time, target, side="right") - 1
    source_idx = np.clip(source_idx, 0, len(trace_time) - 1)
    return trace_soh[source_idx].astype(np.float32), metadata


def _scenario_metadata(info: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in info.items() if key not in {"freeze_mask", "disturbance_mask"}}


def prepare_trace_frame(
    frame: pd.DataFrame,
    scenario: str,
    args,
    start_row: int,
    max_rows: int,
    context_rows: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any], dict[str, int]]:
    """Build an undisturbed causal prefix followed by the evaluation scenario."""
    start = int(start_row)
    stop = start + int(max_rows) if max_rows > 0 else len(frame)
    if context_rows <= 0:
        evaluation = frame.iloc[start:stop].copy().reset_index(drop=True)
        evaluation, scenario_info = apply_measurement_scenario(evaluation, scenario, args)
        freeze = np.asarray(scenario_info.get("freeze_mask", np.zeros(len(evaluation), dtype=bool)), dtype=bool)
        return evaluation, freeze, scenario_info, {
            "context_start_row": start,
            "context_rows_actual": 0,
            "evaluation_start_index": 0,
        }

    context_start = max(0, start - int(context_rows))
    context = frame.iloc[context_start:start].copy().reset_index(drop=True)
    evaluation = frame.iloc[start:stop].copy().reset_index(drop=True)
    context, context_info = apply_measurement_scenario(context, "baseline", args)
    evaluation, scenario_info = apply_measurement_scenario(evaluation, scenario, args)
    context_freeze = np.asarray(
        context_info.get("freeze_mask", np.zeros(len(context), dtype=bool)), dtype=bool
    )
    evaluation_freeze = np.asarray(
        scenario_info.get("freeze_mask", np.zeros(len(evaluation), dtype=bool)), dtype=bool
    )
    combined = pd.concat([context, evaluation], ignore_index=True, sort=False)
    freeze = np.concatenate([context_freeze, evaluation_freeze])
    return combined, freeze, scenario_info, {
        "context_start_row": context_start,
        "context_rows_actual": int(len(context)),
        "evaluation_start_index": int(len(context)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one causal SOH trace shared by all JES2 SOC estimators.")
    parser.add_argument("--mode", choices=["lstm", "reference"], required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data_root", default="/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
    parser.add_argument("--soh_config")
    parser.add_argument("--soh_ckpt")
    parser.add_argument("--soh_scaler")
    parser.add_argument("--soh_init", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--interval_seconds", type=int, default=3600)
    parser.add_argument("--publish_every_intervals", type=int, default=1,
                        help="Publish only every Nth completed SOH interval; model state still updates hourly.")
    parser.add_argument("--nominal_capacity_ah", type=float, default=1.8)
    parser.add_argument("--current_sign", type=float, default=1.0)
    parser.add_argument("--v_max", type=float, default=3.65)
    parser.add_argument("--v_tol", type=float, default=0.02)
    parser.add_argument("--cv_seconds", type=float, default=300.0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--context_rows", type=int, default=0,
                        help="Undisturbed rows before start_row used only to initialize the causal SOH state.")
    parser.add_argument("--warmup_seconds", type=float, default=0.0)
    add_common_scenario_args(parser)
    args = parser.parse_args()

    if args.mode == "lstm" and not all([args.soh_config, args.soh_ckpt, args.soh_scaler]):
        parser.error("LSTM mode requires --soh_config, --soh_ckpt, and --soh_scaler")

    source_start = max(0, int(args.start_row) - int(args.context_rows))
    relative_start = int(args.start_row) - source_start
    source_max_rows = relative_start + int(args.max_rows) if args.max_rows > 0 else 0
    source_frame = load_cell_dataframe(
        args.data_root, args.cell, source_start, source_max_rows
    ).replace([np.inf, -np.inf], np.nan)
    frame, freeze_mask, scenario_info, context_metadata = prepare_trace_frame(
        source_frame,
        args.scenario,
        args,
        start_row=relative_start,
        max_rows=args.max_rows,
        context_rows=args.context_rows,
    )
    context_metadata["context_start_row"] = source_start
    frame = frame.dropna(subset=["Testtime[s]", "Current[A]", "Voltage[V]", "SOC"]).reset_index(drop=True)
    if len(freeze_mask) != len(frame):
        raise ValueError("SOH freeze mask no longer aligns after mandatory-column filtering")
    frame = build_online_aux_features(
        df=frame,
        freeze_mask=freeze_mask,
        current_sign=args.current_sign,
        v_max=args.v_max,
        v_tol=args.v_tol,
        cv_seconds=args.cv_seconds,
        nominal_capacity_ah=args.nominal_capacity_ah,
        # SOH is a shared measurement-derived service and must not inherit a
        # branch-specific SOC initialization perturbation.
        initial_soc_delta=0.0,
        q_c_reset_voltage_v=args.q_c_reset_voltage_v,
        q_c_reset_current_a=args.q_c_reset_current_a,
        q_c_capacity_ah=args.q_c_capacity_ah,
    )

    if args.mode == "lstm":
        soh, source_metadata = build_lstm_soh_trace(
            df=frame,
            freeze_mask=freeze_mask,
            interval_seconds=args.interval_seconds,
            soh_config=args.soh_config,
            soh_checkpoint=args.soh_ckpt,
            soh_scaler=args.soh_scaler,
            soh_init=args.soh_init,
            device=args.device,
            nominal_capacity_ah=args.nominal_capacity_ah,
            current_sign=args.current_sign,
            v_max=args.v_max,
            v_tol=args.v_tol,
            cv_seconds=args.cv_seconds,
            publish_every_intervals=args.publish_every_intervals,
        )
    else:
        soh, source_metadata = build_reference_soh_trace(
            frame,
            freeze_mask,
            args.interval_seconds,
            publish_every_intervals=args.publish_every_intervals,
        )

    metadata = {
        **source_metadata,
        "cell": args.cell,
        "scenario": args.scenario,
        "seed": int(args.seed),
        "interval_seconds": int(source_metadata["interval_seconds"]),
        "causal_update": "completed_interval_applied_to_next_interval",
        "rows": int(len(frame)),
        "start_row": int(args.start_row),
        "max_rows": int(args.max_rows),
        "requested_context_rows": int(args.context_rows),
        **context_metadata,
        "data_root": str(Path(args.data_root).resolve()),
        "scenario_meta": _scenario_metadata(scenario_info),
    }
    save_soh_trace(args.out, frame["Testtime[s]"].to_numpy(dtype=np.float64), soh, metadata)
    print(json.dumps({"trace": str(Path(args.out).resolve()), **metadata}, indent=2))


if __name__ == "__main__":
    main()
