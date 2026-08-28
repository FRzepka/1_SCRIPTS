import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCENARIO_CHOICES = [
    "baseline",
    "current_offset",
    "voltage_offset",
    "temp_offset",
    "current_noise",
    "voltage_noise",
    "temp_noise",
    "adc_quantization",
    "spikes",
    "initial_soc_error",
    "missing_samples",
    "irregular_sampling",
    "missing_gap",
    "temp_mask",
    "downsample",
    "missing_segments",
]

STRATIFICATION_PROTOCOL = {
    "soh_state": {
        "aged": "SOH < 0.80",
        "mid_life": "0.80 <= SOH < 0.90",
        "fresh": "SOH >= 0.90",
    },
    "temperature_state": {
        "nominal": "T <= 30 degC",
        "elevated": "30 < T <= 35 degC",
        "hot": "T > 35 degC",
    },
    "instantaneous_load": {
        "low": "|C-rate| < 0.5",
        "medium": "0.5 <= |C-rate| < 1.5",
        "high": "|C-rate| >= 1.5",
    },
    "soc_state": {
        "low": "SOC < 0.20",
        "middle": "0.20 <= SOC <= 0.80",
        "high": "SOC > 0.80",
    },
}


def load_cell_dataframe(
    data_root: str,
    cell: str,
    start_row: int = 0,
    max_rows: int = 0,
) -> pd.DataFrame:
    path = os.path.join(data_root, f"df_FE_{cell.split('_')[-1]}.parquet")
    if not os.path.exists(path):
        path = os.path.join(data_root, f"df_FE_{cell}.parquet")
    if not os.path.exists(path):
        cid = cell[-3:]
        alt = os.path.join(data_root, f"df_FE_C{cid}.parquet")
        if os.path.exists(alt):
            path = alt
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not locate parquet for cell {cell} in {data_root}")
    start = int(start_row)
    limit = int(max_rows)
    if start < 0 or limit < 0:
        raise ValueError("start_row and max_rows must not be negative")
    if start == 0 and limit == 0:
        return pd.read_parquet(path)

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    stop = start + limit if limit > 0 else parquet.metadata.num_rows
    group_offsets: list[tuple[int, int]] = []
    offset = 0
    for group in range(parquet.num_row_groups):
        group_stop = offset + parquet.metadata.row_group(group).num_rows
        if group_stop > start and offset < stop:
            group_offsets.append((group, offset))
        offset = group_stop
    if not group_offsets:
        return pd.DataFrame()
    groups = [group for group, _ in group_offsets]
    loaded_start = group_offsets[0][1]
    frame = parquet.read_row_groups(groups).to_pandas()
    local_start = start - loaded_start
    local_stop = local_start + limit if limit > 0 else None
    return frame.iloc[local_start:local_stop].copy().reset_index(drop=True)


def add_common_scenario_args(ap) -> None:
    ap.add_argument("--scenario", default="baseline", choices=SCENARIO_CHOICES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--current_offset_a", type=float, default=None)
    ap.add_argument("--current_offset_pct", type=float, default=None)
    ap.add_argument("--voltage_offset_v", type=float, default=None)
    ap.add_argument("--temp_offset_c", type=float, default=None)
    ap.add_argument("--current_noise_std", type=float, default=None)
    ap.add_argument("--voltage_noise_std", type=float, default=None)
    ap.add_argument("--temp_noise_std", type=float, default=None)
    ap.add_argument("--temp_constant", type=float, default=None)
    ap.add_argument("--quantize_current_a", type=float, default=None)
    ap.add_argument("--quantize_voltage_v", type=float, default=None)
    ap.add_argument("--quantize_temp_c", type=float, default=None)
    ap.add_argument("--spike_channel", choices=["Current[A]", "Voltage[V]", "Temperature[°C]"], default="Voltage[V]")
    ap.add_argument("--spike_magnitude", type=float, default=None)
    ap.add_argument("--spike_period", type=int, default=None)
    ap.add_argument("--spike_prob", type=float, default=None)
    ap.add_argument("--soc_init_error", type=float, default=0.0,
                    help="Additive SOC init error in fraction, e.g. 0.1 for +10%.")
    ap.add_argument("--missing_gap_seconds", type=float, default=0.0,
                    help="Length of one central burst-dropout/freeze window in seconds.")
    ap.add_argument(
        "--missing_gap_placement",
        choices=["center", "max_abs_net_charge"],
        default="center",
        help=(
            "Place the burst dropout at the trajectory center or at the measurement-only "
            "window with the largest absolute unobserved net charge."
        ),
    )
    ap.add_argument("--missing_gap_min_pre_seconds", type=float, default=0.0,
                    help="Required observed duration before a charge-severity-selected burst dropout.")
    ap.add_argument("--missing_gap_min_post_seconds", type=float, default=0.0,
                    help="Required observed duration after a charge-severity-selected burst dropout.")
    ap.add_argument("--missing_samples_every", type=int, default=None,
                    help="Freeze every Nth sample.")
    ap.add_argument("--missing_samples_pct", type=float, default=None,
                    help="Randomly freeze this fraction of samples.")
    ap.add_argument("--irregular_dt_jitter", type=float, default=None,
                    help="Uniform +/- jitter in seconds added to each sampling interval.")
    ap.add_argument("--downsample_hz", type=float, default=None)
    ap.add_argument("--drop_pct", type=float, default=None)
    ap.add_argument("--drop_segment_len", type=int, default=None)
    ap.add_argument("--recovery_abs_error_threshold", type=float, default=0.02,
                    help="Common absolute SOC-error threshold for cross-model recovery metrics.")
    ap.add_argument("--recovery_sustain_seconds", type=float, default=300.0,
                    help="Required continuous time inside the common recovery band.")
    ap.add_argument("--recovery_horizon_seconds", type=float, default=86400.0,
                    help="Post-event horizon for common recovery metrics.")
    ap.add_argument("--summary_only", action="store_true",
                    help="Write summary.json only; omit per-sample CSV and diagnostic plots.")
    ap.add_argument(
        "--temporal_metrics_seconds", type=float, default=0.0,
        help="Write compact time-binned prediction/error metrics even in summary-only mode.",
    )
    ap.add_argument("--q_c_reset_voltage_v", type=float, default=3.6002,
                    help="Online Q_c full-charge reset threshold used by training data and firmware.")
    ap.add_argument("--q_c_reset_current_a", type=float, default=0.1,
                    help="Minimum charging current required for an online Q_c voltage reset.")
    ap.add_argument("--q_c_capacity_ah", type=float, default=None,
                    help="Online Q_c lower-clamp magnitude; defaults to nominal_capacity_ah.")


def write_temporal_error_metrics(
    out_dir: str,
    cell: str,
    time_s: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    voltage_v: np.ndarray,
    interval_seconds: float,
    reference_soh: Optional[np.ndarray] = None,
    reset_voltage_v: float = 3.63,
    reset_sustain_seconds: float = 300.0,
) -> tuple[str, str]:
    """Write exact binned MAE and model-independent full-charge events."""
    interval = float(interval_seconds)
    if interval <= 0:
        raise ValueError("temporal_metrics_seconds must be positive")
    t = np.asarray(time_s, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    voltage = np.asarray(voltage_v, dtype=np.float64)
    if not (len(t) == len(truth) == len(pred) == len(voltage)):
        raise ValueError("Temporal metric inputs must have identical lengths")
    valid = np.isfinite(t) & np.isfinite(truth) & np.isfinite(pred) & np.isfinite(voltage)
    if not np.any(valid):
        raise ValueError("No finite samples available for temporal metrics")
    t, truth, pred, voltage = t[valid], truth[valid], pred[valid], voltage[valid]
    soh = None if reference_soh is None else np.asarray(reference_soh, dtype=np.float64)[valid]

    bin_id = np.floor((t - t[0]) / interval).astype(np.int64)
    count = np.bincount(bin_id)

    def bin_mean(values: np.ndarray) -> np.ndarray:
        return np.bincount(bin_id, weights=values, minlength=len(count)) / count

    signed_error = pred - truth
    frame = pd.DataFrame({
        "bin": np.arange(len(count), dtype=np.int64),
        "time_s": t[0] + (np.arange(len(count)) + 0.5) * interval,
        "elapsed_h": (np.arange(len(count)) + 0.5) * interval / 3600.0,
        "n_samples": count,
        "soc_true_mean": bin_mean(truth),
        "soc_pred_mean": bin_mean(pred),
        "mae": bin_mean(np.abs(signed_error)),
        "rmse": np.sqrt(bin_mean(np.square(signed_error))),
        "signed_error": bin_mean(signed_error),
        "voltage_mean": bin_mean(voltage),
    })
    if soh is not None:
        frame["soh_mean"] = bin_mean(soh)

    high = voltage >= float(reset_voltage_v)
    starts = np.flatnonzero(high & np.r_[True, ~high[:-1]])
    stops = np.flatnonzero(high & np.r_[~high[1:], True])
    durations = t[stops] - t[starts]
    eligible = durations >= float(reset_sustain_seconds)
    reset_times = t[starts[eligible]] + float(reset_sustain_seconds)
    reset_indices = np.searchsorted(t, reset_times, side="left")
    reset_indices = np.clip(reset_indices, 0, len(t) - 1)
    events = pd.DataFrame({
        "event": np.arange(1, len(reset_times) + 1, dtype=np.int64),
        "time_s": reset_times,
        "elapsed_h": (reset_times - t[0]) / 3600.0,
        "source_index": reset_indices,
        "voltage_v": voltage[reset_indices],
    })
    if soh is not None:
        events["soh"] = soh[reset_indices]

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"temporal_metrics_{cell}.csv")
    events_path = os.path.join(out_dir, f"full_charge_events_{cell}.csv")
    frame.to_csv(metrics_path, index=False)
    events.to_csv(events_path, index=False)
    return metrics_path, events_path


def compute_center_window_mask(t: np.ndarray, gap_seconds: float) -> np.ndarray:
    if gap_seconds is None or gap_seconds <= 0 or len(t) == 0:
        return np.zeros(len(t), dtype=bool)
    t0 = float(t[0])
    t1 = float(t[-1])
    span = t1 - t0
    if span <= gap_seconds:
        return np.zeros(len(t), dtype=bool)
    start = t0 + (span - gap_seconds) * 0.5
    end = start + gap_seconds
    return (t >= start) & (t <= end)


def compute_max_abs_net_charge_window_mask(
    t: np.ndarray,
    current_a: np.ndarray,
    gap_seconds: float,
    min_pre_seconds: float = 0.0,
    min_post_seconds: float = 0.0,
) -> np.ndarray:
    """Select a fixed-duration gap using measured current only.

    The eligible interval must retain the requested observed context before and
    after the dropout. Among eligible starts, the earliest window with maximum
    absolute integrated current is selected. No target or estimator output is
    used, so selection remains causal-estimator independent and reproducible.
    """
    time_s = np.asarray(t, dtype=np.float64)
    current = np.asarray(current_a, dtype=np.float64)
    if len(time_s) == 0 or len(time_s) != len(current) or gap_seconds <= 0.0:
        return np.zeros(len(time_s), dtype=bool)
    if float(time_s[-1] - time_s[0]) <= float(gap_seconds):
        return np.zeros(len(time_s), dtype=bool)

    earliest = float(time_s[0]) + max(float(min_pre_seconds), 0.0)
    latest_start = float(time_s[-1]) - max(float(min_post_seconds), 0.0) - float(gap_seconds)
    starts = np.flatnonzero((time_s >= earliest) & (time_s <= latest_start))
    if len(starts) == 0:
        return compute_center_window_mask(time_s, gap_seconds)

    dt_s = np.diff(time_s, prepend=time_s[0])
    dt_s[~np.isfinite(dt_s) | (dt_s < 0.0)] = 0.0
    charge_ah = np.nan_to_num(current, nan=0.0) * dt_s / 3600.0
    prefix = np.concatenate(([0.0], np.cumsum(charge_ah, dtype=np.float64)))
    ends = np.searchsorted(time_s, time_s[starts] + float(gap_seconds), side="right") - 1
    valid = ends >= starts
    starts = starts[valid]
    ends = ends[valid]
    if len(starts) == 0:
        return compute_center_window_mask(time_s, gap_seconds)

    net_charge = prefix[ends + 1] - prefix[starts]
    winner = int(np.nanargmax(np.abs(net_charge)))
    start = float(time_s[starts[winner]])
    end = start + float(gap_seconds)
    return (time_s >= start) & (time_s <= end)


def _quantize(arr: np.ndarray, step: float) -> np.ndarray:
    if step is None or step <= 0:
        return arr
    return np.round(arr / float(step)) * float(step)


def _scenario_rng(seed: int):
    return np.random.default_rng(int(seed))


def _interp_numeric_to_timebase(src_t: np.ndarray, src_y: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    valid = np.isfinite(src_t) & np.isfinite(src_y)
    if valid.sum() == 0:
        return np.full(len(dst_t), np.nan, dtype=np.float64)
    if valid.sum() == 1:
        return np.full(len(dst_t), float(src_y[valid][0]), dtype=np.float64)
    return np.interp(dst_t, src_t[valid], src_y[valid], left=float(src_y[valid][0]), right=float(src_y[valid][-1]))


def _resample_irregular_timebase(df: pd.DataFrame, jitter_s: float, rng) -> Tuple[pd.DataFrame, Dict]:
    out = df.sort_values("Testtime[s]").reset_index(drop=True).copy()
    t_src = out["Testtime[s]"].to_numpy(dtype=np.float64)
    if len(t_src) < 2:
        return out, {"nominal_dt_s": 0.0, "resampled_span_s": 0.0}

    dt_src = np.diff(t_src)
    positive_dt = dt_src[dt_src > 0]
    nominal_dt = float(np.median(positive_dt)) if len(positive_dt) else 1.0
    if not np.isfinite(nominal_dt) or nominal_dt <= 0.0:
        nominal_dt = 1.0

    dt_new = nominal_dt + rng.uniform(-float(jitter_s), float(jitter_s), size=len(out) - 1)
    dt_new = np.clip(dt_new, 0.05, None)

    target_span = float(t_src[-1] - t_src[0])
    sum_dt = float(dt_new.sum())
    if target_span > 0.0 and sum_dt > 0.0:
        dt_new *= target_span / sum_dt

    t_dst = np.empty(len(out), dtype=np.float64)
    t_dst[0] = float(t_src[0])
    t_dst[1:] = float(t_src[0]) + np.cumsum(dt_new)

    out["Testtime[s]"] = t_dst
    out["_source_time_s"] = t_src

    for col in out.columns:
        if col in {"Testtime[s]", "_source_time_s"}:
            continue
        series = out[col]
        if pd.api.types.is_numeric_dtype(series):
            out[col] = _interp_numeric_to_timebase(
                src_t=t_src,
                src_y=series.to_numpy(dtype=np.float64),
                dst_t=t_dst,
            )

    return out, {
        "nominal_dt_s": nominal_dt,
        "resampled_span_s": float(t_dst[-1] - t_dst[0]),
    }


def apply_measurement_scenario(df: pd.DataFrame, scenario: str, args) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    for source, target in [
        ("SOH", "_reference_soh"),
        ("Temperature[°C]", "_reference_temperature_c"),
        ("C_Rate", "_reference_c_rate"),
    ]:
        if source in out.columns and target not in out.columns:
            out[target] = out[source].to_numpy(dtype=np.float64, copy=True)
    rng = _scenario_rng(getattr(args, "seed", 42))
    n = len(out)
    freeze_mask = np.zeros(n, dtype=bool)
    disturbance_mask = np.zeros(n, dtype=bool)
    meta: Dict[str, object] = {
        "scenario": scenario,
        "freeze_mask": freeze_mask,
        "disturbance_mask": disturbance_mask,
        "soc_init_delta": 0.0,
        "uses_only_measurement_manipulation": True,
    }

    if scenario == "baseline":
        return out, meta

    if scenario == "initial_soc_error":
        meta["soc_init_delta"] = float(getattr(args, "soc_init_error", 0.0) or 0.0)
        meta["uses_only_measurement_manipulation"] = False
        meta["perturbation_scope"] = "explicit_soc_state_only"
        return out, meta

    if scenario == "current_offset":
        if getattr(args, "current_offset_a", None) is not None:
            out["Current[A]"] = out["Current[A]"] + float(args.current_offset_a)
            disturbance_mask[:] = True
            meta["current_offset_a"] = float(args.current_offset_a)
        elif getattr(args, "current_offset_pct", None) is not None:
            out["Current[A]"] = out["Current[A]"] * (1.0 + float(args.current_offset_pct))
            disturbance_mask[:] = True
            meta["current_offset_pct"] = float(args.current_offset_pct)
        return out, meta

    if scenario == "voltage_offset":
        out["Voltage[V]"] = out["Voltage[V]"] + float(getattr(args, "voltage_offset_v", 0.0) or 0.0)
        disturbance_mask[:] = True
        meta["voltage_offset_v"] = float(getattr(args, "voltage_offset_v", 0.0) or 0.0)
        return out, meta

    if scenario == "temp_offset":
        if "Temperature[°C]" in out.columns:
            out["Temperature[°C]"] = out["Temperature[°C]"] + float(getattr(args, "temp_offset_c", 0.0) or 0.0)
            disturbance_mask[:] = True
        meta["temp_offset_c"] = float(getattr(args, "temp_offset_c", 0.0) or 0.0)
        return out, meta

    if scenario == "current_noise":
        std = float(getattr(args, "current_noise_std", 0.0) or 0.0)
        out["Current[A]"] = out["Current[A]"] + rng.normal(0.0, std, size=n)
        disturbance_mask[:] = std > 0.0
        meta["current_noise_std"] = std
        return out, meta

    if scenario == "voltage_noise":
        std = float(getattr(args, "voltage_noise_std", 0.0) or 0.0)
        out["Voltage[V]"] = out["Voltage[V]"] + rng.normal(0.0, std, size=n)
        disturbance_mask[:] = std > 0.0
        meta["voltage_noise_std"] = std
        return out, meta

    if scenario == "temp_noise":
        std = float(getattr(args, "temp_noise_std", 0.0) or 0.0)
        if "Temperature[°C]" in out.columns:
            out["Temperature[°C]"] = out["Temperature[°C]"] + rng.normal(0.0, std, size=n)
            disturbance_mask[:] = std > 0.0
        meta["temp_noise_std"] = std
        return out, meta

    if scenario == "adc_quantization":
        if "Current[A]" in out.columns:
            step = getattr(args, "quantize_current_a", None)
            out["Current[A]"] = _quantize(out["Current[A]"].to_numpy(dtype=np.float64), 0.01 if step is None else float(step))
        if "Voltage[V]" in out.columns:
            step = getattr(args, "quantize_voltage_v", None)
            out["Voltage[V]"] = _quantize(out["Voltage[V]"].to_numpy(dtype=np.float64), 0.005 if step is None else float(step))
        if "Temperature[°C]" in out.columns:
            step = getattr(args, "quantize_temp_c", None)
            out["Temperature[°C]"] = _quantize(out["Temperature[°C]"].to_numpy(dtype=np.float64), 0.5 if step is None else float(step))
        disturbance_mask[:] = True
        meta["quantize_current_a"] = float(getattr(args, "quantize_current_a", 0.01) or 0.01)
        meta["quantize_voltage_v"] = float(getattr(args, "quantize_voltage_v", 0.005) or 0.005)
        meta["quantize_temp_c"] = float(getattr(args, "quantize_temp_c", 0.5) or 0.5)
        return out, meta

    if scenario == "spikes":
        channel = str(getattr(args, "spike_channel", "Voltage[V]"))
        if channel not in out.columns:
            return out, meta
        mag = float(getattr(args, "spike_magnitude", 0.0) or 0.0)
        if getattr(args, "spike_period", None):
            idx = np.arange(0, n, int(args.spike_period))
        else:
            prob = float(getattr(args, "spike_prob", 0.001) or 0.001)
            idx = np.flatnonzero(rng.random(n) < prob)
        if len(idx):
            signs = rng.choice([-1.0, 1.0], size=len(idx))
            out.loc[idx, channel] = out.loc[idx, channel].to_numpy(dtype=np.float64) + signs * mag
            disturbance_mask[idx] = True
        meta["spike_channel"] = channel
        meta["spike_magnitude"] = mag
        meta["spike_count"] = int(disturbance_mask.sum())
        return out, meta

    if scenario == "temp_mask":
        if "Temperature[°C]" in out.columns:
            if getattr(args, "temp_constant", None) is not None:
                out["Temperature[°C]"] = float(args.temp_constant)
            else:
                out["Temperature[°C]"] = np.nan
            disturbance_mask[:] = True
        meta["temp_constant"] = getattr(args, "temp_constant", None)
        return out, meta

    if scenario == "downsample":
        if "Testtime[s]" not in out.columns:
            return out, meta
        dt = out["Testtime[s]"].diff().median()
        if not np.isfinite(dt) or dt <= 0:
            return out, meta
        orig_hz = 1.0 / dt
        target_hz = float(getattr(args, "downsample_hz", 1.0) or 1.0)
        stride = max(1, int(round(orig_hz / target_hz)))
        out = out.iloc[::stride].reset_index(drop=True)
        disturbance_mask = np.ones(len(out), dtype=bool)
        meta["freeze_mask"] = np.zeros(len(out), dtype=bool)
        meta["disturbance_mask"] = disturbance_mask
        meta["downsample_hz"] = target_hz
        meta["downsample_stride"] = stride
        return out, meta

    if scenario == "missing_segments":
        drop_pct = float(getattr(args, "drop_pct", 0.1) or 0.0)
        seg_len = int(getattr(args, "drop_segment_len", 1000) or 1000)
        if drop_pct <= 0:
            return out, meta
        to_drop = set()
        n_drop = int(n * drop_pct)
        while len(to_drop) < n_drop:
            start = int(rng.integers(0, max(1, n - seg_len)))
            for i in range(start, min(n, start + seg_len)):
                to_drop.add(i)
                if len(to_drop) >= n_drop:
                    break
        keep_idx = [i for i in range(n) if i not in to_drop]
        out = out.iloc[keep_idx].reset_index(drop=True)
        disturbance_mask = np.ones(len(out), dtype=bool)
        meta["freeze_mask"] = np.zeros(len(out), dtype=bool)
        meta["disturbance_mask"] = disturbance_mask
        meta["drop_pct"] = drop_pct
        meta["drop_segment_len"] = seg_len
        return out, meta

    if scenario == "missing_gap":
        t = out["Testtime[s]"].to_numpy(dtype=np.float64)
        gap_seconds = float(getattr(args, "missing_gap_seconds", 0.0) or 0.0)
        placement = str(getattr(args, "missing_gap_placement", "center") or "center")
        if placement == "max_abs_net_charge":
            freeze_mask = compute_max_abs_net_charge_window_mask(
                t,
                out["Current[A]"].to_numpy(dtype=np.float64),
                gap_seconds,
                min_pre_seconds=float(getattr(args, "missing_gap_min_pre_seconds", 0.0) or 0.0),
                min_post_seconds=float(getattr(args, "missing_gap_min_post_seconds", 0.0) or 0.0),
            )
        else:
            freeze_mask = compute_center_window_mask(t, gap_seconds)
        meta["freeze_mask"] = freeze_mask
        meta["disturbance_mask"] = freeze_mask.copy()
        meta["missing_gap_seconds"] = gap_seconds
        meta["missing_gap_placement"] = placement
        meta["missing_gap_min_pre_seconds"] = float(
            getattr(args, "missing_gap_min_pre_seconds", 0.0) or 0.0
        )
        meta["missing_gap_min_post_seconds"] = float(
            getattr(args, "missing_gap_min_post_seconds", 0.0) or 0.0
        )
        if np.any(freeze_mask):
            gap_indices = np.flatnonzero(freeze_mask)
            meta["missing_gap_start_time_s"] = float(t[gap_indices[0]])
            meta["missing_gap_end_time_s"] = float(t[gap_indices[-1]])
        return out, meta

    if scenario == "missing_samples":
        every = getattr(args, "missing_samples_every", None)
        pct = float(getattr(args, "missing_samples_pct", 0.0) or 0.0)
        if every and int(every) > 1:
            freeze_mask[np.arange(int(every) - 1, n, int(every))] = True
        elif pct > 0.0:
            count = int(round(n * pct))
            if count > 0:
                idx = rng.choice(n, size=min(n, count), replace=False)
                freeze_mask[idx] = True
        meta["freeze_mask"] = freeze_mask
        meta["disturbance_mask"] = freeze_mask.copy()
        meta["missing_samples_every"] = every
        meta["missing_samples_pct"] = pct
        return out, meta

    if scenario == "irregular_sampling":
        if "Testtime[s]" not in out.columns or len(out) < 2:
            return out, meta
        jitter = float(getattr(args, "irregular_dt_jitter", 0.0) or 0.0)
        out, resample_meta = _resample_irregular_timebase(out, jitter_s=jitter, rng=rng)
        disturbance_mask[:] = True
        meta["disturbance_mask"] = disturbance_mask
        meta["irregular_dt_jitter"] = jitter
        meta["nominal_dt_s"] = float(resample_meta["nominal_dt_s"])
        meta["resampled_span_s"] = float(resample_meta["resampled_span_s"])
        meta["resampled_measurements"] = True
        return out, meta

    raise ValueError(f"Unknown scenario: {scenario}")


def build_online_aux_features(
    df: pd.DataFrame,
    freeze_mask: np.ndarray,
    current_sign: float,
    v_max: float,
    v_tol: float,
    cv_seconds: float,
    nominal_capacity_ah: float,
    initial_soc_delta: float = 0.0,
    q_c_reset_voltage_v: float = 3.6002,
    q_c_reset_current_a: float = 0.1,
    q_c_capacity_ah: Optional[float] = None,
) -> pd.DataFrame:
    out = df.copy()
    if "_protocol_current_a" not in out.columns and "Current[A]" in out.columns:
        out["_protocol_current_a"] = out["Current[A]"].to_numpy(dtype=np.float64, copy=True)
    has_freeze = bool(np.any(freeze_mask))
    base_cols = [c for c in ["Current[A]", "Voltage[V]", "Temperature[°C]"] if c in out.columns]
    if has_freeze:
        for c in base_cols:
            out.loc[freeze_mask, c] = np.nan
            out[c] = out[c].ffill().bfill()

    t = out["Testtime[s]"].to_numpy(dtype=np.float64)
    i = out["Current[A]"].to_numpy(dtype=np.float64)
    v = out["Voltage[V]"].to_numpy(dtype=np.float64)

    dt_s = np.diff(t, prepend=t[0])
    dt_s[dt_s < 0] = 0.0
    if has_freeze:
        nominal_dt = np.median(dt_s[(~freeze_mask) & (dt_s > 0)])
        if not np.isfinite(nominal_dt) or nominal_dt <= 0:
            nominal_dt = 1.0
        dt_s[freeze_mask] = 0.0
        for k in range(1, len(dt_s)):
            if freeze_mask[k - 1] and not freeze_mask[k]:
                dt_s[k] = nominal_dt

    di = np.diff(i, prepend=i[0])
    du = np.diff(v, prepend=v[0])
    d_i_dt = np.zeros(len(i), dtype=np.float64)
    d_u_dt = np.zeros(len(v), dtype=np.float64)
    valid = dt_s > 0
    d_i_dt[valid] = di[valid] / dt_s[valid]
    d_u_dt[valid] = du[valid] / dt_s[valid]

    q_c = np.zeros(len(i), dtype=np.float64)
    efc = np.zeros(len(i), dtype=np.float64)
    cap_ref = max(float(nominal_capacity_ah), 1e-9)
    q_c_cap = max(float(q_c_capacity_ah if q_c_capacity_ah is not None else cap_ref), 1e-9)
    q_now = float(initial_soc_delta) * cap_ref
    throughput_ah = 0.0
    for k in range(len(i)):
        dt = float(dt_s[k])
        q_now += float(current_sign) * float(i[k]) * dt / 3600.0
        if v[k] >= float(q_c_reset_voltage_v) and i[k] > float(q_c_reset_current_a):
            q_now = 0.0
        q_now = min(0.0, max(-q_c_cap, q_now))
        throughput_ah += abs(float(i[k])) * dt / 3600.0
        q_c[k] = q_now
        efc[k] = throughput_ah / cap_ref

    out["Q_c"] = q_c
    out["EFC"] = efc
    out["dI_dt[A/s]"] = d_i_dt
    out["dU_dt[V/s]"] = d_u_dt
    out["_dt_s_online"] = dt_s
    return out


def compute_robustness_metrics(
    time_s: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    warmup_seconds: float = 0.0,
    disturbance_mask: np.ndarray = None,
    jump_threshold: float = 0.05,
) -> Dict[str, float]:
    t = np.asarray(time_s, dtype=np.float64)
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    abs_err = np.abs(yp - yt)
    err = yp - yt
    metrics: Dict[str, float] = {}

    warm_mask = t >= float(warmup_seconds)
    if not np.any(warm_mask):
        warm_mask = np.ones(len(t), dtype=bool)

    t_w = t[warm_mask]
    yp_w = yp[warm_mask]
    err_w = err[warm_mask]
    abs_w = abs_err[warm_mask]

    metrics["rmse"] = float(np.sqrt(np.mean(err_w ** 2)))
    metrics["mae"] = float(np.mean(abs_w))
    metrics["bias"] = float(np.mean(err_w))
    metrics["max_error"] = float(np.max(abs_w))
    metrics["p95_error"] = float(np.percentile(abs_w, 95.0))
    jumps = np.abs(np.diff(yp_w, prepend=yp_w[0]))
    metrics["jump_count_gt_5pct"] = int(np.sum(jumps > jump_threshold))
    metrics["output_variance"] = float(np.var(yp_w))
    metrics["abs_error_variance"] = float(np.var(abs_w))
    metrics["drift_rate_soc_per_h"] = _fit_slope_per_hour(t_w, yp_w)
    metrics["drift_rate_abs_err_per_h"] = _fit_slope_per_hour(t_w, abs_w)

    if disturbance_mask is not None and len(disturbance_mask) == len(t):
        dm = np.asarray(disturbance_mask, dtype=bool)
        dm_w = dm[warm_mask]
        metrics["disturbed_fraction"] = float(np.mean(dm_w)) if len(dm_w) else 0.0
        if np.any(dm_w):
            metrics["disturbed_mae"] = float(np.mean(abs_w[dm_w]))
            metrics["disturbed_rmse"] = float(np.sqrt(np.mean(err_w[dm_w] ** 2)))
            metrics["disturbed_max_error"] = float(np.max(abs_w[dm_w]))
        calm_pre = warm_mask & (~dm)
        if np.any(calm_pre):
            calm_abs = abs_err[calm_pre]
            metrics["calm_mae"] = float(np.mean(calm_abs))
        rec = _recovery_metrics(t, abs_err, dm, float(warmup_seconds))
        metrics.update(rec)

    return metrics


def compute_stratified_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference_soh: Optional[np.ndarray] = None,
    reference_temperature_c: Optional[np.ndarray] = None,
    reference_c_rate: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Compute run-level metrics for fixed physical-state strata."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) != len(yp):
        raise ValueError("Stratified metrics require aligned targets and predictions")

    strata: list[tuple[str, str, np.ndarray]] = [
        ("soc_state", "low", yt < 0.20),
        ("soc_state", "middle", (yt >= 0.20) & (yt <= 0.80)),
        ("soc_state", "high", yt > 0.80),
    ]
    if reference_soh is not None:
        soh = np.asarray(reference_soh, dtype=np.float64)
        strata.extend([
            ("soh_state", "aged", soh < 0.80),
            ("soh_state", "mid_life", (soh >= 0.80) & (soh < 0.90)),
            ("soh_state", "fresh", soh >= 0.90),
        ])
    if reference_temperature_c is not None:
        temperature = np.asarray(reference_temperature_c, dtype=np.float64)
        strata.extend([
            ("temperature_state", "nominal", temperature <= 30.0),
            ("temperature_state", "elevated", (temperature > 30.0) & (temperature <= 35.0)),
            ("temperature_state", "hot", temperature > 35.0),
        ])
    if reference_c_rate is not None:
        c_rate = np.abs(np.asarray(reference_c_rate, dtype=np.float64))
        strata.extend([
            ("instantaneous_load", "low", c_rate < 0.5),
            ("instantaneous_load", "medium", (c_rate >= 0.5) & (c_rate < 1.5)),
            ("instantaneous_load", "high", c_rate >= 1.5),
        ])

    finite_error = np.isfinite(yt) & np.isfinite(yp)
    rows: List[Dict[str, Any]] = []
    for dimension, stratum, mask in strata:
        mask = np.asarray(mask, dtype=bool) & finite_error
        n_samples = int(mask.sum())
        if n_samples == 0:
            continue
        error = yp[mask] - yt[mask]
        absolute = np.abs(error)
        rows.append({
            "dimension": dimension,
            "stratum": stratum,
            "n_samples": n_samples,
            "coverage_fraction": float(n_samples / max(int(finite_error.sum()), 1)),
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
            "p95_error": float(np.percentile(absolute, 95.0)),
        })
    return rows


def compute_common_recovery_metrics(
    time_s: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    start_index: int,
    threshold: float = 0.02,
    sustain_seconds: float = 300.0,
    horizon_seconds: float = 86400.0,
) -> Dict[str, float]:
    """Measure recovery with one estimator-independent absolute error band."""
    t = np.asarray(time_s, dtype=np.float64)
    abs_err = np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
    if len(t) == 0 or len(t) != len(abs_err):
        return {}

    start_index = int(np.clip(start_index, 0, len(t) - 1))
    horizon_end = float(t[start_index]) + float(horizon_seconds)
    stop_index = int(np.searchsorted(t, horizon_end, side="right"))
    stop_index = max(start_index + 1, min(stop_index, len(t)))
    t_post = t[start_index:stop_index]
    err_post = abs_err[start_index:stop_index]
    valid = err_post <= float(threshold)

    recovery_time_s = None
    recovery_index = None
    candidates = np.flatnonzero(valid)
    if len(candidates):
        bad = np.flatnonzero(~valid)
        bad_positions = np.searchsorted(bad, candidates, side="left")
        next_bad = np.full(len(candidates), len(valid), dtype=np.int64)
        has_bad = bad_positions < len(bad)
        next_bad[has_bad] = bad[bad_positions[has_bad]]
        sustain_end = np.searchsorted(
            t_post,
            t_post[candidates] + float(sustain_seconds),
            side="left",
        )
        full_window = (t_post[candidates] + float(sustain_seconds)) <= t_post[-1]
        recovered = full_window & (next_bad >= sustain_end)
        if np.any(recovered):
            first = int(candidates[np.flatnonzero(recovered)[0]])
            recovery_index = first
            recovery_time_s = float(t_post[first] - t_post[0])

    relapse_index = None
    if recovery_index is not None:
        later_bad = np.flatnonzero(~valid[recovery_index:])
        if len(later_bad):
            relapse_index = int(recovery_index + later_bad[0])

    bad = np.flatnonzero(~valid)
    stable_start = 0 if len(bad) == 0 else int(bad[-1] + 1)
    stable_recovery_time_s = None
    if stable_start < len(valid):
        stable_duration = float(t_post[-1] - t_post[stable_start])
        if valid[stable_start] and stable_duration >= float(sustain_seconds):
            stable_recovery_time_s = float(t_post[stable_start] - t_post[0])

    elapsed_h = (t_post - t_post[0]) / 3600.0
    observed_horizon_s = float(t_post[-1] - t_post[0])
    capped_time_s = (
        recovery_time_s
        if recovery_time_s is not None
        else min(float(horizon_seconds), observed_horizon_s)
    )
    stable_capped_time_s = (
        stable_recovery_time_s
        if stable_recovery_time_s is not None
        else min(float(horizon_seconds), observed_horizon_s)
    )
    excess = np.maximum(err_post - float(threshold), 0.0)
    metrics: Dict[str, float] = {
        "common_recovery_threshold_abs_err": float(threshold),
        "common_recovery_sustain_seconds": float(sustain_seconds),
        "common_recovery_horizon_seconds": float(horizon_seconds),
        "common_recovery_initial_abs_err": float(err_post[0]),
        "common_recovery_excess_auc_soc_h": float(np.trapz(excess, elapsed_h)) if len(excess) > 1 else 0.0,
        "common_recovery_time_s": recovery_time_s,
        "common_recovery_time_h": None if recovery_time_s is None else recovery_time_s / 3600.0,
        "common_recovery_or_censor_time_h": capped_time_s / 3600.0,
        "common_recovery_observed_horizon_h": observed_horizon_s / 3600.0,
        "common_recovery_censored": recovery_time_s is None,
        "common_recovery_relapsed": relapse_index is not None,
        "common_recovery_first_relapse_time_h": (
            None if relapse_index is None else float(t_post[relapse_index] - t_post[0]) / 3600.0
        ),
        "common_stable_recovery_time_h": (
            None if stable_recovery_time_s is None else stable_recovery_time_s / 3600.0
        ),
        "common_stable_recovery_or_censor_time_h": stable_capped_time_s / 3600.0,
        "common_stable_recovery_censored": stable_recovery_time_s is None,
    }
    for hours in (1.0, 6.0):
        mask = elapsed_h <= hours
        if np.any(mask):
            metrics[f"common_recovery_mae_{int(hours)}h"] = float(np.mean(err_post[mask]))
    return metrics


def compute_protocol_event_metrics(
    scenario: str,
    time_s: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_a: np.ndarray,
    freeze_mask: np.ndarray,
    threshold: float,
    sustain_seconds: float,
    horizon_seconds: float,
) -> Dict[str, float]:
    """Return fair recovery and physical gap metrics for JES2 event scenarios."""
    t = np.asarray(time_s, dtype=np.float64)
    current = np.asarray(current_a, dtype=np.float64)
    freeze = np.asarray(freeze_mask, dtype=bool)
    if scenario == "initial_soc_error":
        return compute_common_recovery_metrics(
            t, y_true, y_pred, 0, threshold, sustain_seconds, horizon_seconds
        )
    if scenario != "missing_gap" or not np.any(freeze):
        return {}

    gap_indices = np.flatnonzero(freeze)
    first_gap = int(gap_indices[0])
    last_gap = int(gap_indices[-1])
    recovery_start = min(last_gap + 1, len(t) - 1)
    dt_s = np.diff(t, prepend=t[0])
    dt_s[dt_s < 0.0] = 0.0
    gap_dt = dt_s[freeze]
    gap_current = current[freeze]
    pre_gap = max(0, first_gap - 1)
    event = {
        "gap_net_charge_ah": float(np.sum(gap_current * gap_dt) / 3600.0),
        "gap_throughput_ah": float(np.sum(np.abs(gap_current) * gap_dt) / 3600.0),
        "gap_reference_soc_change": float(np.asarray(y_true)[last_gap] - np.asarray(y_true)[pre_gap]),
        "gap_start_time_s": float(t[first_gap]),
        "gap_end_time_s": float(t[last_gap]),
    }
    event.update(
        compute_common_recovery_metrics(
            t,
            y_true,
            y_pred,
            recovery_start,
            threshold,
            sustain_seconds,
            horizon_seconds,
        )
    )
    return event


def _fit_slope_per_hour(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    t_h = (t - t[0]) / 3600.0
    if np.allclose(t_h, 0.0):
        return 0.0
    slope = np.polyfit(t_h, y, 1)[0]
    return float(slope)


def _recovery_metrics(t: np.ndarray, abs_err: np.ndarray, disturbance_mask: np.ndarray, warmup_seconds: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    dm = np.asarray(disturbance_mask, dtype=bool)
    if not np.any(dm):
        return out
    idx = np.flatnonzero(dm)
    end_idx = int(idx[-1])
    start_idx = int(idx[0])
    post_mask = np.arange(len(t)) > end_idx
    pre_mask = (np.arange(len(t)) < start_idx) & (t >= warmup_seconds)
    if not np.any(post_mask):
        return out

    baseline_mae = float(np.mean(abs_err[pre_mask])) if np.any(pre_mask) else float(np.mean(abs_err[t >= warmup_seconds]))
    threshold = max(baseline_mae * 1.2, 1e-6)
    out["pre_disturbance_mae"] = baseline_mae
    out["recovery_threshold_abs_err"] = threshold

    post_err = abs_err[post_mask]
    post_t = t[post_mask]
    out["post_disturbance_mae"] = float(np.mean(post_err))
    out["post_disturbance_rmse"] = float(math.sqrt(np.mean(post_err ** 2)))

    window = min(300, len(post_err))
    if window >= 5:
        kernel = np.ones(window, dtype=np.float64) / float(window)
        smoothed = np.convolve(post_err, kernel, mode="same")
    else:
        smoothed = post_err

    rec_idx = np.flatnonzero(smoothed <= threshold)
    if len(rec_idx):
        rec0 = int(rec_idx[0])
        out["recovery_time_s"] = float(post_t[rec0] - t[end_idx])
        out["recovery_time_h"] = float((post_t[rec0] - t[end_idx]) / 3600.0)
        out["residual_error_after_recovery"] = float(smoothed[rec0])
    else:
        out["recovery_time_s"] = None
        out["recovery_time_h"] = None
        out["residual_error_after_recovery"] = None
    return out
