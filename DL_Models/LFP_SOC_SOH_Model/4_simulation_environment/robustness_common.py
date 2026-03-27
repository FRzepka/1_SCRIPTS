import math
import os
from typing import Dict, Tuple

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


def load_cell_dataframe(data_root: str, cell: str) -> pd.DataFrame:
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
    return pd.read_parquet(path)


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
    ap.add_argument("--missing_samples_every", type=int, default=None,
                    help="Freeze every Nth sample.")
    ap.add_argument("--missing_samples_pct", type=float, default=None,
                    help="Randomly freeze this fraction of samples.")
    ap.add_argument("--irregular_dt_jitter", type=float, default=None,
                    help="Uniform +/- jitter in seconds added to each sampling interval.")
    ap.add_argument("--downsample_hz", type=float, default=None)
    ap.add_argument("--drop_pct", type=float, default=None)
    ap.add_argument("--drop_segment_len", type=int, default=None)


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
        freeze_mask = compute_center_window_mask(t, float(getattr(args, "missing_gap_seconds", 0.0) or 0.0))
        meta["freeze_mask"] = freeze_mask
        meta["disturbance_mask"] = freeze_mask.copy()
        meta["missing_gap_seconds"] = float(getattr(args, "missing_gap_seconds", 0.0) or 0.0)
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
) -> pd.DataFrame:
    out = df.copy()
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
    q_now = float(initial_soc_delta) * cap_ref
    cv_now = 0.0
    throughput_ah = 0.0
    v_thr = float(v_max) - float(v_tol)
    for k in range(len(i)):
        dt = float(dt_s[k])
        if v[k] >= v_thr:
            cv_now += dt
        else:
            cv_now = 0.0
        if cv_now >= float(cv_seconds):
            q_now = 0.0
        else:
            q_now += float(current_sign) * float(i[k]) * dt / 3600.0
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
