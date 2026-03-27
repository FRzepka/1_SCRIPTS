import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIM_ENV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(SIM_ENV_DIR)
CC_SOH_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "2_models",
        "CC_SOH_1.0.0",
    )
)
sys.path.append(CC_SOH_DIR)
from cc_soh_model import CCSOHConfig, CCSOHModel
from fast_ecm import FastBatteryEKF, FastECMTable
from robustness_common import (
    add_common_scenario_args,
    apply_measurement_scenario,
    build_online_aux_features,
    compute_robustness_metrics,
    load_cell_dataframe,
)

WEEK_SECONDS = 7 * 24 * 3600
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _progress(prefix: str, idx: int, total: int, start_s: float, extra: str = ""):
    if total <= 0:
        return
    frac = idx / total
    elapsed = max(time.time() - start_s, 1e-9)
    rate = idx / elapsed if idx > 0 else 0.0
    eta_s = (total - idx) / rate if rate > 0 else float("inf")
    eta_min = eta_s / 60.0 if np.isfinite(eta_s) else float("inf")
    msg = f"[{prefix}] {frac * 100:5.1f}% ({idx}/{total}) rate={rate:,.0f} step/s"
    if np.isfinite(eta_min):
        msg += f" eta={eta_min:,.1f} min"
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)


def _soh_cache_path(cell: str, args) -> Path:
    key = "|".join([
        cell,
        str(Path(args.soh_ckpt).resolve()),
        str(Path(args.soh_scaler).resolve()),
        f"cap={args.capacity_ah:.6f}",
        f"hold={args.soh_hold_seconds}",
    ])
    digest = hashlib.md5(key.encode("ascii", "ignore")).hexdigest()[:12]
    return CACHE_DIR / f"{cell}_soh_weekly_{digest}.npz"


def compute_or_load_weekly_soh(raw_df: pd.DataFrame, cell: str, args):
    cache_path = _soh_cache_path(cell, args)
    if cache_path.exists():
        cache = np.load(cache_path)
        print(f"[SOH] using cached weekly SOH: {cache_path}", flush=True)
        return {
            "week_bins": cache["week_bins"].astype(np.int64),
            "week_soh": cache["week_soh"].astype(np.float64),
            "cache_path": str(cache_path),
            "cache_hit": True,
        }

    print("[SOH] computing weekly SOH cache from raw baseline measurements", flush=True)
    freeze_mask = np.zeros(len(raw_df), dtype=bool)
    raw_aux = build_online_aux_features(
        df=raw_df.copy(),
        freeze_mask=freeze_mask,
        current_sign=float(args.current_sign),
        v_max=float(args.v_max),
        v_tol=float(args.v_tol),
        cv_seconds=float(args.cv_seconds),
        nominal_capacity_ah=float(args.capacity_ah),
        initial_soc_delta=0.0,
    )
    cfg = CCSOHConfig(
        soh_config=args.soh_config,
        soh_checkpoint=args.soh_ckpt,
        soh_scaler=args.soh_scaler,
        nominal_capacity_ah=float(args.capacity_ah),
        soh_init=float(args.soh_const),
        device=args.device,
        soc_init=float(args.soc_init),
        current_sign=float(args.current_sign),
        v_max=float(args.v_max),
        v_tol=float(args.v_tol),
        cv_seconds=float(args.cv_seconds),
    )
    soh_model = CCSOHModel(cfg)
    if args.require_gpu and soh_model.device.type != "cuda":
        raise RuntimeError("GPU required (--require_gpu), but CUDA is not available.")
    soh_hourly, bins = soh_model.predict_soh_hourly(raw_aux)
    soh_per_row = soh_model.expand_soh_to_rows(raw_aux, bins, soh_hourly)

    t_raw = raw_aux["Testtime[s]"].to_numpy(dtype=np.float64)
    week_index = np.floor_divide(t_raw.astype(np.int64), int(args.soh_hold_seconds)).astype(np.int64)
    unique_weeks = np.unique(week_index)
    week_soh = np.zeros(len(unique_weeks), dtype=np.float64)
    for j, week in enumerate(unique_weeks):
        mask = week_index == week
        week_soh[j] = float(np.median(soh_per_row[mask]))

    np.savez_compressed(cache_path, week_bins=unique_weeks, week_soh=week_soh)
    print(f"[SOH] saved weekly SOH cache: {cache_path}", flush=True)
    return {
        "week_bins": unique_weeks,
        "week_soh": week_soh,
        "cache_path": str(cache_path),
        "cache_hit": False,
    }


def expand_weekly_soh_to_rows(time_s: np.ndarray, week_bins: np.ndarray, week_soh: np.ndarray, hold_seconds: int) -> np.ndarray:
    max_week = int(max(np.max(week_bins), np.max(np.floor_divide(time_s.astype(np.int64), hold_seconds))))
    lookup = np.zeros(max_week + 1, dtype=np.float64)
    last = float(week_soh[0])
    src = {int(w): float(s) for w, s in zip(week_bins, week_soh)}
    for week in range(max_week + 1):
        if week in src:
            last = src[week]
        lookup[week] = last
    row_bins = np.floor_divide(time_s.astype(np.int64), hold_seconds).astype(np.int64)
    return lookup[row_bins]


def main():
    ap = argparse.ArgumentParser(description="Run ECM_0.0.3 scenario (fast ECM + cached shared SOH).")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--data_root", default="/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--soc_init", type=float, default=1.0)
    ap.add_argument("--capacity_ah", type=float, default=1.8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--require_gpu", action="store_true")
    ap.add_argument("--dt_mode", choices=["data", "fixed"], default="data")
    ap.add_argument("--dt_s", type=float, default=1.0)
    ap.add_argument("--max_rows", type=int, default=0, help="Limit rows (0 = full)")
    ap.add_argument("--downsample", type=int, default=1, help="Take every Nth row (>=1)")
    ap.add_argument("--warmup_seconds", type=float, default=600.0)
    ap.add_argument("--soh_mode", choices=["gru", "data", "const"], default="gru")
    ap.add_argument("--soh_const", type=float, default=1.0)
    ap.add_argument("--soh_hold_seconds", type=int, default=WEEK_SECONDS)
    ap.add_argument("--soh_config", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/train_soh.yaml")
    ap.add_argument("--soh_ckpt", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/best_epoch0093_rmse0.02165.pt")
    ap.add_argument("--soh_scaler", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/scaler_robust.joblib")
    ap.add_argument("--current_sign", type=float, default=1.0)
    ap.add_argument("--v_max", type=float, default=3.65)
    ap.add_argument("--v_tol", type=float, default=0.02)
    ap.add_argument("--cv_seconds", type=float, default=300.0)
    add_common_scenario_args(ap)
    args = ap.parse_args()

    np.random.seed(int(args.seed))
    load_start = time.time()
    raw_df = load_cell_dataframe(args.data_root, args.cell)
    raw_df = raw_df.replace([np.inf, -np.inf], np.nan)
    print(f"[LOAD] rows={len(raw_df):,} hours={raw_df['Testtime[s]'].iloc[-1] / 3600.0:.1f} load_s={time.time() - load_start:.1f}", flush=True)

    soh_source = "const"
    soh_cache_info = {"cache_path": None, "cache_hit": False}
    if args.soh_mode == "gru":
        soh_cache_info = compute_or_load_weekly_soh(raw_df, args.cell, args)
        soh_source = f"shared_recurrent_weekly_cached::{Path(args.soh_config).resolve().parent.name}"
    
    df, scenario_info = apply_measurement_scenario(raw_df, args.scenario, args)
    df = df.dropna(subset=["Testtime[s]", "Current[A]", "Voltage[V]", "SOC"]).reset_index(drop=True)
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    if args.downsample and args.downsample > 1:
        df = df.iloc[::args.downsample].reset_index(drop=True)

    t = df["Testtime[s]"].to_numpy(dtype=np.float64)
    if args.dt_mode == "data":
        dt_s = np.diff(t, prepend=t[0])
        dt_s[dt_s < 0] = 0.0
    else:
        dt_s = np.full(len(df), float(args.dt_s), dtype=np.float64)

    freeze_mask = np.asarray(scenario_info.get("freeze_mask", np.zeros(len(df), dtype=bool)), dtype=bool)
    has_gap = bool(np.any(freeze_mask))
    if has_gap:
        nominal_dt = np.median(dt_s[(~freeze_mask) & (dt_s > 0)])
        if not np.isfinite(nominal_dt) or nominal_dt <= 0:
            nominal_dt = 1.0
        dt_s[freeze_mask] = 0.0
        for k in range(1, len(dt_s)):
            if freeze_mask[k - 1] and not freeze_mask[k]:
                dt_s[k] = nominal_dt

    if args.soh_mode == "gru":
        soh = expand_weekly_soh_to_rows(
            time_s=t,
            week_bins=soh_cache_info["week_bins"],
            week_soh=soh_cache_info["week_soh"],
            hold_seconds=int(args.soh_hold_seconds),
        )
    elif args.soh_mode == "data" and "SOH" in df.columns:
        soh = df["SOH"].to_numpy(dtype=np.float64)
        soh_source = "data"
    else:
        soh = np.full(len(df), float(args.soh_const), dtype=np.float64)

    if has_gap and np.any(freeze_mask):
        first_gap = int(np.argmax(freeze_mask))
        hold = float(soh[first_gap - 1]) if first_gap > 0 else float(args.soh_const)
        soh[freeze_mask] = hold

    table = FastECMTable()
    model = FastBatteryEKF(float(soh[0]), table=table)
    model.x[0] = float(np.clip(float(args.soc_init) + float(scenario_info.get("soc_init_delta", 0.0)), 0.0, 1.0))
    soc_true = df["SOC"].to_numpy(dtype=np.float64)
    u_true = df["Voltage[V]"].to_numpy(dtype=np.float64)
    i = df["Current[A]"].to_numpy(dtype=np.float64)

    print(
        f"[SETUP] scenario={args.scenario} rows={len(df):,} weekly_soh_bins={len(np.unique(np.floor_divide(t.astype(np.int64), int(args.soh_hold_seconds)))):,} cache_hit={soh_cache_info.get('cache_hit', False)}",
        flush=True,
    )

    soc_est = np.zeros(len(df), dtype=np.float64)
    u_est = np.zeros(len(df), dtype=np.float64)
    loop_start = time.time()
    progress_every = max(1, len(df) // 100)
    for k in range(len(df)):
        if has_gap and freeze_mask[k]:
            soc_est[k] = float(np.clip(float(args.soc_init) + float(scenario_info.get("soc_init_delta", 0.0)), 0.0, 1.0)) if k == 0 else soc_est[k - 1]
            u_est[k] = u_true[k] if k == 0 else u_est[k - 1]
        else:
            model.soh = float(soh[k])
            model.Cb = model.C0 * model.soh
            model.deltaT = float(dt_s[k] if args.dt_mode == "data" else args.dt_s)
            x_k, _, y_k = model.predict_update(float(i[k]), float(u_true[k]))
            soc_est[k] = float(x_k[0])
            u_est[k] = float(y_k)
        if (k + 1) % progress_every == 0 or (k + 1) == len(df):
            _progress("ECM_0.0.3", k + 1, len(df), loop_start, extra=f"scenario={args.scenario}")

    abs_err = np.abs(soc_true - soc_est)
    metrics = compute_robustness_metrics(
        time_s=t,
        y_true=soc_true,
        y_pred=soc_est,
        warmup_seconds=float(args.warmup_seconds),
        disturbance_mask=np.asarray(scenario_info.get("disturbance_mask", freeze_mask), dtype=bool),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(
        {
            "index": np.arange(len(df)),
            "time_s": t,
            "I": i,
            "U": u_true,
            "SOH": soh,
            "soc_true": soc_true,
            "soc_ecm": soc_est,
            "u_ecm": u_est,
            "abs_err": abs_err,
        }
    )
    out_df.to_csv(out_dir / f"ecm_soc_fullcell_{args.cell}.csv", index=False)

    mask = out_df["time_s"] >= float(args.warmup_seconds)
    summary = {
        "model": "ECM_0.0.3",
        "cell": args.cell,
        "scenario": args.scenario,
        "soc_init": float(np.clip(float(args.soc_init) + float(scenario_info.get("soc_init_delta", 0.0)), 0.0, 1.0)),
        "capacity_ah": float(args.capacity_ah),
        "dt_mode": args.dt_mode,
        "dt_s": float(args.dt_s),
        "max_rows": int(args.max_rows),
        "downsample": int(args.downsample),
        "warmup_seconds": float(args.warmup_seconds),
        "missing_gap_seconds": float(args.missing_gap_seconds),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "engine": "ECM_v2_qinnan_fast_python",
        "soh_mode": args.soh_mode,
        "soh_source": soh_source,
        "soh_hold_seconds": int(args.soh_hold_seconds),
        "soh_cache_path": soh_cache_info.get("cache_path"),
        "soh_cache_hit": bool(soh_cache_info.get("cache_hit", False)),
        "scenario_meta": {k: v for k, v in scenario_info.items() if k not in ("freeze_mask", "disturbance_mask")},
    }
    summary.update(metrics)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(out_df["time_s"] / 3600.0, out_df["soc_true"], label="SOC true", linewidth=1.0)
    ax1.plot(out_df["time_s"] / 3600.0, out_df["soc_ecm"], label="SOC ECM", linewidth=1.0, alpha=0.8)
    ax1.set_title(f"ECM_0.0.3 – Full Cell ({args.cell}) [{args.scenario}]")
    ax1.set_ylabel("SOC")
    ax1.legend(loc="best")
    fig.text(0.12, 0.93, f"MAE: {summary['mae']:.5f} | RMSE: {summary['rmse']:.5f} | P95: {summary['p95_error']:.5f}", fontsize=13,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    ax2.plot(out_df.loc[mask, "time_s"] / 3600.0, out_df.loc[mask, "abs_err"], label="Absolute Error", linewidth=1.0, color="tab:red")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Abs Error")
    ax2.set_ylim(0.0, 0.4)
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"ecm_soc_fullcell_{args.cell}.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
