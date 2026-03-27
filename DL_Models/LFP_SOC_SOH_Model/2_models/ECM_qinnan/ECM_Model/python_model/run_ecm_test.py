import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def apply_scenario(df: pd.DataFrame, scenario: str, args) -> pd.DataFrame:
    out = df.copy()
    if scenario in ('baseline', 'missing_gap'):
        return out
    if scenario == 'current_offset':
        if args.current_offset_a is not None:
            out['Current[A]'] = out['Current[A]'] + float(args.current_offset_a)
        elif args.current_offset_pct is not None:
            out['Current[A]'] = out['Current[A]'] * (1.0 + float(args.current_offset_pct))
        return out
    if scenario == 'current_noise':
        std = float(args.current_noise_std or 0.0)
        out['Current[A]'] = out['Current[A]'] + np.random.normal(0.0, std, size=len(out))
        return out
    if scenario == 'voltage_offset':
        out['Voltage[V]'] = out['Voltage[V]'] + float(args.voltage_offset_v or 0.0)
        return out
    if scenario == 'voltage_noise':
        std = float(args.voltage_noise_std or 0.0)
        out['Voltage[V]'] = out['Voltage[V]'] + np.random.normal(0.0, std, size=len(out))
        return out
    if scenario == 'temp_mask':
        if args.temp_constant is not None and 'Temperature[°C]' in out.columns:
            out['Temperature[°C]'] = float(args.temp_constant)
        return out
    if scenario == 'downsample':
        if 'Testtime[s]' not in out.columns:
            return out
        dt = out['Testtime[s]'].diff().median()
        if not np.isfinite(dt) or dt <= 0:
            return out
        orig_hz = 1.0 / dt
        target_hz = float(args.downsample_hz or 1.0)
        stride = max(1, int(round(orig_hz / target_hz)))
        return out.iloc[::stride].reset_index(drop=True)
    if scenario == 'missing_segments':
        drop_pct = float(args.drop_pct or 0.1)
        seg_len = int(args.drop_segment_len or 1000)
        if drop_pct <= 0:
            return out
        n = len(out)
        to_drop = set()
        n_drop = int(n * drop_pct)
        rng = np.random.default_rng(int(args.seed or 42))
        while len(to_drop) < n_drop:
            start = int(rng.integers(0, max(1, n - seg_len)))
            for i in range(start, min(n, start + seg_len)):
                to_drop.add(i)
                if len(to_drop) >= n_drop:
                    break
        keep_idx = [i for i in range(n) if i not in to_drop]
        return out.iloc[keep_idx].reset_index(drop=True)
    raise ValueError(f"Unknown scenario: {scenario}")


def compute_missing_gap_mask(t: np.ndarray, gap_seconds: float) -> np.ndarray:
    if gap_seconds is None or gap_seconds <= 0:
        return np.zeros(len(t), dtype=bool)
    t0 = float(t[0])
    t1 = float(t[-1])
    span = t1 - t0
    if span <= gap_seconds:
        return np.zeros(len(t), dtype=bool)
    start = t0 + (span - gap_seconds) * 0.5
    end = start + gap_seconds
    return (t >= start) & (t <= end)


def main():
    ap = argparse.ArgumentParser(description="Run ECM EKF (python) on FE dataset.")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--data_root", default="/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--soc_init", type=float, default=1.0)
    ap.add_argument("--capacity_ah", type=float, default=1.7838)
    ap.add_argument("--dt_mode", choices=["data", "fixed"], default="data")
    ap.add_argument("--dt_s", type=float, default=60.0)
    ap.add_argument("--max_rows", type=int, default=0, help="Limit rows (0 = full)")
    ap.add_argument("--downsample", type=int, default=1, help="Take every Nth row (>=1)")
    ap.add_argument("--warmup_seconds", type=float, default=600.0)
    ap.add_argument("--missing_gap_seconds", type=float, default=0.0,
                    help="If >0, create a single missing-data gap (freeze) in the middle.")
    ap.add_argument("--update_covariance", action="store_true")
    ap.add_argument("--ecm_param_c", default=None, help="Path to ECM_parameter.c")
    ap.add_argument("--native", action="store_true", help="Use native C EKF (battery_ekf.c)")
    ap.add_argument("--native_lib", default=None, help="Path to libecm_ekf.so/libecm_ekf_new.so (native)")
    # Default to const to avoid leakage from dataset-only SOH labels.
    ap.add_argument("--soh_mode", choices=["data", "const"], default="const")
    ap.add_argument("--soh_const", type=float, default=1.0)
    ap.add_argument("--scenario", default="baseline",
                    choices=["baseline", "missing_gap", "current_offset", "current_noise", "voltage_offset", "voltage_noise", "temp_mask", "downsample", "missing_segments"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--current_offset_a", type=float, default=None)
    ap.add_argument("--current_offset_pct", type=float, default=None)
    ap.add_argument("--current_noise_std", type=float, default=None)
    ap.add_argument("--voltage_offset_v", type=float, default=None)
    ap.add_argument("--voltage_noise_std", type=float, default=None)
    ap.add_argument("--temp_constant", type=float, default=None)
    ap.add_argument("--downsample_hz", type=float, default=None)
    ap.add_argument("--drop_pct", type=float, default=None)
    ap.add_argument("--drop_segment_len", type=int, default=None)
    args = ap.parse_args()

    ecm_param_c = args.ecm_param_c
    if ecm_param_c is None:
        ecm_param_c = os.path.join(os.path.dirname(__file__), "..", "ECM_parameter.c")
    ecm_param_c = os.path.abspath(ecm_param_c)

    native_lib = args.native_lib
    if native_lib is None:
        root = Path(__file__).resolve().parent.parent
        cand_new = root / "libecm_ekf_new.so"
        cand_old = root / "libecm_ekf.so"
        native_lib = str(cand_new if cand_new.exists() else cand_old)

    df = load_cell_dataframe(args.data_root, args.cell)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = apply_scenario(df, args.scenario, args)
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

    gap_mask = compute_missing_gap_mask(t, float(args.missing_gap_seconds))
    has_gap = bool(np.any(gap_mask))
    if has_gap:
        nominal_dt = np.median(dt_s[(~gap_mask) & (dt_s > 0)])
        if not np.isfinite(nominal_dt) or nominal_dt <= 0:
            nominal_dt = 1.0
        dt_s[gap_mask] = 0.0
        for k in range(1, len(dt_s)):
            if gap_mask[k - 1] and not gap_mask[k]:
                dt_s[k] = nominal_dt

    soh_source = "const"
    if args.soh_mode == "data":
        if "SOH" in df.columns:
            soh = df["SOH"].to_numpy(dtype=np.float64)
            soh_source = "data"
        else:
            print("Warning: --soh_mode=data requested but SOH column is missing; falling back to const.")
            soh = np.full(len(df), float(args.soh_const), dtype=np.float64)
    else:
        soh = np.full(len(df), float(args.soh_const), dtype=np.float64)

    model = None
    if args.native:
        from ecm_native import ECMNativeEKF
        model = ECMNativeEKF(lib_path=native_lib, soc_init=float(args.soc_init))
    else:
        raise SystemExit(
            "Python EKF model was removed; run with --native to use the C implementation."
        )

    soc_true = df["SOC"].to_numpy(dtype=np.float64)
    u_true = df["Voltage[V]"].to_numpy(dtype=np.float64)
    I = df["Current[A]"].to_numpy(dtype=np.float64)

    soc_est = np.zeros(len(df), dtype=np.float64)
    u_est = np.zeros(len(df), dtype=np.float64)

    for k in range(len(df)):
        if has_gap and gap_mask[k]:
            if k == 0:
                soc_est[k] = float(args.soc_init)
                u_est[k] = u_true[k]
            else:
                soc_est[k] = soc_est[k - 1]
                u_est[k] = u_est[k - 1]
            continue
        if args.native:
            soc_est[k], u_est[k] = model.step(I[k], u_true[k], soh=soh[k])
        else:
            soc_est[k], u_est[k] = model.step(I[k], u_true[k], soh=soh[k], dt_s=dt_s[k])

    abs_err = np.abs(soc_true - soc_est)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame({
        "index": np.arange(len(df)),
        "time_s": t,
        "I": I,
        "U": u_true,
        "SOH": soh,
        "soc_true": soc_true,
        "soc_ecm": soc_est,
        "u_ecm": u_est,
        "abs_err": abs_err,
    })
    out_df.to_csv(out_dir / f"ecm_soc_fullcell_{args.cell}.csv", index=False)

    mask = out_df["time_s"] >= float(args.warmup_seconds)
    err_w = out_df.loc[mask, "soc_ecm"] - out_df.loc[mask, "soc_true"]

    summary = {
        "cell": args.cell,
        "scenario": args.scenario,
        "soc_init": float(args.soc_init),
        "capacity_ah": float(args.capacity_ah),
        "dt_mode": args.dt_mode,
        "dt_s": float(args.dt_s),
        "max_rows": int(args.max_rows),
        "downsample": int(args.downsample),
        "update_covariance": bool(args.update_covariance),
        "warmup_seconds": float(args.warmup_seconds),
        "missing_gap_seconds": float(args.missing_gap_seconds),
        "rmse": float(np.sqrt(np.mean((soc_true - soc_est) ** 2))),
        "mae": float(np.mean(np.abs(soc_true - soc_est))),
        "rmse_after_warmup": float(np.sqrt(np.mean(err_w ** 2))) if mask.any() else None,
        "mae_after_warmup": float(np.mean(np.abs(err_w))) if mask.any() else None,
        "ecm_param_c": ecm_param_c,
        "engine": "native" if args.native else "python",
        "native_fixed_dt_s": 60.0 if args.native else None,
        "soh_mode": args.soh_mode,
        "soh_source": soh_source,
        "soh_const": float(args.soh_const),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(out_df["time_s"] / 3600.0, out_df["soc_true"], label="SOC true", linewidth=1.0)
        ax1.plot(out_df["time_s"] / 3600.0, out_df["soc_ecm"], label="SOC ECM", linewidth=1.0, alpha=0.8)
        ax1.set_title(f"ECM EKF – Full Cell ({args.cell}) [{args.scenario}]")
        ax1.set_ylabel("SOC")
        ax1.legend(loc="best")
        fig.text(0.12, 0.93, f"MAE: {summary['mae']:.5f} | RMSE: {summary['rmse']:.5f}", fontsize=14,
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

        t_plot = out_df.loc[mask, "time_s"] / 3600.0
        err_plot = out_df.loc[mask, "abs_err"]
        ax2.plot(t_plot, err_plot, label="Absolute Error", linewidth=1.0, color="tab:red")
        ax2.set_xlabel("Time [h]")
        ax2.set_ylabel("Abs Error")
        ax2.set_ylim(0.0, 0.4)
        ax2.legend(loc="best")

        fig.tight_layout()
        fig.savefig(out_dir / f"ecm_soc_fullcell_{args.cell}.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
