import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean_blocks(x: np.ndarray, block: int = 60) -> np.ndarray:
    n = len(x) // block
    return x[: n * block].reshape(n, block).mean(axis=1)


def last_blocks(x: np.ndarray, block: int = 60) -> np.ndarray:
    n = len(x) // block
    return x[block - 1 : n * block : block]


def load_fe_cell(data_root: Path, cell: str) -> pd.DataFrame:
    # Accept both MGFarm_18650_C07 and C07 styles.
    cid = cell.split("_")[-1] if "_" in cell else cell
    path = data_root / f"df_FE_{cid}.parquet"
    if not path.exists():
        path = data_root / f"df_FE_{cell}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Could not find parquet for {cell} in {data_root}")
    return pd.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick functional test for ECM_v2_qinnan on one FE cell.")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--data_root", default="/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
    ap.add_argument("--ecm_v2_dir", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/ECM_v2_qinnan")
    ap.add_argument("--max_steps", type=int, default=20000, help="Number of minute-steps to evaluate.")
    ap.add_argument("--zoom_hours", type=float, default=6.0, help="Zoom window from start in hours.")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    ecm_v2_dir = Path(args.ecm_v2_dir).resolve()
    if str(ecm_v2_dir) not in sys.path:
        sys.path.insert(0, str(ecm_v2_dir))

    from EKF_fcn import BatteryEKF  # pylint: disable=import-error

    data_root = Path(args.data_root)
    df = load_fe_cell(data_root, args.cell)
    need = ["Current[A]", "Voltage[V]", "SOC", "SOH", "Testtime[s]"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in FE parquet: {miss}")

    i_raw = df["Current[A]"].to_numpy(dtype=np.float64)
    u_raw = df["Voltage[V]"].to_numpy(dtype=np.float64)
    soc_raw = df["SOC"].to_numpy(dtype=np.float64)
    soh_raw = df["SOH"].to_numpy(dtype=np.float64)
    t_raw = df["Testtime[s]"].to_numpy(dtype=np.float64)

    i_m = mean_blocks(i_raw, 60)
    u_m = mean_blocks(u_raw, 60)
    soc_true = last_blocks(soc_raw, 60)
    soh_m = last_blocks(soh_raw, 60)
    t_m = last_blocks(t_raw, 60)

    n = min(int(args.max_steps), len(i_m))
    i_m = i_m[:n]
    u_m = u_m[:n]
    soc_true = soc_true[:n]
    soh_m = soh_m[:n]
    t_m = t_m[:n]

    ekf = BatteryEKF(float(soh_m[0]))
    soc_est = np.zeros(n, dtype=np.float64)
    u_est = np.zeros(n, dtype=np.float64)

    for k in range(n):
        x, _, y = ekf.predict_update(float(i_m[k]), float(u_m[k]))
        soc_est[k] = float(x[0])
        u_est[k] = float(y)

    abs_err = np.abs(soc_true - soc_est)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((soc_true - soc_est) ** 2)))

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.utcnow().strftime("%Y-%m-%d_%H%M")
        out_dir = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/3_test/ECM_model/ecm_v2_single_cell") / f"{ts}_{args.cell}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(
        {
            "time_s": t_m,
            "current_mean_60s": i_m,
            "voltage_mean_60s": u_m,
            "soh_last_60s": soh_m,
            "soc_true": soc_true,
            "soc_est": soc_est,
            "u_est": u_est,
            "abs_err": abs_err,
        }
    )
    out_csv = out_dir / f"ecm_v2_soc_{args.cell}.csv"
    out_df.to_csv(out_csv, index=False)

    summary = {
        "model": "ECM_v2_qinnan",
        "cell": args.cell,
        "steps": int(n),
        "downsample_mode": "60-sample mean for I/U + last sample for SOC/SOH",
        "mae": mae,
        "rmse": rmse,
        "data_root": str(data_root),
        "ecm_v2_dir": str(ecm_v2_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    xh = t_m / 3600.0
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(xh, soc_true, label="SOC true", linewidth=1.0)
    ax[0].plot(xh, soc_est, label="SOC est (ECM v2)", linewidth=1.0, alpha=0.9)
    ax[0].set_ylabel("SOC")
    ax[0].set_title(f"ECM_v2_qinnan - {args.cell}")
    ax[0].legend(loc="best")
    ax[1].plot(xh, abs_err, label="Abs Error", color="tab:red", linewidth=0.9)
    ax[1].set_xlabel("Time [h]")
    ax[1].set_ylabel("Abs Error")
    ax[1].legend(loc="best")
    fig.text(
        0.12,
        0.95,
        f"MAE: {mae:.5f} | RMSE: {rmse:.5f}",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"ecm_v2_soc_full_{args.cell}.png", dpi=170)
    plt.close()

    zoom_h = float(args.zoom_hours)
    m = xh <= (xh[0] + zoom_h)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(xh[m], soc_true[m], label="SOC true", linewidth=1.0)
    ax[0].plot(xh[m], soc_est[m], label="SOC est (ECM v2)", linewidth=1.0, alpha=0.9)
    ax[0].set_ylabel("SOC")
    ax[0].set_title(f"ECM_v2_qinnan - Zoom first {zoom_h:g} h ({args.cell})")
    ax[0].legend(loc="best")
    ax[1].plot(xh[m], abs_err[m], label="Abs Error", color="tab:red", linewidth=0.9)
    ax[1].set_xlabel("Time [h]")
    ax[1].set_ylabel("Abs Error")
    ax[1].legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"ecm_v2_soc_zoom_{args.cell}.png", dpi=170)
    plt.close()

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
