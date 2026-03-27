import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecm_native import ECMNativeEKF


def load_fe_cell(data_root: Path, cell: str) -> pd.DataFrame:
    c_short = cell.split("_")[-1]
    candidates = [
        data_root / f"df_FE_{c_short}.parquet",
        data_root / f"df_FE_{cell}.parquet",
        data_root / f"df_FE_C{cell[-3:]}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"Could not locate FE parquet for {cell} in {data_root}")


def main():
    ap = argparse.ArgumentParser(
        description="Minimal ECM native run on FE data (battery_ekf.c via ctypes)."
    )
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--data_root", default="/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--native_lib", default=None, help="Defaults to ../libecm_ekf_new.so")
    ap.add_argument("--soh_const", type=float, default=1.0)
    ap.add_argument("--soc_init", type=float, default=1.0)
    ap.add_argument("--max_rows", type=int, default=1193040)
    ap.add_argument("--downsample", type=int, default=60)
    ap.add_argument("--warmup_seconds", type=float, default=600.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    lib = Path(args.native_lib) if args.native_lib else (root / "libecm_ekf_new.so")
    if not lib.exists():
        raise FileNotFoundError(f"Native lib not found: {lib}")

    df = load_fe_cell(Path(args.data_root), args.cell)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Testtime[s]", "Current[A]", "Voltage[V]", "SOC"]).reset_index(drop=True)
    if args.max_rows > 0:
        df = df.head(int(args.max_rows)).copy()
    if args.downsample > 1:
        df = df.iloc[:: int(args.downsample)].reset_index(drop=True)

    t = df["Testtime[s]"].to_numpy(dtype=np.float64)
    i = df["Current[A]"].to_numpy(dtype=np.float64)
    u = df["Voltage[V]"].to_numpy(dtype=np.float64)
    soc_true = df["SOC"].to_numpy(dtype=np.float64)

    ekf = ECMNativeEKF(lib_path=str(lib), soc_init=float(args.soc_init))
    soc_pred = np.zeros(len(df), dtype=np.float64)
    u_hat = np.zeros(len(df), dtype=np.float64)
    for k in range(len(df)):
        soc_pred[k], u_hat[k] = ekf.step(i[k], u[k], soh=float(args.soh_const))

    abs_err = np.abs(soc_true - soc_pred)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(
        {
            "index": np.arange(len(df)),
            "time_s": t,
            "I": i,
            "U": u,
            "soc_true": soc_true,
            "soc_ecm": soc_pred,
            "u_ecm": u_hat,
            "abs_err": abs_err,
        }
    )
    out_csv = out_dir / f"ecm_soc_fullcell_{args.cell}.csv"
    out_df.to_csv(out_csv, index=False)

    mask = out_df["time_s"] >= float(args.warmup_seconds)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((soc_true - soc_pred) ** 2)))
    summary = {
        "cell": args.cell,
        "rows": int(len(df)),
        "max_rows": int(args.max_rows),
        "downsample": int(args.downsample),
        "warmup_seconds": float(args.warmup_seconds),
        "soh_const": float(args.soh_const),
        "soc_init": float(args.soc_init),
        "native_lib": str(lib),
        "mae": mae,
        "rmse": rmse,
        "mae_after_warmup": float(np.mean(np.abs(out_df.loc[mask, "soc_ecm"] - out_df.loc[mask, "soc_true"])))
        if mask.any()
        else None,
        "rmse_after_warmup": float(np.sqrt(np.mean((out_df.loc[mask, "soc_ecm"] - out_df.loc[mask, "soc_true"]) ** 2)))
        if mask.any()
        else None,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(out_df["time_s"] / 3600.0, out_df["soc_true"], label="SOC true", linewidth=1.0)
    ax1.plot(out_df["time_s"] / 3600.0, out_df["soc_ecm"], label="SOC ECM", linewidth=1.0, alpha=0.8)
    ax1.set_ylabel("SOC")
    ax1.set_title(f"ECM native simple ({args.cell})")
    ax1.legend(loc="best")
    fig.text(0.12, 0.93, f"MAE: {mae:.5f} | RMSE: {rmse:.5f}", fontsize=14,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    ax2.plot(out_df["time_s"] / 3600.0, out_df["abs_err"], label="Absolute Error", linewidth=1.0, color="tab:red")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Abs Error")
    ax2.set_ylim(0.0, 0.4)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / f"ecm_soc_fullcell_{args.cell}.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
