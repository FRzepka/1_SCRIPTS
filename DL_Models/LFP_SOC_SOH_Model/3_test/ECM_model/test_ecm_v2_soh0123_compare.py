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

import torch
import yaml
from joblib import load as joblib_load


def mean_blocks(x: np.ndarray, block: int = 60) -> np.ndarray:
    n = len(x) // block
    return x[: n * block].reshape(n, block).mean(axis=1)


def last_blocks(x: np.ndarray, block: int = 60) -> np.ndarray:
    n = len(x) // block
    return x[block - 1 : n * block : block]


def load_fe_cell(data_root: Path, cell: str) -> pd.DataFrame:
    cid = cell.split("_")[-1] if "_" in cell else cell
    path = data_root / f"df_FE_{cid}.parquet"
    if not path.exists():
        path = data_root / f"df_FE_{cell}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Could not find parquet for {cell} in {data_root}")
    return pd.read_parquet(path)


def build_hourly_soh_predictions(
    df_raw: pd.DataFrame,
    soh_config_path: Path,
    soh_ckpt_path: Path,
    soh_scaler_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    # Import SOH model helpers from 0.1.2.3 package.
    soh_module_dir = soh_config_path.parent
    if str(soh_module_dir) not in sys.path:
        sys.path.insert(0, str(soh_module_dir))
    from predict_soh import (  # pylint: disable=import-error
        SOH_LSTM_Seq2Seq,
        aggregate_hourly,
        expand_features_for_sampling,
    )

    with open(soh_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_features = cfg["model"]["features"]
    embed_size = int(cfg["model"]["embed_size"])
    hidden_size = int(cfg["model"]["hidden_size"])
    mlp_hidden = int(cfg["model"]["mlp_hidden"])
    num_layers = int(cfg["model"].get("num_layers", 2))
    res_blocks = int(cfg["model"].get("res_blocks", 2))
    bidirectional = bool(cfg["model"].get("bidirectional", False))
    dropout = float(cfg["model"].get("dropout", 0.15))
    sampling_cfg = cfg.get("sampling", {})
    interval_seconds = int(sampling_cfg.get("interval_seconds", 3600))
    feature_aggs = sampling_cfg.get("feature_aggs", ["mean", "std", "min", "max"])

    hourly = aggregate_hourly(df_raw, base_features, interval_seconds, feature_aggs)
    if hourly.empty:
        raise ValueError("No hourly rows for SOH prediction.")
    feature_cols = expand_features_for_sampling(base_features, feature_aggs)

    scaler = joblib_load(str(soh_scaler_path))
    X = hourly[feature_cols].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)

    model = SOH_LSTM_Seq2Seq(
        in_features=len(feature_cols),
        embed_size=embed_size,
        hidden_size=hidden_size,
        mlp_hidden=mlp_hidden,
        num_layers=num_layers,
        res_blocks=res_blocks,
        bidirectional=bidirectional,
        dropout=dropout,
    ).to(device)
    state = torch.load(str(soh_ckpt_path), map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    preds = []
    stateful = None
    with torch.no_grad():
        for i in range(len(Xs)):
            x_step = torch.from_numpy(Xs[i : i + 1]).unsqueeze(1).to(device)
            y_seq, stateful = model(x_step, state=stateful, return_state=True)
            preds.append(float(y_seq.squeeze().detach().cpu().numpy()))

    out = pd.DataFrame(
        {
            "bin": hourly["bin"].to_numpy(dtype=np.int64),
            "time_s": hourly["bin"].to_numpy(dtype=np.int64) * interval_seconds,
            "soh_pred": np.array(preds, dtype=np.float64),
        }
    )
    return out


def map_hourly_to_minute(t_minute: np.ndarray, hourly: pd.DataFrame) -> np.ndarray:
    bins_minute = np.floor(t_minute / 3600.0).astype(np.int64)
    pred_map = pd.Series(hourly["soh_pred"].to_numpy(dtype=np.float64), index=hourly["bin"].to_numpy(dtype=np.int64))
    soh_m = pd.Series(bins_minute).map(pred_map)
    soh_m = soh_m.ffill().bfill()
    return soh_m.to_numpy(dtype=np.float64)


def run_ekf(i_m: np.ndarray, u_m: np.ndarray, soh_seq: np.ndarray, ecm_v2_dir: Path) -> np.ndarray:
    if str(ecm_v2_dir) not in sys.path:
        sys.path.insert(0, str(ecm_v2_dir))
    from EKF_fcn import BatteryEKF  # pylint: disable=import-error

    ekf = BatteryEKF(float(soh_seq[0]))
    soh_min = float(np.min(ekf.ecm.soh))
    soh_max = float(np.max(ekf.ecm.soh))

    soc_est = np.zeros(len(i_m), dtype=np.float64)
    for k in range(len(i_m)):
        ekf.soh = float(np.clip(soh_seq[k], soh_min, soh_max))
        ekf.Cb = ekf.C0 * ekf.soh
        x, _, _ = ekf.predict_update(float(i_m[k]), float(u_m[k]))
        soc_est[k] = float(x[0])
    return soc_est


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare ECM_v2 with fixed SOH vs SOH model 0.1.2.3.")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--data_root", default="/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
    ap.add_argument("--ecm_v2_dir", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/ECM_v2_qinnan")
    ap.add_argument("--soh_config", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/train_soh.yaml")
    ap.add_argument("--soh_ckpt", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/best_epoch0093_rmse0.02165.pt")
    ap.add_argument("--soh_scaler", default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/scaler_robust.joblib")
    ap.add_argument("--max_steps", type=int, default=0, help="0 means full cell.")
    ap.add_argument("--zoom_hours", type=float, default=6.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    ecm_v2_dir = Path(args.ecm_v2_dir).resolve()
    soh_config_path = Path(args.soh_config).resolve()
    soh_ckpt_path = Path(args.soh_ckpt).resolve()
    soh_scaler_path = Path(args.soh_scaler).resolve()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    df = load_fe_cell(data_root, args.cell)
    required = ["Current[A]", "Voltage[V]", "SOC", "SOH", "Testtime[s]"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    i_m = mean_blocks(df["Current[A]"].to_numpy(dtype=np.float64), 60)
    u_m = mean_blocks(df["Voltage[V]"].to_numpy(dtype=np.float64), 60)
    soc_true = last_blocks(df["SOC"].to_numpy(dtype=np.float64), 60)
    soh_true = last_blocks(df["SOH"].to_numpy(dtype=np.float64), 60)
    t_m = last_blocks(df["Testtime[s]"].to_numpy(dtype=np.float64), 60)

    if int(args.max_steps) > 0:
        n = min(int(args.max_steps), len(i_m))
        i_m, u_m, soc_true, soh_true, t_m = i_m[:n], u_m[:n], soc_true[:n], soh_true[:n], t_m[:n]

    hourly_soh = build_hourly_soh_predictions(df, soh_config_path, soh_ckpt_path, soh_scaler_path, device)
    soh_pred_m = map_hourly_to_minute(t_m, hourly_soh)

    soh_fixed = np.full(len(t_m), float(soh_true[0]), dtype=np.float64)
    soc_fixed = run_ekf(i_m, u_m, soh_fixed, ecm_v2_dir)
    soc_dyn_pred = run_ekf(i_m, u_m, soh_pred_m, ecm_v2_dir)
    soc_dyn_true = run_ekf(i_m, u_m, soh_true, ecm_v2_dir)

    m_fixed = metrics(soc_true, soc_fixed)
    m_dyn_pred = metrics(soc_true, soc_dyn_pred)
    m_dyn_true = metrics(soc_true, soc_dyn_true)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.utcnow().strftime("%Y-%m-%d_%H%M")
        out_dir = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/3_test/ECM_model/ecm_v2_soh0123_compare") / f"{ts}_{args.cell}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(
        {
            "time_s": t_m,
            "soc_true": soc_true,
            "soc_ecm_soh_fixed": soc_fixed,
            "soc_ecm_soh_pred0123": soc_dyn_pred,
            "soc_ecm_soh_true": soc_dyn_true,
            "soh_true": soh_true,
            "soh_pred_0123": soh_pred_m,
            "abs_err_fixed": np.abs(soc_true - soc_fixed),
            "abs_err_pred0123": np.abs(soc_true - soc_dyn_pred),
            "abs_err_true_soh": np.abs(soc_true - soc_dyn_true),
        }
    )
    out_df.to_csv(out_dir / f"ecm_v2_soh_compare_{args.cell}.csv", index=False)
    hourly_soh.to_csv(out_dir / f"soh_hourly_pred_0123_{args.cell}.csv", index=False)

    summary = {
        "model": "ECM_v2_qinnan",
        "cell": args.cell,
        "steps": int(len(t_m)),
        "device": str(device),
        "data_source": str(data_root),
        "soh_model": {
            "config": str(soh_config_path),
            "checkpoint": str(soh_ckpt_path),
            "scaler": str(soh_scaler_path),
        },
        "metrics": {
            "ecm_soh_fixed": m_fixed,
            "ecm_soh_pred0123": m_dyn_pred,
            "ecm_soh_true_oracle": m_dyn_true,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    xh = t_m / 3600.0
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(xh, soc_true, label="SOC true", linewidth=1.0, color="black")
    ax[0].plot(xh, soc_fixed, label=f"ECM fixed SOH ({m_fixed['mae']:.4f} MAE)", linewidth=0.9)
    ax[0].plot(xh, soc_dyn_pred, label=f"ECM + SOH_0.1.2.3 ({m_dyn_pred['mae']:.4f} MAE)", linewidth=0.9)
    ax[0].plot(xh, soc_dyn_true, label=f"ECM + true SOH oracle ({m_dyn_true['mae']:.4f} MAE)", linewidth=0.9, alpha=0.85)
    ax[0].set_ylabel("SOC")
    ax[0].set_title(f"ECM_v2 SOH comparison - full ({args.cell})")
    ax[0].legend(loc="best")
    ax[1].plot(xh, np.abs(soc_true - soc_fixed), label="|err| fixed SOH", linewidth=0.9)
    ax[1].plot(xh, np.abs(soc_true - soc_dyn_pred), label="|err| SOH_0.1.2.3", linewidth=0.9)
    ax[1].plot(xh, np.abs(soc_true - soc_dyn_true), label="|err| true SOH oracle", linewidth=0.9)
    ax[1].set_xlabel("Time [h]")
    ax[1].set_ylabel("Abs Error")
    ax[1].legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"ecm_v2_soh_compare_full_{args.cell}.png", dpi=170)
    plt.close()

    zh = float(args.zoom_hours)
    mask = xh <= (xh[0] + zh)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(xh[mask], soc_true[mask], label="SOC true", linewidth=1.0, color="black")
    ax[0].plot(xh[mask], soc_fixed[mask], label="ECM fixed SOH", linewidth=0.9)
    ax[0].plot(xh[mask], soc_dyn_pred[mask], label="ECM + SOH_0.1.2.3", linewidth=0.9)
    ax[0].plot(xh[mask], soc_dyn_true[mask], label="ECM + true SOH oracle", linewidth=0.9, alpha=0.85)
    ax[0].set_ylabel("SOC")
    ax[0].set_title(f"ECM_v2 SOH comparison - zoom first {zh:g} h ({args.cell})")
    ax[0].legend(loc="best")
    ax[1].plot(xh[mask], np.abs(soc_true[mask] - soc_fixed[mask]), label="|err| fixed SOH", linewidth=0.9)
    ax[1].plot(xh[mask], np.abs(soc_true[mask] - soc_dyn_pred[mask]), label="|err| SOH_0.1.2.3", linewidth=0.9)
    ax[1].plot(xh[mask], np.abs(soc_true[mask] - soc_dyn_true[mask]), label="|err| true SOH oracle", linewidth=0.9)
    ax[1].set_xlabel("Time [h]")
    ax[1].set_ylabel("Abs Error")
    ax[1].legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"ecm_v2_soh_compare_zoom_{args.cell}.png", dpi=170)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(xh, soh_true, label="SOH true", linewidth=0.9)
    ax.plot(xh, soh_pred_m, label="SOH pred 0.1.2.3 (hourly hold)", linewidth=0.9)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("SOH")
    ax.set_title(f"SOH trace used by ECM ({args.cell})")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"soh_trace_compare_{args.cell}.png", dpi=170)
    plt.close()

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
