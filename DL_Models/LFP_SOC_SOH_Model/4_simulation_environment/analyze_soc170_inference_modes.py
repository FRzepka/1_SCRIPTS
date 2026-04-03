#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import load
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustness_common import build_online_aux_features, load_cell_dataframe

import sys

TRAIN_SCRIPT_DIR = Path(
    "/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/1_training/1.7.0.0/scripts"
)
sys.path.append(str(TRAIN_SCRIPT_DIR))
from train_soc import GRUMLP  # noqa: E402


def load_soc_bundle(config_path: Path, ckpt_path: Path, scaler_path: Path, device: torch.device):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = GRUMLP(
        in_features=len(cfg["model"]["features"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.05)),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    scaler = load(scaler_path)
    return cfg, model, scaler


def prepare_dataframe(df: pd.DataFrame, features, nominal_capacity_ah: float) -> pd.DataFrame:
    freeze_mask = np.zeros(len(df), dtype=bool)
    df = build_online_aux_features(
        df=df,
        freeze_mask=freeze_mask,
        current_sign=1.0,
        v_max=3.65,
        v_tol=0.02,
        cv_seconds=300.0,
        nominal_capacity_ah=nominal_capacity_ah,
        initial_soc_delta=0.0,
    )

    if "dt_s" in features:
        if "_dt_s_online" in df.columns:
            df["dt_s"] = df["_dt_s_online"].astype(np.float32)
        else:
            t = df["Testtime[s]"].to_numpy(dtype=np.float32)
            dt = np.empty_like(t)
            if len(t) > 1:
                dt[0] = max(float(t[1] - t[0]), 1e-6)
                dt[1:] = np.diff(t)
            elif len(t) == 1:
                dt[0] = 1.0
            df["dt_s"] = dt

    required = sorted(set(features + ["SOC", "SOH", "Testtime[s]"]))
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).reset_index(drop=True)
    for col in features:
        df[col] = df[col].ffill().bfill()
    return df


def stateful_stream_predict(
    Xs: np.ndarray,
    model: GRUMLP,
    device: torch.device,
) -> np.ndarray:
    preds = np.empty(len(Xs), dtype=np.float32)
    state = None
    with torch.no_grad():
        for i in range(len(Xs)):
            xb = torch.from_numpy(Xs[i : i + 1]).unsqueeze(1).to(device)
            pred, state = model(xb, state=state, return_state=True)
            preds[i] = float(pred.squeeze().detach().cpu().item())
    return preds


def stateful_window_predict(
    Xs: np.ndarray,
    chunk: int,
    model: GRUMLP,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    n_pred = len(Xs) - chunk + 1
    preds = []
    with torch.no_grad():
        for start in range(0, n_pred, batch_size):
            end = min(start + batch_size, n_pred)
            windows = np.stack([Xs[j : j + chunk] for j in range(start, end)], axis=0)
            state = None
            last_pred = None
            for t in range(chunk):
                xb = torch.from_numpy(windows[:, t : t + 1, :]).to(device)
                last_pred, state = model(xb, state=state, return_state=True)
            preds.append(last_pred.detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def stateless_window_predict(
    Xs: np.ndarray,
    chunk: int,
    model: GRUMLP,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    n_pred = len(Xs) - chunk + 1
    preds = []
    with torch.no_grad():
        for start in range(0, n_pred, batch_size):
            end = min(start + batch_size, n_pred)
            xb_np = np.stack([Xs[j : j + chunk] for j in range(start, end)], axis=0)
            xb = torch.from_numpy(xb_np).to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(abs_err)),
        "bias": float(np.mean(err)),
    }


def compare(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = a - b
    ad = np.abs(d)
    return {
        "mae_diff": float(np.mean(ad)),
        "rmse_diff": float(np.sqrt(np.mean(d**2))),
        "max_abs_diff": float(np.max(ad)),
    }


def _downsample_idx(n: int, max_points: int = 20000) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_points, dtype=np.int64)


def save_plots(
    out_dir: Path,
    time_s: np.ndarray,
    y_ref: np.ndarray,
    pred_stream: np.ndarray,
    pred_stateful_window: np.ndarray,
    pred_stateless_window: np.ndarray,
) -> None:
    idx = _downsample_idx(len(time_s), max_points=20000)
    th = (time_s[idx] - time_s[idx][0]) / 3600.0

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(th, y_ref[idx], label="SOC_ref", linewidth=1.3, color="black")
    ax.plot(th, pred_stream[idx], label="stateful_stream", linewidth=1.0)
    ax.plot(th, pred_stateful_window[idx], label="stateful_window", linewidth=1.0)
    ax.plot(th, pred_stateless_window[idx], label="stateless_window", linewidth=1.0, alpha=0.9)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("SOC [-]")
    ax.set_title("SOC_1.7.0.0 inference-mode comparison over aligned tail")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "soc170_inference_modes_full_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(th, np.abs(pred_stream[idx] - y_ref[idx]), label="|err| stateful_stream", linewidth=1.0)
    ax.plot(th, np.abs(pred_stateful_window[idx] - y_ref[idx]), label="|err| stateful_window", linewidth=1.0)
    ax.plot(th, np.abs(pred_stateless_window[idx] - y_ref[idx]), label="|err| stateless_window", linewidth=1.0)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Absolute SOC error [-]")
    ax.set_title("Absolute error comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "soc170_inference_modes_abs_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(th, pred_stream[idx] - pred_stateless_window[idx], label="stream - stateless_window", linewidth=1.0)
    ax.plot(th, pred_stateful_window[idx] - pred_stateless_window[idx], label="stateful_window - stateless_window", linewidth=1.0)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Prediction difference [-]")
    ax.set_title("Pairwise prediction differences")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "soc170_inference_modes_pairwise_diff.png", dpi=180)
    plt.close(fig)


def memory_report(
    cfg: dict,
    model: GRUMLP,
    stateful_onnx_path: Path,
    stateless_onnx_path: Path,
) -> Dict[str, Dict[str, float]]:
    features = len(cfg["model"]["features"])
    hidden = int(cfg["model"]["hidden_size"])
    mlp_hidden = int(cfg["model"]["mlp_hidden"])
    chunk = int(cfg["training"]["seq_chunk_size"])
    layers = int(cfg["model"].get("num_layers", 1))
    params = sum(p.numel() for p in model.parameters())
    param_bytes = params * 4

    step_input = features * 4
    hidden_state = layers * hidden * 4
    gru_output_step = hidden * 4
    mlp_hidden_bytes = mlp_hidden * 4
    gate_scratch = 3 * hidden * 4
    stream_lower = step_input + hidden_state + gru_output_step + mlp_hidden_bytes + 4 + gate_scratch

    window_input = chunk * features * 4
    window_output_seq = chunk * hidden * 4
    window_stepwise = window_input + stream_lower
    window_stateless = window_input + window_output_seq + hidden_state + mlp_hidden_bytes + 4

    return {
        "shared_model": {
            "parameter_count": int(params),
            "parameter_bytes_float32": int(param_bytes),
            "checkpoint_bytes": int(Path(cfg["paths"]["model_out_root"]).joinpath("soc_epoch0002_rmse0.01488.pt").stat().st_size),
        },
        "stateful_streaming": {
            "flash_onnx_bytes": int(stateful_onnx_path.stat().st_size),
            "ram_lower_bound_bytes": int(stream_lower),
            "ram_lower_bound_kib": float(stream_lower / 1024.0),
        },
        "stateful_window_replay": {
            "flash_onnx_bytes": int(stateful_onnx_path.stat().st_size),
            "ram_lower_bound_bytes": int(window_stepwise),
            "ram_lower_bound_kib": float(window_stepwise / 1024.0),
        },
        "stateless_full_window": {
            "flash_onnx_bytes": int(stateless_onnx_path.stat().st_size),
            "ram_lower_bound_bytes": int(window_stateless),
            "ram_lower_bound_kib": float(window_stateless / 1024.0),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Compare SOC_1.7.0.0 inference modes without SOH model coupling.")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=6000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--out_dir",
        default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/3_test/SOC_1.7.0.0/inference_mode_check",
    )
    ap.add_argument(
        "--soc_config",
        default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/train_soc.yaml",
    )
    ap.add_argument(
        "--soc_ckpt",
        default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/soc_epoch0002_rmse0.01488.pt",
    )
    ap.add_argument(
        "--soc_scaler",
        default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/scaler_robust.joblib",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cfg, model, scaler = load_soc_bundle(
        config_path=Path(args.soc_config),
        ckpt_path=Path(args.soc_ckpt),
        scaler_path=Path(args.soc_scaler),
        device=device,
    )

    features = list(cfg["model"]["features"])
    chunk = int(cfg["training"]["seq_chunk_size"])
    data_root = cfg["paths"]["data_root"]

    df = load_cell_dataframe(data_root, args.cell)
    stop_row = args.start_row + args.max_rows if args.max_rows > 0 else None
    df = df.iloc[args.start_row:stop_row].copy().reset_index(drop=True)
    df = prepare_dataframe(df, features=features, nominal_capacity_ah=1.8)

    X = df[features].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    y = df["SOC"].to_numpy(dtype=np.float32)
    t = df["Testtime[s]"].to_numpy(dtype=np.float64)

    if len(df) < chunk:
        raise ValueError(f"Need at least {chunk} rows, got {len(df)}")

    results: Dict[str, object] = {
        "cell": args.cell,
        "start_row": int(args.start_row),
        "rows_used": int(len(df)),
        "hours_covered": float((t[-1] - t[0]) / 3600.0),
        "chunk_size": int(chunk),
        "features": features,
        "soh_source": "offline SOH column from dataframe; no separate SOH model used",
    }

    tic = time.perf_counter()
    pred_stream = stateful_stream_predict(Xs, model, device)
    results["timing_stateful_stream_s"] = float(time.perf_counter() - tic)

    tic = time.perf_counter()
    pred_stateful_window = stateful_window_predict(Xs, chunk, model, device, batch_size=args.batch_size)
    results["timing_stateful_window_s"] = float(time.perf_counter() - tic)

    tic = time.perf_counter()
    pred_stateless_window = stateless_window_predict(Xs, chunk, model, device, batch_size=args.batch_size)
    results["timing_stateless_window_s"] = float(time.perf_counter() - tic)

    y_tail = y[chunk - 1 :]
    pred_stream_tail = pred_stream[chunk - 1 :]

    results["metrics"] = {
        "stateful_streaming_tail_aligned": metrics(y_tail, pred_stream_tail),
        "stateful_window": metrics(y_tail, pred_stateful_window),
        "stateless_window": metrics(y_tail, pred_stateless_window),
    }
    results["pairwise_prediction_difference"] = {
        "stateful_window_vs_stateless_window": compare(pred_stateful_window, pred_stateless_window),
        "stateful_streaming_vs_stateless_window": compare(pred_stream_tail, pred_stateless_window),
        "stateful_streaming_vs_stateful_window": compare(pred_stream_tail, pred_stateful_window),
    }
    results["memory_estimate"] = memory_report(
        cfg=cfg,
        model=model,
        stateful_onnx_path=Path(args.soc_ckpt).with_name("soc_best_epoch0002_stateful.onnx"),
        stateless_onnx_path=Path(args.soc_ckpt).with_name("soc_best_epoch0002.onnx"),
    )

    pd.DataFrame(
        {
            "Testtime[s]": t[chunk - 1 :],
            "SOC_ref": y_tail,
            "SOC_stateful_stream": pred_stream_tail,
            "SOC_stateful_window": pred_stateful_window,
            "SOC_stateless_window": pred_stateless_window,
        }
    ).to_csv(out_dir / f"soc170_inference_modes_{args.cell}_{args.start_row}_{len(df)}.csv", index=False)

    save_plots(
        out_dir=out_dir,
        time_s=t[chunk - 1 :],
        y_ref=y_tail,
        pred_stream=pred_stream_tail,
        pred_stateful_window=pred_stateful_window,
        pred_stateless_window=pred_stateless_window,
    )

    with open(out_dir / f"soc170_inference_modes_{args.cell}_{args.start_row}_{len(df)}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
