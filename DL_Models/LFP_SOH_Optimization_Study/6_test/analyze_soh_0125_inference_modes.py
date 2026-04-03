#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = Path(
    "/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Base/0.1.2.5_base_h160"
)
TRAIN_SCRIPT = MODEL_DIR / "scripts"
sys.path.append(str(TRAIN_SCRIPT))
import train_soh as train_mod  # noqa: E402


def load_model_bundle(config_path: Path, ckpt_path: Path, scaler_path: Path, device: torch.device):
    cfg = yaml.safe_load(config_path.read_text())
    model = train_mod.SOH_LSTM_Seq2Seq(
        in_features=len(train_mod.expand_features_for_sampling(cfg["model"]["features"], cfg.get("sampling", {}))),
        embed_size=int(cfg["model"]["embed_size"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        num_layers=int(cfg["model"].get("num_layers", 2)),
        res_blocks=int(cfg["model"].get("res_blocks", 2)),
        bidirectional=bool(cfg["model"].get("bidirectional", False)),
        dropout=float(cfg["model"].get("dropout", 0.15)),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    model.eval()
    scaler = joblib.load(scaler_path)
    return cfg, model, scaler


def resolve_env_default(val: str) -> str:
    text = str(val)
    m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\:-?(.*)\}", text)
    if m:
        env_name, fallback = m.group(1), m.group(2)
        return os.environ.get(env_name, fallback)
    return os.path.expandvars(text)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else float("nan")
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "max_abs_error": float(np.max(np.abs(y_pred - y_true))),
        "bias": float(np.mean(y_pred - y_true)),
    }


def compare(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = a - b
    return {
        "mae_diff": float(np.mean(np.abs(d))),
        "rmse_diff": float(np.sqrt(np.mean(d**2))),
        "max_abs_diff": float(np.max(np.abs(d))),
    }


def stateful_stepwise_predict(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    preds = []
    state = None
    with torch.inference_mode():
        for i in range(len(X)):
            xb = torch.from_numpy(X[i : i + 1]).unsqueeze(0).to(device)
            y_seq, state = model(xb, state=state, return_state=True)
            preds.append(float(y_seq[0, 0].detach().cpu().item()))
    return np.asarray(preds, dtype=np.float32)


def stateful_chunk_predict(model, X: np.ndarray, chunk: int, device: torch.device) -> np.ndarray:
    preds = []
    state = None
    with torch.inference_mode():
        for start in range(0, len(X), chunk):
            end = min(start + chunk, len(X))
            xb = torch.from_numpy(X[start:end]).unsqueeze(0).to(device)
            y_seq, state = model(xb, state=state, return_state=True)
            preds.append(y_seq.squeeze(0).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def stateless_chunk_predict(model, X: np.ndarray, chunk: int, device: torch.device) -> np.ndarray:
    preds = []
    with torch.inference_mode():
        for start in range(0, len(X), chunk):
            end = min(start + chunk, len(X))
            xb = torch.from_numpy(X[start:end]).unsqueeze(0).to(device)
            y_seq = model(xb)
            preds.append(y_seq.squeeze(0).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def rolling_last_step_predict(model, X: np.ndarray, chunk: int, device: torch.device, batch_size: int) -> np.ndarray:
    n_pred = len(X) - chunk + 1
    preds = []
    with torch.inference_mode():
        for start in range(0, n_pred, batch_size):
            end = min(start + batch_size, n_pred)
            xb_np = np.stack([X[j : j + chunk] for j in range(start, end)], axis=0)
            xb = torch.from_numpy(xb_np).to(device)
            y_seq = model(xb)
            preds.append(y_seq[:, -1].detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def save_plots(out_dir: Path, y_true: np.ndarray, preds: Dict[str, np.ndarray]) -> None:
    n = len(y_true)
    max_points = 10000
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    else:
        idx = np.arange(n, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(idx, y_true[idx], label="SOH_ref", linewidth=1.2, color="black")
    for name, arr in preds.items():
        if len(arr) == len(y_true):
            ax.plot(idx, arr[idx], label=name, linewidth=0.9)
    ax.set_xlabel("Hourly index")
    ax.set_ylabel("SOH [-]")
    ax.set_title("SOH inference-mode comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "soh_0125_modes_timeseries.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    for name, arr in preds.items():
        if len(arr) == len(y_true):
            ax.plot(idx, np.abs(arr[idx] - y_true[idx]), label=f"|err| {name}", linewidth=0.9)
    ax.set_xlabel("Hourly index")
    ax.set_ylabel("Absolute SOH error [-]")
    ax.set_title("SOH absolute error comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "soh_0125_modes_abs_error.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compare inference modes for SOH LSTM model 0.1.2.5_base_h160.")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument(
        "--out_dir",
        default="/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/6_test/0.1.2.5_base_h160_inference_modes",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = MODEL_DIR / "config" / "train_soh.yaml"
    ckpt_path = MODEL_DIR / "checkpoints" / "best_epoch0045_rmse0.01516.pt"
    scaler_path = MODEL_DIR / "scaler_robust.joblib"

    device = torch.device(args.device)
    cfg, model, scaler = load_model_bundle(config_path, ckpt_path, scaler_path, device)

    data_root = resolve_env_default(cfg["paths"]["data_root"])
    base_features = list(cfg["model"]["features"])
    target = str(cfg["training"]["target"])
    sampling_cfg = cfg.get("sampling", {})
    features = train_mod.expand_features_for_sampling(base_features, sampling_cfg)
    chunk = int(cfg["training"]["seq_chunk_size"])

    df = train_mod.load_cell_parquet(data_root, args.cell)
    df = train_mod.maybe_aggregate_hourly(df, base_features, target, sampling_cfg)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target]).reset_index(drop=True)

    X = df[features].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)
    y_true = df[target].to_numpy(dtype=np.float32)

    results: Dict[str, object] = {
        "cell": args.cell,
        "rows_hourly": int(len(df)),
        "chunk_size_hours": int(chunk),
        "sampling_interval_s": int(sampling_cfg.get("interval_seconds", 3600)),
        "training_window_stride": int(cfg["training"].get("window_stride", 1)),
        "model_type": cfg["model"]["type"],
        "interpretation": {
            "training": "window-based seq2seq training on overlapping hourly windows",
            "inference_reference": "stateful chunked inference with hidden/cell carry across chunk boundaries",
        },
    }

    tic = time.perf_counter()
    pred_step = stateful_stepwise_predict(model, X_scaled, device)
    results["timing_stateful_stepwise_s"] = float(time.perf_counter() - tic)

    tic = time.perf_counter()
    pred_stateful_chunk = stateful_chunk_predict(model, X_scaled, chunk, device)
    results["timing_stateful_chunk_s"] = float(time.perf_counter() - tic)

    tic = time.perf_counter()
    pred_stateless_chunk = stateless_chunk_predict(model, X_scaled, chunk, device)
    results["timing_stateless_chunk_s"] = float(time.perf_counter() - tic)

    tic = time.perf_counter()
    pred_rolling_last = rolling_last_step_predict(model, X_scaled, chunk, device, args.batch_size)
    results["timing_rolling_laststep_s"] = float(time.perf_counter() - tic)

    y_tail = y_true[chunk - 1 :]
    step_tail = pred_step[chunk - 1 :]
    stateful_chunk_tail = pred_stateful_chunk[chunk - 1 :]
    stateless_chunk_tail = pred_stateless_chunk[chunk - 1 :]

    results["metrics"] = {
        "stateful_stepwise_tail_aligned": metrics(y_tail, step_tail),
        "stateful_chunk_tail_aligned": metrics(y_tail, stateful_chunk_tail),
        "stateless_chunk_tail_aligned": metrics(y_tail, stateless_chunk_tail),
        "rolling_laststep": metrics(y_tail, pred_rolling_last),
    }
    results["pairwise_prediction_difference"] = {
        "stateful_stepwise_vs_stateful_chunk_tail": compare(step_tail, stateful_chunk_tail),
        "stateful_stepwise_vs_stateless_chunk_tail": compare(step_tail, stateless_chunk_tail),
        "stateful_stepwise_vs_rolling_laststep": compare(step_tail, pred_rolling_last),
        "stateful_chunk_tail_vs_rolling_laststep": compare(stateful_chunk_tail, pred_rolling_last),
        "stateless_chunk_tail_vs_rolling_laststep": compare(stateless_chunk_tail, pred_rolling_last),
    }

    pd.DataFrame(
        {
            "hour_idx": np.arange(len(y_true)),
            "SOH_ref": y_true,
            "SOH_stateful_stepwise": pred_step,
            "SOH_stateful_chunk": pred_stateful_chunk,
            "SOH_stateless_chunk": pred_stateless_chunk,
        }
    ).to_csv(out_dir / f"soh_0125_modes_{args.cell}_full_hourly.csv", index=False)

    pd.DataFrame(
        {
            "hour_idx_tail": np.arange(chunk - 1, len(y_true)),
            "SOH_ref_tail": y_tail,
            "SOH_stateful_stepwise_tail": step_tail,
            "SOH_stateful_chunk_tail": stateful_chunk_tail,
            "SOH_stateless_chunk_tail": stateless_chunk_tail,
            "SOH_rolling_laststep": pred_rolling_last,
        }
    ).to_csv(out_dir / f"soh_0125_modes_{args.cell}_tail_aligned.csv", index=False)

    save_plots(
        out_dir,
        y_true,
        {
            "stateful_stepwise": pred_step,
            "stateful_chunk": pred_stateful_chunk,
            "stateless_chunk": pred_stateless_chunk,
        },
    )

    with open(out_dir / f"soh_0125_modes_{args.cell}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
