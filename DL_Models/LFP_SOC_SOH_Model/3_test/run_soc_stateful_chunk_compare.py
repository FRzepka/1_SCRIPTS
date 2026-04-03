#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Tuple
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/florianr/MG_Farm/1_Scripts")
SIM_ENV_DIR = ROOT / "DL_Models/LFP_SOC_SOH_Model/4_simulation_environment"
sys.path.append(str(SIM_ENV_DIR))
from robustness_common import build_online_aux_features, load_cell_dataframe  # noqa: E402

MODELS = {
    "SOC_1.6.0.0": {
        "config": ROOT / "DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/train_soc.yaml",
        "ckpt": ROOT / "DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/soc_epoch0005_rmse0.01393.pt",
        "scaler": ROOT / "DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/scaler_robust.joblib",
        "train_script": ROOT / "DL_Models/LFP_SOC_SOH_Model/1_training/1.6.0.0/scripts/train_soc.py",
        "class_name": "LSTMMLP",
        "color": "#3165d4",
    },
    "SOC_1.7.0.0": {
        "config": ROOT / "DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/train_soc.yaml",
        "ckpt": ROOT / "DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/soc_epoch0002_rmse0.01488.pt",
        "scaler": ROOT / "DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/scaler_robust.joblib",
        "train_script": ROOT / "DL_Models/LFP_SOC_SOH_Model/1_training/1.7.0.0/scripts/train_soc.py",
        "class_name": "GRUMLP",
        "color": "#08bdba",
    },
}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_feature_dataframe(cell: str) -> pd.DataFrame:
    data_root = "/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"
    df = load_cell_dataframe(data_root, cell)
    required = ["Testtime[s]", "Voltage[V]", "Current[A]", "Temperature[°C]", "SOC", "SOH"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)
    freeze_mask = np.zeros(len(df), dtype=bool)
    df = build_online_aux_features(
        df=df,
        freeze_mask=freeze_mask,
        current_sign=1.0,
        v_max=3.65,
        v_tol=0.02,
        cv_seconds=300.0,
        nominal_capacity_ah=1.8,
        initial_soc_delta=0.0,
    )
    if "_dt_s_online" in df.columns:
        df["dt_s"] = df["_dt_s_online"].astype(np.float32)
    return df


def stateful_chunk_predict(model, Xs: np.ndarray, chunk: int, device: torch.device):
    preds = np.empty(len(Xs), dtype=np.float32)
    state = None
    with torch.inference_mode():
        for start in range(0, len(Xs), chunk):
            end = min(start + chunk, len(Xs))
            xb = torch.from_numpy(Xs[start:end]).unsqueeze(0).to(device)
            if hasattr(model, "lstm"):
                out, state = model.lstm(xb, state)
            elif hasattr(model, "gru"):
                out, state = model.gru(xb, state)
            else:
                raise TypeError("Unsupported model: neither LSTM nor GRU core found")
            flat = out.reshape(-1, out.shape[-1])
            y_all = model.mlp(flat).reshape(out.shape[0], out.shape[1]).detach().cpu().numpy()
            preds[start:end] = y_all.squeeze(0)
    return preds


def stateful_stepwise_predict(model, Xs: np.ndarray, device: torch.device):
    preds = np.empty(len(Xs), dtype=np.float32)
    state = None
    with torch.inference_mode():
        for i in range(len(Xs)):
            xb = torch.from_numpy(Xs[i : i + 1]).unsqueeze(0).to(device)
            y, state = model(xb, state=state, return_state=True)
            preds[i] = float(y.squeeze().detach().cpu().item())
    return preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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


def plot_fullcell(out_dir: Path, t_hours: np.ndarray, y_true: np.ndarray, preds: Dict[str, np.ndarray]):
    max_points = 25000
    if len(t_hours) > max_points:
        idx = np.linspace(0, len(t_hours) - 1, max_points, dtype=np.int64)
    else:
        idx = np.arange(len(t_hours), dtype=np.int64)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(t_hours[idx], y_true[idx], color="black", linewidth=1.2, label="SOC_ref")
    for name, series in preds.items():
        ax.plot(t_hours[idx], series[idx], linewidth=1.0, label=name, color=MODELS[name]["color"])
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("SOC [-]")
    ax.set_title("Stateful chunk inference over full cell trajectory")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "soc_stateful_chunk_fullcell_compare.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6))
    for name, series in preds.items():
        ax.plot(t_hours[idx], np.abs(series[idx] - y_true[idx]), linewidth=1.0, label=f"|err| {name}", color=MODELS[name]["color"])
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Absolute SOC error [-]")
    ax.set_title("Absolute error under stateful chunk inference")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "soc_stateful_chunk_abs_error_compare.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compare SOC_1.6.0.0 and SOC_1.7.0.0 in stateful chunk mode.")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check_rows", type=int, default=6000)
    ap.add_argument(
        "--out_dir",
        default=str(ROOT / "DL_Models/LFP_SOC_SOH_Model/3_test/stateful_chunk_compare_2026-03-27"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    df = build_feature_dataframe(args.cell)
    t_hours = (df["Testtime[s]"].to_numpy(dtype=np.float64) - float(df["Testtime[s]"].iloc[0])) / 3600.0
    y_true = df["SOC"].to_numpy(dtype=np.float32)

    results: Dict[str, object] = {
        "cell": args.cell,
        "rows_fullcell": int(len(df)),
        "hours_fullcell": float(t_hours[-1] if len(t_hours) else 0.0),
        "mode_definition": "non-overlapping chunk inference with persistent hidden-state carry across chunk boundaries",
        "models": {},
    }
    preds = {}

    for model_name, spec in MODELS.items():
        cfg = yaml.safe_load(Path(spec["config"]).read_text())
        mod = load_module(Path(spec["train_script"]), f"{model_name}_train_soc")
        model_cls = getattr(mod, spec["class_name"])
        model = model_cls(
            in_features=len(cfg["model"]["features"]),
            hidden_size=int(cfg["model"]["hidden_size"]),
            mlp_hidden=int(cfg["model"]["mlp_hidden"]),
            num_layers=int(cfg["model"].get("num_layers", 1)),
            dropout=float(cfg["model"].get("dropout", 0.05)),
        ).to(device)
        state = torch.load(spec["ckpt"], map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()

        features = list(cfg["model"]["features"])
        work = df.copy()
        for col in features:
            work[col] = work[col].ffill().bfill()
        X = work[features].to_numpy(dtype=np.float32)
        Xs = joblib.load(spec["scaler"]).transform(X).astype(np.float32)
        chunk = int(cfg["training"]["seq_chunk_size"])

        tic = time.perf_counter()
        pred_chunk = stateful_chunk_predict(model, Xs, chunk, device)
        t_chunk = float(time.perf_counter() - tic)
        preds[model_name] = pred_chunk

        check_n = min(int(args.check_rows), len(Xs))
        pred_step = stateful_stepwise_predict(model, Xs[:check_n], device)
        diff = compare(pred_step, pred_chunk[:check_n])

        results["models"][model_name] = {
            "chunk_size": int(chunk),
            "features": features,
            "timing_fullcell_stateful_chunk_s": t_chunk,
            "fullcell_metrics": compute_metrics(y_true, pred_chunk),
            "short_check_rows": int(check_n),
            "stateful_stepwise_vs_stateful_chunk_first_rows": diff,
        }

    plot_fullcell(out_dir, t_hours, y_true, preds)

    with open(out_dir / f"summary_{args.cell}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import time
    main()
