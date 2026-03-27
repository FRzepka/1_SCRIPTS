#!/usr/bin/env python3
"""
SOC Benchmark (PC): Base/Pruned/Quant – Python vs C

Vergleicht Python- und C-Inferenz (Float) sowie Quant-C (Int8) auf den ersten N Samples.
Speichert Metriken, Arrays und Overlays.
"""
import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# Matplotlib only if plots requested
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]  # .../DL_Models/LFP_LSTM_MLP
DATA_ROOT_DEF = Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")

# Model paths
BASE_CKPT = REPO / "2_models/base/soc_1.5.0.0_base/1.5.0.0_soc_epoch0001_rmse0.02897.pt"
PRUNED_CKPT = REPO / "2_models/pruned/soc_1.5.0.0_pruned/prune_30pct_20250916_140404/soc_pruned_hidden45.pt"
SCALER_PATH = REPO / "1_training/1.5.0.0/outputs/scaler_robust.joblib"

# C libs (compiled .so)
LIB_BASE = REPO / "2_models/base/soc_1.5.0.0_base/c_implementation/liblstm_soc_base.so"
LIB_PRUNED = REPO / "2_models/pruned/soc_1.5.0.0_pruned/prune_30pct_20250916_140404/c_implementation/liblstm_soc_pruned.so"
LIB_QUANT = REPO / "2_models/quantized/soc_1.5.0.0_quantized/liblstm_soc_quant.so"

FEATURES = ["Voltage[V]", "Current[A]", "Temperature[°C]", "Q_c", "dU_dt[V/s]", "dI_dt[A/s]"]

# Force line buffered stdout for live tqdm in long runs
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass


class LSTMMLP_SOC(nn.Module):
    def __init__(self, in_features, hidden_size, mlp_hidden, num_layers=1, dropout=0.0, sigmoid_head=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        head = [nn.Linear(hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1)]
        if sigmoid_head:
            head.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*head)

    def forward(self, x, state=None, return_state=False):
        out, new_state = self.lstm(x, state)
        last = out[:, -1, :]
        pred = self.mlp(last).squeeze(-1)
        if return_state:
            return pred, new_state
        return pred


def load_ckpt(path: Path, device: torch.device):
    raw = torch.load(path, map_location=device)
    sd = raw.get("model_state_dict") or raw
    h = sd["lstm.weight_hh_l0"].shape[1]
    in_f = sd["lstm.weight_ih_l0"].shape[1]
    mlp_h = sd["mlp.0.weight"].shape[0]
    model = LSTMMLP_SOC(in_f, h, mlp_h).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    features = raw.get("features", FEATURES)
    chunk = raw.get("chunk") or raw.get("seq_len") or raw.get("window") or 1
    return model, list(features), int(chunk), h


def map_features(df: pd.DataFrame, features):
    cols = list(df.columns)
    mapped = []
    for f in features:
        if f in cols:
            mapped.append(f)
            continue
        # degree symbol robustness
        if "Temperature" in f:
            for c in cols:
                if "Temperature" in c:
                    mapped.append(c)
                    break
            else:
                raise KeyError(f"Feature {f} not in dataframe")
        else:
            raise KeyError(f"Feature {f} not in dataframe")
    return mapped


def predict_python(model, Xs: np.ndarray, device: torch.device, limit: int):
    preds = []
    state = None
    target = Xs.shape[0] if limit <= 0 else min(limit, Xs.shape[0])
    pbar = tqdm(total=target, desc="py-fp32", unit="step", mininterval=0.5, smoothing=0, leave=True, ascii=True, file=sys.stdout)
    i = 0
    with torch.no_grad():
        while i < target:
            x = torch.from_numpy(Xs[i]).view(1, 1, -1).to(device)
            y, state = model(x, state=state, return_state=True)
            preds.append(float(y.squeeze().cpu().item()))
            i += 1
            pbar.update(1)
            if i % 5000 == 0:
                tqdm.write(f"[py] {i}/{target}", file=sys.stdout)
    pbar.close()
    return np.asarray(preds, dtype=np.float32)


def predict_c_float(lib_path: Path, Xs: np.ndarray, hidden_size: int, limit: int):
    lib = ctypes.CDLL(str(lib_path))
    class LSTMState(ctypes.Structure):
        _fields_ = [("h", ctypes.c_float * hidden_size), ("c", ctypes.c_float * hidden_size)]
    class LSTMModel(ctypes.Structure):
        _fields_ = [("state", LSTMState), ("initialized", ctypes.c_int)]
    lib.lstm_model_init.argtypes = [ctypes.POINTER(LSTMModel)]
    lib.lstm_model_inference.argtypes = [ctypes.POINTER(LSTMModel), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.lstm_model_inference.restype = None
    lib.lstm_model_init.restype = None
    model = LSTMModel()
    lib.lstm_model_init(ctypes.byref(model))
    target = Xs.shape[0] if limit <= 0 else min(limit, Xs.shape[0])
    preds = np.zeros(target, dtype=np.float32)
    pbar = tqdm(total=target, desc="c-float", unit="step", mininterval=0.5, smoothing=0, leave=True, ascii=True, file=sys.stdout)
    for i in range(target):
        x = (ctypes.c_float * Xs.shape[1])(*Xs[i].tolist())
        out = ctypes.c_float()
        lib.lstm_model_inference(ctypes.byref(model), x, ctypes.byref(out))
        preds[i] = out.value
        if (i + 1) % 5000 == 0:
            tqdm.write(f"[c-float] {i+1}/{target}", file=sys.stdout)
        pbar.update(1)
    pbar.close()
    return preds


def predict_c_quant(lib_path: Path, X_raw: np.ndarray, limit: int):
    lib = ctypes.CDLL(str(lib_path))
    from ctypes import c_float
    INPUT_SIZE = X_raw.shape[1]
    HIDDEN_SIZE = 64  # defined in model_weights_lstm_int8_manual.h
    class LSTMState(ctypes.Structure):
        _fields_ = [("h", c_float * HIDDEN_SIZE), ("c", c_float * HIDDEN_SIZE)]
    lib.lstm_model_init.argtypes = [ctypes.POINTER(LSTMState)]
    lib.lstm_model_forward.argtypes = [(c_float * INPUT_SIZE), ctypes.POINTER(LSTMState)]
    lib.lstm_model_init.restype = None
    lib.lstm_model_forward.restype = c_float
    state = LSTMState()
    lib.lstm_model_init(ctypes.byref(state))
    target = X_raw.shape[0] if limit <= 0 else min(limit, X_raw.shape[0])
    preds = np.zeros(target, dtype=np.float32)
    pbar = tqdm(total=target, desc="c-quant", unit="step", mininterval=0.5, smoothing=0, leave=True, ascii=True, file=sys.stdout)
    for i in range(target):
        xin = (c_float * INPUT_SIZE)(*X_raw[i].tolist())
        y = lib.lstm_model_forward(xin, ctypes.byref(state))
        preds[i] = float(y)
        if (i + 1) % 5000 == 0:
            tqdm.write(f"[c-quant] {i+1}/{target}", file=sys.stdout)
        pbar.update(1)
    pbar.close()
    return preds


def overlay_plot(y, preds_dict, out_path: Path, max_points: int = 20000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(y)
    step = max(1, math.ceil(n / max_points))
    y_d = y[::step]
    plt.figure(figsize=(12, 4))
    plt.plot(y_d, label="GT", linewidth=1.0, alpha=0.9, color="black")
    colors = {
        "Base Py": "#1f77b4",
        "Base C": "#1f77b4",
        "Pruned Py": "#9467bd",
        "Pruned C": "#9467bd",
        "Quant C": "#2ca02c",
    }
    styles = {
        "Base Py": "-",
        "Base C": "--",
        "Pruned Py": "-",
        "Pruned C": "--",
        "Quant C": "-.",
    }
    for name, arr in preds_dict.items():
        plt.plot(arr[::step], label=name, linewidth=0.9, alpha=0.8, color=colors.get(name), linestyle=styles.get(name, "-"))
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("SOC")
    plt.title(f"SOC first {n} steps (downsample x{step})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT_DEF)
    ap.add_argument("--cell", type=str, default="MGFarm_18650_C07")
    ap.add_argument("--limit", type=int, default=20000, help="Number of steps (<=0 for full)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cuda" if args.device == "cuda" else "cpu"))

    # Paths and output
    out_dir = REPO / "5_benchmark/PC/SOC" / f"BENCH_SOC_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    parquet_path = args.data_root / f"df_FE_{args.cell.split('_')[-1]}.parquet"
    if not parquet_path.exists():
        parquet_path = args.data_root / f"df_FE_{args.cell}.parquet"
    df = pd.read_parquet(parquet_path)
    cols = map_features(df, FEATURES)
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["SOC"])
    if args.limit > 0:
        clean = clean.iloc[: args.limit + 10]  # a few extra in case of drop
    X_raw = clean[cols].to_numpy(dtype=np.float32)
    y_true = clean["SOC"].to_numpy(dtype=np.float32)

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    # Load models
    base_model, base_feats, _, base_hidden = load_ckpt(BASE_CKPT, device)
    pruned_model, pruned_feats, _, pruned_hidden = load_ckpt(PRUNED_CKPT, device)
    assert base_feats == pruned_feats == FEATURES, f"Feature mismatch: {base_feats} vs {pruned_feats}"

    # Python preds
    py_base = predict_python(base_model, X_scaled, device, args.limit)
    py_pruned = predict_python(pruned_model, X_scaled, device, args.limit)

    # C preds (float)
    c_base = predict_c_float(LIB_BASE, X_scaled, base_hidden, args.limit)
    c_pruned = predict_c_float(LIB_PRUNED, X_scaled, pruned_hidden, args.limit)

    # C quant (uses raw input, scales inside)
    c_quant = predict_c_quant(LIB_QUANT, X_raw, args.limit)

    n = min(len(y_true), len(py_base), len(py_pruned), len(c_base), len(c_pruned), len(c_quant))
    y = y_true[:n]
    preds = {
        "Base Py": py_base[:n],
        "Base C": c_base[:n],
        "Pruned Py": py_pruned[:n],
        "Pruned C": c_pruned[:n],
        "Quant C": c_quant[:n],
    }

    # Metrics
    rows = []
    for name, arr in preds.items():
        mae, rmse = compute_metrics(y, arr)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse})
    # Pairwise python vs C deltas
    rows.append({
        "Model": "Base C vs Py",
        "MAE": float(mean_absolute_error(preds["Base Py"], preds["Base C"])),
        "RMSE": float(math.sqrt(mean_squared_error(preds["Base Py"], preds["Base C"])))
    })
    rows.append({
        "Model": "Pruned C vs Py",
        "MAE": float(mean_absolute_error(preds["Pruned Py"], preds["Pruned C"])),
        "RMSE": float(math.sqrt(mean_squared_error(preds["Pruned Py"], preds["Pruned C"])))
    })

    # Save
    np.savez_compressed(out_dir / "arrays.npz", y=y, **{k.replace(' ', '_').lower(): v for k, v in preds.items()})
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    if not args.no_plots:
        overlay_plot(y, preds, out_dir / "overlay_firstN.png")
        # Error plots
        plt.figure(figsize=(10, 4))
        for name, arr in preds.items():
            plt.plot(arr - y, label=name, linewidth=0.7, alpha=0.8)
        plt.axhline(0.0, color="k", linestyle="--", alpha=0.4)
        plt.legend()
        plt.title("Error (pred - GT)")
        plt.tight_layout()
        plt.savefig(out_dir / "errors_firstN.png", dpi=140)
        plt.close()

    # Print table
    print("\nModel\tMAE\tRMSE")
    for r in rows:
        print(f"{r['Model']}\t{r['MAE']:.6f}\t{r['RMSE']:.6f}")
    print(f"\n[done] Saved benchmark to {out_dir}")


if __name__ == "__main__":
    main()
