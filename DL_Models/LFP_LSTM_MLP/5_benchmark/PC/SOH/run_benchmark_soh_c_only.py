#!/usr/bin/env python3
"""
SOH Benchmark (PC, C-only):
- Base C (FP32)
- Pruned C (FP32)
- Quant C (INT8 LSTM + FP32 MLP)

Outputs:
- BENCH_SOH_<timestamp>/arrays.npz (y, base_c, pruned_c, quant_c)
- metrics.json (MAE/RMSE)
- overlay_full.png, error_hist.png, model_sizes.png, streaming_dashboard.png
"""
import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Line buffered stdout for live tqdm
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parents[3]  # .../DL_Models/LFP_LSTM_MLP
DATA_ROOT_DEF = Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")

# Paths
BASE_SO = REPO / "2_models/base/soh_2.1.0.0_base/c_implementation/liblstm_soh_base.so"
PRUNED_SO = REPO / "2_models/pruned/soh_2.1.0.0/prune_30pct_20251122_010142/c_implementation/liblstm_soh_pruned.so"
QUANT_SO = REPO / "2_models/quantized/soh_2.1.0.0_quantized/c_implementation/liblstm_soh_quant.so"
SCALER_PATH = REPO / "1_training/2.1.0.0/outputs/scaler_robust.joblib"

FEATURES = ["Testtime[s]", "Voltage[V]", "Current[A]", "Temperature[°C]", "EFC", "Q_c"]


def map_features(df: pd.DataFrame, feats: list):
    cols = list(df.columns)
    mapped = []
    for f in feats:
        if f in cols:
            mapped.append(f)
            continue
        # handle degree symbol encoding
        if "Temperature" in f:
            hit = None
            for c in cols:
                if "Temperature" in c:
                    hit = c
                    break
            if hit:
                mapped.append(hit)
                continue
        raise KeyError(f"Feature {f} not in dataframe")
    return mapped


def predict_c_base_pruned(lib_path: Path, X_raw: np.ndarray, hidden_size: int, desc: str, limit: int, chunk_size: int = 100000):
    lib = ctypes.CDLL(str(lib_path))
    class LSTMState(ctypes.Structure):
        _fields_ = [("h", ctypes.c_float * hidden_size), ("c", ctypes.c_float * hidden_size)]
    class LSTMModel(ctypes.Structure):
        _fields_ = [("state", LSTMState), ("initialized", ctypes.c_int)]
    lib.lstm_model_soh_init.argtypes = [ctypes.POINTER(LSTMModel)]
    lib.lstm_model_soh_inference_batch.argtypes = [ctypes.POINTER(LSTMModel), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.lstm_model_soh_init.restype = None
    lib.lstm_model_soh_inference_batch.restype = None

    model = LSTMModel()
    lib.lstm_model_soh_init(ctypes.byref(model))

    n = X_raw.shape[0] if limit <= 0 else min(limit, X_raw.shape[0])
    out = np.zeros(n, dtype=np.float32)
    
    pbar = tqdm(total=n, desc=desc, unit="step", mininterval=0.5, leave=True, ascii=True, smoothing=0, file=sys.stdout)
    
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        count = end - i
        inp_chunk = X_raw[i:end].astype(np.float32).ravel()
        
        # Pointer arithmetic for output buffer
        out_ptr = ctypes.cast(out.ctypes.data + i * ctypes.sizeof(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
        
        lib.lstm_model_soh_inference_batch(
            ctypes.byref(model),
            inp_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_ptr,
            count,
        )
        pbar.update(count)
        
    pbar.close()
    return out


def predict_c_quant(lib_path: Path, X_raw: np.ndarray, hidden_size: int, desc: str, limit: int, chunk_size: int = 100000):
    lib = ctypes.CDLL(str(lib_path))
    from ctypes import c_float
    
    class LSTMState(ctypes.Structure):
        _fields_ = [("h", c_float * hidden_size), ("c", c_float * hidden_size)]
        
    # Try to find batch function first
    has_batch = hasattr(lib, 'lstm_model_soh_int8_inference_batch')
    
    if has_batch:
        lib.lstm_model_soh_int8_inference_batch.argtypes = [ctypes.POINTER(c_float), ctypes.POINTER(LSTMState), ctypes.POINTER(c_float), ctypes.c_int]
        lib.lstm_model_soh_int8_inference_batch.restype = None
    else:
        # Fallback (should not happen if lib is correct)
        print(f"Warning: Batch function not found in {lib_path}, falling back to slow loop.")
        lib.lstm_model_soh_int8_init.argtypes = [ctypes.POINTER(LSTMState)]
        lib.lstm_model_soh_int8_forward.argtypes = [(c_float * X_raw.shape[1]), ctypes.POINTER(LSTMState)]
        lib.lstm_model_soh_int8_init.restype = None
        lib.lstm_model_soh_int8_forward.restype = c_float

    state = LSTMState()
    # Init state (memset 0)
    ctypes.memset(ctypes.addressof(state), 0, ctypes.sizeof(state))

    n = X_raw.shape[0] if limit <= 0 else min(limit, X_raw.shape[0])
    out = np.zeros(n, dtype=np.float32)
    pbar = tqdm(total=n, desc=desc, unit="step", mininterval=0.5, leave=True, ascii=True, smoothing=0, file=sys.stdout)
    
    if has_batch:
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            count = end - i
            inp_chunk = X_raw[i:end].astype(np.float32).ravel()
            out_ptr = ctypes.cast(out.ctypes.data + i * ctypes.sizeof(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
            
            lib.lstm_model_soh_int8_inference_batch(
                inp_chunk.ctypes.data_as(ctypes.POINTER(c_float)),
                ctypes.byref(state),
                out_ptr,
                count
            )
            pbar.update(count)
    else:
        INPUT_SIZE = X_raw.shape[1]
        for i in range(n):
            xin = (c_float * INPUT_SIZE)(*X_raw[i].tolist())
            y = lib.lstm_model_soh_int8_forward(xin, ctypes.byref(state))
            out[i] = float(y)
            if (i + 1) % 5000 == 0:
                pbar.update(5000)
        pbar.update(n % 5000)
        
    pbar.close()
    return out


def compute_metrics(y, pred):
    mae = float(mean_absolute_error(y, pred))
    rmse = float(math.sqrt(mean_squared_error(y, pred)))
    return mae, rmse


def plot_overlay(y, preds, out_path: Path, max_points: int = 200000):
    n = len(y)
    step = max(1, math.ceil(n / max_points))
    plt.figure(figsize=(12, 4))
    plt.plot(y[::step], label="GT", linewidth=1.0, alpha=0.9, color="black")
    colors = {"Base C": "#1f77b4", "Pruned C": "#9467bd", "Quant C": "#2ca02c"}
    for name, arr in preds.items():
        plt.plot(arr[::step], label=name, linewidth=0.9, alpha=0.85, color=colors.get(name))
    plt.legend(); plt.grid(alpha=0.2)
    plt.xlabel("step"); plt.ylabel("SOH")
    plt.title(f"SOH streaming (first {n} steps, downsample x{step})")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_error_hist(y, preds, out_path: Path, bins: int = 120, max_samples: int = 200000):
    plt.figure(figsize=(7, 4))
    for name, arr in preds.items():
        err = arr - y
        if len(err) > max_samples:
            idx = np.random.RandomState(0).choice(len(err), size=max_samples, replace=False)
            err = err[idx]
        plt.hist(err, bins=bins, alpha=0.4, label=name, histtype="stepfilled")
    plt.xlabel("Error (pred - GT)"); plt.ylabel("count")
    plt.title("SOH error distribution"); plt.legend(); plt.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_model_sizes(info: dict, out_path: Path):
    names = list(info.keys())
    params = [info[n]["params_m"] for n in names]
    flash = [info[n]["flash_kb"] for n in names]
    so = [info[n]["so_kb"] for n in names]
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(x - w, params, width=w, label="Params [M]")
    ax.bar(x, flash, width=w, label="Est. flash (params*4B) [KB]")
    ax.bar(x + w, so, width=w, label=".so size [KB]")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("Model size comparison (SOH)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_dashboard(y, preds, out_path: Path):
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axs[0].plot(y, label="GT", color="black", linewidth=1.0)
    for name, arr in preds.items():
        axs[0].plot(arr, label=name, linewidth=0.9)
    axs[0].legend(); axs[0].set_ylabel("SOH"); axs[0].set_title("SOH streaming")
    colors = {"Base C": "#1f77b4", "Pruned C": "#9467bd", "Quant C": "#2ca02c"}
    for name, arr in preds.items():
        axs[1].plot(arr - y, label=name, linewidth=0.8, color=colors.get(name))
    axs[1].axhline(0, color="k", linestyle="--", alpha=0.3)
    axs[1].set_ylabel("Error"); axs[1].legend(); axs[1].grid(alpha=0.2)
    axs[2].plot(np.abs(preds["Base C"] - preds["Quant C"]), label="|BaseC - QuantC|", linewidth=0.8, color="tab:red")
    axs[2].set_ylabel("Abs diff"); axs[2].set_xlabel("step"); axs[2].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT_DEF)
    ap.add_argument("--cell", type=str, default="MGFarm_18650_C07")
    ap.add_argument("--limit", type=int, default=20000, help="steps (<=0 for full)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / "5_benchmark/PC/SOH" / f"BENCH_SOH_C_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet = args.data_root / f"df_FE_{args.cell.split('_')[-1]}.parquet"
    if not parquet.exists():
        parquet = args.data_root / f"df_FE_{args.cell}.parquet"
    df = pd.read_parquet(parquet)
    cols = map_features(df, FEATURES)
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["SOH"])
    
    # Sort by time to ensure correct sequence for LSTM
    if "Testtime[s]" in clean.columns:
        print("Sorting data by Testtime[s]...", flush=True)
        clean = clean.sort_values("Testtime[s]")
    
    if args.limit > 0:
        clean = clean.iloc[: args.limit + 10]
    X_raw = clean[cols].to_numpy(dtype=np.float32)
    y_true = clean["SOH"].to_numpy(dtype=np.float32)

    # Predictions
    preds = {}
    preds["Base C"] = predict_c_base_pruned(BASE_SO, X_raw, hidden_size=128, desc="base-c", limit=args.limit)
    preds["Pruned C"] = predict_c_base_pruned(PRUNED_SO, X_raw, hidden_size=90, desc="pruned-c", limit=args.limit)
    preds["Quant C"] = predict_c_quant(QUANT_SO, X_raw, hidden_size=128, desc="quant-c", limit=args.limit)

    n = min(len(y_true), *(len(v) for v in preds.values()))
    y = y_true[:n]
    preds = {k: v[:n] for k, v in preds.items()}

    # Metrics
    rows = []
    for name, arr in preds.items():
        mae, rmse = compute_metrics(y, arr)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse})
    # Save arrays
    np.savez_compressed(out_dir / "arrays.npz", y=y, base_c=preds["Base C"], pruned_c=preds["Pruned C"], quant_c=preds["Quant C"])
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    # Sizes
    def est_params(hidden):
        inp = 6
        mlp = 64
        lstm_params = 4 * hidden * inp + 4 * hidden * hidden + 8 * hidden
        mlp_params = mlp * hidden + mlp + mlp + 1
        return lstm_params + mlp_params
    size_info = {
        "Base": {
            "params_m": est_params(128) / 1e6,
            "flash_kb": est_params(128) * 4 / 1024.0,
            "so_kb": BASE_SO.stat().st_size / 1024.0,
        },
        "Pruned": {
            "params_m": est_params(90) / 1e6,
            "flash_kb": est_params(90) * 4 / 1024.0,
            "so_kb": PRUNED_SO.stat().st_size / 1024.0,
        },
        "Quant": {
            "params_m": est_params(128) / 1e6,  # logical params
            "flash_kb": (4 * 128 * 6 + 4 * 128 * 128) / 1024.0 + (64 * 128 + 2 * 64 + 1) * 4 / 1024.0,  # rough: int8 LSTM + fp32 MLP
            "so_kb": QUANT_SO.stat().st_size / 1024.0,
        },
    }

    if not args.no_plots:
        plot_overlay(y, preds, out_dir / "overlay_full.png")
        plot_error_hist(y, preds, out_dir / "error_hist.png")
        plot_model_sizes(size_info, out_dir / "model_sizes.png")
        plot_dashboard(y, preds, out_dir / "streaming_dashboard.png")

    # Print summary
    print("\nModel\tMAE\tRMSE")
    for r in rows:
        print(f"{r['Model']}\t{r['MAE']:.6f}\t{r['RMSE']:.6f}")
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
