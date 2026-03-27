#!/usr/bin/env python3
"""
Comprehensive SOH Benchmark: Base vs Pruned vs Quantized (Python vs C)
"""
import os
import sys
import time
import ctypes
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# --- Configuration ---
DATA_ROOT = Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
CELL = "MGFarm_18650_C07"
LIMIT = 0 # 0 for full

# Add argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=0, help='Limit number of samples (0 for full)')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
parser.add_argument('--no-calib', action='store_true', help='Disable calibration to GT[0]')
args = parser.parse_args()
LIMIT = args.limit
DEVICE = args.device

BASE_CKPT = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/2_models/base/soh_2.1.0.0_base/2.1.0.0_soh_epoch0120_rmse0.03359.pt")
PRUNED_CKPT = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/2_models/pruned/soh_2.1.0.0/prune_30pct_20251122_010142/soh_pruned_hidden90.pt")
QUANT_NPZ = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/2_models/quantized/soh_2.1.0.0_quantized/quant_state_soh_int16hh_p99_5.npz")

LIB_BASE = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/2_models/base/soh_2.1.0.0_base/c_implementation/liblstm_soh_base.so")
LIB_PRUNED = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/2_models/pruned/soh_2.1.0.0/prune_30pct_20251122_010142/c_implementation/liblstm_soh_pruned.so")
LIB_QUANT = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/2_models/quantized/soh_2.1.0.0_quantized/c_implementation/liblstm_soh_quant.so")

OUTPUT_DIR = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOH")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Filter Config ---
FILTER_CFG = {
    'rel_cap': 1e-4,
    'abs_cap': 1e-5,
    'ema_alpha': 0.02,
    'calib_start_one': True,
    'calib_kind': 'scale'
}

# --- Python Models ---
class LSTMMLP_SOH(nn.Module):
    def __init__(self, in_features, hidden_size, mlp_hidden, num_layers=1, dropout=0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1)
        )
    def forward(self, x, state=None, return_state=False):
        out, new_state = self.lstm(x, state)
        pred = self.mlp(out).squeeze(-1)
        if return_state: return pred, new_state
        return pred

def load_py_model(ckpt_path, device):
    raw = torch.load(ckpt_path, map_location=device)
    sd = raw.get('model_state_dict') or raw
    in_f = sd['lstm.weight_ih_l0'].shape[1]
    h_size = sd['lstm.weight_hh_l0'].shape[1]
    mlp_h = sd['mlp.0.weight'].shape[0]
    model = LSTMMLP_SOH(in_f, h_size, mlp_h).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, raw.get('features'), raw.get('scaler_path')

# --- C Wrappers ---
def run_c_inference(lib_path, input_data, hidden_size, is_quant=False, chunk_size=100000):
    lib = ctypes.CDLL(str(lib_path))
    n_samples, n_features = input_data.shape
    
    # Define structs dynamically to match C memory layout exactly
    class LSTMState(ctypes.Structure):
        _fields_ = [("h", ctypes.c_float * hidden_size), ("c", ctypes.c_float * hidden_size)]

    class LSTMModel(ctypes.Structure):
        _fields_ = [("state", LSTMState), ("initialized", ctypes.c_int)]
    
    # Prepare output array
    output = np.zeros(n_samples, dtype=np.float32)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    # Initialize State/Model
    if is_quant:
        # Quantized: void lstm_model_soh_int8_inference_batch(float* input_flat, LSTMStateSOH* state, float* output, int n_samples)
        state = LSTMState()
        ctypes.memset(ctypes.addressof(state), 0, ctypes.sizeof(state))
        
        lib.lstm_model_soh_int8_inference_batch.argtypes = [c_float_p, ctypes.POINTER(LSTMState), c_float_p, ctypes.c_int]
        
        # Loop in chunks
        pbar = tqdm(total=n_samples, desc=f"C Quant Inference (h={hidden_size})", unit="sample", mininterval=1.0)
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            chunk_len = end - i
            
            # Prepare chunk input
            chunk_in = input_data[i:end].astype(np.float32).ravel()
            # Prepare chunk output buffer (view into main output)
            # We need to be careful with numpy views and ctypes. 
            # Best to pass the pointer to the correct offset in the main output array.
            
            out_ptr = output.ctypes.data + (i * ctypes.sizeof(ctypes.c_float))
            
            lib.lstm_model_soh_int8_inference_batch(
                chunk_in.ctypes.data_as(c_float_p),
                ctypes.byref(state),
                ctypes.cast(out_ptr, c_float_p),
                chunk_len
            )
            pbar.update(chunk_len)
        pbar.close()
        
    else:
        # Float: void lstm_model_soh_inference_batch(LSTMModelSOH* model, const float* input_flat, float* output, int n_samples)
        model = LSTMModel()
        model.initialized = 0 # Will be set to 1 by C code after first run if logic dictates, or we rely on 0 init.
        # Actually C code checks initialized. If 0, it resets state. 
        # We want it to reset state at start (which it is 0).
        # Then it sets initialized=1.
        # Subsequent calls will see initialized=1 and preserve state.
        
        lib.lstm_model_soh_inference_batch.argtypes = [ctypes.POINTER(LSTMModel), c_float_p, c_float_p, ctypes.c_int]
        
        pbar = tqdm(total=n_samples, desc=f"C Float Inference (h={hidden_size})", unit="sample", mininterval=1.0)
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            chunk_len = end - i
            
            chunk_in = input_data[i:end].astype(np.float32).ravel()
            out_ptr = output.ctypes.data + (i * ctypes.sizeof(ctypes.c_float))
            
            lib.lstm_model_soh_inference_batch(
                ctypes.byref(model),
                chunk_in.ctypes.data_as(c_float_p),
                ctypes.cast(out_ptr, c_float_p),
                chunk_len
            )
            pbar.update(chunk_len)
        pbar.close()
        
    return output

LIB_FILTER = OUTPUT_DIR / "libfilter.so"

# --- Helper Functions ---
def apply_filter(preds, rel_cap, abs_cap, ema_alpha):
    if not LIB_FILTER.exists():
        # Fallback to slow python
        print("Warning: libfilter.so not found, using slow python filter")
        filtered = np.empty_like(preds)
        last = preds[0]
        ema = preds[0]
        filtered[0] = last
        for i in range(1, len(preds)):
            v = preds[i]
            caps = []
            if rel_cap: caps.append(abs(last) * rel_cap)
            if abs_cap: caps.append(abs_cap)
            if caps:
                cap = min(caps)
                delta = v - last
                if abs(delta) > cap:
                    v = last + (cap if delta > 0 else -cap)
            if ema_alpha:
                ema = ema_alpha * v + (1 - ema_alpha) * ema
                v = ema
            filtered[i] = v
            last = v
        return filtered
    
    # C Filter
    lib = ctypes.CDLL(str(LIB_FILTER))
    n = len(preds)
    output = np.zeros(n, dtype=np.float32)
    preds_c = preds.astype(np.float32)
    
    # void apply_filter_c(const float* input, float* output, int n, float rel_cap, float abs_cap, float ema_alpha)
    lib.apply_filter_c.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    
    lib.apply_filter_c(
        preds_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
        float(rel_cap) if rel_cap else 0.0,
        float(abs_cap) if abs_cap else 0.0,
        float(ema_alpha) if ema_alpha else 0.0
    )
    return output

def calibrate(preds, gt):
    if len(preds) == 0: return preds
    factor = gt[0] / preds[0] if preds[0] != 0 else 1.0
    return preds * factor

def run_py_inference_batch(model, Xs, device, chunk_size=8192):
    model.eval()
    preds = []
    state = None
    
    # Prime
    prime_len = 2048 # Match training chunk size
    if len(Xs) > prime_len:
        prime = torch.from_numpy(Xs[:prime_len]).unsqueeze(0).to(device)
        with torch.no_grad():
            _, state = model(prime, state=None, return_state=True)
        start_idx = prime_len
    else:
        start_idx = 0

    Xs_target = Xs[start_idx:]
    
    # Add tqdm
    pbar = tqdm(total=len(Xs_target), desc="Py Inference", unit="step", mininterval=1.0)
    
    with torch.no_grad():
        for i in range(0, len(Xs_target), chunk_size):
            end = min(i + chunk_size, len(Xs_target))
            x = torch.from_numpy(Xs_target[i:end]).unsqueeze(0).to(device)
            p, state = model(x, state=state, return_state=True)
            state = (state[0].detach(), state[1].detach())
            preds.append(p.squeeze(0).cpu().numpy())
            pbar.update(end - i)
            
    pbar.close()
            
    if preds:
        full_preds = np.concatenate(preds)
        # Prepend zeros or prime predictions? For simplicity, we align by cutting GT
        return full_preds, start_idx
    return np.array([]), 0

# --- Main ---
def main():
    print("Loading data...", flush=True)
    pq_path = DATA_ROOT / f"df_FE_{CELL.split('_')[-1]}.parquet"
    if not pq_path.exists(): pq_path = DATA_ROOT / f"df_FE_{CELL}.parquet"
    df = pd.read_parquet(pq_path)
    
    # Features (assume standard set from base model)
    features = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'EFC', 'Q_c'] # Hardcoded for now or load from ckpt
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['SOH'])
    
    # Sort by time to ensure correct sequence for LSTM
    if 'Testtime[s]' in df.columns:
        print("Sorting data by Testtime[s]...", flush=True)
        df = df.sort_values('Testtime[s]')
    else:
        print("Warning: Testtime[s] not found, assuming data is already sorted.", flush=True)

    if LIMIT > 0: df = df.iloc[:LIMIT]
    
    X_raw = df[features].to_numpy(dtype=np.float32)
    y_true = df['SOH'].to_numpy(dtype=np.float32)
    print(f"Data loaded: {len(X_raw)} samples", flush=True)
    
    # Scaler (Base)
    scaler_path = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/1_training/2.1.0.0/outputs/scaler_robust.joblib")
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    
    results = {}
    
    # 1. Python Base
    print(f"Running Python Base on {DEVICE}...", flush=True)
    model_base, _, _ = load_py_model(BASE_CKPT, DEVICE)
    h_base = model_base.lstm.hidden_size
    print(f"Base Hidden Size: {h_base}")
    py_base_raw, start_base = run_py_inference_batch(model_base, X_scaled, DEVICE)
    print(f"Python Base done. {len(py_base_raw)} predictions.", flush=True)
    
    # 2. Python Pruned
    print(f"Running Python Pruned on {DEVICE}...", flush=True)
    model_pruned, _, _ = load_py_model(PRUNED_CKPT, DEVICE)
    h_pruned = model_pruned.lstm.hidden_size
    print(f"Pruned Hidden Size: {h_pruned}")
    py_pruned_raw, start_pruned = run_py_inference_batch(model_pruned, X_scaled, DEVICE)
    print(f"Python Pruned done. {len(py_pruned_raw)} predictions.", flush=True)
    
    # Align start indices (C code runs from 0, but Python primes)
    # For C code, we pass full X_raw (it scales internally)
    # But wait, C code stateful also needs priming or starts at 0.
    # We will run C code on full sequence.
    
    # 3. C Base
    print(f"Running C Base (h={h_base})...", flush=True)
    c_base_raw = run_c_inference(LIB_BASE, X_raw, h_base, is_quant=False)
    print(f"C Base done. {len(c_base_raw)} predictions.", flush=True)
    
    # 4. C Pruned
    print(f"Running C Pruned (h={h_pruned})...", flush=True)
    c_pruned_raw = run_c_inference(LIB_PRUNED, X_raw, h_pruned, is_quant=False)
    print(f"C Pruned done. {len(c_pruned_raw)} predictions.", flush=True)
    
    # 5. C Quantized
    print("Running C Quantized (h=128)...", flush=True)
    c_quant_raw = run_c_inference(LIB_QUANT, X_raw, 128, is_quant=True) 
    print(f"C Quantized done. {len(c_quant_raw)} predictions.", flush=True)
    
    # Alignment
    # Python skipped 'start_base' samples. C ran all.
    # We slice C results to match Python valid range for fair comparison vs GT
    # Or we slice GT to match Python.
    
    common_start = max(start_base, start_pruned)
    
    # Slice all to common length
    y_gt = y_true[common_start:]
    
    res_map = {
        'Py_Base': py_base_raw[common_start-start_base:],
        'Py_Pruned': py_pruned_raw[common_start-start_pruned:],
        'C_Base': c_base_raw[common_start:],
        'C_Pruned': c_pruned_raw[common_start:],
        'C_Quant': c_quant_raw[common_start:]
    }
    
    # Filter & Calibrate
    print("Filtering and Calibrating...")
    final_results = {}
    metrics = {}
    
    for name, raw in res_map.items():
        # Calibrate first
        if not args.no_calib:
            cal = calibrate(raw, y_gt)
        else:
            cal = raw
            
        # Filter
        filt = apply_filter(cal, FILTER_CFG['rel_cap'], FILTER_CFG['abs_cap'], FILTER_CFG['ema_alpha'])
        final_results[name] = filt
        
        # Metrics
        mae = mean_absolute_error(y_gt, filt)
        rmse = np.sqrt(mean_squared_error(y_gt, filt))
        metrics[name] = {'MAE': float(mae), 'RMSE': float(rmse)}
        print(f"{name}: MAE={mae:.6f}, RMSE={rmse:.6f}")

    # Save
    np.savez_compressed(OUTPUT_DIR / "benchmark_results.npz", y_gt=y_gt, **final_results)
    import json
    with open(OUTPUT_DIR / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Done. Results saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
