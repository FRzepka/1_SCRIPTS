"""
Build and test the Base FP32 C implementation (lstm_model.c) on the PC and compare
against the PyTorch FP32 reference over N sequential samples.

This isolates whether the issue is on-device (UART/timing/printf) or in the C math itself.

Outputs: overlay plots + metrics in 6_test/STM32/base/PC_C_BASE_YYYYMMDD_HHMMSS
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[5]  # repo root (1_Scripts)
CFG_PATH = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '1.5.0.0' / 'config' / 'train_soc.yaml'
SCALER_PATH = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '1.5.0.0' / 'outputs' / 'scaler_robust.joblib'


def build_fp32_reference(state_dict: dict, input_size: int, hidden_size: int):
    cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
    with torch.no_grad():
        cell.weight_ih.copy_(state_dict['lstm.weight_ih_l0'])
        cell.weight_hh.copy_(state_dict['lstm.weight_hh_l0'])
        cell.bias_ih.copy_(state_dict['lstm.bias_ih_l0'])
        cell.bias_hh.copy_(state_dict['lstm.bias_hh_l0'])
    cell.eval()
    mlp0_w = state_dict['mlp.0.weight'].numpy(); mlp0_b = state_dict['mlp.0.bias'].numpy()
    mlp1_w = state_dict['mlp.3.weight'].numpy(); mlp1_b = state_dict['mlp.3.bias'].numpy()

    def mlp(h):
        x = h @ mlp0_w.T + mlp0_b
        x = np.maximum(0.0, x)
        x = x @ mlp1_w.T + mlp1_b
        return 1.0 / (1.0 + np.exp(-x))

    return cell, mlp


def compile_c_base(c_dir: Path, out_dir: Path) -> Path:
    """Compile a small C app that wraps lstm_model.c and prints SOC for each input line.

    IMPORTANT: Mirror STM32 path:
      - Keep one persistent LSTMModel instance (stateful across steps)
      - Apply RobustScaler (scaler_transform) before inference
    """
    main_c = out_dir / 'test_main_base.c'
    exe = out_dir / ('test_base_fp32.exe' if sys.platform == 'win32' else 'test_base_fp32')
    main_c.write_text(r'''
#include <stdio.h>
#include "lstm_model.h"
#include "model_weights.h"
#include "scaler_params.h"

int main(){
  float in[INPUT_SIZE];
  float in_scaled[INPUT_SIZE];
  LSTMModel model; lstm_model_init(&model);
  for(;;){
    int n = scanf("%f %f %f %f %f %f", &in[0],&in[1],&in[2],&in[3],&in[4],&in[5]);
    if(n!=INPUT_SIZE) break;
    float y=0.0f;
    // Mirror device: scale then stateful inference
    scaler_transform(in, in_scaled);
    lstm_model_inference(&model, in_scaled, &y);
    printf("SOC: %.6f\n", y);
    fflush(stdout);
  }
  return 0;
}
''')
    # Try GCC/Clang/MSVC
    cmd = ['gcc', '-O2', '-I', str(c_dir), str(main_c), str(c_dir / 'lstm_model.c'), '-o', str(exe), '-lm']
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return exe
    except subprocess.CalledProcessError as cpe:
        # Print useful debug info for GCC failure
        print("\n--- C compile failed (gcc) ---")
        print('CMD:', ' '.join(cmd))
        print('PATH:', os.environ.get('PATH'))
        try:
            stdout = cpe.stdout.decode('utf-8', errors='replace') if cpe.stdout is not None else ''
            stderr = cpe.stderr.decode('utf-8', errors='replace') if cpe.stderr is not None else ''
            print('--- stdout ---')
            print(stdout)
            print('--- stderr ---')
            print(stderr)
        except Exception:
            pass
        # Try Clang
        try:
            cmd = ['clang', '-O2', '-I', str(c_dir), str(main_c), str(c_dir / 'lstm_model.c'), '-o', str(exe)]
            subprocess.run(cmd, check=True, capture_output=True)
            return exe
        except subprocess.CalledProcessError as cpe2:
            print("\n--- C compile failed (clang) ---")
            print('CMD:', ' '.join(cmd))
            try:
                print(cpe2.stdout.decode('utf-8', errors='replace'))
                print(cpe2.stderr.decode('utf-8', errors='replace'))
            except Exception:
                pass
        except Exception:
            pass
        # Try MSVC cl
        cl = shutil.which('cl')
        if cl:
            try:
                cmd = ['cl', '/O2', f'/I{str(c_dir)}', str(main_c), str(c_dir / 'lstm_model.c'), '/Fe:'+str(exe)]
                subprocess.run(cmd, check=True)
                return exe
            except subprocess.CalledProcessError as cpe3:
                print("\n--- C compile failed (cl) ---")
                print('CMD:', ' '.join(cmd))
        raise RuntimeError(f"C compiler not found or failed: {cpe}")


def run_c_exe(exe: Path, X_raw: np.ndarray) -> np.ndarray:
    lines = [' '.join(f'{v:.8f}' for v in row) for row in X_raw]
    payload = '\n'.join(lines)+'\n'
    res = subprocess.run([str(exe)], input=payload, text=True, capture_output=True, check=True)
    ys = []
    for line in res.stdout.strip().splitlines():
        line=line.strip()
        if 'SOC' in line.upper():
            toks = line.replace(':',' ').split()
            for t in toks:
                try:
                    ys.append(float(t)); break
                except: pass
    return np.array(ys, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='FP32 base checkpoint (.pt)')
    ap.add_argument('--num_samples', type=int, default=5000)
    ap.add_argument('--parquet', type=str, default='')
    ap.add_argument('--out_dir', type=str, default='')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = ckpt['model_state_dict']
    H = int(sd['lstm.weight_hh_l0'].shape[1])
    In = int(sd['lstm.weight_ih_l0'].shape[1])

    # Data
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scaler = joblib.load(SCALER_PATH)
    feat_names = cfg['model']['features']
    # Pick a default parquet like earlier scripts; user should pass via their own harness if needed
    # Here we rely on user's run script for real UART runs; for PC C vs FP32 we just need any DF
    # We'll require the caller to prepare the Parquet as needed; using the same path as UART tests is fine.

    # For simplicity, reuse df path from earlier runs by prompting minimal input:
    # But to keep non-interactive, we fall back to same file used in your UART tests if available
    df_path = Path(args.parquet) if args.parquet else Path(r'C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet')
    if not df_path.exists():
        raise FileNotFoundError(f'Parquet not found: {df_path}. Pass --parquet PATH to override.')
    df = pd.read_parquet(df_path)
    missing = [c for c in feat_names if c not in df.columns]
    if missing:
        raise KeyError(f'Missing columns in parquet: {missing}')
    X_raw = df[feat_names].values[:args.num_samples].astype(np.float32)
    y_true = df['SOC'].values[:args.num_samples].astype(np.float32)
    Xs = scaler.transform(X_raw).astype(np.float32)

    # FP32 reference (streaming)
    cell, mlp = build_fp32_reference(sd, In, H)
    h = np.zeros((H,), dtype=np.float32)
    c = np.zeros((H,), dtype=np.float32)
    preds_fp32 = []
    for t in range(len(Xs)):
        x = Xs[t]
        h_t, c_t = cell(torch.from_numpy(x).unsqueeze(0), (torch.from_numpy(h).unsqueeze(0), torch.from_numpy(c).unsqueeze(0)))
        h = h_t.detach().squeeze(0).numpy().astype(np.float32)
        c = c_t.detach().squeeze(0).numpy().astype(np.float32)
        y = mlp(h)[0]
        preds_fp32.append(float(y))
    preds_fp32 = np.array(preds_fp32, dtype=np.float32)

    # Build & run C model
    c_dir = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '2_models' / 'base' / 'c_implementation'
    out_dir = Path(args.out_dir) if args.out_dir else (BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'STM32' / 'base' / f'PC_C_BASE_{time.strftime("%Y%m%d_%H%M%S")}')
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = compile_c_base(c_dir, out_dir)
    preds_c = run_c_exe(exe, X_raw)

    # Metrics
    n = min(len(preds_c), len(preds_fp32))
    preds_c = preds_c[:n]
    preds_fp32 = preds_fp32[:n]
    y_true = y_true[:n]
    diffs = np.abs(preds_fp32 - preds_c)
    mae = float(np.mean(np.abs(preds_fp32 - preds_c)))
    rmse = float(np.sqrt(np.mean((preds_fp32 - preds_c)**2)))
    mxd = float(np.max(diffs))

    (out_dir / 'metrics.json').write_text(json.dumps({'MAE': mae, 'RMSE': rmse, 'MAX': mxd, 'N': int(n)}, indent=2))

    # Plots
    plt.figure(figsize=(12,4)); plt.plot(y_true, label='GT', alpha=0.6); plt.plot(preds_fp32, label='FP32'); plt.plot(preds_c, label='C_BASE'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'overlay_full.png', dpi=150); plt.close()
    firstN = min(500, n)
    plt.figure(figsize=(12,4)); plt.plot(y_true[:firstN], label='GT', alpha=0.6); plt.plot(preds_fp32[:firstN], label='FP32'); plt.plot(preds_c[:firstN], label='C_BASE'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'overlay_firstN.png', dpi=150); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(diffs, bins=100, alpha=0.8, color='tab:blue', edgecolor='black'); plt.title('Abs diff C_BASE vs FP32'); plt.tight_layout(); plt.savefig(out_dir/'diff_hist.png', dpi=150); plt.close()

    print(f"Saved results to: {out_dir}")


if __name__ == '__main__':
    main()
