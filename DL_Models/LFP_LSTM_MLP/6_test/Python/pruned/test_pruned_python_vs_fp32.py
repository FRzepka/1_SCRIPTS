#!/usr/bin/env python3
"""
Compare PRUNED FP32 model vs FP32 base reference on PC (no C compiler).

Steps:
 - Load base FP32 checkpoint (1.5.0.0) and pruned checkpoint
 - Apply RobustScaler to raw inputs
 - Stream through both LSTM+MLP paths
 - Save overlay plots + metrics, and report approximate C weight memory

Outputs: DL_Models/LFP_LSTM_MLP/6_test/Python/pruned/PC_PY_PRUNED_YYYYMMDD_HHMMSS
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[5]
CFG_PATH = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '1.5.0.0' / 'config' / 'train_soc.yaml'
SCALER_PATH = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '1.5.0.0' / 'outputs' / 'scaler_robust.joblib'


def build_ref(sd: dict, In: int, H: int):
    cell = nn.LSTMCell(input_size=In, hidden_size=H)
    with torch.no_grad():
        cell.weight_ih.copy_(sd['lstm.weight_ih_l0'])
        cell.weight_hh.copy_(sd['lstm.weight_hh_l0'])
        cell.bias_ih.copy_(sd['lstm.bias_ih_l0'])
        cell.bias_hh.copy_(sd['lstm.bias_hh_l0'])
    cell.eval()
    mlp0_w = sd['mlp.0.weight'].numpy(); mlp0_b = sd['mlp.0.bias'].numpy()
    mlp1_w = sd['mlp.3.weight'].numpy(); mlp1_b = sd['mlp.3.bias'].numpy()
    def mlp(h: np.ndarray) -> float:
        x = h @ mlp0_w.T + mlp0_b
        x = np.maximum(0.0, x)
        x = x @ mlp1_w.T + mlp1_b  # shape (1,)
        y = 1.0 / (1.0 + np.exp(-x))
        return float(np.squeeze(y))
    return cell, mlp


def run_ckpt(sd: dict, Xs: np.ndarray) -> np.ndarray:
    H = int(sd['lstm.weight_hh_l0'].shape[1])
    In = int(sd['lstm.weight_ih_l0'].shape[1])
    cell, mlp = build_ref(sd, In, H)
    h = np.zeros((H,), dtype=np.float32)
    c = np.zeros((H,), dtype=np.float32)
    out = []
    for t in range(Xs.shape[0]):
        x = Xs[t]
        h_t, c_t = cell(torch.from_numpy(x).unsqueeze(0), (torch.from_numpy(h).unsqueeze(0), torch.from_numpy(c).unsqueeze(0)))
        h = h_t.detach().squeeze(0).numpy().astype(np.float32)
        c = c_t.detach().squeeze(0).numpy().astype(np.float32)
        out.append(mlp(h))
    return np.array(out, dtype=np.float32)


def approx_c_bytes(sd: dict) -> int:
    # 4H*In + 4H*H + 4H + M*H + M + 1*M + 1
    w_ih = sd['lstm.weight_ih_l0'].shape
    w_hh = sd['lstm.weight_hh_l0'].shape
    b = sd['lstm.bias_ih_l0'].shape
    mlp0_w = sd['mlp.0.weight'].shape
    mlp0_b = sd['mlp.0.bias'].shape
    mlp1_w = sd['mlp.3.weight'].shape
    mlp1_b = sd['mlp.3.bias'].shape
    total_floats = np.prod(w_ih) + np.prod(w_hh) + np.prod(b) + np.prod(mlp0_w) + np.prod(mlp0_b) + np.prod(mlp1_w) + np.prod(mlp1_b)
    return int(total_floats * 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', default=str(BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '2_models' / 'base' / '1.5.0.0_soc_epoch0001_rmse0.02897.pt'))
    ap.add_argument('--pruned_ckpt', default=str(BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '2_models' / 'pruned' / 'soc_1.5.0.0' / 'prune_30pct_20250916_140404' / 'soc_pruned_hidden45.pt'))
    ap.add_argument('--parquet', default=r'C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet')
    ap.add_argument('--num_samples', type=int, default=5000)
    ap.add_argument('--out_dir', default='')
    args = ap.parse_args()

    base_sd = torch.load(args.base_ckpt, map_location='cpu')['model_state_dict']
    pruned_sd = torch.load(args.pruned_ckpt, map_location='cpu')
    pruned_sd = pruned_sd['model_state_dict'] if 'model_state_dict' in pruned_sd else pruned_sd

    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scaler = joblib.load(SCALER_PATH)
    feats = cfg['model']['features']
    df = pd.read_parquet(args.parquet)
    X_raw = df[feats].values[:args.num_samples].astype(np.float32)
    y_true = df['SOC'].values[:args.num_samples].astype(np.float32)
    Xs = scaler.transform(X_raw).astype(np.float32)

    preds_base = run_ckpt(base_sd, Xs)
    preds_pruned = run_ckpt(pruned_sd, Xs)

    n = min(len(preds_base), len(preds_pruned))
    preds_base = preds_base[:n]
    preds_pruned = preds_pruned[:n]
    y_true = y_true[:n]
    diffs = np.abs(preds_base - preds_pruned)
    mae = float(np.mean(diffs))
    rmse = float(np.sqrt(np.mean(diffs**2)))
    mxd = float(np.max(diffs))

    out_dir = Path(args.out_dir) if args.out_dir else (BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'Python' / 'pruned' / f'PC_PY_PRUNED_{time.strftime("%Y%m%d_%H%M%S")}')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics.json').write_text(json.dumps({
        'N': int(n),
        'MAE_base_vs_pruned': mae,
        'RMSE_base_vs_pruned': rmse,
        'MAX_base_vs_pruned': mxd,
        'C_bytes_base': approx_c_bytes(base_sd),
        'C_bytes_pruned': approx_c_bytes(pruned_sd),
    }, indent=2))

    plt.figure(figsize=(12,4)); plt.plot(y_true, label='GT', alpha=0.6); plt.plot(preds_base, label='FP32'); plt.plot(preds_pruned, label='PRUNED'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'overlay_full.png', dpi=150); plt.close()
    firstN = min(500, n)
    plt.figure(figsize=(12,4)); plt.plot(y_true[:firstN], label='GT', alpha=0.6); plt.plot(preds_base[:firstN], label='FP32'); plt.plot(preds_pruned[:firstN], label='PRUNED'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'overlay_firstN.png', dpi=150); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(diffs, bins=100, alpha=0.8, color='tab:purple', edgecolor='black'); plt.title('Abs diff PRUNED vs FP32'); plt.tight_layout(); plt.savefig(out_dir/'diff_hist.png', dpi=150); plt.close()

    print(f'Saved results to: {out_dir}')


if __name__ == '__main__':
    main()
