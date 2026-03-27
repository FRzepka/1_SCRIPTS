"""
Pure-Python simulation of the Base C math (LSTM+MLP) vs PyTorch FP32 reference.

No C compiler required. Mirrors the STM32 path:
 - RobustScaler on raw inputs
 - Stateful LSTMCell step-by-step
 - MLP: Linear -> ReLU -> Linear -> Sigmoid

Outputs: overlay plots + metrics under 6_test/STM32/base/PC_PY_CSIM_BASE_YYYYMMDD_HHMMSS
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

    def mlp(h: np.ndarray) -> np.ndarray:
        x = h @ mlp0_w.T + mlp0_b
        x = np.maximum(0.0, x)
        x = x @ mlp1_w.T + mlp1_b
        return 1.0 / (1.0 + np.exp(-x))

    return cell, mlp


def csim_predict_sequence(sd: dict, X_scaled: np.ndarray) -> np.ndarray:
    """Python reimplementation of lstm_model.c math using state_dict weights.
    Equations and gate order match PyTorch: gates=[i,f,g,o].
    """
    W_ih = sd['lstm.weight_ih_l0'].numpy().astype(np.float32)  # [4H, In]
    W_hh = sd['lstm.weight_hh_l0'].numpy().astype(np.float32)  # [4H, H]
    b_ih = sd['lstm.bias_ih_l0'].numpy().astype(np.float32)    # [4H]
    b_hh = sd['lstm.bias_hh_l0'].numpy().astype(np.float32)    # [4H]
    b = (b_ih + b_hh).astype(np.float32)

    H = W_hh.shape[1]
    h = np.zeros((H,), dtype=np.float32)
    c = np.zeros((H,), dtype=np.float32)
    out = []

    mlp0_w = sd['mlp.0.weight'].numpy().astype(np.float32)
    mlp0_b = sd['mlp.0.bias'].numpy().astype(np.float32)
    mlp1_w = sd['mlp.3.weight'].numpy().astype(np.float32)
    mlp1_b = sd['mlp.3.bias'].numpy().astype(np.float32)

    for t in range(X_scaled.shape[0]):
        x = X_scaled[t].astype(np.float32)  # [In]
        # gates: [4H]
        gates = x @ W_ih.T + h @ W_hh.T + b
        i = 1.0 / (1.0 + np.exp(-gates[:H]))
        f = 1.0 / (1.0 + np.exp(-gates[H:2*H]))
        g = np.tanh(gates[2*H:3*H])
        o = 1.0 / (1.0 + np.exp(-gates[3*H:4*H]))
        c = f * c + i * g
        h = o * np.tanh(c)

        # MLP
        z = h @ mlp0_w.T + mlp0_b
        z = np.maximum(0.0, z)
        y = z @ mlp1_w.T + mlp1_b
        y = 1.0 / (1.0 + np.exp(-y))
        out.append(float(y[0]))

    return np.array(out, dtype=np.float32)


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

    # Data & scaler
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scaler = joblib.load(SCALER_PATH)
    feat_names = cfg['model']['features']

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

    # FP32 reference
    cell, mlp = build_fp32_reference(sd, In, H)
    h = np.zeros((H,), dtype=np.float32)
    c = np.zeros((H,), dtype=np.float32)
    preds_ref = []
    for t in range(Xs.shape[0]):
        x = Xs[t]
        h_t, c_t = cell(torch.from_numpy(x).unsqueeze(0), (torch.from_numpy(h).unsqueeze(0), torch.from_numpy(c).unsqueeze(0)))
        h = h_t.detach().squeeze(0).numpy().astype(np.float32)
        c = c_t.detach().squeeze(0).numpy().astype(np.float32)
        y = mlp(h)[0]
        preds_ref.append(float(y))
    preds_ref = np.array(preds_ref, dtype=np.float32)

    # Python C-sim path
    preds_csim = csim_predict_sequence(sd, Xs)

    # Metrics
    n = min(len(preds_csim), len(preds_ref))
    preds_csim = preds_csim[:n]
    preds_ref = preds_ref[:n]
    y_true = y_true[:n]
    diffs = np.abs(preds_ref - preds_csim)
    mae = float(np.mean(np.abs(preds_ref - preds_csim)))
    rmse = float(np.sqrt(np.mean((preds_ref - preds_csim)**2)))
    mxd = float(np.max(diffs))

    out_dir = Path(args.out_dir) if args.out_dir else (BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'STM32' / 'base' / f'PC_PY_CSIM_BASE_{time.strftime("%Y%m%d_%H%M%S")}')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics.json').write_text(json.dumps({'MAE': mae, 'RMSE': rmse, 'MAX': mxd, 'N': int(n)}, indent=2))

    # Plots
    plt.figure(figsize=(12,4)); plt.plot(y_true, label='GT', alpha=0.6); plt.plot(preds_ref, label='FP32'); plt.plot(preds_csim, label='PY_CSIM'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'overlay_full.png', dpi=150); plt.close()
    firstN = min(500, n)
    plt.figure(figsize=(12,4)); plt.plot(y_true[:firstN], label='GT', alpha=0.6); plt.plot(preds_ref[:firstN], label='FP32'); plt.plot(preds_csim[:firstN], label='PY_CSIM'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'overlay_firstN.png', dpi=150); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(diffs, bins=100, alpha=0.8, color='tab:blue', edgecolor='black'); plt.title('Abs diff PY_CSIM vs FP32'); plt.tight_layout(); plt.savefig(out_dir/'diff_hist.png', dpi=150); plt.close()

    print(f"Saved results to: {out_dir}")


if __name__ == '__main__':
    main()

