#!/usr/bin/env python3
"""
Compare SOH seq2many (PyTorch) vs quantized step-by-step (manual INT8 weights, FP32 activations).

Outputs under 6_test/Python/quantized_SOH/COMPARE_S2M_VS_QUANT_STEP_<timestamp>:
- overlay_full.png, overlay_firstN.png
- metrics.json (MAE/RMSE for each vs GT and seq2many vs quantized)

Notes:
- Uses checkpoint's feature list and refits a RobustScaler on the selected cell.
- Handles minor column name encoding issues by fuzzy matching (ä→a, ü→u, °C issue, etc.).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import load as joblib_load
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Reuse online seq2many script (placed under 6_test/Python/soh_test_2.0.1.1)
REPO = Path(__file__).resolve().parents[5]
SEQ2MANY_PY = REPO / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'Python' / 'soh_test_2.0.1.1' / 'online_predict_soh_seq2many.py'

import importlib.util
spec = importlib.util.spec_from_file_location('online_predict_soh_seq2many', str(SEQ2MANY_PY))
seq2m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seq2m)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(m):
        return {"mae": None, "rmse": None}
    mae = mean_absolute_error(y_true[m], y_pred[m])
    rmse = float(np.sqrt(np.mean((y_true[m] - y_pred[m])**2)))
    return {"mae": float(mae), "rmse": rmse}


def norm(s: str) -> str:
    # Minimal de-noise for misencoded symbols
    return (s.replace('°', 'o').replace('�', '')
             .replace('ä', 'a').replace('Ä', 'A')
             .replace('ö', 'o').replace('Ö', 'O')
             .replace('ü', 'u').replace('Ü', 'U')
             .replace('ß', 'ss').replace("'", ""))


def map_features(df: pd.DataFrame, features: list) -> list:
    cols = list(df.columns)
    cols_norm = [norm(c) for c in cols]
    mapped = []
    for f in features:
        if f in df.columns:
            mapped.append(f); continue
        fn = norm(f)
        idx = None
        # direct normalized match
        for i, cn in enumerate(cols_norm):
            if cn == fn:
                idx = i; break
        if idx is None:
            # substring fallback
            for i, cn in enumerate(cols_norm):
                if fn in cn:
                    idx = i; break
        if idx is None:
            raise KeyError(f"Feature '{f}' not found in dataframe columns")
        mapped.append(cols[idx])
    return mapped


def map_features_strict(df: pd.DataFrame, features: list) -> list:
    """Strict feature mapping with only Temperature encoding fallback."""
    mapped = []
    for f in features:
        if f in df.columns:
            mapped.append(f)
            continue
        if f == 'Temperature[°C]' and 'Temperature[�C]' in df.columns:
            mapped.append('Temperature[�C]')
            continue
        if f == 'Temperature[�C]' and 'Temperature[°C]' in df.columns:
            mapped.append('Temperature[°C]')
            continue
        raise KeyError(f"Feature '{f}' not found in dataframe columns")
    return mapped


def quantize_per_row(W: np.ndarray):
    rows, cols = W.shape
    scales = np.zeros(rows, dtype=np.float32)
    Wq = np.zeros((rows, cols), dtype=np.int8)
    eps = 1e-12
    for r in range(rows):
        max_abs = np.max(np.abs(W[r]))
        scale = max(max_abs / 127.0, eps)
        scales[r] = scale
        Wq[r] = np.clip(np.round(W[r] / scale), -127, 127).astype(np.int8)
    return Wq, scales


def build_quant_state(state_dict: dict):
    W_ih = state_dict['lstm.weight_ih_l0'].cpu().numpy()
    W_hh = state_dict['lstm.weight_hh_l0'].cpu().numpy()
    b_ih = state_dict['lstm.bias_ih_l0'].cpu().numpy()
    b_hh = state_dict['lstm.bias_hh_l0'].cpu().numpy()
    B = b_ih + b_hh
    W_ih_q, S_ih = quantize_per_row(W_ih)
    W_hh_q, S_hh = quantize_per_row(W_hh)
    mlp0_w = state_dict['mlp.0.weight'].cpu().numpy(); mlp0_b = state_dict['mlp.0.bias'].cpu().numpy()
    mlp1_w = state_dict['mlp.3.weight'].cpu().numpy(); mlp1_b = state_dict['mlp.3.bias'].cpu().numpy()
    return {
        'W_ih_q': W_ih_q, 'S_ih': S_ih, 'W_hh_q': W_hh_q, 'S_hh': S_hh, 'B': B,
        'mlp0_w': mlp0_w, 'mlp0_b': mlp0_b, 'mlp1_w': mlp1_w, 'mlp1_b': mlp1_b,
        'input_size': int(W_ih.shape[1]), 'hidden_size': int(W_hh.shape[1])
    }


def _lstm_step_int8(h: np.ndarray, c: np.ndarray, x: np.ndarray, q: dict):
    H = q['hidden_size']
    W_ih_q, S_ih, W_hh_q, S_hh, B = q['W_ih_q'], q['S_ih'], q['W_hh_q'], q['S_hh'], q['B']
    gates = (x @ W_ih_q.T).astype(np.float32) * S_ih + (h @ W_hh_q.T).astype(np.float32) * S_hh + B
    i_pre = gates[0:H]; f_pre = gates[H:2*H]; g_pre = gates[2*H:3*H]; o_pre = gates[3*H:4*H]
    i = 1.0 / (1.0 + np.exp(-i_pre))
    f = 1.0 / (1.0 + np.exp(-f_pre))
    g = np.tanh(g_pre)
    o = 1.0 / (1.0 + np.exp(-o_pre))
    c = f * c + i * g
    h = o * np.tanh(c)
    return h, c


def quant_step_predict(Xs: np.ndarray, q: dict, chunk: int, progress: bool = True):
    # Ensure input dim match (pad/truncate if necessary)
    In = q['input_size']
    if Xs.shape[1] != In:
        if Xs.shape[1] < In:
            pad = np.zeros((Xs.shape[0], In - Xs.shape[1]), dtype=np.float32)
            Xs_use = np.concatenate([Xs, pad], axis=1)
        else:
            Xs_use = Xs[:, :In]
    else:
        Xs_use = Xs

    T = Xs_use.shape[0]
    H = q['hidden_size']
    W_ih_q, S_ih, W_hh_q, S_hh, B = q['W_ih_q'], q['S_ih'], q['W_hh_q'], q['S_hh'], q['B']
    mlp0_w, mlp0_b, mlp1_w, mlp1_b = q['mlp0_w'], q['mlp0_b'], q['mlp1_w'], q['mlp1_b']

    # Initialize state
    h = np.zeros((H,), dtype=np.float32)
    c = np.zeros((H,), dtype=np.float32)
    # prime with chunk-1 if possible (proper state advancement, no outputs)
    start = 0
    if T >= chunk:
        for t in range(chunk-1):
            h, c = _lstm_step_int8(h, c, Xs_use[t], q)
        start = chunk - 1
    preds = []
    rng = range(start, T)
    for t in rng:
        x = Xs_use[t]
        h, c = _lstm_step_int8(h, c, x, q)
        # linear MLP head
        x0 = np.maximum(0.0, h @ mlp0_w.T + mlp0_b)
        y = float((x0 @ mlp1_w.T + mlp1_b).reshape(()))
        preds.append(y)
    return np.array(preds, dtype=np.float64), start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--cell', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='')
    ap.add_argument('--block_len', type=int, default=8192)
    ap.add_argument('--strict_filter', action='store_true')
    # optional filter for quantized path to mirror seq2many smoothing
    ap.add_argument('--step_filter', action='store_true')
    ap.add_argument('--step_max_rel', type=float, default=None)
    ap.add_argument('--step_max_abs', type=float, default=None)
    ap.add_argument('--step_ema_alpha', type=float, default=None)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--n', type=int, default=0, help='number of rows to use (0=all)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, features, chunk, cfg, scaler_path, pooling = seq2m.load_checkpoint(args.checkpoint, device)

    # Data
    pq = seq2m.locate_cell_parquet(args.data_root, args.cell)
    df_all = pd.read_parquet(pq)
    # Map features strictly (Temperature fallback only) using full df
    mapped_features_full = map_features_strict(df_all, features)
    # Scaler handling like base script: fit on full cell if no usable checkpoint scaler
    scaler = None
    try:
        if scaler_path and Path(scaler_path).exists():
            scaler = joblib_load(scaler_path)
    except Exception:
        scaler = None
    if scaler is None or (getattr(scaler, 'n_features_in_', None) is not None and int(getattr(scaler, 'n_features_in_', 0)) != len(mapped_features_full)):
        X_pre = df_all.replace([np.inf, -np.inf], np.nan)[mapped_features_full].dropna().to_numpy(dtype=np.float32)
        scaler = RobustScaler().fit(X_pre)
    # Now apply slicing and prepare arrays
    df = df_all
    if args.start or args.n:
        s = max(0, int(args.start))
        e = s + int(args.n) if int(args.n) > 0 else None
        df = df.iloc[s:e].reset_index(drop=True)
    mapped_features = map_features_strict(df, features)
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=mapped_features + (['SOH'] if 'SOH' in df.columns else []))
    X = clean[mapped_features].to_numpy(dtype=np.float32)
    y_true = clean['SOH'].to_numpy(dtype=np.float64) if 'SOH' in clean.columns else None

    # seq2many predictions (PyTorch)
    y_seq2m, y_true_s2m, _eff = seq2m.seq2many_with_state(clean, mapped_features, scaler, model, chunk, device,
                                                          block_len=args.block_len, progress=True,
                                                          strict_filter=args.strict_filter)

    # quantized step predictions
    state_dict = (torch.load(args.checkpoint, map_location='cpu').get('model_state_dict'))
    q = build_quant_state(state_dict)
    Xs = scaler.transform(X).astype(np.float32)
    y_q, start_q = quant_step_predict(Xs, q, chunk, progress=True)
    # optional simple caps + EMA on quantized path
    if args.step_filter or args.strict_filter:
        # defaults to mirror seq2many strict
        step_max_abs = args.step_max_abs if args.step_max_abs is not None else (2e-5 if args.strict_filter else None)
        step_max_rel = args.step_max_rel if args.step_max_rel is not None else (5e-4 if args.strict_filter else None)
        alpha = args.step_ema_alpha if args.step_ema_alpha is not None else (0.005 if args.strict_filter else None)
        if any(v is not None for v in [step_max_abs, step_max_rel, alpha]):
            out = []
            last = None
            for p in y_q:
                v = float(p)
                if last is not None:
                    caps = []
                    if step_max_rel is not None:
                        caps.append(abs(last) * step_max_rel)
                    if step_max_abs is not None:
                        caps.append(step_max_abs)
                    if caps:
                        cap = min(caps)
                        dv = v - last
                        if dv > cap:
                            v = last + cap
                        elif dv < -cap:
                            v = last - cap
                    if alpha is not None and alpha > 0:
                        v = (1.0 - alpha) * last + alpha * v
                out.append(v)
                last = v
            y_q = np.array(out, dtype=np.float64)
    y_true_q = y_true[start_q:start_q+len(y_q)] if y_true is not None else None

    # Align lengths for overlay
    n = min(len(y_seq2m), len(y_q))
    y_seq2m = y_seq2m[:n]
    y_q = y_q[:n]
    yt_aligned = None
    if y_true_s2m is not None:
        yt_aligned = y_true_s2m[:n]

    # Metrics
    metrics = {
        'seq2many_vs_quant': compute_metrics(y_seq2m, y_q),
    }
    if yt_aligned is not None:
        metrics['seq2many_vs_true'] = compute_metrics(yt_aligned, y_seq2m)
        metrics['quant_vs_true'] = compute_metrics(yt_aligned, y_q)

    # Output dir
    out_dir = Path(args.out_dir) if args.out_dir else (REPO / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'Python' / 'quantized_SOH' / f'COMPARE_S2M_VS_QUANT_STEP_{time.strftime("%Y%m%d_%H%M%S")}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plots
    def overlay_plot(y_seq, y_qu, yt, path, title):
        plt.figure(figsize=(12,4))
        if yt is not None:
            plt.plot(yt, label='GT', alpha=0.5)
        plt.plot(y_seq, label='seq2many', linewidth=1.0)
        plt.plot(y_qu, label='quant_step', linewidth=1.0, alpha=0.9)
        plt.legend(); plt.title(title); plt.tight_layout()
        plt.savefig(path, dpi=150); plt.close()

    overlay_plot(y_seq2m, y_q, yt_aligned, out_dir / 'overlay_full.png', f'{args.cell} – seq2many vs quant_step (full)')
    firstN = min(5000, n)
    overlay_plot(y_seq2m[:firstN], y_q[:firstN], (yt_aligned[:firstN] if yt_aligned is not None else None), out_dir / 'overlay_firstN.png', f'{args.cell} – seq2many vs quant_step (first {firstN})')

    # Save metrics
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Saved results to', out_dir)


if __name__ == '__main__':
    main()
