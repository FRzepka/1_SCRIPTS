#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def map_features(df: pd.DataFrame, feats):
    cols = list(df.columns)
    mapped = []
    for f in feats:
        if f in df.columns:
            mapped.append(f)
        else:
            raise KeyError(f"Feature '{f}' not found in dataframe")
    return mapped


def strict_filter(series: np.ndarray, rel, abs_, alpha):
    if series.size == 0:
        return series
    out = series.astype(np.float64, copy=True)
    prev = out[0]
    for i in range(1, len(out)):
        d = out[i] - prev
        if abs_ is not None:
            d = np.clip(d, -abs_, abs_)
        if rel is not None:
            cap = max(abs(prev) * rel, 1e-9)
            d = np.clip(d, -cap, cap)
        prev = prev + d
        out[i] = prev
    if alpha is not None and 0 < alpha < 1:
        ema = out[0]
        for i in range(1, len(out)):
            ema = alpha * out[i] + (1 - alpha) * ema
            out[i] = ema
    return out.astype(np.float32)


def apply_pin_before_filter(y: np.ndarray, start_one: bool) -> np.ndarray:
    if not start_one or y.size == 0:
        return y
    y = y.copy(); y[0] = 1.0
    return y


def quant_step(Xs: np.ndarray, q: dict, chunk: int):
    In = int(q['input_size'])
    Xs_use = Xs[:, :In] if Xs.shape[1] >= In else np.concatenate([Xs, np.zeros((Xs.shape[0], In - Xs.shape[1]), dtype=np.float32)], axis=1)
    T = Xs_use.shape[0]; H = int(q['hidden_size'])
    h = np.zeros((H,), dtype=np.float32); c = np.zeros((H,), dtype=np.float32)
    start = 0
    if T >= chunk:
        for t in range(chunk-1):
            x = Xs_use[t]
            gates = (x @ q['W_ih_q'].T).astype(np.float32) * q['S_ih'] + (h @ q['W_hh_q'].T).astype(np.float32) * q['S_hh'] + q['B']
            i = 1.0 / (1.0 + np.exp(-gates[:H])); f = 1.0 / (1.0 + np.exp(-gates[H:2*H]))
            g = np.tanh(gates[2*H:3*H]); o = 1.0 / (1.0 + np.exp(-gates[3*H:4*H]))
            c = f * c + i * g; h = o * np.tanh(c)
        start = chunk - 1
    preds = []
    try:
        from tqdm import tqdm
        it = tqdm(range(start, T), desc='quant-only', dynamic_ncols=True)
    except Exception:
        it = range(start, T)
    for t in it:
        x = Xs_use[t]
        gates = (x @ q['W_ih_q'].T).astype(np.float32) * q['S_ih'] + (h @ q['W_hh_q'].T).astype(np.float32) * q['S_hh'] + q['B']
        i = 1.0 / (1.0 + np.exp(-gates[:H])); f = 1.0 / (1.0 + np.exp(-gates[H:2*H]))
        g = np.tanh(gates[2*H:3*H]); o = 1.0 / (1.0 + np.exp(-gates[3*H:4*H]))
        c = f * c + i * g; h = o * np.tanh(c)
        z = np.maximum(0.0, h @ q['mlp0_w'].T + q['mlp0_b'])
        y = float((z @ q['mlp1_w'].T + q['mlp1_b']).reshape(()))
        preds.append(y)
    return np.array(preds, dtype=np.float32), start


def overlay(y, ytrue, out, title, plot_max: int = 100000):
    out.parent.mkdir(parents=True, exist_ok=True)
    if plot_max and plot_max > 0 and len(y) > plot_max:
        import math as _m
        step = int(_m.ceil(len(y) / float(plot_max)))
        yp = y[::step]
        ytp = (ytrue[::step] if ytrue is not None else None)
    else:
        yp, ytp = y, ytrue
    plt.figure(figsize=(12,4))
    if ytp is not None:
        plt.plot(ytp, label='SOH true', alpha=0.6, lw=0.6)
    plt.plot(yp, label='QUANT', lw=0.8)
    plt.legend(); plt.title(title); plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--ckpt', required=True, help='PyTorch checkpoint to read feature list and chunk size')
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--num-samples', type=int, default=500000)
    ap.add_argument('--prime', action='store_true')
    ap.add_argument('--strict-filter', action='store_true')
    ap.add_argument('--post-max-rel', type=float, default=None)
    ap.add_argument('--post-max-abs', type=float, default=None)
    ap.add_argument('--post-ema-alpha', type=float, default=None)
    ap.add_argument('--filter-passes', type=int, default=1)
    ap.add_argument('--calib-start-one', action='store_true')
    ap.add_argument('--calib-kind', choices=['scale','shift'], default='scale')
    ap.add_argument('--calib-anchor', choices=['by_pred','by_true'], default='by_pred')
    ap.add_argument('--calib-apply', choices=['before_filter','after_filter'], default='before_filter')
    ap.add_argument('--plot-max', type=int, default=100000)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    feats = ck.get('features') or ck.get('feature_list')
    if not feats:
        raise RuntimeError('Checkpoint missing features list')
    chunk = int(ck.get('chunk') or ck.get('window') or ck.get('seq_len') or 2048)

    df_all = pd.read_parquet(args.parquet).reset_index(drop=True)
    df = df_all.iloc[:args.num_samples]
    cols = map_features(df, feats)
    clean = df.replace([np.inf,-np.inf], np.nan).dropna(subset=cols + (['SOH'] if 'SOH' in df.columns else []))
    X = clean[cols].to_numpy(dtype=np.float32)
    ytrue = clean['SOH'].to_numpy(dtype=np.float32) if 'SOH' in clean.columns else None

    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    z = np.load(args.npz)
    q = {k: z[k] for k in z.files}
    y, start = quant_step(Xs, q, chunk)

    # align + calibration order
    y_al = y
    if args.prime:
        y_al = y
        if ytrue is not None:
            ytrue = ytrue[start:start+len(y_al)]

    if args.calib_start_one and args.calib_apply == 'before_filter':
        y_al = apply_pin_before_filter(y_al, True)
    if args.strict_filter:
        y_al = strict_filter(y_al, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
        for _ in range(max(0, args.filter_passes-1)):
            y_al = strict_filter(y_al, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
    if args.calib_start_one and args.calib_apply == 'after_filter':
        y_al = apply_pin_before_filter(y_al, True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    overlay(y_al, ytrue, out_dir / 'overlay_full.png', 'QUANT only (full)', plot_max=int(args.plot_max))
    firstN = min(2000, len(y_al))
    overlay(y_al[:firstN], (ytrue[:firstN] if ytrue is not None else None), out_dir / 'overlay_firstN.png', f'first {firstN}')
    (out_dir / 'metrics.json').write_text(json.dumps({
        'N': int(len(y_al)), 'chunk': chunk, 'prime': bool(args.prime),
        'strict_filter': bool(args.strict_filter), 'post_max_rel': args.post_max_rel,
        'post_max_abs': args.post_max_abs, 'post_ema_alpha': args.post_ema_alpha,
    }, indent=2))
    try:
        import sys
        cmd = 'python ' + Path(__file__).as_posix() + ' ' + ' '.join(sys.argv[1:])
        (out_dir / 'RUN_CMD.md').write_text('# Run command\n\n```\n' + cmd + '\n```\n')
    except Exception:
        pass
    print(f'[done] Wrote results to {out_dir}')


if __name__ == '__main__':
    main()

