#!/usr/bin/env python3
"""
Step-by-step comparison: SOH baseline vs pruned.

Primed mit chunk-1, dann ein Sample pro Schritt (hidden state wird weitergeführt).
Filter (per-step Caps + EMA) wird online angewandt. Optional Kalibrierung auf 1.
Speichert arrays/metrics und optional Plots.

Outputs: 6_test/Python/pruned_SOH/STEP_COMPARE_<timestamp>/<cell>/
"""
import argparse
import json
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load as joblib_load, dump as joblib_dump
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import sys
# Force line-buffered stdout to see progress in long runs
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

print("Imports complete. Starting script...", flush=True)

REPO = Path(__file__).resolve().parents[5]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LSTMMLP_SOHTarget(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int, num_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x, state=None, return_state: bool = False):
        out, new_state = self.lstm(x, state)
        # Apply MLP to all time steps
        pred = self.mlp(out).squeeze(-1)
        if return_state:
            return pred, new_state
        return pred


def load_checkpoint(ckpt_path: str, device: torch.device):
    raw = torch.load(ckpt_path, map_location=device)
    if not isinstance(raw, dict):
        raise ValueError('Unsupported checkpoint format')
    sd = raw.get('model_state_dict') or raw.get('state_dict') or raw
    cfg = raw.get('config') or raw.get('cfg') or {}
    features = raw.get('features')
    chunk = raw.get('chunk') or raw.get('window') or raw.get('seq_len')
    in_features = int(sd['lstm.weight_ih_l0'].shape[1])
    hidden_size = int(sd['lstm.weight_hh_l0'].shape[1])
    mlp_hidden = int(sd['mlp.0.weight'].shape[0])
    num_layers = int(cfg.get('model', {}).get('num_layers', 1)) if isinstance(cfg, dict) else 1
    dropout = float(cfg.get('model', {}).get('dropout', 0.05)) if isinstance(cfg, dict) else 0.05

    model = LSTMMLP_SOHTarget(in_features, hidden_size, mlp_hidden, num_layers=num_layers, dropout=dropout).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    scaler_path = raw.get('scaler_path')
    return model, list(features), int(chunk), scaler_path


def map_features(df: pd.DataFrame, features: list) -> list:
    cols = list(df.columns)
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
        # fallback: substring match
        hit = None
        for c in cols:
            if f.replace('°','') in c.replace('°',''):
                hit = c; break
        if hit is None:
            raise KeyError(f"Feature {f} not found")
        mapped.append(hit)
    return mapped


def load_or_refit_scaler(df: pd.DataFrame, features: list, scaler_hint: str, out_dir: Path):
    scaler = None
    used = None
    if scaler_hint and Path(scaler_hint).exists():
        try:
            scaler = joblib_load(scaler_hint)
            used = scaler_hint
        except Exception:
            scaler = None
    if scaler is None:
        clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
        X = clean[features].to_numpy(dtype=np.float32)
        scaler = RobustScaler(); scaler.fit(X)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / 'scaler_robust_refit.joblib'
        try:
            joblib_dump(scaler, save_path)
            used = str(save_path)
        except Exception:
            used = 'refit_in_memory'
    else:
        try:
            n_feat = getattr(scaler, 'n_features_in_', None)
            if n_feat is not None and int(n_feat) != len(features):
                clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
                X = clean[features].to_numpy(dtype=np.float32)
                scaler = RobustScaler(); scaler.fit(X)
                save_path = out_dir / 'scaler_robust_refit.joblib'
                try:
                    joblib_dump(scaler, save_path)
                    used = str(save_path)
                except Exception:
                    used = 'refit_in_memory'
        except Exception:
            pass
    return scaler, used


def step_predict_with_filter(Xs: np.ndarray, model: nn.Module, device: torch.device, chunk: int,
                             rel_cap: float, abs_cap: float, ema_alpha: float, limit: int):
    total = Xs.shape[0]
    state = None
    start = 0
    
    # Priming
    if total >= chunk:
        prime_input = Xs[:chunk-1]
        with torch.no_grad():
            prime_tensor = torch.from_numpy(prime_input).unsqueeze(0).to(device) # [1, chunk-1, F]
            _, state = model(prime_tensor, state=None, return_state=True)
        start = chunk - 1
    
    target = (total - start) if limit <= 0 else min(total - start, limit)
    print(f"[loop] starting batched predict: total={total} start={start} target={target}", flush=True)
    
    # Generate raw predictions in chunks
    raw_preds = []
    CHUNK_SIZE = 8192 # Process 8k steps at a time
    
    # Slice the relevant part of Xs
    Xs_target = Xs[start : start + target]
    
    pbar = tqdm(total=target, unit='step', desc='inference', dynamic_ncols=True, mininterval=1.0, file=sys.stdout)
    
    with torch.no_grad():
        for i in range(0, len(Xs_target), CHUNK_SIZE):
            end_i = min(i + CHUNK_SIZE, len(Xs_target))
            x_chunk = Xs_target[i:end_i] # [L, F]
            x_tensor = torch.from_numpy(x_chunk).unsqueeze(0).to(device) # [1, L, F]
            
            pred, state = model(x_tensor, state=state, return_state=True)
            # pred is [1, L]
            
            # Important: Detach state to be safe (though no_grad handles graph)
            # State is (h, c)
            state = (state[0].detach(), state[1].detach())
            
            raw_preds.append(pred.squeeze(0).cpu().numpy())
            pbar.update(end_i - i)
            
    pbar.close()
    
    raw_preds = np.concatenate(raw_preds)
    
    # Apply filter offline
    print(f"[filter] Applying filter (rel={rel_cap}, abs={abs_cap}, ema={ema_alpha})...", flush=True)
    filtered_preds = np.empty_like(raw_preds)
    
    last = None
    ema = None
    
    use_cap = (rel_cap is not None) or (abs_cap is not None)
    use_ema = (ema_alpha is not None) and (0 < ema_alpha < 1)
    
    for i in range(len(raw_preds)):
        v = float(raw_preds[i])
        
        if last is not None and use_cap:
            caps = []
            if rel_cap is not None:
                caps.append(abs(last) * rel_cap)
            if abs_cap is not None:
                caps.append(abs_cap)
            
            if caps:
                cap = min(caps)
                delta = v - last
                if abs(delta) > cap:
                    v = last + (cap if delta > 0 else -cap)
        
        if use_ema:
            ema = v if ema is None else (ema_alpha * v + (1 - ema_alpha) * ema)
            v = ema
            
        filtered_preds[i] = v
        last = v
        
    return filtered_preds, start


def overlay_plot(y_true, y_base, y_pruned, out_path: Path, title: str, max_points: int = 200000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def _ds(arr):
        n = len(arr)
        if n <= max_points:
            return arr, 1
        step = int(np.ceil(n / max_points))
        return arr[::step], step
    yt, st = (None,1)
    if y_true is not None:
        yt, st = _ds(y_true)
    yb, sb = _ds(y_base)
    yp, sp = _ds(y_pruned)
    step_used = max(st, sb, sp)
    plt.figure(figsize=(12,4))
    if yt is not None:
        plt.plot(yt, label='SOH true', linewidth=0.6, alpha=0.6, color='gray')
    plt.plot(yb, label='base', linewidth=1.0, alpha=0.8, color='blue')
    plt.plot(yp, label='pruned', linewidth=1.0, alpha=0.8, color='orange', linestyle='--')
    if step_used > 1:
        plt.title(f"{title} (downsample {step_used}x)")
    else:
        plt.title(title)
    plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


def diff_hist(diff, out_path: Path, max_samples: int = 200000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(diff) > max_samples:
        idx = np.random.RandomState(0).choice(len(diff), size=max_samples, replace=False)
        diff_plot = diff[idx]
    else:
        diff_plot = diff
    plt.figure(figsize=(6,4))
    plt.hist(diff_plot, bins=120, alpha=0.85, color='tab:purple', edgecolor='black')
    plt.title('Base - Pruned residuals'); plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.size == 0 or y_pred.size == 0:
        return {'mae': None, 'rmse': None, 'r2': None, 'n': int(min(len(y_true), len(y_pred)))}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    var = float(np.var(y_true))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)) if var > 0 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'n': int(len(y_pred))}


def parse_args():
    default_base = Path(__file__).resolve().parents[5] / 'DL_Models' / 'LFP_LSTM_MLP' / '2_models' / 'base' / 'soh_2.1.0.0_base' / '2.1.0.0_soh_epoch0120_rmse0.03359.pt'
    ap = argparse.ArgumentParser(description='Step-by-step SOH baseline vs pruned')
    ap.add_argument('--baseline-ckpt', type=str, default=str(default_base))
    ap.add_argument('--pruned-ckpt', type=str, required=True)
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    ap.add_argument('--data-root', type=str, default='/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE')
    ap.add_argument('--cell', type=str, default='MGFarm_18650_C07')
    ap.add_argument('--limit', type=int, default=50000, help='Predictions after priming (<=0 for full)')
    ap.add_argument('--strict-filter', action='store_true')
    ap.add_argument('--post-max-rel', type=float, default=None)
    ap.add_argument('--post-max-abs', type=float, default=None)
    ap.add_argument('--post-ema-alpha', type=float, default=None)
    ap.add_argument('--calib-start-one', action='store_true')
    ap.add_argument('--calib-kind', choices=['scale','shift'], default='scale')
    ap.add_argument('--no-plots', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[config] baseline={args.baseline_ckpt}\n pruned={args.pruned_ckpt}\n cell={args.cell}\n limit={args.limit}", flush=True)
    print(f"[config] device={device.type}", flush=True)

    print("[model] Loading baseline checkpoint...", flush=True)
    base_model, base_features, base_chunk, base_scaler_hint = load_checkpoint(args.baseline_ckpt, device)
    print("[model] Loading pruned checkpoint...", flush=True)
    pruned_model, pruned_features, pruned_chunk, pruned_scaler_hint = load_checkpoint(args.pruned_ckpt, device)
    print(f"[model] base_feat={len(base_features)} chunk={base_chunk} | pruned_feat={len(pruned_features)} chunk={pruned_chunk}", flush=True)
    if base_features != pruned_features:
        raise ValueError(f'Feature lists differ:\nbase={base_features}\npruned={pruned_features}')
    features = list(base_features)
    chunk = min(base_chunk, pruned_chunk)
    print(f"[use] features={features} | chunk={chunk}", flush=True)

    run_root = REPO / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'Python' / 'pruned_SOH' / f"STEP_COMPARE_{time.strftime('%Y%m%d_%H%M%S')}"
    cell_dir = run_root / args.cell
    cell_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(args.data_root) / f"df_FE_{args.cell.split('_')[-1]}.parquet"
    if not parquet_path.exists():
        parquet_path = Path(args.data_root) / f"df_FE_{args.cell}.parquet"
    if not parquet_path.exists():
        cid = args.cell[-2:]
        alt = Path(args.data_root) / f"df_FE_C{cid}.parquet"
        if alt.exists():
            parquet_path = alt
    print(f"[data] loading parquet: {parquet_path}", flush=True)
    df = pd.read_parquet(parquet_path)
    print(f"[data] rows total={len(df)} cols={len(df.columns)}", flush=True)
    cols = map_features(df, features)
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ['SOH'])
    print(f"[data] after cleaning rows={len(clean)}", flush=True)

    scaler, scaler_used = load_or_refit_scaler(clean, cols, base_scaler_hint, cell_dir)
    print(f"[scaler] using: {scaler_used}", flush=True)
    X = clean[cols].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    y_full = clean['SOH'].to_numpy(dtype=np.float32)

    # Step-by-step predictions with online filter
    preds_base, start_base = step_predict_with_filter(Xs, base_model, device, chunk, args.post_max_rel, args.post_max_abs, args.post_ema_alpha, args.limit if args.limit else -1)
    preds_pruned, start_pruned = step_predict_with_filter(Xs, pruned_model, device, chunk, args.post_max_rel, args.post_max_abs, args.post_ema_alpha, args.limit if args.limit else -1)
    print(f"[pred] base_len={len(preds_base)} start={start_base} | pruned_len={len(preds_pruned)} start={start_pruned}", flush=True)

    # Optional calibration on first sample
    def apply_calib(arr: np.ndarray, y_true_seg: np.ndarray):
        if not (args.calib_start_one and arr.size > 0):
            return arr
        base = float(arr[0]) if (args.calib_kind == 'scale' or y_true_seg is None or y_true_seg.size == 0) else float(y_true_seg[0])
        if args.calib_kind == 'scale':
            gain = 1.0 / base if base != 0 else 1.0
            return (arr * gain).astype(np.float32)
        else:
            delta = 1.0 - base
            return (arr + delta).astype(np.float32)

    # Align
    end_base = start_base + len(preds_base)
    end_pruned = start_pruned + len(preds_pruned)
    span_start = max(start_base, start_pruned)
    span_end = min(end_base, end_pruned)
    if span_end <= span_start:
        raise RuntimeError('Prediction ranges do not overlap')
    span_len = span_end - span_start
    print(f"[align] span_start={span_start} span_end={span_end} span_len={span_len}", flush=True)

    def slice_span(arr, start_idx):
        off = span_start - start_idx
        return arr[off:off+span_len]

    y_base = slice_span(preds_base, start_base)
    y_pruned = slice_span(preds_pruned, start_pruned)
    y_true = y_full[span_start:span_end]

    # calibration after alignment (same factor would be applied online in a real system at first output)
    y_base = apply_calib(y_base, y_true)
    y_pruned = apply_calib(y_pruned, y_true)

    mets_base = {'mae': None, 'rmse': None, 'r2': None, 'n': 0}
    mets_pruned = {'mae': None, 'rmse': None, 'r2': None, 'n': 0}
    diff_metrics = {'mae': None, 'rmse': None, 'max_abs': None}

    nan_mask = np.isnan(y_base) | np.isnan(y_pruned) | np.isnan(y_true)
    if nan_mask.any():
        print(f"Warning: Found {nan_mask.sum()} NaNs; filtering for metrics", flush=True)
        valid = ~nan_mask
        y_true_v = y_true[valid]; y_base_v = y_base[valid]; y_pruned_v = y_pruned[valid]
    else:
        y_true_v = y_true; y_base_v = y_base; y_pruned_v = y_pruned

    if y_true_v.size > 0:
        mets_base = compute_metrics(y_true_v, y_base_v)
        mets_pruned = compute_metrics(y_true_v, y_pruned_v)
        diff = y_base_v - y_pruned_v
        diff_metrics = {
            'mae': float(np.mean(np.abs(diff))) if diff.size else None,
            'rmse': float(np.sqrt(np.mean(diff**2))) if diff.size else None,
            'max_abs': float(np.max(np.abs(diff))) if diff.size else None,
        }

    print(f"[metrics] base_vs_gt mae={mets_base.get('mae')} rmse={mets_base.get('rmse')}", flush=True)
    print(f"[metrics] pruned_vs_gt mae={mets_pruned.get('mae')} rmse={mets_pruned.get('rmse')}", flush=True)
    print(f"[metrics] base_vs_pruned mae={diff_metrics.get('mae')} rmse={diff_metrics.get('rmse')}", flush=True)

    np.savez_compressed(cell_dir / 'arrays.npz', y_true=y_true, y_base=y_base, y_pruned=y_pruned)
    if not args.no_plots:
        overlay_plot(y_true, y_base, y_pruned, cell_dir / 'overlay_full.png', f'{args.cell} – step base vs pruned')
        firstN = min(2000, span_len)
        overlay_plot(y_true[:firstN], y_base[:firstN], y_pruned[:firstN], cell_dir / f'overlay_first{firstN}.png', f'{args.cell} – first {firstN} samples')
        if diff_metrics.get('mae') is not None:
            diff = y_base_v - y_pruned_v
            diff_hist(diff, cell_dir / 'diff_hist.png')

    report = {
        'cell': args.cell,
        'n_samples': int(span_len),
        'start_idx': int(span_start),
        'data_root': args.data_root,
        'parquet': str(parquet_path),
        'baseline_ckpt': args.baseline_ckpt,
        'pruned_ckpt': args.pruned_ckpt,
        'scaler_used': scaler_used,
        'filter': {
            'strict': bool(args.strict_filter),
            'post_max_rel': args.post_max_rel,
            'post_max_abs': args.post_max_abs,
            'post_ema_alpha': args.post_ema_alpha,
            'calib_start_one': bool(args.calib_start_one),
            'calib_kind': args.calib_kind,
        },
        'metrics': {
            'baseline_vs_gt': mets_base,
            'pruned_vs_gt': mets_pruned,
            'baseline_vs_pruned': diff_metrics,
        },
    }
    with open(cell_dir / 'metrics.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[done] Saved comparison to {cell_dir}")


if __name__ == '__main__':
    main()
