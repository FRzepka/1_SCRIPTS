#!/usr/bin/env python3
"""
Self-contained SOH seq2many runner (no external helpers).

- Loads the FP32 base checkpoint and reconstructs the LSTM+MLP from the
  state_dict shapes (robust to config differences).
- Locates the parquet by either explicit --parquet or (--data-root, --cell).
- Maps feature names robustly to handle encoding/typo variants.
- Loads the RobustScaler from the checkpoint hint if available, else refits on
  the full dataframe and saves a refit copy next to the outputs.
- Primes with chunk-1 steps, then runs in blocks and emits one prediction per
  step (seq2many). Optional strict post-filtering is available.

Outputs under 6_test/Python/base_SOH/python/SEQ2MANY_RUN_<timestamp>/:
  - arrays.npz (y_true, y_pred)
  - metrics.json
  - overlay_full.png, overlay_first2000.png
  - scaler_robust_refit.joblib (if refit)
"""
import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load as joblib_load, dump as joblib_dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import warnings
from sklearn.exceptions import InconsistentVersionWarning

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -------------------------
# Model
# -------------------------

class LSTMMLP_SOHTarget(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int, num_layers: int = 1, dropout: float = 0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_state: bool = False, all_steps: bool = False):
        out, new_state = self.lstm(x, state)
        if all_steps:
            T = out.size(1)
            hs = out.reshape(-1, out.size(-1))
            preds = self.mlp(hs).reshape(x.size(0), T, 1).squeeze(-1)
            if return_state:
                return preds, new_state
            return preds
        last = out[:, -1, :]
        pred = self.mlp(last).squeeze(-1)
        if return_state:
            return pred, new_state
        return pred


def load_checkpoint(ckpt_path: str, device: torch.device):
    raw = torch.load(ckpt_path, map_location=device)
    if not isinstance(raw, dict):
        # assume raw is a state_dict
        sd = raw
        cfg = {}
        features = None
        chunk = None
    else:
        sd = raw.get('model_state_dict') or raw.get('state_dict') or raw
        cfg = raw.get('config') or raw.get('cfg') or {}
        features = raw.get('features') or raw.get('feature_list')
        chunk = raw.get('chunk') or raw.get('window') or raw.get('seq_len')
    if sd is None:
        raise KeyError('No weights found in checkpoint')

    # infer model sizes from weights if needed
    in_features = int(sd['lstm.weight_ih_l0'].shape[1])
    hidden_size = int(sd['lstm.weight_hh_l0'].shape[1])
    mlp_hidden = int(sd['mlp.0.weight'].shape[0])
    num_layers = int(cfg.get('model', {}).get('num_layers', 1)) if isinstance(cfg, dict) else 1
    dropout = float(cfg.get('model', {}).get('dropout', 0.05)) if isinstance(cfg, dict) else 0.05

    model = LSTMMLP_SOHTarget(
        in_features=in_features,
        hidden_size=hidden_size,
        mlp_hidden=mlp_hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    if features is None:
        # fall back to canonical SOH feature list (6 features; no Capacity)
        features = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'EFC', 'Q_c']
    if chunk is None:
        # safe default if missing
        chunk = int(cfg.get('training', {}).get('chunk', 2048)) if isinstance(cfg, dict) else 2048

    scaler_hint = raw.get('scaler_path') if isinstance(raw, dict) else None
    if scaler_hint is None and isinstance(cfg, dict):
        out_root = (cfg.get('paths', {}) or {}).get('out_root')
        if out_root:
            candidate = os.path.join(out_root, 'scaler_robust.joblib')
            if os.path.exists(candidate):
                scaler_hint = candidate
    # Additional fallback: use training scaler under 1_training/<ver>/outputs/scaler_robust.joblib
    if scaler_hint is None:
        m = re.search(r"(\d+\.\d+\.\d+\.\d+)_soh_", os.path.basename(ckpt_path))
        if m:
            ver = m.group(1)
            REPO = Path(__file__).resolve().parents[4]  # .../LFP_LSTM_MLP
            cand = REPO / '1_training' / ver / 'outputs' / 'scaler_robust.joblib'
            if cand.exists():
                scaler_hint = str(cand)
    return model, list(features), int(chunk), scaler_hint


# -------------------------
# Feature mapping / data helpers
# -------------------------

def _norm_name(s: str) -> str:
    s = s.strip().lower()
    # normalize common odd encodings and remove non-alnum
    repl = {
        '°c': 'c', 'Â°c': 'c', '℃': 'c', 'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
        'â°c': 'c', '�c': 'c', '°': '', ' ': '', '[': '', ']': '', '(': '', ')': '', '/s': 's', '/v': 'v', '/a': 'a'
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return re.sub(r'[^a-z0-9]+', '', s)


def map_features_with_fallback(df: pd.DataFrame, features: List[str]) -> List[str]:
    cols = list(df.columns)
    cols_norm = [_norm_name(c) for c in cols]
    mapped = []
    for feat in features:
        if feat in df.columns:
            mapped.append(feat)
            continue
        fn = _norm_name(feat)
        idx = None
        # exact normalized match
        for i, cn in enumerate(cols_norm):
            if cn == fn:
                idx = i
                break
        if idx is None:
            # substring fallback
            for i, cn in enumerate(cols_norm):
                if fn in cn:
                    idx = i
                    break
        if idx is None:
            raise KeyError(f"Feature '{feat}' not found in dataframe columns")
        mapped.append(cols[idx])
    return mapped


def locate_cell_parquet(data_root: str, cell: str) -> str:
    # try df_FE_C07.parquet, df_FE_07.parquet, MGFarm_18650_C07.parquet
    c = cell
    m = re.search(r'_C(\d{2})$', cell)
    suffix = m.group(1) if m else cell[-2:]
    cands = [
        os.path.join(data_root, f'df_FE_C{suffix}.parquet'),
        os.path.join(data_root, f'df_FE_{suffix}.parquet'),
        os.path.join(data_root, f'MGFarm_18650_C{suffix}.parquet'),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None found: {cands}")


def _fit_scaler(df: pd.DataFrame, features: List[str]) -> RobustScaler:
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    X = clean[features].to_numpy(dtype=np.float32)
    sc = RobustScaler()
    sc.fit(X)
    return sc


def load_or_refit_scaler(df_full: pd.DataFrame, features: List[str], scaler_hint: Optional[str], out_dir: Path):
    scaler = None
    used = None
    if scaler_hint and os.path.exists(scaler_hint):
        try:
            scaler = joblib_load(scaler_hint)
            used = scaler_hint
        except Exception:
            scaler = None
    if scaler is None:
        scaler = _fit_scaler(df_full, features)
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
                scaler = _fit_scaler(df_full, features)
                save_path = out_dir / 'scaler_robust_refit.joblib'
                try:
                    joblib_dump(scaler, save_path)
                    used = str(save_path)
                except Exception:
                    used = 'refit_in_memory'
        except Exception:
            scaler = _fit_scaler(df_full, features)
            save_path = out_dir / 'scaler_robust_refit.joblib'
            try:
                joblib_dump(scaler, save_path)
                used = str(save_path)
            except Exception:
                used = 'refit_in_memory'
    return scaler, used


def strict_filter(series: np.ndarray, rel: Optional[float], abs_: Optional[float], alpha: Optional[float]) -> np.ndarray:
    if series.size == 0:
        return series
    out = series.astype(np.float64, copy=True)
    prev = out[0]
    for i in range(1, len(out)):
        diff = out[i] - prev
        if abs_ is not None:
            diff = np.clip(diff, -abs_, abs_)
        if rel is not None:
            cap = max(abs(prev) * rel, 1e-9)
            diff = np.clip(diff, -cap, cap)
        prev = prev + diff
        out[i] = prev
    if alpha is not None and 0 < alpha < 1:
        ema = out[0]
        for i in range(1, len(out)):
            ema = alpha * out[i] + (1 - alpha) * ema
            out[i] = ema
    return out.astype(np.float32)


def _ema_pass(x: np.ndarray, alpha: float) -> np.ndarray:
    if not (0 < alpha < 1):
        return x
    y = x.astype(np.float64, copy=True)
    ema = y[0]
    for i in range(1, len(y)):
        ema = alpha * y[i] + (1 - alpha) * ema
        y[i] = ema
    return y.astype(np.float32)


def bidirectional_ema(series: np.ndarray, alpha: float) -> np.ndarray:
    """Phase-lag-arme Glättung: EMA vorwärts + rückwärts und mitteln."""
    if series.size == 0 or not (0 < alpha < 1):
        return series
    fwd = _ema_pass(series, alpha)
    bwd = _ema_pass(series[::-1], alpha)[::-1]
    return ((fwd.astype(np.float64) + bwd.astype(np.float64)) * 0.5).astype(np.float32)


def median_filter_1d(series: np.ndarray, k: int) -> np.ndarray:
    """Einfache Median-Glättung; für große N und große k kostenintensiv.
    Verwende kleine ungerade Fenster (z.B. 3,5,7) oder deaktivieren (k<=1)."""
    if series.size == 0 or k is None or k <= 1:
        return series
    k = int(k) | 1  # ungerade
    pad = k // 2
    x = np.pad(series.astype(np.float64), (pad, pad), mode='edge')
    out = np.empty_like(series, dtype=np.float64)
    # naive sliding median (ok für kleine k)
    for i in range(len(series)):
        out[i] = np.median(x[i:i + k])
    return out.astype(np.float32)


def seq2many_predictions(Xs: np.ndarray, model: nn.Module, device: torch.device, chunk: int, block_len: int, limit: int, prime: bool = True) -> Tuple[np.ndarray, int]:
    total = Xs.shape[0]
    preds = []
    state = None
    start = 0
    with torch.no_grad():
        if prime and total >= chunk:
            prime_seq = torch.from_numpy(Xs[:chunk-1]).unsqueeze(0).to(device)
            _, state = model(prime_seq, state=None, return_state=True)
            start = chunk - 1
        target = (total - start) if limit <= 0 else min(total - start, limit)
        pbar = tqdm(total=target, unit='step', desc='seq2many', dynamic_ncols=True)
        i = start
        while i < total and len(preds) < target:
            end = min(total, i + block_len)
            block = torch.from_numpy(Xs[i:end]).unsqueeze(0).to(device)
            out_seq, state = model(block, state=state, return_state=True, all_steps=True)
            block_preds = out_seq.squeeze(0).detach().cpu().numpy()
            remaining = target - len(preds)
            take = int(min(len(block_preds), remaining))
            preds.extend(block_preds[:take].tolist())
            pbar.update(take)
            i = end
        pbar.close()
    return np.asarray(preds, dtype=np.float32), start


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.size == 0 or y_pred.size == 0:
        return {'mae': None, 'rmse': None, 'r2': None, 'n': int(min(len(y_true), len(y_pred)))}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else float('nan')
    return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2), 'n': int(len(y_pred))}


def overlay_plot(y_pred: np.ndarray, y_true: Optional[np.ndarray], out_path: Path, title: str, firstN: Optional[int] = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    if y_true is not None:
        plt.plot(y_true, label='SOH true', linewidth=0.6, alpha=0.6)
    plt.plot(y_pred, label='seq2many', linewidth=0.8)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    if firstN is not None:
        n = min(firstN, len(y_pred))
        plt.figure(figsize=(12, 4))
        if y_true is not None:
            plt.plot(y_true[:n], label='SOH true', linewidth=0.6, alpha=0.6)
        plt.plot(y_pred[:n], label='seq2many', linewidth=0.8)
        plt.title(f'{title} (first {n})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path.parent / 'overlay_first2000.png', dpi=140)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description='Run SOH seq2many (FP32 base)')
    ap.add_argument('--ckpt', required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--parquet', default=None)
    g.add_argument('--data-root', default=None)
    ap.add_argument('--cell', default='MGFarm_18650_C07')
    ap.add_argument('--limit', type=int, default=0, help='number of predictions after priming (<=0 for full)')
    ap.add_argument('--block-len', type=int, default=8192)
    ap.add_argument('--no-prime', action='store_true', help='disable initial chunk-1 priming')
    ap.add_argument('--strict-filter', action='store_true')
    ap.add_argument('--post-max-rel', type=float, default=None)
    ap.add_argument('--post-max-abs', type=float, default=None)
    ap.add_argument('--post-ema-alpha', type=float, default=None)
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--plot-max', type=int, default=50000, help='max points to plot in overlay_full (downsample if longer)')
    ap.add_argument('--arrays', choices=['full','firstN','none'], default=None, help='arrays.npz saving mode (auto if omitted)')
    ap.add_argument('--scaler', default=None, help='optional path to RobustScaler.joblib to force use (no refit)')
    ap.add_argument('--scaler-mode', choices=['train','refit','hybrid_center','hybrid_scale','none'], default='train', help='train: TRAIN scaler; refit: refit on cell; hybrid_center: TRAIN scale_ + REFIT center_; hybrid_scale: TRAIN center_ + REFIT scale_; none: no scaling')
    ap.add_argument('--filter-passes', type=int, default=1, help='repeat strict filter N times')
    ap.add_argument('--filter-bidir', action='store_true', help='apply bidirectional EMA after strict filter to reduce phase lag')
    ap.add_argument('--median-window', type=int, default=0, help='optional median filter window (odd, small e.g. 3/5). Warning: slow for huge series')
    # optional calibration that affects predictions (online-tauglich)
    ap.add_argument('--calib-start-one', action='store_true', help='calibrate predictions so first emitted value equals 1 (affects outputs/metrics)')
    ap.add_argument('--calib-kind', choices=['scale','shift'], default='scale', help='scale: multiply by 1/y0; shift: add (1-y0)')
    ap.add_argument('--calib-anchor', choices=['by_pred','by_true'], default='by_pred', help='by_pred: first prediction; by_true: first GT (offline only)')
    ap.add_argument('--calib-mode', choices=['uniform','pin','decay'], default='uniform', help='uniform: global gain/offset; pin: only first sample; decay: correction fades out')
    ap.add_argument('--calib-decay-tau', type=int, default=5000, help='time constant (samples) for decay mode (1/e)')
    ap.add_argument('--norm-start-one', choices=['off','per_series','by_true'], default='off', help='Normalize curves for plotting to start at 1 (plot only)')
    # where to apply calibration relative to strict-filter
    ap.add_argument('--calib-apply', choices=['after_filter','before_filter'], default='after_filter', help='Apply calibration before or after strict-filtering')
    ap.add_argument('--norm-kind', choices=['scale','shift'], default='scale', help='scale: divide by first value; shift: subtract first and add 1 (plot only)')

    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, features, chunk, scaler_hint = load_checkpoint(args.ckpt, device)

    if args.parquet:
        parquet_path = args.parquet
    else:
        parquet_path = locate_cell_parquet(args.data_root, args.cell)
    df = pd.read_parquet(parquet_path)

    cols = map_features_with_fallback(df, features)
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ['SOH'] if 'SOH' in df.columns else cols)
    if clean.empty:
        raise ValueError('No valid rows after cleaning/filtering')

    out_root = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parents[0] / f'SEQ2MANY_RUN_{time.strftime("%Y%m%d_%H%M%S")}')
    out_root.mkdir(parents=True, exist_ok=True)

    # prefer CLI scaler if provided
    scaler, scaler_used = (None, None)
    if args.scaler and os.path.exists(args.scaler):
        try:
            scaler = joblib_load(args.scaler)
            scaler_used = args.scaler
        except Exception:
            scaler = None
            scaler_used = None
    if scaler is None:
        scaler, scaler_used = load_or_refit_scaler(clean, cols, scaler_hint, out_root)

    # optional REFIT scaler for hybrid modes
    refit_scaler = None
    refit_used = None
    if args.scaler_mode in ('refit','hybrid_center','hybrid_scale'):
        refit_scaler, refit_used = load_or_refit_scaler(clean, cols, None, out_root)

    X = clean[cols].to_numpy(dtype=np.float32)
    def _transform_with(train_sc, ref_sc, Xnp, mode: str):
        if mode == 'none':
            return Xnp.astype(np.float32)
        if mode == 'train':
            return train_sc.transform(Xnp).astype(np.float32)
        if mode == 'refit' and ref_sc is not None:
            return ref_sc.transform(Xnp).astype(np.float32)
        # hybrid modes: (X - center_) / scale_
        if not hasattr(train_sc, 'center_') or not hasattr(train_sc, 'scale_'):
            return train_sc.transform(Xnp).astype(np.float32)
        center_tr = np.asarray(getattr(train_sc, 'center_'))
        scale_tr = np.asarray(getattr(train_sc, 'scale_'))
        center_rf = np.asarray(getattr(ref_sc, 'center_')) if (ref_sc is not None and hasattr(ref_sc, 'center_')) else center_tr
        scale_rf = np.asarray(getattr(ref_sc, 'scale_')) if (ref_sc is not None and hasattr(ref_sc, 'scale_')) else scale_tr
        if mode == 'hybrid_center':
            return ((Xnp - center_rf) / (scale_tr + 1e-12)).astype(np.float32)
        if mode == 'hybrid_scale':
            return ((Xnp - center_tr) / (scale_rf + 1e-12)).astype(np.float32)
        return train_sc.transform(Xnp).astype(np.float32)

    Xs = _transform_with(scaler, refit_scaler, X, args.scaler_mode)

    preds, start_idx = seq2many_predictions(
        Xs, model, device, chunk, args.block_len,
        args.limit if args.limit else -1,
        prime=not args.no_prime,
    )

    y_true = None
    if 'SOH' in clean.columns:
        y_full = clean['SOH'].to_numpy(dtype=np.float32)
        y_true = y_full[start_idx:start_idx + len(preds)]

    def apply_calibration(pred_series: np.ndarray, y_true_series: Optional[np.ndarray]):
        info = {'calib_kind': None, 'calib_anchor': None, 'calib_mode': None}
        if not (args.calib_start_one and len(pred_series) > 0):
            return pred_series, info
        eps = 1e-12
        if args.calib_anchor == 'by_true' and y_true_series is not None and len(y_true_series) > 0:
            base = float(y_true_series[0])
        else:
            base = float(pred_series[0])
        mode = args.calib_mode
        y = pred_series
        if mode == 'pin':
            # only the first emitted sample is forced to 1; rest unchanged
            y = y.copy()
            y[0] = 1.0
            kind = 'scale' if args.calib_kind == 'scale' else 'shift'
            info = {'calib_kind': kind, 'calib_anchor': args.calib_anchor, 'calib_mode': 'pin'}
            return y, info
        elif mode == 'decay':
            T = y.shape[0]
            w = np.exp(-np.arange(T, dtype=np.float64) / max(1, int(args.calib_decay_tau)))
            if args.calib_kind == 'scale':
                g = (1.0 / base) if abs(base) > eps else 1.0
                y = (y.astype(np.float64) * (1.0 + w * (g - 1.0))).astype(np.float32)
                info = {'calib_kind': 'scale', 'calib_anchor': args.calib_anchor, 'calib_mode': 'decay', 'gain': float(g), 'tau': int(args.calib_decay_tau)}
            else:
                dlt = 1.0 - base
                y = (y.astype(np.float64) + w * dlt).astype(np.float32)
                info = {'calib_kind': 'shift', 'calib_anchor': args.calib_anchor, 'calib_mode': 'decay', 'delta': float(dlt), 'tau': int(args.calib_decay_tau)}
            return y, info
        else:
            if args.calib_kind == 'scale':
                gain = (1.0 / base) if abs(base) > eps else 1.0
                y = (y * gain).astype(np.float32)
                info = {'calib_kind': 'scale', 'calib_anchor': args.calib_anchor, 'calib_mode': 'uniform', 'gain': float(gain)}
            else:
                delta = 1.0 - base
                y = (y + delta).astype(np.float32)
                info = {'calib_kind': 'shift', 'calib_anchor': args.calib_anchor, 'calib_mode': 'uniform', 'delta': float(delta)}
            return y, info

    calib_info = {'calib_kind': None, 'calib_anchor': None, 'calib_mode': None}
    if args.calib_apply == 'before_filter':
        preds, calib_info = apply_calibration(preds, y_true)

    if args.strict_filter:
        # core strict filter
        preds = strict_filter(preds, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
        # extra passes
        for _ in range(max(0, int(args.filter_passes) - 1)):
            preds = strict_filter(preds, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
        # optional bidirectional ema on top (reduces lag)
        if args.filter_bidir and args.post_ema_alpha and 0 < args.post_ema_alpha < 1:
            preds = bidirectional_ema(preds, args.post_ema_alpha)
        # optional median filter (small window recommended)
        if args.median_window and int(args.median_window) > 1:
            preds = median_filter_1d(preds, int(args.median_window))

    if args.calib_apply == 'after_filter':
        preds, calib_info = apply_calibration(preds, y_true)

    mets = compute_metrics(y_true if y_true is not None else np.array([], dtype=np.float32), preds)

    # save arrays (avoid huge compressed files that take long)
    save_mode = args.arrays
    if save_mode is None:
        save_mode = 'full' if len(preds) <= 2000000 else 'firstN'
    if save_mode == 'full':
        np.savez(out_root / 'arrays.npz', y_true=y_true, y_pred=preds)
    elif save_mode == 'firstN':
        n_first = min(500000, len(preds))
        np.savez(out_root / 'arrays.npz', y_true=(None if y_true is None else y_true[:n_first]), y_pred=preds[:n_first])
    with open(out_root / 'metrics.json', 'w') as f:
        json.dump({
            'checkpoint': args.ckpt,
            'parquet': parquet_path,
            'features': features,
            'mapped_cols': cols,
            'chunk': int(chunk),
            'prime': not args.no_prime,
            'start_index': int(start_idx),
            'limit': int(args.limit),
            'block_len': int(args.block_len),
            'strict_filter': bool(args.strict_filter),
            'post_max_rel': args.post_max_rel,
            'post_max_abs': args.post_max_abs,
            'post_ema_alpha': args.post_ema_alpha,
            'filter_passes': int(args.filter_passes),
            'filter_bidir': bool(args.filter_bidir),
            'median_window': int(args.median_window),
            'scaler_used': scaler_used,
            'scaler_mode': args.scaler_mode,
            'refit_scaler': refit_used,
            **calib_info,
            'norm_start_one': args.norm_start_one,
            'norm_kind': args.norm_kind,
            **mets
        }, f, indent=2)

    title = f"{Path(parquet_path).stem} seq2many"
    # normalization for plotting (optional)
    yp_plot = preds
    yt_plot = y_true
    if args.norm_start_one != 'off':
        eps = 1e-12
        if args.norm_start_one == 'per_series':
            if yt_plot is not None and len(yt_plot) > 0:
                if args.norm_kind == 'scale':
                    d = float(yt_plot[0]) if abs(float(yt_plot[0])) > eps else 1.0
                    yt_plot = yt_plot / d
                else:
                    yt_plot = yt_plot - yt_plot[0] + 1.0
            if len(yp_plot) > 0:
                if args.norm_kind == 'scale':
                    d = float(yp_plot[0]) if abs(float(yp_plot[0])) > eps else 1.0
                    yp_plot = yp_plot / d
                else:
                    yp_plot = yp_plot - yp_plot[0] + 1.0
        elif args.norm_start_one == 'by_true' and yt_plot is not None and len(yt_plot) > 0:
            if args.norm_kind == 'scale':
                d = float(yt_plot[0]) if abs(float(yt_plot[0])) > eps else 1.0
                yt_plot = yt_plot / d
                yp_plot = yp_plot / d
            else:
                delta = 1.0 - float(yt_plot[0])
                yt_plot = yt_plot + delta
                yp_plot = yp_plot + delta

    # downsample for plotting to keep it fast on very long series
    if args.plot_max and len(yp_plot) > int(args.plot_max):
        import math as _math
        step = int(_math.ceil(len(yp_plot) / float(int(args.plot_max))))
        yp_plot = yp_plot[::step]
        if yt_plot is not None:
            yt_plot = yt_plot[::step]
    overlay_plot(yp_plot, yt_plot, out_root / 'overlay_full.png', title, firstN=2000)

    print(f"Saved results to: {out_root}")


if __name__ == '__main__':
    main()
