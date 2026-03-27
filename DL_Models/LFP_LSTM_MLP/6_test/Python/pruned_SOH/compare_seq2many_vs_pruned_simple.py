#!/usr/bin/env python3
"""
Simple comparison: SOH seq2many FP32 baseline vs pruned checkpoint.

Uses the shared seq2many helpers from base_SOH/run_seq2many.py to ensure
identical preprocessing and inference path.

Outputs under:
  DL_Models/LFP_LSTM_MLP/6_test/Python/pruned_SOH/COMPARE_SEQ2MANY_VS_PRUNED_<timestamp>/<cell>/
    - arrays.npz (y_true, y_base, y_pruned)
    - overlay_full.png, overlay_firstN.png, diff_hist.png
    - metrics.json (baseline vs GT, pruned vs GT, baseline vs pruned)
"""
import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import importlib.util


REPO = Path(__file__).resolve().parents[5]
SEQ2MANY_PY = REPO / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'Python' / 'base_SOH' / 'python' / 'run_seq2many.py'
spec = importlib.util.spec_from_file_location('run_seq2many', str(SEQ2MANY_PY))
seq2m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seq2m)


def overlay_plot(y_true, y_base, y_pruned, out_path: Path, title: str, first_only: bool = False, max_points: int = 200000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def _maybe_downsample(arr):
        n = len(arr)
        if n <= max_points:
            return arr, 1
        step = int(np.ceil(n / max_points))
        return arr[::step], step

    yt, step_t = (None, 1)
    if y_true is not None:
        yt, step_t = _maybe_downsample(y_true)
    yb, step_b = _maybe_downsample(y_base)
    yp, step_p = _maybe_downsample(y_pruned)
    step_used = max(step_t, step_b, step_p)

    plt.figure(figsize=(12, 4))
    if yt is not None:
        plt.plot(yt, label='SOH true', linewidth=0.6, alpha=0.6, color='gray')
    plt.plot(yb, label='seq2many base', linewidth=1.0, alpha=0.8, color='blue')
    plt.plot(yp, label='seq2many pruned', linewidth=1.0, alpha=0.8, color='orange', linestyle='--')
    if step_used > 1:
        plt.title(f"{title} (downsampled by {step_used}x)")
    else:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def diff_hist(diff, out_path: Path, max_samples: int = 1000000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(diff) > max_samples:
        idx = np.random.RandomState(0).choice(len(diff), size=max_samples, replace=False)
        diff_plot = diff[idx]
    else:
        diff_plot = diff
    plt.figure(figsize=(6, 4))
    plt.hist(diff_plot, bins=120, alpha=0.85, color='tab:purple', edgecolor='black')
    plt.title('Base - Pruned residuals')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def seq2many_predictions_online_filtered(Xs: np.ndarray, model: torch.nn.Module, device: torch.device, chunk: int,
                                         block_len: int, limit: int, rel_cap: float, abs_cap: float, ema_alpha: float,
                                         prime: bool = True):
    total = Xs.shape[0]
    preds = []
    state = None
    start = 0
    last = None
    ema = None
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
            for bp in block_preds[:take]:
                v = float(bp)
                if last is not None:
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
                if ema_alpha is not None and 0 < ema_alpha < 1:
                    ema = v if ema is None else (ema_alpha * v + (1 - ema_alpha) * ema)
                    v = ema
                preds.append(v)
                last = v
            pbar.update(take)
            i = end
        pbar.close()
    return np.asarray(preds, dtype=np.float32), start


def parse_args():
    default_base = REPO / 'DL_Models' / 'LFP_LSTM_MLP' / '2_models' / 'base' / 'soh_2.1.0.0_base' / '2.1.0.0_soh_epoch0120_rmse0.03359.pt'
    ap = argparse.ArgumentParser(description='Compare seq2many SOH baseline vs pruned checkpoint (simple seq2many-only)')
    ap.add_argument('--baseline-ckpt', type=str, default=str(default_base))
    ap.add_argument('--pruned-ckpt', type=str, required=True)
    ap.add_argument('--data-root', type=str, default='/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE')
    ap.add_argument('--cell', type=str, default='MGFarm_18650_C13')
    ap.add_argument('--out-dir', type=str, default='')
    ap.add_argument('--limit', type=int, default=500000, help='Predictions after priming (<=0 for full)')
    ap.add_argument('--block-len', type=int, default=8192)
    ap.add_argument('--quiet', action='store_true')
    
    # Filter / Calibration args (copied from run_seq2many.py)
    ap.add_argument('--strict-filter', action='store_true')
    ap.add_argument('--post-max-rel', type=float, default=None)
    ap.add_argument('--post-max-abs', type=float, default=None)
    ap.add_argument('--post-ema-alpha', type=float, default=None)
    ap.add_argument('--filter-passes', type=int, default=1)
    ap.add_argument('--filter-bidir', action='store_true')
    ap.add_argument('--median-window', type=int, default=0)
    
    ap.add_argument('--calib-start-one', action='store_true')
    ap.add_argument('--calib-kind', choices=['scale','shift'], default='scale')
    ap.add_argument('--calib-anchor', choices=['by_pred','by_true'], default='by_pred')
    ap.add_argument('--calib-mode', choices=['uniform','pin','decay'], default='uniform')
    ap.add_argument('--calib-decay-tau', type=int, default=5000)
    ap.add_argument('--calib-apply', choices=['after_filter','before_filter'], default='after_filter')
    ap.add_argument('--filter-mode', choices=['post','online'], default='post', help='post: filter after full preds; online: step-wise filter during postprocess loop')
    ap.add_argument('--no-plots', action='store_true', help='Skip plotting (metrics/arrays still saved)')

    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[config] baseline={args.baseline_ckpt}\n         pruned={args.pruned_ckpt}\n         cell={args.cell}\n         limit={args.limit}", flush=True)

    base_model, base_features, base_chunk, base_scaler_hint = seq2m.load_checkpoint(args.baseline_ckpt, device)
    pruned_model, pruned_features, pruned_chunk, pruned_scaler_hint = seq2m.load_checkpoint(args.pruned_ckpt, device)

    print(f"[model] base_feat={len(base_features)} chunk={base_chunk} | pruned_feat={len(pruned_features)} chunk={pruned_chunk}", flush=True)

    if base_features != pruned_features:
        raise ValueError(f'Feature lists differ between baseline and pruned checkpoints:\nbase={base_features}\npruned={pruned_features}')
    features = list(base_features)
    chunk = min(base_chunk, pruned_chunk)

    print(f"[use] features={features} | chunk={chunk}", flush=True)

    run_root = Path(args.out_dir) if args.out_dir else (REPO / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'Python' / 'pruned_SOH' / f'COMPARE_SEQ2MANY_VS_PRUNED_{time.strftime("%Y%m%d_%H%M%S")}')
    run_root.mkdir(parents=True, exist_ok=True)
    cell_dir = run_root / args.cell
    cell_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = seq2m.locate_cell_parquet(args.data_root, args.cell)
    print(f"[data] loading parquet: {parquet_path}", flush=True)
    df = pd.read_parquet(parquet_path)
    print(f"[data] rows total={len(df)} cols={len(df.columns)}", flush=True)
    cols = seq2m.map_features_with_fallback(df, features)
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ['SOH'])
    if clean.empty:
        raise ValueError('No valid rows after cleaning/filtering')
    print(f"[data] after cleaning rows={len(clean)}", flush=True)

    # single scaler, based on baseline hint (or refit)
    scaler, scaler_used = seq2m.load_or_refit_scaler(clean, cols, base_scaler_hint, cell_dir)
    print(f"[scaler] using: {scaler_used}", flush=True)

    X = clean[cols].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    y_full = clean['SOH'].to_numpy(dtype=np.float32)

    # Choose prediction path (online filter vs. normal)
    use_online = args.strict_filter and args.filter_mode == 'online'
    if use_online:
        print(f"[pred] using ONLINE filter during inference (rel={args.post_max_rel}, abs={args.post_max_abs}, ema={args.post_ema_alpha})", flush=True)
        if args.filter_pases > 1 or args.filter_bidir or args.median_window:
            print("[warn] extra filter_passes/bidir/median are ignored in online mode", flush=True)
        preds_base, start_base = seq2many_predictions_online_filtered(
            Xs, base_model, device, chunk, args.block_len,
            args.limit if args.limit else -1,
            args.post_max_rel, args.post_max_abs, args.post_ema_alpha,
            prime=True
        )
        preds_pruned, start_pruned = seq2many_predictions_online_filtered(
            Xs, pruned_model, device, chunk, args.block_len,
            args.limit if args.limit else -1,
            args.post_max_rel, args.post_max_abs, args.post_ema_alpha,
            prime=True
        )
    else:
        preds_base, start_base = seq2m.seq2many_predictions(
            Xs, base_model, device, chunk,
            args.block_len, args.limit if args.limit else -1, prime=True
        )
        preds_pruned, start_pruned = seq2m.seq2many_predictions(
            Xs, pruned_model, device, chunk,
            args.block_len, args.limit if args.limit else -1, prime=True
        )

    print(f"[pred] base_len={len(preds_base)} start={start_base} | pruned_len={len(preds_pruned)} start={start_pruned}", flush=True)

    # --- Apply Filters / Calibration (Identical logic for both) ---
    def apply_post_processing(preds, y_true_full, start_idx):
        # Extract corresponding GT for calibration if needed
        y_true_segment = None
        if y_true_full is not None:
            y_true_segment = y_true_full[start_idx : start_idx + len(preds)]

        # Helper to call seq2m calibration logic (re-implementing briefly since seq2m main() has it inline)
        # Actually, seq2m has helper functions: strict_filter, bidirectional_ema, median_filter_1d
        # But calibration logic is inside main() in run_seq2many.py. We need to replicate it or extract it.
        # Since we can't easily extract it without editing run_seq2many.py, we'll replicate the logic here.
        
        def _apply_calib(p, yt):
            if not (args.calib_start_one and len(p) > 0):
                return p
            eps = 1e-12
            if args.calib_anchor == 'by_true' and yt is not None and len(yt) > 0:
                base = float(yt[0])
            else:
                base = float(p[0])
            
            mode = args.calib_mode
            y = p
            if mode == 'pin':
                y = y.copy()
                y[0] = 1.0
                return y
            elif mode == 'decay':
                T = y.shape[0]
                w = np.exp(-np.arange(T, dtype=np.float64) / max(1, int(args.calib_decay_tau)))
                if args.calib_kind == 'scale':
                    g = (1.0 / base) if abs(base) > eps else 1.0
                    y = (y.astype(np.float64) * (1.0 + w * (g - 1.0))).astype(np.float32)
                else:
                    dlt = 1.0 - base
                    y = (y.astype(np.float64) + w * dlt).astype(np.float32)
                return y
            else: # uniform
                if args.calib_kind == 'scale':
                    gain = (1.0 / base) if abs(base) > eps else 1.0
                    y = (y * gain).astype(np.float32)
                else:
                    delta = 1.0 - base
                    y = (y + delta).astype(np.float32)
                return y

        # 1. Calibration (Before)
        if args.calib_apply == 'before_filter':
            preds = _apply_calib(preds, y_true_segment)

        # 2. Strict Filter
        if args.strict_filter:
            if args.filter_mode == 'online':
                print(f"[filter] applying ONLINE strict_filter (len={len(preds)}, rel={args.post_max_rel}, abs={args.post_max_abs}, ema={args.post_ema_alpha})", flush=True)
                # step-wise filter
                out = []
                last = None
                ema = None
                rel = args.post_max_rel
                abs_cap = args.post_max_abs
                alpha = args.post_ema_alpha
                for p in preds:
                    v = float(p)
                    if last is not None:
                        if rel is not None:
                            cap_rel = abs(last) * rel
                        else:
                            cap_rel = None
                        cap_abs = abs_cap
                        if cap_rel is not None or cap_abs is not None:
                            caps = []
                            if cap_rel is not None:
                                caps.append(cap_rel)
                            if cap_abs is not None:
                                caps.append(cap_abs)
                            if caps:
                                cap = min(caps)
                                delta = v - last
                                if abs(delta) > cap:
                                    v = last + (cap if delta > 0 else -cap)
                    if alpha is not None and 0 < alpha < 1:
                        ema = v if ema is None else (alpha * v + (1 - alpha) * ema)
                        v = ema
                    out.append(v)
                    last = v
                preds = np.asarray(out, dtype=np.float32)
                # extra passes + bidir + median can still apply on top
                for _ in range(max(0, int(args.filter_passes) - 1)):
                    preds = seq2m.strict_filter(preds, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
                if args.filter_bidir and args.post_ema_alpha and 0 < args.post_ema_alpha < 1:
                    preds = seq2m.bidirectional_ema(preds, args.post_ema_alpha)
                if args.median_window and int(args.median_window) > 1:
                    preds = seq2m.median_filter_1d(preds, int(args.median_window))
            else:
                print(f"[filter] applying strict_filter (len={len(preds)}, rel={args.post_max_rel}, abs={args.post_max_abs}, ema={args.post_ema_alpha})", flush=True)
                preds = seq2m.strict_filter(preds, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
                for _ in range(max(0, int(args.filter_passes) - 1)):
                    preds = seq2m.strict_filter(preds, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
                if args.filter_bidir and args.post_ema_alpha and 0 < args.post_ema_alpha < 1:
                    preds = seq2m.bidirectional_ema(preds, args.post_ema_alpha)
                if args.median_window and int(args.median_window) > 1:
                    preds = seq2m.median_filter_1d(preds, int(args.median_window))

        # 3. Calibration (After)
        if args.calib_apply == 'after_filter':
            preds = _apply_calib(preds, y_true_segment)
            
        return preds

    preds_base = apply_post_processing(preds_base, y_full, start_base)
    preds_pruned = apply_post_processing(preds_pruned, y_full, start_pruned)
    # ------------------------------------------------------------

    end_base = start_base + len(preds_base)
    end_pruned = start_pruned + len(preds_pruned)
    span_start = max(start_base, start_pruned)
    span_end = min(end_base, end_pruned)
    if span_end <= span_start:
        raise RuntimeError('Prediction ranges do not overlap between baseline and pruned outputs')
    span_len = span_end - span_start
    print(f"[align] span_start={span_start} span_end={span_end} span_len={span_len}", flush=True)

    def slice_span(preds, start_idx):
        offset = span_start - start_idx
        return preds[offset:offset + span_len]

    y_base = slice_span(preds_base, start_base)
    y_pruned = slice_span(preds_pruned, start_pruned)
    y_true = y_full[span_start:span_end]

    # Check for NaNs and handle them gracefully
    nan_mask = np.isnan(y_base) | np.isnan(y_pruned) | np.isnan(y_true)
    if nan_mask.any():
        n_nans = nan_mask.sum()
        print(f"Warning: Found {n_nans} samples with NaNs in predictions or ground truth.")
        if n_nans == len(y_true):
            print("Error: All samples contain NaNs. Cannot compute metrics.")
            mets_base = {'mae': None, 'rmse': None, 'r2': None, 'n': 0}
            mets_pruned = {'mae': None, 'rmse': None, 'r2': None, 'n': 0}
            diff_metrics = {'mae': None, 'rmse': None, 'max_abs': None}
            diff = np.array([], dtype=np.float32)
        else:
            print("Filtering out NaN samples for metric computation.")
            valid_mask = ~nan_mask
            y_true_valid = y_true[valid_mask]
            y_base_valid = y_base[valid_mask]
            y_pruned_valid = y_pruned[valid_mask]

            mets_base = seq2m.compute_metrics(y_true_valid, y_base_valid)
            mets_pruned = seq2m.compute_metrics(y_true_valid, y_pruned_valid)

            diff_valid = y_base_valid - y_pruned_valid
            diff_metrics = {
                'mae': float(np.mean(np.abs(diff_valid))) if diff_valid.size else None,
                'rmse': float(np.sqrt(np.mean(diff_valid ** 2))) if diff_valid.size else None,
                'max_abs': float(np.max(np.abs(diff_valid))) if diff_valid.size else None,
            }
            diff = diff_valid
    else:
        mets_base = seq2m.compute_metrics(y_true, y_base)
        mets_pruned = seq2m.compute_metrics(y_true, y_pruned)
        diff = y_base - y_pruned
        diff_metrics = {
            'mae': float(np.mean(np.abs(diff))) if diff.size else None,
            'rmse': float(np.sqrt(np.mean(diff ** 2))) if diff.size else None,
            'max_abs': float(np.max(np.abs(diff))) if diff.size else None,
        }

    print(f"[metrics] base_vs_gt mae={mets_base.get('mae')} rmse={mets_base.get('rmse')}", flush=True)
    print(f"[metrics] pruned_vs_gt mae={mets_pruned.get('mae')} rmse={mets_pruned.get('rmse')} ", flush=True)
    print(f"[metrics] base_vs_pruned mae={diff_metrics['mae']} rmse={diff_metrics['rmse']}", flush=True)

    np.savez_compressed(cell_dir / 'arrays.npz', y_true=y_true, y_base=y_base, y_pruned=y_pruned)
    if not args.no_plots:
        overlay_plot(y_true, y_base, y_pruned, cell_dir / 'overlay_full.png', f'{args.cell} – seq2many base vs pruned')
        firstN = min(2000, span_len)
        overlay_plot(y_true[:firstN], y_base[:firstN], y_pruned[:firstN], cell_dir / f'overlay_first{firstN}.png', f'{args.cell} – first {firstN} samples', first_only=True)
        if not nan_mask.all():
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
