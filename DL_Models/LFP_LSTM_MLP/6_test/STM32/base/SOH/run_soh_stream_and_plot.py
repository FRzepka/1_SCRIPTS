#!/usr/bin/env python3
"""
STM32 SOH Base (FP32) – stream N samples, collect SOH and save plots + metrics.

Sends 7 raw SOH features per row from a Parquet file to the STM32 over UART.
Expects the board to reply with lines like: "SOH: <float>".

Outputs in DL_Models/LFP_LSTM_MLP/6_test/STM32/base/SOH/results/STM32_SOH_STREAM_<ts>/
 - overlay_full.png, overlay_firstN.png, diff_hist.png, metrics.json, log.txt
"""
import argparse
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import serial
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

SCRIPT_DIR = Path(__file__).resolve().parent
# Repo structure: .../1_Scripts/DL_Models/LFP_LSTM_MLP/6_test/STM32/base/SOH/
# parents[6] -> 1_Scripts (repo root)
REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_YAML = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '2.1.0.0' / 'config' / 'train_soh.yaml'


def _yaml_features(yaml_path: Optional[Path]) -> Optional[List[str]]:
    if not yaml_path:
        return None
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        feats = data.get('model', {}).get('features')
        if isinstance(feats, list) and len(feats) == 7:
            return [str(x) for x in feats]
    except Exception:
        pass
    return None


def pick_columns(df: pd.DataFrame, need: int, cols_arg: Optional[str], yaml_path: Optional[Path]) -> List[str]:
    feats = _yaml_features(yaml_path)
    if feats and all(c in df.columns or (c=='Temperature[°C]' and 'Temperature[�C]' in df.columns) or (c=='Temperature[�C]' and 'Temperature[°C]' in df.columns) for c in feats):
        mapped = []
        for f in feats:
            if f in df.columns: mapped.append(f)
            elif f=='Temperature[°C]' and 'Temperature[�C]' in df.columns: mapped.append('Temperature[�C]')
            elif f=='Temperature[�C]' and 'Temperature[°C]' in df.columns: mapped.append('Temperature[°C]')
        if len(mapped)==need:
            return mapped
    if cols_arg:
        parts = [c.strip() for c in cols_arg.split(',') if c.strip()]
        if len(parts) != need:
            raise ValueError(f'Please provide exactly {need} columns (got {len(parts)})')
        for p in parts:
            if p not in df.columns:
                raise ValueError(f"Column '{p}' not found")
        bad = [c for c in parts if not is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f'Selected non-numeric columns: {bad}')
        return parts
    # fallback: first numeric columns
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if len(num_cols) >= need:
        return num_cols[:need]
    raise ValueError(f'Could not determine {need} feature columns; provide --yaml or --cols')


def main():
    ap = argparse.ArgumentParser(description='STM32 SOH Base: stream N samples and plot')
    ap.add_argument('--port', required=True)
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--yaml', default=str(DEFAULT_YAML))
    ap.add_argument('--ckpt', default='', help='Optional .pt to read exact feature order (like quantized script)')
    ap.add_argument('--cols', default='')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--delay', type=float, default=0.01)
    ap.add_argument('--timeout', type=float, default=2.0)
    ap.add_argument('--prime', type=int, default=2047, help='Priming samples (no capture)')
    ap.add_argument('--strict_filter', action='store_true', help='Apply caps + EMA like PC test')
    ap.add_argument('--post_max_rel', type=float, default=None)
    ap.add_argument('--post_max_abs', type=float, default=None)
    ap.add_argument('--post_ema_alpha', type=float, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    yaml_path = Path(args.yaml) if args.yaml else None
    cols = None
    if args.ckpt:
        try:
            import torch  # lazy import
            ckpt = torch.load(args.ckpt, map_location='cpu')
            feats = ckpt.get('features')
            if isinstance(feats, list) and len(feats) == 7 and all((f in df.columns or (f=='Temperature[\u00B0C]' and 'Temperature[��C]' in df.columns) or (f=='Temperature[��C]' and 'Temperature[\u00B0C]' in df.columns)) for f in feats):
                mapped = []
                for f in feats:
                    if f in df.columns: mapped.append(f)
                    elif f=='Temperature[\u00B0C]' and 'Temperature[��C]' in df.columns: mapped.append('Temperature[��C]')
                    elif f=='Temperature[��C]' and 'Temperature[\u00B0C]' in df.columns: mapped.append('Temperature[\u00B0C]')
                if len(mapped)==7:
                    cols = mapped
        except Exception:
            cols = None
    if cols is None:
        cols = pick_columns(df, need=7, cols_arg=args.cols or None, yaml_path=yaml_path)
    if 'SOH' not in df.columns:
        raise ValueError('SOH column not found in parquet')

    start = int(args.start)
    end = min(len(df), start + max(0, int(args.n)))
    if start >= end:
        raise ValueError('Empty selection: check --start/--n')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Save in results folder: .../6_test/STM32/base/SOH/results/STM32_SOH_STREAM_*
    out_dir = SCRIPT_DIR / 'results' / f'STM32_SOH_STREAM_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'log.txt'

    cap_start = start + max(0, int(args.prime))
    cap_start = min(cap_start, end)
    y_true = df.loc[cap_start:end-1, 'SOH'].astype('float32').to_numpy()
    stm_preds = np.full((end - cap_start,), np.nan, dtype=np.float32)
    timeouts = 0

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser, open(log_path, 'w', encoding='utf-8') as logf:
        ser.reset_input_buffer(); ser.reset_output_buffer(); time.sleep(0.2)
        boot = ser.read(ser.in_waiting or 1).decode(errors='ignore')
        if boot:
            print(boot.strip()); logf.write(boot)
        # Prime
        for idx in range(start, cap_start):
            row = df.loc[idx, cols].astype('float32').tolist()
            line = ' '.join(f'{v:.6f}' for v in row) + '\n'
            ser.write(line.encode('ascii')); ser.flush()
            if args.delay > 0: time.sleep(args.delay)
            # Drain any pending device output during priming to avoid buffer buildup
            _ = ser.read(ser.in_waiting or 0)
            if (idx - start + 1) % 500 == 0:
                print(f'Primed {idx - start + 1}/{cap_start - start} samples...')

        # Drain leftover lines for a short window before capture
        drain_until = time.time() + 1.0
        while time.time() < drain_until:
            if ser.in_waiting:
                _ = ser.read(ser.in_waiting)
            else:
                time.sleep(0.02)
        # Capture
        t0 = time.time()
        for i, idx in enumerate(range(cap_start, end)):
            row = df.loc[idx, cols].astype('float32').tolist()
            line = ' '.join(f'{v:.6f}' for v in row) + '\n'
            ser.write(line.encode('ascii')); ser.flush()
            if args.delay > 0: time.sleep(args.delay)
            # Read until SOH or timeout
            deadline = time.time() + args.timeout
            parts = []; pred = None
            while time.time() < deadline:
                # Non-blocking drain then bounded blocking read
                if ser.in_waiting:
                    raw = ser.readline()
                else:
                    raw = ser.readline()
                if not raw: continue
                text = raw.decode(errors='ignore').strip()
                if text: parts.append(text)
                if 'SOH' in text.upper():
                    toks = text.replace(':', ' ').split()
                    for t in toks:
                        try: pred = float(t); break
                        except Exception: pass
                    if pred is not None: break
            if pred is None: timeouts += 1
            else: stm_preds[i] = pred
            logf.write(f'[{idx}] {" ".join(map(str, row))} | {" | ".join(parts)}\n')
            if (i+1) % 500 == 0:
                rate = (i+1)/max(1e-6, (time.time()-t0))
                print(f'{i+1}/{end-cap_start} samples | {rate:.1f} samples/s | timeouts={timeouts}')

    # Optional filter
    if args.strict_filter:
        post_max_abs = args.post_max_abs if args.post_max_abs is not None else 2e-5
        post_max_rel = args.post_max_rel if args.post_max_rel is not None else 5e-4
        alpha = args.post_ema_alpha if args.post_ema_alpha is not None else 0.005
        out = []
        last = None
        for p in stm_preds:
            v = float(p)
            if last is not None and np.isfinite(last) and np.isfinite(v):
                caps = []
                if post_max_rel is not None: caps.append(abs(last) * post_max_rel)
                if post_max_abs is not None: caps.append(post_max_abs)
                if caps:
                    cap = min(caps)
                    dv = v - last
                    if dv > cap: v = last + cap
                    elif dv < -cap: v = last - cap
                if alpha is not None and alpha > 0:
                    v = (1.0 - alpha) * last + alpha * v
            out.append(v)
            last = v
        stm_preds = np.array(out, dtype=np.float32)

    # Metrics + plots
    valid = ~np.isnan(stm_preds)
    diffs = np.abs(stm_preds[valid] - y_true[valid])
    mae = float(np.mean(diffs)) if valid.any() else float('nan')
    rmse = float(np.sqrt(np.mean((stm_preds[valid] - y_true[valid])**2))) if valid.any() else float('nan')
    mxd = float(np.max(diffs)) if valid.any() else float('nan')

    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'N': int(end - cap_start),
            'valid': int(valid.sum()),
            'timeouts': int(timeouts),
            'MAE_vs_GT': mae,
            'RMSE_vs_GT': rmse,
            'MAX_vs_GT': mxd,
            'y_true_min': float(np.nanmin(y_true)) if len(y_true) else float('nan'),
            'y_true_max': float(np.nanmax(y_true)) if len(y_true) else float('nan'),
            'pred_min': float(np.nanmin(stm_preds)) if len(stm_preds) else float('nan'),
            'pred_max': float(np.nanmax(stm_preds)) if len(stm_preds) else float('nan'),
            'features': cols,
            'parquet': str(Path(args.parquet).resolve()),
            'yaml': str(Path(args.yaml).resolve()) if args.yaml else '',
            'port': args.port,
            'baud': args.baud,
            'prime': int(max(0, int(args.prime))),
            'strict_filter': bool(args.strict_filter),
        }, f, indent=2)

    # Plots
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label='GT', alpha=0.6)
    plt.plot(stm_preds, label='STM32 SOH (base)', alpha=0.8)
    plt.title(f'Overlay full ({len(stm_preds)})')
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'overlay_full.png', dpi=150); plt.close()

    firstN = min(500, len(stm_preds))
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:firstN], label='GT', alpha=0.6)
    plt.plot(stm_preds[:firstN], label='STM32 SOH (base)', alpha=0.8)
    plt.title(f'Overlay first {firstN}')
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'overlay_firstN.png', dpi=150); plt.close()

    if valid.any():
        plt.figure(figsize=(6, 4))
        plt.hist(diffs, bins=100, alpha=0.8, color='tab:blue', edgecolor='black')
        plt.title('Abs error vs GT (STM32 SOH base)')
        plt.tight_layout(); plt.savefig(out_dir / 'diff_hist.png', dpi=150); plt.close()

    print(f"Saved results to: {out_dir}")


if __name__ == '__main__':
    main()
