#!/usr/bin/env python3
"""
Stream 6 features to STM32 over UART for N samples, collect predicted SOC,
compare to ground-truth SOC from parquet, and save plots + metrics under 6_tests.

Outputs in DL_Models/LFP_LSTM_MLP/6_test/STM32/quantized/SOC/results/STM32_SOC_QUANTIZED_STREAM_YYYYMMDD_HHMMSS:
 - overlay_full.png         (GT vs Pred over all N)
 - overlay_firstN.png       (first 500)
 - diff_hist.png            (abs error histogram)
 - metrics.json             (MAE, RMSE, MAX, N, timeouts)
 - log.txt                  (per-line log)
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import serial  # pyserial
import yaml
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
# Repo structure: .../1_Scripts/DL_Models/LFP_LSTM_MLP/6_test/STM32/quantized/SOC/
# parents[6] -> 1_Scripts (repo root)
REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_YAML = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '1.5.0.0' / 'config' / 'train_soc.yaml'


DEFAULT_FEATURES = [
    'Voltage[V]',
    'Current[A]',
    'Temperature[°C]',
    'Q_c',
    'dU_dt[V/s]',
    'dI_dt[A/s]',
]


def _try_yaml_features(yaml_path: Optional[Path]) -> Optional[List[str]]:
    if not yaml_path:
        return None
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        feats = data.get('model', {}).get('features')
        if isinstance(feats, list) and len(feats) == 6:
            return [str(x) for x in feats]
    except Exception:
        pass
    return None


def pick_columns(df: pd.DataFrame, need: int, cols_arg: Optional[str], yaml_path: Optional[Path]) -> List[str]:
    feats = _try_yaml_features(yaml_path)
    if feats and all(c in df.columns for c in feats):
        return feats
    if cols_arg:
        parts = [c.strip() for c in cols_arg.split(',') if c.strip()]
        cols: List[str] = []
        for p in parts:
            if p.isdigit():
                idx = int(p)
                if idx < 0 or idx >= len(df.columns):
                    raise ValueError(f'Column index {idx} out of range (0..{len(df.columns)-1})')
                cols.append(df.columns[idx])
            else:
                if p not in df.columns:
                    raise ValueError(f"Column '{p}' not found in parquet columns")
                cols.append(p)
        if len(cols) != need:
            raise ValueError(f'Please provide exactly {need} columns (got {len(cols)})')
        bad = [c for c in cols if not is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f'Selected non-numeric columns: {bad}')
        return cols
    if all(c in df.columns for c in DEFAULT_FEATURES):
        return DEFAULT_FEATURES
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if len(num_cols) >= need:
        return num_cols[:need]
    raise ValueError('Could not determine 6 feature columns. Provide --cols or --yaml.')


def main():
    ap = argparse.ArgumentParser(description='STM32 Quantized: N-step stream and plot')
    ap.add_argument('--port', required=True, help='Serial port (e.g., COM9)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--parquet', required=True, help='Path to parquet with features and SOC')
    ap.add_argument('--yaml', default=str(DEFAULT_YAML), help='YAML config to pick features (model.features)')
    ap.add_argument('--cols', default='', help='Comma-separated column names or indices (overrides YAML)')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--delay', type=float, default=0.01, help='Delay after write (s)')
    ap.add_argument('--timeout', type=float, default=2.0, help='Per-sample max wait for SOC (s)')
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    yaml_path = Path(args.yaml) if args.yaml else None
    cols = pick_columns(df, need=6, cols_arg=args.cols or None, yaml_path=yaml_path)
    soc_col = None
    for candidate in ['SOC', 'soc', 'SOC[%]', 'SoC']:
        if candidate in df.columns:
            soc_col = candidate
            break
    if not soc_col:
        raise ValueError('No SOC column found (tried: SOC, soc, SOC[%], SoC)')

    end = min(len(df), args.start + max(args.n, 0))
    if args.start >= end:
        raise ValueError('Empty selection: check --start/--n')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Save in results folder: .../6_test/STM32/quantized/SOC/results/STM32_SOC_QUANTIZED_STREAM_*
    out_dir = SCRIPT_DIR / 'results' / f'STM32_SOC_QUANTIZED_STREAM_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'log.txt'

    y_true = df.loc[args.start:end-1, soc_col].astype('float32').to_numpy()
    stm_preds = np.full((end - args.start,), np.nan, dtype=np.float32)
    timeouts = 0

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser, open(log_path, 'w', encoding='utf-8') as logf:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.2)
        boot = ser.read(ser.in_waiting or 1).decode(errors='ignore')
        if boot:
            print(boot.strip())
            logf.write(boot)
        t0 = time.time()
        for i, idx in enumerate(range(args.start, end)):
            row = df.loc[idx, cols].astype('float32').tolist()
            line = ' '.join(f'{v:.6f}' for v in row) + '\n'
            ser.write(line.encode('ascii'))
            ser.flush()
            if args.delay > 0:
                time.sleep(args.delay)
            # Read until SOC or timeout
            deadline = time.time() + args.timeout
            parts = []
            pred = None
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                text = raw.decode(errors='ignore').strip()
                if text:
                    parts.append(text)
                if 'SOC' in text.upper():
                    toks = text.replace(':', ' ').split()
                    for t in toks:
                        try:
                            pred = float(t)
                            break
                        except Exception:
                            pass
                    if pred is not None:
                        break
            resp = ' | '.join(parts)
            if pred is None:
                timeouts += 1
            else:
                stm_preds[i] = pred
            logf.write(f'[{idx}] {" ".join(map(str, row))} | {resp}\n')
            if (i+1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i+1)/elapsed if elapsed > 0 else 0.0
                print(f'{i+1}/{end-args.start} samples | {rate:.1f} samples/s | timeouts={timeouts}')

    # Metrics vs ground truth
    valid = ~np.isnan(stm_preds)
    diffs = np.abs(stm_preds[valid] - y_true[valid])
    mae = float(np.mean(np.abs(stm_preds[valid] - y_true[valid]))) if valid.any() else float('nan')
    rmse = float(np.sqrt(np.mean((stm_preds[valid] - y_true[valid])**2))) if valid.any() else float('nan')
    mxd = float(np.max(diffs)) if valid.any() else float('nan')

    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'N': int(end - args.start),
            'valid': int(valid.sum()),
            'timeouts': int(timeouts),
            'MAE_vs_GT': mae,
            'RMSE_vs_GT': rmse,
            'MAX_vs_GT': mxd,
            'features': cols,
            'parquet': str(Path(args.parquet).resolve()),
            'yaml': str(Path(args.yaml).resolve()) if args.yaml else '',
            'port': args.port,
            'baud': args.baud,
        }, f, indent=2)

    # Plots
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label='GT', alpha=0.6)
    plt.plot(stm_preds, label='STM32', alpha=0.8)
    plt.title(f'Overlay full ({len(stm_preds)})')
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'overlay_full.png', dpi=150); plt.close()

    firstN = min(500, len(stm_preds))
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:firstN], label='GT', alpha=0.6)
    plt.plot(stm_preds[:firstN], label='STM32', alpha=0.8)
    plt.title(f'Overlay first{firstN}')
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'overlay_firstN.png', dpi=150); plt.close()

    if valid.any():
        plt.figure(figsize=(6, 4))
        plt.hist(diffs, bins=100, alpha=0.8, color='tab:blue', edgecolor='black')
        plt.title('Abs error vs GT (STM32)')
        plt.tight_layout(); plt.savefig(out_dir / 'diff_hist.png', dpi=150); plt.close()

    print(f'\nSaved results to: {out_dir}')


if __name__ == '__main__':
    main()
