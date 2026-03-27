#!/usr/bin/env python3
"""
Live streaming to STM32 (quantized SOH) with rolling plot.

Streams SOH features row-by-row from a parquet file to the STM32 over UART,
parses lines like "SOH: <float>", and updates a live plot showing the last
--window samples (default 10000). Runs continuously until Ctrl-C; optionally
wraps at EOF (see --repeat).

Notes
- The STM32 firmware must print lines containing "SOH:" per received row.
- This script sends RAW features; the model on STM32 applies its own scaler.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # avoid OpenMP runtime conflicts

import argparse
import time
from pathlib import Path
from typing import List, Optional
from collections import deque

import numpy as np
import pandas as pd
import serial  # pyserial
import yaml

import matplotlib
matplotlib.use('TkAgg')  # prefer Tk over Qt to avoid OpenMP issues
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
# Repo structure: .../1_Scripts/DL_Models/LFP_LSTM_MLP/6_test/STM32/quantized/SOH/
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


def pick_columns(df: pd.DataFrame, need: int, cols_arg: Optional[str], yaml_path: Optional[Path], ckpt_path: Optional[Path]) -> List[str]:
    # Prefer checkpoint feature order if provided
    if ckpt_path:
        try:
            import torch
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            feats = ckpt.get('features')
            if isinstance(feats, list) and len(feats) == need:
                mapped = []
                for f in feats:
                    if f in df.columns:
                        mapped.append(f)
                    elif f == 'Temperature[°C]' and 'Temperature[�C]' in df.columns:
                        mapped.append('Temperature[�C]')
                    elif f == 'Temperature[�C]' and 'Temperature[°C]' in df.columns:
                        mapped.append('Temperature[°C]')
                    else:
                        mapped = []
                        break
                if len(mapped) == need:
                    return mapped
        except Exception:
            pass
    # Next: YAML features
    feats = _yaml_features(yaml_path)
    if feats and all(c in df.columns or (c=='Temperature[°C]' and 'Temperature[�C]' in df.columns) or (c=='Temperature[�C]' and 'Temperature[°C]' in df.columns) for c in feats):
        mapped = []
        for f in feats:
            if f in df.columns: mapped.append(f)
            elif f=='Temperature[°C]' and 'Temperature[�C]' in df.columns: mapped.append('Temperature[�C]')
            elif f=='Temperature[�C]' and 'Temperature[°C]' in df.columns: mapped.append('Temperature[°C]')
        if len(mapped)==need:
            return mapped
    # Finally: explicit list
    if cols_arg:
        parts = [c.strip() for c in cols_arg.split(',') if c.strip()]
        if len(parts) != need:
            raise ValueError(f'Please provide exactly {need} columns (got {len(parts)})')
        for p in parts:
            if p not in df.columns:
                raise ValueError(f"Column '{p}' not in dataframe")
        return parts
    raise ValueError('Could not determine feature columns; provide --ckpt or --yaml or --cols')


def main():
    ap = argparse.ArgumentParser(description='STM32 Quantized SOH live plot with rolling window')
    ap.add_argument('--port', required=True, help='Serial port (e.g., COM9)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--parquet', required=True, help='Path to parquet with features + SOH')
    ap.add_argument('--yaml', default=str(DEFAULT_YAML), help='YAML for feature names (fallback)')
    ap.add_argument('--ckpt', default='', help='Optional .pt with exact feature order')
    ap.add_argument('--cols', default='', help='Comma-separated column names (overrides YAML)')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--window', type=int, default=10000)
    ap.add_argument('--delay', type=float, default=0.01, help='Delay after write (s)')
    ap.add_argument('--timeout', type=float, default=1.0, help='Per-sample max wait (s)')
    ap.add_argument('--repeat', action='store_true', help='Loop over parquet when reaching EOF')
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    yaml_path = Path(args.yaml) if args.yaml else None
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    cols = pick_columns(df, need=7, cols_arg=args.cols or None, yaml_path=yaml_path, ckpt_path=ckpt_path)
    soh_col = 'SOH' if 'SOH' in df.columns else None
    if not soh_col:
        raise ValueError('SOH column not found in parquet')

    # Prepare rolling buffers
    win = max(100, int(args.window))
    y_true = deque(maxlen=win)
    y_pred = deque(maxlen=win)
    x_idx = deque(maxlen=win)

    # Matplotlib live setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))
    ln_true, = ax.plot([], [], label='GT', alpha=0.6)
    ln_pred, = ax.plot([], [], label='STM32 SOH', alpha=0.85)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Live SOH (window={win})')
    ax.set_xlabel('sample index')
    ax.set_ylabel('SOH')

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        ser.reset_input_buffer(); ser.reset_output_buffer(); time.sleep(0.2)
        # flush boot text
        _ = ser.read(ser.in_waiting or 1)

        i = int(args.start)
        total = len(df)
        t0 = time.time(); n_sent = 0
        try:
            while True:
                if i >= total:
                    if args.repeat:
                        i = 0
                    else:
                        break
                row = df.loc[i, cols].astype('float32').tolist()
                gt = float(df.loc[i, soh_col]) if soh_col else np.nan
                line = ' '.join(f'{v:.6f}' for v in row) + '\n'
                ser.write(line.encode('ascii')); ser.flush()
                if args.delay > 0:
                    time.sleep(args.delay)
                # read response
                pred = np.nan
                deadline = time.time() + args.timeout
                while time.time() < deadline:
                    raw = ser.readline()
                    if not raw:
                        continue
                    text = raw.decode(errors='ignore').strip()
                    if 'SOH' in text.upper():
                        toks = text.replace(':', ' ').split()
                        for t in toks:
                            try:
                                pred = float(t)
                                break
                            except Exception:
                                pass
                        break
                # update buffers
                x_idx.append(i)
                y_true.append(gt)
                y_pred.append(pred)
                ln_true.set_data(range(len(y_true)), list(y_true))
                ln_pred.set_data(range(len(y_pred)), list(y_pred))
                ax.set_xlim(0, max(10, len(y_true)))
                # auto-scale y to recent data with small margin
                vals = [v for v in list(y_true)+list(y_pred) if np.isfinite(v)]
                if vals:
                    ymin, ymax = min(vals), max(vals)
                    if ymin == ymax:
                        ymin -= 0.001; ymax += 0.001
                    mrg = 0.02 * (ymax - ymin)
                    ax.set_ylim(ymin - mrg, ymax + mrg)
                fig.canvas.draw(); fig.canvas.flush_events()

                n_sent += 1
                if n_sent % 500 == 0:
                    elapsed = time.time() - t0
                    rate = n_sent/elapsed if elapsed>0 else 0.0
                    print(f"{n_sent} sent | {rate:.1f} samples/s")
                i += 1
        except KeyboardInterrupt:
            pass

    # keep window open until closed
    print('Stopped. Close the plot window to exit.')
    plt.ioff(); plt.show()


if __name__ == '__main__':
    main()

