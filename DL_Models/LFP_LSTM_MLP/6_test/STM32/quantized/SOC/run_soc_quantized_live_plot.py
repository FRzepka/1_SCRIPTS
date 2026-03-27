#!/usr/bin/env python3
"""
STM32 Quantized Model: Live rolling plot (default window=1000)
"""
import argparse
import re
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import serial
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
# Repo structure: .../1_Scripts/DL_Models/LFP_LSTM_MLP/6_test/STM32/quantized/SOC/
# parents[6] -> 1_Scripts (repo root)
REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_YAML = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '1.5.0.0' / 'config' / 'train_soc.yaml'

DEFAULT_FEATURES = ['Voltage[V]','Current[A]','Temperature[°C]','Q_c','dU_dt[V/s]','dI_dt[A/s]']


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


def extract_soc(text: str):
    t = text.strip()
    if not t:
        return None
    if 'SOC' in t.upper():
        toks = t.replace(':', ' ').split()
        for tok in toks:
            try:
                return float(tok)
            except Exception:
                pass
    m = re.search(r'([-+]?\d+(?:\.\d+)?)', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description='STM32 Quantized Live Plot (rolling window)')
    ap.add_argument('--port', required=True)
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--yaml', default=str(DEFAULT_YAML))
    ap.add_argument('--cols', default='')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--delay', type=float, default=0.01)
    ap.add_argument('--timeout', type=float, default=1.5)
    ap.add_argument('--window', type=int, default=1000)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    yaml_path = Path(args.yaml) if args.yaml else None
    cols = pick_columns(df, 6, args.cols or None, yaml_path)
    soc_col = next((c for c in ['SOC','soc','SOC[%]','SoC'] if c in df.columns), None)
    if not soc_col:
        raise ValueError('No SOC column found')
    end = len(df) if args.n == 0 else min(len(df), args.start + max(args.n, 0))
    if args.start >= end:
        raise ValueError('Empty selection')

    ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    time.sleep(0.2)
    ser.reset_input_buffer(); ser.reset_output_buffer()
    _ = ser.read(ser.in_waiting or 1)

    xs, gt, pr, er = deque(maxlen=args.window), deque(maxlen=args.window), deque(maxlen=args.window), deque(maxlen=args.window)
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
    fig.suptitle('STM32 Quantized Live')
    l_gt, = ax1.plot([],[],'b-',label='GT'); l_pr, = ax1.plot([],[],'r--',label='Pred'); ax1.legend(); ax1.grid(True, alpha=0.3)
    l_er, = ax2.plot([],[],'g-',label='Error'); ax2.axhline(0,color='k',linestyle='--',alpha=0.3); ax2.legend(); ax2.grid(True,alpha=0.3)

    try:
        for idx in range(args.start, end):
            row = df.loc[idx, cols].astype('float32').tolist()
            y = float(df.loc[idx, soc_col])
            line = ' '.join(f'{v:.6f}' for v in row) + '\n'
            ser.write(line.encode('ascii')); ser.flush()
            if args.delay>0: time.sleep(args.delay)
            # read until soc or timeout
            t_dead = time.time() + args.timeout
            pred=None
            while time.time()<t_dead:
                raw = ser.readline()
                if not raw: continue
                p = extract_soc(raw.decode(errors='ignore'))
                if p is not None:
                    pred=p; break
            if pred is None: continue
            xs.append(idx); gt.append(y); pr.append(pred); er.append(y-pred)
            l_gt.set_data(list(xs), list(gt)); l_pr.set_data(list(xs), list(pr)); l_er.set_data(list(xs), list(er))
            if len(xs)>1:
                ax1.set_xlim(xs[0], xs[-1])
                y_all=list(gt)+list(pr); y_min, y_max=min(y_all), max(y_all); marg=(y_max-y_min)*0.1 or 0.01
                ax1.set_ylim(y_min-marg, y_max+marg)
                e_min, e_max=min(er), max(er); e_m=max(abs(e_min),abs(e_max))*0.1 or 0.01
                ax2.set_ylim(e_min-e_m, e_max+e_m)
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close(); plt.ioff(); plt.show()


if __name__=='__main__':
    main()

