#!/usr/bin/env python3
"""
STM32 SOH PRUNED (FP32) – stream N samples, collect SOH predictions, and save plots + metrics.

This mirrors `run_base_stream_and_plot_soh.py` but targets the pruned STM32 firmware
(`AI_Project_LSTM_SOH_pruned`). Use it after flashing the board with
`STM32/workspace_1.17.0/AI_Project_LSTM_SOH_pruned`.

Outputs land in:
  DL_Models/LFP_LSTM_MLP/6_test/STM32/pruned/SOH/STM32_SOH_PRUNED_STREAM_<timestamp>/
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
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_YAML = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '2.1.0.0' / 'config' / 'train_soh.yaml'
DEFAULT_CKPT = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '2_models' / 'pruned' / 'soh_2.1.0.0' / 'prune_30pct_20251110_140853' / 'soh_pruned_hidden90.pt'


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


def _norm_name(s: str) -> str:
    return (str(s)
            .replace('��', 'o').replace('���', '')
            .replace('��', 'a').replace('�"', 'A')
            .replace('��', 'o').replace('�-', 'O')
            .replace('Ǭ', 'u').replace('�o', 'U')
            .replace('�Y', 'ss')
            .replace("'", "")
            .strip())

def _map_feature_name(name: str, df: pd.DataFrame) -> Optional[str]:
    if name in df.columns:
        return name
    # handle Temperature encodings flexibly
    if 'Temperature' in str(name):
        for cand in ('Temperature[��C]', 'Temperature[���C]', 'Temperature[°C]'):
            if cand in df.columns:
                return cand
    cols = list(df.columns)
    cols_norm = [_norm_name(c) for c in cols]
    nn = _norm_name(name)
    for i, cn in enumerate(cols_norm):
        if cn == nn:
            return cols[i]
    for i, cn in enumerate(cols_norm):
        if nn in cn:
            return cols[i]
    return None


def _map_temp(col_name: str, df: pd.DataFrame) -> Optional[str]:
    if col_name in df.columns:
        return col_name
    if col_name == 'Temperature[°C]' and 'Temperature[�C]' in df.columns:
        return 'Temperature[�C]'
    if col_name == 'Temperature[�C]' and 'Temperature[°C]' in df.columns:
        return 'Temperature[°C]'
    return None


def pick_columns(df: pd.DataFrame, need: int, cols_arg: Optional[str], yaml_path: Optional[Path], ckpt_path: Optional[Path], no_fallback: bool = True, allow_short: bool = False) -> List[str]:
    feats = None
    if ckpt_path and ckpt_path.exists():
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location='cpu')
            feats = ckpt.get('features')
        except Exception:
            feats = None
    if feats is None:
        feats = _yaml_features(yaml_path)
    if feats:
        mapped = []
        for f in feats:
            m = _map_feature_name(f, df)
            if m is None:
                mapped = []
                break
            mapped.append(m)
        if len(mapped) == need:
            return mapped
    if cols_arg:
        parts = [c.strip() for c in cols_arg.split(',') if c.strip()]
        if len(parts) != need:
            if allow_short and 0 < len(parts) < need:
                print(f"[warn] Allowing fewer columns than MCU INPUT_SIZE: got {len(parts)} < {need}. MCU will pad missing with scaler centers.")
            else:
                raise ValueError(f'Please provide exactly {need} columns (got {len(parts)})')
        for p in parts:
            if p not in df.columns:
                raise ValueError(f"Column '{p}' not found in dataframe")
        bad = [c for c in parts if not is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f'Selected non-numeric columns: {bad}')
        return parts
    if not no_fallback:
        numeric = [c for c in df.columns if is_numeric_dtype(df[c])]
        if len(numeric) >= need:
            print('[warn] Falling back to first numeric columns; verify order matches training features!', flush=True)
            return numeric[:need]
    raise ValueError((
        f'Could not resolve {need} feature columns from ckpt/yaml. Pass --cols explicitly (comma-separated).\n'
    ) + f'Available columns: {list(df.columns)}')


def _read_mcu_input_size() -> Optional[int]:
    """Try to read INPUT_SIZE from the pruned STM32 header, otherwise None."""
    try:
        hdr = (REPO_ROOT / 'STM32' / 'workspace_1.17.0' / 'AI_Project_LSTM_SOH_pruned' / 'Core' / 'Inc' / 'model_weights_soh.h')
        if not hdr.exists():
            return None
        txt = hdr.read_text(encoding='utf-8', errors='ignore')
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith('#define') and 'INPUT_SIZE' in line:
                parts = line.split()
                for i,p in enumerate(parts):
                    if p == 'INPUT_SIZE' and i+1 < len(parts):
                        return int(parts[i+1])
        return None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description='STM32 SOH PRUNED: stream samples over UART and plot results')
    ap.add_argument('--port', required=True, help='Serial port (e.g. /dev/ttyUSB0)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--parquet', required=True, help='Parquet file with SOH + features')
    ap.add_argument('--yaml', default=str(DEFAULT_YAML), help='YAML config to read feature order')
    ap.add_argument('--ckpt', default=str(DEFAULT_CKPT), help='Checkpoint to read feature order (overrides YAML if present)')
    ap.add_argument('--cols', default='', help='Comma-separated column names to force feature order (default: MCU INPUT_SIZE)')
    ap.add_argument('--allow-missing', action='store_true', help='Allow fewer columns than MCU INPUT_SIZE; MCU will pad missing dims with scaler centers')
    ap.add_argument('--no-fallback', action='store_true', help='Do not auto-pick first numeric columns when features cannot be resolved')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--delay', type=float, default=0.01)
    ap.add_argument('--timeout', type=float, default=2.0)
    ap.add_argument('--prime', type=int, default=2047, help='Priming samples (no capture, just to fill LSTM state)')
    ap.add_argument('--strict-filter', action='store_true', help='Apply step caps + EMA like PC sanity tests')
    ap.add_argument('--post-max-rel', type=float, default=None)
    ap.add_argument('--post-max-abs', type=float, default=None)
    ap.add_argument('--post-ema-alpha', type=float, default=None)
    ap.add_argument('--reset-state', action='store_true', help='Send RESET command before priming')
    ap.add_argument('--out-dir', default='', help='Override output directory (no timestamp if set)')
    # Optional: compare against Python seq2many baseline on the same slice
    ap.add_argument('--pc-seq2many', action='store_true', help='Also run Python seq2many on the same data slice and compare with STM32')
    ap.add_argument('--pc-block', type=int, default=8192, help='Seq2many block length for PC baseline')
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if 'SOH' not in df.columns:
        raise ValueError("Parquet lacks 'SOH' column")
    yaml_path = Path(args.yaml) if args.yaml else None
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    # Enforce exact MCU input dimensionality
    need = _read_mcu_input_size() or 7
    try:
        import torch, json
        st = torch.load(str(ckpt_path), map_location='cpu') if ckpt_path and ckpt_path.exists() else {}
        feats = st.get('features') or st.get('feature_list')
    except Exception:
        feats = None
    if feats:
        print(f"[info] CKPT features (order): {feats}")
    print(f"[info] MCU INPUT_SIZE from header: {need}")
    cols = pick_columns(df, need=need, cols_arg=args.cols or None, yaml_path=yaml_path, ckpt_path=ckpt_path, no_fallback=True, allow_short=args.allow_missing)
    print(f"[info] Using RAW columns (MCU scales): {cols}")
    if feats and set(map(str,feats)) != set(map(str,cols)):
        print("[warn] CKPT feature set != selected columns; ensure scaler header matches CKPT features exactly.")

    start = max(0, int(args.start))
    end = min(len(df), start + max(0, int(args.n)))
    if start >= end:
        raise ValueError('Empty range: adjust --start/--n')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir) if args.out_dir else (SCRIPT_DIR / f'STM32_SOH_PRUNED_STREAM_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'log.txt'

    prime_len = max(0, int(args.prime))
    cap_start = min(end, start + prime_len)
    y_true = df.loc[cap_start:end-1, 'SOH'].astype('float32').to_numpy()
    stm_preds = np.full((end - cap_start,), np.nan, dtype=np.float32)
    timeouts = 0

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser, open(log_path, 'w', encoding='utf-8') as logf:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.2)
        boot = ser.read(ser.in_waiting or 1).decode(errors='ignore')
        if boot:
            print(boot.strip()); logf.write(boot)
        if args.reset_state:
            ser.write(b'RESET\n'); ser.flush(); time.sleep(0.05)
            ack = ser.read(ser.in_waiting or 0).decode(errors='ignore')
            if ack:
                print(ack.strip()); logf.write(ack)
        # Prime state
        for idx in range(start, cap_start):
            row = df.loc[idx, cols].astype('float32').tolist()
            line = ' '.join(f'{v:.6f}' for v in row) + '\n'
            ser.write(line.encode('ascii')); ser.flush()
            if args.delay > 0:
                time.sleep(args.delay)
            if (idx - start + 1) % 500 == 0:
                print(f'Primed {idx - start + 1}/{cap_start - start} samples...')
        # Drain leftover bytes
        drain_until = time.time() + 1.0
        while time.time() < drain_until:
            if ser.in_waiting:
                _ = ser.read(ser.in_waiting)
            else:
                time.sleep(0.02)

        t0 = time.time()
        for i, idx in enumerate(range(cap_start, end)):
            row = df.loc[idx, cols].astype('float32').tolist()
            payload = ' '.join(f'{v:.6f}' for v in row) + '\n'
            ser.write(payload.encode('ascii')); ser.flush()
            if args.delay > 0:
                time.sleep(args.delay)
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
                if 'SOH' in text.upper():
                    toks = text.replace(':', ' ').split()
                    for tok in toks:
                        try:
                            pred = float(tok)
                            break
                        except Exception:
                            continue
                    if pred is not None:
                        break
            if pred is None:
                timeouts += 1
            else:
                stm_preds[i] = pred
            logf.write(f'[{idx}] {" ".join(map(str, row))} | {" | ".join(parts)}\n')
            if (i + 1) % 500 == 0:
                rate = (i + 1) / max(1e-6, (time.time() - t0))
                print(f'{i + 1}/{end - cap_start} samples | {rate:.1f} samples/s | timeouts={timeouts}')

    preds_proc = stm_preds.copy()
    if args.strict_filter:
        post_max_abs = args.post_max_abs if args.post_max_abs is not None else 2e-5
        post_max_rel = args.post_max_rel if args.post_max_rel is not None else 5e-4
        alpha = args.post_ema_alpha if args.post_ema_alpha is not None else 0.005
        out = []
        last = None
        for p in preds_proc:
            v = float(p)
            if np.isnan(v):
                out.append(np.nan)
                continue
            if last is not None and np.isfinite(last):
                cap = min(post_max_abs, abs(last) * post_max_rel)
                delta = v - last
                if abs(delta) > cap:
                    v = last + np.sign(delta) * cap
                v = alpha * v + (1.0 - alpha) * last
            out.append(v)
            last = v
        preds_proc = np.array(out, dtype=np.float32)

    valid = np.isfinite(preds_proc)
    diffs = np.abs(preds_proc[valid] - y_true[valid]) if valid.any() else np.array([], dtype=np.float32)
    mae = float(np.mean(np.abs(preds_proc[valid] - y_true[valid]))) if valid.any() else float('nan')
    rmse = float(np.sqrt(np.mean((preds_proc[valid] - y_true[valid]) ** 2))) if valid.any() else float('nan')
    mxd = float(np.max(diffs)) if valid.any() else float('nan')

    metrics = {
        'N': int(end - cap_start),
        'valid': int(valid.sum()),
        'timeouts': int(timeouts),
        'MAE_vs_GT': mae,
        'RMSE_vs_GT': rmse,
        'MAX_vs_GT': mxd,
        'parquet': str(Path(args.parquet).resolve()),
        'features': cols,
        'start': start,
        'n': int(end - start),
        'prime': prime_len,
        'strict_filter': bool(args.strict_filter),
        'post_max_abs': args.post_max_abs,
        'post_max_rel': args.post_max_rel,
        'post_ema_alpha': args.post_ema_alpha,
    }
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label='SOH true', alpha=0.6)
    plt.plot(preds_proc, label='STM32 SOH PRUNED', alpha=0.8)
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'overlay_full.png', dpi=150); plt.close()

    firstN = min(500, len(preds_proc))
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:firstN], label='SOH true', alpha=0.6)
    plt.plot(preds_proc[:firstN], label='STM32 SOH PRUNED', alpha=0.8)
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'overlay_firstN.png', dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    if diffs.size > 0:
        plt.hist(diffs, bins=80, color='tab:purple', edgecolor='black', alpha=0.85)
    plt.title('Abs diff STM32 vs GT'); plt.tight_layout()
    plt.savefig(out_dir / 'diff_hist.png', dpi=150); plt.close()

    np.savez_compressed(out_dir / 'arrays.npz', y_true=y_true, stm32=preds_proc, raw_preds=stm_preds)
    print(f"[done] Saved STM32 SOH PRUNED run to: {out_dir}")

    # =============== Optional: PC seq2many baseline on the same slice ===============
    if args.pc_seq2many:
        try:
            import re
            import torch
            import torch.nn as nn
        except Exception as e:
            print(f"[warn] PC seq2many skipped (PyTorch not available): {e}")
            return

        # 1) Build model from pruned checkpoint by inferring shapes
        ckpt_path = Path(args.ckpt)
        try:
            state = torch.load(str(ckpt_path), map_location='cpu')
        except Exception as e:
            print(f"[warn] PC seq2many skipped (cannot load ckpt): {e}")
            return
        sd = state.get('model_state_dict') if isinstance(state, dict) else None
        if sd is None:
            sd = state if isinstance(state, dict) else None
        if sd is None:
            print("[warn] PC seq2many skipped (invalid checkpoint format)")
            return
        try:
            w_ih = sd['lstm.weight_ih_l0'].cpu().numpy()  # [4H, In]
            w_hh = sd['lstm.weight_hh_l0'].cpu().numpy()  # [4H, H]
            b_ih = sd['lstm.bias_ih_l0'].cpu().numpy()    # [4H]
            b_hh = sd['lstm.bias_hh_l0'].cpu().numpy()    # [4H]
            bias = (b_ih + b_hh).astype(np.float32)
            H4, In = w_ih.shape
            H = H4 // 4
            mlp0_w = sd['mlp.0.weight'].cpu().numpy()  # [M, H]
            mlp0_b = sd['mlp.0.bias'].cpu().numpy()    # [M]
            mlp3_w = sd['mlp.3.weight'].cpu().numpy()  # [1, M]
            mlp3_b = sd['mlp.3.bias'].cpu().numpy()    # [1]
            M = mlp0_w.shape[0]
        except Exception as e:
            print(f"[warn] PC seq2many skipped (missing expected weights): {e}")
            return

        class LSTMMLP_SOH_Min(nn.Module):
            def __init__(self, in_features: int, hidden_size: int, mlp_hidden: int):
                super().__init__()
                self.lstm = nn.LSTM(in_features, hidden_size, num_layers=1, batch_first=True)
                self.mlp0 = nn.Linear(hidden_size, mlp_hidden)
                self.mlp1 = nn.Linear(mlp_hidden, 1)
                self.relu = nn.ReLU()
            def forward(self, x, state=None, return_state=False, all_steps=False):
                out, new_state = self.lstm(x, state)
                if all_steps:
                    hs = out.reshape(-1, out.size(-1))
                    y = self.mlp1(self.relu(self.mlp0(hs))).reshape(x.size(0), x.size(1), 1).squeeze(-1)
                else:
                    last = out[:, -1, :]
                    y = self.mlp1(self.relu(self.mlp0(last))).squeeze(-1)
                return (y, new_state) if return_state else y

        device = torch.device('cpu')
        model_pc = LSTMMLP_SOH_Min(In, H, M).to(device)
        # load weights
        with torch.no_grad():
            model_pc.lstm.weight_ih_l0.copy_(torch.from_numpy(w_ih.astype(np.float32)))
            model_pc.lstm.weight_hh_l0.copy_(torch.from_numpy(w_hh.astype(np.float32)))
            model_pc.lstm.bias_ih_l0.copy_(torch.from_numpy(bias))
            model_pc.lstm.bias_hh_l0.zero_()
            model_pc.mlp0.weight.copy_(torch.from_numpy(mlp0_w.astype(np.float32)))
            model_pc.mlp0.bias.copy_(torch.from_numpy(mlp0_b.astype(np.float32)))
            model_pc.mlp1.weight.copy_(torch.from_numpy(mlp3_w.astype(np.float32)))
            model_pc.mlp1.bias.copy_(torch.from_numpy(mlp3_b.astype(np.float32)))
        model_pc.eval()

        # 2) Build a scaler from the MCU header (exact same transform)
        try:
            hdr = (REPO_ROOT / 'STM32' / 'workspace_1.17.0' / 'AI_Project_LSTM_SOH_pruned' / 'Core' / 'Inc' / 'scaler_params_soh.h').read_text(encoding='utf-8', errors='ignore')
            def parse_arr(name):
                m = re.search(rf"{name}\[.*?\]\s*=\s*\{{([^\}}]+)\}}", hdr, flags=re.S)
                if not m:
                    raise RuntimeError(f'missing {name}')
                nums = []
                for tok in m.group(1).replace('\n',' ').split(','):
                    tok = tok.strip().rstrip('f')
                    if not tok: continue
                    try:
                        nums.append(float(tok))
                    except: pass
                return np.array(nums, dtype=np.float32)
            cen = parse_arr('SCALER_SOH_CENTER')
            scl = parse_arr('SCALER_SOH_SCALE')
            if cen.size != In or scl.size != In:
                print(f"[warn] scaler dims mismatch: center={cen.size} scale={scl.size} In={In}")
        except Exception as e:
            print(f"[warn] PC seq2many skipped (cannot parse scaler header): {e}")
            return

        # 3) Prepare data slice [start:end) and run seq2many blocks with the same prime (= chunk-1)
        df_slice = df.iloc[start:end]
        X = df_slice[cols].to_numpy(dtype=np.float32)
        # scale; if checkpoint expects more inputs than provided columns, pad missing dims with their centers
        In = int(In)
        if X.shape[1] < In:
            pad_dims = In - X.shape[1]
            pad = np.tile(cen[X.shape[1]:X.shape[1]+pad_dims], (X.shape[0], 1)) if cen.size >= In else np.tile(cen[-pad_dims:], (X.shape[0], 1))
            X_full = np.concatenate([X, pad], axis=1)
        else:
            X_full = X[:, :In]
        Xs = (X_full - cen) / scl
        # chunk inferred from prime
        chunk = int(args.prime) + 1
        total = Xs.shape[0]
        if total < chunk:
            print('[warn] PC seq2many skipped (slice shorter than chunk)')
            return
        # start inside the slice
        sidx = chunk - 1
        # block loop
        i = sidx
        preds_pc = []
        state = None
        while i < total:
            j = min(total, i + int(args.pc_block))
            blk = torch.from_numpy(Xs[i:j]).unsqueeze(0).to(device)
            with torch.no_grad():
                out_seq, state = model_pc(blk, state=state, return_state=True, all_steps=True)
            preds_pc.extend(out_seq.squeeze(0).cpu().numpy().tolist())
            i = j
        preds_pc = np.array(preds_pc, dtype=np.float32)
        y_true_pc = y_true  # same target slice length

        # optional same filter as STM32 post-process
        preds_pc_proc = preds_pc.copy()
        if args.strict_filter:
            post_max_abs = args.post_max_abs if args.post_max_abs is not None else 2e-5
            post_max_rel = args.post_max_rel if args.post_max_rel is not None else 5e-4
            alpha = args.post_ema_alpha if args.post_ema_alpha is not None else 0.005
            out = []
            last = None
            for p in preds_pc_proc:
                v = float(p)
                if last is not None and np.isfinite(last):
                    cap = min(post_max_abs, abs(last) * post_max_rel)
                    delta = v - last
                    if abs(delta) > cap:
                        v = last + np.sign(delta) * cap
                    v = alpha * v + (1.0 - alpha) * last
                out.append(v)
                last = v
            preds_pc_proc = np.array(out, dtype=np.float32)

        # Align lengths (defensive)
        L = min(len(preds_pc_proc), len(preds_proc))
        a_pc = preds_pc_proc[:L]
        a_stm = preds_proc[:L]
        t = y_true_pc[:L]

        # Compute extra metrics
        mae_pc = float(np.mean(np.abs(a_pc - t))) if L else float('nan')
        rmse_pc = float(np.sqrt(np.mean((a_pc - t)**2))) if L else float('nan')
        mae_stm_pc = float(np.mean(np.abs(a_stm - a_pc))) if L else float('nan')
        rmse_stm_pc = float(np.sqrt(np.mean((a_stm - a_pc)**2))) if L else float('nan')
        max_stm_pc = float(np.max(np.abs(a_stm - a_pc))) if L else float('nan')

        # Append to metrics.json
        try:
            with open(out_dir / 'metrics.json', 'r', encoding='utf-8') as f:
                met = json.load(f)
        except Exception:
            met = {}
        met.update({
            'PC_seq2many': True,
            'PC_chunk': int(chunk),
            'PC_block': int(args.pc_block),
            'MAE_PC_vs_GT': mae_pc,
            'RMSE_PC_vs_GT': rmse_pc,
            'MAE_STM_vs_PC': mae_stm_pc,
            'RMSE_STM_vs_PC': rmse_stm_pc,
            'MAX_STM_vs_PC': max_stm_pc,
        })
        with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(met, f, indent=2)

        # Plots: PC vs GT, STM vs PC, and combined overlay
        Nplot = min(500, L)
        plt.figure(figsize=(12, 4))
        plt.plot(t, label='SOH true', alpha=0.6)
        plt.plot(a_pc, label='PC seq2many (pruned)', alpha=0.8)
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'overlay_pc_vs_gt_full.png', dpi=150); plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(t[:Nplot], label='SOH true', alpha=0.6)
        plt.plot(a_pc[:Nplot], label='PC seq2many (pruned)', alpha=0.8)
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'overlay_pc_vs_gt_firstN.png', dpi=150); plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(a_stm[:Nplot], label='STM32', alpha=0.8)
        plt.plot(a_pc[:Nplot], label='PC seq2many', alpha=0.8)
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'overlay_stm_vs_pc_firstN.png', dpi=150); plt.close()

        plt.figure(figsize=(6, 4))
        if L:
            plt.hist(np.abs(a_stm - a_pc), bins=80, color='tab:green', edgecolor='black', alpha=0.85)
        plt.title('Abs diff STM32 vs PC seq2many'); plt.tight_layout()
        plt.savefig(out_dir / 'diff_hist_stm_vs_pc.png', dpi=150); plt.close()

        # Combined overlay first N: GT, PC, STM32
        plt.figure(figsize=(12, 4))
        plt.plot(t[:Nplot], label='SOH true', alpha=0.7)
        plt.plot(a_pc[:Nplot], label='PC seq2many', alpha=0.8)
        plt.plot(a_stm[:Nplot], label='STM32', alpha=0.8)
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'overlay_combined_firstN.png', dpi=150); plt.close()


if __name__ == '__main__':
    main()
