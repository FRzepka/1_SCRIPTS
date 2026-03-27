#!/usr/bin/env python3
"""
Compare SOH seq2many (PyTorch) vs step-by-step C implementation (compiled LSTM/MLP or Python C-sim).

Outputs: DL_Models/.../6_test/Python/base_SOH/c_code/C_VS_SEQ2MANY_<timestamp>/
"""
import argparse
import os
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_CKPT = PROJECT_ROOT / '2_models' / 'base' / '2.1.0.0_soh_epoch0120_rmse0.03359.pt'
SOH_C_DIR = PROJECT_ROOT / '2_models' / 'base' / 'SOH_c_implementation'


def normalize_name(name: str) -> str:
    return (name.replace('��', 'o').replace('���', '')
                .replace('��', 'a').replace('�"', 'A')
                .replace('��', 'o').replace('�-', 'O')
                .replace('Ǭ', 'u').replace('�o', 'U')
                .replace('�Y', 'ss').replace("'", "").lower())


def map_features(df: pd.DataFrame, feats: List[str]) -> List[str]:
    mapped = []
    cols = list(df.columns)
    cols_norm = [normalize_name(c) for c in cols]
    for f in feats:
        if f in df.columns:
            mapped.append(f)
            continue
        fn = normalize_name(f)
        idx = None
        for i, cn in enumerate(cols_norm):
            if cn == fn:
                idx = i; break
        if idx is None:
            for i, cn in enumerate(cols_norm):
                if fn in cn:
                    idx = i; break
        if idx is None:
            raise KeyError(f"Feature '{f}' not found in dataframe columns")
        mapped.append(cols[idx])
    return mapped


def strict_filter(series: np.ndarray, rel: Optional[float], abs_: Optional[float], alpha: Optional[float]) -> np.ndarray:
    if series.size == 0:
        return series
    out = series.copy()
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
    return out

# -------------------------
# Online post-processing (filter + calibration)
# -------------------------

class OnlinePost:
    def __init__(self,
                 rel: Optional[float] = None,
                 abs_: Optional[float] = None,
                 ema_alpha: Optional[float] = None,
                 passes: int = 1,
                 calib_mode: str = 'off',
                 calib_kind: str = 'scale',
                 calib_anchor: str = 'by_pred',
                 calib_apply: str = 'after_filter',
                 calib_decay_tau: int = 5000):
        self.rel = rel
        self.abs = abs_
        self.alpha = ema_alpha if (ema_alpha is not None and 0 < ema_alpha < 1) else None
        self.passes = max(1, int(passes))
        self.prev_vals = [None] * self.passes
        self.ema = None
        self.calib_mode = calib_mode
        self.calib_kind = calib_kind
        self.calib_anchor = calib_anchor
        self.calib_apply = calib_apply
        self.calib_decay_tau = max(1, int(calib_decay_tau))
        self._calib_ready = (calib_mode == 'off')
        self._gain = 1.0
        self._delta = 0.0
        self._t = 0  # sample index since start

    def set_base_true(self, y_true0: Optional[float]):
        if self.calib_mode != 'off' and self.calib_anchor == 'by_true' and y_true0 is not None:
            base = float(y_true0)
            if self.calib_kind == 'scale':
                self._gain = (1.0 / base) if abs(base) > 1e-12 else 1.0
                self._delta = 0.0
            else:
                self._gain = 1.0
                self._delta = 1.0 - base
            self._calib_ready = True

    def _apply_filter_once(self, y: float, idx: int) -> float:
        prev = self.prev_vals[idx]
        if prev is None:
            self.prev_vals[idx] = float(y)
            return float(y)
        diff = float(y) - prev
        if self.abs is not None:
            a = float(self.abs)
            if a >= 0:
                if diff > a: diff = a
                elif diff < -a: diff = -a
        if self.rel is not None:
            cap = max(abs(prev) * float(self.rel), 1e-9)
            if diff > cap: diff = cap
            elif diff < -cap: diff = -cap
        out = prev + diff
        self.prev_vals[idx] = out
        return out

    def update(self, y_pred: float, y_true: Optional[float] = None) -> float:
        y = float(y_pred)
        # Establish calibration anchor if needed
        if self.calib_mode != 'off' and not self._calib_ready:
            if self.calib_anchor == 'by_true' and y_true is not None:
                base = float(y_true)
            elif self.calib_anchor == 'by_pred':
                base = float(y)
            else:
                base = None
            if base is not None:
                if self.calib_mode == 'pin':
                    # pin does not need gain/delta beyond the first sample
                    self._gain = 1.0
                    self._delta = 0.0
                elif self.calib_kind == 'scale':
                    self._gain = (1.0 / base) if abs(base) > 1e-12 else 1.0
                    self._delta = 0.0
                else:
                    self._gain = 1.0
                    self._delta = 1.0 - base
                self._calib_ready = True

        # Helper to apply calibration with mode and time index
        def _apply_calib(val: float, t: int) -> float:
            if self.calib_mode == 'off':
                return val
            if self.calib_mode == 'pin':
                # Only force first emitted sample to 1.0
                if t == 0:
                    return 1.0
                return val
            if self.calib_mode == 'decay':
                w = np.exp(-float(t) / float(self.calib_decay_tau))
                if self.calib_kind == 'scale':
                    g = self._gain
                    return float(val * (1.0 + w * (g - 1.0)))
                else:
                    return float(val + w * self._delta)
            # uniform
            return float(val * self._gain + self._delta)

        # Optionally apply calibration before filtering
        if self.calib_apply == 'before_filter':
            y = _apply_calib(y, self._t)

        # Apply cascaded filter passes online
        for p in range(self.passes):
            y = self._apply_filter_once(y, p)

        # Optional EMA smoothing (online)
        if self.alpha is not None:
            if self.ema is None:
                self.ema = y
            else:
                self.ema = self.alpha * y + (1.0 - self.alpha) * self.ema
            y = self.ema

        # Or apply calibration after filtering
        if self.calib_apply == 'after_filter':
            y = _apply_calib(y, self._t)

        self._t += 1
        return float(y)


# -------------------------
# Minimal local seq2many stack (self-contained)
# -------------------------
import torch
import torch.nn as nn


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

    def forward(self, x, state=None, return_state: bool = False, all_steps: bool = False):
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


def load_checkpoint_local(path: str, device: torch.device):
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        sd = state
        cfg = {}
        features = None
        chunk = None
        scaler_hint = None
    else:
        sd = state.get('model_state_dict') or state.get('state_dict') or state
        cfg = state.get('config') or state.get('cfg') or {}
        features = state.get('features') or state.get('feature_list')
        chunk = state.get('chunk') or state.get('window') or state.get('seq_len')
        scaler_hint = state.get('scaler_path')
        if scaler_hint is None and isinstance(cfg, dict):
            out_root = (cfg.get('paths', {}) or {}).get('out_root')
            if out_root:
                candidate = Path(out_root) / 'scaler_robust.joblib'
                if candidate.exists():
                    scaler_hint = str(candidate)
    in_features = int(sd['lstm.weight_ih_l0'].shape[1])
    hidden_size = int(sd['lstm.weight_hh_l0'].shape[1])
    mlp_hidden = int(sd['mlp.0.weight'].shape[0])
    num_layers = int(cfg.get('model', {}).get('num_layers', 1)) if isinstance(cfg, dict) else 1
    dropout = float(cfg.get('model', {}).get('dropout', 0.05)) if isinstance(cfg, dict) else 0.05
    model = LSTMMLP_SOHTarget(in_features, hidden_size, mlp_hidden, num_layers=num_layers, dropout=dropout).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    if not features:
        features = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'EFC', 'Q_c']
    if chunk is None:
        chunk = 2048
    return model, list(features), int(chunk), cfg, scaler_hint, sd


def seq2many_with_state(df: pd.DataFrame,
                        cols: List[str],
                        scaler: RobustScaler,
                        model: nn.Module,
                        chunk: int,
                        device: torch.device,
                        block_len: int = 8192,
                        max_preds: int = -1,
                        progress: bool = True,
                        strict_filter: bool = False,
                        post_max_rel: Optional[float] = None,
                        post_max_abs: Optional[float] = None,
                        post_ema_alpha: Optional[float] = None,
                        filter_passes: int = 1,
                        calib_mode: str = 'off',
                        calib_kind: str = 'scale',
                        calib_anchor: str = 'by_pred',
                        calib_apply: str = 'after_filter',
                        calib_decay_tau: int = 5000):
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + (['SOH'] if 'SOH' in df.columns else []))
    X = clean[cols].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    total = len(Xs)
    preds = []
    state = None
    start = 0
    # Online post-processor for streaming
    post = None
    with torch.no_grad():
        if total >= chunk:
            prime = torch.from_numpy(Xs[:chunk-1]).unsqueeze(0).to(device)
            _, state = model(prime, state=None, return_state=True)
            start = chunk - 1
        target = (total - start) if max_preds <= 0 else min(total - start, max_preds)
        it = tqdm(total=target, unit='step', desc='seq2many', dynamic_ncols=True) if progress else None
        # Initialize online post-processing after priming
        if strict_filter or (calib_mode != 'off') or (post_ema_alpha is not None):
            post = OnlinePost(rel=post_max_rel, abs_=post_max_abs, ema_alpha=post_ema_alpha,
                              passes=filter_passes, calib_mode=calib_mode, calib_kind=calib_kind,
                              calib_anchor=calib_anchor, calib_apply=calib_apply, calib_decay_tau=calib_decay_tau)
            # if by_true, set base from the first true value after priming
            if calib_mode != 'off' and calib_anchor == 'by_true' and 'SOH' in clean.columns and start < len(clean):
                post.set_base_true(float(clean['SOH'].iloc[start]))
        i = start
        while i < total and len(preds) < target:
            end = min(total, i + block_len)
            block = torch.from_numpy(Xs[i:end]).unsqueeze(0).to(device)
            out_seq, state = model(block, state=state, return_state=True, all_steps=True)
            block_preds = out_seq.squeeze(0).detach().cpu().numpy().tolist()
            remaining = target - len(preds)
            take = int(min(len(block_preds), remaining))
            # stream each prediction through online post
            if post is None:
                preds.extend(block_preds[:take])
            else:
                off = 0
                for k in range(take):
                    yk = float(block_preds[k])
                    preds.append(post.update(yk))
            if it: it.update(take)
            i = end
        if it: it.close()
    y_true = None
    if 'SOH' in clean.columns:
        y = clean['SOH'].to_numpy(dtype=np.float32)
        y_true = y[start:start + len(preds)]
    preds = np.asarray(preds, dtype=np.float32)
    return preds, y_true, {'start_index': start, 'processed': int(len(preds))}


def strict_filter_fn(series: np.ndarray, rel: Optional[float], abs_: Optional[float], alpha: Optional[float]) -> np.ndarray:
    return strict_filter(series, rel, abs_, alpha)


def compile_c_tester(work_dir: Path) -> Optional[Path]:
    tester = work_dir / ('test_soh_c.exe' if sys.platform == 'win32' else 'test_soh_c')
    main_c = work_dir / 'test_main_soh.c'
    main_c.write_text(r'''
#include <stdio.h>
#include "lstm_model_soh.h"
#include "model_weights_soh.h"
#include "scaler_params_soh.h"

int main(){
  float in[INPUT_SIZE];
  float in_scaled[INPUT_SIZE];
  LSTMModelSOH model; lstm_model_soh_init(&model);
  while (1){
    int ok = 0;
    if (INPUT_SIZE==6){
      ok = scanf("%f %f %f %f %f %f", &in[0],&in[1],&in[2],&in[3],&in[4],&in[5]);
      if (ok!=6) break;
    } else {
      for (int i=0;i<INPUT_SIZE;i++){
        if (scanf("%f", &in[i])!=1){ ok=0; break; }
        ok++;
      }
      if (ok!=INPUT_SIZE) break;
    }
    scaler_soh_transform(in, in_scaled);
    float y=0.0f; lstm_model_soh_inference(&model, in_scaled, &y);
    printf("SOH %.6f\n", y);
    fflush(stdout);
  }
  return 0;
}
''')
    inc = str(SOH_C_DIR)
    cmd = ['gcc', '-O2', '-I', inc, str(main_c), str(SOH_C_DIR / 'lstm_model_soh.c'), '-o', str(tester), '-lm']
    for compiler in (cmd,
                     ['clang', '-O2', '-I', inc, str(main_c), str(SOH_C_DIR / 'lstm_model_soh.c'), '-o', str(tester)],
                     ['cl', '/O2', f'/I{inc}', str(main_c), str(SOH_C_DIR / 'lstm_model_soh.c'), '/Fe:'+str(tester)]):
        try:
            if compiler[0] == 'cl':
                subprocess.run(compiler, check=True, capture_output=False)
            else:
                subprocess.run(compiler, check=True, capture_output=True)
            return tester
        except Exception:
            continue
    return None


def run_c_binary(exe: Path, X_raw: np.ndarray) -> np.ndarray:
    lines = [' '.join(f'{v:.8f}' for v in row) for row in X_raw]
    payload = '\n'.join(lines) + '\n'
    res = subprocess.run([str(exe)], input=payload, text=True, capture_output=True, check=True)
    preds = []
    for line in res.stdout.splitlines():
        if 'SOH' in line.upper():
            tokens = line.replace(':', ' ').split()
            for tok in tokens:
                try:
                    preds.append(float(tok))
                    break
                except ValueError:
                    continue
    return np.array(preds, dtype=np.float32)


def run_c_binary_streaming(exe: Path, X_raw: np.ndarray, post: Optional[OnlinePost] = None) -> np.ndarray:
    """Feed the C binary step-by-step and read predictions line-by-line,
    applying online post-processing per sample.
    """
    proc = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    preds: List[float] = []
    try:
        for row in X_raw:
            line = ' '.join(f'{v:.8f}' for v in row) + '\n'
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(line)
            proc.stdin.flush()
            out_line = proc.stdout.readline()
            if not out_line:
                break
            val = None
            tokens = out_line.replace(':', ' ').split()
            for tok in tokens:
                try:
                    val = float(tok)
                    break
                except ValueError:
                    continue
            if val is None:
                continue
            if post is not None:
                val = post.update(val)
            preds.append(float(val))
    finally:
        # Close stdin to signal EOF
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
        # Consume remaining output and ensure process terminates
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    return np.array(preds, dtype=np.float32)


def run_csim(sd: dict, Xs: np.ndarray, show_progress: bool = True, post: Optional[OnlinePost] = None) -> np.ndarray:
    W_ih = sd['lstm.weight_ih_l0'].detach().cpu().numpy().astype(np.float32)
    W_hh = sd['lstm.weight_hh_l0'].detach().cpu().numpy().astype(np.float32)
    b_ih = sd['lstm.bias_ih_l0'].detach().cpu().numpy().astype(np.float32)
    b_hh = sd['lstm.bias_hh_l0'].detach().cpu().numpy().astype(np.float32)
    B = b_ih + b_hh
    mlp0_w = sd['mlp.0.weight'].detach().cpu().numpy().astype(np.float32)
    mlp0_b = sd['mlp.0.bias'].detach().cpu().numpy().astype(np.float32)
    mlp1_w = sd['mlp.3.weight'].detach().cpu().numpy().astype(np.float32)
    mlp1_b = sd['mlp.3.bias'].detach().cpu().numpy().astype(np.float32)
    H = W_hh.shape[1]
    h = np.zeros((H,), dtype=np.float32)
    c = np.zeros((H,), dtype=np.float32)
    preds = []
    iterator = tqdm(range(Xs.shape[0]), desc='c-sim', dynamic_ncols=True) if show_progress else range(Xs.shape[0])
    for t in iterator:
        x = Xs[t]
        gates = x @ W_ih.T + h @ W_hh.T + B
        i = 1.0 / (1.0 + np.exp(-gates[:H]))
        f = 1.0 / (1.0 + np.exp(-gates[H:2*H]))
        g = np.tanh(gates[2*H:3*H])
        o = 1.0 / (1.0 + np.exp(-gates[3*H:4*H]))
        c = f * c + i * g
        h = o * np.tanh(c)
        z = np.maximum(0.0, h @ mlp0_w.T + mlp0_b)
        y = float((z @ mlp1_w.T + mlp1_b).reshape(()))
        if post is not None:
            y = post.update(y)
        preds.append(y)
    return np.array(preds, dtype=np.float32)


def parse_args():
    ap = argparse.ArgumentParser(description='Compare SOH seq2many vs C step-by-step path')
    ap.add_argument('--ckpt', default=str(DEFAULT_CKPT))
    ap.add_argument('--parquet', required=True, help='Path or name of parquet. If not an existing path, will be resolved relative to --data-root')
    ap.add_argument('--num-samples', type=int, default=20000)
    ap.add_argument('--prime', action='store_true')
    ap.add_argument('--strict-filter', action='store_true')
    ap.add_argument('--post-max-rel', type=float, default=None)
    ap.add_argument('--post-max-abs', type=float, default=None)
    ap.add_argument('--post-ema-alpha', type=float, default=None)
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--scaler', default='', help='Optional path to RobustScaler.joblib to force use (overrides checkpoint hint/refit)')
    ap.add_argument('--no-compile', action='store_true', help='Skip native compile; always use Python csim')
    ap.add_argument('--data-root', default='${DATA_ROOT:-/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE}', help='Base folder for parquet if --parquet is not an absolute path')
    ap.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars')
    ap.add_argument('--plot-max', type=int, default=100000, help='Max points to plot in overlay_full (<=0 for all)')
    # online filter options (applied during generation, not post)
    ap.add_argument('--filter-passes', type=int, default=1, help='Repeat strict filter online N times per sample')
    ap.add_argument('--calib-mode', choices=['off','pin','decay','uniform'], default='pin', help='online calibration mode')
    ap.add_argument('--calib-kind', choices=['scale','shift'], default='scale', help='scale: multiply by 1/y0; shift: add (1-y0)')
    ap.add_argument('--calib-anchor', choices=['by_pred','by_true'], default='by_pred', help='use first prediction or first GT (if available) as anchor')
    ap.add_argument('--calib-apply', choices=['after_filter','before_filter'], default='after_filter', help='apply calibration before or after filtering')
    ap.add_argument('--calib-decay-tau', type=int, default=5000, help='tau for decay mode (samples)')
    return ap.parse_args()


def _resolve_env_default(s: str) -> str:
    if not isinstance(s, str):
        return s
    import re
    m = re.match(r'^\$\{([^}:]+)(?::-(.+))?\}$', s)
    if m:
        var = m.group(1)
        default = m.group(2) if m.group(2) is not None else ''
        return os.environ.get(var, default)
    return os.path.expandvars(s)


def _resolve_parquet_path(arg: str, data_root: str) -> str:
    p = Path(arg)
    if p.exists():
        return str(p)
    # If no path separators, try resolving relative to data_root
    if os.sep not in arg:
        candidates = []
        # direct filename
        candidates.append(Path(data_root) / arg)
        # df_FE_*.parquet
        base = arg
        if base.lower().endswith('.parquet'):
            base = base[:-8]
        # normalize token, allow C07 or 07
        token = base
        token = token.replace('df_fe_', '').replace('DF_FE_', '')
        if not token.upper().startswith('C'):
            token = 'C' + token
        candidates.append(Path(data_root) / f'df_FE_{token}.parquet')
        for c in candidates:
            if c.exists():
                return str(c)
    # Fallback: join with data_root
    fallback = Path(data_root) / arg
    return str(fallback)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, feats, chunk, _cfg, scaler_hint, sd = load_checkpoint_local(args.ckpt, device)
    chunk = int(chunk)

    data_root = _resolve_env_default(args.data_root)
    parquet_path = _resolve_parquet_path(args.parquet, data_root)
    print(f"[config] data_root={data_root} | parquet={parquet_path}")
    df_all = pd.read_parquet(parquet_path)
    df_slice = df_all.iloc[:args.num_samples].reset_index(drop=True)
    cols = map_features(df_slice, feats)
    if df_slice.empty:
        raise ValueError('Parquet slice empty; increase data or adjust --num-samples')

    scaler = None
    # prefer CLI scaler if provided
    if args.scaler:
        try:
            scaler = joblib_load(args.scaler)
        except Exception:
            scaler = None
    if scaler is None and scaler_hint and Path(scaler_hint).exists():
        try:
            scaler = joblib_load(scaler_hint)
            scaler_used = scaler_hint
        except Exception:
            scaler = None
            scaler_used = None
    if scaler is None or getattr(scaler, 'n_features_in_', len(cols)) != len(cols):
        scaler = RobustScaler()
        scaler.fit(df_slice[cols].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32))
        scaler_used = 'refit_on_slice'

    seq_preds, seq_true, eff = seq2many_with_state(
        df_slice,
        list(cols),
        scaler,
        model,
        chunk,
        device,
        block_len=8192,
        max_preds=args.num_samples,
        progress=not args.no_progress,
        strict_filter=args.strict_filter,
        post_max_rel=args.post_max_rel,
        post_max_abs=args.post_max_abs,
        post_ema_alpha=args.post_ema_alpha,
        filter_passes=int(args.filter_passes),
        calib_mode=args.calib_mode,
        calib_kind=args.calib_kind,
        calib_anchor=args.calib_anchor,
        calib_apply=args.calib_apply,
        calib_decay_tau=int(args.calib_decay_tau),
    )

    work_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / '6_test' / 'Python' / 'base_SOH' / 'c_code' / f'C_VS_SEQ2MANY_{time.strftime("%Y%m%d_%H%M%S")}')
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build a clean dataframe exactly like seq2many uses (ensures alignment)
    df_clean = df_slice.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + (['SOH'] if 'SOH' in df_slice.columns else []))
    if df_clean.empty:
        raise ValueError('No valid rows after cleaning/filtering')
    X_raw = df_clean[cols].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X_raw).astype(np.float32)

    if not args.no_compile:
        exe = compile_c_tester(work_dir)
    else:
        exe = None
    # Prepare online post-processor for C path as well
    post = None
    if args.strict_filter or (args.calib_mode != 'off') or (args.post_ema_alpha is not None):
        post = OnlinePost(rel=args.post_max_rel, abs_=args.post_max_abs, ema_alpha=args.post_ema_alpha,
                          passes=int(args.filter_passes), calib_mode=args.calib_mode,
                          calib_kind=args.calib_kind, calib_anchor=args.calib_anchor,
                          calib_apply=args.calib_apply, calib_decay_tau=args.calib_decay_tau)
        if args.calib_mode != 'off' and args.calib_anchor == 'by_true' and 'SOH' in df_slice.columns:
            start_idx = eff.get('start_index', chunk - 1 if chunk else 0)
            if 0 <= start_idx < len(df_slice):
                post.set_base_true(float(df_slice['SOH'].iloc[start_idx]))

    if exe:
        c_preds_full = run_c_binary_streaming(exe, X_raw, post)
        mode = 'C_BINARY'
    else:
        c_preds_full = run_csim(sd, Xs, show_progress=not args.no_progress, post=post)
        mode = 'PY_CSIM'

    prime_skip = eff.get('start_index', chunk - 1 if chunk else 0) if args.prime else 0
    seq_aligned = seq_preds
    c_aligned = c_preds_full[prime_skip:prime_skip + len(seq_aligned)]
    if seq_true is not None:
        y_aligned = seq_true[:len(seq_aligned)]
    else:
        y_full = df_slice['SOH'].to_numpy(dtype=np.float32) if 'SOH' in df_slice.columns else None
        y_aligned = y_full[prime_skip:prime_skip + len(seq_aligned)] if y_full is not None else None

    # Note: filtering already applied online via OnlinePost to both seq and C paths; do not re-apply here

    n = min(len(seq_aligned), len(c_aligned))
    seq_aligned = seq_aligned[:n]
    c_aligned = c_aligned[:n]
    if y_aligned is not None:
        y_aligned = y_aligned[:n]

    diffs = np.abs(seq_aligned - c_aligned)
    mae = float(np.mean(diffs)) if n else float('nan')
    rmse = float(np.sqrt(np.mean(diffs**2))) if n else float('nan')
    maxd = float(np.max(diffs)) if n else float('nan')

    (work_dir / 'metrics.json').write_text(json.dumps({
        'mode': mode,
        'N': int(n),
        'MAE_seq_vs_c': mae,
        'RMSE_seq_vs_c': rmse,
        'MAX_seq_vs_c': maxd,
        'prime_skip': int(prime_skip),
        'strict_filter': bool(args.strict_filter),
        'post_max_rel': args.post_max_rel,
        'post_max_abs': args.post_max_abs,
        'post_ema_alpha': args.post_ema_alpha,
        'filter_passes': int(args.filter_passes),
        'calib_mode': args.calib_mode,
        'calib_kind': args.calib_kind,
        'calib_anchor': args.calib_anchor,
        'checkpoint': args.ckpt,
        'parquet': parquet_path,
        'scaler_used': scaler_used,
    }, indent=2))

    # Optional downsampling for overlay_full to keep image size manageable
    pm = int(args.plot_max) if hasattr(args, 'plot_max') else 100000
    if pm and pm > 0 and n > pm:
        import math as _math
        step = int(_math.ceil(n / float(pm)))
        seq_plot = seq_aligned[::step]
        c_plot = c_aligned[::step]
        y_plot = (y_aligned[::step] if y_aligned is not None else None)
    else:
        seq_plot = seq_aligned
        c_plot = c_aligned
        y_plot = y_aligned

    plt.figure(figsize=(12, 4))
    if y_plot is not None:
        plt.plot(y_plot, label='SOH true', alpha=0.5)
    plt.plot(seq_plot, label='seq2many')
    plt.plot(c_plot, label=mode)
    plt.legend(); plt.tight_layout(); plt.savefig(work_dir / 'overlay_full.png', dpi=150); plt.close()

    firstN = min(500, n)
    plt.figure(figsize=(12, 4))
    if y_aligned is not None:
        plt.plot(y_aligned[:firstN], label='SOH true', alpha=0.5)
    plt.plot(seq_aligned[:firstN], label='seq2many')
    plt.plot(c_aligned[:firstN], label=mode)
    plt.legend(); plt.tight_layout(); plt.savefig(work_dir / 'overlay_firstN.png', dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    if diffs.size:
        plt.hist(diffs, bins=120, alpha=0.85, color='tab:purple', edgecolor='black')
    plt.title('Abs diff seq2many vs C')
    plt.tight_layout(); plt.savefig(work_dir / 'diff_hist.png', dpi=150); plt.close()

    print(f'[done] Saved comparison to {work_dir}')


if __name__ == '__main__':
    main()
