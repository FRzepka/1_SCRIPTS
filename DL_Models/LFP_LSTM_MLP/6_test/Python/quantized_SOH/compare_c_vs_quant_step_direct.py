#!/usr/bin/env python3
"""
Compare SOH C step-by-step (FP32 weights) vs quantized step-by-step (manual INT8 weights, FP32 activations)
on the same 1D stream (with priming), using a refit RobustScaler on the selected cell.

Outputs under 6_test/Python/quantized_SOH/COMPARE_C_VS_QUANT_STEP_<timestamp>:
- overlay_full.png, overlay_firstN.png, diff_hist.png
- metrics.json (MAE/RMSE between C and QUANT step; optional vs GT if available)
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def normalize_name(name: str) -> str:
    return (name.replace('������', 'o').replace('���������', '')
                .replace('������', 'a').replace('���"', 'A')
                .replace('������', 'o').replace('���-', 'O')
                .replace('��', 'u').replace('���o', 'U')
                .replace('���Y', 'ss').replace("'", "").lower())


def map_features(df: pd.DataFrame, feats):
    cols = list(df.columns)
    cols_norm = [normalize_name(c) for c in cols]
    mapped = []
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
            raise KeyError(f"Feature '{f}' not in dataframe columns")
        mapped.append(cols[idx])
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


def apply_calibration(series: np.ndarray,
                      y_true_first: float | None,
                      mode: str = 'pin',
                      kind: str = 'scale',
                      anchor: str = 'by_pred',
                      apply_where: str = 'after_filter',
                      decay_tau: int = 5000) -> tuple[np.ndarray, dict]:
    info = {'calib_kind': None, 'calib_anchor': None, 'calib_mode': None}
    if series.size == 0 or mode == 'off':
        return series, info
    y = series.astype(np.float32, copy=True)
    eps = 1e-12
    base = None
    if anchor == 'by_true' and y_true_first is not None:
        base = float(y_true_first)
    else:
        base = float(y[0])

    def _apply(yarr: np.ndarray) -> np.ndarray:
        if mode == 'pin':
            yarr = yarr.copy(); yarr[0] = 1.0
            return yarr
        if mode == 'decay':
            T = yarr.shape[0]
            w = np.exp(-np.arange(T, dtype=np.float64) / max(1, int(decay_tau)))
            if kind == 'scale':
                g = (1.0 / base) if abs(base) > eps else 1.0
                return (yarr.astype(np.float64) * (1.0 + w * (g - 1.0))).astype(np.float32)
            else:
                dlt = 1.0 - base
                return (yarr.astype(np.float64) + w * dlt).astype(np.float32)
        # uniform
        if kind == 'scale':
            gain = (1.0 / base) if abs(base) > eps else 1.0
            return (yarr * gain).astype(np.float32)
        else:
            delta = 1.0 - base
            return (yarr + delta).astype(np.float32)

    if apply_where == 'before_filter':
        y = _apply(y)
    # caller will run strict_filter next
    if apply_where == 'after_filter':
        # defer; caller calls again after filtering
        pass
    info = {'calib_kind': kind, 'calib_anchor': anchor, 'calib_mode': mode}
    return y, info


def run_csim(sd: dict, Xs: np.ndarray, chunk: int) -> (np.ndarray, int):
    W_ih = sd['lstm.weight_ih_l0'].numpy().astype(np.float32)
    W_hh = sd['lstm.weight_hh_l0'].numpy().astype(np.float32)
    B = (sd['lstm.bias_ih_l0'] + sd['lstm.bias_hh_l0']).numpy().astype(np.float32)
    mlp0_w = sd['mlp.0.weight'].numpy().astype(np.float32)
    mlp0_b = sd['mlp.0.bias'].numpy().astype(np.float32)
    mlp1_w = sd['mlp.3.weight'].numpy().astype(np.float32)
    mlp1_b = sd['mlp.3.bias'].numpy().astype(np.float32)
    H = W_hh.shape[1]
    h = np.zeros((H,), dtype=np.float32)
    c = np.zeros((H,), dtype=np.float32)
    T = Xs.shape[0]
    start = 0
    if T >= chunk:
        for t in range(chunk-1):
            x = Xs[t]
            gates = x @ W_ih.T + h @ W_hh.T + B
            i = 1.0 / (1.0 + np.exp(-gates[:H]))
            f = 1.0 / (1.0 + np.exp(-gates[H:2*H]))
            g = np.tanh(gates[2*H:3*H])
            o = 1.0 / (1.0 + np.exp(-gates[3*H:4*H]))
            c = f * c + i * g
            h = o * np.tanh(c)
        start = chunk - 1
    preds = []
    from tqdm import tqdm
    iterator = tqdm(range(start, T), desc='c-fp32', dynamic_ncols=True)
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
        preds.append(y)
    return np.array(preds, dtype=np.float64), start


def _row_scale(vals: np.ndarray, mode: str) -> float:
    eps = 1e-12
    a = np.abs(vals)
    if mode == 'max':
        mx = float(np.max(a))
    elif mode == 'p99_9':
        mx = float(np.percentile(a, 99.9))
    elif mode == 'p99_5':
        mx = float(np.percentile(a, 99.5))
    else:
        mx = float(np.max(a))
    return max(mx, eps)


def quantize_per_row(W: np.ndarray, scale_mode: str = 'max', bits: int = 8):
    rows, cols = W.shape
    if bits == 8:
        qmax = 127.0
        dtype = np.int8
    elif bits == 16:
        qmax = 32767.0
        dtype = np.int16
    else:
        raise ValueError('bits must be 8 or 16')
    scales = np.zeros(rows, dtype=np.float32)
    Wq = np.zeros((rows, cols), dtype=dtype)
    for r in range(rows):
        sc = _row_scale(W[r], scale_mode) / qmax
        scales[r] = sc
        Wq[r] = np.clip(np.round(W[r] / sc), -qmax, qmax).astype(dtype)
    return Wq, scales


def build_quant_state(sd: dict, scale_mode: str = 'max', hh_bits: int = 8):
    W_ih = sd['lstm.weight_ih_l0'].cpu().numpy()
    W_hh = sd['lstm.weight_hh_l0'].cpu().numpy()
    B = (sd['lstm.bias_ih_l0'] + sd['lstm.bias_hh_l0']).cpu().numpy()
    # input weights INT8, recurrent optionally INT16
    W_ih_q, S_ih = quantize_per_row(W_ih, scale_mode=scale_mode, bits=8)
    W_hh_q, S_hh = quantize_per_row(W_hh, scale_mode=scale_mode, bits=hh_bits)
    mlp0_w = sd['mlp.0.weight'].cpu().numpy(); mlp0_b = sd['mlp.0.bias'].cpu().numpy()
    mlp1_w = sd['mlp.3.weight'].cpu().numpy(); mlp1_b = sd['mlp.3.bias'].cpu().numpy()
    return {
        'W_ih_q': W_ih_q.astype(W_ih_q.dtype), 'S_ih': S_ih.astype(np.float32),
        'W_hh_q': W_hh_q.astype(W_hh_q.dtype), 'S_hh': S_hh.astype(np.float32),
        'B': B.astype(np.float32), 'mlp0_w': mlp0_w.astype(np.float32), 'mlp0_b': mlp0_b.astype(np.float32),
        'mlp1_w': mlp1_w.astype(np.float32), 'mlp1_b': mlp1_b.astype(np.float32),
        'input_size': int(W_ih.shape[1]), 'hidden_size': int(W_hh.shape[1])
    }


def quant_step(Xs: np.ndarray, q: dict, chunk: int):
    In = q['input_size']
    Xs_use = Xs[:, :In] if Xs.shape[1] >= In else np.concatenate([Xs, np.zeros((Xs.shape[0], In - Xs.shape[1]), dtype=np.float32)], axis=1)
    T = Xs_use.shape[0]; H = q['hidden_size']
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
    from tqdm import tqdm
    iterator = tqdm(range(start, T), desc='c-int8/16', dynamic_ncols=True)
    for t in iterator:
        x = Xs_use[t]
        gates = (x @ q['W_ih_q'].T).astype(np.float32) * q['S_ih'] + (h @ q['W_hh_q'].T).astype(np.float32) * q['S_hh'] + q['B']
        i = 1.0 / (1.0 + np.exp(-gates[:H])); f = 1.0 / (1.0 + np.exp(-gates[H:2*H]))
        g = np.tanh(gates[2*H:3*H]); o = 1.0 / (1.0 + np.exp(-gates[3*H:4*H]))
        c = f * c + i * g; h = o * np.tanh(c)
        z = np.maximum(0.0, h @ q['mlp0_w'].T + q['mlp0_b'])
        y = float((z @ q['mlp1_w'].T + q['mlp1_b']).reshape(()))
        preds.append(y)
    return np.array(preds, dtype=np.float64), start


def overlay(y1, y2, ytrue, out, title, plot_max: int = 100000):
    out.parent.mkdir(parents=True, exist_ok=True)
    if plot_max and plot_max > 0 and len(y1) > plot_max:
        import math as _math
        step = int(_math.ceil(len(y1) / float(plot_max)))
        y1p = y1[::step]; y2p = y2[::step]
        ytp = (ytrue[::step] if ytrue is not None else None)
    else:
        y1p, y2p, ytp = y1, y2, ytrue
    plt.figure(figsize=(12,4))
    if ytp is not None:
        plt.plot(ytp, label='SOH true', alpha=0.6, lw=0.6)
    plt.plot(y1p, label='C_step', lw=0.8)
    plt.plot(y2p, label='QUANT_step', lw=0.8)
    plt.legend(); plt.title(title); plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--num-samples', type=int, default=500000)
    ap.add_argument('--prime', action='store_true')
    ap.add_argument('--strict-filter', action='store_true')
    ap.add_argument('--post-max-rel', type=float, default=None)
    ap.add_argument('--post-max-abs', type=float, default=None)
    ap.add_argument('--post-ema-alpha', type=float, default=None)
    ap.add_argument('--filter-passes', type=int, default=1, help='Repeat strict filter N times')
    ap.add_argument('--out-dir', default='')
    # calibration like python runner
    ap.add_argument('--calib-start-one', action='store_true')
    ap.add_argument('--calib-kind', choices=['scale','shift'], default='scale')
    ap.add_argument('--calib-anchor', choices=['by_pred','by_true'], default='by_pred')
    ap.add_argument('--calib-mode', choices=['uniform','pin','decay','off'], default='pin')
    ap.add_argument('--calib-apply', choices=['before_filter','after_filter'], default='before_filter')
    ap.add_argument('--calib-decay-tau', type=int, default=5000)
    ap.add_argument('--plot-max', type=int, default=100000)
    ap.add_argument('--export-quantized-to', default='', help='Optional dir to save quantized state (.npz) for reuse as C model')
    # quantization options
    ap.add_argument('--quant-scale', choices=['max','p99_9','p99_5'], default='max')
    ap.add_argument('--quant-hh-precision', choices=['int8','int16'], default='int8')
    # optional affine correction of QUANT vs C before filtering/calib
    ap.add_argument('--affine-correct', type=int, default=0, help='If >0, fit yC ~ a*yQ+b on first N samples (after prime) and correct QUANT before filtering/calib')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    feats = ck.get('features') or ck.get('feature_list')
    if not feats:
        raise RuntimeError('Checkpoint missing features')
    chunk = int(ck.get('chunk') or ck.get('window') or ck.get('seq_len') or 2048)

    df_all = pd.read_parquet(args.parquet)
    df_all = df_all.reset_index(drop=True)
    df = df_all.iloc[:args.num_samples].copy()
    cols = map_features(df, feats)
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + (['SOH'] if 'SOH' in df.columns else []))
    X = clean[cols].to_numpy(dtype=np.float32)
    ytrue = clean['SOH'].to_numpy(dtype=np.float32) if 'SOH' in clean.columns else None

    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    c_preds, start = run_csim(sd, Xs, chunk)
    hh_bits = 16 if args.quant_hh_precision == 'int16' else 8
    q = build_quant_state(sd, scale_mode=args.quant_scale, hh_bits=hh_bits)
    if args.export_quantized_to:
        exp = Path(args.export_quantized_to); exp.mkdir(parents=True, exist_ok=True)
        tag_bits = 'int16hh' if hh_bits == 16 else 'int8'
        fname = f'quant_state_soh_{tag_bits}_{args.quant_scale}.npz'
        np.savez(exp / fname, **q)
        # write metadata and a brief README for reproducibility
        meta = {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'ckpt': str(Path(args.ckpt)),
            'features': feats,
            'chunk': int(chunk),
            'quant_scale': args.quant_scale,
            'quant_hh_precision': args.quant_hh_precision,
            'npz_file': fname,
            'notes': 'Manual per-row quantization. W_ih INT8. W_hh INT8 or INT16 depending on quant_hh_precision. Scales S_ih/S_hh are per-row.'
        }
        (exp / 'quant_meta.json').write_text(json.dumps(meta, indent=2))
        (exp / 'README.txt').write_text(
            'Quantized SOH LSTM state (manual per-row)\n\n'
            f'- NPZ: {fname}\n'
            f'- Scale mode: {args.quant_scale}\n'
            f'- Recurrent precision: {args.quant_hh_precision}\n\n'
            'Keys in NPZ:\n'
            '  W_ih_q (int8), S_ih (float32), W_hh_q (int8/int16), S_hh (float32), B (float32),\n'
            '  mlp0_w/mlp0_b (float32), mlp1_w/mlp1_b (float32), input_size, hidden_size.\n'
            'Use compare_c_vs_quant_step_direct.py to validate or export headers.'
        )
    q_preds, start_q = quant_step(Xs, q, chunk)
    # align by prime
    if args.prime:
        s = start
        c_al = c_preds
        q_al = q_preds[:len(c_al)] if len(q_preds) >= len(c_al) else q_preds
        n = min(len(c_al), len(q_al))
        c_al = c_al[:n]; q_al = q_al[:n]
        y_al = None
        if ytrue is not None:
            y_al = ytrue[s:s+n]
    else:
        n = min(len(c_preds), len(q_preds))
        c_al = c_preds[:n]; q_al = q_preds[:n]
        y_al = ytrue[:n] if ytrue is not None else None

    # Optional affine correction (match QUANT to C on early window, then proceed identically)
    if int(args.affine_correct) > 0:
        ncal = min(int(args.affine_correct), len(c_al), len(q_al))
        if ncal > 16:
            A = np.vstack([q_al[:ncal], np.ones(ncal)]).T
            coeff, *_ = np.linalg.lstsq(A, c_al[:ncal], rcond=None)
            a, b = float(coeff[0]), float(coeff[1])
        else:
            a, b = 1.0, 0.0
        q_al = a * q_al + b
        affine_info = {'affine_a': a, 'affine_b': b, 'affine_window': int(ncal)}
    else:
        affine_info = {}

    # Apply calibration and filtering with correct order
    calib_info = {}
    if args.calib_start_one:
        y0 = (float(y_al[0]) if (y_al is not None and len(y_al) > 0 and args.calib_anchor=='by_true') else None)
        c_al, ci = apply_calibration(c_al, y0, mode=args.calib_mode, kind=args.calib_kind,
                                     anchor=args.calib_anchor, apply_where=args.calib_apply,
                                     decay_tau=int(args.calib_decay_tau))
        q_al, _ = apply_calibration(q_al, y0, mode=args.calib_mode, kind=args.calib_kind,
                                    anchor=args.calib_anchor, apply_where=args.calib_apply,
                                    decay_tau=int(args.calib_decay_tau))
        calib_info.update(ci)
    if args.strict_filter:
        # core strict filter + extra passes
        c_al = strict_filter(c_al, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
        q_al = strict_filter(q_al, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
        for _ in range(max(0, int(args.filter_passes) - 1)):
            c_al = strict_filter(c_al, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
            q_al = strict_filter(q_al, args.post_max_rel, args.post_max_abs, args.post_ema_alpha)
    if args.calib_start_one and args.calib_apply == 'after_filter':
        y0 = (float(y_al[0]) if (y_al is not None and len(y_al) > 0 and args.calib_anchor=='by_true') else None)
        c_al, _ = apply_calibration(c_al, y0, mode=args.calib_mode, kind=args.calib_kind,
                                    anchor=args.calib_anchor, apply_where='after_filter',
                                    decay_tau=int(args.calib_decay_tau))
        q_al, _ = apply_calibration(q_al, y0, mode=args.calib_mode, kind=args.calib_kind,
                                    anchor=args.calib_anchor, apply_where='after_filter',
                                    decay_tau=int(args.calib_decay_tau))

    diff = np.abs(c_al - q_al)
    mae = float(np.mean(diff)) if diff.size else float('nan')
    rmse = float(np.sqrt(np.mean(diff**2))) if diff.size else float('nan')
    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parents[0] / f'COMPARE_C_VS_QUANT_STEP_{time.strftime("%Y%m%d_%H%M%S")}')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save metrics and the exact command as MD for reproducibility
    (out_dir / 'metrics.json').write_text(json.dumps({
        'N': int(len(c_al)), 'MAE_c_vs_quant': mae, 'RMSE_c_vs_quant': rmse,
        'chunk': chunk, 'prime': bool(args.prime), **calib_info,
        'strict_filter': bool(args.strict_filter), 'post_max_rel': args.post_max_rel,
        'post_max_abs': args.post_max_abs, 'post_ema_alpha': args.post_ema_alpha,
        **affine_info,
        'quant_scale': args.quant_scale, 'quant_hh_precision': args.quant_hh_precision,
    }, indent=2))
    # Save the command used
    try:
        import sys
        cmd = 'python ' + Path(__file__).as_posix() + ' ' + ' '.join(sys.argv[1:])
        (out_dir / 'RUN_CMD.md').write_text('# Run command\n\n```\n' + cmd + '\n```\n')
    except Exception:
        pass
    overlay(c_al, q_al, y_al, out_dir / 'overlay_full.png', 'C step vs QUANT step (full)', plot_max=int(args.plot_max))
    firstN = min(2000, len(c_al))
    overlay(c_al[:firstN], q_al[:firstN], (y_al[:firstN] if y_al is not None else None), out_dir / 'overlay_firstN.png', f'first {firstN}')
    # diff hist
    plt.figure(figsize=(6,4))
    if diff.size:
        plt.hist(diff, bins=120, alpha=0.85, color='tab:purple', edgecolor='black')
    plt.title('Abs diff C vs QUANT')
    plt.tight_layout(); plt.savefig(out_dir / 'diff_hist.png', dpi=140); plt.close()
    print(f'[done] Saved comparison to {out_dir}')


if __name__ == '__main__':
    main()
