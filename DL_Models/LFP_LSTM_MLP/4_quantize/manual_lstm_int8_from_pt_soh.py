"""
Manual INT8 quantization of LSTM weights from a SOH PyTorch checkpoint (no ONNX).

What it does:
- Loads FP32 checkpoint (LSTM+MLP)
- Quantizes LSTM weights (W_ih, W_hh) to int8 with per-row symmetric scales
- Runs a streaming comparison FP32 vs int8-weights LSTM (activations FP32)
- Exports C headers package for STM32 with:
  * model_weights_lstm_int8_manual_soh.h (int8 weights + scales + bias)
  * mlp_weights_fp32_soh.h (FP32 MLP head weights; linear head)
  * scaler_params_soh.h (RobustScaler for SOH features)
- Saves metrics and plots under 6_tests/INT8_MANUAL_SOH_<timestamp>

Usage:
  python manual_lstm_int8_from_pt_soh.py \
    --ckpt DL_Models/LFP_LSTM_MLP/2_models/base/2.1.0.0_soh_epoch0120_rmse0.03359.pt \
    --num_samples 5000 \
    --package_out DL_Models/LFP_LSTM_MLP/2_models/quantized/manual_int8_lstm_soh
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
# Robustly locate repo root that contains DL_Models
_p = _HERE
while _p.name != 'DL_Models' and _p != _p.parent:
    _p = _p.parent
if _p.name != 'DL_Models':
    # Fallback to 4 levels up as typical layout
    BASE = _HERE.parents[4]
else:
    BASE = _p.parent  # 1_Scripts root
SOH_CONFIG = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '2.1.0.0' / 'config' / 'train_soh.yaml'
SOH_SCALER = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '2.1.0.0' / 'outputs' / 'scaler_robust.joblib'
SOH_DATA   = Path(r'C:\\Users\\Florian\\SynologyDrive\\TUB\\3_Projekte\\MG_Farm\\5_Data\\01_LFP\\00_Data\\Versuch_18650_standart\\MGFarm_18650_FE\\df_FE_C07.parquet')


def norm(s: str) -> str:
    return (s.replace('��', 'a').replace('�"', 'A')
              .replace('��', 'o').replace('�-', 'O')
              .replace('Ǭ', 'u').replace('�o', 'U')
              .replace('�Y', 'ss').replace("'", ""))


def load_data(num_samples=1000):
    with open(SOH_CONFIG, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scaler = joblib.load(SOH_SCALER)
    df = pd.read_parquet(SOH_DATA)
    feature_names = cfg['model']['features']
    feats = []
    for feat in feature_names:
        if feat in df.columns:
            feats.append(feat)
        else:
            fkey = norm(feat)
            found = next((c for c in df.columns if norm(c) == fkey or fkey in norm(c)), None)
            if not found:
                raise KeyError(f'Feature {feat} not found')
            feats.append(found)
    X_raw = df[feats].values[:num_samples].astype(np.float32)
    y = df['SOH'].values[:num_samples].astype(np.float32) if 'SOH' in df.columns else np.zeros((min(num_samples, len(df)),), dtype=np.float32)
    Xs = scaler.transform(X_raw).astype(np.float32)
    return Xs, y, cfg, scaler


def quantize_per_row(W: np.ndarray):
    """Symmetric per-row int8 quantization for 2D weight matrix."""
    rows, cols = W.shape
    scales = np.zeros(rows, dtype=np.float32)
    Wq = np.zeros((rows, cols), dtype=np.int8)
    eps = 1e-12
    for r in range(rows):
        max_abs = np.max(np.abs(W[r]))
        scale = max(max_abs / 127.0, eps)
        scales[r] = scale
        Wq[r] = np.clip(np.round(W[r] / scale), -127, 127).astype(np.int8)
    return Wq, scales


def fp32_lstmcell_from_state(W_ih, W_hh, b_ih, b_hh, input_size, hidden_size):
    cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
    with torch.no_grad():
        cell.weight_ih.copy_(torch.from_numpy(W_ih))
        cell.weight_hh.copy_(torch.from_numpy(W_hh))
        cell.bias_ih.copy_(torch.from_numpy(b_ih))
        cell.bias_hh.copy_(torch.from_numpy(b_hh))
    cell.eval()
    return cell


def run_streaming_compare(Xs: np.ndarray, state_dict: dict, hidden_size: int):
    """Compare FP32 LSTMCell vs int8-weight manual forward over sequence Xs for SOH (linear head)."""
    W_ih = state_dict['lstm.weight_ih_l0'].numpy()
    W_hh = state_dict['lstm.weight_hh_l0'].numpy()
    b_ih = state_dict['lstm.bias_ih_l0'].numpy()
    b_hh = state_dict['lstm.bias_hh_l0'].numpy()
    b_total = b_ih + b_hh

    input_size = int(W_ih.shape[1])
    H = hidden_size

    # FP32 reference cell
    cell = fp32_lstmcell_from_state(W_ih, W_hh, b_ih, b_hh, input_size, H)

    # INT8 weights (per-row scales)
    W_ih_q, S_ih = quantize_per_row(W_ih)
    W_hh_q, S_hh = quantize_per_row(W_hh)

    # Buffers
    h_ref = np.zeros((H,), dtype=np.float32)
    c_ref = np.zeros((H,), dtype=np.float32)
    h_i8 = np.zeros((H,), dtype=np.float32)
    c_i8 = np.zeros((H,), dtype=np.float32)

    preds_ref = []
    preds_i8 = []

    # Build MLP (linear head) from checkpoint
    mlp0_w = state_dict['mlp.0.weight'].numpy()
    mlp0_b = state_dict['mlp.0.bias'].numpy()
    mlp3_w = state_dict['mlp.3.weight'].numpy()
    mlp3_b = state_dict['mlp.3.bias'].numpy()

    def mlp_forward(h):
        x = h @ mlp0_w.T + mlp0_b
        x = np.maximum(0.0, x)  # ReLU
        x = x @ mlp3_w.T + mlp3_b  # linear head (no sigmoid)
        return x

    # Streaming
    # Ensure input dims match expected
    if Xs.shape[1] != input_size:
        if Xs.shape[1] < input_size:
            pad = np.zeros((Xs.shape[0], input_size - Xs.shape[1]), dtype=np.float32)
            Xs_use = np.concatenate([Xs, pad], axis=1)
        else:
            Xs_use = Xs[:, :input_size]
    else:
        Xs_use = Xs

    for t in range(Xs_use.shape[0]):
        x = Xs_use[t]

        # FP32 path
        h_ref_t, c_ref_t = cell(torch.from_numpy(x).unsqueeze(0), (torch.from_numpy(h_ref).unsqueeze(0), torch.from_numpy(c_ref).unsqueeze(0)))
        h_ref = h_ref_t.detach().squeeze(0).numpy().astype(np.float32)
        c_ref = c_ref_t.detach().squeeze(0).numpy().astype(np.float32)
        y_ref = mlp_forward(h_ref)[0]
        preds_ref.append(float(y_ref))

        # INT8-weight path (activations FP32)
        gates_ih = (x @ W_ih_q.T).astype(np.float32) * S_ih
        gates_hh = (h_i8 @ W_hh_q.T).astype(np.float32) * S_hh
        gates = gates_ih + gates_hh + b_total
        i_pre = gates[0:H]
        f_pre = gates[H:2*H]
        g_pre = gates[2*H:3*H]
        o_pre = gates[3*H:4*H]
        i = 1.0 / (1.0 + np.exp(-i_pre))
        f = 1.0 / (1.0 + np.exp(-f_pre))
        g = np.tanh(g_pre)
        o = 1.0 / (1.0 + np.exp(-o_pre))
        c_i8 = f * c_i8 + i * g
        h_i8 = o * np.tanh(c_i8)
        y_i8 = mlp_forward(h_i8)[0]
        preds_i8.append(float(y_i8))

    preds_ref = np.array(preds_ref)
    preds_i8 = np.array(preds_i8)
    diffs = np.abs(preds_ref - preds_i8)
    mae = float(np.mean(np.abs(preds_ref - preds_i8)))
    rmse = float(np.sqrt(np.mean((preds_ref - preds_i8)**2)))
    mxd = float(np.max(diffs))

    return preds_ref, preds_i8, {'MAE': mae, 'RMSE': rmse, 'MAX': mxd}, (W_ih_q, S_ih, W_hh_q, S_hh, b_total)


def export_header(out_dir: Path, W_ih_q, S_ih, W_hh_q, S_hh, B, hidden_size: int, input_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    H = hidden_size
    header = out_dir / 'model_weights_lstm_int8_manual_soh.h'
    with open(header, 'w') as f:
        f.write('/* Manual LSTM INT8 Weights (per-row scales) for SOH */\n')
        f.write('#ifndef MODEL_WEIGHTS_LSTM_INT8_MANUAL_SOH_H\n#define MODEL_WEIGHTS_LSTM_INT8_MANUAL_SOH_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write(f'#define INPUT_SIZE {input_size}\n')
        f.write(f'#define HIDDEN_SIZE {H}\n')
        f.write(f'#define LSTM_CHANNELS {4*H}\n\n')
        f.write(f'static const int8_t LSTM_W_IH[LSTM_CHANNELS][INPUT_SIZE] = {{\n')
        for r in range(4*H):
            row = ', '.join(str(int(v)) for v in W_ih_q[r].tolist())
            f.write(f'  {{{row}}}{"," if r<4*H-1 else ""}\n')
        f.write('};\n\n')
        f.write(f'static const int8_t LSTM_W_HH[LSTM_CHANNELS][HIDDEN_SIZE] = {{\n')
        for r in range(4*H):
            row = ', '.join(str(int(v)) for v in W_hh_q[r].tolist())
            f.write(f'  {{{row}}}{"," if r<4*H-1 else ""}\n')
        f.write('};\n\n')
        f.write(f'static const float LSTM_W_IH_SCALE[LSTM_CHANNELS] = {{\n  ' + ', '.join(f'{float(s):.8e}f' for s in S_ih.tolist()) + '\n};\n\n')
        f.write(f'static const float LSTM_W_HH_SCALE[LSTM_CHANNELS] = {{\n  ' + ', '.join(f'{float(s):.8e}f' for s in S_hh.tolist()) + '\n};\n\n')
        f.write(f'static const float LSTM_B[LSTM_CHANNELS] = {{\n  ' + ', '.join(f'{float(b):.8e}f' for b in B.tolist()) + '\n};\n\n')
        f.write('#endif /* MODEL_WEIGHTS_LSTM_INT8_MANUAL_SOH_H */\n')
    return header


def export_mlp_fp32_header(out_dir: Path, state_dict: dict):
    mlp0_w = state_dict['mlp.0.weight'].numpy(); mlp0_b = state_dict['mlp.0.bias'].numpy()
    mlp1_w = state_dict['mlp.3.weight'].numpy(); mlp1_b = state_dict['mlp.3.bias'].numpy()
    mlp_header_path = out_dir / 'mlp_weights_fp32_soh.h'
    with open(mlp_header_path, 'w') as f:
        out_dim0, in_dim0 = mlp0_w.shape
        out_dim1, in_dim1 = mlp1_w.shape
        f.write('/* MLP FP32 Weights for SOH head: ReLU -> Linear */\n')
        f.write('#ifndef MLP_WEIGHTS_FP32_SOH_H\n#define MLP_WEIGHTS_FP32_SOH_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write(f'#define MLP0_OUT_DIM {out_dim0}\n#define MLP0_IN_DIM {in_dim0}\n')
        f.write(f'#define MLP1_OUT_DIM {out_dim1}\n#define MLP1_IN_DIM {in_dim1}\n\n')
        f.write('static const float MLP0_WEIGHT[MLP0_OUT_DIM][MLP0_IN_DIM] = {\n')
        for r in range(out_dim0):
            row = ', '.join(f'{float(v):.8e}f' for v in mlp0_w[r].tolist())
            f.write(f'  {{{row}}}{"," if r<out_dim0-1 else ""}\n')
        f.write('};\n')
        f.write('static const float MLP0_BIAS[MLP0_OUT_DIM] = {\n  ' + ', '.join(f'{float(v):.8e}f' for v in mlp0_b.tolist()) + '\n};\n\n')
        f.write('static const float MLP1_WEIGHT[MLP1_OUT_DIM][MLP1_IN_DIM] = {\n')
        for r in range(out_dim1):
            row = ', '.join(f'{float(v):.8e}f' for v in mlp1_w[r].tolist())
            f.write(f'  {{{row}}}{"," if r<out_dim1-1 else ""}\n')
        f.write('};\n')
        f.write('static const float MLP1_BIAS[MLP1_OUT_DIM] = {\n  ' + ', '.join(f'{float(v):.8e}f' for v in mlp1_b.tolist()) + '\n};\n\n')
        f.write('#endif /* MLP_WEIGHTS_FP32_SOH_H */\n')
    return mlp_header_path


def export_scaler_header(out_dir: Path, scaler, input_size: int):
    out_h = out_dir / 'scaler_params_soh.h'
    center = None
    scale = None
    if hasattr(scaler, 'center_'):
        c = np.asarray(getattr(scaler, 'center_'), dtype=np.float32)
        s = np.asarray(getattr(scaler, 'scale_'), dtype=np.float32)
        if c.shape[0] == input_size:
            center, scale = c, s
    if center is None or scale is None:
        # Fallback to identity scaler matching weight input size
        center = np.zeros((input_size,), dtype=np.float32)
        scale = np.ones((input_size,), dtype=np.float32)
    n = center.shape[0]
    lines = []
    lines += [
        '/* RobustScaler parameters for SOH preprocessing */\n',
        '#ifndef SCALER_PARAMS_SOH_H\n#define SCALER_PARAMS_SOH_H\n\n',
        f'#define SCALER_NUM_FEATURES {n}\n',
        'static const float SCALER_SOH_CENTER[SCALER_NUM_FEATURES] = {\n  ' + ', '.join(f'{float(v):.10f}f' for v in center) + '\n};\n',
        'static const float SCALER_SOH_SCALE[SCALER_NUM_FEATURES] = {\n  ' + ', '.join(f'{float(v):.10f}f' for v in scale) + '\n};\n',
        'static inline void scaler_soh_transform(float inout[SCALER_NUM_FEATURES]){\n',
        '  for (int i=0;i<SCALER_NUM_FEATURES;i++){ inout[i] = (inout[i] - SCALER_SOH_CENTER[i]) / SCALER_SOH_SCALE[i]; }\n',
        '}\n\n',
        '#endif /* SCALER_PARAMS_SOH_H */\n'
    ]
    out_h.write_text(''.join(lines))
    return out_h


def save_plots(out_dir: Path, y_true, fp32, int8):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(fp32)
    plt.figure(figsize=(12,4))
    if len(y_true) == n:
        plt.plot(y_true, label='GT', alpha=0.5)
    plt.plot(fp32, label='FP32')
    plt.plot(int8, label='INT8_W')
    plt.title(f'Overlay full ({n} samples)')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / 'overlay_full.png', dpi=150); plt.close()

    firstN = min(5000, n)
    plt.figure(figsize=(12,4))
    if len(y_true) >= firstN:
        plt.plot(y_true[:firstN], label='GT', alpha=0.5)
    plt.plot(fp32[:firstN], label='FP32')
    plt.plot(int8[:firstN], label='INT8_W')
    plt.title(f'Overlay first {firstN}')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / 'overlay_firstN.png', dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--num_samples', type=int, default=5000)
    ap.add_argument('--package_out', type=str, default=str(BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '2_models' / 'quantized' / 'manual_int8_lstm_soh'))
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    hidden_size = int(state_dict['lstm.weight_hh_l0'].shape[1])
    input_size = int(state_dict['lstm.weight_ih_l0'].shape[1])

    Xs, y_true, cfg, scaler = load_data(args.num_samples)

    fp32_preds, int8_preds, metrics, pack = run_streaming_compare(Xs, state_dict, hidden_size)
    W_ih_q, S_ih, W_hh_q, S_hh, B = pack

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '6_tests' / f'INT8_MANUAL_SOH_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    save_plots(out_dir, y_true, fp32_preds, int8_preds)

    pkg_dir = Path(args.package_out)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    header = export_header(pkg_dir, W_ih_q, S_ih, W_hh_q, S_hh, B, hidden_size, input_size)
    mlp_header = export_mlp_fp32_header(pkg_dir, state_dict)
    scaler_header = export_scaler_header(pkg_dir, scaler, input_size)

    manifest = {
        'checkpoint': str(Path(args.ckpt)),
        'hidden_size': hidden_size,
        'input_size': input_size,
        'headers': [str(header), str(mlp_header), str(scaler_header)],
        'metrics': metrics,
    }
    with open(pkg_dir / 'model_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    fp32_lstm_bytes = (4 * ((4*hidden_size)*input_size + (4*hidden_size)*hidden_size) + 4 * (4*hidden_size))
    int8_lstm_bytes = (1 * ((4*hidden_size)*input_size + (4*hidden_size)*hidden_size) + 4 * (2*(4*hidden_size)) + 4 * (4*hidden_size))
    plt.figure(figsize=(5,4))
    labels = ['FP32 LSTM', 'INT8 LSTM']
    values = [fp32_lstm_bytes/1024.0, int8_lstm_bytes/1024.0]
    plt.bar(labels, values)
    plt.ylabel('KB'); plt.title('LSTM Memory Footprint (SOH)'); plt.tight_layout()
    plt.savefig(pkg_dir / 'size_comparison.png', dpi=150); plt.close()

    print('='*100)
    print('Manual INT8 LSTM (weights-only) for SOH built')
    print('Metrics:', metrics)
    print('Packaged to:', pkg_dir)
    print('Test results saved to:', out_dir)
    print('='*100)


if __name__ == '__main__':
    main()
