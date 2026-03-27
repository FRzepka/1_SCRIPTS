"""
Manual INT8 quantization of LSTM weights from PyTorch checkpoint (no ONNX).

What it does:
- Loads FP32 checkpoint (LSTM+MLP)
- Quantizes LSTM weights (W_ih, W_hh) to int8 with per-row symmetric scales
- Runs a streaming comparison FP32 vs int8-weights LSTM (activations FP32)
- Exports a C header with int8 weights + per-row scales + FP32 bias
- Saves metrics and plots under 6_tests/INT8_MANUAL_<timestamp>

Usage:
  python manual_lstm_int8_from_pt.py \
    --ckpt DL_Models/LFP_LSTM_MLP/2_models/base/1.5.0.0_soc_epoch0001_rmse0.02897.pt \
    --num_samples 5000
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

BASE = Path(__file__).resolve().parent.parent.parent.parent  # -> 1_Scripts/
CONFIG_PATH = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '1.5.0.0' / 'config' / 'train_soc.yaml'
SCALER_PATH = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '1_training' / '1.5.0.0' / 'outputs' / 'scaler_robust.joblib'
DATA_PATH = Path('/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE/df_FE_C07.parquet')


def norm(s: str) -> str:
    return (s.replace('ä', 'a').replace('Ä', 'A')
              .replace('ö', 'o').replace('Ö', 'O')
              .replace('ü', 'u').replace('Ü', 'U')
              .replace('ß', 'ss').replace("'", ""))


def load_data(num_samples=1000):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_parquet(DATA_PATH)
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
    y = df['SOC'].values[:num_samples].astype(np.float32)
    Xs = scaler.transform(X_raw).astype(np.float32)
    return Xs, y, cfg


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


def run_streaming_compare(
    Xs: np.ndarray,
    state_dict: dict,
    hidden_size: int,
    show_progress: bool = False,
    desc: str = "Streaming",
):
    """Compare FP32 LSTMCell vs int8-weight manual forward over sequence Xs.

    Wenn ``show_progress=True`` wird eine tqdm-Progressbar verwendet (falls verfügbar).
    """
    W_ih = state_dict['lstm.weight_ih_l0'].numpy()
    W_hh = state_dict['lstm.weight_hh_l0'].numpy()
    b_ih = state_dict['lstm.bias_ih_l0'].numpy()
    b_hh = state_dict['lstm.bias_hh_l0'].numpy()
    b_total = b_ih + b_hh

    input_size = Xs.shape[1]
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

    # Build MLP from checkpoint
    mlp0_w = state_dict['mlp.0.weight'].numpy()
    mlp0_b = state_dict['mlp.0.bias'].numpy()
    mlp3_w = state_dict['mlp.3.weight'].numpy()
    mlp3_b = state_dict['mlp.3.bias'].numpy()

    def mlp_forward(h):
        x = h @ mlp0_w.T + mlp0_b
        x = np.maximum(0.0, x)
        x = x @ mlp3_w.T + mlp3_b
        return 1.0 / (1.0 + np.exp(-x))  # sigmoid

    # Streaming
    iterator = range(Xs.shape[0])
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(iterator, total=Xs.shape[0], desc=desc)
        except Exception:
            # Fallback ohne Progressbar, falls tqdm nicht installiert ist
            pass

    for t in iterator:
        x = Xs[t]  # [In]

        # FP32 path
        h_ref_t, c_ref_t = cell(torch.from_numpy(x).unsqueeze(0), (torch.from_numpy(h_ref).unsqueeze(0), torch.from_numpy(c_ref).unsqueeze(0)))
        h_ref = h_ref_t.detach().squeeze(0).numpy().astype(np.float32)
        c_ref = c_ref_t.detach().squeeze(0).numpy().astype(np.float32)
        y_ref = mlp_forward(h_ref)[0]
        preds_ref.append(float(y_ref))

        # INT8-weight path (activations FP32)
        # gates = (x @ W_ih_q.T) * S_ih + (h_i8 @ W_hh_q.T) * S_hh + b_total
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
    header = out_dir / 'model_weights_lstm_int8_manual.h'
    with open(header, 'w') as f:
        f.write('/* Manual LSTM INT8 Weights (per-row scales) */\n')
        f.write('#ifndef MODEL_WEIGHTS_LSTM_INT8_MANUAL_H\n#define MODEL_WEIGHTS_LSTM_INT8_MANUAL_H\n\n')
        f.write('#include <stdint.h>\n\n')
        # Dimensions
        f.write(f'#define INPUT_SIZE {input_size}\n')
        f.write(f'#define HIDDEN_SIZE {H}\n')
        f.write(f'#define LSTM_CHANNELS {4*H}\n\n')
        # W_ih
        f.write(f'static const int8_t LSTM_W_IH[LSTM_CHANNELS][INPUT_SIZE] = {{\n')
        for r in range(4*H):
            row = ', '.join(str(int(v)) for v in W_ih_q[r].tolist())
            f.write(f'  {{{row}}}{"," if r<4*H-1 else ""}\n')
        f.write('};\n\n')
        # W_hh
        f.write(f'static const int8_t LSTM_W_HH[LSTM_CHANNELS][HIDDEN_SIZE] = {{\n')
        for r in range(4*H):
            row = ', '.join(str(int(v)) for v in W_hh_q[r].tolist())
            f.write(f'  {{{row}}}{"," if r<4*H-1 else ""}\n')
        f.write('};\n\n')
        # Scales
        f.write(f'static const float LSTM_W_IH_SCALE[LSTM_CHANNELS] = {{\n  ' + ', '.join(f'{float(s):.8e}f' for s in S_ih.tolist()) + '\n};\n\n')
        f.write(f'static const float LSTM_W_HH_SCALE[LSTM_CHANNELS] = {{\n  ' + ', '.join(f'{float(s):.8e}f' for s in S_hh.tolist()) + '\n};\n\n')
        # Bias
        f.write(f'static const float LSTM_B[LSTM_CHANNELS] = {{\n  ' + ', '.join(f'{float(b):.8e}f' for b in B.tolist()) + '\n};\n\n')
        f.write('#endif /* MODEL_WEIGHTS_LSTM_INT8_MANUAL_H */\n')
    return header


def save_plots(out_dir: Path, y_true, fp32, int8):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(y_true)
    plt.figure(figsize=(12,4))
    plt.plot(y_true, label='GT', alpha=0.6)
    plt.plot(fp32, label='FP32')
    plt.plot(int8, label='INT8_W')
    plt.title(f'Overlay full ({n} samples)')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / 'overlay_full.png', dpi=150); plt.close()

    firstN = min(5000, n)
    plt.figure(figsize=(12,4))
    plt.plot(y_true[:firstN], label='GT', alpha=0.6)
    plt.plot(fp32[:firstN], label='FP32')
    plt.plot(int8[:firstN], label='INT8_W')
    plt.title(f'Overlay first {firstN}')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / 'overlay_firstN.png', dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--num_samples', type=int, default=5000)
    ap.add_argument('--package_out', type=str, default=str(BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '2_models' / 'quantized' / 'manual_int8_lstm'))
    args = ap.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    hidden_size = int(state_dict['lstm.weight_hh_l0'].shape[1])
    input_size = int(state_dict['lstm.weight_ih_l0'].shape[1])

    # Load data
    Xs, y_true, cfg = load_data(args.num_samples)

    # Compare streams
    fp32_preds, int8_preds, metrics, pack = run_streaming_compare(Xs, state_dict, hidden_size)
    W_ih_q, S_ih, W_hh_q, S_hh, B = pack

    # Save artifacts
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = BASE / 'DL_Models' / 'LFP_LSTM_MLP' / '6_tests' / f'INT8_MANUAL_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    save_plots(out_dir, y_true, fp32_preds, int8_preds)

    # Package headers to quantized folder
    pkg_dir = Path(args.package_out)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    header = export_header(pkg_dir, W_ih_q, S_ih, W_hh_q, S_hh, B, hidden_size, input_size)
    # Export MLP FP32 header
    mlp0_w = state_dict['mlp.0.weight'].numpy(); mlp0_b = state_dict['mlp.0.bias'].numpy()
    mlp1_w = state_dict['mlp.3.weight'].numpy(); mlp1_b = state_dict['mlp.3.bias'].numpy()
    # write mlp header
    mlp_header_path = pkg_dir / 'mlp_weights_fp32.h'
    with open(mlp_header_path, 'w') as f:
        out_dim0, in_dim0 = mlp0_w.shape
        out_dim1, in_dim1 = mlp1_w.shape
        f.write('/* MLP FP32 Weights for head: ReLU -> Sigmoid */\n')
        f.write('#ifndef MLP_WEIGHTS_FP32_H\n#define MLP_WEIGHTS_FP32_H\n\n')
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
        f.write('#endif /* MLP_WEIGHTS_FP32_H */\n')
    # Manifest and size plot
    manifest = {
        'checkpoint': str(Path(args.ckpt)),
        'hidden_size': hidden_size,
        'input_size': input_size,
        'headers': [str(header), str(mlp_header_path)]
    }
    with open(pkg_dir / 'model_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    fp32_lstm_bytes = (4 * ((4*hidden_size)*input_size + (4*hidden_size)*hidden_size) + 4 * (4*hidden_size))
    int8_lstm_bytes = (1 * ((4*hidden_size)*input_size + (4*hidden_size)*hidden_size) + 4 * (2*(4*hidden_size)) + 4 * (4*hidden_size))
    plt.figure(figsize=(5,4))
    labels = ['FP32 LSTM', 'INT8 LSTM']
    values = [fp32_lstm_bytes/1024.0, int8_lstm_bytes/1024.0]
    plt.bar(labels, values)
    plt.ylabel('KB'); plt.title('LSTM Memory Footprint'); plt.tight_layout()
    plt.savefig(pkg_dir / 'size_comparison.png', dpi=150); plt.close()

    print('='*100)
    print('Manual INT8 LSTM (weights-only) built')
    print('Metrics:', metrics)
    print('Packaged to:', pkg_dir)
    print('Test results saved to:', out_dir)
    print('='*100)


if __name__ == '__main__':
    main()

