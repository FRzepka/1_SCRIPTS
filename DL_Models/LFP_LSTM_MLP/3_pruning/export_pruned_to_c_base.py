#!/usr/bin/env python3
"""
Export a PRUNED FP32 LSTM+MLP checkpoint to a C header for the base STM32 project (AI_Project_LSTM).

The generated header mirrors 2_models/base/c_implementation/model_weights.h, but uses the
pruned hidden size (and pruned MLP input accordingly). The base C code uses the macros
INPUT_SIZE, HIDDEN_SIZE, MLP_HIDDEN, NUM_LAYERS – so no code changes are needed as long as
the header is dropped in.

Usage (example):
  python export_pruned_to_c_base.py \
    --ckpt DL_Models/LFP_LSTM_MLP/2_models/pruned/soc_1.5.0.0/prune_30pct_20250916_140404/soc_pruned_hidden45.pt \
    --out_dir DL_Models/LFP_LSTM_MLP/2_models/pruned/soc_1.5.0.0/prune_30pct_20250916_140404/c_implementation

Then copy the generated model_weights.h into the STM32 base project (Core/Inc/model_weights.h)
and rebuild.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def format_array_c(arr: np.ndarray, name: str, dtype: str = 'float') -> str:
    flat = arr.astype(np.float32).ravel()
    lines = [f"const {dtype} {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        values = ', '.join([f'{float(v):.8f}f' for v in chunk])
        lines.append(f"    {values},")
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(',')
    lines.append("};")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Export pruned FP32 LSTM+MLP checkpoint to C header (base STM32)')
    ap.add_argument('--ckpt', required=True, help='Path to pruned .pt checkpoint (from 3_pruning)')
    ap.add_argument('--out_dir', required=False, default='', help='Output directory (default: alongside ckpt in c_implementation)')
    ap.add_argument('--header-name', default='model_weights.h', help='Header filename to emit (default: model_weights.h)')
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    # Decide out_dir
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = ckpt_path.parent / 'c_implementation'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h = out_dir / args.header_name

    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    # Infer sizes from state_dict shapes
    w_ih = sd['lstm.weight_ih_l0'].cpu().numpy()  # [4H, In]
    w_hh = sd['lstm.weight_hh_l0'].cpu().numpy()  # [4H, H]
    b_ih = sd['lstm.bias_ih_l0'].cpu().numpy()    # [4H]
    b_hh = sd['lstm.bias_hh_l0'].cpu().numpy()    # [4H]
    bias = b_ih + b_hh

    H4, In = w_ih.shape
    H = H4 // 4

    # MLP weights from sequential: [0]=Linear(in=H, out=M), [3]=Linear(in=M, out=1)
    mlp0_w = sd['mlp.0.weight'].cpu().numpy()  # [M, H]
    mlp0_b = sd['mlp.0.bias'].cpu().numpy()    # [M]
    mlp3_w = sd['mlp.3.weight'].cpu().numpy()  # [1, M]
    mlp3_b = sd['mlp.3.bias'].cpu().numpy()    # [1]
    M = mlp0_w.shape[0]

    # Compose header
    header = []
    header.append("/*")
    header.append(" * Auto-generated PRUNED model weights for LSTM-MLP SOC prediction (FP32 base)")
    header.append(f" * Checkpoint: {ckpt_path.name}")
    header.append(" * DO NOT EDIT MANUALLY")
    header.append(" */\n")
    header.append("#ifndef MODEL_WEIGHTS_H")
    header.append("#define MODEL_WEIGHTS_H\n")
    header.append("/* Model configuration */")
    header.append(f"#define INPUT_SIZE {In}")
    header.append(f"#define HIDDEN_SIZE {H}")
    header.append(f"#define MLP_HIDDEN {M}")
    header.append(f"#define NUM_LAYERS 1\n")

    # LSTM arrays
    header.append(f"/* LSTM input weights: [4*HIDDEN_SIZE={4*H}, INPUT_SIZE={In}] */")
    header.append(format_array_c(w_ih, 'LSTM_WEIGHT_IH'))
    header.append("")
    header.append(f"/* LSTM hidden weights: [4*HIDDEN_SIZE={4*H}, HIDDEN_SIZE={H}] */")
    header.append(format_array_c(w_hh, 'LSTM_WEIGHT_HH'))
    header.append("")
    header.append(f"/* LSTM bias: [4*HIDDEN_SIZE={4*H}] */")
    header.append(format_array_c(bias, 'LSTM_BIAS'))
    header.append("")

    # MLP arrays
    header.append("/* MLP Weights */")
    header.append(f"/* FC1: [{M}, {H}] */")
    header.append(format_array_c(mlp0_w, 'MLP_FC1_WEIGHT'))
    header.append("")
    header.append(f"/* FC1 bias: [{M}] */")
    header.append(format_array_c(mlp0_b, 'MLP_FC1_BIAS'))
    header.append("")
    header.append(f"/* FC2: [1, {M}] */")
    header.append(format_array_c(mlp3_w, 'MLP_FC2_WEIGHT'))
    header.append("")
    header.append("/* FC2 bias: [1] */")
    header.append(format_array_c(mlp3_b, 'MLP_FC2_BIAS'))
    header.append("")
    header.append("#endif /* MODEL_WEIGHTS_H */\n")

    out_h.write_text('\n'.join(header))
    print(f"[export] C header written: {out_h}")
    print(f"          INPUT_SIZE={In} HIDDEN_SIZE={H} MLP_HIDDEN={M}")


if __name__ == '__main__':
    main()
