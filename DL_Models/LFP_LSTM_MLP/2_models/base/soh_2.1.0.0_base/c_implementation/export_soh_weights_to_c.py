#!/usr/bin/env python3
"""
Export a FP32 SOH LSTM+MLP checkpoint (2.1.0.0) to a C header for the SOH C implementation.

Writes model_weights_soh.h with macros and arrays for LSTM and MLP.
"""
import argparse
from pathlib import Path
import numpy as np
import torch


def format_array_c(arr: np.ndarray, name: str) -> str:
    flat = arr.astype(np.float32).ravel()
    out = [f"const float {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        out.append("    " + ", ".join(f"{float(v):.8f}f" for v in chunk) + ",")
    if len(out) > 1:
        out[-1] = out[-1].rstrip(',')
    out.append("};")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(Path('DL_Models/LFP_LSTM_MLP/2_models/base/2.1.0.0_soh_epoch0120_rmse0.03359.pt').resolve()))
    ap.add_argument('--out_dir', default=str(Path('DL_Models/LFP_LSTM_MLP/2_models/base/SOH_c_implementation').resolve()))
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h = out_dir / 'model_weights_soh.h'

    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    W_ih = sd['lstm.weight_ih_l0'].cpu().numpy()  # [4H, In]
    W_hh = sd['lstm.weight_hh_l0'].cpu().numpy()  # [4H, H]
    b = (sd['lstm.bias_ih_l0'] + sd['lstm.bias_hh_l0']).cpu().numpy()
    H4, In = W_ih.shape
    H = H4 // 4

    mlp0_w = sd['mlp.0.weight'].cpu().numpy()  # [M, H]
    mlp0_b = sd['mlp.0.bias'].cpu().numpy()    # [M]
    mlp2_w = sd['mlp.3.weight'].cpu().numpy()  # [1, M]
    mlp2_b = sd['mlp.3.bias'].cpu().numpy()    # [1]
    M = mlp0_w.shape[0]

    lines = []
    lines += ["/*",
              " * Auto-generated SOH model weights (2.1.0.0)",
              f" * Checkpoint: {ckpt_path.name}",
              " * DO NOT EDIT MANUALLY",
              " */\n",
              "#ifndef MODEL_WEIGHTS_SOH_H",
              "#define MODEL_WEIGHTS_SOH_H\n",
              "/* Model configuration */",
              f"#define INPUT_SIZE {In}",
              f"#define HIDDEN_SIZE {H}",
              f"#define MLP_HIDDEN {M}",
              f"#define NUM_LAYERS 1\n",
              f"/* LSTM input weights: [4*H={4*H}, In={In}] */",
              format_array_c(W_ih, 'LSTM_WEIGHT_IH'),
              "",
              f"/* LSTM hidden weights: [4*H={4*H}, H={H}] */",
              format_array_c(W_hh, 'LSTM_WEIGHT_HH'),
              "",
              f"/* LSTM bias: [4*H={4*H}] */",
              format_array_c(b, 'LSTM_BIAS'),
              "",
              "/* MLP */",
              f"/* FC1: [{M}, {H}] */",
              format_array_c(mlp0_w, 'MLP_FC1_WEIGHT'),
              "",
              f"/* FC1 bias: [{M}] */",
              format_array_c(mlp0_b, 'MLP_FC1_BIAS'),
              "",
              f"/* FC2: [1, {M}] */",
              format_array_c(mlp2_w, 'MLP_FC2_WEIGHT'),
              "",
              "/* FC2 bias: [1] */",
              format_array_c(mlp2_b, 'MLP_FC2_BIAS'),
              "",
              "#endif /* MODEL_WEIGHTS_SOH_H */\n"]

    out_h.write_text("\n".join(lines))
    print(f"[export] Wrote {out_h}")
    print(f"         INPUT_SIZE={In} HIDDEN_SIZE={H} MLP_HIDDEN={M}")


if __name__ == '__main__':
    main()

