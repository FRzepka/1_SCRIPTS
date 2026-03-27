#!/usr/bin/env python3
"""
Export Pruned SOH LSTM+MLP checkpoint to C headers.
Combines weight export and scaler export.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import joblib

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

def export_weights(ckpt_path: Path, out_dir: Path):
    print(f"[weights] Loading {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    W_ih = sd['lstm.weight_ih_l0'].cpu().numpy()  # [4H, In]
    W_hh = sd['lstm.weight_hh_l0'].cpu().numpy()  # [4H, H]
    
    # Handle bias (some pruned models might separate them, but usually they are summed in export)
    # Check if both exist
    b_ih = sd['lstm.bias_ih_l0'].cpu().numpy()
    b_hh = sd['lstm.bias_hh_l0'].cpu().numpy()
    b = b_ih + b_hh

    H4, In = W_ih.shape
    H = H4 // 4

    mlp0_w = sd['mlp.0.weight'].cpu().numpy()  # [M, H]
    mlp0_b = sd['mlp.0.bias'].cpu().numpy()    # [M]
    mlp2_w = sd['mlp.3.weight'].cpu().numpy()  # [1, M]
    mlp2_b = sd['mlp.3.bias'].cpu().numpy()    # [1]
    M = mlp0_w.shape[0]

    out_h = out_dir / 'model_weights_soh.h'
    
    lines = []
    lines += ["/*",
              " * Auto-generated Pruned SOH model weights",
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
    print(f"[weights] Wrote {out_h}")
    print(f"          INPUT_SIZE={In} HIDDEN_SIZE={H} MLP_HIDDEN={M}")

def export_scaler(scaler_path: Path, out_dir: Path):
    print(f"[scaler] Loading {scaler_path}...")
    scaler = joblib.load(str(scaler_path))
    center = np.asarray(scaler.center_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    n = center.shape[0]

    out_h = out_dir / 'scaler_params_soh.h'

    lines = []
    lines += ["/*",
              " * RobustScaler parameters for SOH preprocessing",
              " * DO NOT EDIT MANUALLY",
              " */\n",
              "#ifndef SCALER_PARAMS_SOH_H",
              "#define SCALER_PARAMS_SOH_H\n",
              f"#define SCALER_NUM_FEATURES {n}\n",
              "const float SCALER_SOH_CENTER[SCALER_NUM_FEATURES] = {",
              "    " + ", ".join(f"{float(v):.10f}f" for v in center),
              "};\n",
              "const float SCALER_SOH_SCALE[SCALER_NUM_FEATURES] = {",
              "    " + ", ".join(f"{float(v):.10f}f" for v in scale),
              "};\n",
              "static inline void scaler_soh_transform(const float in[SCALER_NUM_FEATURES], float out[SCALER_NUM_FEATURES]){",
              "    for (int i=0;i<SCALER_NUM_FEATURES;i++){ out[i] = (in[i] - SCALER_SOH_CENTER[i]) / SCALER_SOH_SCALE[i]; }",
              "}\n",
              "#endif /* SCALER_PARAMS_SOH_H */\n"]

    out_h.write_text("\n".join(lines))
    print(f"[scaler] Wrote {out_h}")

def main():
    # Hardcoded paths based on user request context
    base_dir = Path(__file__).parent
    ckpt_path = base_dir.parent / 'soh_pruned_hidden90.pt'
    scaler_path = Path('/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/1_training/2.1.0.0/outputs/scaler_robust.joblib')
    
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
    if not scaler_path.exists():
        print(f"Error: Scaler not found at {scaler_path}")
        return

    export_weights(ckpt_path, base_dir)
    export_scaler(scaler_path, base_dir)
    print("[done] C implementation headers generated.")

if __name__ == '__main__':
    main()
