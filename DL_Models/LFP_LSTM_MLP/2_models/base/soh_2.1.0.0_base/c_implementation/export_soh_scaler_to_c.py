#!/usr/bin/env python3
"""
Export RobustScaler parameters for SOH (2.1.0.0) to a C header (scaler_params_soh.h).
"""
import argparse
from pathlib import Path
import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scaler', default=str(Path('DL_Models/LFP_LSTM_MLP/1_training/2.1.0.0/outputs/scaler_robust.joblib').resolve()))
    ap.add_argument('--out_dir', default=str(Path('DL_Models/LFP_LSTM_MLP/2_models/base/SOH_c_implementation').resolve()))
    args = ap.parse_args()

    scaler_path = Path(args.scaler)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h = out_dir / 'scaler_params_soh.h'

    scaler = joblib.load(str(scaler_path))
    center = np.asarray(scaler.center_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    n = center.shape[0]

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
    print(f"[export] Wrote {out_h}")


if __name__ == '__main__':
    main()

