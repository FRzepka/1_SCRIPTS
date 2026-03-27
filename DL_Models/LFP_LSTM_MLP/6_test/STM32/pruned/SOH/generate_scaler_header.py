#!/usr/bin/env python3
"""
Generate scaler_params_soh.h from a pruned SOH checkpoint's RobustScaler

Usage:
  python generate_scaler_header.py --ckpt <soh_pruned.pt> \
      --out STM32/workspace_1.17.0/AI_Project_LSTM_SOH_pruned/Core/Inc/scaler_params_soh.h

Notes:
  - Reads 'features' and 'scaler_path' from the checkpoint.
  - Requires joblib scaler with center_ and scale_ attributes.
  - Writes C header with SCALER_SOH_CENTER and SCALER_SOH_SCALE matching the checkpoint feature order.
"""
import argparse
from pathlib import Path
import json

import numpy as np

def load_ckpt(ckpt_path: Path):
    import torch
    st = torch.load(str(ckpt_path), map_location='cpu')
    if not isinstance(st, dict):
        raise ValueError('Unsupported checkpoint format')
    feats = st.get('features') or st.get('feature_list')
    if not feats or not isinstance(feats, (list, tuple)):
        raise ValueError("Checkpoint lacks 'features' list")
    scaler_path = st.get('scaler_path')
    return feats, scaler_path

def load_scaler(p: Path):
    from joblib import load as joblib_load
    sc = joblib_load(str(p))
    # Accept RobustScaler-like
    cen = getattr(sc, 'center_', None)
    scl = getattr(sc, 'scale_', None)
    if cen is None or scl is None:
        raise ValueError('Scaler does not expose center_/scale_ (expect RobustScaler)')
    return np.array(cen, dtype=np.float32), np.array(scl, dtype=np.float32)

HDR_TEMPLATE = """/* RobustScaler parameters for SOH preprocessing (auto-generated) */
#ifndef SCALER_PARAMS_SOH_H
#define SCALER_PARAMS_SOH_H

#define SCALER_NUM_FEATURES {N}
static const float SCALER_SOH_CENTER[SCALER_NUM_FEATURES] = {{
{CENTER}
}};
static const float SCALER_SOH_SCALE[SCALER_NUM_FEATURES] = {{
{SCALE}
}};
static inline void scaler_soh_transform(float inout[SCALER_NUM_FEATURES]){{
  for (int i=0;i<SCALER_NUM_FEATURES;i++){{ inout[i] = (inout[i] - SCALER_SOH_CENTER[i]) / SCALER_SOH_SCALE[i]; }}
}}

#endif /* SCALER_PARAMS_SOH_H */
"""

def fmt_arr(a):
    # Format as C floats with sufficient precision
    return ", ".join(f"  {x:.10f}f" for x in a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='Pruned SOH checkpoint (.pt)')
    ap.add_argument('--out', required=True, help='Output header path (scaler_params_soh.h)')
    ap.add_argument('--scaler', default='', help='Explicit scaler path (overrides ckpt.scaler_path)')
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    feats, scaler_path = load_ckpt(ckpt)
    if args.scaler:
        scaler_path = args.scaler
    if not scaler_path:
        raise SystemExit('No scaler_path in ckpt; please pass --scaler path/to/scaler_robust.joblib')
    sp = Path(scaler_path)
    if not sp.exists():
        raise SystemExit(f'scaler not found: {sp}')

    cen, scl = load_scaler(sp)
    if len(cen) != len(feats) or len(scl) != len(feats):
        raise SystemExit(f'scaler dims mismatch: center={len(cen)} scale={len(scl)} features={len(feats)}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hdr = HDR_TEMPLATE.format(N=len(feats), CENTER=fmt_arr(cen), SCALE=fmt_arr(scl))
    out.write_text(hdr, encoding='utf-8')

    print('[ok] wrote', out)
    print('[info] feature order:', json.dumps(feats))

if __name__ == '__main__':
    main()

