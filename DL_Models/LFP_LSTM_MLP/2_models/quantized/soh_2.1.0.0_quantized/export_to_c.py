#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np


def _write_array_1d(f, ctype, name, arr):
    flat = arr.reshape(-1)
    f.write('static const %s %s[%d] = {\n' % (ctype, name, flat.size))
    line = []
    for v in flat:
        line.append(str(int(v)) if ctype in ('int8_t','int16_t') else repr(float(v)))
        if len(line) >= 16:
            f.write(', '.join(line) + ',\n'); line = []
    if line:
        f.write(', '.join(line) + '\n')
    f.write('};\n\n')


def _write_array_2d(f, ctype, name, arr):
    rows, cols = arr.shape
    f.write('static const %s %s[%d][%d] = {\n' % (ctype, name, rows, cols))
    for r in range(rows):
        line = ', '.join(str(int(v)) if ctype in ('int8_t','int16_t') else repr(float(v)) for v in arr[r])
        f.write('  {' + line + '},\n')
    f.write('};\n\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.npz)
    q = {k: z[k] for k in z.files}
    In = int(q['input_size']); H = int(q['hidden_size'])
    G = 4 * H
    Wih = q['W_ih_q'].reshape(G, In)
    Whh = q['W_hh_q'].reshape(G, H)
    B = q['B'].reshape(G)
    mlp0_w = q['mlp0_w']  # [O0, H]
    mlp0_b = q['mlp0_b']
    mlp1_w = q['mlp1_w']  # [1, O0]
    mlp1_b = q['mlp1_b']
    O0 = int(mlp0_b.shape[0])

    # Params + scales with compatibility aliases
    with (out / 'quant_params_soh.h').open('w', encoding='utf-8') as f:
        f.write('#pragma once\n#include <stdint.h>\n\n')
        f.write(f'#define SOH_IN_SIZE {In}\n')
        f.write(f'#define SOH_HIDDEN_SIZE {H}\n')
        f.write(f'#define INPUT_SIZE {In}\n')
        f.write(f'#define HIDDEN_SIZE {H}\n')
        f.write(f'#define LSTM_CHANNELS {G}\n')
        f.write(f'#define MLP0_IN_DIM {H}\n')
        f.write(f'#define MLP0_OUT_DIM {O0}\n')
        f.write(f'#define MLP1_IN_DIM {O0}\n')
        f.write(f'#define MLP1_OUT_DIM 1\n\n')
        f.write('// Per-row scales for dequantization\n')
        _write_array_1d(f, 'float', 'S_IH', q['S_ih'].astype(np.float32))
        _write_array_1d(f, 'float', 'S_HH', q['S_hh'].astype(np.float32))
        # Compatibility aliases
        _write_array_1d(f, 'float', 'LSTM_W_IH_SCALE', q['S_ih'].astype(np.float32))
        _write_array_1d(f, 'float', 'LSTM_W_HH_SCALE', q['S_hh'].astype(np.float32))

    # Weights/bias as 2D arrays with legacy names
    with (out / 'quant_weights_soh.h').open('w', encoding='utf-8') as f:
        f.write('#pragma once\n#include <stdint.h>\n\n')
        hh_dtype = 'int16_t' if q['W_hh_q'].dtype == np.int16 else 'int8_t'
        _write_array_2d(f, 'int8_t', 'LSTM_W_IH', Wih.astype(np.int8))
        _write_array_2d(f, hh_dtype, 'LSTM_W_HH', Whh)
        _write_array_1d(f, 'float', 'LSTM_B', B.astype(np.float32))
        _write_array_2d(f, 'float', 'MLP0_WEIGHT', mlp0_w.astype(np.float32))
        _write_array_1d(f, 'float', 'MLP0_BIAS', mlp0_b.astype(np.float32))
        _write_array_2d(f, 'float', 'MLP1_WEIGHT', mlp1_w.astype(np.float32))
        _write_array_1d(f, 'float', 'MLP1_BIAS', mlp1_b.astype(np.float32))

    (out / 'README.txt').write_text(
        'Quantized SOH LSTM export (headers)\n\n'
        '- quant_params_soh.h: sizes + scales (incl. compatibility aliases).\n'
        '- quant_weights_soh.h: weights/bias with legacy names (drop-in).\n'
    )
    print(f'[done] Wrote headers to {out}')


if __name__ == '__main__':
    main()
