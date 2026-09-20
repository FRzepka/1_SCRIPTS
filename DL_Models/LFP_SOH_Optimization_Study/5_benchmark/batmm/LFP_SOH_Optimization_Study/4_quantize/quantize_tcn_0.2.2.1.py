#!/usr/bin/env python3
from pathlib import Path

from quantize_soh_model import run_quantize


def main():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / '2_models' / 'TCN' / 'Base' / '0.2.2.1'
    out_dir = root / '2_models' / 'TCN' / 'Quantized' / '0.2.2.1'
    run_quantize(model_dir=model_dir, out_dir=out_dir, ckpt_path=None)


if __name__ == '__main__':
    main()
