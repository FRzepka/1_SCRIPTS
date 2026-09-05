#!/usr/bin/env python3
"""Write local SOH predictions as uncompressed NPZ archives for .NET interop."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pruned", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.base_pruned, allow_pickle=False) as archive:
        y_true = np.asarray(archive["y_true"], dtype=np.float32)
        y_base = np.asarray(archive["y_base"], dtype=np.float32)
        y_pruned = np.asarray(archive["y_pruned"], dtype=np.float32)
        alignment_start = np.asarray(archive["alignment_start"], dtype=np.int64)

    # The quantized archive contains a complete prediction vector. Its redundant
    # y_true ZIP member may be unavailable after cloud synchronization, so the
    # verified target vector from the same C07 replay is reused here.
    with np.load(args.quantized, allow_pickle=False) as archive:
        y_quant = np.asarray(archive["y_quant"], dtype=np.float32)
        quantized_alignment = np.asarray(
            archive["alignment_start"], dtype=np.int64
        )

    if len(y_quant) != len(y_true):
        raise ValueError(
            f"Prediction lengths differ: quantized={len(y_quant)}, target={len(y_true)}"
        )
    if int(quantized_alignment) != int(alignment_start):
        raise ValueError(
            "Alignment differs between the local model re-executions: "
            f"quantized={int(quantized_alignment)}, "
            f"base/pruned={int(alignment_start)}"
        )
    for name, values in (
        ("y_true", y_true),
        ("y_base", y_base),
        ("y_pruned", y_pruned),
        ("y_quant", y_quant),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    base_pruned_output = args.output_dir / "base_pruned_soh_raw_windows_stored.npz"
    quantized_output = args.output_dir / "quantized_soh_raw_predictions_stored.npz"
    np.savez(
        base_pruned_output,
        y_true=y_true,
        y_base=y_base,
        y_pruned=y_pruned,
        alignment_start=alignment_start,
    )
    np.savez(
        quantized_output,
        y_true=y_true,
        y_quant=y_quant,
        alignment_start=alignment_start,
    )
    print(base_pruned_output)
    print(quantized_output)


if __name__ == "__main__":
    main()
