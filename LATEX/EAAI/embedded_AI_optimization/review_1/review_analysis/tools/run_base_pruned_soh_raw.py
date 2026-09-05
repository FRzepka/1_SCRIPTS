#!/usr/bin/env python3
"""Run Base and Pruned SOH C models with the same local PC pipeline."""

import argparse
import ctypes
import time
from pathlib import Path

import numpy as np

from run_quantized_soh_raw import load_data


def make_model_types(hidden_size):
    class State(ctypes.Structure):
        _fields_ = [
            ("h", ctypes.c_float * hidden_size),
            ("c", ctypes.c_float * hidden_size),
        ]

    class Model(ctypes.Structure):
        _fields_ = [("state", State), ("initialized", ctypes.c_int)]

    return Model


def run_model(name, dll_path, hidden_size, features, chunk_size):
    model_type = make_model_types(hidden_size)
    model = model_type()
    model.initialized = 0

    dll = ctypes.CDLL(str(dll_path))
    function = dll.lstm_model_soh_inference_batch
    float_pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [
        ctypes.POINTER(model_type),
        float_pointer,
        float_pointer,
        ctypes.c_int,
    ]
    function.restype = None

    output = np.empty(features.shape[0], dtype=np.float32)
    started = time.time()
    next_report = 0.05
    for start in range(0, len(output), chunk_size):
        end = min(start + chunk_size, len(output))
        input_chunk = np.ascontiguousarray(features[start:end], dtype=np.float32)
        output_pointer = ctypes.cast(
            output.ctypes.data + start * output.itemsize, float_pointer
        )
        function(
            ctypes.byref(model),
            input_chunk.ctypes.data_as(float_pointer),
            output_pointer,
            end - start,
        )
        fraction = end / len(output)
        if fraction >= next_report or end == len(output):
            elapsed = time.time() - started
            rate = end / max(elapsed, 1e-9)
            remaining = (len(output) - end) / max(rate, 1e-9)
            print(
                f"{name} {100*fraction:5.1f}% | {rate:,.0f} samples/s | "
                f"ETA {remaining/60:.1f} min",
                flush=True,
            )
            next_report += 0.05
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--base-dll", type=Path, required=True)
    parser.add_argument("--pruned-dll", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--alignment-start", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=100000)
    args = parser.parse_args()

    features, target = load_data(args.parquet, args.limit)
    base = run_model("Base", args.base_dll, 128, features, args.chunk_size)
    pruned = run_model("Pruned", args.pruned_dll, 90, features, args.chunk_size)

    alignment = min(args.alignment_start, len(target))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.output}...", flush=True)
    np.savez_compressed(
        args.output,
        y_true=np.asarray(target[alignment:], dtype=np.float32),
        y_base=np.asarray(base[alignment:], dtype=np.float32),
        y_pruned=np.asarray(pruned[alignment:], dtype=np.float32),
        alignment_start=np.asarray(alignment, dtype=np.int64),
    )


if __name__ == "__main__":
    main()
