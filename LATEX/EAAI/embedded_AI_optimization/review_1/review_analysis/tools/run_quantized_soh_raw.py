#!/usr/bin/env python3
"""Reproduce the missing raw Quantized SOH trajectory on the local PC."""

import argparse
import ctypes
import json
import time
from pathlib import Path

import duckdb
import numpy as np


FEATURES = [
    "Testtime[s]",
    "Voltage[V]",
    "Current[A]",
    "Temperature[\u00b0C]",
    "EFC",
    "Q_c",
]


class LSTMState(ctypes.Structure):
    _fields_ = [("h", ctypes.c_float * 128), ("c", ctypes.c_float * 128)]


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def load_data(parquet_path, limit):
    required = FEATURES + ["SOH"]
    aliases = [f"f{i}" for i in range(len(FEATURES))] + ["target"]
    select = ", ".join(
        f"CAST({quoted(column)} AS FLOAT) AS {alias}"
        for column, alias in zip(required, aliases)
    )
    finite = " AND ".join(f"isfinite({quoted(column)})" for column in required)
    query = f"SELECT {select} FROM read_parquet(?) WHERE {finite}"

    print("Reading the C07 feature sequence in original Parquet row order...", flush=True)
    result = duckdb.connect().execute(query, [str(parquet_path)]).fetchnumpy()
    print("Reproducing pandas.sort_values default quicksort order...", flush=True)
    order = np.argsort(np.asarray(result["f0"]), kind="quicksort")
    if limit > 0:
        order = order[:limit]
    n = len(order)
    x = np.empty((n, len(FEATURES)), dtype=np.float32)
    for i in range(len(FEATURES)):
        x[:, i] = np.asarray(result[f"f{i}"], dtype=np.float32)[order]
    target = np.asarray(result["target"], dtype=np.float32)[order].copy()
    print(f"Loaded {n:,} valid samples.", flush=True)
    return x, target


def run_quantized(dll_path, features, chunk_size):
    dll = ctypes.CDLL(str(dll_path))
    function = dll.lstm_model_soh_int8_inference_batch
    float_pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [
        float_pointer,
        ctypes.POINTER(LSTMState),
        float_pointer,
        ctypes.c_int,
    ]
    function.restype = None

    state = LSTMState()
    ctypes.memset(ctypes.addressof(state), 0, ctypes.sizeof(state))
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
            input_chunk.ctypes.data_as(float_pointer),
            ctypes.byref(state),
            output_pointer,
            end - start,
        )
        fraction = end / len(output)
        if fraction >= next_report or end == len(output):
            elapsed = time.time() - started
            rate = end / max(elapsed, 1e-9)
            remaining = (len(output) - end) / max(rate, 1e-9)
            print(
                f"Inference {100*fraction:5.1f}% | {rate:,.0f} samples/s | "
                f"ETA {remaining/60:.1f} min",
                flush=True,
            )
            next_report += 0.05
    return output


def benchmark_filter(values, rel_cap=1e-4, abs_cap=1e-5, alpha=0.02):
    output = np.empty_like(values, dtype=np.float32)
    last = np.float32(values[0])
    ema = np.float32(values[0])
    output[0] = last
    for i in range(1, len(values)):
        value = np.float32(values[i])
        cap = np.float32(min(abs(float(last)) * rel_cap, abs_cap))
        delta = np.float32(value - last)
        if abs(float(delta)) > float(cap):
            value = np.float32(last + (cap if delta > 0 else -cap))
        ema = np.float32(alpha * value + (1.0 - alpha) * ema)
        output[i] = ema
        last = ema
    return output


def validate_against_saved(raw, target, benchmark_npz):
    with np.load(benchmark_npz) as archive:
        saved_target = np.asarray(archive["y_gt"], dtype=np.float32)
        saved_quantized = np.asarray(archive["C_Quant"], dtype=np.float32)
    n = min(len(raw), len(target), len(saved_target), len(saved_quantized))
    factor = np.float32(target[0] / raw[0]) if raw[0] != 0 else np.float32(1.0)
    reproduced = benchmark_filter(np.asarray(raw[:n] * factor, dtype=np.float32))
    difference = np.asarray(reproduced - saved_quantized[:n], dtype=np.float64)
    target_difference = np.asarray(target[:n] - saved_target[:n], dtype=np.float64)
    return {
        "samples_compared": int(n),
        "calibration_factor": float(factor),
        "target_mae": float(np.mean(np.abs(target_difference))),
        "target_max_abs": float(np.max(np.abs(target_difference))),
        "filtered_output_mae": float(np.mean(np.abs(difference))),
        "filtered_output_max_abs": float(np.max(np.abs(difference))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-npz", type=Path)
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--alignment-start", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=100000)
    args = parser.parse_args()

    features, target = load_data(args.parquet, args.limit)
    raw = run_quantized(args.dll, features, args.chunk_size)
    alignment = min(args.alignment_start, len(raw))
    aligned_raw = np.asarray(raw[alignment:], dtype=np.float32)
    aligned_target = np.asarray(target[alignment:], dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.output}...", flush=True)
    np.savez_compressed(
        args.output,
        y_true=aligned_target,
        y_quant=aligned_raw,
        alignment_start=np.asarray(alignment, dtype=np.int64),
    )

    if args.benchmark_npz:
        validation = validate_against_saved(
            aligned_raw, aligned_target, args.benchmark_npz
        )
        print(json.dumps(validation, indent=2), flush=True)
        if args.validation_json:
            args.validation_json.parent.mkdir(parents=True, exist_ok=True)
            args.validation_json.write_text(
                json.dumps(validation, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
