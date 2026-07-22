#!/usr/bin/env python3
"""Minimal transient input-bitflip recovery test for the six C models.

The test flips one bit in one raw FP32 input feature for exactly one sample.
The recurrent model then receives clean inputs for the configured recovery
horizon. Clean and disturbed runs start from the same copied LSTM state.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np


HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parents[2]
WORKSPACE = MODEL_ROOT.parents[1]
BUILD_DIR = HERE / "build"
RESULTS_DIR = HERE / "results"

DEFAULT_DATA = Path(
    r"C:\Users\flori\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP"
    r"\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet"
)
DEFAULT_ZIG = (
    WORKSPACE
    / "LATEX/EAAI/elsarticle/elsarticle/review_1/review_analysis/runtime"
    / "zig-0.15.2/zig-aarch64-windows-0.15.2/zig.exe"
)


@dataclass(frozen=True)
class ModelSpec:
    task: str
    name: str
    define: str
    hidden_size: int
    source: Path
    include_dir: Path

    @property
    def key(self) -> str:
        return f"{self.task.lower()}_{self.name.lower()}"


MODEL_SPECS = (
    ModelSpec(
        "SOC",
        "Base",
        "RUNNER_SOC_FP32",
        64,
        MODEL_ROOT / "2_models/base/soc_1.5.0.0_base/c_implementation/lstm_model.c",
        MODEL_ROOT / "2_models/base/soc_1.5.0.0_base/c_implementation",
    ),
    ModelSpec(
        "SOC",
        "Pruned",
        "RUNNER_SOC_FP32",
        45,
        MODEL_ROOT
        / "2_models/pruned/soc_1.5.0.0_pruned/prune_30pct_20250916_140404"
        / "c_implementation/lstm_model.c",
        MODEL_ROOT
        / "2_models/pruned/soc_1.5.0.0_pruned/prune_30pct_20250916_140404"
        / "c_implementation",
    ),
    ModelSpec(
        "SOC",
        "Quantized",
        "RUNNER_SOC_INT8",
        64,
        MODEL_ROOT
        / "2_models/quantized/soc_1.5.0.0_quantized"
        / "lstm_model_lstm_int8_fp32mlp.c",
        MODEL_ROOT / "2_models/quantized/soc_1.5.0.0_quantized",
    ),
    ModelSpec(
        "SOH",
        "Base",
        "RUNNER_SOH_FP32",
        128,
        MODEL_ROOT / "2_models/base/soh_2.1.0.0_base/c_implementation/lstm_model_soh.c",
        MODEL_ROOT / "2_models/base/soh_2.1.0.0_base/c_implementation",
    ),
    ModelSpec(
        "SOH",
        "Pruned",
        "RUNNER_SOH_FP32",
        90,
        MODEL_ROOT
        / "2_models/pruned/soh_2.1.0.0/prune_30pct_20251122_010142"
        / "c_implementation/lstm_model_soh.c",
        MODEL_ROOT
        / "2_models/pruned/soh_2.1.0.0/prune_30pct_20251122_010142"
        / "c_implementation",
    ),
    ModelSpec(
        "SOH",
        "Quantized",
        "RUNNER_SOH_INT8",
        128,
        MODEL_ROOT
        / "2_models/quantized/soh_2.1.0.0_quantized/c_implementation"
        / "lstm_model_soh_int8.c",
        MODEL_ROOT / "2_models/quantized/soh_2.1.0.0_quantized/c_implementation",
    ),
)


class Progress:
    def __init__(self, total: int) -> None:
        self.total = max(1, total)
        self.done = 0
        self.started = time.perf_counter()
        self.last_print = 0.0

    def advance(self, count: int = 1, label: str = "") -> None:
        self.done += count
        now = time.perf_counter()
        if self.done < self.total and now - self.last_print < 0.75:
            return
        self.last_print = now
        elapsed = now - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else math.inf
        width = 28
        fraction = min(1.0, self.done / self.total)
        filled = int(round(width * fraction))
        bar = "#" * filled + "-" * (width - filled)
        eta = format_duration(remaining) if math.isfinite(remaining) else "--"
        print(
            f"\r[{bar}] {100.0 * fraction:6.2f}%  "
            f"elapsed {format_duration(elapsed)}  ETA {eta}  {label:24.24s}",
            end="",
            flush=True,
        )
        if self.done >= self.total:
            print()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def newest_source_mtime(spec: ModelSpec) -> float:
    paths = [HERE / "model_runner_wrapper.c", spec.source]
    if spec.key == "soh_quantized":
        paths.append(HERE / "soh_quantized_scaler_compat.h")
    paths.extend(spec.include_dir.glob("*.h"))
    return max(path.stat().st_mtime for path in paths)


def build_models(zig: Path, force: bool) -> dict[str, Path]:
    if not zig.exists():
        raise FileNotFoundError(f"Zig compiler not found: {zig}")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    dlls: dict[str, Path] = {}
    print("Building native model runners:")
    build_start = time.perf_counter()
    for index, spec in enumerate(MODEL_SPECS, start=1):
        dll = BUILD_DIR / f"{spec.key}.dll"
        dlls[spec.key] = dll
        rebuild = force or not dll.exists() or dll.stat().st_mtime < newest_source_mtime(spec)
        status = "compile" if rebuild else "cached"
        print(f"  [{index}/6] {spec.task:3s} {spec.name:9s} {status}", flush=True)
        if not rebuild:
            continue
        command = [
            str(zig),
            "cc",
            "-target",
            "aarch64-windows-gnu",
            "-O2",
            "-shared",
            f"-D{spec.define}",
            "-DINPUT_SIZE=6",
            f"-DHIDDEN_SIZE={spec.hidden_size}",
            "-DMLP_HIDDEN=128" if spec.task == "SOH" else "-DMLP_HIDDEN=64",
        ]
        if spec.key == "soh_quantized":
            command.extend(["-include", str(HERE / "soh_quantized_scaler_compat.h")])
        command.extend(
            [
                str(HERE / "model_runner_wrapper.c"),
                str(spec.source),
                "-I",
                str(spec.include_dir),
                "-o",
                str(dll),
            ]
        )
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Compilation failed for {spec.key}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
    print(f"Native runners ready after {format_duration(time.perf_counter() - build_start)}.")
    return dlls


class ModelRunner:
    def __init__(self, spec: ModelSpec, dll_path: Path) -> None:
        self.spec = spec
        self.lib = ctypes.CDLL(str(dll_path))
        self.lib.runner_state_size.argtypes = []
        self.lib.runner_state_size.restype = ctypes.c_size_t
        self.lib.runner_input_size.argtypes = []
        self.lib.runner_input_size.restype = ctypes.c_int
        self.lib.runner_hidden_size.argtypes = []
        self.lib.runner_hidden_size.restype = ctypes.c_int
        self.lib.runner_init.argtypes = [ctypes.c_void_p]
        self.lib.runner_init.restype = None
        self.lib.runner_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.runner_step.restype = ctypes.c_float
        self.state_size = int(self.lib.runner_state_size())
        if int(self.lib.runner_input_size()) != 6:
            raise RuntimeError(f"Unexpected input size for {spec.key}")
        if int(self.lib.runner_hidden_size()) != spec.hidden_size:
            raise RuntimeError(f"Unexpected hidden size for {spec.key}")

    def new_state(self) -> ctypes.Array:
        state = ctypes.create_string_buffer(self.state_size)
        self.lib.runner_init(state)
        return state

    def copy_state(self, state: ctypes.Array) -> ctypes.Array:
        copied = ctypes.create_string_buffer(self.state_size)
        ctypes.memmove(copied, state, self.state_size)
        return copied

    def step(self, state: ctypes.Array, row: np.ndarray) -> float:
        pointer = row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return float(self.lib.runner_step(state, pointer))


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_data(path: Path, rows: int) -> tuple[dict[str, np.ndarray], str]:
    if not path.exists():
        raise FileNotFoundError(f"Test data not found: {path}")
    connection = duckdb.connect()
    schema = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
    ).fetchall()
    names = [row[0] for row in schema]
    temperature = next((name for name in names if name.startswith("Temperature[")), None)
    if temperature is None:
        raise KeyError("No temperature column found in parquet data")

    columns = [
        "Testtime[s]",
        "Voltage[V]",
        "Current[A]",
        temperature,
        "EFC",
        "Q_c",
        "dU_dt[V/s]",
        "dI_dt[A/s]",
        "SOC",
        "SOH",
    ]
    missing = [column for column in columns if column not in names]
    if missing:
        raise KeyError(f"Missing data columns: {missing}")
    selected = ", ".join(quote_identifier(column) for column in columns)
    finite = " AND ".join(
        f"isfinite({quote_identifier(column)})" for column in columns
    )
    query = f"SELECT {selected} FROM read_parquet(?) WHERE {finite} LIMIT ?"
    frame = connection.execute(query, [str(path), rows]).fetchdf()
    connection.close()
    if len(frame) < rows:
        print(f"Warning: requested {rows} rows, loaded {len(frame)} finite rows.")

    soc_features = ["Voltage[V]", "Current[A]", temperature, "Q_c", "dU_dt[V/s]", "dI_dt[A/s]"]
    soh_features = ["Testtime[s]", "Voltage[V]", "Current[A]", temperature, "EFC", "Q_c"]
    arrays = {
        "SOC_X": np.ascontiguousarray(frame[soc_features].to_numpy(dtype=np.float32)),
        "SOC_y": np.ascontiguousarray(frame["SOC"].to_numpy(dtype=np.float32)),
        "SOH_X": np.ascontiguousarray(frame[soh_features].to_numpy(dtype=np.float32)),
        "SOH_y": np.ascontiguousarray(frame["SOH"].to_numpy(dtype=np.float32)),
    }
    return arrays, temperature


def flip_float32_bit(value: np.float32, bit: int) -> np.float32:
    raw = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    flipped = raw ^ (1 << bit)
    return np.float32(struct.unpack("<f", struct.pack("<I", flipped))[0])


def recovery_time(delta_pp: np.ndarray) -> float | None:
    if not np.all(np.isfinite(delta_pp)):
        return None
    peak_index = int(np.argmax(delta_pp))
    peak = float(delta_pp[peak_index])
    if peak <= 1.0e-6:
        return 0.0
    threshold = max(0.1 * peak, 1.0e-4)
    consecutive = 5
    for start in range(peak_index + 1, len(delta_pp) - consecutive + 1):
        if np.all(delta_pp[start : start + consecutive] <= threshold):
            return float(start)
    return None


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if len(array) else math.nan


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(rows: list[dict], horizon: int) -> list[dict]:
    summary: list[dict] = []
    for task in ("SOC", "SOH"):
        for model in ("Base", "Pruned", "Quantized"):
            group = [row for row in rows if row["task"] == task and row["model"] == model]
            recoveries = [float(row["recovery_s"]) for row in group if row["recovery_s"] != ""]
            summary.append(
                {
                    "task": task,
                    "model": model,
                    "trials": len(group),
                    "median_peak_deviation_pp": percentile((row["peak_deviation_pp"] for row in group), 50),
                    "p95_peak_deviation_pp": percentile((row["peak_deviation_pp"] for row in group), 95),
                    "median_extra_mae_pp": percentile((row["extra_mae_pp"] for row in group), 50),
                    "p95_extra_mae_pp": percentile((row["extra_mae_pp"] for row in group), 95),
                    "median_residual_at_horizon_pp": percentile((row["residual_at_horizon_pp"] for row in group), 50),
                    "p95_residual_at_horizon_pp": percentile((row["residual_at_horizon_pp"] for row in group), 95),
                    "median_recovery_s": percentile(recoveries, 50),
                    "not_recovered_by_horizon_pct": 100.0 * (len(group) - len(recoveries)) / max(1, len(group)),
                    "recovery_horizon_s": horizon,
                }
            )
    return summary


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2.4"/>'


def write_trace_svg(path: Path, trace_rows: list[dict], horizon: int) -> None:
    width, height = 1240, 470
    margin_left, margin_right, margin_top, margin_bottom = 78, 34, 52, 68
    gap = 82
    panel_width = (width - margin_left - margin_right - gap) / 2
    plot_height = height - margin_top - margin_bottom
    colors = {"Base": "#C83E4D", "Pruned": "#2F6F8F", "Quantized": "#6A4C93"}
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222;letter-spacing:0}.axis{stroke:#333;stroke-width:1.2}.grid{stroke:#ddd;stroke-width:1}.tick{font-size:14px}.label{font-size:17px}.panel{font-size:18px;font-weight:bold}.legend{font-size:15px}</style>',
    ]
    for panel_index, task in enumerate(("SOC", "SOH")):
        x0 = margin_left + panel_index * (panel_width + gap)
        y0 = margin_top
        task_rows = [row for row in trace_rows if row["task"] == task]
        ymax = max((float(row["abs_deviation_pp"]) for row in task_rows), default=1.0)
        ymax = max(0.01, ymax * 1.08)
        for tick in range(5):
            value = ymax * tick / 4
            y = y0 + plot_height - plot_height * tick / 4
            chunks.append(f'<line class="grid" x1="{x0:.2f}" y1="{y:.2f}" x2="{x0 + panel_width:.2f}" y2="{y:.2f}"/>')
            chunks.append(f'<text class="tick" x="{x0 - 10:.2f}" y="{y + 5:.2f}" text-anchor="end">{value:.2f}</text>')
        for tick in (0, 15, 30, 45, 60):
            if tick > horizon:
                continue
            x = x0 + panel_width * tick / horizon
            chunks.append(f'<text class="tick" x="{x:.2f}" y="{y0 + plot_height + 25:.2f}" text-anchor="middle">{tick}</text>')
        chunks.append(f'<line class="axis" x1="{x0:.2f}" y1="{y0 + plot_height:.2f}" x2="{x0 + panel_width:.2f}" y2="{y0 + plot_height:.2f}"/>')
        chunks.append(f'<line class="axis" x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y0 + plot_height:.2f}"/>')
        chunks.append(f'<text class="panel" x="{x0:.2f}" y="28">({chr(97 + panel_index)}) {task}</text>')
        chunks.append(f'<text class="label" x="{x0 + panel_width / 2:.2f}" y="{height - 20}" text-anchor="middle">Time after one-sample input bitflip [s]</text>')
        if panel_index == 0:
            chunks.append(f'<text class="label" x="22" y="{y0 + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 22 {y0 + plot_height / 2:.2f})">Absolute disturbed-clean deviation [pp]</text>')
        for model in ("Base", "Pruned", "Quantized"):
            model_rows = sorted(
                (row for row in task_rows if row["model"] == model),
                key=lambda row: int(row["seconds_after_fault"]),
            )
            points = [
                (
                    x0 + panel_width * int(row["seconds_after_fault"]) / horizon,
                    y0 + plot_height - plot_height * float(row["abs_deviation_pp"]) / ymax,
                )
                for row in model_rows
            ]
            chunks.append(svg_polyline(points, colors[model]))
        legend_y = y0 + 12
        legend_x = x0 + panel_width - 235
        for legend_index, model in enumerate(("Base", "Pruned", "Quantized")):
            lx = legend_x + legend_index * 82
            chunks.append(f'<line x1="{lx:.2f}" y1="{legend_y:.2f}" x2="{lx + 24:.2f}" y2="{legend_y:.2f}" stroke="{colors[model]}" stroke-width="2.8"/>')
            chunks.append(f'<text class="legend" x="{lx + 29:.2f}" y="{legend_y + 5:.2f}">{model}</text>')
    chunks.append("</svg>")
    path.write_text("\n".join(chunks), encoding="utf-8")


def run_test(args: argparse.Namespace, dlls: dict[str, Path]) -> Path:
    print(f"Loading {args.rows} finite C07 samples with DuckDB...", flush=True)
    load_start = time.perf_counter()
    arrays, temperature_column = load_data(args.data, args.rows)
    n = min(len(arrays["SOC_y"]), len(arrays["SOH_y"]))
    if n <= args.warmup + args.recovery + 1:
        raise ValueError("Not enough rows for warmup and recovery horizon")
    positions = np.linspace(
        args.warmup,
        n - args.recovery - 1,
        num=args.trials,
        dtype=np.int64,
    )
    positions = np.unique(positions)
    sensor_names = ("Voltage[V]", "Current[A]", temperature_column)
    task_feature_indices = {
        "SOC": {"Voltage[V]": 0, "Current[A]": 1, temperature_column: 2},
        "SOH": {"Voltage[V]": 1, "Current[A]": 2, temperature_column: 3},
    }
    trial_plan = [
        {
            "trial_id": index + 1,
            "position": int(position),
            "feature": sensor_names[index % len(sensor_names)],
        }
        for index, position in enumerate(positions)
    ]
    print(
        f"Loaded {n} samples in {format_duration(time.perf_counter() - load_start)}. "
        f"Trials: {len(trial_plan)}, recovery: {args.recovery} s, bit: {args.bit}."
    )

    total_steps = len(MODEL_SPECS) * (n + len(trial_plan) * (args.recovery + 1))
    progress = Progress(total_steps)
    result_rows: list[dict] = []
    trace_rows: list[dict] = []
    representative = min(
        (trial for trial in trial_plan if trial["feature"] == "Voltage[V]"),
        key=lambda trial: abs(trial["position"] - n / 2),
    )

    for spec in MODEL_SPECS:
        runner = ModelRunner(spec, dlls[spec.key])
        features = arrays[f"{spec.task}_X"]
        target = arrays[f"{spec.task}_y"]
        state = runner.new_state()
        baseline = np.empty(n, dtype=np.float32)
        saved_states: dict[int, bytes] = {}
        planned_positions = {trial["position"] for trial in trial_plan}

        for sample_index in range(n):
            if sample_index in planned_positions:
                saved_states[sample_index] = bytes(state.raw)
            baseline[sample_index] = runner.step(state, features[sample_index])
            progress.advance(1, f"{spec.task} {spec.name} clean")

        for trial in trial_plan:
            position = trial["position"]
            feature = trial["feature"]
            feature_index = task_feature_indices[spec.task][feature]
            branch = ctypes.create_string_buffer(runner.state_size)
            ctypes.memmove(branch, saved_states[position], runner.state_size)
            disturbed = np.empty(args.recovery + 1, dtype=np.float32)
            original_value = np.float32(features[position, feature_index])
            corrupted_value = flip_float32_bit(original_value, args.bit)

            for offset in range(args.recovery + 1):
                input_row = features[position + offset]
                if offset == 0:
                    input_row = input_row.copy()
                    input_row[feature_index] = corrupted_value
                disturbed[offset] = runner.step(branch, input_row)
                progress.advance(1, f"{spec.task} {spec.name} fault")

            clean = baseline[position : position + args.recovery + 1]
            reference = target[position : position + args.recovery + 1]
            delta_pp = np.abs(disturbed.astype(np.float64) - clean.astype(np.float64)) * 100.0
            clean_error = np.abs(clean.astype(np.float64) - reference.astype(np.float64)) * 100.0
            disturbed_error = np.abs(disturbed.astype(np.float64) - reference.astype(np.float64)) * 100.0
            recovered = recovery_time(delta_pp)
            result_rows.append(
                {
                    "task": spec.task,
                    "model": spec.name,
                    "trial_id": trial["trial_id"],
                    "sample_index": position,
                    "feature": feature,
                    "flipped_bit": args.bit,
                    "original_input": float(original_value),
                    "corrupted_input": float(corrupted_value),
                    "peak_deviation_pp": float(np.max(delta_pp)),
                    "clean_window_mae_pp": float(np.mean(clean_error)),
                    "fault_window_mae_pp": float(np.mean(disturbed_error)),
                    "delta_window_mae_pp": float(np.mean(disturbed_error) - np.mean(clean_error)),
                    "extra_mae_pp": float(np.mean(disturbed_error) - np.mean(clean_error)),
                    "residual_at_horizon_pp": float(delta_pp[-1]),
                    "integrated_deviation_pp_s": float(np.sum(delta_pp)),
                    "recovery_s": "" if recovered is None else recovered,
                    "nonfinite_output": int(not np.all(np.isfinite(disturbed))),
                    "clean_out_of_range_output": int(np.any((clean < 0.0) | (clean > 1.0))),
                    "fault_induced_out_of_range_output": int(
                        np.any(
                            ((disturbed < 0.0) | (disturbed > 1.0))
                            & ((clean >= 0.0) & (clean <= 1.0))
                        )
                    ),
                }
            )

            if trial["trial_id"] == representative["trial_id"]:
                for offset, value in enumerate(delta_pp):
                    trace_rows.append(
                        {
                            "task": spec.task,
                            "model": spec.name,
                            "trial_id": trial["trial_id"],
                            "sample_index": position,
                            "feature": feature,
                            "seconds_after_fault": offset,
                            "abs_deviation_pp": float(value),
                            "clean_prediction": float(clean[offset]),
                            "disturbed_prediction": float(disturbed[offset]),
                        }
                    )

    progress.advance(0, "complete")
    summary = aggregate_results(result_rows, args.recovery)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"BITFLIP_INPUT_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trial_results.csv", result_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "representative_trace.csv", trace_rows)
    write_trace_svg(output_dir / "representative_trace.svg", trace_rows, args.recovery)
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "data": str(args.data),
        "samples_loaded": n,
        "warmup_minimum_samples": args.warmup,
        "trials": len(trial_plan),
        "recovery_horizon_seconds": args.recovery,
        "fault": {
            "type": "single-sample FP32 input-buffer bitflip",
            "bit": args.bit,
            "bit_class": "mantissa" if args.bit <= 22 else ("exponent" if args.bit <= 30 else "sign"),
            "features": list(sensor_names),
        },
        "models": [
            {
                "task": spec.task,
                "name": spec.name,
                "hidden_size": spec.hidden_size,
                "source": str(spec.source),
            }
            for spec in MODEL_SPECS
        ],
        "trial_plan": trial_plan,
        "recovery_definition": (
            "First time after the peak at which disturbed-clean absolute deviation "
            "stays below max(10% of peak, 0.0001 pp) for five samples."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nSummary across transient input bitflips:")
    print(
        "Task Model       Peak median/P95 [pp]  Extra MAE median/P95 [pp]  "
        f"Not recovered by {args.recovery}s"
    )
    for row in summary:
        print(
            f"{row['task']:4s} {row['model']:10s} "
            f"{row['median_peak_deviation_pp']:8.4f}/{row['p95_peak_deviation_pp']:8.4f}        "
            f"{row['median_extra_mae_pp']:8.4f}/{row['p95_extra_mae_pp']:8.4f}             "
            f"{row['not_recovered_by_horizon_pct']:6.1f}%"
        )
    print(f"\nResults: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--zig", type=Path, default=DEFAULT_ZIG)
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--recovery", type=int, default=60)
    parser.add_argument(
        "--bit",
        type=int,
        default=22,
        choices=range(32),
        metavar="0..31",
        help="FP32 bit to flip. Default 22 is the most significant mantissa bit.",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Transient input-bitflip recovery benchmark")
    print(f"Python: {sys.version.split()[0]} ({platform.machine()})")
    print(
        "Protocol: one input bitflip for one 1-Hz sample, followed by "
        f"{args.recovery} clean samples."
    )
    if args.rows <= 0 or args.trials <= 0 or args.recovery <= 0 or args.warmup < 0:
        raise ValueError("rows, trials and recovery must be positive; warmup must be non-negative")
    dlls = build_models(args.zig, args.force_rebuild)
    run_test(args, dlls)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
