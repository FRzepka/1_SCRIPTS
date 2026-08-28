#!/usr/bin/env python3
"""Run the JES2 nominal SOC benchmark against one flashed STM32 firmware."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import serial


PROTOCOL = "JES2_HW_V1"
MODELS = ("DM", "HDM", "HECM", "DD")
INPUT_COLUMNS = (
    "voltage_v",
    "current_a",
    "temperature_c",
    "soh",
    "q_c_ah",
    "dv_dt_v_s",
    "di_dt_a_s",
    "dt_s",
)
RESULT_COLUMNS = (
    "round",
    "sample_id",
    "segment_id",
    "model",
    "status",
    "soc_device",
    "soc_reference",
    "soc_dataset",
    "error",
    "abs_error",
    "dataset_error",
    "dataset_abs_error",
    "cycles",
    "device_time_us",
    "host_rtt_ms",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p95": percentile(finite, 0.95),
        "maximum": max(finite) if finite else None,
    }


def load_vectors(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "segment_id", "reset", *INPUT_COLUMNS}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing vector columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Test-vector file is empty")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id must be unique within the vector file")
    return rows


def read_nonempty_line(port: serial.Serial, deadline: float) -> str:
    while time.monotonic() < deadline:
        line = port.readline().decode("ascii", errors="replace").strip()
        if line:
            return line
    raise TimeoutError("Timed out waiting for STM32 response")


def wait_ready(port: serial.Serial, timeout_s: float) -> dict[str, str | int]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = read_nonempty_line(port, deadline)
        fields = line.split(",")
        if len(fields) == 5 and fields[0] == "READY":
            protocol, model, revision, clock = fields[1:]
            if protocol != PROTOCOL:
                raise RuntimeError(f"Firmware protocol is {protocol}, expected {PROTOCOL}")
            return {
                "protocol": protocol,
                "model": model,
                "firmware_revision": revision,
                "clock_hz": int(clock),
            }
    raise TimeoutError("Firmware did not emit a READY line")


def reset_device(port: serial.Serial, timeout_s: float) -> None:
    port.write(b"RESET\n")
    port.flush()
    line = read_nonempty_line(port, time.monotonic() + timeout_s)
    if line != "ACK,RESET":
        raise RuntimeError(f"Unexpected RESET response: {line}")


def parse_result(line: str, expected_id: str, expected_model: str) -> dict[str, object]:
    if line.startswith("ERROR,"):
        raise RuntimeError(f"STM32 reported {line}")
    fields = line.split(",")
    if len(fields) != 6 or fields[0] != "RESULT":
        raise RuntimeError(f"Malformed STM32 response: {line}")
    _, sample_id, model, soc_text, cycles_text, status = fields
    if sample_id != expected_id:
        raise RuntimeError(f"Response ID {sample_id} does not match sent ID {expected_id}")
    if model != expected_model:
        raise RuntimeError(f"Firmware returned model {model}, expected {expected_model}")
    if status not in {"OK", "WARMUP"}:
        raise RuntimeError(f"Unexpected result status: {status}")
    soc = None if soc_text.lower() == "nan" else float(soc_text)
    return {"sample_id": sample_id, "model": model, "soc": soc, "cycles": int(cycles_text), "status": status}


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required; install requirements.txt") from exc

    rows = load_vectors(args.vectors)
    expected_column = f"expected_{args.model.lower()}"
    if expected_column not in rows[0]:
        raise ValueError(f"Missing software-reference column: {expected_column}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "measurements.csv"
    started = utc_now()
    measurements: list[dict[str, object]] = []

    with serial.Serial(args.port, args.baud, timeout=0.1) as port:
        time.sleep(args.reset_wait_s)
        port.reset_input_buffer()
        # DTR reset is board/driver dependent, so request an explicit firmware banner.
        port.write(b"HELLO\n")
        port.flush()
        device = wait_ready(port, args.timeout_s)
        if device["model"] != args.model:
            raise RuntimeError(f"Flashed firmware is {device['model']}, requested {args.model}")

        with result_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for round_index in range(1, args.rounds + 1):
                reset_device(port, args.timeout_s)
                for row in rows:
                    if row["reset"].strip().lower() in {"1", "true", "yes"}:
                        reset_device(port, args.timeout_s)
                    payload = ["STEP", row["sample_id"], *(f"{float(row[name]):.9g}" for name in INPUT_COLUMNS)]
                    t0 = time.perf_counter_ns()
                    port.write((",".join(payload) + "\n").encode("ascii"))
                    port.flush()
                    line = read_nonempty_line(port, time.monotonic() + args.timeout_s)
                    t1 = time.perf_counter_ns()
                    parsed = parse_result(line, row["sample_id"], args.model)

                    reference_text = row[expected_column].strip()
                    reference = float(reference_text) if reference_text else None
                    dataset_text = row.get("soc_dataset", "").strip()
                    dataset_soc = float(dataset_text) if dataset_text else None
                    prediction = parsed["soc"]
                    error = prediction - reference if prediction is not None and reference is not None else None
                    dataset_error = prediction - dataset_soc if prediction is not None and dataset_soc is not None else None
                    cycles = int(parsed["cycles"])
                    clock_hz = int(device["clock_hz"])
                    record = {
                        "round": round_index,
                        "sample_id": row["sample_id"],
                        "segment_id": row["segment_id"],
                        "model": args.model,
                        "status": parsed["status"],
                        "soc_device": prediction,
                        "soc_reference": reference,
                        "soc_dataset": dataset_soc,
                        "error": error,
                        "abs_error": abs(error) if error is not None else None,
                        "dataset_error": dataset_error,
                        "dataset_abs_error": abs(dataset_error) if dataset_error is not None else None,
                        "cycles": cycles,
                        "device_time_us": cycles * 1e6 / clock_hz if cycles else None,
                        "host_rtt_ms": (t1 - t0) / 1e6,
                    }
                    measurements.append(record)
                    writer.writerow(record)
                    output.flush()

    valid = [row for row in measurements if row["status"] == "OK"]
    errors = [float(row["error"]) for row in valid if row["error"] is not None]
    dataset_errors = [float(row["dataset_error"]) for row in valid if row["dataset_error"] is not None]
    summary = {
        "schema_version": 1,
        "started_utc": started,
        "finished_utc": utc_now(),
        "model": args.model,
        "port": args.port,
        "baud": args.baud,
        "rounds": args.rounds,
        "vectors": str(args.vectors.resolve()),
        "vectors_sha256": file_sha256(args.vectors),
        "device": device,
        "rows_total": len(measurements),
        "rows_ok": len(valid),
        "rows_warmup": sum(row["status"] == "WARMUP" for row in measurements),
        "device_time_us": numeric_summary(row["device_time_us"] for row in valid if row["device_time_us"] is not None),
        "cycles": numeric_summary(row["cycles"] for row in valid),
        "host_rtt_ms_diagnostic": numeric_summary(row["host_rtt_ms"] for row in valid),
        "reference_difference": {
            "n": len(errors),
            "mae": statistics.fmean(abs(value) for value in errors) if errors else None,
            "rmse": math.sqrt(statistics.fmean(value * value for value in errors)) if errors else None,
            "maximum_absolute_error": max((abs(value) for value in errors), default=None),
        },
        "dataset_difference": {
            "n": len(dataset_errors),
            "mae": statistics.fmean(abs(value) for value in dataset_errors) if dataset_errors else None,
            "rmse": math.sqrt(statistics.fmean(value * value for value in dataset_errors)) if dataset_errors else None,
            "maximum_absolute_error": max((abs(value) for value in dataset_errors), default=None),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Serial port, for example COM7 or /dev/ttyACM0")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--reset-wait-s", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
