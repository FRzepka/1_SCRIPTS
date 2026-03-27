#!/usr/bin/env python3
"""
Simple RAM bandwidth benchmark inspired by the STREAM test.
Creates large numpy buffers, measures common memory-bound kernels,
and writes a Markdown report summarizing the results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import textwrap
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

MiB = 1024 ** 2
GiB = 1024 ** 3


def parse_size_token(token: str) -> int:
    """Parse tokens like '256', '512M', '1G', '2GiB' into a byte count."""
    if not token:
        raise ValueError("Empty size token")
    token = token.strip().lower()
    num_part = []
    unit_part = []
    for ch in token:
        if ch.isdigit() or ch == ".":
            num_part.append(ch)
        else:
            unit_part.append(ch)
    if not num_part:
        raise ValueError(f"Invalid size token: {token}")
    unit = "".join(unit_part).strip()
    value = float("".join(num_part))
    multipliers = {
        "": MiB,
        "b": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": MiB,
        "mib": MiB,
        "gb": GiB,
        "gib": GiB,
    }
    if unit not in multipliers:
        raise ValueError(f"Unknown unit '{unit}' in size token '{token}'")
    return int(value * multipliers[unit])


def _cpu_model_name() -> str:
    cpuinfo = "/proc/cpuinfo"
    try:
        with open(cpuinfo, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or "unknown"


def _mem_total_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = float(parts[1])
                        return int(kb * 1024)
    except FileNotFoundError:
        pass
    return None


def gather_env_info() -> Dict[str, Optional[str]]:
    mem_total = _mem_total_bytes()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model_name(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mem_total_bytes": mem_total,
    }


def benchmark_size(
    size_bytes: int,
    dtype: np.dtype,
    iterations: int,
    warmup: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    element_size = dtype.itemsize
    elements = max(1, size_bytes // element_size)
    actual_bytes = elements * element_size

    try:
        a = rng.random(elements, dtype=dtype)
        b = rng.random(elements, dtype=dtype)
        c = np.zeros(elements, dtype=dtype)
        tmp = np.zeros(elements, dtype=dtype)
    except MemoryError as exc:
        return {
            "requested_bytes": size_bytes,
            "size_bytes": actual_bytes,
            "elements": elements,
            "dtype": dtype.name,
            "error": f"Memory allocation failed: {exc}",
        }

    scalar = dtype.type(3.0)

    def op_copy():
        np.copyto(c, a)

    def op_scale():
        np.multiply(a, scalar, out=c)

    def op_add():
        np.add(a, b, out=c)

    def op_triad():
        np.multiply(c, scalar, out=tmp)
        np.add(b, tmp, out=a)

    operations = [
        ("copy", op_copy, 2),
        ("scale", op_scale, 2),
        ("add", op_add, 3),
        ("triad", op_triad, 3),
    ]

    for _ in range(max(0, warmup)):
        for _, fn, _ in operations:
            fn()

    op_results = []
    for name, fn, streams in operations:
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        elapsed = time.perf_counter() - start
        per_iter_bytes = streams * actual_bytes
        bytes_total = per_iter_bytes * iterations
        bandwidth = bytes_total / elapsed if elapsed > 0 else float("nan")
        op_results.append(
            {
                "op": name,
                "streams": streams,
                "seconds": elapsed,
                "bandwidth_bytes_per_s": bandwidth,
                "bytes_per_iteration": per_iter_bytes,
            }
        )

    return {
        "requested_bytes": size_bytes,
        "size_bytes": actual_bytes,
        "elements": elements,
        "dtype": dtype.name,
        "ops": op_results,
    }


def format_bandwidth(op_result: Optional[Dict[str, float]]) -> str:
    if not op_result:
        return "--"
    value = op_result["bandwidth_bytes_per_s"] / GiB
    if np.isnan(value) or value <= 0:
        return "--"
    return f"{value:6.2f}"


def benchmark_file_io(path: str) -> Dict[str, object]:
    import pandas as pd
    import gc

    result: Dict[str, object] = {
        "path": path,
        "exists": os.path.exists(path),
    }
    if not result["exists"]:
        result["error"] = "file does not exist"
        return result

    result["file_size_bytes"] = os.path.getsize(path)
    start = time.perf_counter()
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        result["error"] = f"failed to read parquet: {exc}"
        return result
    load_time = time.perf_counter() - start
    result["load_seconds"] = load_time
    result["rows"] = len(df)
    result["columns"] = list(df.columns)
    result["memory_bytes"] = int(df.memory_usage(deep=True).sum())
    if load_time > 0:
        result["effective_bandwidth_bytes_per_s"] = result["memory_bytes"] / load_time

    gc_start = time.perf_counter()
    del df
    gc.collect()
    result["release_seconds"] = time.perf_counter() - gc_start
    return result


def render_markdown(
    env: Dict[str, Optional[str]],
    args: argparse.Namespace,
    benchmarks: List[Dict[str, object]],
    timestamp: str,
    io_result: Optional[Dict[str, object]] = None,
) -> str:
    lines: List[str] = []
    lines.append("# RAM Bandwidth Benchmark")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- Timestamp: {timestamp}")
    lines.append(f"- Hostname: {env.get('hostname', 'unknown')}")
    lines.append(f"- Platform: {env.get('platform', 'unknown')}")
    lines.append(f"- CPU: {env.get('cpu_model', 'unknown')}")
    mem_total = env.get("mem_total_bytes")
    if isinstance(mem_total, (int, float)):
        lines.append(f"- Installed RAM: {mem_total / GiB:,.2f} GiB")
    lines.append(f"- Python: {env.get('python', 'n/a')}")
    lines.append(f"- NumPy: {env.get('numpy', 'n/a')}")
    lines.append("")

    requested_sizes = ", ".join(args.sizes)
    lines.append("## Configuration")
    lines.append(f"- Requested sizes: {requested_sizes} (per array)")
    lines.append(f"- Dtype: {np.dtype(args.dtype).name}")
    lines.append(f"- Iterations: {args.iterations}")
    lines.append(f"- Warmup rounds: {args.warmup}")
    lines.append("")

    successful = [b for b in benchmarks if "error" not in b]
    failures = [b for b in benchmarks if "error" in b]

    if successful:
        lines.append("## Results (GiB/s)")
        header = "| Size (MiB) | Elements (M) | Copy | Scale | Add | Triad |"
        lines.append(header)
        lines.append("|-----------:|-------------:|-----:|------:|----:|------:|")
        for bench in successful:
            size_mib = bench["size_bytes"] / MiB
            elements_m = bench["elements"] / 1e6
            op_map = {entry["op"]: entry for entry in bench["ops"]}
            row = "| {size:10.0f} | {elems:11.2f} | {copy} | {scale} | {add} | {triad} |".format(
                size=size_mib,
                elems=elements_m,
                copy=format_bandwidth(op_map.get("copy")),
                scale=format_bandwidth(op_map.get("scale")),
                add=format_bandwidth(op_map.get("add")),
                triad=format_bandwidth(op_map.get("triad")),
            )
            lines.append(row)
        lines.append("")

    if failures:
        lines.append("### Sizes skipped")
        for bench in failures:
            request_mib = bench["requested_bytes"] / MiB
            lines.append(f"- {request_mib:,.1f} MiB request -> {bench['error']}")
        lines.append("")

    lines.append("## Raw Data")
    lines.append("```json")
    lines.append(json.dumps(benchmarks, indent=2))
    lines.append("```")
    lines.append("")

    if io_result:
        lines.append("## Sample File I/O")
        lines.append(f"- Path: {io_result['path']}")
        if "error" in io_result:
            lines.append(f"- Error: {io_result['error']}")
        else:
            lines.append(f"- File size: {io_result['file_size_bytes'] / GiB:,.2f} GiB")
            lines.append(f"- Rows: {io_result['rows']:,}")
            lines.append(f"- Columns: {len(io_result['columns'])}")
            lines.append(f"- Load time: {io_result['load_seconds']:.3f} s")
            bw = io_result.get("effective_bandwidth_bytes_per_s")
            if bw:
                lines.append(f"- Effective load BW: {bw / GiB:,.2f} GiB/s")
            lines.append(f"- Release time: {io_result['release_seconds']:.4f} s")
        lines.append("")

    return "\n".join(lines)


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark RAM bandwidth using simple STREAM-like kernels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        default=["256", "512", "1024", "2048"],
        help="Array sizes to test (accepts suffixes like 256, 1G, 512MiB).",
    )
    parser.add_argument("--dtype", type=str, default="float64", help="Floating dtype for the arrays.")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations per kernel.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup rounds before timing.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the Markdown report (defaults to script directory).",
    )
    parser.add_argument("--output-name", type=str, default=None, help="Optional custom report file name.")
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for numpy's PRNG (controls buffer initialization).",
    )
    parser.add_argument(
        "--io-sample",
        type=str,
        default=None,
        help="Optional data file to read once as a practical load example (e.g., parquet).",
    )
    return parser


def main():
    parser = create_argparser()
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(args.seed)

    sizes_bytes = [parse_size_token(token) for token in args.sizes]
    benchmarks = []
    for size in sizes_bytes:
        result = benchmark_size(size, dtype, args.iterations, args.warmup, rng)
        benchmarks.append(result)

    env = gather_env_info()
    io_result = benchmark_file_io(args.io_sample) if args.io_sample else None
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    report = render_markdown(env, args, benchmarks, timestamp, io_result=io_result)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    if args.output_name:
        filename = args.output_name
    else:
        filename = f"ram_benchmark_{timestamp.replace(':', '').replace('-', '')}.md"
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(textwrap.dedent(
        f"""
        [ram-benchmark] Completed.
        Report saved to: {out_path}
        Sizes tested: {', '.join(args.sizes)}
        """
    ).strip())


if __name__ == "__main__":
    main()
