#!/usr/bin/env python3
"""
Run a single timed STM32 streaming benchmark for SOH.

This script wraps the existing streaming scripts and attaches timing + metadata.
It does NOT modify existing scripts. Output directory is created by the called script.
We detect the newest run folder and write bench_meta.json with duration/throughput.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[5]


def newest_run_dir(root: Path, prefix: str) -> Optional[Path]:
    if not root.exists():
        return None
    cand = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def call_stream_script(model: str, args) -> Path:
    import subprocess

    if model == 'base' or model == 'pruned':
        script = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'STM32' / 'base' / 'run_base_stream_and_plot_soh.py'
        out_root = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'STM32' / 'base'
        prefix = 'STM32_BASE_SOH_STREAM_'
    elif model == 'quantized':
        script = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'STM32' / 'quantized' / 'run_quantized_stream_and_plot_soh.py'
        out_root = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'STM32' / 'quantized'
        prefix = 'STM32_QUANTIZED_SOH_STREAM_'
    else:
        raise ValueError('model must be one of: base, pruned, quantized')

    cmd = [sys.executable, str(script),
           '--port', args.port, '--baud', str(args.baud),
           '--parquet', str(args.parquet), '--yaml', str(args.yaml),
           '--n', str(args.n), '--prime', str(args.prime),
           '--delay', str(args.delay), '--timeout', str(args.timeout)]
    if args.strict:
        cmd.append('--strict_filter')
    if args.ckpt:
        cmd += ['--ckpt', str(args.ckpt)]
    if args.cols:
        cmd += ['--cols', args.cols]

    t0 = time.perf_counter()
    print(f"[run] {model}: launching {script.name}")
    # Stream stdout live and show a lightweight tqdm by parsing progress lines
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    pbar = tqdm(total=args.n, unit='samples', desc=f'{model}', dynamic_ncols=True)
    parsed_total = args.n
    try:
        for line in proc.stdout:
            # Echo underlying script output
            sys.stdout.write(line)
            sys.stdout.flush()
            # Progress lines look like: "500/1488 samples | ..."
            if 'samples' in line and '/' in line:
                try:
                    head = line.strip().split('samples')[0].strip()
                    nums = head.split('/')
                    cur = int(nums[0].strip())
                    tot = int(nums[1].strip())
                    # Update total if script computed different N (safety)
                    if tot != parsed_total:
                        pbar.total = tot
                        parsed_total = tot
                    pbar.n = cur
                    pbar.refresh()
                except Exception:
                    pass
        proc.wait()
    finally:
        pbar.close()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    dt = time.perf_counter() - t0

    # Detect newest run dir
    run_dir = newest_run_dir(out_root, prefix)
    if not run_dir:
        raise RuntimeError(f"Could not locate output folder under {out_root}")

    # Augment metrics with duration/throughput
    metrics_path = run_dir / 'metrics.json'
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = {}
    valid = int(metrics.get('valid', 0))
    throughput = valid / dt if dt > 0 else 0.0

    bench_meta = {
        'model': model,
        'port': args.port,
        'baud': int(args.baud),
        'duration_sec': dt,
        'throughput_samples_per_s': throughput,
        'parquet': str(args.parquet),
        'yaml': str(args.yaml),
        'ckpt': str(args.ckpt or ''),
        'cols': args.cols or '',
    }
    (run_dir / 'bench_meta.json').write_text(json.dumps(bench_meta, indent=2))
    print(f"[done] {model}: duration={dt:.2f}s | throughput={throughput:.1f} samp/s | {run_dir}")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description='Run a timed STM32 SOH stream for base/quantized/pruned')
    ap.add_argument('--model', required=True, help='base|quantized|pruned (or comma-separated list)')
    ap.add_argument('--port', required=True)
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--yaml', required=True)
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--cols', default='')
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--prime', type=int, default=2047)
    ap.add_argument('--delay', type=float, default=0.01)
    ap.add_argument('--timeout', type=float, default=2.5)
    ap.add_argument('--strict', action='store_true')
    args = ap.parse_args()

    models = [m.strip().lower() for m in args.model.split(',') if m.strip()]
    for m in models:
        call_stream_script(m, args)


if __name__ == '__main__':
    main()
