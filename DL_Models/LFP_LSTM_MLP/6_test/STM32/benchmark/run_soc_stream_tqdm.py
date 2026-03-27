#!/usr/bin/env python3
"""
Wrapper for the SOC quantized streaming tool with a clean tqdm progress bar.

It calls the original tool (stream_parquet_features.py), captures its stdout,
suppresses the per-line prints, and updates a progress bar by parsing lines like:
  IN[954]: ...
and METRICS lines like:
  METRICS: cycles=... us=... E_uJ=...

Usage (same args as the original tool, but parquet is positional at the end):
  python .../run_soc_stream_tqdm.py --port COM7 --yaml ... --delay 0.01 --timeout 2.0 --n 10000 "C:\\...\\df.parquet"
"""
import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[5]
SOC_TOOL = REPO_ROOT / 'STM32' / 'workspace_1.17.0' / 'AI_Project_LSTM_quantized' / 'tools' / 'stream_parquet_features.py'


def main():
    ap = argparse.ArgumentParser(description='Run SOC quantized stream with tqdm progress')
    ap.add_argument('--port', required=True)
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--cols', default='')
    ap.add_argument('--yaml', default='')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--delay', type=float, default=0.01)
    ap.add_argument('--timeout', type=float, default=2.0)
    ap.add_argument('--step', action='store_true')
    ap.add_argument('--echo', action='store_true', help='If set, forward tool output verbosely')
    ap.add_argument('parquet', help='Path to parquet file (positional, like original tool)')
    args = ap.parse_args()

    if not SOC_TOOL.exists():
        print(f'[error] SOC tool not found: {SOC_TOOL}', file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(SOC_TOOL), '--port', args.port, '--baud', str(args.baud)]
    if args.cols:
        cmd += ['--cols', args.cols]
    if args.yaml:
        cmd += ['--yaml', args.yaml]
    if args.start:
        cmd += ['--start', str(args.start)]
    if args.n:
        cmd += ['--n', str(args.n)]
    if args.delay is not None:
        cmd += ['--delay', str(args.delay)]
    if args.timeout is not None:
        cmd += ['--timeout', str(args.timeout)]
    if args.step:
        cmd += ['--step']
    # Only pass --echo if user really wants the verbose output
    if args.echo:
        cmd += ['--echo']
    cmd.append(str(args.parquet))

    t0 = time.perf_counter()
    if args.echo:
        # Verbose mode: just run and stream through
        proc = subprocess.Popen(cmd)
        proc.wait()
        dt = time.perf_counter() - t0
        print(f'[done] duration={dt:.2f}s')
        sys.exit(proc.returncode or 0)

    # Quiet mode: capture stdout and show tqdm by parsing
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    total = args.n if args.n > 0 else None
    pbar = tqdm(total=total, unit='samples', dynamic_ncols=True, desc='SOC')
    last_in_idx = -1
    last_metrics = ''
    try:
        for line in proc.stdout:
            # Try to parse IN[...] lines for progress
            # Example: "IN[954]: Real SOC=..." or similar
            if line.startswith('IN['):
                try:
                    idx_str = line.split('IN[')[1].split(']')[0]
                    cur = int(idx_str)
                    # progress shows processed count (cur+1)
                    if cur + 1 > last_in_idx:
                        last_in_idx = cur + 1
                        pbar.n = last_in_idx
                        pbar.refresh()
                except Exception:
                    pass
                continue
            # Parse METRICS lines to show short status
            if 'METRICS:' in line:
                last_metrics = line.strip().replace('METRICS:', '').strip()
                pbar.set_postfix_str(last_metrics[:60])
                continue
            # Suppress other per-line prints
            # (If needed for debugging, set --echo to see full output)
        proc.wait()
    finally:
        pbar.close()
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print('[error] SOC stream failed', file=sys.stderr)
        sys.exit(proc.returncode)
    print(f'[done] duration={dt:.2f}s | processed={last_in_idx} samples')


if __name__ == '__main__':
    main()

