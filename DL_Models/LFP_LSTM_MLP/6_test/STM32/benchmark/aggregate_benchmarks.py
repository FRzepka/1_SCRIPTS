#!/usr/bin/env python3
"""
Aggregate STM32 SOH benchmark runs into a single CSV summary.

Scans under 6_test/STM32/{base,quantized,pruned}/ for run folders, reads metrics.json
and bench_meta.json, and optionally parses .map files to extract Flash/RAM splits.
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[5]


def parse_map_sizes(map_path: Path) -> Dict[str, float]:
    """Parse a GCC .map file and return sizes in KB: {flash_kb, ram_kb, text_kb, rodata_kb, data_kb, bss_kb}.
    Heuristic parser: sums section sizes if recognizable; otherwise returns empty dict.
    """
    if not map_path or not map_path.exists():
        return {}
    text = map_path.read_text(errors='ignore', encoding='utf-8')
    sizes = {'text': 0, 'rodata': 0, 'data': 0, 'bss': 0}
    # Match lines like:  .text          0x08001234       0x1f0 ...
    sec_re = re.compile(r"^\s*\.(text|rodata|data|bss)\s+0x[0-9a-fA-F]+\s+0x([0-9a-fA-F]+)", re.MULTILINE)
    for m in sec_re.finditer(text):
        sec = m.group(1)
        sz = int(m.group(2), 16)
        sizes[sec] += sz
    out: Dict[str, float] = {}
    out['text_kb'] = sizes['text'] / 1024.0
    out['rodata_kb'] = sizes['rodata'] / 1024.0
    out['data_kb'] = sizes['data'] / 1024.0
    out['bss_kb'] = sizes['bss'] / 1024.0
    out['flash_kb'] = (sizes['text'] + sizes['rodata'] + sizes['data']) / 1024.0
    out['ram_kb'] = (sizes['data'] + sizes['bss']) / 1024.0
    return out


def collect_runs(root: Path, prefix: str, label: str) -> Dict[Path, Dict]:
    runs: Dict[Path, Dict] = {}
    if not root.exists():
        return runs
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            row: Dict = {'model': label, 'run_dir': str(d)}
            m = d / 'metrics.json'
            if m.exists():
                try:
                    row.update(json.loads(m.read_text()))
                except Exception:
                    pass
            b = d / 'bench_meta.json'
            if b.exists():
                try:
                    row.update(json.loads(b.read_text()))
                except Exception:
                    pass
            runs[d] = row
    return runs


def main():
    ap = argparse.ArgumentParser(description='Aggregate STM32 benchmark runs to CSV')
    ap.add_argument('--base_map', default='')
    ap.add_argument('--quant_map', default='')
    ap.add_argument('--pruned_map', default='')
    ap.add_argument('--out', default=str(Path(__file__).with_name('summary.csv')))
    args = ap.parse_args()

    # Corrected paths
    base_root = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'STM32' / 'base'
    quant_root = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'STM32' / 'quantized'
    pruned_root = REPO_ROOT / 'DL_Models' / 'LFP_LSTM_MLP' / '6_test' / 'STM32' / 'base'  # pruned labeled runs live under base

    rows = {}
    rows.update(collect_runs(base_root, 'STM32_BASE_SOH_STREAM_', 'base'))
    rows.update(collect_runs(quant_root, 'STM32_QUANTIZED_SOH_STREAM_', 'quantized'))
    # If you label pruned runs by using --model pruned in run_bench_stream_soh.py, bench_meta.json contains model='pruned'
    # We still scan base folder for such runs.

    # Inject map sizes if provided
    sizes_map: Dict[str, Dict] = {}
    if args.base_map:
        sizes_map['base'] = parse_map_sizes(Path(args.base_map))
    if args.quant_map:
        sizes_map['quantized'] = parse_map_sizes(Path(args.quant_map))
    if args.pruned_map:
        sizes_map['pruned'] = parse_map_sizes(Path(args.pruned_map))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'model', 'run_dir', 'N', 'valid', 'timeouts', 'MAE_vs_GT', 'RMSE_vs_GT', 'MAX_vs_GT',
        'duration_sec', 'throughput_samples_per_s', 'port', 'baud', 'prime', 'strict_filter',
        'flash_kb', 'ram_kb', 'text_kb', 'rodata_kb', 'data_kb', 'bss_kb',
    ]
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d, row in sorted(rows.items(), key=lambda kv: kv[0].name):
            model = row.get('model', '')
            if model in sizes_map:
                row.update(sizes_map[model])
            # keep only known fields
            out_row = {k: row.get(k, '') for k in fields}
            w.writerow(out_row)
    print(f"[done] Wrote {out_path}")


if __name__ == '__main__':
    main()


