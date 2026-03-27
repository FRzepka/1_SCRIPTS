#!/usr/bin/env python3
"""
Produce a compact visual report from summary.csv created by aggregate_benchmarks.py.
Generates a few engineer-friendly plots: MAE/RMSE bars, throughput bars, size bars (if map used).
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def plot_bars(ax, labels, values, title, ylabel):
    idx = np.arange(len(labels))
    ax.bar(idx, values, color=['tab:green','tab:orange','tab:red','tab:blue'][:len(labels)])
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description='Plot simple STM32 benchmark report from summary.csv')
    ap.add_argument('--summary', default=str(Path(__file__).with_name('summary.csv')))
    ap.add_argument('--out_dir', default=str(Path(__file__).with_name('report')))
    args = ap.parse_args()

    rows = load_summary(Path(args.summary))
    if not rows:
        print('[warn] No rows in summary.csv')
        return

    # Reduce to last run per model (or all models aggregate)
    # Simple pick: keep the latest line for each model
    by_model: Dict[str, Dict] = {}
    for r in rows:
        by_model[r['model']] = r

    labels = list(by_model.keys())
    mae = [to_float(by_model[m].get('MAE_vs_GT')) for m in labels]
    rmse = [to_float(by_model[m].get('RMSE_vs_GT')) for m in labels]
    thr = [to_float(by_model[m].get('throughput_samples_per_s')) for m in labels]
    flash = [to_float(by_model[m].get('flash_kb')) for m in labels]
    ram = [to_float(by_model[m].get('ram_kb')) for m in labels]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot MAE & RMSE
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    plot_bars(ax[0], labels, mae, 'MAE vs GT', 'MAE')
    plot_bars(ax[1], labels, rmse, 'RMSE vs GT', 'RMSE')
    fig.tight_layout(); fig.savefig(out_dir / 'accuracy.png', dpi=150); plt.close(fig)

    # Throughput
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_bars(ax, labels, thr, 'Throughput (end-to-end)', 'samples/s')
    fig.tight_layout(); fig.savefig(out_dir / 'throughput.png', dpi=150); plt.close(fig)

    # Size (if available)
    if not all(np.isnan(flash)) or not all(np.isnan(ram)):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        plot_bars(ax[0], labels, flash, 'Flash footprint', 'KB')
        plot_bars(ax[1], labels, ram, 'RAM footprint', 'KB')
        fig.tight_layout(); fig.savefig(out_dir / 'footprint.png', dpi=150); plt.close(fig)

    print(f"[done] Plots saved under: {out_dir}")


if __name__ == '__main__':
    main()

