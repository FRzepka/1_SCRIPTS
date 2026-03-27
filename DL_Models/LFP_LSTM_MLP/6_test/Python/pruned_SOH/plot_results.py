import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

def overlay_plot(y_true, y_base, y_pruned, out_path: Path, title: str, max_points: int = 200000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def _ds(arr):
        n = len(arr)
        if n <= max_points:
            return arr, 1
        step = int(np.ceil(n / max_points))
        return arr[::step], step
    yt, st = (None,1)
    if y_true is not None:
        yt, st = _ds(y_true)
    yb, sb = _ds(y_base)
    yp, sp = _ds(y_pruned)
    step_used = max(st, sb, sp)
    plt.figure(figsize=(12,4))
    if yt is not None:
        plt.plot(yt, label='SOH true', linewidth=0.6, alpha=0.6, color='gray')
    plt.plot(yb, label='base', linewidth=1.0, alpha=0.8, color='blue')
    plt.plot(yp, label='pruned', linewidth=1.0, alpha=0.8, color='orange', linestyle='--')
    if step_used > 1:
        plt.title(f"{title} (downsample {step_used}x)")
    else:
        plt.title(title)
    plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()

def diff_hist(diff, out_path: Path, max_samples: int = 200000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(diff) > max_samples:
        idx = np.random.RandomState(0).choice(len(diff), size=max_samples, replace=False)
        diff_plot = diff[idx]
    else:
        diff_plot = diff
    plt.figure(figsize=(6,4))
    plt.hist(diff_plot, bins=120, alpha=0.85, color='tab:purple', edgecolor='black')
    plt.title('Base - Pruned residuals'); plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir', type=str, help='Directory containing arrays.npz')
    args = ap.parse_args()
    
    d = Path(args.dir)
    npz_path = d / 'arrays.npz'
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        return

    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    y_true = data['y_true']
    y_base = data['y_base']
    y_pruned = data['y_pruned']
    
    print("Plotting overlay_full.png...")
    overlay_plot(y_true, y_base, y_pruned, d / 'overlay_full.png', 'Step base vs pruned')
    
    span_len = len(y_true)
    firstN = min(2000, span_len)
    print(f"Plotting overlay_first{firstN}.png...")
    overlay_plot(y_true[:firstN], y_base[:firstN], y_pruned[:firstN], d / f'overlay_first{firstN}.png', f'First {firstN} samples')
    
    print("Plotting diff_hist.png...")
    diff = y_base - y_pruned
    diff_hist(diff, d / 'diff_hist.png')
    
    print("Done.")

if __name__ == '__main__':
    main()
