#!/usr/bin/env python3
"""Plot all available SOH curves and highlight test cells."""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study")
OUT_DIR = ROOT / "6_test" / "CURRENT_MODELS_BASE_VS_OPTIMIZED"
CFG_PATH = ROOT / "2_models" / "CNN" / "Base" / "0.4.2.1_hp" / "config" / "train_soh.yaml"


def normalize_cell_name(raw: str) -> str:
    m = re.search(r"C\d{2}", raw)
    if not m:
        return raw
    return f"MGFarm_18650_{m.group(0)}"


def to_parquet_path(data_root: Path, cell: str) -> Path:
    return data_root / f"df_FE_{cell.split('_')[-1]}.parquet"


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_points:
        return x, y
    step = max(1, len(y) // max_points)
    return x[::step], y[::step]


def main():
    with CFG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["paths"]["data_root"])
    interval_seconds = int(cfg.get("sampling", {}).get("interval_seconds", 3600))
    hours_per_step = float(interval_seconds) / 3600.0 if interval_seconds > 0 else 1.0
    test_cells = {normalize_cell_name(c) for c in cfg["cells"]["test"]}

    parquet_files = sorted(data_root.glob("df_FE_*.parquet"))
    cell_to_path = {}
    for p in parquet_files:
        m = re.search(r"df_FE_(C\d{2}|MGFarm_18650_C\d{2})\.parquet$", p.name)
        if not m:
            continue
        cell_to_path[normalize_cell_name(m.group(1))] = p

    all_cells = sorted(cell_to_path.keys())
    if not all_cells:
        raise RuntimeError(f"No parquet files found in {data_root}")

    test_palette = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    test_color = {}
    for i, c in enumerate(sorted(test_cells)):
        test_color[c] = test_palette[i % len(test_palette)]

    plt.figure(figsize=(15, 7))

    non_test_label_used = False
    for cell in all_cells:
        df = pd.read_parquet(cell_to_path[cell])
        if "SOH" not in df.columns:
            continue
        y = df["SOH"].to_numpy(dtype=np.float32)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            continue
        x_hours = np.arange(len(y), dtype=np.float32) * hours_per_step
        x_hours, y = downsample_xy(x_hours, y, max_points=6000)

        if cell in test_cells:
            plt.plot(
                x_hours,
                y,
                linewidth=1.5,
                alpha=0.95,
                color=test_color[cell],
                label=f"{cell} (test)",
            )
        else:
            plt.plot(
                x_hours,
                y,
                linewidth=1.0,
                alpha=0.45,
                color="#b0b0b0",
                label="other cells" if not non_test_label_used else None,
            )
            non_test_label_used = True

    plt.title("SOH Overview (All Cells) - Test Cells Highlighted")
    plt.xlabel("Time in hours")
    plt.ylabel("SOH")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_path = OUT_DIR / "all_soh_overview_test_highlight.png"
    plt.savefig(out_path, dpi=170)
    plt.close()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
