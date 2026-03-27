#!/usr/bin/env python3
"""
Make a PNG plot from `stm32_hw_c11_with_pc.csv`:
  SOH true vs SOH stm32 vs SOH pc (0.1.2.3)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw-dir", required=True, help="HW run dir, e.g. .../HW_C11_YYYYMMDD_HHMMSS")
    args = ap.parse_args()

    hw_dir = Path(args.hw_dir)
    csv_path = hw_dir / "stm32_hw_c11_with_pc.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    df = pd.read_csv(csv_path)
    t = df["t_s"] if "t_s" in df.columns else range(len(df))

    plt.figure(figsize=(14, 5))
    if "soh_true" in df.columns:
        plt.plot(t, df["soh_true"], label="SOH true", linewidth=0.9, alpha=0.7)
    if "soh_stm32" in df.columns:
        plt.plot(t, df["soh_stm32"], label="SOH stm32", linewidth=0.9, alpha=0.85)
    if "soh_pc" in df.columns:
        plt.plot(t, df["soh_pc"], label="SOH pc (0.1.2.3)", linewidth=1.2, alpha=0.9)

    plt.ylim(0.0, 1.05)
    plt.xlabel("t [s]")
    plt.ylabel("SOH")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best")

    out_path = hw_dir / "stm32_hw_c11_compare.png"
    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot: {out_path}")


if __name__ == "__main__":
    main()

