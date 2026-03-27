#!/usr/bin/env python3
"""
Plot COMPLETE PC vs STM32 SOH comparison (all data from 0 to end).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Plot complete PC vs STM32 SOH comparison")
    ap.add_argument("--csv", type=str, required=True, help="Path to pc_vs_stm32_soh_full.csv")
    ap.add_argument("--out-dir", type=str, default="", help="Output directory (default: same as CSV)")
    args = ap.parse_args()
    
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Extract data
    t_s = df["t_s"].values
    t_h = t_s / 3600.0  # Convert to hours
    soh_pc = df["soh_pc"].values if "soh_pc" in df.columns else None
    soh_stm32 = df["soh_stm32"].values if "soh_stm32" in df.columns else None
    soh_true = df["soh_true"].values if "soh_true" in df.columns else None
    
    # Stats
    n_total = len(df)
    n_pc = int(np.sum(np.isfinite(soh_pc))) if soh_pc is not None else 0
    n_stm32 = int(np.sum(np.isfinite(soh_stm32))) if soh_stm32 is not None else 0
    n_true = int(np.sum(np.isfinite(soh_true))) if soh_true is not None else 0
    
    print(f"\nValid data points:")
    print(f"  PC:     {n_pc}/{n_total}")
    print(f"  STM32:  {n_stm32}/{n_total}")
    print(f"  True:   {n_true}/{n_total}")
    
    # Compute metrics where both PC and STM32 are valid
    if soh_pc is not None and soh_stm32 is not None:
        valid_both = np.isfinite(soh_pc) & np.isfinite(soh_stm32)
        if np.sum(valid_both) > 0:
            diff = soh_pc[valid_both] - soh_stm32[valid_both]
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            max_err = float(np.max(np.abs(diff)))
            print(f"\nPC vs STM32 (where both valid):")
            print(f"  MAE:  {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  Max:  {max_err:.6f}")
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Full timeseries (all three)
    ax1 = fig.add_subplot(gs[0, :])
    if soh_true is not None:
        mask_true = np.isfinite(soh_true)
        ax1.plot(t_h[mask_true], soh_true[mask_true], 'b-', alpha=0.6, linewidth=0.8, label='SOH True')
    if soh_pc is not None:
        mask_pc = np.isfinite(soh_pc)
        ax1.plot(t_h[mask_pc], soh_pc[mask_pc], 'g-', alpha=0.7, linewidth=0.9, label='SOH PC (Base)')
    if soh_stm32 is not None:
        mask_stm32 = np.isfinite(soh_stm32)
        ax1.plot(t_h[mask_stm32], soh_stm32[mask_stm32], 'r-', alpha=0.7, linewidth=0.9, label='SOH STM32 (Quantized)')
    ax1.set_xlabel("Time [hours]")
    ax1.set_ylabel("SOH")
    ax1.set_title("Complete SOH Comparison (Full Timeline)")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. PC vs STM32 detail (where both valid)
    ax2 = fig.add_subplot(gs[1, 0])
    if soh_pc is not None and soh_stm32 is not None:
        mask_both = np.isfinite(soh_pc) & np.isfinite(soh_stm32)
        if np.sum(mask_both) > 0:
            ax2.plot(t_h[mask_both], soh_pc[mask_both], 'g-', alpha=0.7, linewidth=0.9, label='PC')
            ax2.plot(t_h[mask_both], soh_stm32[mask_both], 'r-', alpha=0.7, linewidth=0.9, label='STM32')
            ax2.set_xlabel("Time [hours]")
            ax2.set_ylabel("SOH")
            ax2.set_title("PC vs STM32 (Overlap Region)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # 3. Error over time (PC - STM32)
    ax3 = fig.add_subplot(gs[1, 1])
    if soh_pc is not None and soh_stm32 is not None:
        mask_both = np.isfinite(soh_pc) & np.isfinite(soh_stm32)
        if np.sum(mask_both) > 0:
            err = soh_pc[mask_both] - soh_stm32[mask_both]
            ax3.plot(t_h[mask_both], err, 'k-', alpha=0.6, linewidth=0.7)
            ax3.axhline(0, color='r', linestyle='--', linewidth=1)
            ax3.set_xlabel("Time [hours]")
            ax3.set_ylabel("Error (PC - STM32)")
            ax3.set_title(f"PC-STM32 Error Over Time (MAE={mae:.6f})")
            ax3.grid(True, alpha=0.3)
    
    # 4. Scatter: PC vs STM32
    ax4 = fig.add_subplot(gs[2, 0])
    if soh_pc is not None and soh_stm32 is not None:
        mask_both = np.isfinite(soh_pc) & np.isfinite(soh_stm32)
        if np.sum(mask_both) > 0:
            pc_vals = soh_pc[mask_both]
            stm32_vals = soh_stm32[mask_both]
            ax4.scatter(stm32_vals, pc_vals, alpha=0.3, s=1, c='blue')
            lim_min = min(pc_vals.min(), stm32_vals.min())
            lim_max = max(pc_vals.max(), stm32_vals.max())
            ax4.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1, label='Perfect match')
            ax4.set_xlabel("SOH STM32")
            ax4.set_ylabel("SOH PC")
            ax4.set_title("PC vs STM32 Scatter")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_aspect('equal', adjustable='box')
    
    # 5. Scatter: PC vs True
    ax5 = fig.add_subplot(gs[2, 1])
    if soh_pc is not None and soh_true is not None:
        mask_both = np.isfinite(soh_pc) & np.isfinite(soh_true)
        if np.sum(mask_both) > 0:
            pc_vals = soh_pc[mask_both]
            true_vals = soh_true[mask_both]
            ax5.scatter(true_vals, pc_vals, alpha=0.3, s=1, c='green')
            lim_min = min(pc_vals.min(), true_vals.min())
            lim_max = max(pc_vals.max(), true_vals.max())
            ax5.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1, label='Perfect match')
            ax5.set_xlabel("SOH True")
            ax5.set_ylabel("SOH PC")
            ax5.set_title("PC vs True Scatter")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_aspect('equal', adjustable='box')
    
    # 6. Error histogram (PC - STM32)
    ax6 = fig.add_subplot(gs[3, 0])
    if soh_pc is not None and soh_stm32 is not None:
        mask_both = np.isfinite(soh_pc) & np.isfinite(soh_stm32)
        if np.sum(mask_both) > 0:
            err = soh_pc[mask_both] - soh_stm32[mask_both]
            ax6.hist(err, bins=50, alpha=0.7, edgecolor='black')
            ax6.axvline(0, color='r', linestyle='--', linewidth=1)
            ax6.set_xlabel("Error (PC - STM32)")
            ax6.set_ylabel("Count")
            ax6.set_title(f"Error Distribution (RMSE={rmse:.6f})")
            ax6.grid(True, alpha=0.3)
    
    # 7. Scatter: STM32 vs True
    ax7 = fig.add_subplot(gs[3, 1])
    if soh_stm32 is not None and soh_true is not None:
        mask_both = np.isfinite(soh_stm32) & np.isfinite(soh_true)
        if np.sum(mask_both) > 0:
            stm32_vals = soh_stm32[mask_both]
            true_vals = soh_true[mask_both]
            ax7.scatter(true_vals, stm32_vals, alpha=0.3, s=1, c='red')
            lim_min = min(stm32_vals.min(), true_vals.min())
            lim_max = max(stm32_vals.max(), true_vals.max())
            ax7.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1, label='Perfect match')
            ax7.set_xlabel("SOH True")
            ax7.set_ylabel("SOH STM32")
            ax7.set_title("STM32 vs True Scatter")
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            ax7.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f"Complete PC vs STM32 SOH Analysis ({n_total} samples, {t_h[-1]:.1f} hours)", fontsize=14, fontweight='bold')
    
    out_png = out_dir / "pc_vs_stm32_soh_COMPLETE.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_png}")
    
    # Save summary
    summary = {
        "n_total": n_total,
        "duration_hours": float(t_h[-1]),
        "n_pc_valid": n_pc,
        "n_stm32_valid": n_stm32,
        "n_true_valid": n_true,
    }
    
    if soh_pc is not None and soh_stm32 is not None:
        mask_both = np.isfinite(soh_pc) & np.isfinite(soh_stm32)
        if np.sum(mask_both) > 0:
            diff = soh_pc[mask_both] - soh_stm32[mask_both]
            summary["pc_vs_stm32_mae"] = float(np.mean(np.abs(diff)))
            summary["pc_vs_stm32_rmse"] = float(np.sqrt(np.mean(diff**2)))
            summary["pc_vs_stm32_max_error"] = float(np.max(np.abs(diff)))
    
    out_json = out_dir / "pc_vs_stm32_soh_COMPLETE_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {out_json}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
