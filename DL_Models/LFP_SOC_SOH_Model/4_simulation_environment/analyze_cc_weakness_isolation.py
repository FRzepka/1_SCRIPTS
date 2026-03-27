import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/4_simulation_environment")


def latest_summary(model: str, scenario: str, contains: Optional[str] = None) -> Optional[Path]:
    base = ROOT / model / "runs" / scenario
    if not base.exists():
        return None
    cands = sorted(base.glob("*/summary.json"))
    if contains:
        cands = [p for p in cands if contains in str(p.parent.name)]
    if not cands:
        return None
    return cands[-1]


def csv_from_run_dir(run_dir: Path) -> Path:
    cands = list(run_dir.glob("*.csv"))
    if not cands:
        raise FileNotFoundError(f"No CSV in {run_dir}")
    if len(cands) == 1:
        return cands[0]
    # Prefer SOC prediction CSV over hourly SOH for SOC_SOH model.
    for c in cands:
        if "soc_pred_fullcell" in c.name:
            return c
    return sorted(cands)[0]


def load_soc_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "soc_cc" in df.columns:
        pred = "soc_cc"
    elif "soc_ecm" in df.columns:
        pred = "soc_ecm"
    elif "soc_pred" in df.columns:
        pred = "soc_pred"
    else:
        raise ValueError(f"Could not identify prediction SOC column in {csv_path}")
    need = ["time_s", "soc_true", pred]
    if "abs_err" not in df.columns:
        df["abs_err"] = np.abs(df[pred] - df["soc_true"])
    else:
        need.append("abs_err")
    out = df[["time_s", "soc_true", pred, "abs_err"]].copy()
    out = out.rename(columns={pred: "soc_pred"})
    return out


def compute_gap_bounds(t: np.ndarray, gap_seconds: float) -> Tuple[float, float]:
    t0, t1 = float(t[0]), float(t[-1])
    start = t0 + (t1 - t0 - gap_seconds) * 0.5
    return start, start + gap_seconds


def first_cc_reset_after_gap(cc_df: pd.DataFrame, gap_end_s: float) -> Optional[float]:
    if "q_m_new" not in cc_df.columns:
        return None
    q = cc_df["q_m_new"].to_numpy(dtype=float)
    t = cc_df["time_s"].to_numpy(dtype=float)
    dq = np.diff(q, prepend=q[0])
    idx = np.where((t >= gap_end_s) & (dq > 0.5))[0]
    if len(idx) == 0:
        return None
    return float(t[idx[0]])


def plot_current_offset_soc(models: Dict[str, Dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for name, cfg in models.items():
        if cfg.get("offset_df") is None:
            continue
        df = cfg["offset_df"]
        xh = df["time_s"].to_numpy(dtype=float) / 3600.0
        axes[0].plot(xh, df["soc_pred"], label=name, linewidth=0.9)
        axes[1].plot(xh, df["abs_err"], label=name, linewidth=0.9)
    if models:
        # plot one shared ground truth for readability
        any_df = next((cfg["offset_df"] for cfg in models.values() if cfg.get("offset_df") is not None), None)
        if any_df is not None:
            xh = any_df["time_s"].to_numpy(dtype=float) / 3600.0
            axes[0].plot(xh, any_df["soc_true"], color="black", linewidth=1.1, label="SOC true")
    axes[0].set_title("Current Offset +20mA: SOC trajectories")
    axes[0].set_ylabel("SOC")
    axes[0].legend(loc="best", ncol=2)
    axes[1].set_title("Current Offset +20mA: |error| over time")
    axes[1].set_ylabel("Abs Error")
    axes[1].set_xlabel("Time [h]")
    axes[1].legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "01_current_offset_soc_and_error.png", dpi=180)
    plt.close()


def plot_current_noise_error(models: Dict[str, Dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for name, cfg in models.items():
        if cfg.get("noise_df") is None:
            continue
        df = cfg["noise_df"]
        xh = df["time_s"].to_numpy(dtype=float) / 3600.0
        ax.plot(xh, df["abs_err"], label=name, linewidth=0.9)
    ax.set_title("Current Noise (std=20mA): |error| over time")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Abs Error")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "05_current_noise_abs_error.png", dpi=180)
    plt.close()


def plot_missing_gap_prereset(models: Dict[str, Dict], gap_start: float, gap_end: float, first_reset: float, out_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for name, cfg in models.items():
        df = cfg.get("missing_df")
        if df is None:
            continue
        mask = (df["time_s"] >= gap_end) & (df["time_s"] <= first_reset)
        if not mask.any():
            continue
        xh = (df.loc[mask, "time_s"].to_numpy(dtype=float) - gap_end) / 3600.0
        ax.plot(xh, df.loc[mask, "abs_err"], label=name, linewidth=1.1)
    ax.set_title("Post-gap to first CC-reset: Abs Error (reset-free comparison window)")
    ax.set_xlabel("Hours after gap end")
    ax.set_ylabel("Abs Error")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "02_missing_gap_pre_reset_abs_error.png", dpi=180)
    plt.close()


def plot_cc_reset_vs_noreset(reset_df: pd.DataFrame, noreset_df: pd.DataFrame, gap_end: float, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for lbl, df in [("reset enabled", reset_df), ("reset disabled", noreset_df)]:
        xh = (df["time_s"].to_numpy(dtype=float) - gap_end) / 3600.0
        mask = xh >= 0.0
        ax.plot(xh[mask], df["abs_err"].to_numpy(dtype=float)[mask], label=lbl, linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("Hours after gap end")
    ax.set_ylabel("Abs Error")
    ax.set_xlim(0, 36)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze CC weaknesses vs data-driven models.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "CC_1.0.0": {"dir": "CC_1.0.0"},
        "CC_SOH_1.0.0": {"dir": "CC_SOH_1.0.0"},
        "ECM_0.0.1": {"dir": "ECM_0.0.1"},
        "SOC_SOH_1.6.0.0_GRU_0.3.1.2": {"dir": "SOC_SOH_1.6.0.0_0.1.2.3"},
    }

    rows = []
    for name, cfg in models.items():
        model_dir = cfg["dir"]
        baseline = latest_summary(model_dir, "baseline")
        offset = latest_summary(model_dir, "current_offset", "20mA")
        noise = latest_summary(model_dir, "current_noise", "std20mA")
        missing = latest_summary(model_dir, "missing_gap", "all_sensors_3600")

        cfg["baseline_summary"] = baseline
        cfg["offset_summary"] = offset
        cfg["noise_summary"] = noise
        cfg["missing_summary"] = missing
        cfg["offset_df"] = None
        cfg["noise_df"] = None
        cfg["missing_df"] = None

        b_mae = b_rmse = np.nan
        if baseline and baseline.exists():
            b = json.loads(baseline.read_text())
            b_mae = float(b.get("mae", np.nan))
            b_rmse = float(b.get("rmse", np.nan))

        o_mae = o_rmse = np.nan
        if offset and offset.exists():
            o = json.loads(offset.read_text())
            o_mae = float(o.get("mae", np.nan))
            o_rmse = float(o.get("rmse", np.nan))
            cfg["offset_df"] = load_soc_csv(csv_from_run_dir(offset.parent))

        n_mae = n_rmse = np.nan
        if noise and noise.exists():
            n = json.loads(noise.read_text())
            n_mae = float(n.get("mae", np.nan))
            n_rmse = float(n.get("rmse", np.nan))
            cfg["noise_df"] = load_soc_csv(csv_from_run_dir(noise.parent))

        m_mae = m_rmse = np.nan
        if missing and missing.exists():
            m = json.loads(missing.read_text())
            m_mae = float(m.get("mae", np.nan))
            m_rmse = float(m.get("rmse", np.nan))
            cfg["missing_df"] = load_soc_csv(csv_from_run_dir(missing.parent))
            cfg["missing_gap_seconds"] = float(m.get("missing_gap_seconds", 0.0))

        rows.append(
            {
                "model": name,
                "baseline_mae": b_mae,
                "baseline_rmse": b_rmse,
                "offset20mA_mae": o_mae,
                "offset20mA_rmse": o_rmse,
                "offset20mA_delta_mae": o_mae - b_mae if np.isfinite(o_mae) and np.isfinite(b_mae) else np.nan,
                "noise20mA_mae": n_mae,
                "noise20mA_rmse": n_rmse,
                "noise20mA_delta_mae": n_mae - b_mae if np.isfinite(n_mae) and np.isfinite(b_mae) else np.nan,
                "missinggap_mae": m_mae,
                "missinggap_rmse": m_rmse,
                "missinggap_delta_mae": m_mae - b_mae if np.isfinite(m_mae) and np.isfinite(b_mae) else np.nan,
                "baseline_run": str(baseline.parent) if baseline else "",
                "offset20mA_run": str(offset.parent) if offset else "",
                "noise20mA_run": str(noise.parent) if noise else "",
                "missinggap_run": str(missing.parent) if missing else "",
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "cc_weakness_summary.csv", index=False)

    # Missing-gap recovery in reset-free window until first CC reset
    cc_cfg = models["CC_1.0.0"]
    if cc_cfg.get("missing_summary") and cc_cfg.get("missing_df") is not None:
        cc_run = csv_from_run_dir(cc_cfg["missing_summary"].parent)
        cc_raw = pd.read_csv(cc_run)
        cc_missing = cc_cfg["missing_df"]
        gap_s = float(cc_cfg.get("missing_gap_seconds", 3600.0))
        g0, g1 = compute_gap_bounds(cc_missing["time_s"].to_numpy(dtype=float), gap_s)
        reset_t = first_cc_reset_after_gap(cc_raw, g1)
        if reset_t is not None:
            pre_rows = []
            for name, cfg in models.items():
                df = cfg.get("missing_df")
                if df is None:
                    continue
                mask = (df["time_s"] >= g1) & (df["time_s"] <= reset_t)
                if not mask.any():
                    continue
                pre_rows.append(
                    {
                        "model": name,
                        "mae_postgap_to_first_cc_reset": float(df.loc[mask, "abs_err"].mean()),
                        "window_hours": float((reset_t - g1) / 3600.0),
                    }
                )
            pd.DataFrame(pre_rows).to_csv(out_dir / "missing_gap_pre_reset_window.csv", index=False)
            plot_missing_gap_prereset(models, g0, g1, reset_t, out_dir)

    # CC reset vs no-reset counterfactual plots
    cc_reset = latest_summary("CC_1.0.0", "missing_gap", "all_sensors_3600")
    cc_noreset = latest_summary("CC_1.0.0", "missing_gap", "noreset")
    if cc_reset and cc_noreset:
        df_r = load_soc_csv(csv_from_run_dir(cc_reset.parent))
        df_n = load_soc_csv(csv_from_run_dir(cc_noreset.parent))
        g0, g1 = compute_gap_bounds(df_r["time_s"].to_numpy(dtype=float), 3600.0)
        plot_cc_reset_vs_noreset(
            df_r,
            df_n,
            g1,
            out_dir / "03_cc_reset_vs_noreset.png",
            "CC_1.0.0: Missing gap recovery (reset enabled vs disabled)",
        )

    ccsoh_reset = latest_summary("CC_SOH_1.0.0", "missing_gap", "all_sensors_3600")
    ccsoh_noreset = latest_summary("CC_SOH_1.0.0", "missing_gap", "noreset")
    if ccsoh_reset and ccsoh_noreset:
        df_r = load_soc_csv(csv_from_run_dir(ccsoh_reset.parent))
        df_n = load_soc_csv(csv_from_run_dir(ccsoh_noreset.parent))
        g0, g1 = compute_gap_bounds(df_r["time_s"].to_numpy(dtype=float), 3600.0)
        plot_cc_reset_vs_noreset(
            df_r,
            df_n,
            g1,
            out_dir / "04_cc_soh_reset_vs_noreset.png",
            "CC_SOH_1.0.0: Missing gap recovery (reset enabled vs disabled)",
        )

    plot_current_offset_soc(models, out_dir)
    plot_current_noise_error(models, out_dir)

    md = []
    md.append("|model|baseline_mae|offset20mA_mae|offset20mA_delta_mae|noise20mA_mae|noise20mA_delta_mae|missinggap_mae|missinggap_delta_mae|")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in table.sort_values("offset20mA_delta_mae", na_position="last").iterrows():
        md.append(
            f"|{r['model']}|{r['baseline_mae']:.6f}|{r['offset20mA_mae']:.6f}|"
            f"{r['offset20mA_delta_mae']:.6f}|{r['noise20mA_mae']:.6f}|{r['noise20mA_delta_mae']:.6f}|"
            f"{r['missinggap_mae']:.6f}|{r['missinggap_delta_mae']:.6f}|"
        )
    md.append("")
    md.append("Interpretation focus:")
    md.append("- `offset20mA_delta_mae` isolates current-integration sensitivity (CC weakness).")
    md.append("- `missinggap_delta_mae` plus reset/no-reset plots isolate dependence on reset events.")
    (out_dir / "CC_WEAKNESS_REPORT.md").write_text("\n".join(md))

    print(f"Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()
