import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
OUT_ROOT = RESULTS_DIR / "bias_percent_0p5_1p5_3p0_v2"
MODEL_RUN_DIRS = {
    "CC_1.0.0": ROOT / "CC_1.0.0" / "runs",
    "CC_SOH_1.0.0": ROOT / "CC_SOH_1.0.0" / "runs",
    "ECM_0.0.3": ROOT / "ECM_0.0.3" / "runs",
    "SOC_SOH_1.7.0.0_0.1.2.3": ROOT / "SOC_SOH_1.7.0.0_0.1.2.3" / "runs",
}

if str(RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_curated_paper_results_v2 import MODEL_ORDER, MODEL_META, _find_pred_col  # noqa: E402
from robustness_common import apply_measurement_scenario, load_cell_dataframe  # noqa: E402


DISPLAY = {
    "baseline": ("0.0%", 0.0),
    "current_bias_0p5pct": ("0.5%", 0.5),
    "current_bias_1p5pct": ("1.5%", 1.5),
    "current_bias_3p0pct": ("3.0%", 3.0),
}


def load_rows(campaign_dir: Path, campaign_overrides: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    rows = []
    campaign_overrides = campaign_overrides or {}
    for model_name, runs_root in MODEL_RUN_DIRS.items():
        if not runs_root.exists():
            continue
        model_tag = campaign_overrides.get(model_name, campaign_dir.name)
        for alias in DISPLAY:
            alias_dir = runs_root / alias
            if not alias_dir.exists():
                continue
            tagged_runs = sorted(
                [p for p in alias_dir.iterdir() if p.is_dir() and model_tag in p.name],
                key=lambda p: p.stat().st_mtime,
            )
            if not tagged_runs:
                continue
            latest = tagged_runs[-1]
            summary_path = latest / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            label, pct = DISPLAY[alias]
            amp = float(summary.get("scenario_meta", {}).get("current_offset_a", 0.0))
            rows.append(
                {
                    "model": summary["model"],
                    "cell": summary["cell"],
                    "alias": alias,
                    "bias_label": label,
                    "bias_pct": pct,
                    "bias_a": amp,
                    "mae": summary["mae"],
                    "rmse": summary["rmse"],
                    "p95_error": summary.get("p95_error"),
                    "run_dir": str(latest),
                    "campaign_tag": model_tag,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["model", "bias_pct"]).reset_index(drop=True)
    base = df.loc[df["alias"] == "baseline", ["model", "mae", "rmse"]].rename(columns={"mae": "baseline_mae", "rmse": "baseline_rmse"})
    df = df.merge(base, on="model", how="left")
    df["delta_mae"] = df["mae"] - df["baseline_mae"]
    df["delta_rmse"] = df["rmse"] - df["baseline_rmse"]
    return df


def load_soc_csv(run_dir: Path) -> pd.DataFrame:
    for p in sorted(run_dir.glob("*.csv")):
        if "soh_hourly" in p.name:
            continue
        df = pd.read_csv(p)
        pred_col = _find_pred_col(df)
        out = df.rename(columns={pred_col: "soc_pred"}).copy()
        return out
    raise FileNotFoundError(f"No SOC csv in {run_dir}")


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "bias_summary.csv", index=False)
    try:
        (out_dir / "bias_summary.md").write_text(df.to_markdown(index=False))
    except Exception:
        (out_dir / "bias_summary.md").write_text(df.to_csv(index=False))


def make_mae_plot(df: pd.DataFrame, out_dir: Path, i_max: float) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("bias_pct")
        ax.plot(sub["bias_pct"], 100.0 * sub["mae"], marker="o", lw=2.2, color=MODEL_META[model]["color"], label=MODEL_META[model]["short"])
    ax.set_xlabel("Current bias [% of |I|max]")
    ax.set_ylabel("MAE [SOC percentage points]")
    ax.set_title(
        "Current-bias sweep: baseline and 0.5 / 1.5 / 3.0 % of |I|max\n"
        f"|I|max = {i_max:.4f} A -> 0.5%={i_max*0.005:.4f} A, 1.5%={i_max*0.015:.4f} A, 3.0%={i_max*0.03:.4f} A"
    )
    ax.legend(frameon=True, ncol=2, loc="upper left")
    fig.savefig(out_dir / "mae_vs_bias_percent.png", dpi=220)
    plt.close(fig)


def make_zoom_plot(df: pd.DataFrame, out_dir: Path, cell: str, i_max: float) -> None:
    raw = load_cell_dataframe("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE", cell)
    if "Testtime[s]" in raw.columns:
        raw["time_s"] = raw["Testtime[s]"].astype(float)
    t0 = float(raw["time_s"].min())
    # first-sequence 30 min zoom
    t_start = t0
    t_end = t0 + 1800.0
    raw_win = raw[(raw["time_s"] >= t_start) & (raw["time_s"] <= t_end)].copy()
    noisy_args = type("Args", (), {
        "seed": 42,
        "current_offset_a": i_max * 0.03,
        "current_offset_pct": None,
        "voltage_offset_v": None,
        "temp_offset_c": None,
        "current_noise_std": None,
        "voltage_noise_std": None,
        "temp_noise_std": None,
        "temp_constant": None,
        "quantize_current_a": None,
        "quantize_voltage_v": None,
        "quantize_temp_c": None,
        "spike_channel": "Voltage[V]",
        "spike_magnitude": None,
        "spike_period": None,
        "spike_prob": None,
        "soc_init_error": 0.0,
        "missing_gap_seconds": 0.0,
        "missing_samples_every": None,
        "missing_samples_pct": None,
        "irregular_dt_jitter": None,
        "downsample_hz": None,
        "drop_pct": None,
        "drop_segment_len": None,
    })()
    raw_biased, _ = apply_measurement_scenario(raw, "current_offset", noisy_args)
    raw_biased = raw_biased[(raw_biased["time_s"] >= t_start) & (raw_biased["time_s"] <= t_end)].copy()

    fig = plt.figure(figsize=(12.8, 10.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.30, wspace=0.22)
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot((raw_win["time_s"] - t_start) / 60.0, raw_win["Current[A]"], color="#222222", lw=1.6, label="Current baseline")
    ax_top.plot((raw_biased["time_s"] - t_start) / 60.0, raw_biased["Current[A]"], color="#cc4c02", lw=1.2, alpha=0.9, label=f"Current with +3.0% bias ({i_max*0.03:.4f} A)")
    ax_top.set_title("30 min input zoom: baseline current vs +3.0% current bias")
    ax_top.set_ylabel("Current [A]")
    ax_top.set_xlabel("Time in 30 min window [min]")
    ax_top.legend(frameon=True, loc="upper right")

    for ax, model in zip(
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
        MODEL_ORDER,
    ):
        base_row = df[(df["model"] == model) & (df["alias"] == "baseline")].iloc[0]
        bias_row = df[(df["model"] == model) & (df["alias"] == "current_bias_3p0pct")].iloc[0]
        base_soc = load_soc_csv(Path(base_row["run_dir"]))
        bias_soc = load_soc_csv(Path(bias_row["run_dir"]))
        base_soc = base_soc[(base_soc["time_s"] >= t_start) & (base_soc["time_s"] <= t_end)]
        bias_soc = bias_soc[(bias_soc["time_s"] >= t_start) & (bias_soc["time_s"] <= t_end)]
        x_min = (base_soc["time_s"] - t_start) / 60.0
        ax.plot(x_min, base_soc["soc_true"], color="#111111", lw=1.1, label="SOC true")
        ax.plot(x_min, base_soc["soc_pred"], color=MODEL_META[model]["color"], lw=1.0, alpha=0.85, label="baseline pred")
        ax.plot((bias_soc["time_s"] - t_start) / 60.0, bias_soc["soc_pred"], color="#cc4c02", lw=1.0, alpha=0.85, label="+3.0% bias pred")
        ax.set_title(MODEL_META[model]["short"])
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("SOC [-]")
        ax.legend(frameon=True, fontsize=8, loc="best")

    fig.suptitle(
        "30 min zoom under current bias\n"
        f"|I|max = {i_max:.4f} A, bias levels: 0.5%={i_max*0.005:.4f} A, 1.5%={i_max*0.015:.4f} A, 3.0%={i_max*0.03:.4f} A",
        fontsize=16,
        y=0.995,
    )
    fig.savefig(out_dir / "zoom_30min_current_and_soc_3p0pct.png", dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign_tag", required=True)
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--ecm_campaign_tag", default=None)
    args = ap.parse_args()

    campaign_dir = ROOT / "campaigns" / args.campaign_tag
    out_dir = OUT_ROOT / args.campaign_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    if args.ecm_campaign_tag:
        overrides["ECM_0.0.3"] = args.ecm_campaign_tag
    df = load_rows(campaign_dir, overrides)
    if df.empty:
        raise SystemExit(f"No completed runs found in {campaign_dir}")
    i_max = float(load_cell_dataframe("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE", args.cell)["Current[A]"].abs().max())
    write_tables(df, out_dir)
    make_mae_plot(df, out_dir, i_max)
    make_zoom_plot(df, out_dir, args.cell, i_max)
    summary = {
        "campaign_tag": args.campaign_tag,
        "cell": args.cell,
        "max_abs_current_a": i_max,
        "bias_levels_pct": [0.5, 1.5, 3.0],
        "bias_levels_a": [i_max * 0.005, i_max * 0.015, i_max * 0.03],
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
