import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results.build_paper_results_v2 import load_campaign_rows
from results.build_curated_paper_results_v2 import MODEL_META, MODEL_ORDER, _load_run_series, _thin

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "analysis_local_focus" / "2026-03-24_local_recovery_v2"


def sustained_recovery_time_seconds(time_s: np.ndarray, abs_err: np.ndarray, threshold: float, sustain_s: float) -> float:
    if len(time_s) == 0:
        return float("nan")
    for i in range(len(time_s)):
        end_t = time_s[i] + sustain_s
        j = np.searchsorted(time_s, end_t, side="right")
        if j <= i:
            continue
        if np.all(abs_err[i:j] <= threshold):
            return float(time_s[i])
    return float("nan")


def rolling_mae(abs_err: np.ndarray, window_pts: int = 15) -> np.ndarray:
    if len(abs_err) == 0:
        return np.array([], dtype=float)
    return pd.Series(abs_err).rolling(window=window_pts, min_periods=1).mean().to_numpy(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign_tag", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_campaign_rows(ROOT / "campaigns" / args.campaign_tag)
    if df.empty:
        raise SystemExit(f"No completed runs found for {args.campaign_tag}")

    local_rows = []

    # Initial SOC error recovery
    init_models = [m for m in MODEL_ORDER if not df[(df["model"] == m) & (df["alias"] == "initial_soc_error")].empty]
    if init_models:
        fig, axes = plt.subplots(len(init_models), 1, figsize=(12, 3.0 * len(init_models)), sharex=True)
        if len(init_models) == 1:
            axes = [axes]
        for ax, model in zip(axes, init_models):
            base_row = df[(df["model"] == model) & (df["alias"] == "baseline")].iloc[0]
            init_row = df[(df["model"] == model) & (df["alias"] == "initial_soc_error")].iloc[0]
            base_summary = pd.read_json(Path(base_row["run_dir"]) / "summary.json", typ="series")
            init_df = _load_run_series(Path(init_row["run_dir"]), model)
            init_df = init_df[init_df["time_s"] <= init_df["time_s"].iloc[0] + 12.0 * 3600.0].copy()
            strict = max(float(base_summary["p95_error"]), float(base_summary["mae"]) * 1.5)
            fair = max(float(base_summary["p95_error"]) * 2.0, float(base_summary["mae"]) * 3.0, 0.02)
            t_strict = sustained_recovery_time_seconds(init_df["time_s"].to_numpy(float), init_df["abs_err"].to_numpy(float), strict, 600.0)
            t_fair = sustained_recovery_time_seconds(init_df["time_s"].to_numpy(float), init_df["abs_err"].to_numpy(float), fair, 600.0)
            local_rows.extend([
                {"model": model, "focus_scenario": "initial_soc_error", "local_metric": "recovery_time_to_baseline_band_strict_h", "value": t_strict / 3600.0 if np.isfinite(t_strict) else np.nan, "threshold": strict},
                {"model": model, "focus_scenario": "initial_soc_error", "local_metric": "recovery_time_to_baseline_band_fair_h", "value": t_fair / 3600.0 if np.isfinite(t_fair) else np.nan, "threshold": fair},
            ])
            t_h = (init_df["time_s"] - init_df["time_s"].iloc[0]) / 3600.0
            ax.plot(t_h, init_df["abs_err"], color=MODEL_META[model]["color"], label=MODEL_META[model]["short"])
            ax.axhline(strict, color="tab:red", linestyle="--", label="strict band")
            ax.axhline(fair, color="tab:orange", linestyle="--", label="fair band")
            if np.isfinite(t_fair):
                ax.axvline(t_fair / 3600.0, color="tab:green", linestyle=":", label="fair recovered")
            ax.set_ylabel("Abs error")
            ax.set_title(model)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Hours since evaluation start")
        plt.tight_layout()
        plt.savefig(out_dir / "01_initial_soc_error_local_recovery.png", dpi=180)
        plt.close(fig)

    # Spikes local response
    spike_models = [m for m in MODEL_ORDER if not df[(df["model"] == m) & (df["alias"] == "spikes_high")].empty]
    if spike_models:
        fig, axes = plt.subplots(len(spike_models), 1, figsize=(12, 3.0 * len(spike_models)), sharex=True)
        if len(spike_models) == 1:
            axes = [axes]
        for ax, model in zip(axes, spike_models):
            base_row = df[(df["model"] == model) & (df["alias"] == "baseline")].iloc[0]
            spike_row = df[(df["model"] == model) & (df["alias"] == "spikes_high")].iloc[0]
            base_summary = pd.read_json(Path(base_row["run_dir"]) / "summary.json", typ="series")
            spike_df = _load_run_series(Path(spike_row["run_dir"]), model)
            threshold = max(float(base_summary["p95_error"]), float(base_summary["mae"]) * 1.5)
            spike_period_rows = 1000
            recoveries = []
            for idx in np.arange(spike_period_rows, len(spike_df), spike_period_rows)[:200]:
                sub_t = spike_df["time_s"].to_numpy(float)[idx:] - float(spike_df["time_s"].iloc[idx])
                sub_e = spike_df["abs_err"].to_numpy(float)[idx:]
                rec = sustained_recovery_time_seconds(sub_t, sub_e, threshold, 30.0)
                if np.isfinite(rec):
                    recoveries.append(rec)
            local_rows.append({
                "model": model,
                "focus_scenario": "spikes",
                "local_metric": "median_spike_recovery_time_s",
                "value": float(np.median(recoveries)) if recoveries else np.nan,
                "threshold": threshold,
            })
            ref_idx = spike_period_rows if len(spike_df) > spike_period_rows + 240 else max(1, len(spike_df) // 2)
            lo = max(0, ref_idx - 120)
            hi = min(len(spike_df), ref_idx + 240)
            ref_t0 = float(spike_df["time_s"].iloc[ref_idx])
            ax.plot(spike_df["time_s"].iloc[lo:hi] - ref_t0, spike_df["abs_err"].iloc[lo:hi], color=MODEL_META[model]["color"], label=MODEL_META[model]["short"])
            ax.axhline(threshold, color="tab:red", linestyle="--", label="baseline band")
            ax.axvline(0.0, color="tab:purple", linestyle=":", label="spike")
            ax.set_ylabel("Abs error")
            ax.set_title(model)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Time since spike [s]")
        plt.tight_layout()
        plt.savefig(out_dir / "02_spikes_local_recovery.png", dpi=180)
        plt.close(fig)

    # Current-noise local trend
    noise_models = [m for m in MODEL_ORDER if not df[(df["model"] == m) & (df["alias"] == "current_noise_high")].empty]
    if noise_models:
        fig, axes = plt.subplots(len(noise_models), 1, figsize=(12, 3.0 * len(noise_models)), sharex=True)
        if len(noise_models) == 1:
            axes = [axes]
        for ax, model in zip(axes, noise_models):
            base_row = df[(df["model"] == model) & (df["alias"] == "baseline")].iloc[0]
            low_row = df[(df["model"] == model) & (df["alias"] == "current_noise_low")].iloc[0]
            high_row = df[(df["model"] == model) & (df["alias"] == "current_noise_high")].iloc[0]
            base = _thin(_load_run_series(Path(base_row["run_dir"]), model), max_points=5000)
            low = _thin(_load_run_series(Path(low_row["run_dir"]), model), max_points=5000)
            high = _thin(_load_run_series(Path(high_row["run_dir"]), model), max_points=5000)
            common = base[["time_s", "abs_err"]].merge(low[["time_s", "abs_err"]], on="time_s", suffixes=("_base", "_low")).merge(high[["time_s", "abs_err"]], on="time_s")
            common = common.rename(columns={"abs_err": "abs_err_high"})
            base_roll = rolling_mae(common["abs_err_base"].to_numpy(float))
            low_roll = rolling_mae(common["abs_err_low"].to_numpy(float)) - base_roll
            high_roll = rolling_mae(common["abs_err_high"].to_numpy(float)) - base_roll
            time_h = common["time_s"].to_numpy(float) / 3600.0
            keep = common["time_s"].to_numpy(float) >= 900.0
            ax.plot(time_h[keep], low_roll[keep], linestyle="--", label="noise_0.02A - baseline")
            ax.plot(time_h[keep], high_roll[keep], linestyle="-.", label="noise_0.10A - baseline")
            ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.2)
            mid = len(high_roll) // 2
            local_rows.extend([
                {"model": model, "focus_scenario": "current_noise_high", "local_metric": "late_minus_early_excess_rolling_mae", "value": float(np.mean(high_roll[mid:]) - np.mean(high_roll[:mid])) if mid > 0 else np.nan, "threshold": np.nan},
                {"model": model, "focus_scenario": "current_noise_high", "local_metric": "mean_excess_rolling_mae_high_noise", "value": float(np.mean(high_roll)), "threshold": np.nan},
            ])
            ax.set_ylabel("Excess rolling MAE")
            ax.set_title(model)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Time [h]")
        plt.tight_layout()
        plt.savefig(out_dir / "03_current_noise_local_trend.png", dpi=180)
        plt.close(fig)

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(out_dir / "local_metrics.csv", index=False)
    try:
        (out_dir / "local_metrics.md").write_text(local_df.to_markdown(index=False))
    except Exception:
        (out_dir / "local_metrics.md").write_text(local_df.to_csv(index=False))
    print(f"Wrote {out_dir / 'local_metrics.csv'}")


if __name__ == "__main__":
    main()
