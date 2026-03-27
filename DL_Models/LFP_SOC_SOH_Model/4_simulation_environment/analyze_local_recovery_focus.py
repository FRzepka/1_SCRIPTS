import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
MINIMAL_TAG = "2026-03-12_minimal_matrix_fullc07"
OUT_DIR = ROOT / "analysis_local_focus" / "2026-03-12_local_recovery"
MODELS = [
    "CC_1.0.0",
    "CC_SOH_1.0.0",
    "ECM_0.0.1",
    "SOC_SOH_1.6.0.0_GRU_0.3.1.2",
]
CSV_NAME_CANDIDATES = [
    "soc_cc_fullcell_MGFarm_18650_C07.csv",
    "soc_cc_soh_fullcell_MGFarm_18650_C07.csv",
    "ecm_soc_fullcell_MGFarm_18650_C07.csv",
    "soc_pred_fullcell_MGFarm_18650_C07.csv",
]
PRED_COLS = ["soc_cc", "soc_ecm", "soc_pred"]


def load_minimal_summary() -> pd.DataFrame:
    path = ROOT / "campaigns" / MINIMAL_TAG / "analysis" / "minimal_matrix_summary.csv"
    return pd.read_csv(path)


def find_csv(run_dir: Path) -> Path:
    for name in CSV_NAME_CANDIDATES:
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No prediction csv found in {run_dir}")


def load_summary_and_csv_path(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text())
    csv_path = find_csv(run_dir)
    return summary, csv_path


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    pred_col = next(c for c in PRED_COLS if c in df.columns)
    df = df.rename(columns={pred_col: "soc_pred"})
    if "abs_err" not in df.columns:
        df["abs_err"] = (df["soc_true"] - df["soc_pred"]).abs()
    return df


def read_csv_head(csv_path: Path, nrows: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows)
    return normalize_df(df)


def read_csv_slice(csv_path: Path, row_start: int, row_stop: int) -> pd.DataFrame:
    skip = range(1, row_start + 1)
    nrows = max(0, row_stop - row_start)
    df = pd.read_csv(csv_path, skiprows=skip, nrows=nrows)
    return normalize_df(df)


def read_csv_downsample(csv_path: Path, stride: int = 60, chunk_size: int = 500000) -> pd.DataFrame:
    parts = []
    row0 = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk = normalize_df(chunk)
        idx = np.arange(row0, row0 + len(chunk))
        keep = (idx % stride) == 0
        parts.append(chunk.loc[keep].copy())
        row0 += len(chunk)
    if not parts:
        return pd.DataFrame(columns=["time_s", "soc_true", "soc_pred", "abs_err"])
    return pd.concat(parts, ignore_index=True)


def sustained_recovery_time_seconds(time_s: np.ndarray, abs_err: np.ndarray, threshold: float, sustain_s: float) -> float:
    if len(time_s) == 0:
        return float("nan")
    for i in range(len(time_s)):
        t0 = time_s[i]
        end_t = t0 + sustain_s
        j = np.searchsorted(time_s, end_t, side="right")
        if j <= i:
            continue
        if np.all(abs_err[i:j] <= threshold):
            return float(t0)
    return float("nan")


def spike_recovery_times(time_s: np.ndarray, abs_err: np.ndarray, threshold: float, spike_period_rows: int = 1000, sustain_s: float = 30.0, max_spikes: int = 200):
    recoveries = []
    spike_rows = np.arange(spike_period_rows, len(time_s), spike_period_rows)[:max_spikes]
    for idx in spike_rows:
        t0 = time_s[idx]
        sub_t = time_s[idx:] - t0
        sub_e = abs_err[idx:]
        rec = sustained_recovery_time_seconds(sub_t, sub_e, threshold, sustain_s)
        recoveries.append((idx, t0, rec))
    return recoveries


def rolling_mae(abs_err: np.ndarray, window_pts: int = 15) -> np.ndarray:
    if len(abs_err) == 0:
        return np.array([], dtype=float)
    s = pd.Series(abs_err)
    return s.rolling(window=window_pts, min_periods=1).mean().to_numpy(dtype=float)


def latest_high_noise_dir(model: str) -> Path:
    if model == "SOC_SOH_1.6.0.0_GRU_0.3.1.2":
        base = ROOT / "SOC_SOH_1.6.0.0_0.1.2.3" / "runs" / "current_noise"
    else:
        base = ROOT / model / "runs" / "current_noise"
    dirs = sorted(base.glob("*current_noise_high_0p10"))
    if not dirs:
        raise FileNotFoundError(f"No high-noise run found for {model}")
    return dirs[-1]


def model_runs_base(model: str, scenario: str) -> Path:
    if model == "SOC_SOH_1.6.0.0_GRU_0.3.1.2":
        return ROOT / "SOC_SOH_1.6.0.0_0.1.2.3" / "runs" / scenario
    return ROOT / model / "runs" / scenario


def latest_matching_dir(model: str, scenario: str, pattern: str) -> Path:
    base = model_runs_base(model, scenario)
    dirs = sorted(base.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"No run found for {model} {scenario} matching {pattern}")
    return dirs[-1]


def first_cv_start_time_seconds(cell: str, v_max: float = 3.65, v_tol: float = 0.02, cv_seconds: float = 300.0) -> float:
    cid = cell.split("_")[-1]
    path = DATA_ROOT / f"df_FE_{cid}.parquet"
    if not path.exists():
        path = DATA_ROOT / f"df_FE_{cell}.parquet"
    df = pd.read_parquet(path, columns=["Testtime[s]", "Voltage[V]"])
    high = df["Voltage[V]"].to_numpy(dtype=float) >= (v_max - v_tol)
    min_len = int(cv_seconds)
    start = None
    for i, val in enumerate(high):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if (i - start) >= min_len:
                return float(df["Testtime[s]"].iloc[start])
            start = None
    if start is not None and (len(df) - start) >= min_len:
        return float(df["Testtime[s]"].iloc[start])
    return float("inf")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = load_minimal_summary()
    run_map = {(row["model"], row["scenario"]): Path(row["run_dir"]) for _, row in summary_df.iterrows()}
    local_rows = []

    # 1) Initial SOC error local recovery
    init_models = ["CC_1.0.0", "CC_SOH_1.0.0", "ECM_0.0.1", "SOC_SOH_1.6.0.0_GRU_0.3.1.2"]
    fig, axes = plt.subplots(len(init_models), 1, figsize=(12, 9), sharex=True)
    for ax, model in zip(axes, init_models):
        base_summary, _ = load_summary_and_csv_path(run_map[(model, "baseline")])
        if (model, "initial_soc_error") in run_map:
            init_summary, init_csv = load_summary_and_csv_path(run_map[(model, "initial_soc_error")])
        else:
            init_summary, init_csv = load_summary_and_csv_path(latest_matching_dir(model, "initial_soc_error", "*initial_soc_error*"))
        cv_start_s = first_cv_start_time_seconds(init_summary["cell"])
        eval_end_s = min(float(cv_start_s), 12.0 * 3600.0)
        init_df = read_csv_head(init_csv, int(eval_end_s) + 1)
        init_df = init_df.loc[init_df["time_s"] <= eval_end_s].copy()
        strict_threshold = max(float(base_summary["p95_error"]), float(base_summary["mae"]) * 1.5)
        fair_threshold = max(float(base_summary["p95_error"]) * 2.0, float(base_summary["mae"]) * 3.0, 0.02)
        t_rec_strict = sustained_recovery_time_seconds(
            init_df["time_s"].to_numpy(dtype=float),
            init_df["abs_err"].to_numpy(dtype=float),
            threshold=strict_threshold,
            sustain_s=600.0,
        )
        t_rec_fair = sustained_recovery_time_seconds(
            init_df["time_s"].to_numpy(dtype=float),
            init_df["abs_err"].to_numpy(dtype=float),
            threshold=fair_threshold,
            sustain_s=600.0,
        )
        local_rows.append({
            "model": model,
            "focus_scenario": "initial_soc_error",
            "local_metric": "recovery_time_to_baseline_band_strict_h",
            "value": t_rec_strict / 3600.0 if np.isfinite(t_rec_strict) else np.nan,
            "threshold": strict_threshold,
        })
        local_rows.append({
            "model": model,
            "focus_scenario": "initial_soc_error",
            "local_metric": "recovery_time_to_baseline_band_fair_h",
            "value": t_rec_fair / 3600.0 if np.isfinite(t_rec_fair) else np.nan,
            "threshold": fair_threshold,
        })
        ax.plot(init_df["time_s"] / 3600.0, init_df["abs_err"], label="abs error")
        ax.axhline(strict_threshold, color="tab:red", linestyle="--", label="strict band")
        ax.axhline(fair_threshold, color="tab:orange", linestyle="--", label="fair band")
        if np.isfinite(t_rec_strict):
            ax.axvline(t_rec_strict / 3600.0, color="tab:green", linestyle=":", label="strict recovered")
        if np.isfinite(t_rec_fair):
            ax.axvline(t_rec_fair / 3600.0, color="tab:olive", linestyle=":", label="fair recovered")
        ax.set_title(f"{model} - initial_soc_error")
        ax.set_ylabel("Abs Error")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time [h]")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_initial_soc_error_local_recovery.png", dpi=180)
    plt.close(fig)

    # 2) Spikes local recovery using first 200 spike events max and one representative zoom.
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(12, 11), sharex=True)
    for ax, model in zip(axes, MODELS):
        base_summary, _ = load_summary_and_csv_path(run_map[(model, "baseline")])
        try:
            spike_summary, spike_csv = load_summary_and_csv_path(latest_matching_dir(model, "spikes", "*spikes_high*"))
        except FileNotFoundError:
            spike_summary, spike_csv = load_summary_and_csv_path(run_map[(model, "spikes")])
        threshold = max(float(base_summary["p95_error"]), float(base_summary["mae"]) * 1.5)
        spike_df = read_csv_head(spike_csv, 250000)
        recs = spike_recovery_times(
            spike_df["time_s"].to_numpy(dtype=float),
            spike_df["abs_err"].to_numpy(dtype=float),
            threshold=threshold,
            spike_period_rows=1000,
            sustain_s=30.0,
            max_spikes=200,
        )
        rec_vals = [r[2] for r in recs if np.isfinite(r[2])]
        median_rec = float(np.median(rec_vals)) if rec_vals else float("nan")
        local_rows.append({
            "model": model,
            "focus_scenario": "spikes",
            "local_metric": "median_spike_recovery_time_s",
            "value": median_rec,
            "threshold": threshold,
        })

        ref_idx = 1000 if len(spike_df) > 1240 else max(1, len(spike_df) // 2)
        lo = max(0, ref_idx - 120)
        hi = min(len(spike_df), ref_idx + 240)
        ref_t0 = float(spike_df["time_s"].iloc[ref_idx])
        ax.plot((spike_df["time_s"].iloc[lo:hi] - ref_t0), spike_df["abs_err"].iloc[lo:hi], label="abs error")
        ax.axhline(threshold, color="tab:red", linestyle="--", label="baseline band")
        ax.axvline(0.0, color="tab:purple", linestyle=":", label="spike")
        ax.set_title(f"{model} - spikes (representative spike window)")
        ax.set_ylabel("Abs Error")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time since spike [s]")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_spikes_local_recovery.png", dpi=180)
    plt.close(fig)

    # 3) Current noise local trend. Evaluate excess local error relative to baseline.
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(12, 11), sharex=True)
    for ax, model in zip(axes, MODELS):
        _, base_csv = load_summary_and_csv_path(run_map[(model, "baseline")])
        _, low_csv = load_summary_and_csv_path(run_map[(model, "current_noise")])
        _, high_csv = load_summary_and_csv_path(latest_high_noise_dir(model))
        curves = {}
        for label, csv_path, style in [
            ("baseline", base_csv, "-"),
            ("noise_0.02A", low_csv, "--"),
            ("noise_0.10A", high_csv, "-."),
        ]:
            df_cur = read_csv_downsample(csv_path, stride=60, chunk_size=250000)
            roll = rolling_mae(df_cur["abs_err"].to_numpy(dtype=float), window_pts=15)
            curves[label] = {"df": df_cur, "roll": roll, "style": style}

        base_roll = curves["baseline"]["roll"]
        low_roll = curves["noise_0.02A"]["roll"] - base_roll
        high_roll = curves["noise_0.10A"]["roll"] - base_roll
        time_h = curves["baseline"]["df"]["time_s"].to_numpy(dtype=float) / 3600.0

        keep = curves["baseline"]["df"]["time_s"].to_numpy(dtype=float) >= 900.0
        ax.plot(time_h[keep], low_roll[keep], linestyle="--", label="noise_0.02A - baseline")
        ax.plot(time_h[keep], high_roll[keep], linestyle="-.", label="noise_0.10A - baseline")
        ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.2)

        mid = len(high_roll) // 2
        early = float(np.mean(high_roll[:mid])) if mid > 0 else float("nan")
        late = float(np.mean(high_roll[mid:])) if mid > 0 else float("nan")
        local_rows.append({
            "model": model,
            "focus_scenario": "current_noise_high",
            "local_metric": "late_minus_early_excess_rolling_mae",
            "value": late - early,
            "threshold": np.nan,
        })
        local_rows.append({
            "model": model,
            "focus_scenario": "current_noise_high",
            "local_metric": "mean_excess_rolling_mae_high_noise",
            "value": float(np.mean(high_roll)),
            "threshold": np.nan,
        })
        ax.set_title(f"{model} - excess rolling abs error under current noise")
        ax.set_ylabel("Excess rolling MAE")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time [h]")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_current_noise_local_trend.png", dpi=180)
    plt.close(fig)

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(OUT_DIR / "local_metrics.csv", index=False)
    try:
        md = local_df.to_markdown(index=False)
    except Exception:
        md = local_df.to_csv(index=False)
    (OUT_DIR / "local_metrics.md").write_text(md)

    notes = [
        "# Local Robustness Notes",
        "",
        "- `initial_soc_error`: recovery time is measured in a CV-free evaluation window.",
        "- `initial_soc_error`: `strict` band = `max(p95_error, 1.5 * mae)` and `fair` band = `max(2 * p95_error, 3 * mae, 0.02)`.",
        "- `initial_soc_error`: recovery is the first time the absolute error re-enters the band and stays there for 10 minutes.",
        "- `spikes`: recovery is measured after periodic voltage spikes; the reported value is the median recovery time across the first 200 spikes with a 30 s sustain requirement.",
        "- `current_noise_high`: local trend is assessed via a 15-minute rolling MAE on 60 s downsampled traces.",
        "- `current_noise_high`: the plotted and summarized quantity is `excess_rolling_mae = rolling_mae(noise_run) - rolling_mae(baseline_run)`.",
        "- `current_noise_high`: `late_minus_early_excess_rolling_mae > 0` indicates that the extra local error caused by noise grows over time beyond the baseline trend.",
    ]
    (OUT_DIR / "NOTES.md").write_text("\n".join(notes))

    print(f"Wrote {OUT_DIR / 'local_metrics.csv'}")
    print(f"Wrote {OUT_DIR / '01_initial_soc_error_local_recovery.png'}")
    print(f"Wrote {OUT_DIR / '02_spikes_local_recovery.png'}")
    print(f"Wrote {OUT_DIR / '03_current_noise_local_trend.png'}")


if __name__ == "__main__":
    main()
