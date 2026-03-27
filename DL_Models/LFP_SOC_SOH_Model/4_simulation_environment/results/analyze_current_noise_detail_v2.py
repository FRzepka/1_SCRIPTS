import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
OUT_DIR = RESULTS_DIR / "noise_detail_v2"
ARCHIVE_ROOT = ROOT / "archive"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_DIR))

from build_paper_results_v2 import MODEL_DIRS
from build_curated_paper_results_v2 import MODEL_ORDER, MODEL_META


CELL = "MGFarm_18650_C07"
LEVELS = [0.02, 0.10, 0.15, 0.20]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f7f7",
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.7,
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
            "savefig.dpi": 240,
        }
    )


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def _iter_model_roots(model: str) -> list[Path]:
    active_root = MODEL_DIRS[model]
    root_names = {model, active_root.parent.name}
    roots = [active_root]
    if ARCHIVE_ROOT.exists():
        for name in sorted(root_names):
            roots.extend(sorted(ARCHIVE_ROOT.glob(f"*/runs/{name}")))
            roots.extend(sorted(ARCHIVE_ROOT.glob(f"*/{name}")))
    uniq = []
    seen = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(root)
    return uniq


def _latest_run(model: str, scenario: str, level: Optional[float] = None) -> tuple[Path, dict]:
    scenario_candidates = [scenario]
    if scenario == "current_noise" and level is not None:
        if abs(float(level) - 0.02) < 1e-9:
            scenario_candidates = ["current_noise", "current_noise_low"]
        elif abs(float(level) - 0.10) < 1e-9:
            scenario_candidates = ["current_noise", "current_noise_high"]

    run_roots = _iter_model_roots(model)

    hits = []
    for root in run_roots:
        if not root.exists():
            continue
        for scenario_name in scenario_candidates:
            base = root / scenario_name
            if not base.exists():
                continue
            for run_dir in sorted(base.iterdir()):
                if not run_dir.is_dir():
                    continue
                summary_path = run_dir / "summary.json"
                if not summary_path.exists():
                    continue
                s = _summary(run_dir)
                if s.get("cell") != CELL:
                    continue
                if level is not None:
                    std = s.get("scenario_meta", {}).get("current_noise_std")
                    if std is None or abs(float(std) - float(level)) > 1e-9:
                        continue
                hits.append((run_dir.stat().st_mtime, run_dir, s))
    if not hits:
        raise FileNotFoundError(f"Missing run for {model} / {scenario} / {level}")
    _, run_dir, s = max(hits, key=lambda x: x[0])
    return run_dir, s


def _find_csv(run_dir: Path) -> Path:
    for p in sorted(run_dir.glob("*.csv")):
        if "soh_hourly" in p.name:
            continue
        return p
    raise FileNotFoundError(f"No csv in {run_dir}")


def _load_abs_err_full(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, usecols=["time_s", "abs_err"])


def _load_abs_err(csv_path: Path, stride: int, chunk_size: int = 250000) -> pd.DataFrame:
    parts = []
    row0 = 0
    for ch in pd.read_csv(csv_path, usecols=["time_s", "abs_err"], chunksize=chunk_size):
        idx = np.arange(row0, row0 + len(ch))
        keep = (idx % stride) == 0
        if keep.any():
            parts.append(ch.loc[keep].copy())
        row0 += len(ch)
    if not parts:
        return pd.DataFrame(columns=["time_s", "abs_err"])
    return pd.concat(parts, ignore_index=True)


def _rolling_mean(series: pd.Series, window_pts: int) -> pd.Series:
    return series.rolling(window=window_pts, min_periods=1).mean()


def collect_metrics() -> tuple[pd.DataFrame, dict]:
    rows = []
    full_curves = {}
    for model in MODEL_ORDER:
        base_dir, base_sum = _latest_run(model, "baseline")
        base_df = _load_abs_err(_find_csv(base_dir), stride=60)
        base_df["rolling_mae"] = _rolling_mean(base_df["abs_err"], window_pts=15)
        for level in LEVELS:
            noise_dir, noise_sum = _latest_run(model, "current_noise", level)
            noise_df = _load_abs_err(_find_csv(noise_dir), stride=60)
            noise_df["rolling_mae"] = _rolling_mean(noise_df["abs_err"], window_pts=15)
            merged = noise_df.merge(base_df, on="time_s", suffixes=("_noise", "_base"))
            merged["delta_abs_err"] = merged["abs_err_noise"] - merged["abs_err_base"]
            merged["abs_delta_abs_err"] = merged["delta_abs_err"].abs()
            merged["rolling_mean_delta_abs_err"] = _rolling_mean(merged["delta_abs_err"], window_pts=15)
            merged = merged.loc[merged["time_s"] >= 900.0].copy()
            mid = len(merged) // 2
            rows.append(
                {
                    "model": model,
                    "class": MODEL_META[model]["short"],
                    "current_noise_std": level,
                    "mae": float(noise_sum["mae"]),
                    "delta_mae": float(noise_sum["mae"]) - float(base_sum["mae"]),
                    "rmse": float(noise_sum["rmse"]),
                    "delta_rmse": float(noise_sum["rmse"]) - float(base_sum["rmse"]),
                    "mean_abs_delta_abs_err": float(merged["abs_delta_abs_err"].mean()),
                    "p95_abs_delta_abs_err": float(merged["abs_delta_abs_err"].quantile(0.95)),
                    "p99_abs_delta_abs_err": float(merged["abs_delta_abs_err"].quantile(0.99)),
                    "max_abs_delta_abs_err": float(merged["abs_delta_abs_err"].max()),
                    "std_delta_abs_err": float(merged["delta_abs_err"].std(ddof=0)),
                    "late_minus_early_mean_delta_abs_err": float(
                        merged["rolling_mean_delta_abs_err"].iloc[mid:].mean()
                        - merged["rolling_mean_delta_abs_err"].iloc[:mid].mean()
                    ) if mid > 0 else float("nan"),
                    "run_dir": str(noise_dir),
                }
            )
            full_curves[(model, level)] = merged[
                [
                    "time_s",
                    "delta_abs_err",
                    "abs_delta_abs_err",
                    "rolling_mean_delta_abs_err",
                    "rolling_mae_noise",
                    "rolling_mae_base",
                ]
            ].copy()
    return pd.DataFrame(rows).sort_values(["model", "current_noise_std"]).reset_index(drop=True), full_curves


def representative_zoom(full_curves: dict) -> tuple[float, float]:
    level = 0.20
    merged_all = None
    for model in MODEL_ORDER:
        cur = full_curves[(model, level)][["time_s", "abs_delta_abs_err"]].rename(columns={"abs_delta_abs_err": model})
        merged_all = cur if merged_all is None else merged_all.merge(cur, on="time_s")

    if merged_all is not None and not merged_all.empty:
        merged_all["mean_abs_delta"] = merged_all[MODEL_ORDER].mean(axis=1)
        idx = int(merged_all["mean_abs_delta"].idxmax())
        t0 = float(merged_all.loc[idx, "time_s"])
        return t0 - 3 * 3600.0, t0 + 3 * 3600.0

    # Fallback: pick the strongest single-model spike when models do not share a common time grid.
    best_model = None
    best_peak = -np.inf
    best_time = 0.0
    for model in MODEL_ORDER:
        cur = full_curves[(model, level)]
        if cur.empty:
            continue
        idx = cur["abs_delta_abs_err"].idxmax()
        peak = float(cur.loc[idx, "abs_delta_abs_err"])
        if peak > best_peak:
            best_peak = peak
            best_model = model
            best_time = float(cur.loc[idx, "time_s"])

    if best_model is None:
        raise ValueError("No current-noise curves available for representative zoom")
    return best_time - 3 * 3600.0, best_time + 3 * 3600.0


def make_figures(df: pd.DataFrame, full_curves: dict) -> None:
    _setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("current_noise_std")
        color = MODEL_META[model]["color"]
        label = MODEL_META[model]["short"]
        axes[0, 0].plot(sub["current_noise_std"], sub["delta_mae"], marker="o", lw=2.2, color=color, label=label)
        axes[0, 1].plot(sub["current_noise_std"], sub["p95_abs_delta_abs_err"], marker="o", lw=2.2, color=color, label=label)
        axes[1, 0].plot(sub["current_noise_std"], sub["max_abs_delta_abs_err"], marker="o", lw=2.2, color=color, label=label)
        axes[1, 1].plot(sub["current_noise_std"], sub["late_minus_early_mean_delta_abs_err"], marker="o", lw=2.2, color=color, label=label)

    axes[0, 0].set_title("Global MAE penalty vs current-noise level")
    axes[0, 0].set_ylabel(r"$\Delta$MAE")
    axes[0, 1].set_title("Local error spikes: p95 |Δ abs error|")
    axes[0, 1].set_ylabel("p95 |Δ abs error|")
    axes[1, 0].set_title("Local error spikes: max |Δ abs error|")
    axes[1, 0].set_ylabel("max |Δ abs error|")
    axes[1, 1].set_title("Drift check: late - early mean(rolling Δ abs error)")
    axes[1, 1].set_ylabel("Late - early mean")

    for ax in axes.ravel():
        ax.set_xlabel(r"Current-noise std $\sigma_I$ [A]")
        ax.set_xticks(LEVELS)
        ax.legend(frameon=True, ncol=2, loc="upper left")

    fig.suptitle("Current-noise sensitivity", fontsize=18, y=0.995)
    fig.savefig(OUT_DIR / "Figure_noise_sweep_metrics.png")
    plt.close(fig)

    z0, z1 = representative_zoom(full_curves)
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.0), sharex=True)
    for model in MODEL_ORDER:
        cur = full_curves[(model, 0.20)]
        sub = cur[(cur["time_s"] >= z0) & (cur["time_s"] <= z1)]
        t_h = (sub["time_s"] - z0) / 3600.0
        axes[0].plot(t_h, sub["delta_abs_err"], color=MODEL_META[model]["color"], lw=2.0, label=MODEL_META[model]["short"])
        axes[1].plot(t_h, sub["rolling_mean_delta_abs_err"], color=MODEL_META[model]["color"], lw=2.0, label=MODEL_META[model]["short"])
    axes[0].axhline(0.0, color="#666666", linestyle=":", linewidth=1.2)
    axes[1].axhline(0.0, color="#666666", linestyle=":", linewidth=1.2)
    axes[0].set_title("Representative local error fluctuations at $\\sigma_I=0.20$ A")
    axes[0].set_ylabel(r"$\Delta$ abs error")
    axes[1].set_title("15-min rolling mean of local error difference at $\\sigma_I=0.20$ A")
    axes[1].set_ylabel(r"Rolling mean $\Delta$ abs error")
    axes[1].set_xlabel("Time inside selected 6 h window [h]")
    axes[0].legend(frameon=True, ncol=2, loc="upper left")
    axes[1].legend(frameon=True, ncol=2, loc="upper left")
    fig.suptitle("Current-noise local behaviour", fontsize=18, y=0.995)
    fig.savefig(OUT_DIR / "Figure_noise_local_window.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.8, 4.8))
    for model in MODEL_ORDER:
        cur = full_curves[(model, 0.20)]
        t_h = cur["time_s"] / 3600.0
        ax.plot(t_h, cur["rolling_mean_delta_abs_err"], color=MODEL_META[model]["color"], lw=2.0, label=MODEL_META[model]["short"])
    ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.2)
    ax.set_title("Full-run drift check under strong current noise ($\\sigma_I=0.20$ A)")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(r"15-min rolling mean $\Delta$ abs error")
    ax.legend(frameon=True, ncol=2, loc="upper left")
    fig.savefig(OUT_DIR / "Figure_noise_drift_check.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), sharex=True, sharey=False)
    axes = axes.ravel()
    for ax, model in zip(axes, MODEL_ORDER):
        cur = full_curves[(model, 0.20)]
        t_h = cur["time_s"] / 3600.0
        ax.plot(t_h, cur["rolling_mae_base"], color="#666666", lw=2.0, label="baseline rolling MAE")
        ax.plot(t_h, cur["rolling_mae_noise"], color=MODEL_META[model]["color"], lw=2.0, label=r"noise rolling MAE ($\sigma_I=0.20$ A)")
        ax.set_title(f"{MODEL_META[model]['short']}: baseline vs noise")
        ax.set_ylabel("Rolling MAE (15 min)")
        ax.legend(frameon=True, loc="upper left")
    for ax in axes[-2:]:
        ax.set_xlabel("Time [h]")
    fig.suptitle("Full-run rolling MAE: baseline vs current-noise run", fontsize=18, y=0.995)
    fig.savefig(OUT_DIR / "Figure_noise_baseline_vs_noise_fullrun.png")
    plt.close(fig)


def make_fullres_baseline_vs_noise_plot() -> None:
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.8), sharex=False, sharey=False)
    axes = axes.ravel()
    common_warmup_s = 600.0
    roll_window_s = 900.0

    for ax, model in zip(axes, MODEL_ORDER):
        base_dir, base_sum = _latest_run(model, "baseline")
        noise_dir, noise_sum = _latest_run(model, "current_noise", 0.20)
        base_df = _load_abs_err_full(_find_csv(base_dir))
        noise_df = _load_abs_err_full(_find_csv(noise_dir))

        base_df["rolling_mae"] = _rolling_mean(base_df["abs_err"], window_pts=900)
        noise_df["rolling_mae"] = _rolling_mean(noise_df["abs_err"], window_pts=900)
        merged = noise_df.merge(base_df, on="time_s", suffixes=("_noise", "_base"))

        base_t0 = float(base_df["time_s"].min())
        noise_t0 = float(noise_df["time_s"].min())
        warmup_s = max(
            common_warmup_s,
            float(base_sum.get("warmup_seconds") or 0.0),
            float(noise_sum.get("warmup_seconds") or 0.0),
        )
        plot_start_s = max(base_t0, noise_t0, warmup_s) + roll_window_s
        merged = merged.loc[merged["time_s"] >= plot_start_s].copy()

        t_h = merged["time_s"] / 3600.0
        ax.plot(t_h, merged["rolling_mae_base"], color="#666666", lw=1.2, label="baseline rolling MAE")
        ax.plot(
            t_h,
            merged["rolling_mae_noise"],
            color=MODEL_META[model]["color"],
            lw=1.2,
            label=r"noise rolling MAE ($\sigma_I=0.20$ A)",
        )
        ax.set_title(MODEL_META[model]["short"])
        ax.set_ylabel("Rolling MAE (15 min)")
        ax.legend(frameon=True, loc="upper left")

    for ax in axes[-2:]:
        ax.set_xlabel("Time [h]")
    fig.suptitle(
        "Full-run rolling MAE at 1 s resolution: baseline vs current noise\n"
        "(first 15 min after warmup omitted to remove start artifact)",
        fontsize=16,
        y=0.995,
    )
    fig.savefig(OUT_DIR / "Figure_noise_baseline_vs_noise_fullrun_every1s_no_artifact.png")
    plt.close(fig)


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, curves = collect_metrics()
    df.to_csv(OUT_DIR / "current_noise_detail.csv", index=False)
    try:
        (OUT_DIR / "current_noise_detail.md").write_text(df.to_markdown(index=False))
    except Exception:
        (OUT_DIR / "current_noise_detail.md").write_text(df.to_csv(index=False))
    make_figures(df, curves)
    make_fullres_baseline_vs_noise_plot()
    notes = [
        "# Current-noise detail",
        "",
        "- `Δ abs error = abs_err(noise_run) - abs_err(baseline_run)`",
        "- Local spike metrics are based on `|Δ abs error|`.",
        "- Drift check uses a 15-minute rolling mean of `Δ abs error` on 60 s downsampled traces.",
        "- If the full-run rolling mean stays near zero, there is no evidence for a systematic noise-driven drift beyond baseline.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(notes))
    print(f"Wrote {OUT_DIR / 'current_noise_detail.csv'}")
    print(f"Wrote {OUT_DIR / 'Figure_noise_sweep_metrics.png'}")
    print(f"Wrote {OUT_DIR / 'Figure_noise_local_window.png'}")
    print(f"Wrote {OUT_DIR / 'Figure_noise_drift_check.png'}")
    print(f"Wrote {OUT_DIR / 'Figure_noise_baseline_vs_noise_fullrun.png'}")


if __name__ == "__main__":
    main()
