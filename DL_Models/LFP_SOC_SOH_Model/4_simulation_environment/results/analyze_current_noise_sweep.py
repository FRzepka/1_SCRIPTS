import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
OUT_DIR = RESULTS_DIR / "noise_sweep"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_DIR))

from build_paper_results import MODEL_DIRS
from build_curated_paper_results import MODEL_ORDER, MODEL_META, _load_run_series, _rolling_abs_err


TARGET_LEVELS = [0.02, 0.10, 0.15, 0.20]
CELL = "MGFarm_18650_C07"


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


def _read_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def _latest_baseline(model: str) -> Path:
    runs_root = MODEL_DIRS[model] / "baseline"
    candidates = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = _read_summary(run_dir)
        if summary.get("cell") != CELL:
            continue
        candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No baseline run for {model} {CELL}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_current_noise(model: str, target_std: float) -> Path:
    runs_root = MODEL_DIRS[model] / "current_noise"
    candidates = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = _read_summary(run_dir)
        if summary.get("cell") != CELL:
            continue
        meta = summary.get("scenario_meta", {})
        std = meta.get("current_noise_std")
        if std is None:
            continue
        if abs(float(std) - target_std) < 1e-9:
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No current-noise run for {model} std={target_std}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _noise_local_metrics(model: str, baseline_dir: Path, noise_dir: Path) -> dict:
    base = _rolling_abs_err(_load_run_series(baseline_dir, model), window_s=900)
    noise = _rolling_abs_err(_load_run_series(noise_dir, model), window_s=900)
    merged = noise.merge(base, on="time_s", suffixes=("_noise", "_base"))
    merged = merged.loc[merged["time_s"] >= 900.0].copy()
    merged["excess"] = merged["rolling_mae_noise"] - merged["rolling_mae_base"]
    mid = len(merged) // 2
    first = merged["excess"].iloc[:mid]
    second = merged["excess"].iloc[mid:]
    return {
        "mean_abs_excess": float(merged["excess"].abs().mean()),
        "p95_abs_excess": float(merged["excess"].abs().quantile(0.95)),
        "std_excess": float(merged["excess"].std(ddof=0)),
        "late_minus_early_excess": float(second.mean() - first.mean()) if mid > 0 else float("nan"),
    }


def collect() -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        base_dir = _latest_baseline(model)
        base_summary = _read_summary(base_dir)
        baseline_mae = float(base_summary["mae"])
        baseline_rmse = float(base_summary["rmse"])
        for std in TARGET_LEVELS:
            noise_dir = _latest_current_noise(model, std)
            summary = _read_summary(noise_dir)
            local = _noise_local_metrics(model, base_dir, noise_dir)
            rows.append(
                {
                    "model": model,
                    "class": MODEL_META[model]["short"],
                    "current_noise_std": std,
                    "mae": float(summary["mae"]),
                    "rmse": float(summary["rmse"]),
                    "delta_mae": float(summary["mae"]) - baseline_mae,
                    "delta_rmse": float(summary["rmse"]) - baseline_rmse,
                    "mean_abs_excess": local["mean_abs_excess"],
                    "p95_abs_excess": local["p95_abs_excess"],
                    "std_excess": local["std_excess"],
                    "late_minus_early_excess": local["late_minus_early_excess"],
                    "run_dir": str(noise_dir),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "current_noise_std"]).reset_index(drop=True)


def plot(df: pd.DataFrame) -> None:
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))
    x_levels = TARGET_LEVELS
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("current_noise_std")
        color = MODEL_META[model]["color"]
        label = MODEL_META[model]["short"]
        axes[0, 0].plot(sub["current_noise_std"], sub["delta_mae"], marker="o", lw=2.0, color=color, label=label)
        axes[0, 1].plot(sub["current_noise_std"], sub["delta_rmse"], marker="o", lw=2.0, color=color, label=label)
        axes[1, 0].plot(sub["current_noise_std"], sub["mean_abs_excess"], marker="o", lw=2.0, color=color, label=label)
        axes[1, 1].plot(sub["current_noise_std"], sub["p95_abs_excess"], marker="o", lw=2.0, color=color, label=label)

    axes[0, 0].set_title("Current-noise penalty: global $\\Delta$MAE")
    axes[0, 0].set_ylabel(r"$\Delta$MAE")
    axes[0, 1].set_title("Current-noise penalty: global $\\Delta$RMSE")
    axes[0, 1].set_ylabel(r"$\Delta$RMSE")
    axes[1, 0].set_title("Extra local error: mean |excess rolling MAE|")
    axes[1, 0].set_ylabel("Mean |excess rolling MAE|")
    axes[1, 1].set_title("Extra local error: p95 |excess rolling MAE|")
    axes[1, 1].set_ylabel("p95 |excess rolling MAE|")

    for ax in axes.ravel():
        ax.set_xlabel(r"Current-noise std $\sigma_I$ [A]")
        ax.set_xticks(x_levels)
        ax.legend(frameon=True, ncol=2, loc="upper left")

    fig.suptitle("Current-noise sweep", fontsize=18, y=0.995)
    fig.savefig(OUT_DIR / "current_noise_sweep.png")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT_DIR / "current_noise_sweep.csv", index=False)
    try:
        (OUT_DIR / "current_noise_sweep.md").write_text(df.to_markdown(index=False))
    except Exception:
        (OUT_DIR / "current_noise_sweep.md").write_text(df.to_csv(index=False))
    plot(df)
    note = [
        "# Current-noise sweep notes",
        "",
        "- Global metrics alone can hide zero-mean noise effects.",
        "- Local noise metrics are baseline-referenced:",
        "  - `excess_rolling_mae(t) = rolling_mae(noise_run,t) - rolling_mae(baseline_run,t)`",
        "- Rolling MAE uses 60 s downsampling and a 15-point rolling window (= 15 min).",
        "- `mean_abs_excess` measures the average extra local error level caused by noise.",
        "- `p95_abs_excess` measures the strong-tail local noise sensitivity.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(note))
    print(f"Wrote {OUT_DIR / 'current_noise_sweep.csv'}")
    print(f"Wrote {OUT_DIR / 'current_noise_sweep.png'}")


if __name__ == "__main__":
    main()
