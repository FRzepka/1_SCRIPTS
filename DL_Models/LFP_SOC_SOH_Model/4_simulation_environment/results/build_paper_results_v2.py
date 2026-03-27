import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
TABLES_DIR = RESULTS_DIR / "tables_v2"
FIGURES_DIR = RESULTS_DIR / "figures_v2"
LOCAL_ANALYSIS_DIR = ROOT / "analysis_local_focus" / "2026-03-24_local_recovery_v2"
MODEL_DIRS = {
    "CC_1.0.0": ROOT / "CC_1.0.0" / "runs",
    "CC_SOH_1.0.0": ROOT / "CC_SOH_1.0.0" / "runs",
    "ECM_0.0.3": ROOT / "ECM_0.0.3" / "runs",
    "SOC_SOH_1.7.0.0_0.1.2.3": ROOT / "SOC_SOH_1.7.0.0_0.1.2.3" / "runs",
}
EXTRA_RUNS = {
    # V2 runs are expected to include initial_soc_error directly in the campaign.
}


def load_campaign_rows(campaign_dir: Path) -> pd.DataFrame:
    rows = []
    campaign_tag = campaign_dir.name

    archive_runs_root = ROOT / "archive"

    def iter_run_roots(model_name: str):
        active_root = MODEL_DIRS[model_name]
        root_names = {model_name, active_root.parent.name}
        if active_root.exists():
            yield active_root
        if archive_runs_root.exists():
            seen = set()
            for name in sorted(root_names):
                for archived_root in sorted(archive_runs_root.glob(f"*/runs/{name}")):
                    key = str(archived_root.resolve()) if archived_root.exists() else str(archived_root)
                    if key in seen:
                        continue
                    seen.add(key)
                    if archived_root.exists():
                        yield archived_root

    for model_name in MODEL_DIRS:
        for runs_root in iter_run_roots(model_name):
            for alias_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
                tagged_runs = sorted(
                    [p for p in alias_dir.iterdir() if p.is_dir() and campaign_tag in p.name],
                    key=lambda p: p.stat().st_mtime,
                )
                if alias_dir.name in {"baseline", "irregular_sampling"} and not tagged_runs:
                    tagged_runs = sorted(
                        [p for p in alias_dir.iterdir() if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                    )
                if not tagged_runs:
                    continue
                latest = tagged_runs[-1]
                summary_path = latest / "summary.json"
                if not summary_path.exists():
                    continue
                summary = json.loads(summary_path.read_text())
                rows.append({
                    "model": summary.get("model", model_name),
                    "cell": summary["cell"],
                    "alias": alias_dir.name,
                    "scenario": summary.get("scenario", alias_dir.name),
                    "mae": summary.get("mae"),
                    "rmse": summary.get("rmse"),
                    "p95_error": summary.get("p95_error"),
                    "max_error": summary.get("max_error"),
                    "bias": summary.get("bias"),
                    "jump_count_gt_5pct": summary.get("jump_count_gt_5pct"),
                    "recovery_time_h": summary.get("recovery_time_h"),
                    "run_dir": str(latest),
                    "run_mtime": latest.stat().st_mtime,
                })

    for (model_name, alias), run_dir in EXTRA_RUNS.items():
        if not run_dir.exists():
            continue
        already = any(r["model"] == model_name and r["alias"] == alias for r in rows)
        if already:
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        rows.append({
            "model": summary.get("model", model_name),
            "cell": summary["cell"],
            "alias": alias,
            "scenario": summary.get("scenario", alias),
            "mae": summary.get("mae"),
            "rmse": summary.get("rmse"),
            "p95_error": summary.get("p95_error"),
            "max_error": summary.get("max_error"),
            "bias": summary.get("bias"),
            "jump_count_gt_5pct": summary.get("jump_count_gt_5pct"),
            "recovery_time_h": summary.get("recovery_time_h"),
            "run_dir": str(run_dir),
            "run_mtime": run_dir.stat().st_mtime,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = (
        df.sort_values("run_mtime")
        .drop_duplicates(subset=["model", "alias"], keep="last")
        .reset_index(drop=True)
    )
    baselines = (
        df.loc[df["alias"] == "baseline", ["model", "mae", "rmse", "p95_error"]]
        .rename(columns={"mae": "baseline_mae", "rmse": "baseline_rmse", "p95_error": "baseline_p95"})
    )
    df = df.merge(baselines, on="model", how="left")
    df["delta_mae"] = df["mae"] - df["baseline_mae"]
    df["delta_rmse"] = df["rmse"] - df["baseline_rmse"]
    return df.sort_values(["model", "alias"]).reset_index(drop=True)


def write_tables(df: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    baseline = (
        df.loc[df["alias"] == "baseline", ["model", "mae", "rmse", "p95_error", "max_error", "bias", "run_dir"]]
        .sort_values("mae")
        .reset_index(drop=True)
    )
    baseline.to_csv(TABLES_DIR / "baseline_table.csv", index=False)
    try:
        (TABLES_DIR / "baseline_table.md").write_text(baseline.to_markdown(index=False))
    except Exception:
        (TABLES_DIR / "baseline_table.md").write_text(baseline.to_csv(index=False))

    delta = (
        df.loc[df["alias"] != "baseline", ["model", "alias", "scenario", "mae", "rmse", "delta_mae", "delta_rmse", "p95_error", "run_dir"]]
        .sort_values(["alias", "model"])
        .reset_index(drop=True)
    )
    delta.to_csv(TABLES_DIR / "scenario_delta_table.csv", index=False)
    try:
        (TABLES_DIR / "scenario_delta_table.md").write_text(delta.to_markdown(index=False))
    except Exception:
        (TABLES_DIR / "scenario_delta_table.md").write_text(delta.to_csv(index=False))

    ranking = (
        df.loc[df["alias"] != "baseline", ["alias", "model", "mae", "delta_mae"]]
        .sort_values(["alias", "mae"])
        .reset_index(drop=True)
    )
    ranking.to_csv(TABLES_DIR / "scenario_ranking_table.csv", index=False)
    try:
        (TABLES_DIR / "scenario_ranking_table.md").write_text(ranking.to_markdown(index=False))
    except Exception:
        (TABLES_DIR / "scenario_ranking_table.md").write_text(ranking.to_csv(index=False))

    local_path = LOCAL_ANALYSIS_DIR / "local_metrics.csv"
    if local_path.exists():
        local = pd.read_csv(local_path)
        local.to_csv(TABLES_DIR / "local_recovery_table.csv", index=False)
        try:
            (TABLES_DIR / "local_recovery_table.md").write_text(local.to_markdown(index=False))
        except Exception:
            (TABLES_DIR / "local_recovery_table.md").write_text(local.to_csv(index=False))


def _save_bar(df: pd.DataFrame, value_col: str, title: str, out_name: str) -> None:
    models = list(df["model"])
    values = list(df[value_col])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(models, values)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / out_name, dpi=180)
    plt.close(fig)


def _save_grouped_bar(df: pd.DataFrame, aliases: list[str], title: str, out_name: str) -> None:
    models = sorted(df["model"].unique())
    x = np.arange(len(models))
    width = 0.8 / max(1, len(aliases))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, alias in enumerate(aliases):
        sub = df[df["alias"] == alias].set_index("model").reindex(models)
        ax.bar(x + (i - (len(aliases) - 1) / 2) * width, sub["mae"], width=width, label=alias)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / out_name, dpi=180)
    plt.close(fig)


def _save_heatmap(df: pd.DataFrame, value_col: str, title: str, out_name: str) -> None:
    piv = df.pivot(index="model", columns="alias", values=value_col)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            txt = "nan" if pd.isna(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / out_name, dpi=180)
    plt.close(fig)


def write_figures(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline = df.loc[df["alias"] == "baseline", ["model", "mae"]].sort_values("mae")
    _save_bar(baseline, "mae", "Baseline MAE", "01_baseline_mae.png")

    heat_df = df.loc[df["alias"] != "baseline", ["model", "alias", "delta_mae"]].copy()
    if not heat_df.empty:
        _save_heatmap(heat_df, "delta_mae", "Delta MAE vs Baseline", "02_delta_mae_heatmap.png")

    offset_aliases = [a for a in ["current_offset", "voltage_offset", "temp_offset"] if a in set(df["alias"])]
    if offset_aliases:
        _save_grouped_bar(df, offset_aliases, "Offset Sensitivity", "03_offset_sensitivity.png")

    integrity_aliases = [a for a in ["missing_samples", "missing_gap", "irregular_sampling"] if a in set(df["alias"])]
    if integrity_aliases:
        _save_grouped_bar(df, integrity_aliases, "Signal Integrity Stress", "04_signal_integrity.png")

    noise_aliases = [a for a in ["current_noise_low", "current_noise_high", "voltage_noise", "temp_noise"] if a in set(df["alias"])]
    if noise_aliases:
        _save_grouped_bar(df, noise_aliases, "Noise Sensitivity", "05_noise_sensitivity.png")

    local_copy_map = {
        "06_initial_soc_local_recovery.png": LOCAL_ANALYSIS_DIR / "01_initial_soc_error_local_recovery.png",
        "07_spikes_local_recovery.png": LOCAL_ANALYSIS_DIR / "02_spikes_local_recovery.png",
        "08_current_noise_local_trend.png": LOCAL_ANALYSIS_DIR / "03_current_noise_local_trend.png",
    }
    for dst, src in local_copy_map.items():
        if src.exists():
            shutil.copy2(src, FIGURES_DIR / dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build paper-facing tables and figures from a completed benchmark campaign.")
    ap.add_argument("--campaign_tag", required=True)
    args = ap.parse_args()

    campaign_dir = ROOT / "campaigns" / args.campaign_tag
    df = load_campaign_rows(campaign_dir)
    if df.empty:
        raise SystemExit(f"No completed runs found in {campaign_dir}")

    write_tables(df)
    write_figures(df)

    summary = {
        "campaign_tag": args.campaign_tag,
        "tables_dir": str(TABLES_DIR),
        "figures_dir": str(FIGURES_DIR),
        "models": sorted(df["model"].unique()),
        "aliases": sorted(df["alias"].unique()),
        "n_runs": int(len(df)),
    }
    (RESULTS_DIR / "results_manifest_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
