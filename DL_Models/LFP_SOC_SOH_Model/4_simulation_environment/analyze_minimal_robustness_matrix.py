import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCENARIO_ORDER = [
    "baseline",
    "current_noise",
    "current_offset",
    "initial_soc_error",
    "missing_samples",
    "spikes",
]


def load_campaign_rows(tag: str):
    campaign_dir = ROOT / "campaigns" / tag
    rows = []
    seen = set()
    for manifest_path in sorted(campaign_dir.glob("*_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        model = manifest["model"]
        for run in manifest["runs"]:
            summary_path = Path(run["out_dir"]) / "summary.json"
            if not summary_path.exists():
                continue
            seen.add(str(summary_path.resolve()))
            summary = json.loads(summary_path.read_text())
            rows.append(
                {
                    "model": model,
                    "scenario": run["scenario"],
                    "cell": summary.get("cell"),
                    "mae": summary.get("mae"),
                    "rmse": summary.get("rmse"),
                    "p95_error": summary.get("p95_error"),
                    "max_error": summary.get("max_error"),
                    "jump_count_gt_5pct": summary.get("jump_count_gt_5pct"),
                    "drift_rate_abs_err_per_h": summary.get("drift_rate_abs_err_per_h"),
                    "disturbed_mae": summary.get("disturbed_mae"),
                    "recovery_time_h": summary.get("recovery_time_h"),
                    "run_dir": run["out_dir"],
                    "status": run["status"],
                }
            )
    for model_dir in ROOT.iterdir():
        runs_dir = model_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for summary_path in runs_dir.glob(f"*/*{tag}*/summary.json"):
            key = str(summary_path.resolve())
            if key in seen:
                continue
            summary = json.loads(summary_path.read_text())
            rows.append(
                {
                    "model": summary.get("model", model_dir.name),
                    "scenario": summary.get("scenario", summary_path.parent.parent.name),
                    "cell": summary.get("cell"),
                    "mae": summary.get("mae"),
                    "rmse": summary.get("rmse"),
                    "p95_error": summary.get("p95_error"),
                    "max_error": summary.get("max_error"),
                    "jump_count_gt_5pct": summary.get("jump_count_gt_5pct"),
                    "drift_rate_abs_err_per_h": summary.get("drift_rate_abs_err_per_h"),
                    "disturbed_mae": summary.get("disturbed_mae"),
                    "recovery_time_h": summary.get("recovery_time_h"),
                    "run_dir": str(summary_path.parent),
                    "status": "completed",
                }
            )
    return pd.DataFrame(rows), campaign_dir


def grouped_bar(df: pd.DataFrame, value_col: str, title: str, out_png: Path) -> None:
    pivot = df.pivot(index="scenario", columns="model", values=value_col).reindex(SCENARIO_ORDER)
    ax = pivot.plot(kind="bar", figsize=(13, 6))
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(value_col)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def heatmap(df: pd.DataFrame, value_col: str, title: str, out_png: Path) -> None:
    pivot = df.pivot(index="scenario", columns="model", values=value_col).reindex(SCENARIO_ORDER)
    vals = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(vals, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)


def dataframe_to_markdown_fallback(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = [
        "|" + "|".join(cols) + "|",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            else:
                vals.append(str(val))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze one minimal robustness benchmark campaign.")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    df, campaign_dir = load_campaign_rows(args.tag)
    if df.empty:
        raise SystemExit(f"No completed summaries found for campaign {args.tag}")

    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    df = df.sort_values(["model", "scenario", "run_dir"]).drop_duplicates(
        subset=["model", "scenario"], keep="last"
    ).reset_index(drop=True)

    baseline = df[df["scenario"] == "baseline"][["model", "mae", "rmse"]].rename(columns={"mae": "baseline_mae", "rmse": "baseline_rmse"})
    df = df.merge(baseline, on="model", how="left")
    df["delta_mae"] = df["mae"] - df["baseline_mae"]
    df["delta_rmse"] = df["rmse"] - df["baseline_rmse"]

    out_dir = campaign_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "minimal_matrix_summary.csv", index=False)
    try:
        md_text = df.to_markdown(index=False)
    except ImportError:
        md_text = dataframe_to_markdown_fallback(df)
    (out_dir / "minimal_matrix_summary.md").write_text(md_text)

    grouped_bar(df, "mae", "Minimal Matrix: MAE by Scenario and Model", out_dir / "01_mae_grouped.png")
    grouped_bar(df, "delta_mae", "Minimal Matrix: Delta MAE vs Baseline", out_dir / "02_delta_mae_grouped.png")
    heatmap(df, "p95_error", "Minimal Matrix: P95 Absolute Error", out_dir / "03_p95_heatmap.png")

    findings = []
    findings.append("|scenario|best_model|best_mae|worst_model|worst_mae|")
    findings.append("|---|---|---:|---|---:|")
    for scenario in SCENARIO_ORDER:
        sub = df[df["scenario"] == scenario].dropna(subset=["mae"])
        if sub.empty:
            continue
        best = sub.sort_values("mae").iloc[0]
        worst = sub.sort_values("mae").iloc[-1]
        findings.append(f"|{scenario}|{best['model']}|{best['mae']:.6f}|{worst['model']}|{worst['mae']:.6f}|")
    findings.append("")
    findings.append("Interpretation guidance:")
    findings.append("- `delta_mae > 0` means the disturbance degraded the estimator relative to its own baseline.")
    findings.append("- `p95_error` is useful for robustness claims because it captures large but non-singleton failures.")
    findings.append("- `jump_count_gt_5pct` highlights unstable estimator outputs, not just average error.")
    (out_dir / "FINDINGS.md").write_text("\n".join(findings))

    print(f"Wrote {out_dir / 'minimal_matrix_summary.csv'}")
    print(f"Wrote {out_dir / 'minimal_matrix_summary.md'}")
    print(f"Wrote {out_dir / '01_mae_grouped.png'}")
    print(f"Wrote {out_dir / '02_delta_mae_grouped.png'}")
    print(f"Wrote {out_dir / '03_p95_heatmap.png'}")
    print(f"Wrote {out_dir / 'FINDINGS.md'}")


if __name__ == "__main__":
    main()
