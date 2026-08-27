from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns" / "jes2_dropout_corrected_preview_20260827"
FULL_CAMPAIGNS = {
    cell: ROOT / "campaigns" / f"jes2_full_{cell}_20260825"
    for cell in ("C09", "C13", "C15", "C25", "C27", "C29")
}
FIGURES = ROOT.parents[2] / "LATEX" / "JES" / "paper_robustness_benchmark" / "figures" / "Results"
TABLES = ROOT / "results" / "figure_09_six_cell_resume_error.csv"

MODEL_ORDER = ("DM", "HDM", "HECM", "DD")
MODEL_COLORS = {"DM": "#2ca02c", "HDM": "#9467bd", "HECM": "#1f77b4", "DD": "#d62728"}
FILES = {
    "DM": ("soc_cc_fullcell_C29.csv", "soc_cc"),
    "HDM": ("soc_cc_soh_fullcell_C29.csv", "soc_cc"),
    "HECM": ("ecm_soc_fullcell_C29.csv", "soc_ecm"),
    "DD": ("soc_pred_fullcell_C29.csv", "soc_pred"),
}


def load_run(condition: str, model: str) -> pd.DataFrame:
    filename, prediction = FILES[model]
    frame = pd.read_csv(CAMPAIGN / condition / model / filename)
    return frame.rename(columns={prediction: "soc_pred"})


def aligned_delta(baseline: pd.DataFrame, dropout: pd.DataFrame) -> pd.DataFrame:
    left = baseline[["time_s", "soc_pred"]].rename(columns={"soc_pred": "baseline"})
    right = dropout[["time_s", "soc_pred", "soc_true", "abs_err"]].rename(
        columns={"soc_pred": "dropout"}
    )
    joined = left.merge(right, on="time_s", how="inner")
    joined["abs_output_deviation"] = np.abs(joined["dropout"] - joined["baseline"])
    return joined


def load_six_cell_recovery() -> pd.DataFrame:
    rows = []
    for cell, campaign in FULL_CAMPAIGNS.items():
        summaries = campaign.glob(f"runs/{cell}/**/missing_gap_1h/seed_42/*/*/summary.json")
        for path in summaries:
            mode = path.parent.parent.name
            model = path.parent.name
            if model not in MODEL_ORDER or mode not in {"lstm_h1", "no_soh"}:
                continue
            summary = json.loads(path.read_text())
            resume_error = summary.get("common_recovery_initial_abs_err")
            if resume_error is None:
                continue
            rows.append({
                "cell": cell,
                "model": model,
                "resume_error": float(resume_error),
            })
    values = pd.DataFrame(rows)
    if values.empty:
        raise RuntimeError("No six-cell recovery summaries found")
    return values


def thin(frame: pd.DataFrame, maximum: int = 4500) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, maximum, dtype=int)]


def shade_gap(ax, gap_start: float, gap_end: float) -> None:
    ax.axvspan(0.0, (gap_end - gap_start) / 3600.0, color="#999999", alpha=0.20)
    ax.axvline(0.0, color="#555555", linewidth=0.8, linestyle=":")
    ax.axvline((gap_end - gap_start) / 3600.0, color="#555555", linewidth=0.8, linestyle=":")


def main() -> None:
    baseline = {model: load_run("baseline_48h", model) for model in MODEL_ORDER}
    dropout = {model: load_run("missing_gap_1h", model) for model in MODEL_ORDER}
    summary = json.loads((CAMPAIGN / "missing_gap_1h" / "DM" / "summary.json").read_text())
    gap_start = float(summary["gap_start_time_s"])
    gap_end = float(summary["gap_end_time_s"])
    panel_start = gap_start - 0.35 * 3600.0
    panel_end = gap_end + 2.0 * 3600.0

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.2), sharex=False)

    truth = dropout["DM"]
    truth = thin(truth[(truth.time_s >= panel_start) & (truth.time_s <= panel_end)])
    axes[0, 0].plot((truth.time_s - gap_start) / 3600.0, truth.soc_true, "k--", lw=1.5, label="Reference SOC")
    for model in MODEL_ORDER:
        part = dropout[model]
        part = thin(part[(part.time_s >= panel_start) & (part.time_s <= panel_end)])
        axes[0, 0].plot((part.time_s - gap_start) / 3600.0, part.soc_pred,
                        color=MODEL_COLORS[model], lw=1.3, label=model)
    shade_gap(axes[0, 0], gap_start, gap_end)
    axes[0, 0].set(xlabel="Time from dropout start [h]", ylabel="SOC",
                   title="(a) Corrected C29 burst-dropout transition")
    axes[0, 0].legend(ncol=3, frameon=False, fontsize=8)

    base_q = thin(baseline["DD"][(baseline["DD"].time_s >= panel_start) & (baseline["DD"].time_s <= panel_end)])
    gap_q = thin(dropout["DD"][(dropout["DD"].time_s >= panel_start) & (dropout["DD"].time_s <= panel_end)])
    axes[0, 1].plot((base_q.time_s - gap_start) / 3600.0, base_q.q_c_online,
                    color="#222222", lw=1.4, linestyle="--", label="No-dropout $Q_c$")
    axes[0, 1].plot((gap_q.time_s - gap_start) / 3600.0, gap_q.q_c_online,
                    color="#d62728", lw=1.5, label="Available online $Q_c$")
    shade_gap(axes[0, 1], gap_start, gap_end)
    axes[0, 1].set(xlabel="Time from dropout start [h]", ylabel="$Q_c$ [Ah]",
                   title="(b) Frozen counter and next voltage reset")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for model in MODEL_ORDER:
        joined = aligned_delta(baseline[model], dropout[model])
        part = thin(joined[(joined.time_s >= gap_end) & (joined.time_s <= gap_end + 4 * 3600.0)])
        axes[1, 0].plot((part.time_s - gap_end) / 3600.0, part.abs_output_deviation,
                        color=MODEL_COLORS[model], lw=1.35, label=model)
    axes[1, 0].set(xlabel="Time after measurements resume [h]",
                   ylabel="Absolute deviation from no-dropout output",
                   title="(c) Dropout-induced estimator deviation")
    axes[1, 0].legend(ncol=2, frameon=False, fontsize=8)

    for model in MODEL_ORDER:
        part = thin(dropout[model][(dropout[model].time_s >= gap_end) & (dropout[model].time_s <= gap_end + 4 * 3600.0)])
        axes[1, 1].plot((part.time_s - gap_end) / 3600.0, part.abs_err,
                        color=MODEL_COLORS[model], lw=1.35, label=model)
    axes[1, 1].axhline(0.02, color="#222222", linestyle="--", linewidth=1.0, label="2% band")
    axes[1, 1].set(xlabel="Time after measurements resume [h]", ylabel="Absolute SOC error",
                   title="(d) Error after resume")
    axes[1, 1].legend(ncol=3, frameon=False, fontsize=8)

    fig.suptitle(
        "One-hour burst dropout across six test cells",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    official_stem = FIGURES / "Figure_09_Burst_Dropout_Transition"
    fig.savefig(official_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(official_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
