from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4] / "DL_Models" / "LFP_SOC_SOH_Model" / "4_simulation_environment"
RESULTS = Path(__file__).resolve().parent / "Results"
CAMPAIGN = ROOT / "campaigns" / "jes2_dropout_corrected_preview_20260827"
CELLS = ("C09", "C13", "C15", "C25", "C27", "C29")
MODEL_ORDER = ("DM", "HDM", "HECM", "DD")
MODEL_COLORS = {"DM": "#2ca02c", "HDM": "#9467bd", "HECM": "#1f77b4", "DD": "#d62728"}
OUTPUT = RESULTS / "Figure_09_Burst_Dropout_Transition_CORR"

sys.path.insert(0, str(ROOT / "results"))
from build_corrected_dropout_preview import aligned_delta, thin, shade_gap  # noqa: E402


FILES = {
    "DM": ("soc_cc_fullcell_C29.csv", "soc_cc"),
    "HDM": ("soc_cc_soh_fullcell_C29.csv", "soc_cc"),
    "HECM": ("ecm_soc_fullcell_C29.csv", "soc_ecm"),
    "DD": ("soc_pred_fullcell_C29.csv", "soc_pred"),
}


def load_source_run(condition: str, model: str) -> pd.DataFrame:
    filename, prediction = FILES[model]
    frame = pd.read_csv(CAMPAIGN / condition / model / filename)
    return frame.rename(columns={prediction: "soc_pred"})


def recovery_statistics() -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        campaign = ROOT / "campaigns" / f"jes2_full_{cell}_20260825"
        for path in campaign.glob(f"runs/{cell}/**/missing_gap_1h/seed_42/*/*/summary.json"):
            mode = path.parent.parent.name
            model = path.parent.name
            if model not in MODEL_ORDER or mode not in {"lstm_h1", "no_soh"}:
                continue
            summary = json.loads(path.read_text())
            value = summary.get("common_recovery_or_censor_time_h")
            if value is not None:
                rows.append({"cell": cell, "model": model, "value": float(value)})
    values = pd.DataFrame(rows)
    by_cell = values.groupby(["cell", "model"], as_index=False)["value"].mean()
    return by_cell.groupby("model")["value"].agg(["mean", "std"]).reindex(MODEL_ORDER)


def main() -> None:
    baseline = {model: load_source_run("baseline_48h", model) for model in MODEL_ORDER}
    dropout = {model: load_source_run("missing_gap_1h", model) for model in MODEL_ORDER}
    summary = json.loads((CAMPAIGN / "missing_gap_1h" / "DM" / "summary.json").read_text())
    gap_start = float(summary["gap_start_time_s"])
    gap_end = float(summary["gap_end_time_s"])
    panel_start = gap_start - 0.35 * 3600.0
    panel_end = gap_end + 2.0 * 3600.0

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig = plt.figure(figsize=(12.5, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.95, 1.25), hspace=0.34, wspace=0.24)
    ax_a = fig.add_subplot(grid[0, :])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_b = fig.add_subplot(grid[1, 1])

    truth = dropout["DM"]
    truth = thin(truth[(truth.time_s >= panel_start) & (truth.time_s <= panel_end)])
    ax_a.plot((truth.time_s - gap_start) / 3600.0, truth.soc_true, "k--", lw=1.5, label="Reference SOC")
    for model in MODEL_ORDER:
        part = dropout[model]
        part = thin(part[(part.time_s >= panel_start) & (part.time_s <= panel_end)])
        ax_a.plot((part.time_s - gap_start) / 3600.0, part.soc_pred,
                  color=MODEL_COLORS[model], lw=1.3, label=model)
    shade_gap(ax_a, gap_start, gap_end)
    ax_a.set(xlabel="Time from dropout start [h]", ylabel="SOC",
             title="(a) Corrected C29 burst-dropout transition")
    ax_a.legend(ncol=5, frameon=False, fontsize=8)

    for model in MODEL_ORDER:
        joined = aligned_delta(baseline[model], dropout[model])
        part = thin(joined[(joined.time_s >= gap_end) & (joined.time_s <= gap_end + 4 * 3600.0)])
        ax_c.plot((part.time_s - gap_end) / 3600.0, part.abs_output_deviation,
                  color=MODEL_COLORS[model], lw=1.35, label=model)
    ax_c.set(xlabel="Time after measurements resume [h]",
             ylabel="Absolute deviation from no-dropout output",
             title="(b) Dropout-induced estimator deviation")
    ax_c.legend(ncol=2, frameon=False, fontsize=8)

    stats = recovery_statistics()
    positions = np.arange(len(MODEL_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        value = float(stats.loc[model, "mean"])
        error = float(stats.loc[model, "std"] or 0.0)
        ax_b.bar(positions[index], value, 0.62, yerr=error, capsize=4,
                 ecolor="#111111", error_kw={"elinewidth": 1.4, "capthick": 1.4},
                 color=MODEL_COLORS[model], alpha=0.42,
                 edgecolor=MODEL_COLORS[model], linewidth=2.2)
    ax_b.set_xticks(positions, MODEL_ORDER)
    ax_b.set(xlabel="Estimator class", ylabel="Recovery/censor time [h]",
             title="(c) Six-cell recovery after burst dropout")
    ax_b.set_ylim(bottom=0)

    fig.suptitle("One-hour burst dropout: transition, estimator deviation, and six-cell recovery", fontsize=12)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.88)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
