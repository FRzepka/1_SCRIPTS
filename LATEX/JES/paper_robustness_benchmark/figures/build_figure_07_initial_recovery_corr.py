from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4] / "DL_Models" / "LFP_SOC_SOH_Model" / "4_simulation_environment"
TRAJECTORIES = ROOT / "campaigns" / "jes2_representative_trajectories_20260826"
RESULTS = Path(__file__).resolve().parent / "Results"
PAPER = Path(__file__).resolve().parents[1]
CELLS = ("C09", "C13", "C15", "C25", "C27", "C29")
MODEL_ORDER = ("DM", "HDM", "HECM", "DD")
MODEL_COLORS = {"DM": "#2ca02c", "HDM": "#9467bd", "HECM": "#1f77b4", "DD": "#d62728"}
EVALUATION_START_SAMPLE = 2023
FILES = {
    "DM": ("soc_cc_fullcell_C29.csv", "soc_cc"),
    "HDM": ("soc_cc_soh_fullcell_C29.csv", "soc_cc"),
    "HECM": ("ecm_soc_fullcell_C29.csv", "soc_ecm"),
    "DD": ("soc_pred_fullcell_C29.csv", "soc_pred"),
}
OUTPUT = RESULTS / "Figure_07_Initial_State_Recovery_CORR"


def load_trajectory(condition: str, model: str) -> pd.DataFrame:
    filename, prediction = FILES[model]
    frame = pd.read_csv(TRAJECTORIES / condition / model / filename)
    return frame.rename(columns={prediction: "soc_pred"})


def source_start_time(frame: pd.DataFrame) -> float:
    time_s = frame["time_s"].to_numpy(float)
    source_index = frame["index"].to_numpy(float)
    dt = np.diff(time_s) / np.diff(source_index)
    finite = dt[np.isfinite(dt) & (dt > 0.0)]
    nominal_dt = float(np.median(finite)) if len(finite) else 1.0
    return float(time_s[0] - source_index[0] * nominal_dt)


def recovery_statistics() -> pd.DataFrame:
    path = PAPER / "JES_2.0" / "results" / "jes2_paired_initial_recovery_statistics.csv"
    values = pd.read_csv(path)
    return (
        values[values["metric"] == "recovery_or_censor_time_h"]
        .set_index("model")
        .reindex(MODEL_ORDER)
    )


def main() -> None:
    baseline = {model: load_trajectory("baseline", model) for model in MODEL_ORDER}
    initial = {model: load_trajectory("initial_soc_error", model) for model in MODEL_ORDER}
    event_time_s = source_start_time(initial["DD"])
    time_h = (initial["DD"]["time_s"] - event_time_s) / 3600.0
    limit = min(12.0, float(time_h.iloc[-1]))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig = plt.figure(figsize=(12.5, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0), hspace=0.34, wspace=0.24)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    reference = initial["DD"]
    mask = (reference["index"] >= EVALUATION_START_SAMPLE) & (
        reference["time_s"] <= event_time_s + limit * 3600.0
    )
    ax_a.plot(time_h[mask], reference.loc[mask, "soc_true"], "k--", lw=1.5, label="Reference SOC")
    for model in MODEL_ORDER:
        frame = initial[model]
        mask = (frame["index"] >= EVALUATION_START_SAMPLE) & (
            frame["time_s"] <= event_time_s + limit * 3600.0
        )
        x = (frame.loc[mask, "time_s"] - event_time_s) / 3600.0
        ax_a.plot(x, frame.loc[mask, "soc_pred"], color=MODEL_COLORS[model], lw=1.3, label=f"{model}: 10% shifted")
        clean = baseline[model]
        clean_mask = (clean["index"] >= EVALUATION_START_SAMPLE) & (
            clean["time_s"] <= event_time_s + limit * 3600.0
        )
        clean_x = (clean.loc[clean_mask, "time_s"] - event_time_s) / 3600.0
        ax_a.plot(clean_x, clean.loc[clean_mask, "soc_pred"], color=MODEL_COLORS[model], lw=1.0, alpha=0.55, linestyle=":", label=f"{model}: correct")
    ax_a.axvline(EVALUATION_START_SAMPLE / 3600.0, color="#777777", linestyle="--", linewidth=0.9)
    ax_a.set(xlabel="Time from initialization intervention [h]", ylabel="SOC",
             title="(a) Correct versus 10% shifted initial SOC")
    ax_a.legend(ncol=4, frameon=False, fontsize=6.5)

    for model in MODEL_ORDER:
        left = baseline[model][["time_s", "soc_pred"]].rename(columns={"soc_pred": "correct"})
        right = initial[model][["time_s", "soc_pred"]].rename(columns={"soc_pred": "shifted"})
        paired = left.merge(right, on="time_s", how="inner")
        paired["difference"] = (paired["shifted"] - paired["correct"]).abs()
        paired = paired[
            (paired["time_s"] >= event_time_s + EVALUATION_START_SAMPLE)
            & (paired["time_s"] <= event_time_s + limit * 3600.0)
        ]
        x = (paired["time_s"] - event_time_s) / 3600.0
        ax_b.plot(x, paired["difference"], color=MODEL_COLORS[model], lw=1.35, label=model)
    ax_b.axhline(0.02, color="#222222", linestyle="--", linewidth=1.2, label="Recovery threshold (2% difference)")
    ax_b.axvline(
        EVALUATION_START_SAMPLE / 3600.0,
        color="#777777",
        linestyle="--",
        linewidth=0.9,
        label="Common scoring starts",
    )
    ax_b.set(xlabel="Time from initialization intervention [h]", ylabel="|Shifted - correct SOC|",
             title="(b) Recovery of the initial-state mismatch")
    ax_b.legend(frameon=False, fontsize=7)
    ax_b.set_ylim(bottom=0)

    stats = recovery_statistics()
    positions = np.arange(len(MODEL_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        value = float(stats.loc[model, "mean"])
        low = float(stats.loc[model, "ci_low"])
        high = float(stats.loc[model, "ci_high"])
        ax_c.bar(positions[index], value, 0.62,
                 color=MODEL_COLORS[model], alpha=0.42,
                 edgecolor=MODEL_COLORS[model], linewidth=2.2)
        ax_c.errorbar(index, value, yerr=[[value - low], [high - value]],
                      color="#111111", capsize=4, linewidth=1.4)
    ax_c.set_xticks(positions, MODEL_ORDER)
    ax_c.set(xlabel="Estimator class", ylabel="Recovery/censor time [h]",
             title="(c) Six-cell mean and hierarchical 95% CI")
    ax_c.set_ylim(bottom=0)

    fig.suptitle("Initial-state recovery after a 10% SOC initialization mismatch", fontsize=12)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.88)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
