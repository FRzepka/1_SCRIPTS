import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
PAPER_TABLES_DIR = RESULTS_DIR / "paper_tables"
PAPER_FIGURES_DIR = RESULTS_DIR / "paper_figures"
LOCAL_ANALYSIS_DIR = ROOT / "archive" / "2026-03-15_cleanup_noise_bias_only" / "analyses" / "analysis_local_focus" / "2026-03-12_local_recovery"
CAMPAIGN_DEFAULT = "2026-03-12_extended_matrix_fullc07"
BIAS_CAMPAIGN_TAG = "bias_0p5_1p5_3p0_percent_fullc07"
BIAS_ECM_CAMPAIGN_TAG = "bias_0p5_1p5_3p0_percent_fullc07_ecm003"
DATA_ROOT = Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_DIR))

from robustness_common import apply_measurement_scenario, load_cell_dataframe
from build_paper_results import load_campaign_rows

MODEL_ORDER = [
    "CC_1.0.0",
    "CC_SOH_1.0.0",
    "ECM_0.0.3",
    "SOC_SOH_1.6.0.0_GRU_0.3.1.2",
]

MODEL_META = {
    "CC_1.0.0": {
        "label": "Direct measurement",
        "short": "DM",
        "color": "#6e2fc4",
        "fill": "#6e2fc4",
    },
    "CC_SOH_1.0.0": {
        "label": "Hybrid direct measurement",
        "short": "HDM",
        "color": "#08bdba",
        "fill": "#08bdba",
    },
    "ECM_0.0.3": {
        "label": "Hybrid ECM",
        "short": "HECM",
        "color": "#d4bbff",
        "fill": "#d4bbff",
    },
    "ECM_0.0.1": {
        "label": "Hybrid ECM",
        "short": "HECM",
        "color": "#d4bbff",
        "fill": "#d4bbff",
    },
    "SOC_SOH_1.6.0.0_GRU_0.3.1.2": {
        "label": "Data-driven",
        "short": "DD",
        "color": "#4589ff",
        "fill": "#4589ff",
    },
}

SCENARIO_SPECS = {
    "baseline": ("baseline", {}),
    "current_noise_low": ("current_noise", {"current_noise_std": 0.02}),
    "current_noise_high": ("current_noise", {"current_noise_std": 0.10}),
    "voltage_noise": ("voltage_noise", {"voltage_noise_std": 0.01}),
    "temp_noise": ("temp_noise", {"temp_noise_std": 1.0}),
    "current_offset": ("current_offset", {"current_offset_a": 0.05}),
    "voltage_offset": ("voltage_offset", {"voltage_offset_v": 0.02}),
    "temp_offset": ("temp_offset", {"temp_offset_c": 3.0}),
    "adc_quantization": ("adc_quantization", {}),
    "initial_soc_error": ("initial_soc_error", {"soc_init_error": -0.10}),
    "missing_samples": ("missing_samples", {"missing_samples_every": 50}),
    "irregular_sampling": ("irregular_sampling", {"irregular_dt_jitter": 0.1}),
    "missing_gap": ("missing_gap", {"missing_gap_seconds": 3600.0}),
    "spikes_high": (
        "spikes",
        {"spike_channel": "Voltage[V]", "spike_magnitude": 0.20, "spike_period": 1000},
    ),
}

BIAS_DISPLAY = {
    "baseline": ("0.0%", 0.0),
    "current_bias_0p5pct": ("0.5%", 0.5),
    "current_bias_1p5pct": ("1.5%", 1.5),
    "current_bias_3p0pct": ("3.0%", 3.0),
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.7,
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "savefig.bbox": "tight",
            "savefig.dpi": 240,
        }
    )


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_order=df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})).sort_values("_order").drop(columns="_order")


def _write_md(df: pd.DataFrame, path: Path) -> None:
    try:
        path.write_text(df.to_markdown(index=False))
    except Exception:
        path.write_text(df.to_csv(index=False))


def _load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def _find_pred_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.startswith("soc_") and c != "soc_true"]
    if not candidates:
        raise ValueError(f"No prediction column found in {df.columns.tolist()}")
    if "soc_pred" in candidates:
        return "soc_pred"
    if "soc_ecm" in candidates:
        return "soc_ecm"
    if "soc_cc" in candidates:
        return "soc_cc"
    return candidates[0]


def _load_run_series(run_dir: Path, model: str) -> pd.DataFrame:
    csv_files = sorted(run_dir.glob("*.csv"))
    chosen = None
    for p in csv_files:
        if "soh_hourly" in p.name:
            continue
        chosen = p
        break
    if chosen is None:
        raise FileNotFoundError(f"No SOC csv found in {run_dir}")
    df = pd.read_csv(chosen)
    pred_col = _find_pred_col(df)
    keep = ["time_s", "soc_true", pred_col, "abs_err"]
    extra = [c for c in ["I", "U", "SOH", "q_m_new", "soh_pred", "u_ecm"] if c in df.columns]
    out = df[keep + extra].copy()
    out = out.rename(columns={pred_col: "soc_pred"})
    out["model"] = model
    return out


def _scenario_args(alias: str) -> SimpleNamespace:
    scenario, kwargs = SCENARIO_SPECS[alias]
    ns = SimpleNamespace(
        scenario=scenario,
        seed=42,
        current_offset_a=None,
        current_offset_pct=None,
        voltage_offset_v=None,
        temp_offset_c=None,
        current_noise_std=None,
        voltage_noise_std=None,
        temp_noise_std=None,
        temp_constant=None,
        quantize_current_a=None,
        quantize_voltage_v=None,
        quantize_temp_c=None,
        spike_channel="Voltage[V]",
        spike_magnitude=None,
        spike_period=None,
        spike_prob=None,
        soc_init_error=0.0,
        missing_gap_seconds=0.0,
        missing_samples_every=None,
        missing_samples_pct=None,
        irregular_dt_jitter=None,
        downsample_hz=None,
        drop_pct=None,
        drop_segment_len=None,
    )
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def _load_measurement_scenario(cell: str, alias: str) -> tuple[pd.DataFrame, dict]:
    df = load_cell_dataframe(str(DATA_ROOT), cell)
    args = _scenario_args(alias)
    out, meta = apply_measurement_scenario(df, SCENARIO_SPECS[alias][0], args)
    return out, meta


def _rolling_abs_err(df: pd.DataFrame, window_s: float = 900.0) -> pd.DataFrame:
    tmp = df[["time_s", "abs_err"]].copy().sort_values("time_s")
    tmp["rolling_mae"] = (
        tmp.set_index(pd.to_timedelta(tmp["time_s"], unit="s"))["abs_err"]
        .rolling(f"{int(window_s)}s", min_periods=1)
        .mean()
        .to_numpy()
    )
    return tmp


def _find_gap_window(meta: dict, time_s: np.ndarray) -> tuple[float, float]:
    mask = np.asarray(meta["freeze_mask"], dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("No missing-gap mask found")
    return float(time_s[idx[0]]), float(time_s[idx[-1]])


def _spike_indices(meta: dict) -> np.ndarray:
    mask = np.asarray(meta["disturbance_mask"], dtype=bool)
    return np.flatnonzero(mask)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _thin(df: pd.DataFrame, max_points: int = 4000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    stride = int(np.ceil(len(df) / max_points))
    return df.iloc[::stride].copy()


def _load_bias_rows() -> pd.DataFrame:
    model_run_dirs = {
        "CC_1.0.0": ROOT / "CC_1.0.0" / "runs",
        "CC_SOH_1.0.0": ROOT / "CC_SOH_1.0.0" / "runs",
        "ECM_0.0.3": ROOT / "ECM_0.0.3" / "runs",
        "SOC_SOH_1.6.0.0_GRU_0.3.1.2": ROOT / "SOC_SOH_1.6.0.0_0.1.2.3" / "runs",
    }
    rows = []
    for model_name, runs_root in model_run_dirs.items():
        if not runs_root.exists():
            continue
        model_tag = BIAS_ECM_CAMPAIGN_TAG if model_name == "ECM_0.0.3" else BIAS_CAMPAIGN_TAG
        for alias, (label, pct) in BIAS_DISPLAY.items():
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
            rows.append(
                {
                    "model": summary.get("model", model_name),
                    "alias": alias,
                    "bias_label": label,
                    "bias_pct": pct,
                    "bias_a": float(summary.get("scenario_meta", {}).get("current_offset_a", 0.0)),
                    "mae": float(summary["mae"]),
                    "rmse": float(summary["rmse"]),
                    "run_dir": str(latest),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["model", "bias_pct"]).reset_index(drop=True)
    base = df.loc[df["alias"] == "baseline", ["model", "mae", "rmse"]].rename(
        columns={"mae": "baseline_mae", "rmse": "baseline_rmse"}
    )
    df = df.merge(base, on="model", how="left")
    df["delta_mae"] = df["mae"] - df["baseline_mae"]
    df["delta_rmse"] = df["rmse"] - df["baseline_rmse"]
    return df


def build_tables(df: pd.DataFrame) -> None:
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _ordered(
        df[df["alias"] == "baseline"][["model", "mae", "rmse", "p95_error", "max_error", "bias"]].copy()
    )
    baseline["class"] = baseline["model"].map(lambda m: MODEL_META[m]["label"])
    baseline = baseline[["class", "mae", "rmse", "p95_error", "max_error", "bias"]]
    baseline.to_csv(PAPER_TABLES_DIR / "table_baseline.csv", index=False)
    _write_md(baseline.round(4), PAPER_TABLES_DIR / "table_baseline.md")

    selected = df[
        df["alias"].isin(
            [
                "current_offset",
                "voltage_noise",
                "temp_noise",
                "missing_samples",
                "missing_gap",
                "irregular_sampling",
                "spikes_high",
                "current_noise_high",
                "initial_soc_error",
            ]
        )
    ].copy()
    selected = _ordered(selected)
    selected["class"] = selected["model"].map(lambda m: MODEL_META[m]["label"])
    selected["scenario_label"] = selected["alias"].map(
        {
            "current_offset": "Current bias",
            "voltage_noise": "Voltage noise",
            "temp_noise": "Temperature noise",
            "missing_samples": "Missing samples",
            "missing_gap": "Burst dropout",
            "irregular_sampling": "Irregular sampling",
            "spikes_high": "Voltage spikes",
            "current_noise_high": "Current noise (high)",
            "initial_soc_error": "Initial SOC error",
        }
    )
    key = selected[["scenario_label", "class", "mae", "delta_mae", "rmse", "delta_rmse", "p95_error"]]
    key.to_csv(PAPER_TABLES_DIR / "table_key_results.csv", index=False)
    _write_md(key.round(4), PAPER_TABLES_DIR / "table_key_results.md")

    local = pd.read_csv(LOCAL_ANALYSIS_DIR / "local_metrics.csv")
    local["class"] = local["model"].map(lambda m: MODEL_META[m]["label"])
    local = local[["class", "focus_scenario", "local_metric", "value", "threshold"]]
    local.to_csv(PAPER_TABLES_DIR / "table_local_behaviour.csv", index=False)
    _write_md(local.round(4), PAPER_TABLES_DIR / "table_local_behaviour.md")

    figure_scope = pd.DataFrame(
        [
            {
                "figure": "Figure_1_baseline_performance",
                "global_evidence": "Baseline MAE and RMSE on the complete C07 run",
                "local_evidence": "none",
                "note": "pure global ranking figure",
            },
            {
                "figure": "Figure_2_current_bias",
                "global_evidence": "MAE vs current-bias level (0.5 / 1.5 / 3.0 % of |I|max)",
                "local_evidence": "30 min current-input zoom plus first-12h SOC divergence at 3.0 % bias",
                "note": "baseline accuracy and bias robustness differ strongly; HECM remains the most controlled under bias",
            },
            {
                "figure": "Figure_3_noise_robustness",
                "global_evidence": "Delta MAE under current, voltage, and temperature noise",
                "local_evidence": "rolling-MAE drift and late-minus-early rolling MAE",
                "note": "use local metrics because global noise penalties are small",
            },
            {
                "figure": "Figure_4_signal_integrity",
                "global_evidence": "Delta MAE for missing samples, irregular sampling, and burst dropout",
                "local_evidence": "recovery time after burst dropout",
                "note": "mixed global/local summary figure",
            },
            {
                "figure": "Figure_5_missing_gap_transition",
                "global_evidence": "none",
                "local_evidence": "sensor and SOC behaviour around the dropout window",
                "note": "pure local transition figure",
            },
            {
                "figure": "Figure_6_missing_gap_recovery",
                "global_evidence": "none",
                "local_evidence": "post-gap absolute-error recovery over time",
                "note": "pure local recovery figure",
            },
            {
                "figure": "Figure_7_initial_state_recovery",
                "global_evidence": "global MAE listed in key-results table",
                "local_evidence": "CV-free initial-state recovery against the fair baseline band",
                "note": "local interpretation is the primary evidence",
            },
            {
                "figure": "Figure_8_spike_response",
                "global_evidence": "Delta MAE under voltage spikes",
                "local_evidence": "aligned per-spike excess-error response",
                "note": "single spikes are globally weak, so local response matters",
            },
        ]
    )
    figure_scope.to_csv(PAPER_TABLES_DIR / "table_figure_scope.csv", index=False)
    _write_md(figure_scope, PAPER_TABLES_DIR / "table_figure_scope.md")


def figure_baseline(df: pd.DataFrame) -> None:
    baseline = _ordered(df[df["alias"] == "baseline"].copy())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, metric, panel in zip(axes, ["mae", "rmse"], ["(a)", "(b)"]):
        vals = baseline[metric].to_numpy()
        labels = [MODEL_META[m]["short"] for m in baseline["model"]]
        colors = [MODEL_META[m]["fill"] for m in baseline["model"]]
        edges = [MODEL_META[m]["color"] for m in baseline["model"]]
        y = np.arange(len(vals))
        ax.barh(y, vals, color=colors, edgecolor=edges, linewidth=2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(metric.upper())
        ax.set_title(panel, loc="left", fontsize=14, fontweight="bold", pad=6)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.4f}", va="center", ha="left", fontsize=11)
    _savefig(PAPER_FIGURES_DIR / "Figure_1_baseline_performance.png")


def figure_offset(_: pd.DataFrame) -> None:
    bias_df = _ordered(_load_bias_rows())
    if bias_df.empty:
        raise FileNotFoundError("No bias-sweep rows found for Figure_2_current_bias")

    raw = load_cell_dataframe(str(DATA_ROOT), "MGFarm_18650_C07").copy()
    if "time_s" in raw.columns:
        raw["time_s"] = raw["time_s"].astype(float)
    elif "Testtime[s]" in raw.columns:
        raw["time_s"] = raw["Testtime[s]"].astype(float)
    else:
        raw["time_s"] = np.arange(len(raw), dtype=float)
    t0 = float(raw["time_s"].min())
    t_start = t0
    t_end = t0 + 1800.0
    i_max = float(raw["Current[A]"].abs().max())
    bias_3pct_a = i_max * 0.03
    bias_args = SimpleNamespace(
        seed=42,
        current_offset_a=bias_3pct_a,
        current_offset_pct=None,
        voltage_offset_v=None,
        temp_offset_c=None,
        current_noise_std=None,
        voltage_noise_std=None,
        temp_noise_std=None,
        temp_constant=None,
        quantize_current_a=None,
        quantize_voltage_v=None,
        quantize_temp_c=None,
        spike_channel="Voltage[V]",
        spike_magnitude=None,
        spike_period=None,
        spike_prob=None,
        soc_init_error=0.0,
        missing_gap_seconds=0.0,
        missing_samples_every=None,
        missing_samples_pct=None,
        irregular_dt_jitter=None,
        downsample_hz=None,
        drop_pct=None,
        drop_segment_len=None,
    )
    raw_bias, _ = apply_measurement_scenario(raw, "current_offset", bias_args)
    raw_win = raw[(raw["time_s"] >= t_start) & (raw["time_s"] <= t_end)].copy()
    raw_bias_win = raw_bias[(raw_bias["time_s"] >= t_start) & (raw_bias["time_s"] <= t_end)].copy()

    fig = plt.figure(figsize=(13.8, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.24)
    ax_top = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    ax_top.plot(
        (raw_win["time_s"] - t_start) / 60.0,
        raw_win["Current[A]"],
        color="#225ea8",
        lw=1.8,
        label="Baseline current",
    )
    ax_top.plot(
        (raw_bias_win["time_s"] - t_start) / 60.0,
        raw_bias_win["Current[A]"],
        color="#6baed6",
        lw=1.4,
        label="+3% bias",
    )
    ax_top.set_xlabel("Time [min]")
    ax_top.set_ylabel("Current [A]")
    ax_top.legend(frameon=True, loc="upper right")
    ax_top.set_title("(a)", loc="left", fontsize=14, fontweight="bold", pad=6)

    for model in MODEL_ORDER:
        sub = bias_df[bias_df["model"] == model].sort_values("bias_pct")
        ax_left.plot(
            sub["bias_pct"],
            100.0 * sub["mae"],
            marker="o",
            ms=7,
            lw=2.2,
            color=MODEL_META[model]["color"],
            label=MODEL_META[model]["short"],
        )
    ax_left.set_xlabel("Current bias [% of |I|max]")
    ax_left.set_ylabel("MAE [SOC pp]")
    ax_left.legend(frameon=True, ncol=2, loc="upper left")
    ax_left.set_title("(b)", loc="left", fontsize=14, fontweight="bold", pad=6)

    t_end_12h = t0 + 12.0 * 3600.0
    first_legend_handles = []
    true_ref = None
    for model in MODEL_ORDER:
        sub = bias_df[bias_df["model"] == model]
        base_row = sub[sub["alias"] == "baseline"].iloc[0]
        bias_row = sub[sub["alias"] == "current_bias_3p0pct"].iloc[0]
        base_series = _load_run_series(Path(base_row["run_dir"]), model)
        bias_series = _load_run_series(Path(bias_row["run_dir"]), model)
        base_win = base_series[(base_series["time_s"] >= t0) & (base_series["time_s"] <= t_end_12h)].copy()
        bias_win = bias_series[(bias_series["time_s"] >= t0) & (bias_series["time_s"] <= t_end_12h)].copy()
        if true_ref is None:
            true_ref = base_win[["time_s", "soc_true"]].copy()
        ax_right.plot(
            (base_win["time_s"] - t0) / 3600.0,
            base_win["soc_pred"],
            color=MODEL_META[model]["color"],
            lw=1.2,
            alpha=0.35,
            ls="--",
        )
        h = ax_right.plot(
            (bias_win["time_s"] - t0) / 3600.0,
            bias_win["soc_pred"],
            color=MODEL_META[model]["color"],
            lw=2.0,
            alpha=0.95,
            label=MODEL_META[model]["short"],
        )[0]
        first_legend_handles.append(h)
    ax_right.plot((true_ref["time_s"] - t0) / 3600.0, true_ref["soc_true"], color="#111111", lw=1.2, label="SOC true")
    ax_right.set_xlabel("Time [h]")
    ax_right.set_ylabel("SOC [-]")
    model_legend = ax_right.legend(handles=first_legend_handles, frameon=True, ncol=2, loc="upper left")
    ax_right.add_artist(model_legend)
    ax_right.legend(
        handles=[
            Line2D([0], [0], color="#666666", lw=1.2, ls="--", alpha=0.6, label="Baseline prediction"),
            Line2D([0], [0], color="#666666", lw=2.0, alpha=1.0, label="+3.0% bias prediction"),
            Line2D([0], [0], color="#111111", lw=1.2, label="SOC true"),
        ],
        frameon=True,
        loc="lower left",
    )
    ax_right.set_title("(c)", loc="left", fontsize=14, fontweight="bold", pad=6)

    fig.subplots_adjust(top=0.90)
    _savefig(PAPER_FIGURES_DIR / "Figure_2_current_bias.png")


def _select_noise_example_window(raw: pd.DataFrame) -> float:
    cur = raw["Current[A]"].astype(float).abs().to_numpy()
    time_s = raw["time_s"].astype(float).to_numpy()
    quiet = cur < 0.05
    run = 0
    for i, is_quiet in enumerate(quiet):
        if is_quiet:
            run += 1
            continue
        if run >= 300:
            start = i - run
            if time_s[start] >= 40.0 * 3600.0:
                # Start 20 min before the CV-to-rest transition so the informative
                # part sits closer to the center of the 30 min panel.
                return max(float(time_s[start] - 1200.0), float(time_s.min()))
        run = 0
    return float(time_s.min())


def figure_noise(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14.2, 12.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.15, 1.0], hspace=0.34, wspace=0.26)
    ax_top = fig.add_subplot(gs[0, :])
    ax_mid = fig.add_subplot(gs[1, :])
    axes = np.array([fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])])
    detail_csv = RESULTS_DIR / "noise_detail" / "current_noise_detail.csv"
    jitter_csv = RESULTS_DIR / "noise_detail" / "current_noise_output_jitter.csv"
    if not detail_csv.exists():
        raise FileNotFoundError(f"Missing noise detail file: {detail_csv}")
    if not jitter_csv.exists():
        raise FileNotFoundError(f"Missing noise jitter file: {jitter_csv}")
    detail = pd.read_csv(detail_csv)
    jitter = pd.read_csv(jitter_csv)
    levels = sorted(detail["current_noise_std"].unique())

    # Re-plot the accepted first-sequence 30 min current window directly so the
    # paper figure uses the same visual language as the other panels.
    current_df = load_cell_dataframe(str(DATA_ROOT), "MGFarm_18650_C07").copy()
    if "time_s" in current_df.columns:
        t_s = current_df["time_s"].to_numpy(dtype=float)
    elif "Testtime[s]" in current_df.columns:
        t_s = current_df["Testtime[s]"].to_numpy(dtype=float)
    else:
        t_s = np.arange(len(current_df), dtype=float)
    current_df["time_s"] = t_s
    noise_args = SimpleNamespace(
        seed=42,
        current_noise_std=0.20,
        current_offset_a=None,
        current_offset_pct=None,
        voltage_offset_v=None,
        temp_offset_c=None,
        voltage_noise_std=None,
        temp_noise_std=None,
        temp_constant=None,
        quantize_current_a=None,
        quantize_voltage_v=None,
        quantize_temp_c=None,
        spike_channel="Voltage[V]",
        spike_magnitude=None,
        spike_period=None,
        spike_prob=None,
        soc_init_error=0.0,
        missing_gap_seconds=0.0,
        missing_samples_every=None,
        missing_samples_pct=None,
        irregular_dt_jitter=None,
        downsample_hz=None,
        drop_pct=None,
        drop_segment_len=None,
    )
    noisy_df, _ = apply_measurement_scenario(current_df, "current_noise", noise_args)
    noise_series = {}
    base_series = {}
    for model in MODEL_ORDER:
        sub = detail[detail["model"] == model].sort_values("current_noise_std")
        row_base = df[(df["model"] == model) & (df["alias"] == "baseline")].iloc[0]
        row_noise = sub.loc[np.isclose(sub["current_noise_std"], 0.20)].iloc[-1]
        base_series[model] = _load_run_series(Path(row_base["run_dir"]), model)
        noise_series[model] = _load_run_series(Path(row_noise["run_dir"]), model)

    t0 = _select_noise_example_window(current_df)
    top_mask = (current_df["time_s"] >= t0) & (current_df["time_s"] <= t0 + 1800.0)
    x_min = (current_df.loc[top_mask, "time_s"].to_numpy(dtype=float) - t0) / 60.0
    ax_top.plot(
        x_min,
        current_df.loc[top_mask, "Current[A]"].to_numpy(dtype=float),
        color="#225ea8",
        lw=2.2,
        label="Baseline current",
    )
    ax_top.plot(
        x_min,
        noisy_df.loc[top_mask, "Current[A]"].to_numpy(dtype=float),
        color="#4589ff",
        lw=1.6,
        alpha=0.95,
        label=r"Current with noise ($\sigma_I = 0.20$ A)",
    )
    ax_top.set_xlabel("Time [min]")
    ax_top.set_ylabel("Current [A]")
    ax_top.legend(frameon=True, ncol=2, loc="upper right")
    ax_top.set_title("(a)", loc="left", fontsize=14, fontweight="bold", pad=6)

    for model in MODEL_ORDER:
        sub = detail[detail["model"] == model].sort_values("current_noise_std")
        jit = jitter[jitter["model"] == model].sort_values("current_noise_std")
        color = MODEL_META[model]["color"]
        axes[0].plot(sub["current_noise_std"], sub["delta_mae"], marker="o", lw=2.2, color=color, label=MODEL_META[model]["short"])
        axes[1].plot(jit["current_noise_std"], jit["delta_p95_step_abs"], marker="o", lw=2.2, color=color, label=MODEL_META[model]["short"])

        base_win = base_series[model][(base_series[model]["time_s"] >= t0) & (base_series[model]["time_s"] <= t0 + 1800.0)].copy()
        noise_win = noise_series[model][(noise_series[model]["time_s"] >= t0) & (noise_series[model]["time_s"] <= t0 + 1800.0)].copy()
        ax_mid.plot(
            (base_win["time_s"] - t0) / 60.0,
            base_win["soc_pred"],
            color=color,
            lw=1.4,
            ls="--",
            alpha=0.55,
        )
        ax_mid.plot(
            (noise_win["time_s"] - t0) / 60.0,
            noise_win["soc_pred"],
            color=color,
            lw=2.0,
            alpha=0.95,
            label=MODEL_META[model]["short"],
        )

    true_ref = base_series[MODEL_ORDER[0]]
    true_ref = true_ref[(true_ref["time_s"] >= t0) & (true_ref["time_s"] <= t0 + 1800.0)].copy()
    ax_mid.plot((true_ref["time_s"] - t0) / 60.0, true_ref["soc_true"], color="#111111", lw=1.2, label="SOC true")
    model_legend = ax_mid.legend(frameon=True, ncol=2, loc="upper left", title="Model colors")
    ax_mid.add_artist(model_legend)
    ax_mid.legend(
        handles=[
            Line2D([0], [0], color="#666666", lw=1.4, ls="--", alpha=0.7, label="Baseline prediction"),
            Line2D([0], [0], color="#666666", lw=2.0, label=r"Noise prediction ($\sigma_I=0.20$ A)"),
            Line2D([0], [0], color="#111111", lw=1.2, label="SOC true"),
        ],
        frameon=True,
        loc="lower right",
    )
    ax_mid.set_xlabel("Time [min]")
    ax_mid.set_ylabel("SOC [-]")
    ax_mid.set_title("(b)", loc="left", fontsize=14, fontweight="bold", pad=6)

    axes[0].set_title("(c)", loc="left", fontsize=14, fontweight="bold", pad=6)
    axes[0].set_xlabel(r"Current-noise std $\sigma_I$ [A]")
    axes[0].set_ylabel(r"$\Delta$MAE")
    axes[0].set_xticks(levels)
    axes[0].legend(frameon=True, ncol=2, loc="upper left")

    axes[1].set_title("(d)", loc="left", fontsize=14, fontweight="bold", pad=6)
    axes[1].set_xlabel(r"Current-noise std $\sigma_I$ [A]")
    axes[1].set_ylabel(r"$\Delta$ p95 $|\Delta \hat{y}|$ per 1 s step")
    axes[1].set_xticks(levels)
    axes[1].legend(frameon=True, ncol=2, loc="upper left")

    _savefig(PAPER_FIGURES_DIR / "Figure_3_noise_robustness.png")


def figure_signal_integrity(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9))
    sub = _ordered(df[df["alias"].isin(["missing_samples", "irregular_sampling", "missing_gap"])].copy())
    width = 0.22
    x = np.arange(len(MODEL_ORDER))
    aliases = ["missing_samples", "irregular_sampling", "missing_gap"]
    labels = {"missing_samples": "Missing samples", "irregular_sampling": "Irregular sampling", "missing_gap": "Burst dropout"}
    hatches = {"missing_samples": "//", "irregular_sampling": "..", "missing_gap": ""}
    for i, alias in enumerate(aliases):
        vals = sub[sub["alias"] == alias].set_index("model").reindex(MODEL_ORDER)["delta_mae"].to_numpy()
        facecolors = []
        edgecolors = []
        for m in MODEL_ORDER:
            edge = MODEL_META[m]["color"]
            fill = MODEL_META[m]["fill"]
            edgecolors.append(edge)
            if alias == "missing_gap":
                facecolors.append(matplotlib.colors.to_rgba(fill, 0.55))
            elif alias == "missing_samples":
                facecolors.append((1.0, 1.0, 1.0, 0.98))
            else:
                facecolors.append(matplotlib.colors.to_rgba(fill, 0.18))
        axes[0].bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=facecolors,
            edgecolor=edgecolors,
            linewidth=2.2,
            hatch=hatches[alias],
            zorder=3,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODEL_META[m]["short"] for m in MODEL_ORDER])
    axes[0].set_ylabel(r"$\Delta$MAE")
    axes[0].set_title("(a)", loc="left", fontsize=14, fontweight="bold", pad=6)
    axes[0].legend(
        handles=[
            Patch(facecolor=(1.0, 1.0, 1.0, 0.98), edgecolor="#666666", linewidth=2.0, hatch=hatches["missing_samples"], label=labels["missing_samples"]),
            Patch(facecolor=matplotlib.colors.to_rgba("#999999", 0.12), edgecolor="#666666", linewidth=2.0, hatch=hatches["irregular_sampling"], label=labels["irregular_sampling"]),
            Patch(facecolor=matplotlib.colors.to_rgba("#999999", 0.50), edgecolor="#666666", linewidth=2.0, hatch=hatches["missing_gap"], label=labels["missing_gap"]),
        ],
        frameon=True,
        loc="upper right",
    )

    gap = _ordered(df[df["alias"] == "missing_gap"].copy())
    vals = gap["recovery_time_h"].fillna(np.nan).to_numpy()
    colors = [MODEL_META[m]["fill"] for m in gap["model"]]
    edges = [MODEL_META[m]["color"] for m in gap["model"]]
    axes[1].bar(np.arange(len(gap)), vals, color=colors, edgecolor=edges, linewidth=2)
    axes[1].set_xticks(np.arange(len(gap)))
    axes[1].set_xticklabels([MODEL_META[m]["short"] for m in gap["model"]])
    axes[1].set_ylabel("Recovery time [h]")
    axes[1].set_title("(b)", loc="left", fontsize=14, fontweight="bold", pad=6)
    for xi, v in enumerate(vals):
        if np.isfinite(v):
            axes[1].text(xi, v + max(vals) * 0.02, f"{v:.1f}", ha="center", va="bottom")

    _savefig(PAPER_FIGURES_DIR / "Figure_4_signal_integrity.png")


def figure_missing_gap_detail(df: pd.DataFrame, cell: str) -> None:
    meas, meta = _load_measurement_scenario(cell, "missing_gap")
    gap_start, gap_end = _find_gap_window(meta, meas["Testtime[s]"].to_numpy(dtype=float))
    win = meas[(meas["Testtime[s]"] >= gap_start - 900) & (meas["Testtime[s]"] <= gap_end + 1800)].copy()

    fig, axes = plt.subplots(4, 1, figsize=(13.2, 10.8), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.4]})
    rel_min = (win["Testtime[s]"] - gap_start) / 60.0
    gap_end_min = (gap_end - gap_start) / 60.0
    axes[0].plot(rel_min, win["Current[A]"], color="#444444", lw=1.8)
    axes[0].set_ylabel("Current [A]")
    axes[0].set_title("(a)", loc="left", fontsize=14, fontweight="bold", pad=6)
    axes[1].plot(rel_min, win["Voltage[V]"], color="#444444", lw=1.8)
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_title("(b)", loc="left", fontsize=14, fontweight="bold", pad=6)
    if "Temperature[°C]" in win.columns:
        axes[2].plot(rel_min, win["Temperature[°C]"], color="#444444", lw=1.8)
    axes[2].set_ylabel("Temp [C]")
    axes[2].set_title("(c)", loc="left", fontsize=14, fontweight="bold", pad=6)
    for ax in axes[:3]:
        ax.axvspan(0.0, gap_end_min, color="#9e9e9e", alpha=0.25)

    for model in MODEL_ORDER:
        row = df[(df["model"] == model) & (df["alias"] == "missing_gap")].iloc[0]
        series = _load_run_series(Path(row["run_dir"]), model)
        seg = series[(series["time_s"] >= gap_start - 900) & (series["time_s"] <= gap_end + 1800)].copy()
        seg = _thin(seg, max_points=2500)
        axes[3].plot((seg["time_s"] - gap_start) / 60.0, seg["soc_pred"], color=MODEL_META[model]["color"], lw=2.2, label=MODEL_META[model]["short"])
    truth = _load_run_series(Path(df[(df["model"] == MODEL_ORDER[0]) & (df["alias"] == "missing_gap")].iloc[0]["run_dir"]), MODEL_ORDER[0])
    truth = truth[(truth["time_s"] >= gap_start - 900) & (truth["time_s"] <= gap_end + 1800)]
    truth = _thin(truth, max_points=2500)
    axes[3].plot((truth["time_s"] - gap_start) / 60.0, truth["soc_true"], color="black", lw=2.0, ls="--", label="Ground truth")
    axes[3].axvspan(0.0, gap_end_min, color="#9e9e9e", alpha=0.25)
    axes[3].set_ylabel("SOC")
    axes[3].set_xlabel("Minutes relative to gap start")
    axes[3].set_title("(d)", loc="left", fontsize=14, fontweight="bold", pad=6)
    axes[3].legend(ncol=3, frameon=True, loc="upper left")

    _savefig(PAPER_FIGURES_DIR / "Figure_5_missing_gap_transition.png")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.8, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    truth = _load_run_series(
        Path(df[(df["model"] == MODEL_ORDER[0]) & (df["alias"] == "missing_gap")].iloc[0]["run_dir"]),
        MODEL_ORDER[0],
    )
    truth = truth[(truth["time_s"] >= gap_end) & (truth["time_s"] <= gap_end + 30 * 3600)].copy()
    truth = _thin(truth, max_points=1800)
    axes[0].plot(
        (truth["time_s"] - gap_end) / 3600.0,
        truth["soc_true"],
        color="black",
        lw=2.0,
        ls="--",
        label="Ground truth",
    )
    for model in MODEL_ORDER:
        row = df[(df["model"] == model) & (df["alias"] == "missing_gap")].iloc[0]
        series = _load_run_series(Path(row["run_dir"]), model)
        seg = series[(series["time_s"] >= gap_end) & (series["time_s"] <= gap_end + 30 * 3600)].copy()
        seg = _thin(seg, max_points=1800)
        rel_h = (seg["time_s"] - gap_end) / 3600.0
        axes[0].plot(rel_h, seg["soc_pred"], color=MODEL_META[model]["color"], lw=2.0, label=MODEL_META[model]["short"])
        axes[1].plot(rel_h, seg["abs_err"], color=MODEL_META[model]["color"], lw=2.0, label=MODEL_META[model]["short"])
        rec = row.get("recovery_time_h")
        if pd.notna(rec):
            axes[1].axvline(float(rec), color=MODEL_META[model]["color"], ls="--", lw=1.2, alpha=0.9)
    axes[0].set_ylabel("SOC")
    axes[0].set_title("(a)", loc="left", fontsize=14, fontweight="bold", pad=6)
    axes[0].legend(ncol=3, frameon=True, loc="upper left")
    axes[1].set_xlabel("Hours after gap end")
    axes[1].set_ylabel("Absolute SOC error")
    axes[1].set_title("(b)", loc="left", fontsize=14, fontweight="bold", pad=6)
    axes[1].legend(ncol=2, frameon=True, loc="upper left")
    _savefig(PAPER_FIGURES_DIR / "Figure_6_missing_gap_recovery.png")


def figure_initial_state(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.34)
    ax_soc = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[1, 0])
    local = pd.read_csv(LOCAL_ANALYSIS_DIR / "local_metrics.csv")
    fair = local[(local["focus_scenario"] == "initial_soc_error") & (local["local_metric"] == "recovery_time_to_baseline_band_fair_h")][["model", "value", "threshold"]]
    fair = fair.rename(columns={"value": "recovery_h", "threshold": "threshold"}).set_index("model")

    plotted_models = []
    start = None
    for model in MODEL_ORDER:
        rows = df[(df["model"] == model) & (df["alias"] == "initial_soc_error")]
        if rows.empty:
            continue
        row = rows.iloc[0]
        series = _load_run_series(Path(row["run_dir"]), model)
        if start is None:
            start = float(series["time_s"].iloc[0])
        seg = series[(series["time_s"] >= start) & (series["time_s"] <= start + 6 * 3600)].copy()
        t_h = (seg["time_s"] - start) / 3600.0
        seg = _thin(seg, max_points=2500)
        t_h = (seg["time_s"] - start) / 3600.0
        ax_soc.plot(t_h, seg["soc_pred"], color=MODEL_META[model]["color"], lw=2.1, label=MODEL_META[model]["short"])
        ax_err.plot(t_h, seg["abs_err"], color=MODEL_META[model]["color"], lw=2.1, label=MODEL_META[model]["short"])
        plotted_models.append(model)
    if start is None or not plotted_models:
        raise ValueError("No initial_soc_error runs available for Figure 7")
    truth = _load_run_series(Path(df[(df["model"] == plotted_models[0]) & (df["alias"] == "initial_soc_error")].iloc[0]["run_dir"]), plotted_models[0])
    truth = truth[(truth["time_s"] >= start) & (truth["time_s"] <= start + 6 * 3600)].copy()
    truth = _thin(truth, max_points=2500)
    ax_soc.plot((truth["time_s"] - start) / 3600.0, truth["soc_true"], color="black", lw=2.0, ls="--", label="Ground truth")

    ax_soc.set_xlabel("Hours since evaluation start")
    ax_soc.set_ylabel("SOC")
    ax_soc.set_title("(a)", loc="left", fontsize=14, fontweight="bold", pad=6)
    ax_soc.legend(frameon=True, ncol=3, loc="upper right")
    ax_err.set_xlabel("Hours since evaluation start")
    ax_err.set_ylabel("Absolute SOC error")
    ax_err.set_title("(b)", loc="left", fontsize=14, fontweight="bold", pad=6)
    ax_err.legend(frameon=True, ncol=2, loc="upper right")

    fig.subplots_adjust(top=0.90)
    _savefig(PAPER_FIGURES_DIR / "Figure_7_initial_state_recovery.png")


def figure_spike_response(df: pd.DataFrame, cell: str) -> None:
    meas, meta = _load_measurement_scenario(cell, "spikes_high")
    spike_idx = _spike_indices(meta)
    time_s = meas["Testtime[s]"].to_numpy(dtype=float)
    if len(spike_idx) == 0:
        return
    rel_grid = np.arange(-60, 181, 1)
    spike_times = time_s[spike_idx]
    candidate = spike_idx[np.searchsorted(spike_times, 6 * 3600.0)] if np.any(spike_times >= 6 * 3600.0) else spike_idx[min(20, len(spike_idx) - 1)]
    t0_rep = int(round(time_s[int(candidate)]))

    fig = plt.figure(figsize=(13.6, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.26)
    ax_soc = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_local = fig.add_subplot(gs[1, 1])
    sub = _ordered(df[df["alias"] == "spikes_high"].copy())
    peak_summary = []

    truth_drawn = False
    for model in MODEL_ORDER:
        row = sub[sub["model"] == model].iloc[0]
        series = _load_run_series(Path(row["run_dir"]), model).copy()
        series["time_int"] = np.round(series["time_s"]).astype(int)
        series = series.drop_duplicates("time_int").set_index("time_int")
        base_row = df[(df["model"] == model) & (df["alias"] == "baseline")].iloc[0]
        base_series = _load_run_series(Path(base_row["run_dir"]), model).copy()
        base_series["time_int"] = np.round(base_series["time_s"]).astype(int)
        base_series = base_series.drop_duplicates("time_int").set_index("time_int")

        sample_times = t0_rep + rel_grid
        rep = series.reindex(sample_times)
        rep_base = base_series.reindex(sample_times)
        if not rep["soc_pred"].isna().any() and not rep_base["soc_pred"].isna().any():
            if not truth_drawn:
                ax_soc.plot(rel_grid, rep["soc_true"].to_numpy(), color="black", lw=2.0, ls="--", label="Ground truth")
                truth_drawn = True
            ax_soc.plot(rel_grid, rep_base["soc_pred"].to_numpy(), color=MODEL_META[model]["color"], lw=1.7, ls="--", alpha=0.75)
            ax_soc.plot(rel_grid, rep["soc_pred"].to_numpy(), color=MODEL_META[model]["color"], lw=2.1, label=MODEL_META[model]["short"])

        aligned = []
        peak_excess = []
        for idx in spike_idx[:40]:
            t0 = int(round(time_s[idx]))
            sample_times = t0 + rel_grid
            window = series.reindex(sample_times)
            if window["abs_err"].isna().any():
                continue
            pre = window.loc[t0 - 10 : t0 - 1, "abs_err"]
            baseline = float(pre.mean()) if len(pre) else float(window["abs_err"].iloc[0])
            excess = window["abs_err"].to_numpy() - baseline
            aligned.append(excess)
            post_mask = rel_grid >= 0
            peak_excess.append(float(np.nanmax(np.maximum(excess[post_mask], 0.0))))
        if aligned:
            arr = np.vstack(aligned)
            mean = np.nanmean(arr, axis=0)
            lo = np.nanpercentile(arr, 25, axis=0)
            hi = np.nanpercentile(arr, 75, axis=0)
            ax_local.plot(rel_grid, mean, color=MODEL_META[model]["color"], lw=2.1, label=MODEL_META[model]["short"])
            ax_local.fill_between(rel_grid, lo, hi, color=MODEL_META[model]["fill"], alpha=0.25)
        peak_summary.append((model, float(np.nanpercentile(peak_excess, 95)) if peak_excess else np.nan))

    peak_df = pd.DataFrame(peak_summary, columns=["model", "p95_peak_excess"]).set_index("model").reindex(MODEL_ORDER).reset_index()
    colors = [MODEL_META[m]["fill"] for m in peak_df["model"]]
    edges = [MODEL_META[m]["color"] for m in peak_df["model"]]
    vals = peak_df["p95_peak_excess"].to_numpy()
    ax_bar.bar(np.arange(len(peak_df)), vals, color=colors, edgecolor=edges, linewidth=2)
    ax_bar.set_xticks(np.arange(len(peak_df)))
    ax_bar.set_xticklabels([MODEL_META[m]["short"] for m in peak_df["model"]])
    ax_bar.set_ylabel("p95 peak excess error")
    ax_bar.set_title("(b)", loc="left", fontsize=14, fontweight="bold", pad=6)
    ax_soc.axvline(0.0, color="black", ls="--", lw=1.2)
    ax_soc.set_xlabel("Seconds relative to a representative voltage spike")
    ax_soc.set_ylabel("SOC")
    ax_soc.set_title("(a)", loc="left", fontsize=14, fontweight="bold", pad=6)
    ax_soc.legend(frameon=True, ncol=3, loc="upper right")

    ax_local.axvline(0.0, color="black", ls="--", lw=1.2)
    ax_local.set_xlabel("Seconds relative to voltage spike")
    ax_local.set_ylabel("Excess absolute error")
    ax_local.set_title("(c)", loc="left", fontsize=14, fontweight="bold", pad=6)
    ax_local.legend(frameon=True, ncol=2, loc="upper right")

    fig.subplots_adjust(top=0.90)
    _savefig(PAPER_FIGURES_DIR / "Figure_8_spike_response.png")


def build_readme(df: pd.DataFrame) -> None:
    bias = df[(df["alias"] == "current_offset")][["model", "mae", "delta_mae"]].copy()
    bias = _ordered(bias)
    bias_lines = []
    for _, row in bias.iterrows():
        meta = MODEL_META[row["model"]]
        bias_lines.append(
            f"- `{meta['short']}`: global MAE `{row['mae']:.4f}`, global $\\Delta$MAE `{row['delta_mae']:.4f}`"
        )

    lines = [
        "# Paper-ready robustness results",
        "",
        "This directory contains curated figures and tables for the robustness-benchmark manuscript.",
        "",
        "## Global vs local evidence",
        "",
        "- `paper_tables/table_key_results.md` contains the complete **global** scenario metrics used for ranking.",
        "- `paper_tables/table_local_behaviour.md` contains the **local** recovery and drift metrics used when global MAE is not informative enough.",
        "- `paper_tables/table_figure_scope.md` maps every paper figure to its global and local evidence.",
        "- Figures that combine global and local evidence are explicitly marked below.",
        "",
        "## Figures",
        "",
        "- `Figure_1_baseline_performance.png`: baseline ranking on clean data (**global**)",
        "- `Figure_2_current_bias.png`: current-bias penalty (**global**) plus early drift trajectory (**local**)",
        "- `Figure_3_noise_robustness.png`: global noise penalties plus local rolling-error drift",
        "- `Figure_4_signal_integrity.png`: global integrity penalties plus local burst-dropout recovery time",
        "- `Figure_5_missing_gap_transition.png`: sensor and SOC transition around the dropout window (**local**)",
        "- `Figure_6_missing_gap_recovery.png`: post-gap recovery trajectory (**local**)",
        "- `Figure_7_initial_state_recovery.png`: CV-free initial-state recovery analysis (**local**; global MAE in key-results table)",
        "- `Figure_8_spike_response.png`: global spike penalty plus local per-spike response",
        "",
        "## Current-bias interpretation to preserve in the manuscript",
        "",
        "- Figure 2 is intentionally mixed-scope.",
        "- The left panel shows the **global** current-bias penalty via $\\Delta$MAE on the complete run.",
        "- The right panel shows a **local** early-window drift view (first 12 h) to expose the physical bias mechanism.",
        "- This figure must not be described as if the right panel were the full-run behaviour.",
        "- For the complete-run global ranking under current bias, use the values in `table_key_results.md`.",
        "- Separate verification plots for the full run are stored under `results/figures_test/`.",
        "",
        "Current-bias global ranking:",
        *bias_lines,
        "",
        "## Tables",
        "",
        "- `table_baseline.md`: clean-data baseline metrics (**global**)",
        "- `table_key_results.md`: key scenario metrics and penalties (**global**)",
        "- `table_local_behaviour.md`: local recovery and drift metrics (**local**)",
        "- `table_figure_scope.md`: explicit global/local mapping for each figure",
        "",
        "## Model classes",
        "",
    ]
    for model in MODEL_ORDER:
        meta = MODEL_META[model]
        lines.append(f"- `{meta['short']}`: {meta['label']}")
    (RESULTS_DIR / "PAPER_RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign_tag", default=CAMPAIGN_DEFAULT)
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    args = ap.parse_args()

    _setup_style()

    campaign_dir = ROOT / "campaigns" / args.campaign_tag
    df = load_campaign_rows(campaign_dir)
    if df.empty:
        raise SystemExit(f"No completed runs found in {campaign_dir}")

    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building tables...", flush=True)
    build_tables(df)
    print("Figure 1 baseline...", flush=True)
    figure_baseline(df)
    print("Figure 2 current bias...", flush=True)
    figure_offset(df)
    print("Figure 3 noise...", flush=True)
    figure_noise(df)
    print("Figure 4 signal integrity...", flush=True)
    figure_signal_integrity(df)
    print("Figure 5-6 missing gap...", flush=True)
    figure_missing_gap_detail(df, args.cell)
    print("Figure 7 initial state...", flush=True)
    figure_initial_state(df)
    print("Figure 8 spikes...", flush=True)
    figure_spike_response(df, args.cell)
    print("Writing README...", flush=True)
    build_readme(df)

    manifest = {
        "campaign_tag": args.campaign_tag,
        "paper_tables_dir": str(PAPER_TABLES_DIR),
        "paper_figures_dir": str(PAPER_FIGURES_DIR),
        "n_rows": int(len(df)),
    }
    (RESULTS_DIR / "paper_results_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
