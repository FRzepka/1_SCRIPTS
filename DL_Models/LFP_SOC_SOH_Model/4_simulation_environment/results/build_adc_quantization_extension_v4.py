from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
TABLES_OUT = RESULTS_DIR / "paper_tables_v4_adc_extension"
FIGURES_OUT = RESULTS_DIR / "paper_figures_v4_adc_extension"
DATA_ROOT = Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
CELL = "MGFarm_18650_C07"

MODEL_ORDER = [
    "CC_1.0.0",
    "CC_SOH_1.0.0",
    "ECM_0.0.3",
    "SOC_SOH_1.7.0.0_0.1.2.3",
]

MODEL_META = {
    "CC_1.0.0": {"label": "Direct measurement", "short": "DM", "color": "#6e2fc4"},
    "CC_SOH_1.0.0": {"label": "Hybrid direct measurement", "short": "HDM", "color": "#08bdba"},
    "ECM_0.0.3": {"label": "Hybrid ECM", "short": "HECM", "color": "#d4bbff"},
    "SOC_SOH_1.7.0.0_0.1.2.3": {"label": "Data-driven", "short": "DD", "color": "#4589ff"},
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _find_latest_summary(model: str, alias: str, tag_fragment: str) -> Path:
    root = ROOT / model / "runs" / alias
    matches = sorted(root.glob(f"*{tag_fragment}*/summary.json"))
    if not matches:
        raise FileNotFoundError(f"No summary found for model={model}, alias={alias}, tag~{tag_fragment}")
    return matches[-1]


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
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
            "savefig.dpi": 240,
        }
    )


def _pick_dynamic_window(df: pd.DataFrame, seconds: int = 300, after_s: int = 600) -> tuple[float, float]:
    cur = df["Current[A]"].to_numpy(dtype=float)
    t = df["Testtime[s]"].to_numpy(dtype=float)
    mask = t >= after_s
    idx = np.flatnonzero(mask)
    if len(idx) < seconds:
        return float(t[0]), float(t[min(len(t) - 1, seconds)])
    start_candidates = idx[: len(idx) - seconds + 1]
    best_i = int(start_candidates[0])
    best_score = -np.inf
    for i in start_candidates[::10]:
        window = cur[i : i + seconds]
        score = float(np.std(window))
        if score > best_score:
            best_score = score
            best_i = int(i)
    return float(t[best_i]), float(t[best_i + seconds - 1])


def _build_table() -> pd.DataFrame:
    rows = []
    tag = "v4_soc170_s30ft_soh0125_s30ft"
    for model in MODEL_ORDER:
        base_summary = _load_json(_find_latest_summary(model, "baseline", f"{tag}_baseline"))
        adc_summary = _load_json(_find_latest_summary(model, "adc_quantization", f"{tag}_adc_quantization"))
        rows.append(
            {
                "model": model,
                "class": MODEL_META[model]["label"],
                "short": MODEL_META[model]["short"],
                "baseline_mae": base_summary["mae"],
                "adc_mae": adc_summary["mae"],
                "delta_mae": adc_summary["mae"] - base_summary["mae"],
                "baseline_rmse": base_summary["rmse"],
                "adc_rmse": adc_summary["rmse"],
                "delta_rmse": adc_summary["rmse"] - base_summary["rmse"],
                "adc_p95_error": adc_summary["p95_error"],
                "quantize_current_a": adc_summary["scenario_meta"]["quantize_current_a"],
                "quantize_voltage_v": adc_summary["scenario_meta"]["quantize_voltage_v"],
                "quantize_temp_c": adc_summary["scenario_meta"]["quantize_temp_c"],
                "run_dir": str(_find_latest_summary(model, "adc_quantization", f"{tag}_adc_quantization").parent),
            }
        )
    df = pd.DataFrame(rows)
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_OUT / "table_adc_quantization_v4.csv", index=False)
    (TABLES_OUT / "table_adc_quantization_v4.md").write_text(df.round(6).to_markdown(index=False), encoding="utf-8")
    return df


def _plot_figure(df_table: pd.DataFrame) -> None:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from robustness_common import apply_measurement_scenario, load_cell_dataframe

    raw = load_cell_dataframe(DATA_ROOT, CELL)
    quant, meta = apply_measurement_scenario(
        raw.copy(),
        "adc_quantization",
        SimpleNamespace(quantize_current_a=0.01, quantize_voltage_v=0.005, quantize_temp_c=0.5, seed=42),
    )

    t0, t1 = _pick_dynamic_window(raw, seconds=300, after_s=600)
    raw_seg = raw[(raw["Testtime[s]"] >= t0) & (raw["Testtime[s]"] <= t1)].copy()
    quant_seg = quant[(quant["Testtime[s]"] >= t0) & (quant["Testtime[s]"] <= t1)].copy()
    rel_t = raw_seg["Testtime[s]"].to_numpy(dtype=float) - t0

    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.15], hspace=0.32, wspace=0.25)
    ax_i = fig.add_subplot(gs[0, 0])
    ax_u = fig.add_subplot(gs[1, 0], sharex=ax_i)
    ax_bar = fig.add_subplot(gs[:, 1])

    ax_i.plot(rel_t, raw_seg["Current[A]"].to_numpy(dtype=float), color="#222222", lw=1.8, label="Raw")
    ax_i.step(rel_t, quant_seg["Current[A]"].to_numpy(dtype=float), where="post", color="#d62728", lw=1.6, label="Quantized")
    ax_i.set_ylabel("Current [A]")
    ax_i.set_title("(a) Current quantization", loc="left", fontweight="bold")
    ax_i.legend(frameon=True, loc="upper right")

    ax_u.plot(rel_t, raw_seg["Voltage[V]"].to_numpy(dtype=float), color="#222222", lw=1.8, label="Raw")
    ax_u.step(rel_t, quant_seg["Voltage[V]"].to_numpy(dtype=float), where="post", color="#d62728", lw=1.6, label="Quantized")
    ax_u.set_ylabel("Voltage [V]")
    ax_u.set_xlabel("Seconds within representative local window")
    ax_u.set_title("(b) Voltage quantization", loc="left", fontweight="bold")
    ax_u.legend(frameon=True, loc="upper right")

    x = np.arange(len(df_table))
    colors = [MODEL_META[m]["color"] for m in df_table["model"]]
    bars = ax_bar.bar(x, df_table["delta_mae"], color=colors, alpha=0.92)
    ax_bar.axhline(0.0, color="#666666", lw=1.0)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(df_table["short"])
    ax_bar.set_ylabel(r"$\Delta$MAE vs baseline")
    ax_bar.set_title("(c) Global ADC-quantization penalty", loc="left", fontweight="bold")
    for bar, (_, row) in zip(bars, df_table.iterrows()):
        y = float(row["delta_mae"])
        va = "bottom" if y >= 0 else "top"
        dy = 0.00005 if y >= 0 else -0.00005
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            y + dy,
            f"{y:+.4f}\nMAE={row['adc_mae']:.4f}",
            ha="center",
            va=va,
            fontsize=9,
        )
    ax_bar.set_ylim(min(df_table["delta_mae"].min() - 0.0002, -0.0005), max(df_table["delta_mae"].max() + 0.00035, 0.0008))
    fig.suptitle(
        "ADC Quantization with v4 Models\n"
        f"Current step = {meta['quantize_current_a']:.2f} A, Voltage step = {meta['quantize_voltage_v']:.3f} V, Temperature step = {meta['quantize_temp_c']:.1f} °C",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_OUT / "Figure_ADC_quantization_v4.png")
    plt.close(fig)


def _write_summary_note(df: pd.DataFrame) -> None:
    best = df.loc[df["delta_mae"].idxmin()]
    worst = df.loc[df["delta_mae"].idxmax()]
    lines = [
        "# ADC Quantization Extension (v4)",
        "",
        "This extension uses the already available `adc_quantization` runs from the v4 benchmark setup.",
        "",
        "Applied quantization steps:",
        f"- Current: `{df['quantize_current_a'].iloc[0]:.2f} A`",
        f"- Voltage: `{df['quantize_voltage_v'].iloc[0]:.3f} V`",
        f"- Temperature: `{df['quantize_temp_c'].iloc[0]:.1f} °C`",
        "",
        "Main observation:",
        "- Under these quantization levels, the global SOC error changes only marginally for all four estimator classes.",
        f"- Largest MAE increase: `{worst['short']}` with `ΔMAE = {worst['delta_mae']:+.6f}`.",
        f"- Smallest / slightly improved case: `{best['short']}` with `ΔMAE = {best['delta_mae']:+.6f}`.",
        "",
        "Interpretation:",
        "- ADC quantization appears to be much less discriminative than current bias, voltage noise, missing samples, or initialization mismatch for the present benchmark configuration.",
        "- This makes it a reasonable appendix or supporting scenario, but not necessarily a core main-text figure unless embedded sensor resolution is a central discussion point.",
        "",
    ]
    (TABLES_OUT / "adc_quantization_note_v4.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _setup_style()
    df = _build_table()
    _plot_figure(df)
    _write_summary_note(df)
    print(f"Wrote {TABLES_OUT / 'table_adc_quantization_v4.csv'}")
    print(f"Wrote {FIGURES_OUT / 'Figure_ADC_quantization_v4.png'}")


if __name__ == "__main__":
    main()
