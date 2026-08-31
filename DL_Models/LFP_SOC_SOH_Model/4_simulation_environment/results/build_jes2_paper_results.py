from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jes2_protocol import (
    MODEL_ORDER,
    PRIMARY_STOCHASTIC_ALIASES,
    SCENARIOS,
    SCENARIO_LABELS,
    STOCHASTIC_ALIASES,
)
from characterize_jes2_cells import CELL_LOAD_CLASSES, characterize_cell
from jes2_plot_style import MODEL_COLORS, TU_RED, clean_axes, model_fill, save_figure, setup_style


PRIMARY_CONDITIONS = {"DM": "none", "HDM": "lstm_h1", "HECM": "lstm_h1", "DD": "lstm_h1"}
INTERNAL_REFERENCE_ALIASES = {"missing_gap_baseline_48h"}
METRICS = [
    "mae",
    "rmse",
    "p95_error",
    "max_error",
    "bias",
    "jump_count_gt_5pct",
    "gap_net_charge_ah",
    "gap_throughput_ah",
    "gap_reference_soc_change",
    "evaluation_samples",
]


def load_campaign(manifest_path: Path) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for record in manifest.get("runs", []):
        if record.get("status") not in {
            "completed", "skipped_existing", "reused_common_interval"
        }:
            continue
        summary_path = Path(
            record.get("source_summary", Path(record["out_dir"]) / "summary.json")
        )
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "cell": record["cell"],
            "alias": record["alias"],
            "scenario": record["scenario"],
            "seed": int(record["seed"]),
            "model": record["model"],
            "soh_mode": record.get("soh_mode", "none"),
            "soh_condition": record.get("soh_condition", "none"),
            "soh_publish_intervals": int(record.get("soh_publish_intervals", 0)),
            "out_dir": record["out_dir"],
            "summary_path": str(summary_path),
            "window_id": record.get("window_id", "single_window"),
            "window_soh_state": record.get("soh_state", "all"),
            "cell_load_class": record.get("cell_load_class", "unassigned"),
            "start_row": int(record.get("start_row", summary.get("start_row", 0))),
            "max_rows": int(record.get("max_rows", summary.get("max_rows", 0))),
            "evaluation_start_sample": int(
                summary.get(
                    "evaluation_start_sample",
                    manifest.get("protocol", {}).get("common_evaluation_start_sample", 0),
                )
            ),
        }
        for metric in METRICS:
            row[metric] = summary.get(metric)
        if row["evaluation_samples"] is None and record["model"] == "DD":
            row["evaluation_samples"] = max(
                0, row["max_rows"] - row["evaluation_start_sample"]
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No completed JES2 runs found in {manifest_path}")
    # Campaign manifests preserve historical metadata. Always apply the current,
    # frozen analysis classification so reclassification never requires reruns.
    frame["cell_load_class"] = frame["cell"].map(
        lambda cell: CELL_LOAD_CLASSES.get(str(cell).rsplit("_", 1)[-1], "unassigned")
    )
    return manifest, frame


def add_baseline_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "window_id" not in frame.columns:
        frame["window_id"] = "single_window"
    if "max_rows" not in frame.columns:
        frame["max_rows"] = 0
    baseline = frame[frame["alias"] == "baseline"].copy()
    baseline = baseline.sort_values("seed").drop_duplicates(
        ["cell", "window_id", "model", "soh_condition"]
    )
    baseline = baseline[["cell", "window_id", "model", "soh_condition", "mae", "rmse"]].rename(
        columns={"mae": "baseline_mae", "rmse": "baseline_rmse"}
    )
    merged = frame.merge(
        baseline, on=["cell", "window_id", "model", "soh_condition"], how="left"
    )
    merged["delta_mae"] = merged["mae"] - merged["baseline_mae"]
    merged["delta_rmse"] = merged["rmse"] - merged["baseline_rmse"]

    event = frame[frame["alias"] == "missing_gap_baseline_48h"].copy()
    event = event.sort_values("seed").drop_duplicates(
        ["cell", "window_id", "model", "soh_condition"]
    )
    event = event[["cell", "window_id", "model", "soh_condition", "mae", "rmse", "max_rows"]].rename(
        columns={"mae": "event_baseline_mae", "rmse": "event_baseline_rmse", "max_rows": "event_baseline_rows"}
    )
    merged = merged.merge(event, on=["cell", "window_id", "model", "soh_condition"], how="left")
    gap = merged["alias"] == "missing_gap_1h"
    duration_matched = gap & (merged["max_rows"] == merged["event_baseline_rows"])
    merged.loc[duration_matched, "baseline_mae"] = merged.loc[duration_matched, "event_baseline_mae"]
    merged.loc[duration_matched, "baseline_rmse"] = merged.loc[duration_matched, "event_baseline_rmse"]
    merged.loc[duration_matched, "delta_mae"] = (
        merged.loc[duration_matched, "mae"] - merged.loc[duration_matched, "event_baseline_mae"]
    )
    merged.loc[duration_matched, "delta_rmse"] = (
        merged.loc[duration_matched, "rmse"] - merged.loc[duration_matched, "event_baseline_rmse"]
    )
    unmatched_gap = gap & ~duration_matched
    merged.loc[unmatched_gap, ["baseline_mae", "baseline_rmse", "delta_mae", "delta_rmse"]] = np.nan
    return merged


def load_stratified_run_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metadata = [
        "cell", "alias", "scenario", "seed", "model", "soh_mode",
        "soh_condition", "soh_publish_intervals", "summary_path", "window_id",
        "window_soh_state", "cell_load_class",
        "max_rows",
    ]
    for record in frame.itertuples(index=False):
        summary = json.loads(Path(record.summary_path).read_text(encoding="utf-8"))
        base = {key: getattr(record, key) for key in metadata}
        for stratum in summary.get("stratified_metrics", []):
            rows.append({**base, **stratum})
    return pd.DataFrame(rows)


def add_stratified_baseline_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = ["cell", "window_id", "model", "soh_condition", "dimension", "stratum"]
    baseline = frame[frame["alias"] == "baseline"].sort_values("seed").drop_duplicates(keys)
    baseline = baseline[keys + ["mae", "rmse"]].rename(
        columns={"mae": "baseline_mae", "rmse": "baseline_rmse"}
    )
    merged = frame.merge(baseline, on=keys, how="left")
    merged["delta_mae"] = merged["mae"] - merged["baseline_mae"]
    merged["delta_rmse"] = merged["rmse"] - merged["baseline_rmse"]
    event = frame[frame["alias"] == "missing_gap_baseline_48h"].sort_values("seed").drop_duplicates(keys)
    event = event[keys + ["mae", "rmse", "max_rows"]].rename(
        columns={"mae": "event_baseline_mae", "rmse": "event_baseline_rmse", "max_rows": "event_baseline_rows"}
    )
    merged = merged.merge(event, on=keys, how="left")
    gap = merged["alias"] == "missing_gap_1h"
    duration_matched = gap & (merged["max_rows"] == merged["event_baseline_rows"])
    merged.loc[duration_matched, "baseline_mae"] = merged.loc[duration_matched, "event_baseline_mae"]
    merged.loc[duration_matched, "baseline_rmse"] = merged.loc[duration_matched, "event_baseline_rmse"]
    merged.loc[duration_matched, "delta_mae"] = merged.loc[duration_matched, "mae"] - merged.loc[duration_matched, "event_baseline_mae"]
    merged.loc[duration_matched, "delta_rmse"] = merged.loc[duration_matched, "rmse"] - merged.loc[duration_matched, "event_baseline_rmse"]
    merged.loc[gap & ~duration_matched, ["baseline_mae", "baseline_rmse", "delta_mae", "delta_rmse"]] = np.nan
    return merged


def primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    mask = np.zeros(len(frame), dtype=bool)
    for model, condition in PRIMARY_CONDITIONS.items():
        mask |= (frame["model"] == model) & (frame["soh_condition"] == condition)
    return frame.loc[mask].copy()


def hierarchical_stats(group: pd.DataFrame, value: str, bootstrap_samples: int, seed: int) -> dict:
    valid = group.dropna(subset=[value])
    if "seed" in valid.columns:
        # Windows are repeated observations within a cell, not independent
        # experimental units. Average them before seed/cell resampling.
        valid = valid.groupby(["cell", "seed"], as_index=False)[value].mean()
    cells = sorted(valid["cell"].unique())
    if not cells:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_cells": 0, "n_runs": 0}
    cell_means = valid.groupby("cell")[value].mean()
    point = float(cell_means.mean())
    if len(cells) == 1 or bootstrap_samples <= 0:
        return {
            "mean": point,
            "ci_low": point,
            "ci_high": point,
            "n_cells": len(cells),
            "n_runs": len(valid),
        }

    rng = np.random.default_rng(seed)
    # Draw cells first, then independently resample the nested seed values for
    # every occurrence of a source cell. This is the vectorized equivalent of
    # the original nested loop and keeps 10,000-draw analyses practical.
    sampled_cell_indices = rng.integers(
        0, len(cells), size=(bootstrap_samples, len(cells))
    )
    nested_means = np.empty_like(sampled_cell_indices, dtype=np.float64)
    for cell_idx, cell in enumerate(cells):
        selected = sampled_cell_indices == cell_idx
        occurrence_count = int(selected.sum())
        if occurrence_count == 0:
            continue
        values = valid.loc[valid["cell"] == cell, value].to_numpy(dtype=float)
        nested_means[selected] = rng.choice(
            values, size=(occurrence_count, len(values)), replace=True
        ).mean(axis=1)
    draws = nested_means.mean(axis=1)
    return {
        "mean": point,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "n_cells": len(cells),
        "n_runs": len(valid),
    }


def aggregate_metrics(frame: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["alias", "model", "soh_mode", "soh_condition", "soh_publish_intervals"]
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        base = dict(zip(keys, group_key))
        for metric in [*METRICS, "delta_mae", "delta_rmse"]:
            stats = hierarchical_stats(group, metric, bootstrap_samples, seed=1905 + len(rows))
            if stats["n_runs"]:
                rows.append({**base, "metric": metric, **stats})
    return pd.DataFrame(rows)


def aggregate_stratified_metrics(
    frame: pd.DataFrame,
    bootstrap_samples: int,
    include_load_class: bool = False,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    keys = ["dimension", "stratum"]
    if include_load_class:
        keys.append("cell_load_class")
    keys.extend(["alias", "model", "soh_mode", "soh_condition", "soh_publish_intervals"])
    rows = []
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        base = dict(zip(keys, group_key))
        for metric in ["mae", "rmse", "bias", "p95_error", "coverage_fraction", "delta_mae", "delta_rmse"]:
            stats = hierarchical_stats(group, metric, bootstrap_samples, seed=3905 + len(rows))
            if stats["n_runs"]:
                rows.append({**base, "metric": metric, **stats})
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = abs(float(np.mean(values)))
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(values))), dtype=np.float64)
    null = np.abs(np.mean(signs * np.abs(values), axis=1))
    return float(np.mean(null >= observed - 1e-15))


def paired_bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) <= 1 or samples <= 0:
        mean = float(np.mean(values)) if len(values) else np.nan
        return mean, mean
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def add_holm_adjustment(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result["p_holm"] = np.nan
    for _, family in result.groupby(group_columns, dropna=False, sort=False):
        valid = family["p_exact"].dropna().sort_values()
        running = 0.0
        total = len(valid)
        for rank, (index, p_value) in enumerate(valid.items()):
            adjusted = min(1.0, float(p_value) * (total - rank))
            running = max(running, adjusted)
            result.loc[index, "p_holm"] = running
    return result


def paired_effect_summary(values: np.ndarray, bootstrap_samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    ci_low, ci_high = paired_bootstrap_ci(values, bootstrap_samples, seed)
    return {
        "mean_difference": mean,
        "median_difference": float(np.median(values)),
        "sd_difference": sd,
        "standardized_effect_dz": mean / sd if sd > 0.0 else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_exact": exact_sign_flip_pvalue(values),
        "n_cells": int(len(values)),
    }


def build_paired_statistical_tests(frame: pd.DataFrame, bootstrap_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = primary_rows(frame)
    scenario_rows = []
    disturbed = primary[primary["alias"] != "baseline"]
    for (alias, model), group in disturbed.groupby(["alias", "model"], sort=True):
        cell_values = group.groupby("cell")["delta_mae"].mean().dropna().to_numpy(dtype=float)
        if len(cell_values) < 2:
            continue
        seed_counts = group.groupby("cell")["seed"].nunique()
        scenario_rows.append({
            "test_family": "scenario_vs_cell_matched_baseline",
            "alias": alias,
            "model": model,
            "metric": "delta_mae",
            "stochastic_scenario": alias in STOCHASTIC_ALIASES,
            "seeds_per_cell_min": int(seed_counts.min()),
            "seeds_per_cell_max": int(seed_counts.max()),
            **paired_effect_summary(cell_values, bootstrap_samples, 4905 + len(scenario_rows)),
        })
    scenario_tests = add_holm_adjustment(
        pd.DataFrame(scenario_rows), ["test_family", "alias"]
    )

    pair_rows = []
    aliases = ["baseline", *sorted(set(primary["alias"]) & STOCHASTIC_ALIASES)]
    for alias in aliases:
        part = primary[primary["alias"] == alias]
        value = "mae" if alias == "baseline" else "delta_mae"
        cell_model = part.groupby(["cell", "model"])[value].mean().unstack("model")
        for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
            if model_a not in cell_model or model_b not in cell_model:
                continue
            difference = (cell_model[model_b] - cell_model[model_a]).dropna().to_numpy(dtype=float)
            if len(difference) < 2:
                continue
            pair_rows.append({
                "test_family": "paired_model_comparison",
                "alias": alias,
                "metric": value,
                "model_a": model_a,
                "model_b": model_b,
                "difference_definition": "model_b_minus_model_a",
                **paired_effect_summary(difference, bootstrap_samples, 5905 + len(pair_rows)),
            })
    pair_tests = add_holm_adjustment(
        pd.DataFrame(pair_rows), ["test_family", "alias"]
    )
    return scenario_tests, pair_tests


def build_soh_ablation(frame: pd.DataFrame, bootstrap_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["cell", "window_id", "alias", "seed", "model"]
    lstm = frame[frame["soh_condition"] == "lstm_h1"][keys + ["mae", "rmse"]]
    reference = frame[frame["soh_condition"] == "reference_h1"][keys + ["mae", "rmse"]]
    paired = lstm.merge(reference, on=keys, suffixes=("_lstm", "_reference"))
    if paired.empty:
        return paired, pd.DataFrame()
    paired["reference_minus_lstm_mae"] = paired["mae_reference"] - paired["mae_lstm"]
    paired["reference_minus_lstm_rmse"] = paired["rmse_reference"] - paired["rmse_lstm"]
    rows = []
    for (alias, model), group in paired.groupby(["alias", "model"], sort=True):
        for metric in ["reference_minus_lstm_mae", "reference_minus_lstm_rmse"]:
            rows.append(
                {
                    "alias": alias,
                    "model": model,
                    "metric": metric,
                    **hierarchical_stats(group, metric, bootstrap_samples, seed=2905 + len(rows)),
                }
            )
    return paired, pd.DataFrame(rows)


def metric_slice(aggregate: pd.DataFrame, alias: str | Iterable[str], metric: str) -> pd.DataFrame:
    aliases = [alias] if isinstance(alias, str) else list(alias)
    rows = aggregate[(aggregate["alias"].isin(aliases)) & (aggregate["metric"] == metric)].copy()
    keep = np.zeros(len(rows), dtype=bool)
    for model, condition in PRIMARY_CONDITIONS.items():
        keep |= (rows["model"] == model) & (rows["soh_condition"] == condition)
    return rows.loc[keep]


def draw_model_bars(ax, rows: pd.DataFrame, ylabel: str, title: str) -> None:
    indexed = rows.set_index("model")
    models = [model for model in MODEL_ORDER if model in indexed.index]
    values = np.asarray([indexed.loc[model, "mean"] for model in models], dtype=float)
    low = np.asarray([indexed.loc[model, "ci_low"] for model in models], dtype=float)
    high = np.asarray([indexed.loc[model, "ci_high"] for model in models], dtype=float)
    yerr = np.vstack([values - low, high - values])
    bars = ax.bar(
        np.arange(len(models)),
        values,
        yerr=yerr,
        capsize=3,
        color=[model_fill(model) for model in models],
        edgecolor=[MODEL_COLORS[model] for model in models],
        linewidth=1.5,
    )
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    clean_axes(ax)


def plot_baseline(raw: pd.DataFrame, aggregate: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for ax, metric, label in zip(axes, ["mae", "rmse"], ["MAE [SOC]", "RMSE [SOC]"]):
        rows = metric_slice(aggregate, "baseline", metric)
        draw_model_bars(ax, rows, label, f"Baseline {metric.upper()}")
        primary = primary_rows(raw)
        primary = primary[primary["alias"] == "baseline"]
        for model_idx, model in enumerate(MODEL_ORDER):
            values = primary.loc[primary["model"] == model, metric].dropna().to_numpy(dtype=float)
            ax.scatter(
                np.full(len(values), model_idx) + np.linspace(-0.09, 0.09, max(len(values), 1)),
                values,
                s=20,
                facecolor="white",
                edgecolor="#222222",
                linewidth=0.7,
                zorder=4,
            )
    fig.suptitle(
        "Independent holdout-cell baseline "
        "(dots: cell/SOH windows; bars: cell-macro mean with 95% CI)"
    )
    fig.tight_layout()
    save_figure(fig, out / "Figure_04_Baseline_Performance.png")


def plot_sweep(aggregate: pd.DataFrame, aliases: list[str], out_path: Path, title: str) -> None:
    rows = metric_slice(aggregate, aliases, "delta_mae")
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    for model in MODEL_ORDER:
        part = rows[rows["model"] == model].set_index("alias").reindex(aliases)
        if part["mean"].isna().all():
            continue
        x = np.arange(len(aliases))
        mean = part["mean"].to_numpy(dtype=float)
        low = part["ci_low"].to_numpy(dtype=float)
        high = part["ci_high"].to_numpy(dtype=float)
        ax.errorbar(x, mean, yerr=np.vstack([mean - low, high - mean]), marker="o", capsize=3,
                    linewidth=2.0, color=MODEL_COLORS[model], label=model)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_xticks(np.arange(len(aliases)), [SCENARIO_LABELS[a] for a in aliases], rotation=18, ha="right")
    ax.set_ylabel(r"$\Delta$MAE [SOC]")
    ax.set_title(title)
    ax.legend(ncol=4, frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_grouped_scenarios(aggregate: pd.DataFrame, aliases: list[str], out_path: Path, title: str) -> None:
    rows = metric_slice(aggregate, aliases, "delta_mae")
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    x = np.arange(len(aliases))
    width = 0.19
    for idx, model in enumerate(MODEL_ORDER):
        part = rows[rows["model"] == model].set_index("alias").reindex(aliases)
        mean = part["mean"].to_numpy(dtype=float)
        low = part["ci_low"].to_numpy(dtype=float)
        high = part["ci_high"].to_numpy(dtype=float)
        xpos = x + (idx - 1.5) * width
        ax.bar(xpos, mean, width, yerr=np.vstack([mean - low, high - mean]), capsize=2,
               color=model_fill(model), edgecolor=MODEL_COLORS[model], linewidth=1.5,
               label=model)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_xticks(x, [SCENARIO_LABELS[a] for a in aliases], rotation=16, ha="right")
    ax.set_ylabel(r"$\Delta$MAE [SOC]")
    ax.set_title(title)
    ax.legend(ncol=4, frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_spikes(aggregate: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    draw_model_bars(axes[0], metric_slice(aggregate, "voltage_spikes", "delta_mae"),
                    r"$\Delta$MAE [SOC]", "Global spike penalty")
    draw_model_bars(axes[1], metric_slice(aggregate, "voltage_spikes", "jump_count_gt_5pct"),
                    "Output jumps >5%", "Transient output susceptibility")
    fig.tight_layout()
    save_figure(fig, out / "Figure_11_Voltage_Spike_Response.png")


def plot_heatmap(aggregate: pd.DataFrame, out: Path) -> None:
    rows = primary_rows_from_aggregate(aggregate, "delta_mae")
    rows = rows[~rows["alias"].isin(["baseline", "initial_soc_error"])]
    pivot = rows.pivot(index="model", columns="alias", values="mean").reindex(MODEL_ORDER)
    aliases = [alias for alias, _, _ in SCENARIOS if alias in pivot.columns]
    pivot = pivot.reindex(columns=aliases)
    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    cmap = LinearSegmentedColormap.from_list("diss_diverging", ["#566b78", "#f7f7f7", "#b6302d"])
    limit = max(float(np.nanmax(np.abs(pivot.to_numpy(dtype=float)))), 1e-6)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(pivot.columns)), [SCENARIO_LABELS[a] for a in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.55 * limit else "#222222")
    fig.colorbar(image, ax=ax, label=r"Cell-macro $\Delta$MAE [SOC]", shrink=0.85)
    ax.set_title("Cross-scenario robustness with independent-cell macro aggregation")
    fig.tight_layout()
    save_figure(fig, out / "Figure_12_Cross_Scenario_Heatmap.png")


def primary_rows_from_aggregate(aggregate: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = aggregate[aggregate["metric"] == metric].copy()
    keep = np.zeros(len(rows), dtype=bool)
    for model, condition in PRIMARY_CONDITIONS.items():
        keep |= (rows["model"] == model) & (rows["soh_condition"] == condition)
    return rows.loc[keep]


def plot_soh_ablation(ablation: pd.DataFrame, out: Path) -> None:
    if ablation.empty or "metric" not in ablation.columns:
        return
    rows = ablation[ablation["metric"] == "reference_minus_lstm_mae"].copy()
    if rows.empty:
        return
    aliases = [alias for alias in SCENARIO_LABELS if alias in set(rows["alias"])]
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    x = np.arange(len(aliases))
    width = 0.25
    models = [model for model in ["HDM", "HECM", "DD"] if model in set(rows["model"])]
    for idx, model in enumerate(models):
        part = rows[rows["model"] == model].set_index("alias").reindex(aliases)
        mean = part["mean"].to_numpy(dtype=float)
        low = part["ci_low"].to_numpy(dtype=float)
        high = part["ci_high"].to_numpy(dtype=float)
        ax.bar(x + (idx - 1) * width, mean, width, yerr=np.vstack([mean - low, high - mean]), capsize=2,
               color=model_fill(model), edgecolor=MODEL_COLORS[model], linewidth=1.5, label=model)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, [SCENARIO_LABELS[a] for a in aliases], rotation=20, ha="right")
    ax.set_ylabel("Reference-SOH MAE minus LSTM-SOH MAE [SOC]")
    ax.set_title("Shared-SOH error-propagation ablation (negative values favor ideal reference SOH)")
    ax.legend(ncol=3, frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, out / "Figure_15_SOH_Ablation.png")


def plot_cell_coverage(cells: pd.DataFrame, out: Path) -> None:
    labels = [cell.split("_")[-1] for cell in cells["cell"]]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    axes[0].bar(labels, cells["soh_min"], color=(0.84, 0.15, 0.16, 0.38),
                edgecolor=TU_RED, linewidth=1.5)
    axes[0].set_ylim(0.5, 1.01)
    axes[0].set_ylabel("Minimum reference SOH")
    axes[0].set_title("Aging-state coverage")
    axes[1].bar(labels, cells["temperature_max_c"] - cells["temperature_min_c"],
                color=(0.58, 0.40, 0.74, 0.38), edgecolor="#9467bd", linewidth=1.5)
    axes[1].set_ylabel("Temperature span [degC]")
    axes[1].set_title("Thermal coverage")
    axes[2].bar(labels, cells["abs_current_p95_a"], color=(0.12, 0.47, 0.71, 0.38),
                edgecolor="#1f77b4", linewidth=1.5)
    axes[2].set_ylabel("95th percentile |I| [A]")
    axes[2].set_title("Load-profile coverage")
    for ax in axes:
        clean_axes(ax)
    fig.suptitle("Independent LFP holdout cells; quantitative coverage, not cross-chemistry validation")
    fig.tight_layout()
    save_figure(fig, out / "Figure_16_Holdout_Cell_Coverage.png")


def plot_statistical_validation(aggregate: pd.DataFrame, out: Path) -> None:
    rows = metric_slice(aggregate, ["current_noise_high", "voltage_noise", "temperature_noise"], "delta_mae")
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    aliases = [a for a in ["current_noise_high", "voltage_noise", "temperature_noise"] if a in set(rows["alias"])]
    positions = []
    labels = []
    y = 0
    for alias in aliases:
        for model in MODEL_ORDER:
            part = rows[(rows["alias"] == alias) & (rows["model"] == model)]
            if part.empty:
                continue
            row = part.iloc[0]
            ax.errorbar(row["mean"], y, xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                        fmt="o", capsize=3, color=MODEL_COLORS[model])
            positions.append(y)
            labels.append(f"{SCENARIO_LABELS[alias]} | {model}")
            y += 1
        y += 0.5
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Cell-macro $\Delta$MAE with hierarchical 95% bootstrap CI [SOC]")
    ax.set_title("Statistical robustness across independent cells and repeated disturbance seeds")
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, out / "Figure_17_Statistical_Robustness.png")


def plot_cadence(aggregate: pd.DataFrame, out: Path) -> None:
    rows = aggregate[(aggregate["alias"] == "baseline") & (aggregate["metric"] == "mae")]
    rows = rows[rows["soh_mode"] == "lstm"]
    if rows["soh_publish_intervals"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    cadences = sorted(rows["soh_publish_intervals"].unique())
    for model in ["HDM", "HECM", "DD"]:
        part = rows[rows["model"] == model].set_index("soh_publish_intervals").reindex(cadences)
        mean = part["mean"].to_numpy(dtype=float)
        low = part["ci_low"].to_numpy(dtype=float)
        high = part["ci_high"].to_numpy(dtype=float)
        ax.errorbar(cadences, mean, yerr=np.vstack([mean - low, high - mean]), marker="o", capsize=3,
                    color=MODEL_COLORS[model], linewidth=2.0, label=model)
    ax.set_xscale("log")
    ax.set_xticks(cadences, [f"{value} h" for value in cadences])
    ax.set_xlabel("SOH publication interval")
    ax.set_ylabel("Baseline MAE [SOC]")
    ax.set_title("SOH-context cadence sensitivity (same hourly LSTM, sample-and-hold publication)")
    ax.legend(ncol=3, frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, out / "Figure_18_SOH_Cadence_Sensitivity.png")


def stratified_metric_slice(
    aggregate: pd.DataFrame,
    dimension: str,
    alias: str,
    metric: str,
) -> pd.DataFrame:
    if aggregate.empty:
        return aggregate
    rows = aggregate[
        (aggregate["dimension"] == dimension)
        & (aggregate["alias"] == alias)
        & (aggregate["metric"] == metric)
    ].copy()
    keep = np.zeros(len(rows), dtype=bool)
    for model, condition in PRIMARY_CONDITIONS.items():
        keep |= (rows["model"] == model) & (rows["soh_condition"] == condition)
    return rows.loc[keep]


def draw_stratified_lines(ax, rows: pd.DataFrame, strata: list[str], labels: list[str]) -> None:
    x = np.arange(len(strata))
    for model in MODEL_ORDER:
        part = rows[rows["model"] == model].set_index("stratum").reindex(strata)
        if part.empty or part["mean"].isna().all():
            continue
        mean = part["mean"].to_numpy(dtype=float)
        low = part["ci_low"].to_numpy(dtype=float)
        high = part["ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            x, mean, yerr=np.vstack([mean - low, high - mean]), marker="o",
            linewidth=2.0, capsize=3, color=MODEL_COLORS[model], label=model,
        )
    ax.set_xticks(x, labels)
    clean_axes(ax)


def plot_soh_state_performance(aggregate: pd.DataFrame, out: Path) -> None:
    rows = stratified_metric_slice(aggregate, "soh_state", "baseline", "mae")
    if rows.empty or not {"fresh", "mid_life", "aged"}.issubset(set(rows["stratum"])):
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    draw_stratified_lines(ax, rows, ["fresh", "mid_life", "aged"], ["Fresh\nSOH >= 0.90", "Mid-life\n0.80-0.90", "Aged\nSOH < 0.80"])
    ax.set_ylabel("Baseline MAE [SOC]")
    ax.set_title("Estimator accuracy across reference-SOH states (equal-weight cell macro)")
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    save_figure(fig, out / "Figure_20_SOH_State_Performance.png")


def plot_load_soh_interaction(aggregate: pd.DataFrame, out: Path) -> None:
    if aggregate.empty:
        return
    rows = aggregate[
        (aggregate["dimension"] == "soh_state")
        & (aggregate["alias"] == "baseline")
        & (aggregate["metric"] == "mae")
    ].copy()
    keep = np.zeros(len(rows), dtype=bool)
    for model, condition in PRIMARY_CONDITIONS.items():
        keep |= (rows["model"] == model) & (rows["soh_condition"] == condition)
    rows = rows.loc[keep]
    if (
        rows.empty
        or not {"fresh", "mid_life", "aged"}.issubset(set(rows["stratum"]))
        or not {"low", "middle", "high"}.issubset(set(rows["cell_load_class"]))
    ):
        return

    load_order = ["low", "middle", "high"]
    soh_order = ["fresh", "mid_life", "aged"]
    matrices = []
    for model in MODEL_ORDER:
        part = rows[rows["model"] == model]
        matrices.append(part.pivot(index="cell_load_class", columns="stratum", values="mean").reindex(
            index=load_order, columns=soh_order
        ))
    finite_values = np.concatenate([matrix.to_numpy(dtype=float).ravel() for matrix in matrices])
    finite_values = finite_values[np.isfinite(finite_values)]
    vmax = float(np.max(finite_values)) if len(finite_values) else 1.0
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8), sharey=True)
    cmap = LinearSegmentedColormap.from_list("diss_sequential", ["#f7f7f7", "#d1887e", "#b6302d"])
    image = None
    for ax, model, matrix in zip(axes, MODEL_ORDER, matrices):
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_xticks(np.arange(3), ["Fresh", "Mid", "Aged"])
        ax.set_title(model)
        for row_idx in range(3):
            for col_idx in range(3):
                value = matrix.iloc[row_idx, col_idx]
                ax.text(col_idx, row_idx, "N/A" if not np.isfinite(value) else f"{value:.3f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(value) and value > 0.60 * vmax else "#222222")
    axes[0].set_yticks(
        np.arange(3),
        ["Low load (n=2)", "Middle load (n=3)", "High load (C29; n=1)"],
    )
    colorbar_axis = fig.add_axes([0.92, 0.18, 0.015, 0.55])
    fig.colorbar(image, cax=colorbar_axis, label="Baseline MAE [SOC]")
    fig.suptitle("Cell-load group x SOH-state interaction (High: exploratory C29 case)")
    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.15, top=0.78, wspace=0.12)
    save_figure(fig, out / "Figure_21_Load_Class_SOH_Interaction.png")


def plot_soh_state_robustness(aggregate: pd.DataFrame, out: Path) -> None:
    aliases = [
        alias for alias in ["current_noise_high", "temperature_noise", "missing_gap_1h"]
        if {"fresh", "mid_life", "aged"}.issubset(
            set(stratified_metric_slice(aggregate, "soh_state", alias, "delta_mae")["stratum"])
        )
    ]
    if not aliases:
        return
    fig, axes = plt.subplots(1, len(aliases), figsize=(4.4 * len(aliases), 4.2), squeeze=False, sharey=True)
    for ax, alias in zip(axes[0], aliases):
        rows = stratified_metric_slice(aggregate, "soh_state", alias, "delta_mae")
        draw_stratified_lines(ax, rows, ["fresh", "mid_life", "aged"], ["Fresh", "Mid", "Aged"])
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_title(SCENARIO_LABELS[alias])
    axes[0, 0].set_ylabel(r"State-specific $\Delta$MAE [SOC]")
    axes[0, -1].legend(ncol=2, frameon=False)
    fig.suptitle("Disturbance sensitivity across battery aging states")
    fig.tight_layout()
    save_figure(fig, out / "Figure_22_SOH_State_Robustness.png")


def plot_operating_state_performance(aggregate: pd.DataFrame, out: Path) -> None:
    dimensions = [
        ("instantaneous_load", ["low", "medium", "high"], ["<0.5 C", "0.5-1.5 C", ">=1.5 C"], "Instantaneous load"),
        ("temperature_state", ["nominal", "elevated", "hot"], ["<=30 C", "30-35 C", ">35 C"], "Temperature state"),
    ]
    if any(
        not set(strata).issubset(
            set(stratified_metric_slice(aggregate, dimension, "baseline", "mae")["stratum"])
        )
        for dimension, strata, *_ in dimensions
    ):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (dimension, strata, labels, title) in zip(axes, dimensions):
        draw_stratified_lines(ax, stratified_metric_slice(aggregate, dimension, "baseline", "mae"), strata, labels)
        ax.set_title(title)
    axes[0].set_ylabel("Baseline MAE [SOC]")
    axes[1].legend(ncol=2, frameon=False)
    fig.suptitle("Accuracy across measured operating regimes")
    fig.tight_layout()
    save_figure(fig, out / "Figure_23_Operating_State_Performance.png")


def validate_coverage(manifest: dict, frame: pd.DataFrame, allow_incomplete: bool) -> None:
    canonical = lambda cell: str(cell).rsplit("_", 1)[-1]
    expected = {canonical(cell) for cell in manifest.get("split", {}).get("holdout", manifest.get("cells", []))}
    requested = {canonical(cell) for cell in manifest.get("cells", [])}
    completed = {canonical(cell) for cell in frame["cell"]}
    missing = sorted(requested - completed)
    if missing and not allow_incomplete:
        raise ValueError(f"Campaign is incomplete; no completed runs for cells: {missing}")
    if not requested.issubset(expected):
        raise ValueError("Campaign contains cells outside the declared holdout split")

    protocol = manifest.get("protocol", {})
    expected_start = int(
        protocol.get(
            "common_evaluation_start_sample",
            manifest.get("correction", {}).get("evaluation_start_sample", 0),
        )
    )
    if expected_start > 0:
        wrong_start = frame[frame["evaluation_start_sample"] != expected_start]
        if not wrong_start.empty:
            raise ValueError(
                f"Common evaluation mask violation: {len(wrong_start)} runs do not start "
                f"at source sample {expected_start}"
            )
        expected_samples = (frame["max_rows"] - expected_start).clip(lower=0)
        wrong_count = frame[frame["evaluation_samples"] != expected_samples]
        if not wrong_count.empty:
            raise ValueError(
                f"Common evaluation mask violation: {len(wrong_count)} runs have an "
                "unexpected evaluation-sample count"
            )

    base_seed = int(manifest.get("base_seed", 42))
    repeats = int(manifest.get("stochastic_repeats", 1))
    secondary_repeats = int(manifest.get("secondary_stochastic_repeats", repeats))
    selected_models = set(manifest.get("models", MODEL_ORDER))
    soh_modes = set(manifest.get("soh_modes", []))
    reference_aliases = set(manifest.get("reference_aliases", []))
    cadence_aliases = set(manifest.get("cadence_aliases", ["baseline"]))
    lstm_intervals = manifest.get("lstm_publish_intervals", [1])
    reference_intervals = manifest.get("reference_publish_intervals", [1])
    scenario_aliases = [row["alias"] for row in manifest.get("protocol", {}).get("scenarios", [])]

    definitions = manifest.get("window", {}).get("definitions", [])
    if not definitions:
        definitions = [
            {"cell": row["cell"], "window_id": row["window_id"]}
            for row in manifest.get("runs", [])
            if row.get("window_id")
        ]
    if definitions:
        expected_windows = sorted({
            (canonical(row["cell"]), str(row["window_id"]))
            for row in definitions
            if canonical(row["cell"]) in requested
        })
    else:
        expected_windows = [(cell, "single_window") for cell in requested]

    expected_runs = set()
    for cell, window_id in expected_windows:
        for alias in scenario_aliases:
            if alias in PRIMARY_STOCHASTIC_ALIASES:
                seed_count = repeats
            elif alias in STOCHASTIC_ALIASES:
                seed_count = secondary_repeats
            else:
                seed_count = 1
            for seed in range(base_seed, base_seed + seed_count):
                if "DM" in selected_models:
                    expected_runs.add((cell, window_id, alias, seed, "none", "DM"))
                for mode in soh_modes:
                    if mode == "reference" and alias not in reference_aliases:
                        continue
                    intervals = reference_intervals if mode == "reference" else lstm_intervals
                    if mode == "lstm" and alias not in cadence_aliases:
                        intervals = [1]
                    for interval in intervals:
                        condition = f"{mode}_h{interval}"
                        if alias == "initial_soc_error":
                            declared = manifest.get("protocol", {}).get(
                                "initial_state_comparison_models", ["DM", "HDM", "HECM"]
                            )
                            models = [
                                model for model in ["HDM", "HECM", "DD"]
                                if model in declared and model in selected_models
                            ]
                        else:
                            models = [model for model in ["HDM", "HECM", "DD"] if model in selected_models]
                        for model in models:
                            expected_runs.add((cell, window_id, alias, seed, condition, model))

    completed_runs = {
        (
            canonical(row.cell), getattr(row, "window_id", "single_window"), row.alias,
            int(row.seed), row.soh_condition, row.model,
        )
        for row in frame.itertuples(index=False)
    }
    missing_runs = sorted(expected_runs - completed_runs)
    if missing_runs and not allow_incomplete:
        preview = ", ".join("/".join(map(str, key)) for key in missing_runs[:8])
        raise ValueError(f"Campaign is incomplete; {len(missing_runs)} expected runs are missing: {preview}")


def has_aliases(frame: pd.DataFrame, aliases: Iterable[str]) -> bool:
    return set(aliases).issubset(set(frame["alias"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewer-facing JES2 tables and DISS-colored figures.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--figures_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"))
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    setup_style()
    manifest, raw = load_campaign(args.manifest)
    validate_coverage(manifest, raw, args.allow_incomplete)
    raw = add_baseline_deltas(raw)
    stratified = load_stratified_run_metrics(raw)
    cells = pd.DataFrame([characterize_cell(args.data_root, cell) for cell in manifest["cells"]])
    if not stratified.empty:
        load_class = {
            str(row.cell).rsplit("_", 1)[-1]: row.cell_load_class
            for row in cells.itertuples(index=False)
        }
        stratified["cell_load_class"] = stratified["cell"].map(
            lambda cell: load_class.get(str(cell).rsplit("_", 1)[-1], "unassigned")
        )
        stratified = add_stratified_baseline_deltas(stratified)
    raw = raw[~raw["alias"].isin(INTERNAL_REFERENCE_ALIASES)].copy()
    if not stratified.empty:
        stratified = stratified[~stratified["alias"].isin(INTERNAL_REFERENCE_ALIASES)].copy()
    aggregate = aggregate_metrics(raw, args.bootstrap_samples)
    paired, ablation = build_soh_ablation(raw, args.bootstrap_samples)
    stratified_aggregate = aggregate_stratified_metrics(stratified, args.bootstrap_samples)
    load_soh_aggregate = aggregate_stratified_metrics(stratified, args.bootstrap_samples, include_load_class=True)
    scenario_tests, model_pair_tests = build_paired_statistical_tests(raw, args.bootstrap_samples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_dir / "jes2_run_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "jes2_macro_statistics.csv", index=False)
    paired.to_csv(args.out_dir / "jes2_soh_ablation_pairs.csv", index=False)
    ablation.to_csv(args.out_dir / "jes2_soh_ablation_statistics.csv", index=False)
    cells.to_csv(args.out_dir / "jes2_cell_characteristics.csv", index=False)
    stratified.to_csv(args.out_dir / "jes2_stratified_run_metrics.csv", index=False)
    stratified_aggregate.to_csv(args.out_dir / "jes2_stratified_statistics.csv", index=False)
    load_soh_aggregate.to_csv(args.out_dir / "jes2_load_class_stratified_statistics.csv", index=False)
    scenario_tests.to_csv(args.out_dir / "jes2_paired_scenario_tests.csv", index=False)
    model_pair_tests.to_csv(args.out_dir / "jes2_paired_model_tests.csv", index=False)
    (args.out_dir / "jes2_statistical_method.txt").write_text(
        "Statistical unit: independent holdout cell. Protocol-defined SOH windows are averaged within each cell and "
        "random seed before inference.\nPoint estimates: equal-weight cell macro means. Uncertainty: hierarchical "
        f"bootstrap with seeds nested within cells ({args.bootstrap_samples} repetitions).\nHypothesis tests: "
        "exact two-sided paired sign-flip tests on cell-level differences. Holm correction controls family-wise "
        "error within each scenario across the four model effects or six model-pair comparisons.\n"
        "Effect size: paired standardized mean difference dz. Per-sample observations are never treated as "
        "independent replicates.\n",
        encoding="utf-8",
    )

    plot_baseline(raw, aggregate, args.figures_dir)
    bias_aliases = ["current_bias_0p5pct", "current_bias_1p5pct", "current_bias_3p0pct"]
    if has_aliases(raw, bias_aliases):
        plot_sweep(
            aggregate,
            bias_aliases,
            args.figures_dir / "Figure_05_Current_Bias.png",
            "Current-gain sensitivity across independent holdout cells",
        )
    noise_aliases = ["current_noise_low", "current_noise_high", "voltage_noise", "temperature_noise"]
    if has_aliases(raw, noise_aliases):
        plot_grouped_scenarios(
            aggregate,
            noise_aliases,
            args.figures_dir / "Figure_06_Noise_Robustness.png",
            "Repeated noise robustness (cell-macro mean and hierarchical 95% CI)",
        )
    signal_aliases = ["missing_samples_periodic", "missing_samples_random", "irregular_sampling_0p1s",
                      "irregular_sampling_0p5s", "irregular_sampling_0p9s"]
    if has_aliases(raw, signal_aliases):
        plot_grouped_scenarios(
            aggregate,
            signal_aliases,
            args.figures_dir / "Figure_08_Signal_Integrity.png",
            "Signal-integrity and timing stress",
        )
    if has_aliases(raw, ["voltage_spikes"]):
        plot_spikes(aggregate, args.figures_dir)
    if len(set(raw["alias"]) - {"baseline", "initial_soc_error"}) > 0:
        plot_heatmap(aggregate, args.figures_dir)
    draw_adc = metric_slice(aggregate, "adc_quantization", "delta_mae")
    if not draw_adc.empty:
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        draw_model_bars(ax, draw_adc, r"$\Delta$MAE [SOC]", "ADC quantization across holdout cells")
        fig.tight_layout()
        save_figure(fig, args.figures_dir / "Figure_14_ADC_Quantization.png")
    plot_soh_ablation(ablation, args.figures_dir)
    plot_cell_coverage(cells, args.figures_dir)
    plot_statistical_validation(aggregate, args.figures_dir)
    plot_cadence(aggregate, args.figures_dir)
    plot_soh_state_performance(stratified_aggregate, args.figures_dir)
    plot_load_soh_interaction(load_soh_aggregate, args.figures_dir)
    plot_soh_state_robustness(stratified_aggregate, args.figures_dir)
    plot_operating_state_performance(stratified_aggregate, args.figures_dir)

    result_manifest = {
        "source_manifest": str(args.manifest.resolve()),
        "source_tag": manifest.get("tag"),
        "cells": manifest.get("cells"),
        "completed_run_records": int(len(raw)),
        "bootstrap_samples": args.bootstrap_samples,
        "aggregation": "windows averaged within cell/seed; equal-weight cell macro; hierarchical bootstrap",
        "common_evaluation_start_sample": manifest.get("protocol", {}).get(
            "common_evaluation_start_sample"
        ),
        "primary_soh_condition": "lstm_h1",
        "reference_soh_role": "paired explanatory ablation",
        "stratified_run_records": int(len(stratified)),
        "paired_scenario_tests": int(len(scenario_tests)),
        "paired_model_tests": int(len(model_pair_tests)),
        "stratification_dimensions": ["cell_load_class", "soh_state", "temperature_state", "instantaneous_load", "soc_state"],
        "palette": "DISS EAAI categorical (green, purple, blue, red)",
    }
    (args.out_dir / "jes2_results_manifest.json").write_text(json.dumps(result_manifest, indent=2), encoding="utf-8")
    print(json.dumps(result_manifest, indent=2))


if __name__ == "__main__":
    main()
