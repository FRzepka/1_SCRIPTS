from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("DM", "HDM", "HECM", "DD")
ALIASES = ("positive_3pct", "negative_3pct")
PROGRESS_GRID = np.linspace(0.0, 100.0, 41)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def load_temporal(campaign: Path, alias: str, model: str) -> pd.DataFrame:
    path = campaign / "runs" / alias / model / "temporal_metrics_C29.csv"
    frame = pd.read_csv(path)
    required = {"time_s", "n_samples", "mae", "soc_pred_mean"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def paired_temporal(campaign: Path, alias: str, model: str) -> pd.DataFrame:
    baseline = load_temporal(campaign, "baseline", model)
    biased = load_temporal(campaign, alias, model)
    paired = baseline.merge(
        biased,
        on=["bin", "time_s"],
        suffixes=("_baseline", "_biased"),
        validate="one_to_one",
    )
    paired["n_samples"] = paired[["n_samples_baseline", "n_samples_biased"]].min(axis=1)
    paired["delta_mae"] = paired["mae_biased"] - paired["mae_baseline"]
    paired["prediction_divergence"] = np.abs(
        paired["soc_pred_mean_biased"] - paired["soc_pred_mean_baseline"]
    )
    return paired


def select_adverse_alias(campaign: Path, model: str) -> tuple[str, dict[str, float]]:
    scores: dict[str, float] = {}
    for alias in ALIASES:
        paired = paired_temporal(campaign, alias, model)
        scores[alias] = weighted_mean(paired.delta_mae, paired.n_samples)
    return max(scores, key=scores.get), scores


def interpolate_cycle(
    frame: pd.DataFrame,
    start_s: float,
    stop_s: float,
    value_columns: tuple[str, ...],
) -> dict[str, np.ndarray] | None:
    part = frame[(frame.time_s >= start_s) & (frame.time_s < stop_s)].copy()
    if len(part) < 2 or stop_s <= start_s:
        return None
    progress = 100.0 * (part.time_s.to_numpy(float) - start_s) / (stop_s - start_s)
    order = np.argsort(progress)
    progress = progress[order]
    unique, unique_indices = np.unique(progress, return_index=True)
    if len(unique) < 2:
        return None
    result = {"progress_pct": PROGRESS_GRID.copy()}
    for column in value_columns:
        values = part[column].to_numpy(float)[order][unique_indices]
        result[column] = np.interp(PROGRESS_GRID, unique, values)
    return result


def aggregate_normalized_cycles(
    paired: pd.DataFrame,
    reset_times: np.ndarray,
    model: str,
    alias: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle_rows = []
    normalized_rows = []
    value_columns = ("mae_baseline", "mae_biased", "delta_mae", "prediction_divergence")
    for cycle, (start_s, stop_s) in enumerate(zip(reset_times[:-1], reset_times[1:]), start=1):
        part = paired[(paired.time_s >= start_s) & (paired.time_s < stop_s)]
        if part.empty:
            continue
        cycle_rows.append({
            "model": model,
            "alias": alias,
            "cycle": cycle,
            "start_h": start_s / 3600.0,
            "stop_h": stop_s / 3600.0,
            "duration_h": (stop_s - start_s) / 3600.0,
            "baseline_mae": weighted_mean(part.mae_baseline, part.n_samples),
            "biased_mae": weighted_mean(part.mae_biased, part.n_samples),
            "delta_mae": weighted_mean(part.delta_mae, part.n_samples),
            "prediction_divergence": weighted_mean(part.prediction_divergence, part.n_samples),
        })
        interpolated = interpolate_cycle(paired, start_s, stop_s, value_columns)
        if interpolated is None:
            continue
        for index, progress in enumerate(interpolated["progress_pct"]):
            normalized_rows.append({
                "model": model,
                "alias": alias,
                "cycle": cycle,
                "progress_pct": progress,
                **{column: interpolated[column][index] for column in value_columns},
            })
    cycles = pd.DataFrame(cycle_rows)
    normalized = pd.DataFrame(normalized_rows)
    return cycles, normalized


def event_window_metrics(
    paired: pd.DataFrame,
    events: pd.DataFrame,
    model: str,
    alias: str,
    window_h: float,
) -> pd.DataFrame:
    rows = []
    window_s = window_h * 3600.0
    for event in events.itertuples(index=False):
        for phase, start_s, stop_s in (
            ("before", event.time_s - window_s, event.time_s),
            ("after", event.time_s, event.time_s + window_s),
        ):
            part = paired[(paired.time_s >= start_s) & (paired.time_s < stop_s)]
            if part.empty:
                continue
            rows.append({
                "model": model,
                "alias": alias,
                "event": int(event.event),
                "event_h": float(event.elapsed_h),
                "soh": float(event.soh),
                "phase": phase,
                "window_h": window_h,
                "n_bins": len(part),
                "baseline_mae": weighted_mean(part.mae_baseline, part.n_samples),
                "biased_mae": weighted_mean(part.mae_biased, part.n_samples),
                "delta_mae": weighted_mean(part.delta_mae, part.n_samples),
                "prediction_divergence": weighted_mean(
                    part.prediction_divergence, part.n_samples
                ),
            })
    return pd.DataFrame(rows)


def summarize_normalized(normalized: pd.DataFrame) -> pd.DataFrame:
    value_columns = ("mae_baseline", "mae_biased", "delta_mae", "prediction_divergence")
    rows = []
    for (model, alias, progress), part in normalized.groupby(
        ["model", "alias", "progress_pct"], sort=False
    ):
        row = {
            "model": model,
            "alias": alias,
            "progress_pct": progress,
            "n_cycles": part.cycle.nunique(),
        }
        for column in value_columns:
            row[f"{column}_mean"] = float(part[column].mean())
            row[f"{column}_min"] = float(part[column].min())
            row[f"{column}_max"] = float(part[column].max())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_model(
    paired: pd.DataFrame,
    cycles: pd.DataFrame,
    normalized: pd.DataFrame,
    event_metrics: pd.DataFrame,
    model: str,
    alias: str,
    alias_scores: dict[str, float],
) -> dict[str, float | int | str | bool]:
    life_progress = (paired.time_s - paired.time_s.min()) / (
        paired.time_s.max() - paired.time_s.min()
    )
    life_start = paired[life_progress <= 0.1]
    life_end = paired[life_progress >= 0.9]
    early = normalized[normalized.progress_pct <= 20.0]
    late = normalized[normalized.progress_pct >= 80.0]
    before = event_metrics[event_metrics.phase == "before"]
    after = event_metrics[event_metrics.phase == "after"]
    event_pairs = before[["event", "delta_mae"]].merge(
        after[["event", "delta_mae"]], on="event", suffixes=("_before", "_after")
    )
    early_delta = float(early.delta_mae.mean())
    late_delta = float(late.delta_mae.mean())
    before_delta = float(before.delta_mae.mean())
    after_delta = float(after.delta_mae.mean())
    early_biased = float(early.mae_biased.mean())
    late_biased = float(late.mae_biased.mean())
    early_baseline = float(early.mae_baseline.mean())
    late_baseline = float(late.mae_baseline.mean())
    before_biased = float(before.biased_mae.mean())
    after_biased = float(after.biased_mae.mean())
    early_divergence = float(early.prediction_divergence.mean())
    late_divergence = float(late.prediction_divergence.mean())
    return {
        "model": model,
        "adverse_alias": alias,
        "bias_percent": 3.0 if alias == "positive_3pct" else -3.0,
        "positive_delta_mae": alias_scores["positive_3pct"],
        "negative_delta_mae": alias_scores["negative_3pct"],
        "baseline_mae_full_life": weighted_mean(paired.mae_baseline, paired.n_samples),
        "biased_mae_full_life": weighted_mean(paired.mae_biased, paired.n_samples),
        "delta_mae_full_life": weighted_mean(paired.delta_mae, paired.n_samples),
        "delta_mae_life_first_10pct": weighted_mean(
            life_start.delta_mae, life_start.n_samples
        ),
        "delta_mae_life_last_10pct": weighted_mean(
            life_end.delta_mae, life_end.n_samples
        ),
        "delta_mae_life_change": (
            weighted_mean(life_end.delta_mae, life_end.n_samples)
            - weighted_mean(life_start.delta_mae, life_start.n_samples)
        ),
        "biased_mae_life_first_10pct": weighted_mean(
            life_start.mae_biased, life_start.n_samples
        ),
        "biased_mae_life_last_10pct": weighted_mean(
            life_end.mae_biased, life_end.n_samples
        ),
        "biased_mae_life_change": (
            weighted_mean(life_end.mae_biased, life_end.n_samples)
            - weighted_mean(life_start.mae_biased, life_start.n_samples)
        ),
        "n_inter_reset_cycles": int(cycles.cycle.nunique()),
        "n_reset_events": int(event_metrics.event.nunique()),
        "delta_mae_cycle_start_0_20pct": early_delta,
        "delta_mae_cycle_end_80_100pct": late_delta,
        "delta_mae_cycle_accumulation": late_delta - early_delta,
        "biased_mae_cycle_start_0_20pct": early_biased,
        "biased_mae_cycle_end_80_100pct": late_biased,
        "biased_mae_cycle_change": late_biased - early_biased,
        "baseline_mae_cycle_change": late_baseline - early_baseline,
        "prediction_divergence_cycle_start_0_20pct": early_divergence,
        "prediction_divergence_cycle_end_80_100pct": late_divergence,
        "prediction_divergence_cycle_accumulation": late_divergence - early_divergence,
        "delta_mae_before_reset": before_delta,
        "delta_mae_after_reset": after_delta,
        "delta_mae_reset_change": after_delta - before_delta,
        "biased_mae_before_reset": before_biased,
        "biased_mae_after_reset": after_biased,
        "biased_mae_reset_change": after_biased - before_biased,
        "events_with_lower_delta_after_percent": float(
            100.0 * (event_pairs.delta_mae_after < event_pairs.delta_mae_before).mean()
        ),
        "bias_degrades_full_life_mae": bool(
            weighted_mean(paired.delta_mae, paired.n_samples) > 0.0
        ),
    }


def main() -> None:
    simulation = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze time-resolved current-bias accumulation and full-charge recovery."
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=simulation / "campaigns/jes2_c29_lifecycle_bias_temporal_20260828",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--event-window-hours", type=float, default=2.0)
    args = parser.parse_args()
    out = args.out_dir or args.campaign / "results"
    out.mkdir(parents=True, exist_ok=True)

    event_path = args.campaign / "runs" / "baseline" / "DM" / "full_charge_events_C29.csv"
    events = pd.read_csv(event_path)
    reset_times = events.time_s.to_numpy(float)
    all_cycles = []
    all_normalized = []
    all_event_metrics = []
    all_selected_temporal = []
    summaries = []
    selected_aliases = {}

    for model in MODELS:
        alias, alias_scores = select_adverse_alias(args.campaign, model)
        selected_aliases[model] = {
            "alias": alias,
            "bias_percent": 3.0 if alias == "positive_3pct" else -3.0,
            "delta_mae_by_alias": alias_scores,
        }
        paired = paired_temporal(args.campaign, alias, model)
        selected_temporal = paired[[
            "time_s", "elapsed_h_baseline", "n_samples", "mae_baseline", "mae_biased",
            "delta_mae", "prediction_divergence", "soh_mean_baseline",
        ]].copy()
        selected_temporal.insert(0, "alias", alias)
        selected_temporal.insert(0, "model", model)
        selected_temporal = selected_temporal.rename(
            columns={"elapsed_h_baseline": "elapsed_h", "soh_mean_baseline": "soh"}
        )
        cycles, normalized = aggregate_normalized_cycles(
            paired, reset_times, model, alias
        )
        event_metrics = event_window_metrics(
            paired, events, model, alias, args.event_window_hours
        )
        all_cycles.append(cycles)
        all_normalized.append(normalized)
        all_event_metrics.append(event_metrics)
        all_selected_temporal.append(selected_temporal)
        summaries.append(
            summarize_model(
                paired, cycles, normalized, event_metrics, model, alias, alias_scores
            )
        )

    cycle_metrics = pd.concat(all_cycles, ignore_index=True)
    normalized_raw = pd.concat(all_normalized, ignore_index=True)
    normalized_summary = summarize_normalized(normalized_raw)
    event_metrics = pd.concat(all_event_metrics, ignore_index=True)
    selected_temporal = pd.concat(all_selected_temporal, ignore_index=True)
    model_summary = pd.DataFrame(summaries)

    cycle_metrics.to_csv(out / "c29_bias_inter_reset_cycles.csv", index=False)
    normalized_summary.to_csv(out / "c29_bias_normalized_cycle.csv", index=False)
    event_metrics.to_csv(out / "c29_bias_reset_event_metrics.csv", index=False)
    lifecycle_rows = []
    for (model, alias), part in selected_temporal.groupby(["model", "alias"], sort=False):
        part = part.sort_values("elapsed_h").copy()
        part["delta_mae_24h"] = part.delta_mae.rolling(
            288, min_periods=72, center=True
        ).mean()
        compact = part.iloc[::12][["elapsed_h", "delta_mae_24h"]].copy()
        compact.insert(0, "alias", alias)
        compact.insert(0, "model", model)
        lifecycle_rows.append(compact)
    pd.concat(lifecycle_rows, ignore_index=True).to_csv(
        out / "c29_bias_lifecycle_24h.csv", index=False
    )
    model_summary.to_csv(out / "c29_bias_temporal_model_summary.csv", index=False)
    protocol = {
        "cell": "C29",
        "temporal_bin_seconds": 300,
        "event_window_hours": args.event_window_hours,
        "full_charge_events": int(len(events)),
        "inter_reset_cycles": int(max(len(events) - 1, 0)),
        "adverse_bias_selection": (
            "For each model, select the sign with the larger full-life paired delta MAE."
        ),
        "selected_aliases": selected_aliases,
        "interpretation": (
            "Delta MAE controls for the baseline drive-profile error. Absolute biased MAE "
            "determines the best practical model under current bias."
        ),
    }
    (out / "c29_bias_temporal_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(model_summary.to_string(index=False))


if __name__ == "__main__":
    main()
