from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "results"))
from jes2_plot_style import NEUTRAL_DARK, clean_axes, save_figure, setup_style
from robustness_common import compute_common_recovery_metrics


LOOKUP_CONDITIONS = {
    "nominal_lookup": {},
    "resistance_minus_10pct": {"--ecm_resistance_scale": "0.90"},
    "resistance_plus_10pct": {"--ecm_resistance_scale": "1.10"},
    "tau_minus_10pct": {"--ecm_tau_scale": "0.90"},
    "tau_plus_10pct": {"--ecm_tau_scale": "1.10"},
    "ocv_minus_10mV": {"--ecm_ocv_offset_v": "-0.010"},
    "ocv_plus_10mV": {"--ecm_ocv_offset_v": "0.010"},
}

LOOKUP_LABELS = {
    "nominal_lookup": "Nominal lookup",
    "resistance_minus_10pct": "Resistance -10%",
    "resistance_plus_10pct": "Resistance +10%",
    "tau_minus_10pct": "Time constants -10%",
    "tau_plus_10pct": "Time constants +10%",
    "ocv_minus_10mV": "OCV -10 mV",
    "ocv_plus_10mV": "OCV +10 mV",
}

LOOKUP_PLOT_LABELS = {
    "resistance_minus_10pct": "R\n-10%",
    "resistance_plus_10pct": "R\n+10%",
    "tau_minus_10pct": r"$\tau$" + "\n-10%",
    "tau_plus_10pct": r"$\tau$" + "\n+10%",
    "ocv_minus_10mV": "OCV\n-10 mV",
    "ocv_plus_10mV": "OCV\n+10 mV",
}

FAMILY_ALIASES = {
    "Current gain": {
        "current_bias_neg_0p5pct",
        "current_bias_0p5pct",
        "current_bias_neg_1p5pct",
        "current_bias_1p5pct",
        "current_bias_neg_3p0pct",
        "current_bias_3p0pct",
    },
    "Sensor noise": {
        "current_noise_low",
        "current_noise_high",
        "voltage_noise",
        "temperature_noise",
    },
    "Sensor offsets": {"voltage_offset", "temperature_offset"},
    "ADC quantization": {"adc_quantization"},
    "Missing samples": {"missing_samples_periodic", "missing_samples_random"},
    "Timing jitter": {
        "irregular_sampling_0p1s",
        "irregular_sampling_0p5s",
        "irregular_sampling_0p9s",
    },
    "Burst dropout": {"missing_gap_1h"},
    "Voltage spikes": {"voltage_spikes"},
    "Initialization MAE": {"initial_soc_error"},
}

ALIAS_LABELS = {
    "adc_quantization": "ADC quantization",
    "current_bias_neg_0p5pct": "Current gain -0.5%",
    "current_bias_0p5pct": "Current gain +0.5%",
    "current_bias_neg_1p5pct": "Current gain -1.5%",
    "current_bias_1p5pct": "Current gain +1.5%",
    "current_bias_neg_3p0pct": "Current gain -3.0%",
    "current_bias_3p0pct": "Current gain +3.0%",
    "current_noise_low": "Current noise 0.02 A",
    "current_noise_high": "Current noise 0.10 A",
    "initial_soc_error": "Initial SOC error",
    "irregular_sampling_0p1s": "Timing jitter 0.1 s",
    "irregular_sampling_0p5s": "Timing jitter 0.5 s",
    "irregular_sampling_0p9s": "Timing jitter 0.9 s",
    "missing_gap_1h": "Burst dropout 1 h",
    "missing_samples_periodic": "Periodic missing samples",
    "missing_samples_random": "Random missing samples",
    "temperature_noise": "Temperature noise",
    "temperature_offset": "Temperature offset",
    "voltage_noise": "Voltage noise",
    "voltage_offset": "Voltage offset",
    "voltage_spikes": "Voltage spikes",
}

DROP_BASELINE_ALIAS = "missing_gap_baseline_48h"
BASELINE_ALIAS = "baseline"
RECOVERY_BASELINE_ALIAS = "recovery_baseline"
INITIAL_ALIAS = "initial_soc_error"
EXPECTED_DISTURBANCE_ALIASES = set().union(*FAMILY_ALIASES.values())
ROBUSTNESS_FAMILIES = {
    family: aliases
    for family, aliases in FAMILY_ALIASES.items()
    if family != "Initialization MAE"
}
FULL_OUTPUT_ALIASES = {RECOVERY_BASELINE_ALIAS, INITIAL_ALIAS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_manifest(path: Path) -> list[dict]:
    return list(json.loads(path.read_text(encoding="utf-8")).get("runs", []))


def is_primary_hecm(record: dict) -> bool:
    return record.get("model") == "HECM" and record.get("soh_condition") == "lstm_h1"


def load_sources(
    main_manifest: Path,
    signed_manifest: Path,
    gap_manifest: Path,
    recovery_manifest: Path,
) -> list[dict]:
    main = [
        row
        for row in read_manifest(main_manifest)
        if is_primary_hecm(row) and row.get("alias") != INITIAL_ALIAS
    ]
    signed = [row for row in read_manifest(signed_manifest) if is_primary_hecm(row)]
    gap = [
        row
        for row in read_manifest(gap_manifest)
        if is_primary_hecm(row) and row.get("alias") == DROP_BASELINE_ALIAS
    ]
    recovery = []
    for row in read_manifest(recovery_manifest):
        if not is_primary_hecm(row) or row.get("alias") not in {BASELINE_ALIAS, INITIAL_ALIAS}:
            continue
        copied = dict(row)
        if copied["alias"] == BASELINE_ALIAS:
            copied["alias"] = RECOVERY_BASELINE_ALIAS
        recovery.append(copied)
    sources = main + signed + gap + recovery
    keys = [
        (row["cell"], row["window_id"], row["alias"], int(row["seed"]))
        for row in sources
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate HECM source records in merged sensitivity protocol")
    aliases = {row["alias"] for row in sources}
    expected_aliases = EXPECTED_DISTURBANCE_ALIASES | {
        BASELINE_ALIAS,
        DROP_BASELINE_ALIAS,
        RECOVERY_BASELINE_ALIAS,
    }
    if aliases != expected_aliases:
        raise ValueError(f"Unexpected source aliases: missing={expected_aliases - aliases}, extra={aliases - expected_aliases}")
    if len(sources) != 1280:
        raise ValueError(f"Expected 1,280 HECM source records, found {len(sources)}")
    windows = {
        row["window_id"] for row in sources if row["alias"] == RECOVERY_BASELINE_ALIAS
    }
    if len(windows) != 16:
        raise ValueError(f"Expected 16 recovery baseline windows, found {len(windows)}")
    return sources


def remove_flag(command: list[str], flag: str, takes_value: bool = True) -> list[str]:
    result = list(command)
    while flag in result:
        index = result.index(flag)
        del result[index]
        if takes_value and index < len(result):
            del result[index]
    return result


def build_command(
    source: dict,
    out_dir: Path,
    lookup_parameters: dict[str, str],
    evaluation_start_sample: int,
    device: str,
) -> list[str]:
    command = list(source["command"])
    command[0] = sys.executable
    for flag, takes_value in (
        ("--out_dir", True),
        ("--device", True),
        ("--require_gpu", False),
        ("--evaluation_start_sample", True),
        ("--ecm_resistance_scale", True),
        ("--ecm_tau_scale", True),
        ("--ecm_ocv_offset_v", True),
    ):
        command = remove_flag(command, flag, takes_value)
    command.extend(
        [
            "--out_dir",
            str(out_dir),
            "--device",
            device,
            "--evaluation_start_sample",
            str(evaluation_start_sample),
        ]
    )
    for flag, value in lookup_parameters.items():
        command.extend([flag, value])
    full_output = source["alias"] in FULL_OUTPUT_ALIASES
    command = remove_flag(command, "--summary_only", takes_value=False)
    if not full_output:
        command.append("--summary_only")
    return command


def build_records(
    sources: list[dict],
    out_root: Path,
    evaluation_start_sample: int,
    device: str,
) -> list[dict]:
    records = []
    for lookup_condition, lookup_parameters in LOOKUP_CONDITIONS.items():
        for source in sources:
            out_dir = (
                out_root
                / "runs"
                / lookup_condition
                / source["cell"]
                / source["window_id"]
                / source["alias"]
                / f"seed_{int(source['seed'])}"
            )
            records.append(
                {
                    "cell": source["cell"],
                    "window_id": source["window_id"],
                    "soh_state": source.get("soh_state", source.get("window_soh_state", "unknown")),
                    "cell_load_class": source.get("cell_load_class", "unassigned"),
                    "alias": source["alias"],
                    "scenario": source.get("scenario", source["alias"]),
                    "seed": int(source["seed"]),
                    "start_row": int(source["start_row"]),
                    "max_rows": int(source["max_rows"]),
                    "lookup_condition": lookup_condition,
                    "lookup_parameters": lookup_parameters,
                    "out_dir": str(out_dir),
                    "command": build_command(
                        source,
                        out_dir,
                        lookup_parameters,
                        evaluation_start_sample,
                        device,
                    ),
                    "status": "pending",
                }
            )
    expected = len(sources) * len(LOOKUP_CONDITIONS)
    if expected != 8960 or len(records) != expected:
        raise ValueError(f"Expected 8,960 full lookup-sensitivity runs, built {len(records)}")
    return records


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def execute_record(record: dict, skip_existing: bool) -> tuple[str, str | None]:
    out_dir = Path(record["out_dir"])
    summary_path = out_dir / "summary.json"
    full_output = record["alias"] in FULL_OUTPUT_ALIASES
    csv_exists = bool(list(out_dir.glob("ecm_soc_fullcell_*.csv")))
    if skip_existing and summary_path.is_file() and (not full_output or csv_exists):
        return "reused_existing", None
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            record["command"],
            cwd=WORKSPACE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        return "failed", f"return code {completed.returncode}; see {log_path}"
    if not summary_path.is_file():
        return "failed", f"missing summary: {summary_path}"
    if full_output and not list(out_dir.glob("ecm_soc_fullcell_*.csv")):
        return "failed", f"missing recovery trajectory: {out_dir}"
    return "completed", None


def cell_bootstrap(values: pd.Series, samples: int, seed: int) -> tuple[float, float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return np.nan, np.nan, np.nan
    point = float(array.mean())
    if len(array) == 1 or samples <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load_metrics(manifest: dict) -> pd.DataFrame:
    rows = []
    evaluation_start = int(manifest["evaluation_start_sample"])
    for record in manifest["runs"]:
        summary_path = Path(record["out_dir"]) / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("evaluation_start_sample", -1)) != evaluation_start:
            raise ValueError(f"Common evaluation mask missing in {summary_path}")
        rows.append(
            {
                key: record[key]
                for key in (
                    "cell",
                    "window_id",
                    "soh_state",
                    "cell_load_class",
                    "alias",
                    "scenario",
                    "seed",
                    "start_row",
                    "max_rows",
                    "lookup_condition",
                    "out_dir",
                )
            }
            | {
                "evaluation_start_sample": int(summary["evaluation_start_sample"]),
                "evaluation_samples": int(summary["evaluation_samples"]),
                "mae": float(summary["mae"]),
                "rmse": float(summary["rmse"]),
                "p95_error": float(summary["p95_error"]),
            }
        )
    metrics = pd.DataFrame(rows)
    if len(metrics) != 8960:
        raise ValueError(f"Expected 8,960 completed metrics, found {len(metrics)}")
    return metrics


def add_interactions(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = (
        metrics[metrics["alias"] == BASELINE_ALIAS]
        .sort_values("seed")
        .drop_duplicates(["lookup_condition", "cell", "window_id"])
        [["lookup_condition", "cell", "window_id", "mae", "rmse"]]
        .rename(columns={"mae": "baseline_mae", "rmse": "baseline_rmse"})
    )
    gap_baseline = (
        metrics[metrics["alias"] == DROP_BASELINE_ALIAS]
        .sort_values("seed")
        .drop_duplicates(["lookup_condition", "cell", "window_id"])
        [["lookup_condition", "cell", "window_id", "mae", "rmse"]]
        .rename(columns={"mae": "gap_baseline_mae", "rmse": "gap_baseline_rmse"})
    )
    disturbed = metrics[metrics["alias"].isin(EXPECTED_DISTURBANCE_ALIASES)].copy()
    disturbed = disturbed.merge(
        baseline,
        on=["lookup_condition", "cell", "window_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        gap_baseline,
        on=["lookup_condition", "cell", "window_id"],
        how="left",
        validate="many_to_one",
    )
    gap = disturbed["alias"] == "missing_gap_1h"
    disturbed.loc[gap, "baseline_mae"] = disturbed.loc[gap, "gap_baseline_mae"]
    disturbed.loc[gap, "baseline_rmse"] = disturbed.loc[gap, "gap_baseline_rmse"]
    disturbed["delta_mae"] = disturbed["mae"] - disturbed["baseline_mae"]
    disturbed["delta_rmse"] = disturbed["rmse"] - disturbed["baseline_rmse"]
    nominal = disturbed[disturbed["lookup_condition"] == "nominal_lookup"][
        ["cell", "window_id", "alias", "seed", "delta_mae", "delta_rmse"]
    ].rename(
        columns={"delta_mae": "nominal_delta_mae", "delta_rmse": "nominal_delta_rmse"}
    )
    disturbed = disturbed.merge(
        nominal,
        on=["cell", "window_id", "alias", "seed"],
        how="left",
        validate="many_to_one",
    )
    disturbed["interaction_delta_delta_mae"] = (
        disturbed["delta_mae"] - disturbed["nominal_delta_mae"]
    )
    disturbed["interaction_delta_delta_rmse"] = (
        disturbed["delta_rmse"] - disturbed["nominal_delta_rmse"]
    )
    baseline_cell = baseline.groupby(["lookup_condition", "cell"], as_index=False).agg(
        baseline_mae=("baseline_mae", "mean")
    )
    nominal_baseline = baseline_cell[baseline_cell["lookup_condition"] == "nominal_lookup"][
        ["cell", "baseline_mae"]
    ].rename(columns={"baseline_mae": "nominal_baseline_mae"})
    baseline_cell = baseline_cell.merge(nominal_baseline, on="cell", validate="many_to_one")
    baseline_cell["baseline_lookup_delta_mae"] = (
        baseline_cell["baseline_mae"] - baseline_cell["nominal_baseline_mae"]
    )
    return disturbed, baseline_cell


def scenario_statistics(
    disturbed: pd.DataFrame, bootstrap_samples: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = (
        disturbed.groupby(["lookup_condition", "alias", "cell"], as_index=False)
        .agg(
            delta_mae=("delta_mae", "mean"),
            interaction_delta_delta_mae=("interaction_delta_delta_mae", "mean"),
        )
    )
    rows = []
    for lookup_index, lookup_condition in enumerate(LOOKUP_CONDITIONS):
        for alias_index, alias in enumerate(sorted(EXPECTED_DISTURBANCE_ALIASES)):
            part = cells[
                (cells["lookup_condition"] == lookup_condition) & (cells["alias"] == alias)
            ]
            mean, ci_low, ci_high = cell_bootstrap(
                part["interaction_delta_delta_mae"],
                bootstrap_samples,
                seed=8200 + lookup_index * 100 + alias_index,
            )
            rows.append(
                {
                    "lookup_condition": lookup_condition,
                    "alias": alias,
                    "alias_label": ALIAS_LABELS[alias],
                    "interaction_mean": mean,
                    "interaction_ci_low": ci_low,
                    "interaction_ci_high": ci_high,
                    "interaction_ci_includes_zero": bool(ci_low <= 0.0 <= ci_high),
                    "n_cells": int(part["cell"].nunique()),
                    "bootstrap_samples": bootstrap_samples,
                }
            )
    return cells, pd.DataFrame(rows)


def load_recovery_trajectory(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, usecols=["index", "soc_ecm", "time_s"])


def infer_event_time(frame: pd.DataFrame) -> float:
    time_s = frame["time_s"].to_numpy(dtype=float)
    source_index = frame["index"].to_numpy(dtype=float)
    dt = np.diff(time_s) / np.diff(source_index)
    finite = dt[np.isfinite(dt) & (dt > 0.0)]
    nominal_dt = float(np.median(finite)) if len(finite) else 1.0
    return float(time_s[0] - source_index[0] * nominal_dt)


def build_recovery_statistics(
    manifest: dict, bootstrap_samples: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.DataFrame(
        [
            {
                "lookup_condition": row["lookup_condition"],
                "cell": row["cell"],
                "window_id": row["window_id"],
                "alias": row["alias"],
                "out_dir": row["out_dir"],
                "max_rows": row["max_rows"],
            }
            for row in manifest["runs"]
            if row["alias"] in FULL_OUTPUT_ALIASES
        ]
    )
    rows = []
    evaluation_start = int(manifest["evaluation_start_sample"])
    for lookup_condition in LOOKUP_CONDITIONS:
        for window_id in sorted(records["window_id"].unique()):
            pair = records[
                (records["lookup_condition"] == lookup_condition)
                & (records["window_id"] == window_id)
            ].set_index("alias")
            if set(pair.index) != FULL_OUTPUT_ALIASES:
                raise ValueError(f"Incomplete recovery pair: {lookup_condition} / {window_id}")
            base_path = next(
                Path(pair.loc[RECOVERY_BASELINE_ALIAS, "out_dir"]).glob(
                    "ecm_soc_fullcell_*.csv"
                )
            )
            shifted_path = next(Path(pair.loc[INITIAL_ALIAS, "out_dir"]).glob("ecm_soc_fullcell_*.csv"))
            baseline = load_recovery_trajectory(base_path)
            shifted = load_recovery_trajectory(shifted_path)
            event_time_s = infer_event_time(baseline)
            baseline = baseline[baseline["index"] >= evaluation_start]
            shifted = shifted[shifted["index"] >= evaluation_start]
            paired = baseline.merge(
                shifted,
                on=["index", "time_s"],
                suffixes=("_baseline", "_shifted"),
                validate="one_to_one",
            )
            recovery_metrics = compute_common_recovery_metrics(
                paired["time_s"].to_numpy(dtype=float),
                paired["soc_ecm_baseline"].to_numpy(dtype=float),
                paired["soc_ecm_shifted"].to_numpy(dtype=float),
                start_index=0,
                threshold=0.02,
                sustain_seconds=300.0,
                horizon_seconds=86400.0,
                event_time_s=event_time_s,
            )
            endpoint_h = float(
                recovery_metrics["common_stable_recovery_or_censor_time_h"]
            )
            censored = bool(recovery_metrics["common_stable_recovery_censored"])
            excess_area = float(recovery_metrics["common_recovery_excess_auc_soc_h"])
            rows.append(
                {
                    "lookup_condition": lookup_condition,
                    "cell": pair.iloc[0]["cell"],
                    "window_id": window_id,
                    "persistent_recovery_or_censor_h": endpoint_h,
                    "persistent_censored": censored,
                    "excess_error_area_soc_h": excess_area,
                }
            )
    recovery = pd.DataFrame(rows)
    cell = recovery.groupby(["lookup_condition", "cell"], as_index=False).agg(
        recovery_h=("persistent_recovery_or_censor_h", "mean"),
        censored_fraction=("persistent_censored", "mean"),
        excess_area=("excess_error_area_soc_h", "mean"),
    )
    nominal = cell[cell["lookup_condition"] == "nominal_lookup"][
        ["cell", "recovery_h", "censored_fraction", "excess_area"]
    ].rename(
        columns={
            "recovery_h": "nominal_recovery_h",
            "censored_fraction": "nominal_censored_fraction",
            "excess_area": "nominal_excess_area",
        }
    )
    cell = cell.merge(nominal, on="cell", validate="many_to_one")
    cell["recovery_delta_h"] = cell["recovery_h"] - cell["nominal_recovery_h"]
    cell["censored_fraction_delta"] = (
        cell["censored_fraction"] - cell["nominal_censored_fraction"]
    )
    cell["excess_area_delta"] = cell["excess_area"] - cell["nominal_excess_area"]
    stats = []
    for lookup_index, lookup_condition in enumerate(LOOKUP_CONDITIONS):
        part = cell[cell["lookup_condition"] == lookup_condition]
        row = {"lookup_condition": lookup_condition, "n_cells": int(part["cell"].nunique())}
        for metric_index, metric in enumerate(
            ["recovery_h", "recovery_delta_h", "censored_fraction", "excess_area"]
        ):
            mean, ci_low, ci_high = cell_bootstrap(
                part[metric], bootstrap_samples, seed=9300 + lookup_index * 10 + metric_index
            )
            row[metric] = mean
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        stats.append(row)
    return recovery, pd.DataFrame(stats)


def build_lookup_summary(
    baseline_cell: pd.DataFrame,
    scenario_stats: pd.DataFrame,
    recovery_stats: pd.DataFrame,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    for lookup_index, lookup_condition in enumerate(LOOKUP_CONDITIONS):
        baseline_part = baseline_cell[baseline_cell["lookup_condition"] == lookup_condition]
        baseline_mean, _, _ = cell_bootstrap(
            baseline_part["baseline_mae"], bootstrap_samples, 10100 + lookup_index
        )
        scenario_part = scenario_stats[
            (scenario_stats["lookup_condition"] == lookup_condition)
            & (scenario_stats["alias"] != INITIAL_ALIAS)
        ].copy()
        worst_index = scenario_part["interaction_mean"].abs().idxmax()
        worst = scenario_part.loc[worst_index]
        recovery = recovery_stats.set_index("lookup_condition").loc[lookup_condition]
        rows.append(
            {
                "lookup_condition": lookup_condition,
                "lookup_label": LOOKUP_LABELS[lookup_condition],
                "baseline_mae": baseline_mean,
                "worst_interaction_alias": worst["alias"],
                "worst_interaction_label": worst["alias_label"],
                "worst_absolute_interaction_mae": abs(float(worst["interaction_mean"])),
                "scenarios_ci_excluding_zero": int(
                    (~scenario_part["interaction_ci_includes_zero"]).sum()
                ),
                "scenario_count": int(len(scenario_part)),
                "recovery_h": float(recovery["recovery_h"]),
                "recovery_delta_h": float(recovery["recovery_delta_h"]),
                "recovery_censored_fraction": float(recovery["censored_fraction"]),
            }
        )
    return pd.DataFrame(rows)


def family_heatmap(scenario_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, aliases in ROBUSTNESS_FAMILIES.items():
        for lookup_condition in list(LOOKUP_CONDITIONS)[1:]:
            part = scenario_stats[
                (scenario_stats["lookup_condition"] == lookup_condition)
                & (scenario_stats["alias"].isin(aliases))
            ]
            selected = part.loc[part["interaction_mean"].abs().idxmax()]
            rows.append(
                {
                    "family": family,
                    "lookup_condition": lookup_condition,
                    "interaction_mean": float(selected["interaction_mean"]),
                    "source_alias": selected["alias"],
                }
            )
    return pd.DataFrame(rows)


def plot_results(
    family: pd.DataFrame,
    recovery_runs: pd.DataFrame,
    recovery_stats: pd.DataFrame,
    path: Path,
) -> None:
    setup_style()
    lookup_order = list(LOOKUP_CONDITIONS)[1:]
    family_order = list(ROBUSTNESS_FAMILIES)
    matrix = (
        family.pivot(index="family", columns="lookup_condition", values="interaction_mean")
        .loc[family_order, lookup_order]
        .to_numpy(dtype=float)
    )
    recovery = recovery_stats.set_index("lookup_condition").loc[lookup_order]
    recovery_cells = recovery_runs.groupby(
        ["lookup_condition", "cell"], as_index=False
    ).agg(
        recovery_h=("persistent_recovery_or_censor_h", "mean"),
        censored=("persistent_censored", "max"),
    )
    nominal_cells = recovery_cells[
        recovery_cells["lookup_condition"] == "nominal_lookup"
    ][["cell", "recovery_h"]].rename(columns={"recovery_h": "nominal_recovery_h"})
    recovery_cells = recovery_cells.merge(nominal_cells, on="cell", validate="many_to_one")
    recovery_cells["recovery_delta_h"] = (
        recovery_cells["recovery_h"] - recovery_cells["nominal_recovery_h"]
    )
    fig, (ax_heat, ax_recovery) = plt.subplots(
        1,
        2,
        figsize=(12.4, 6.1),
        gridspec_kw={"width_ratios": [2.35, 1.05]},
    )
    limit = max(float(np.nanmax(np.abs(matrix))), 1e-5)
    cmap = LinearSegmentedColormap.from_list(
        "diss_diverging", ["#566b78", "#f7f7f7", "#b6302d"]
    )
    image = ax_heat.imshow(
        matrix,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    ax_heat.set_xticks(
        np.arange(len(lookup_order)),
        [LOOKUP_PLOT_LABELS[item] for item in lookup_order],
    )
    ax_heat.set_yticks(np.arange(len(family_order)), family_order)
    ax_heat.set_title("(a) Largest interaction by disturbance family")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if abs(value) > 0.55 * limit else NEUTRAL_DARK
            ax_heat.text(
                column,
                row,
                f"{value:+.4f}",
                ha="center",
                va="center",
                fontsize=7.4,
                color=color,
            )
    colorbar = fig.colorbar(
        image,
        ax=ax_heat,
        orientation="horizontal",
        fraction=0.065,
        pad=0.16,
        aspect=42,
    )
    colorbar.set_label(r"Lookup interaction, $\Delta\Delta$MAE [SOC]")
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    x = np.arange(len(lookup_order), dtype=float)
    means = recovery["recovery_delta_h"].to_numpy(dtype=float)
    low = means - recovery["recovery_delta_h_ci_low"].to_numpy(dtype=float)
    high = recovery["recovery_delta_h_ci_high"].to_numpy(dtype=float) - means
    ax_recovery.errorbar(
        x,
        means,
        yerr=np.vstack([low, high]),
        fmt="o",
        color="#566b78",
        ecolor="#566b78",
        capsize=4,
        linewidth=1.3,
        markersize=5,
    )
    jitter = np.linspace(-0.09, 0.09, 6)
    for lookup_index, lookup_condition in enumerate(lookup_order):
        part = recovery_cells[
            recovery_cells["lookup_condition"] == lookup_condition
        ].sort_values("cell")
        for cell_index, row in enumerate(part.itertuples(index=False)):
            ax_recovery.scatter(
                lookup_index + jitter[cell_index],
                row.recovery_delta_h,
                s=22,
                facecolor="white" if row.censored else "#566b78",
                edgecolor="#b6302d" if row.censored else "white",
                linewidth=1.0 if row.censored else 0.45,
                zorder=5,
            )
            if row.censored:
                ax_recovery.annotate(
                    row.cell,
                    (lookup_index + jitter[cell_index], row.recovery_delta_h),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#b6302d",
                )
    ax_recovery.axhline(0.0, color=NEUTRAL_DARK, linestyle="--", linewidth=0.9)
    ax_recovery.set_xticks(
        x,
        [LOOKUP_PLOT_LABELS[item] for item in lookup_order],
    )
    ax_recovery.set_ylabel("Change in recovery/censor time [h]")
    ax_recovery.set_title("(b) Initialization recovery")
    clean_axes(ax_recovery)
    fig.suptitle("HECM lookup sensitivity across the complete disturbance benchmark", fontweight="bold")
    fig.tight_layout(w_pad=3.0)
    save_figure(fig, path)


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    labels = {
        "nominal_lookup": "Nominal lookup",
        "resistance_minus_10pct": r"Resistance $-10\%$",
        "resistance_plus_10pct": r"Resistance $+10\%$",
        "tau_minus_10pct": r"Time constants $-10\%$",
        "tau_plus_10pct": r"Time constants $+10\%$",
        "ocv_minus_10mV": r"OCV $-10$ mV",
        "ocv_plus_10mV": r"OCV $+10$ mV",
    }
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \caption{HECM lookup sensitivity across the complete disturbance benchmark. The largest robustness interaction is selected from the 20 measurement and signal-integrity subcases after equal-weight cell aggregation. Bracketed counts report subcases whose 95\% interaction interval excludes zero. Recovery is the persistent recovery-or-censor time from the paired initialization analysis, with the equal-weight censored-cell percentage in brackets.}",
        r"    \label{tab:hecm_lookup_sensitivity}",
        r"    \begin{tabularx}{0.99\textwidth}{@{}l>{\centering\arraybackslash}p{0.12\textwidth}>{\centering\arraybackslash}p{0.19\textwidth}Y>{\centering\arraybackslash}p{0.17\textwidth}@{}}",
        r"        \toprule",
        r"        Lookup condition & Baseline MAE & Max. robust $|\Delta\Delta\mathrm{MAE}|$ [CI] & Source subcase & Recovery/censor [h] [censored] \\",
        r"        \midrule",
    ]
    for row in summary.itertuples(index=False):
        if row.lookup_condition == "nominal_lookup":
            worst = "Reference"
            worst_label = "--"
            non_zero = "--"
        else:
            worst = f"{row.worst_absolute_interaction_mae:.5f}"
            worst_label = str(row.worst_interaction_label).replace("%", r"\%")
            non_zero = f"{row.scenarios_ci_excluding_zero}/{row.scenario_count}"
        lines.append(
            "        "
            f"{labels[row.lookup_condition]} & {row.baseline_mae:.4f} & "
            f"{worst} [{non_zero}] & {worst_label} & {row.recovery_h:.2f} "
            f"[{100.0 * row.recovery_censored_fraction:.0f}\\%] "
            + r"\\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabularx}",
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    out_root: Path,
    bootstrap_samples: int,
    figure_path: Path | None,
    table_path: Path | None,
) -> dict:
    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    metrics = load_metrics(manifest)
    disturbed, baseline_cell = add_interactions(metrics)
    scenario_cells, scenarios = scenario_statistics(disturbed, bootstrap_samples)
    recovery_runs, recovery_stats = build_recovery_statistics(manifest, bootstrap_samples)
    summary = build_lookup_summary(
        baseline_cell, scenarios, recovery_stats, bootstrap_samples
    )
    family = family_heatmap(scenarios)
    run_counts = metrics.groupby(
        ["lookup_condition", "alias"], as_index=False
    ).agg(
        runs=("alias", "size"),
        cells=("cell", "nunique"),
        windows=("window_id", "nunique"),
        evaluation_samples_min=("evaluation_samples", "min"),
        evaluation_samples_max=("evaluation_samples", "max"),
    )
    metrics.to_csv(out_root / "hecm_full_lookup_runs.csv", index=False)
    disturbed.to_csv(out_root / "hecm_full_lookup_interactions.csv", index=False)
    baseline_cell.to_csv(out_root / "hecm_full_lookup_baseline_cells.csv", index=False)
    run_counts.to_csv(out_root / "hecm_full_lookup_run_counts.csv", index=False)
    scenario_cells.to_csv(out_root / "hecm_full_lookup_scenario_cells.csv", index=False)
    scenarios.to_csv(out_root / "hecm_full_lookup_scenario_statistics.csv", index=False)
    recovery_runs.to_csv(out_root / "hecm_full_lookup_recovery_runs.csv", index=False)
    recovery_stats.to_csv(out_root / "hecm_full_lookup_recovery_statistics.csv", index=False)
    summary.to_csv(out_root / "hecm_full_lookup_summary.csv", index=False)
    family.to_csv(out_root / "hecm_full_lookup_family_heatmap.csv", index=False)
    protocol = {
        "analysis": "HECM lookup sensitivity across the complete JES2 disturbance benchmark",
        "scope": "HECM-only lookup sensitivity outside the cross-model score",
        "lookup_conditions": LOOKUP_CONDITIONS,
        "disturbance_aliases": sorted(EXPECTED_DISTURBANCE_ALIASES),
        "robustness_families": {
            key: sorted(value) for key, value in ROBUSTNESS_FAMILIES.items()
        },
        "initialization_recovery_alias": INITIAL_ALIAS,
        "runs": int(len(metrics)),
        "cells": sorted(metrics["cell"].unique().tolist()),
        "windows": sorted(metrics["window_id"].unique().tolist()),
        "evaluation_start_sample": int(manifest["evaluation_start_sample"]),
        "aggregation": (
            "seeds and windows averaged within each cell; equal-weight cell macro; "
            f"{bootstrap_samples}-draw cell bootstrap"
        ),
        "interaction_definition": (
            "scenario delta MAE under the perturbed lookup minus scenario delta MAE "
            "under the nominal lookup, each relative to its lookup-matched baseline"
        ),
        "recovery_definition": (
            "canonical paired baseline-versus-initialization recovery with 0.02 SOC "
            "threshold, 300 s qualification, 24 h horizon, and persistent "
            "recovery-or-censor endpoint"
        ),
        "interpretation_boundary": (
            "The analysis covers every declared JES2 disturbance subcase for the fixed HECM "
            "implementation under local one-at-a-time lookup perturbations. It does not cover "
            "combined lookup errors or other HECM structures."
        ),
    }
    (out_root / "hecm_full_lookup_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    if figure_path is None:
        figure_path = out_root / "Figure_HECM_Full_Lookup_Sensitivity.png"
    if table_path is None:
        table_path = out_root / "Table_HECM_Full_Lookup_Sensitivity.tex"
    plot_results(family, recovery_runs, recovery_stats, figure_path)
    write_latex_table(summary, table_path)
    perturbed_summary = summary[summary["lookup_condition"] != "nominal_lookup"]
    result = {
        "runs": int(len(metrics)),
        "disturbance_subcases": int(len(EXPECTED_DISTURBANCE_ALIASES)),
        "robustness_subcases": int(len(EXPECTED_DISTURBANCE_ALIASES - {INITIAL_ALIAS})),
        "lookup_conditions": int(len(LOOKUP_CONDITIONS)),
        "cells": int(metrics["cell"].nunique()),
        "windows": int(metrics["window_id"].nunique()),
        "figure": str(figure_path.resolve()),
        "table": str(table_path.resolve()),
        "largest_worst_case_interaction_mae": float(
            perturbed_summary["worst_absolute_interaction_mae"].max()
        ),
        "maximum_subcases_with_ci_excluding_zero": int(
            perturbed_summary["scenarios_ci_excluding_zero"].max()
        ),
        "maximum_absolute_recovery_change_h": float(
            perturbed_summary["recovery_delta_h"].abs().max()
        ),
    }
    (out_root / "analysis_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HECM lookup sensitivity across the complete JES2 benchmark."
    )
    parser.add_argument("--campaign_manifest", type=Path, required=True)
    parser.add_argument("--signed_manifest", type=Path, required=True)
    parser.add_argument("--gap_manifest", type=Path, required=True)
    parser.add_argument("--recovery_manifest", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--evaluation_start_sample", type=int, default=2023)
    parser.add_argument("--figure_path", type=Path, default=None)
    parser.add_argument("--table_path", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    for name in (
        "campaign_manifest",
        "signed_manifest",
        "gap_manifest",
        "recovery_manifest",
        "out_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.figure_path is not None:
        args.figure_path = args.figure_path.resolve()
    if args.table_path is not None:
        args.table_path = args.table_path.resolve()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.analyze_only:
        print(
            json.dumps(
                analyze(args.out_root, args.bootstrap_samples, args.figure_path, args.table_path),
                indent=2,
            )
        )
        return
    sources = load_sources(
        args.campaign_manifest,
        args.signed_manifest,
        args.gap_manifest,
        args.recovery_manifest,
    )
    records = build_records(
        sources,
        args.out_root,
        args.evaluation_start_sample,
        args.device,
    )
    manifest_path = args.out_root / "manifest.json"
    payload = {
        "analysis": "HECM lookup sensitivity across the complete JES2 disturbance benchmark",
        "started_utc": utc_now(),
        "source_campaign": str(args.campaign_manifest),
        "source_signed_campaign": str(args.signed_manifest),
        "source_gap_campaign": str(args.gap_manifest),
        "source_recovery_campaign": str(args.recovery_manifest),
        "evaluation_start_sample": args.evaluation_start_sample,
        "lookup_conditions": LOOKUP_CONDITIONS,
        "runs": records,
    }
    write_manifest(manifest_path, payload)
    if args.dry_run:
        print(json.dumps({"manifest": str(manifest_path), "runs": len(records)}, indent=2))
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute_record, record, args.skip_existing): record
            for record in records
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                status, error = future.result()
            except Exception as exc:  # pragma: no cover
                status, error = "failed", repr(exc)
            record["status"] = status
            if error is not None:
                record["error"] = error
            completed += 1
            if completed % 25 == 0 or completed == len(records) or status == "failed":
                payload["progress"] = {"completed": completed, "total": len(records)}
                write_manifest(manifest_path, payload)
                print(
                    f"[{completed:04d}/{len(records)}] last={status}: "
                    f"{record['lookup_condition']} / {record['alias']}",
                    flush=True,
                )
    payload["finished_utc"] = utc_now()
    failures = [record for record in records if record["status"] == "failed"]
    payload["failures"] = len(failures)
    write_manifest(manifest_path, payload)
    if failures:
        raise RuntimeError(f"{len(failures)} full HECM lookup-sensitivity runs failed")
    print(
        json.dumps(
            analyze(args.out_root, args.bootstrap_samples, args.figure_path, args.table_path),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
