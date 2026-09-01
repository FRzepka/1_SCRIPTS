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
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
RUNNER = ROOT / "ECM_0.0.3" / "run_ecm_scenario.py"
sys.path.insert(0, str(ROOT / "results"))
from jes2_plot_style import MODEL_COLORS, NEUTRAL_DARK, clean_axes, save_figure, setup_style


TABLE_CONDITIONS = {
    "nominal_lookup": {},
    "resistance_minus_10pct": {"--ecm_resistance_scale": "0.90"},
    "resistance_plus_10pct": {"--ecm_resistance_scale": "1.10"},
    "ocv_minus_10mV": {"--ecm_ocv_offset_v": "-0.010"},
    "ocv_plus_10mV": {"--ecm_ocv_offset_v": "0.010"},
}

GAIN_CONDITIONS = {
    "gain_minus_3pct": {
        "alias": "current_bias_neg_3p0pct",
        "scenario": "current_offset",
        "current_gain_pct": -0.03,
    },
    "gain_nominal": {
        "alias": "baseline",
        "scenario": "baseline",
        "current_gain_pct": 0.0,
    },
    "gain_plus_3pct": {
        "alias": "current_bias_3p0pct",
        "scenario": "current_offset",
        "current_gain_pct": 0.03,
    },
}

DISPLAY_LABELS = {
    "nominal_lookup": "Nominal\nlookup",
    "resistance_minus_10pct": "Resistance\n-10%",
    "resistance_plus_10pct": "Resistance\n+10%",
    "ocv_minus_10mV": "OCV\n-10 mV",
    "ocv_plus_10mV": "OCV\n+10 mV",
}

PERTURBED_TABLE_CONDITIONS = [
    "resistance_minus_10pct",
    "resistance_plus_10pct",
    "ocv_minus_10mV",
    "ocv_plus_10mV",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_value(record: dict, flag: str) -> str | None:
    command = record.get("command", [])
    if flag not in command:
        return None
    index = command.index(flag)
    return command[index + 1] if index + 1 < len(command) else None


def resolve_trace(record: dict) -> Path:
    value = record.get("soh_trace") or command_value(record, "--soh_trace")
    if not value:
        raise ValueError(
            f"No SOH trace recorded for {record.get('window_id')} / {record.get('alias')}"
        )
    path = Path(value)
    if not path.is_absolute():
        path = WORKSPACE / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def load_source_records(main_manifest: Path, signed_manifest: Path) -> list[dict]:
    main = json.loads(main_manifest.read_text(encoding="utf-8"))
    signed = json.loads(signed_manifest.read_text(encoding="utf-8"))
    source_rows = list(main.get("runs", [])) + list(signed.get("runs", []))
    aliases = {item["alias"] for item in GAIN_CONDITIONS.values()}
    selected = [
        row
        for row in source_rows
        if row.get("model") == "HECM"
        and row.get("soh_condition") == "lstm_h1"
        and row.get("alias") in aliases
    ]
    keyed: dict[tuple[str, str], dict] = {}
    for row in selected:
        key = (row["window_id"], row["alias"])
        if key in keyed:
            raise ValueError(f"Duplicate HECM source record: {key}")
        keyed[key] = row

    windows = sorted(
        row["window_id"] for row in selected if row.get("alias") == "baseline"
    )
    if len(windows) != 16:
        raise ValueError(f"Expected 16 baseline HECM windows, found {len(windows)}")
    for window_id in windows:
        for alias in aliases:
            if (window_id, alias) not in keyed:
                raise ValueError(f"Missing HECM source record: {window_id} / {alias}")
    return [keyed[(window_id, alias)] for window_id in windows for alias in aliases]


def build_run_records(
    sources: list[dict],
    out_root: Path,
    data_root: Path,
    device: str,
    evaluation_start_sample: int,
) -> list[dict]:
    source_by_key = {(row["window_id"], row["alias"]): row for row in sources}
    windows = sorted({row["window_id"] for row in sources})
    records = []
    for window_id in windows:
        for table_condition, table_parameters in TABLE_CONDITIONS.items():
            for gain_condition, gain_definition in GAIN_CONDITIONS.items():
                source = source_by_key[(window_id, gain_definition["alias"])]
                out_dir = out_root / "runs" / source["cell"] / window_id / table_condition / gain_condition
                trace = resolve_trace(source)
                command = [
                    sys.executable,
                    str(RUNNER),
                    "--cell",
                    source["cell"],
                    "--scenario",
                    gain_definition["scenario"],
                    "--seed",
                    str(source["seed"]),
                    "--soh_trace",
                    str(trace),
                    "--data_root",
                    str(data_root),
                    "--device",
                    device,
                    "--out_dir",
                    str(out_dir),
                    "--start_row",
                    str(int(source["start_row"])),
                    "--max_rows",
                    str(int(source["max_rows"])),
                    "--evaluation_start_sample",
                    str(evaluation_start_sample),
                    "--summary_only",
                ]
                if gain_definition["scenario"] == "current_offset":
                    command.extend(
                        ["--current_offset_pct", str(gain_definition["current_gain_pct"])]
                    )
                for flag, value in table_parameters.items():
                    command.extend([flag, value])
                records.append(
                    {
                        "cell": source["cell"],
                        "window_id": window_id,
                        "soh_state": source["soh_state"],
                        "cell_load_class": source["cell_load_class"],
                        "start_row": int(source["start_row"]),
                        "max_rows": int(source["max_rows"]),
                        "seed": int(source["seed"]),
                        "table_condition": table_condition,
                        "gain_condition": gain_condition,
                        "current_gain_pct": gain_definition["current_gain_pct"],
                        "source_alias": gain_definition["alias"],
                        "soh_trace": str(trace),
                        "out_dir": str(out_dir),
                        "command": command,
                        "status": "pending",
                    }
                )
    expected = len(windows) * len(TABLE_CONDITIONS) * len(GAIN_CONDITIONS)
    if len(records) != expected or expected != 240:
        raise ValueError(f"Expected 240 HECM sensitivity runs, built {len(records)}")
    return records


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def execute_record(record: dict, skip_existing: bool) -> tuple[str, str | None]:
    out_dir = Path(record["out_dir"])
    summary_path = out_dir / "summary.json"
    if skip_existing and summary_path.is_file():
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


def build_statistics(metrics: pd.DataFrame, bootstrap_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = metrics[metrics["gain_condition"] == "gain_nominal"].copy()
    baseline_cell = (
        baseline.groupby(["cell", "table_condition"], as_index=False)
        .agg(baseline_mae=("mae", "mean"), baseline_rmse=("rmse", "mean"))
    )

    table_baseline = baseline[
        ["cell", "window_id", "table_condition", "mae", "rmse"]
    ].rename(columns={"mae": "table_baseline_mae", "rmse": "table_baseline_rmse"})
    signed = metrics[metrics["gain_condition"] != "gain_nominal"].merge(
        table_baseline,
        on=["cell", "window_id", "table_condition"],
        how="left",
        validate="many_to_one",
    )
    signed["delta_mae"] = signed["mae"] - signed["table_baseline_mae"]
    signed["delta_rmse"] = signed["rmse"] - signed["table_baseline_rmse"]
    signed_cell = (
        signed.groupby(["cell", "table_condition", "gain_condition"], as_index=False)
        .agg(delta_mae=("delta_mae", "mean"), delta_rmse=("delta_rmse", "mean"))
    )
    adverse = signed_cell.loc[
        signed_cell.groupby(["cell", "table_condition"])["delta_mae"].idxmax()
    ].rename(
        columns={
            "gain_condition": "adverse_gain_condition",
            "delta_mae": "adverse_gain_delta_mae",
            "delta_rmse": "adverse_gain_delta_rmse",
        }
    )
    cell_results = baseline_cell.merge(
        adverse,
        on=["cell", "table_condition"],
        how="inner",
        validate="one_to_one",
    )
    nominal_penalty = cell_results[cell_results["table_condition"] == "nominal_lookup"][
        ["cell", "adverse_gain_delta_mae"]
    ].rename(columns={"adverse_gain_delta_mae": "nominal_lookup_gain_delta_mae"})
    cell_results = cell_results.merge(nominal_penalty, on="cell", validate="many_to_one")
    cell_results["gain_interaction_delta_delta_mae"] = (
        cell_results["adverse_gain_delta_mae"]
        - cell_results["nominal_lookup_gain_delta_mae"]
    )

    rows = []
    metrics_to_summarize = [
        "baseline_mae",
        "adverse_gain_delta_mae",
        "gain_interaction_delta_delta_mae",
    ]
    for table_index, table_condition in enumerate(TABLE_CONDITIONS):
        part = cell_results[cell_results["table_condition"] == table_condition]
        for metric_index, metric in enumerate(metrics_to_summarize):
            mean, ci_low, ci_high = cell_bootstrap(
                part[metric],
                bootstrap_samples,
                seed=3110 + 100 * table_index + metric_index,
            )
            rows.append(
                {
                    "table_condition": table_condition,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_cells": int(part["cell"].nunique()),
                    "n_windows": int(metrics["window_id"].nunique()),
                    "bootstrap_samples": bootstrap_samples,
                }
            )
    return cell_results, pd.DataFrame(rows)


def build_compact_statistics(statistics: pd.DataFrame) -> pd.DataFrame:
    indexed = statistics.set_index(["table_condition", "metric"])
    rows = []
    for table_condition in TABLE_CONDITIONS:
        baseline = indexed.loc[(table_condition, "baseline_mae")]
        penalty = indexed.loc[(table_condition, "adverse_gain_delta_mae")]
        interaction = indexed.loc[(table_condition, "gain_interaction_delta_delta_mae")]
        rows.append(
            {
                "table_condition": table_condition,
                "display_label": DISPLAY_LABELS[table_condition].replace("\n", " "),
                "baseline_mae": float(baseline["mean"]),
                "adverse_gain_delta_mae": float(penalty["mean"]),
                "interaction_delta_delta_mae": float(interaction["mean"]),
                "interaction_ci_low": float(interaction["ci_low"]),
                "interaction_ci_high": float(interaction["ci_high"]),
                "interaction_ci_includes_zero": bool(
                    float(interaction["ci_low"]) <= 0.0 <= float(interaction["ci_high"])
                ),
            }
        )
    return pd.DataFrame(rows)


def write_compact_latex_table(compact: pd.DataFrame, path: Path) -> None:
    labels = {
        "nominal_lookup": "Nominal lookup",
        "resistance_minus_10pct": r"Resistance $-10\%$",
        "resistance_plus_10pct": r"Resistance $+10\%$",
        "ocv_minus_10mV": r"OCV $-10$ mV",
        "ocv_plus_10mV": r"OCV $+10$ mV",
    }
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \footnotesize",
        r"    \caption{Compact HECM lookup sensitivity. Lookup calibration changes baseline accuracy, whereas the adverse $\pm3\%$ current-gain interaction remains small and all 95\% intervals include zero.}",
        r"    \label{tab:hecm_lookup_sensitivity}",
        r"    \resizebox{\columnwidth}{!}{%",
        r"    \begin{tabular}{lccc}",
        r"        \toprule",
        r"        Lookup condition & Baseline MAE & Gain $\Delta$MAE & Interaction $\Delta\Delta$MAE [95\% CI] \\",
        r"        \midrule",
    ]
    for row in compact.itertuples(index=False):
        if row.table_condition == "nominal_lookup":
            interaction = "Reference"
        else:
            interaction = (
                f"{row.interaction_delta_delta_mae:+.5f} "
                f"[{row.interaction_ci_low:+.5f}, {row.interaction_ci_high:+.5f}]"
            )
        lines.append(
            "        "
            f"{labels[row.table_condition]} & {row.baseline_mae:.4f} & "
            f"{row.adverse_gain_delta_mae:.4f} & {interaction} "
            + r"\\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    }",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(cell_results: pd.DataFrame, statistics: pd.DataFrame, path: Path) -> None:
    setup_style()
    order = PERTURBED_TABLE_CONDITIONS
    x = np.arange(len(order), dtype=float)
    color = MODEL_COLORS["HECM"]
    metric = "gain_interaction_delta_delta_mae"
    fig, ax = plt.subplots(figsize=(7.1, 4.4))
    cells = sorted(cell_results["cell"].unique())
    jitter = np.linspace(-0.10, 0.10, len(cells))
    stats = statistics[statistics["metric"] == metric].set_index("table_condition").loc[order]
    means = stats["mean"].to_numpy(dtype=float)
    low = means - stats["ci_low"].to_numpy(dtype=float)
    high = stats["ci_high"].to_numpy(dtype=float) - means
    ax.bar(
        x,
        means,
        width=0.64,
        color=to_rgba(color, 0.22),
        edgecolor=color,
        linewidth=1.4,
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([low, high]),
        fmt="none",
        ecolor=color,
        elinewidth=1.25,
        capsize=4,
        capthick=1.25,
        zorder=4,
    )
    for cell_index, cell in enumerate(cells):
        part = cell_results[cell_results["cell"] == cell].set_index("table_condition").loc[order]
        ax.scatter(
            x + jitter[cell_index],
            part[metric],
            s=21,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.78,
            zorder=5,
        )
    ax.axhline(0.0, color=NEUTRAL_DARK, linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xticks(x, [DISPLAY_LABELS[item] for item in order])
    ax.set_ylabel(r"Lookup $\times$ gain interaction, $\Delta\Delta$MAE [SOC]")
    ax.set_title("HECM lookup × current-gain interaction")
    clean_axes(ax)
    span = max(float(np.nanmax(high + means) - np.nanmin(means - low)), 1e-4)
    for position, value, upper in zip(x, means, high):
        vertical = value + upper + 0.035 * span
        ax.text(
            position,
            vertical,
            f"{value:+.5f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=NEUTRAL_DARK,
        )
    ax.text(
        0.02,
        0.97,
        "All 95% intervals include zero",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=NEUTRAL_DARK,
    )
    fig.tight_layout()
    save_figure(fig, path)


def analyze(
    out_root: Path,
    bootstrap_samples: int,
    figure_path: Path | None,
    table_path: Path | None,
) -> dict:
    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for record in manifest["runs"]:
        summary_path = Path(record["out_dir"]) / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluation_start = int(summary.get("evaluation_start_sample", -1))
        evaluation_samples = int(summary.get("evaluation_samples", -1))
        if evaluation_start != manifest["evaluation_start_sample"]:
            raise ValueError(f"Common evaluation start missing in {summary_path}")
        if evaluation_samples != int(record["max_rows"]) - evaluation_start:
            raise ValueError(f"Common evaluation sample count mismatch in {summary_path}")
        rows.append(
            {
                key: record[key]
                for key in (
                    "cell",
                    "window_id",
                    "soh_state",
                    "cell_load_class",
                    "table_condition",
                    "gain_condition",
                    "current_gain_pct",
                    "start_row",
                    "max_rows",
                )
            }
            | {
                "evaluation_start_sample": evaluation_start,
                "evaluation_samples": evaluation_samples,
                "mae": float(summary["mae"]),
                "rmse": float(summary["rmse"]),
                "p95_error": float(summary["p95_error"]),
            }
        )
    metrics = pd.DataFrame(rows)
    if len(metrics) != 240:
        raise ValueError(f"Expected 240 completed metrics, found {len(metrics)}")
    cell_results, statistics = build_statistics(metrics, bootstrap_samples)
    compact = build_compact_statistics(statistics)
    metrics.to_csv(out_root / "hecm_lookup_sensitivity_runs.csv", index=False)
    cell_results.to_csv(out_root / "hecm_lookup_sensitivity_cells.csv", index=False)
    statistics.to_csv(out_root / "hecm_lookup_sensitivity_statistics.csv", index=False)
    compact.to_csv(out_root / "hecm_lookup_sensitivity_compact.csv", index=False)
    protocol = {
        "analysis": "HECM lookup-table x current-gain sensitivity",
        "scope": "HECM-only explanatory analysis outside the cross-model robustness score",
        "table_conditions": TABLE_CONDITIONS,
        "gain_conditions": GAIN_CONDITIONS,
        "evaluation_start_sample": int(manifest["evaluation_start_sample"]),
        "evaluation_samples_per_window": int(metrics["evaluation_samples"].unique().item()),
        "cells": sorted(metrics["cell"].unique().tolist()),
        "windows": sorted(metrics["window_id"].unique().tolist()),
        "aggregation": (
            "windows averaged within each cell; equal-weight cell macro; "
            f"{bootstrap_samples}-draw cell bootstrap confidence intervals"
        ),
        "adverse_gain_definition": (
            "larger cell-level delta MAE from the matched -3% and +3% current-gain runs"
        ),
        "interaction_definition": (
            "adverse current-gain delta MAE under the perturbed lookup minus the "
            "adverse current-gain delta MAE under the nominal lookup"
        ),
        "interpretation_boundary": (
            "The one-at-a-time shifts test whether the observed current-gain response "
            "depends strongly on local resistance and OCV calibration changes. They "
            "are not a general parameter-uncertainty analysis across all disturbances."
        ),
    }
    (out_root / "hecm_lookup_sensitivity_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    if figure_path is None:
        figure_path = out_root / "Figure_HECM_Lookup_Table_Sensitivity.png"
    if table_path is None:
        table_path = out_root / "Table_HECM_Lookup_Table_Sensitivity.tex"
    plot_results(cell_results, statistics, figure_path)
    write_compact_latex_table(compact, table_path)
    perturbed = compact[compact["table_condition"] != "nominal_lookup"]
    summary = {
        "runs": len(metrics),
        "windows": int(metrics["window_id"].nunique()),
        "cells": int(metrics["cell"].nunique()),
        "figure": str(figure_path.resolve()),
        "statistics": str((out_root / "hecm_lookup_sensitivity_statistics.csv").resolve()),
        "compact_statistics": str((out_root / "hecm_lookup_sensitivity_compact.csv").resolve()),
        "compact_table": str(table_path.resolve()),
        "largest_absolute_macro_interaction": float(
            statistics[
                statistics["metric"] == "gain_interaction_delta_delta_mae"
            ]["mean"].abs().max()
        ),
        "all_perturbed_interaction_intervals_include_zero": bool(
            perturbed["interaction_ci_includes_zero"].all()
        ),
    }
    (out_root / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final HECM lookup-table x current-gain sensitivity analysis."
    )
    parser.add_argument("--campaign_manifest", type=Path, required=True)
    parser.add_argument("--signed_manifest", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--evaluation_start_sample", type=int, default=2023)
    parser.add_argument("--figure_path", type=Path, default=None)
    parser.add_argument("--table_path", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.campaign_manifest = args.campaign_manifest.resolve()
    args.signed_manifest = args.signed_manifest.resolve()
    args.out_root = args.out_root.resolve()
    if args.figure_path is not None:
        args.figure_path = args.figure_path.resolve()
    if args.table_path is not None:
        args.table_path = args.table_path.resolve()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if args.analyze_only:
        print(
            json.dumps(
                analyze(
                    args.out_root,
                    args.bootstrap_samples,
                    args.figure_path,
                    args.table_path,
                ),
                indent=2,
            )
        )
        return

    sources = load_source_records(args.campaign_manifest, args.signed_manifest)
    records = build_run_records(
        sources=sources,
        out_root=args.out_root,
        data_root=args.data_root.resolve(),
        device=args.device,
        evaluation_start_sample=args.evaluation_start_sample,
    )
    manifest_path = args.out_root / "manifest.json"
    payload = {
        "analysis": "HECM lookup-table x current-gain sensitivity",
        "scope": "explanatory HECM-only analysis outside the cross-model robustness score",
        "started_utc": utc_now(),
        "source_campaign": str(args.campaign_manifest),
        "source_signed_campaign": str(args.signed_manifest),
        "evaluation_start_sample": args.evaluation_start_sample,
        "table_conditions": TABLE_CONDITIONS,
        "gain_conditions": GAIN_CONDITIONS,
        "runs": records,
    }
    write_manifest(manifest_path, payload)
    if args.dry_run:
        print(json.dumps({"manifest": str(manifest_path), "runs": len(records)}, indent=2))
        return

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_record = {
            executor.submit(execute_record, record, args.skip_existing): record
            for record in records
        }
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                status, error = future.result()
            except Exception as exc:  # pragma: no cover - defensive subprocess boundary
                status, error = "failed", repr(exc)
            record["status"] = status
            if error is not None:
                record["error"] = error
            completed += 1
            payload["progress"] = {"completed": completed, "total": len(records)}
            write_manifest(manifest_path, payload)
            print(
                f"[{completed:03d}/{len(records)}] {status}: "
                f"{record['window_id']} / {record['table_condition']} / {record['gain_condition']}",
                flush=True,
            )

    payload["finished_utc"] = utc_now()
    failures = [record for record in records if record["status"] == "failed"]
    payload["failures"] = len(failures)
    write_manifest(manifest_path, payload)
    if failures:
        raise RuntimeError(f"{len(failures)} HECM sensitivity runs failed")
    print(
        json.dumps(
            analyze(
                args.out_root,
                args.bootstrap_samples,
                args.figure_path,
                args.table_path,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
