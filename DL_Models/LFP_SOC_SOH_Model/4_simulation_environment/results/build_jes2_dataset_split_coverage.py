from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from jes2_plot_style import NEUTRAL_DARK, clean_axes, save_figure, setup_style


WORKSPACE = Path(__file__).resolve().parents[4]


SPLITS = {
    "Training": ["C01", "C03", "C05", "C11", "C17", "C23"],
    "Validation": ["C07", "C19", "C21"],
    "Holdout": ["C09", "C13", "C15", "C25", "C27", "C29"],
}

SPLIT_COLORS = {
    "Training": "#2ca02c",
    "Validation": "#9467bd",
    "Holdout": "#d62728",
}

HOLDOUT_LOAD_CLASS = {
    "C09": "Middle",
    "C13": "Middle",
    "C15": "Middle",
    "C25": "Low",
    "C27": "Low",
    "C29": "High*",
}


def extract_cell(path: Path, cell: str, split: str, stride: int) -> tuple[dict, np.ndarray, np.ndarray]:
    parquet = pq.ParquetFile(path)
    columns = ["Testtime[s]", "SOH", "Temperature[°C]", "C_Rate"]
    time_points: list[np.ndarray] = []
    soh_points: list[np.ndarray] = []
    c_rate_chunks: list[np.ndarray] = []
    row_offset = 0
    time_first = np.nan
    time_last = np.nan
    soh_min = np.inf
    soh_max = -np.inf
    temperature_min = np.inf
    temperature_max = -np.inf

    for batch in parquet.iter_batches(batch_size=500_000, columns=columns):
        names = batch.schema.names
        values = {
            name: batch.column(names.index(name)).to_numpy(zero_copy_only=False)
            for name in columns
        }
        time_s = np.asarray(values["Testtime[s]"], dtype=np.float64)
        soh = np.asarray(values["SOH"], dtype=np.float64)
        temperature = np.asarray(values["Temperature[°C]"], dtype=np.float64)
        c_rate = np.asarray(values["C_Rate"], dtype=np.float64)

        finite_time = time_s[np.isfinite(time_s)]
        if finite_time.size:
            if not np.isfinite(time_first):
                time_first = float(finite_time[0])
            time_last = float(finite_time[-1])

        finite_soh = soh[np.isfinite(soh)]
        if finite_soh.size:
            soh_min = min(soh_min, float(finite_soh.min()))
            soh_max = max(soh_max, float(finite_soh.max()))

        finite_temperature = temperature[np.isfinite(temperature)]
        if finite_temperature.size:
            temperature_min = min(temperature_min, float(finite_temperature.min()))
            temperature_max = max(temperature_max, float(finite_temperature.max()))

        finite_c_rate = np.abs(c_rate[np.isfinite(c_rate)])
        if finite_c_rate.size:
            c_rate_chunks.append(finite_c_rate.astype(np.float32, copy=False))

        first_index = (-row_offset) % stride
        indices = np.arange(first_index, len(time_s), stride, dtype=np.int64)
        valid = np.isfinite(time_s[indices]) & np.isfinite(soh[indices])
        time_points.append(time_s[indices][valid])
        soh_points.append(soh[indices][valid])
        row_offset += len(time_s)

    if not c_rate_chunks or not np.isfinite(time_first) or not np.isfinite(time_last):
        raise ValueError(f"Incomplete measurement data for {cell}: {path}")

    c_rate_values = np.concatenate(c_rate_chunks)
    duration_days = (time_last - time_first) / 86_400.0
    metadata = {
        "cell": cell,
        "split": split,
        "duration_days": float(duration_days),
        "soh_min": float(soh_min),
        "soh_max": float(soh_max),
        "temperature_min_c": float(temperature_min),
        "temperature_max_c": float(temperature_max),
        "abs_c_rate_p95": float(np.percentile(c_rate_values, 95.0)),
        "holdout_load_class": HOLDOUT_LOAD_CLASS.get(cell, "Not applicable"),
    }
    time_days = (np.concatenate(time_points) - time_first) / 86_400.0
    soh_trace = np.concatenate(soh_points)
    return metadata, time_days, soh_trace


def distribute_labels(values: list[float], lower: float, upper: float, gap: float) -> list[float]:
    order = np.argsort(values)
    placed = np.asarray(values, dtype=float)[order]
    placed[0] = max(placed[0], lower)
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)
    overflow = placed[-1] - upper
    if overflow > 0:
        placed -= overflow
        for index in range(len(placed) - 2, -1, -1):
            placed[index] = min(placed[index], placed[index + 1] - gap)
    result = np.empty_like(placed)
    result[order] = placed
    return result.tolist()


def plot_split_coverage(
    traces: dict[str, list[tuple[str, np.ndarray, np.ndarray]]],
    output: Path,
) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4), sharey=True)
    for panel, (ax, split) in enumerate(zip(axes, SPLITS)):
        color = SPLIT_COLORS[split]
        endpoints = []
        for cell, time_days, soh in traces[split]:
            ax.plot(time_days, soh, color=color, linewidth=1.65, alpha=0.78)
            endpoints.append(float(soh[-1]))
        label_positions = distribute_labels(endpoints, 0.605, 0.992, 0.022)
        max_days = max(float(time_days[-1]) for _, time_days, _ in traces[split])
        label_x = max_days * 1.035
        for (cell, time_days, soh), label_y in zip(traces[split], label_positions):
            suffix = ""
            if split == "Holdout":
                suffix = f"  {HOLDOUT_LOAD_CLASS[cell].lower()}"
            ax.plot(
                [time_days[-1], label_x * 0.99],
                [soh[-1], label_y],
                color=color,
                linewidth=0.8,
                alpha=0.55,
            )
            ax.text(
                label_x,
                label_y,
                f"{cell}{suffix}",
                color=color,
                fontsize=8.5,
                fontweight="semibold",
                va="center",
            )
        ax.set_xlim(0, max_days * 1.25)
        ax.set_ylim(0.59, 1.012)
        ax.set_xlabel("Test time [days]")
        ax.set_title(f"({chr(97 + panel)}) {split} cells (n={len(SPLITS[split])})")
        clean_axes(ax)
    axes[0].set_ylabel("Reference SOH [-]")
    fig.suptitle(
        "Cell-disjoint model-development and holdout split across the LFP aging campaign",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.015,
        "Holdout load labels are descriptive groups based on measured 95th-percentile absolute C-rate; high contains C29 only.",
        ha="center",
        fontsize=9,
        color=NEUTRAL_DARK,
    )
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, output)


def write_latex_table(rows: list[dict], output: Path) -> None:
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Measured coverage of all cells in the fixed model-development and holdout split. The 95th-percentile absolute C-rate is calculated over each complete 1-Hz trajectory. Load classes are defined only for the six holdout cells. The high-load class contains C29 only and is exploratory.}",
        r"\label{tab:dataset_cell_split_coverage}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Cell & Split & Duration [d] & SOH range & $T$ range [$^{\circ}$C] & P95 $|C_{\mathrm{rate}}|$ & Holdout load \\",
        r"\midrule",
    ]
    previous_split = None
    for row in rows:
        if previous_split is not None and row["split"] != previous_split:
            lines.append(r"\addlinespace[2pt]")
        load_class = row["holdout_load_class"].replace("Not applicable", "--")
        lines.append(
            f'{row["cell"]} & {row["split"]} & {row["duration_days"]:.1f} & '
            f'{row["soh_min"]:.3f}--{row["soh_max"]:.3f} & '
            f'{row["temperature_min_c"]:.1f}--{row["temperature_max_c"]:.1f} & '
            f'{row["abs_c_rate_p95"]:.2f} & {load_class} \\\\'
        )
        previous_split = row["split"]
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the JES2 full-cell split and aging-coverage figure.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"),
    )
    parser.add_argument("--figure_path", type=Path, required=True)
    parser.add_argument(
        "--source_figure",
        type=Path,
        default=(
            WORKSPACE
            / "LATEX"
            / "DISS"
            / "Florian_Rzepka_Dissertation"
            / "pictures"
            / "eaai_palette"
            / "embedded_soh_all_days.png"
        ),
    )
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--table_path", type=Path, required=True)
    parser.add_argument("--trajectory_stride", type=int, default=3600)
    args = parser.parse_args()
    if args.trajectory_stride < 1:
        parser.error("--trajectory_stride must be at least 1")

    rows = []
    traces: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {split: [] for split in SPLITS}
    for split, cells in SPLITS.items():
        for cell in cells:
            path = args.data_root / f"df_FE_{cell}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            metadata, time_days, soh = extract_cell(path, cell, split, args.trajectory_stride)
            rows.append(metadata)
            traces[split].append((cell, time_days, soh))
            print(f"{cell}: {metadata['duration_days']:.1f} d, SOH {metadata['soh_min']:.3f}-{metadata['soh_max']:.3f}")

    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_latex_table(rows, args.table_path)
    if not args.source_figure.is_file():
        raise FileNotFoundError(args.source_figure)
    args.figure_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source_figure, args.figure_path)
    print(json.dumps({"cells": len(rows), "figure": str(args.figure_path), "table": str(args.table_path)}, indent=2))


if __name__ == "__main__":
    main()
