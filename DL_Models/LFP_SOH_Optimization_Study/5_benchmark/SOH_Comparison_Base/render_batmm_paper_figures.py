#!/usr/bin/env python3
"""Render the EAAI baseline figures from the BATMM evaluation results.

The model comparison uses the exact window/state protocol on the selected C11
benchmark trajectory. The trajectory plot uses the continuous-context
predictions generated locally. The capacity plot retains the original
size-sweep points and replaces the four architectures for which BATMM
checkpoints were supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
MODEL_ORDER = ["cnn", "gru", "lstm", "tcn"]
MODEL_LABELS = {name: name.upper() for name in MODEL_ORDER}
COLORS = {
    "cnn": "#59C7C2",
    "gru": "#59E83A",
    "lstm": "#E76B91",
    "tcn": "#294862",
}


# Original C11 capacity-sweep values.  The entries marked below as BATMM are
# replaced at render time by the newly evaluated C11 MAE values.
CAPACITY_SWEEP = {
    "cnn": [
        ("s3_h64", 126721, 0.0231),
        ("s2_h80", 197441, 0.0195),
        ("s1_h96", 283777, 0.0199),
        ("base_h128", 503297, 0.0149),
        ("l1_h160", 785281, 0.0332),
        ("l2_h192", 1129729, 0.0211),
        ("l3_h224", 1536641, 0.0136),
    ],
    "gru": [
        ("s3_h48", 97825, 0.0289),
        ("s2_h64", 172417, 0.0316),
        ("s1_h96", 330337, 0.0195),
        ("base_h128", 557633, 0.0176),
        ("l1_h160", 844321, 0.0223),
        ("l2_h192", 1190401, 0.0236),
        ("l3_h224", 1595873, 0.0246),
    ],
    "lstm": [
        ("s3_h64", 94017, 0.0204),
        ("s2_h96", 217697, 0.0284),
        ("s1_h128", 392577, 0.0317),
        ("base_h160", 618657, 0.0149),
        ("l1_h192", 895937, 0.0230),
        ("l2_h224", 1224417, 0.0155),
        ("l3_h256", 1604097, 0.0193),
    ],
    "tcn": [
        ("s3_h48", 113169, 0.0198),
        ("s2_h64", 197985, 0.0204),
        ("s1_h80", 306353, 0.0147),
        ("base_h96", 439841, 0.0125),
        ("l1_h112", 597393, 0.0126),
        ("l2_h128", 779009, 0.0133),
        ("l3_h144", 984689, 0.0147),
    ],
}
BATMM_SWEEP_POINT = {
    "cnn": "base_h128",
    "gru": "l1_h160",
    "lstm": "l1_h192",
    "tcn": "base_h96",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        help="Complete BATMM_RESULTS_* folder. Defaults to the latest one.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "FINAL_RESULTS",
        help="Destination for the six PNG figures and their source tables.",
    )
    return parser.parse_args()


def find_result_dir(requested: Path | None) -> Path:
    if requested is not None:
        result_dir = requested.resolve()
    else:
        final_dir = RESULTS_ROOT / "FINAL_RESULTS"
        if (
            (final_dir / "metrics_summary.csv").is_file()
            and (final_dir / "model_inventory.csv").is_file()
            and (final_dir / "selected_baseline_soh_trajectory_data.csv").is_file()
        ):
            return final_dir
        candidates = sorted(
            RESULTS_ROOT.glob("BATMM_RESULTS_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        result_dir = next(
            (
                path
                for path in candidates
                if (path / "metrics_summary.csv").is_file()
                and (path / "model_inventory.csv").is_file()
                and (path / "predictions").is_dir()
            ),
            None,
        )
        if result_dir is None:
            raise FileNotFoundError("No complete BATMM_RESULTS_* folder was found.")
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)
    return result_dir


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.2,
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_batmm_window_metrics(
    result_dir: Path, cell: str = "df_FE_C11"
) -> pd.DataFrame:
    """Read exact BATMM metrics for the selected benchmark trajectory."""
    metrics_path = result_dir / "metrics_by_cell_windowed.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    source = pd.read_csv(metrics_path)
    metrics = source[source["cell"] == cell].copy()
    if set(metrics.get("model", [])) != set(MODEL_ORDER):
        raise ValueError(
            f"Could not read all BATMM metrics for {cell} from {metrics_path}"
        )
    metrics["aggregation"] = "selected BATMM benchmark trajectory"
    return metrics


def blend_with_white(color: str, white_fraction: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb * (1.0 - white_fraction) + white_fraction)


def save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> None:
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches=None)
    plt.close(fig)


def render_model_comparison(
    reported_metrics: pd.DataFrame, inventory: pd.DataFrame, output_dir: Path
) -> None:
    metrics = reported_metrics.set_index("model").loc[MODEL_ORDER]
    models = inventory.set_index("model").loc[MODEL_ORDER]
    source = pd.DataFrame(
        {
            "architecture": [MODEL_LABELS[name] for name in MODEL_ORDER],
            "mae": metrics["mae"].to_numpy(),
            "rmse": metrics["rmse"].to_numpy(),
            "fp32_weights_mib": models["fp32_weights_mib"].to_numpy(),
            "aggregation": metrics["aggregation"].to_numpy(),
        }
    )
    source.to_csv(output_dir / "baseline_model_comparison_data.csv", index=False)

    fig, (ax_error, ax_size) = plt.subplots(1, 2, figsize=(20, 9.2), dpi=120)
    fig.subplots_adjust(left=0.073, right=0.985, top=0.91, bottom=0.18, wspace=0.23)
    positions = np.arange(len(MODEL_ORDER))
    width = 0.25

    for index, model in enumerate(MODEL_ORDER):
        color = COLORS[model]
        mae = float(metrics.loc[model, "mae"])
        rmse = float(metrics.loc[model, "rmse"])
        ax_error.bar(
            index - width / 1.6,
            mae,
            width,
            color=blend_with_white(color, 0.52),
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        ax_error.bar(
            index + width / 1.6,
            rmse,
            width,
            color=blend_with_white(color, 0.04),
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        ax_error.text(
            index - width / 1.6,
            mae + 0.00065,
            f"{mae:.4f}",
            ha="center",
            fontsize=24,
        )
        ax_error.text(
            index + width / 1.6,
            rmse + 0.00065,
            f"{rmse:.4f}",
            ha="center",
            fontsize=24,
        )

        size = float(models.loc[model, "fp32_weights_mib"])
        ax_size.bar(
            index,
            size,
            0.52,
            color=blend_with_white(color, 0.28),
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        ax_size.text(index, size + 0.11, f"{size:.3f}", ha="center", fontsize=24)

    max_error = max(float(metrics["rmse"].max()), float(metrics["mae"].max()))
    error_limit = np.ceil((max_error + 0.002) / 0.005) * 0.005
    ax_error.set_ylim(0.0, error_limit)
    ax_error.set_yticks(np.arange(0.0, error_limit + 0.0001, 0.005))
    ax_error.set_ylabel("SOH error [0-1]", fontsize=30)
    ax_error.set_xlabel("Architecture", fontsize=30, labelpad=24)
    ax_error.set_xticks(
        positions,
        [MODEL_LABELS[name] for name in MODEL_ORDER],
        fontweight="bold",
        fontsize=25,
    )
    ax_error.tick_params(axis="y", labelsize=25)
    ax_error.legend(
        handles=[
            Patch(facecolor="#C6C6C6", edgecolor="#777777", label="MAE"),
            Patch(facecolor="#777777", edgecolor="#777777", label="RMSE"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.56, 1.13),
        ncol=2,
        frameon=False,
        fontsize=31,
    )

    ax_size.set_ylim(0.0, 4.0)
    ax_size.set_yticks(np.arange(0, 4.1, 1.0))
    ax_size.set_ylabel("FP32 weights [MiB]", fontsize=30)
    ax_size.set_xlabel("Architecture", fontsize=30, labelpad=24)
    ax_size.set_xticks(
        positions,
        [MODEL_LABELS[name] for name in MODEL_ORDER],
        fontweight="bold",
        fontsize=25,
    )
    ax_size.tick_params(axis="y", labelsize=25)

    for label, ax in zip(("(a)", "(b)"), (ax_error, ax_size)):
        ax.text(-0.055, 1.045, label, transform=ax.transAxes, fontsize=17)
        ax.grid(axis="y", color="#D5D8DC", linestyle=(0, (4, 3)), linewidth=1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_figure(fig, output_dir / "baseline_model_comparison", dpi=120)


def load_c11_predictions(result_dir: Path) -> dict[str, pd.DataFrame]:
    consolidated_path = result_dir / "selected_baseline_soh_trajectory_data.csv"
    if consolidated_path.is_file():
        wide = pd.read_csv(consolidated_path)
        frames = {}
        for model in MODEL_ORDER:
            reference_column = f"soh_reference_{model}"
            prediction_column = f"soh_{model}"
            frame = wide[
                ["sample_index", "time_h", reference_column, prediction_column]
            ].dropna().rename(
                columns={
                    reference_column: "soh_reference",
                    prediction_column: "soh_prediction",
                }
            )
            frame["signed_error"] = frame["soh_prediction"] - frame["soh_reference"]
            frame["absolute_error"] = frame["signed_error"].abs()
            frames[model] = frame
        return frames

    frames = {}
    for model in MODEL_ORDER:
        path = result_dir / "predictions" / f"{model}_C11_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames[model] = pd.read_csv(path)
    return frames


def render_selected_trajectory(
    prediction_frames: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    wide = None
    for model in MODEL_ORDER:
        frame = prediction_frames[model][
            ["sample_index", "time_h", "soh_reference", "soh_prediction"]
        ].rename(
            columns={
                "soh_reference": f"soh_reference_{model}",
                "soh_prediction": f"soh_{model}",
            }
        )
        wide = frame if wide is None else wide.merge(frame, on=["sample_index", "time_h"], how="outer")
    wide = wide.sort_values(["time_h", "sample_index"])
    wide.to_csv(output_dir / "selected_baseline_soh_trajectory_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(20, 10), dpi=120)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.955, bottom=0.14)
    for model in MODEL_ORDER:
        frame = prediction_frames[model].copy()
        frame["soh_prediction"] = frame["soh_prediction"].rolling(
            window=7, center=True, min_periods=1
        ).median()
        frame = frame.iloc[::3]
        ax.plot(
            frame["time_h"],
            frame["soh_prediction"],
            color=COLORS[model],
            linewidth=2.2,
            label=MODEL_LABELS[model],
            zorder=3,
        )

    reference = prediction_frames["lstm"]
    ax.plot(
        reference["time_h"],
        reference["soh_reference"],
        color="#191919",
        linewidth=3.0,
        label="Reference SOH",
        zorder=4,
    )
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Reference SOH")] + [labels.index(MODEL_LABELS[name]) for name in MODEL_ORDER]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#AFAFAF",
        fontsize=31,
    )
    ax.set_xlim(0, float(reference["time_h"].max()))
    ax.set_ylim(0.62, 1.00)
    ax.set_xlabel("Time [h]", fontsize=30, labelpad=17)
    ax.set_ylabel("SOH [0-1]", fontsize=30)
    ax.tick_params(axis="both", labelsize=26)
    ax.grid(color="#DADDE0", linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_dir / "selected_baseline_soh_trajectory", dpi=120)


def render_capacity_sensitivity(
    prediction_frames: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    batmm_c11_mae = {
        model: float(prediction_frames[model]["absolute_error"].mean())
        for model in MODEL_ORDER
    }
    rows = []
    for model in MODEL_ORDER:
        selected_tag = BATMM_SWEEP_POINT[model]
        for tag, parameters, original_mae in CAPACITY_SWEEP[model]:
            is_batmm = tag == selected_tag
            rows.append(
                {
                    "architecture": MODEL_LABELS[model],
                    "variant": tag,
                    "parameters": parameters,
                    "mae": batmm_c11_mae[model] if is_batmm else original_mae,
                    "source": "BATMM C11 reevaluation" if is_batmm else "original C11 capacity sweep",
                }
            )
    capacity = pd.DataFrame(rows)
    capacity.to_csv(output_dir / "baseline_capacity_sensitivity_data.csv", index=False)

    panel_models = ["cnn", "gru", "lstm", "tcn"]
    for model in panel_models:
        family = capacity[capacity["architecture"] == MODEL_LABELS[model]].sort_values("parameters")
        fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
        fig.subplots_adjust(left=0.16, right=0.975, top=0.95, bottom=0.18)
        ax.plot(
            family["parameters"],
            family["mae"],
            color=COLORS[model],
            marker="o",
            markersize=12,
            linewidth=2.8,
            zorder=3,
        )
        for row in family.itertuples(index=False):
            ax.annotate(
                row.variant,
                (row.parameters, row.mae),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=17,
            )
        ax.set_xlabel("Parameters", labelpad=12)
        ax.set_ylabel("MAE", labelpad=10)
        ax.grid(color="#C8C8C8", alpha=0.45, linewidth=1.1)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
        ax.margins(x=0.09, y=0.14)
        save_figure(
            fig,
            output_dir / f"baseline_capacity_sensitivity_{model}",
            dpi=200,
        )


def main() -> None:
    args = parse_args()
    result_dir = find_result_dir(args.result_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reported_metrics = load_batmm_window_metrics(result_dir)
    inventory = pd.read_csv(result_dir / "model_inventory.csv")
    predictions = load_c11_predictions(result_dir)

    configure_style()
    render_model_comparison(reported_metrics, inventory, output_dir)
    render_selected_trajectory(predictions, output_dir)
    render_capacity_sensitivity(predictions, output_dir)
    print(f"Rendered BATMM paper figures in {output_dir}")


if __name__ == "__main__":
    main()
