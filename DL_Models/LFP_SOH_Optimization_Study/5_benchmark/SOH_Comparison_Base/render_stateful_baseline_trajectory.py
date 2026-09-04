#!/usr/bin/env python3
"""Render the C11 baseline trajectory with continuous model context.

The recurrent models carry their hidden state. CNN and TCN retain the raw input
history required by their causal receptive field, so processing in chunks is
numerically equivalent to one uninterrupted causal sequence.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

import render_batmm_paper_figures as paper_figures
import run_batmm_base_comparison as batmm


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "results" / "FINAL_RESULTS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    return parser.parse_args()


def convolution_receptive_field(model: torch.nn.Module) -> int:
    declared = getattr(model, "receptive_field", None)
    if declared is not None:
        return int(declared)

    kernel_size = int(model.kernel_size)
    dilations = [int(value) for value in model.dilations]
    output_kernel_size = int(getattr(model, "output_kernel_size", 1))
    return 1 + 2 * (kernel_size - 1) * sum(dilations) + output_kernel_size - 1


@torch.inference_mode()
def predict_convolution_continuously(
    model: torch.nn.Module,
    x_scaled: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    receptive_field = convolution_receptive_field(model)
    history_length = receptive_field - 1
    history: np.ndarray | None = None
    predictions = []

    for start in range(0, len(x_scaled), chunk_size):
        stop = min(start + chunk_size, len(x_scaled))
        current = x_scaled[start:stop]
        model_input = current if history is None else np.concatenate((history, current), axis=0)

        inputs = torch.from_numpy(model_input).unsqueeze(0).to(device)
        output = model(inputs).squeeze(0).detach().cpu().numpy()
        predictions.append(output[-len(current) :])

        if history_length > 0:
            history = model_input[-history_length:].copy()

    values = np.concatenate(predictions)
    indices = np.arange(len(values), dtype=np.int64)
    return values, indices, receptive_field


def build_prediction_frame(
    hourly: pd.DataFrame,
    data_utils,
    prediction: np.ndarray,
    indices: np.ndarray,
) -> pd.DataFrame:
    reference = hourly[data_utils.TARGET].to_numpy(dtype=np.float32)[indices]
    return pd.DataFrame(
        {
            "sample_index": indices,
            "time_h": indices.astype(np.float64),
            "soh_reference": reference,
            "soh_prediction": prediction,
            "signed_error": prediction - reference,
            "absolute_error": np.abs(prediction - reference),
        }
    )


def load_wide_prediction_frames(path: Path) -> dict[str, pd.DataFrame]:
    wide = pd.read_csv(path)
    frames = {}
    for model_name in paper_figures.MODEL_ORDER:
        reference_column = f"soh_reference_{model_name}"
        prediction_column = f"soh_{model_name}"
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
        frames[model_name] = frame
    return frames


def draw_trajectory(
    frames: dict[str, pd.DataFrame], output_dir: Path, output_name: str
) -> None:
    fig, ax = plt.subplots(figsize=(20, 10), dpi=120)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.955, bottom=0.14)
    for model_name in paper_figures.MODEL_ORDER:
        frame = frames[model_name]
        ax.plot(
            frame["time_h"],
            frame["soh_prediction"],
            color=paper_figures.COLORS[model_name],
            linewidth=2.2,
            label=paper_figures.MODEL_LABELS[model_name],
            zorder=3,
        )

    reference = frames["lstm"]
    ax.plot(
        reference["time_h"],
        reference["soh_reference"],
        color="#191919",
        linewidth=3.0,
        label="Reference SOH",
        zorder=4,
    )
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Reference SOH")] + [
        labels.index(paper_figures.MODEL_LABELS[name])
        for name in paper_figures.MODEL_ORDER
    ]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#AFAFAF",
    )
    ax.set_xlim(0, float(reference["time_h"].max()))
    ax.set_ylim(0.62, 1.00)
    ax.set_xlabel("Time [h]", labelpad=17)
    ax.set_ylabel("SOH [0-1]")
    ax.grid(color="#DADDE0", linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    paper_figures.save_figure(fig, output_dir / output_name, dpi=120)


def render_trajectory(
    frames: dict[str, pd.DataFrame], output_dir: Path
) -> dict[str, pd.DataFrame]:
    wide = None
    for model_name in paper_figures.MODEL_ORDER:
        frame = frames[model_name][
            ["sample_index", "time_h", "soh_reference", "soh_prediction"]
        ].rename(
            columns={
                "soh_reference": f"soh_reference_{model_name}",
                "soh_prediction": f"soh_{model_name}",
            }
        )
        wide = frame if wide is None else wide.merge(
            frame, on=["sample_index", "time_h"], how="outer"
        )

    wide = wide.sort_values(["time_h", "sample_index"])
    wide.to_csv(
        output_dir / "selected_baseline_soh_trajectory_stateful_data.csv", index=False
    )

    draw_trajectory(
        frames,
        output_dir,
        "selected_baseline_soh_trajectory_stateful_raw",
    )

    display_frames = {}
    for model_name in paper_figures.MODEL_ORDER:
        display = frames[model_name].copy()
        display["soh_prediction"] = display["soh_prediction"].rolling(
            window=7, center=True, min_periods=1
        ).median()
        display_frames[model_name] = display.iloc[::3].reset_index(drop=True)

    display_wide = None
    for model_name in paper_figures.MODEL_ORDER:
        display = display_frames[model_name][
            ["sample_index", "time_h", "soh_reference", "soh_prediction"]
        ].rename(
            columns={
                "soh_reference": f"soh_reference_{model_name}",
                "soh_prediction": f"soh_{model_name}",
            }
        )
        display_wide = display if display_wide is None else display_wide.merge(
            display, on=["sample_index", "time_h"], how="outer"
        )
    display_wide.to_csv(
        output_dir / "selected_baseline_soh_trajectory_stateful_display_data.csv",
        index=False,
    )
    draw_trajectory(
        display_frames,
        output_dir,
        "selected_baseline_soh_trajectory_stateful",
    )
    return display_frames


def write_variant_comparison(
    windowed_frames: dict[str, pd.DataFrame],
    continuous_frames: dict[str, pd.DataFrame],
    display_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    rows = []
    variants = (
        (
            "01_batmm_windowed",
            windowed_frames,
            "independent CNN/TCN windows; continuous GRU/LSTM state",
            "none",
            True,
        ),
        (
            "02_continuous_context",
            continuous_frames,
            "continuous context for every architecture",
            "none",
            True,
        ),
        (
            "03_continuous_display_reduced",
            display_frames,
            "same inference as 02_continuous_context",
            "centered 7-hour rolling median; every third point drawn",
            False,
        ),
    )

    continuous_metrics = {}
    for model_name, frame in continuous_frames.items():
        continuous_metrics[model_name] = batmm.compute_metrics(
            frame["soh_reference"].to_numpy(), frame["soh_prediction"].to_numpy()
        )

    for variant, frames, inference, processing, independent_run in variants:
        for model_name in paper_figures.MODEL_ORDER:
            frame = frames[model_name]
            displayed_metrics = batmm.compute_metrics(
                frame["soh_reference"].to_numpy(),
                frame["soh_prediction"].to_numpy(),
            )
            benchmark_metrics = (
                displayed_metrics
                if variant != "03_continuous_display_reduced"
                else continuous_metrics[model_name]
            )
            rows.append(
                {
                    "variant": variant,
                    "model": model_name,
                    "inference_protocol": inference,
                    "plot_processing": processing,
                    "independent_inference_run": independent_run,
                    "benchmark_samples": (
                        len(continuous_frames[model_name])
                        if variant == "03_continuous_display_reduced"
                        else len(frame)
                    ),
                    "drawn_points": len(frame),
                    "benchmark_mae": benchmark_metrics["mae"],
                    "benchmark_rmse": benchmark_metrics["rmse"],
                    "displayed_points_mae_diagnostic_only": displayed_metrics["mae"],
                    "displayed_points_rmse_diagnostic_only": displayed_metrics["rmse"],
                }
            )

    pd.DataFrame(rows).to_csv(
        output_dir / "selected_baseline_soh_trajectory_three_variants_metrics.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = batmm.resolve_data_root(args.data_root)
    device = batmm.resolve_device(args.device)

    config, data_utils, modules = batmm.import_batmm()
    hourly_cells = batmm.load_hourly_cells(
        data_root, list(config.TEST_CELLS), data_utils
    )
    c11_frames: dict[str, pd.DataFrame] = {}
    prediction_frames: dict[tuple[str, str], pd.DataFrame] = {}
    protocol_rows = []

    previous_cwd = Path.cwd()
    os.chdir(batmm.BATMM_ROOT)
    try:
        for model_name in paper_figures.MODEL_ORDER:
            model, _ = batmm.load_model(model_name, modules[model_name], device)
            model_dir = batmm.BATMM_ROOT / "models" / model_name.upper()
            scaler = joblib.load(model_dir / "scaler_robust.joblib")
            with (model_dir / "train_soh.yaml").open("r", encoding="utf-8") as handle:
                model_config = yaml.safe_load(handle)
            chunk_size = int(model_config["training"]["seq_chunk_size"])
            for cell_name, hourly in hourly_cells.items():
                x_scaled = scaler.transform(
                    hourly[data_utils.FEATURES].to_numpy(dtype=np.float32)
                ).astype(np.float32)

                if model_name in ("gru", "lstm"):
                    prediction, indices = batmm.predict_stateful(
                        model, x_scaled, device, chunk_size
                    )
                    receptive_field = None
                    inference_mode = "continuous_recurrent_state"
                else:
                    prediction, indices, receptive_field = predict_convolution_continuously(
                        model, x_scaled, device, chunk_size
                    )
                    inference_mode = "continuous_convolution_context"

                frame = build_prediction_frame(
                    hourly, data_utils, prediction, indices
                )
                prediction_frames[(model_name, cell_name)] = frame
                if cell_name == "df_FE_C11":
                    c11_frames[model_name] = frame
                metrics = batmm.compute_metrics(
                    frame["soh_reference"].to_numpy(),
                    frame["soh_prediction"].to_numpy(),
                )
                protocol_rows.append(
                    {
                        "model": model_name,
                        "cell": cell_name,
                        "inference_mode": inference_mode,
                        "chunk_size": chunk_size,
                        "receptive_field": receptive_field,
                        "context_samples": None if receptive_field is None else receptive_field - 1,
                        "source_samples": len(hourly),
                        "evaluated_samples": len(frame),
                        "omitted_tail_samples": len(hourly) - len(frame),
                        **metrics,
                    }
                )
                print(
                    f"{model_name.upper()} {cell_name}: {inference_mode}, "
                    f"n={len(frame)}, MAE={metrics['mae']:.6f}, "
                    f"RMSE={metrics['rmse']:.6f}"
                )

            del model, scaler
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        os.chdir(previous_cwd)

    per_cell = pd.DataFrame(protocol_rows)
    summary = batmm.summarize_metrics(per_cell, prediction_frames)
    per_cell.to_csv(
        output_dir / "selected_baseline_soh_trajectory_stateful_metrics.csv",
        index=False,
    )
    per_cell.to_csv(output_dir / "metrics_by_cell_continuous.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary_continuous.csv", index=False)

    paper_figures.configure_style()
    display_frames = render_trajectory(c11_frames, output_dir)

    windowed_data = output_dir / "selected_baseline_soh_trajectory_windowed_data.csv"
    canonical_data = output_dir / "selected_baseline_soh_trajectory_data.csv"
    if not windowed_data.is_file():
        shutil.copy2(canonical_data, windowed_data)
    windowed_frames = load_wide_prediction_frames(windowed_data)
    draw_trajectory(
        windowed_frames,
        output_dir,
        "selected_baseline_soh_trajectory_01_batmm_windowed",
    )
    draw_trajectory(
        c11_frames,
        output_dir,
        "selected_baseline_soh_trajectory_02_continuous_context",
    )
    draw_trajectory(
        display_frames,
        output_dir,
        "selected_baseline_soh_trajectory_03_continuous_display_reduced",
    )
    write_variant_comparison(
        windowed_frames, c11_frames, display_frames, output_dir
    )

    for canonical_name, backup_name in (
        ("metrics_by_cell.csv", "metrics_by_cell_windowed.csv"),
        ("metrics_summary.csv", "metrics_summary_windowed.csv"),
    ):
        canonical_path = output_dir / canonical_name
        backup_path = output_dir / backup_name
        if canonical_path.is_file() and not backup_path.is_file():
            shutil.copy2(canonical_path, backup_path)

    per_cell.to_csv(output_dir / "metrics_by_cell.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    shutil.copy2(
        output_dir / "selected_baseline_soh_trajectory_stateful_data.csv",
        canonical_data,
    )
    print(f"Wrote stateful trajectory outputs to {output_dir}")


if __name__ == "__main__":
    main()
