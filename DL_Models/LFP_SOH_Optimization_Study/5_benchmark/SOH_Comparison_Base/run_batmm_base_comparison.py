#!/usr/bin/env python3
"""Reproduce the BATMM base-model comparison without modifying BATMM.

The script uses the checkpoints, scalers, model definitions, test-cell split,
feature construction, and inference protocol delivered in BATMM_for_Florian.
All generated data and figures are written to a new timestamped result folder.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import types
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parents[1]
BATMM_ROOT = HERE.parent / "BATMM_for_Florian" / "BATMM_for_Florian"
RESULTS_ROOT = HERE / "results"
CURRENT_OVERVIEW = (
    STUDY_ROOT
    / "6_test"
    / "CURRENT_MODELS_BASE_VS_OPTIMIZED"
    / "CURRENT_MODELS_OVERVIEW.md"
)

MODEL_ORDER = ("cnn", "gru", "lstm", "tcn")
MODEL_LABELS = {name: name.upper() for name in MODEL_ORDER}
COLORS = {
    "cnn": "#59C7C2",
    "gru": "#59E83A",
    "lstm": "#E76B91",
    "tcn": "#294862",
}
REFERENCE_COLOR = "#191919"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the four BATMM base models and create paper-style plots."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing df_FE_C11.parquet, df_FE_C23.parquet, and df_FE_C29.parquet.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    parser.add_argument(
        "--rnn-chunk",
        type=int,
        default=256,
        help="Number of hourly samples processed per recurrent call while carrying state.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write numerical results only. Use render_batmm_results.py afterwards.",
    )
    return parser.parse_args()


def resolve_data_root(requested: Path | None) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested)
    if os.getenv("MGFARM_FE_DATA_ROOT"):
        candidates.append(Path(os.environ["MGFARM_FE_DATA_ROOT"]))
    candidates.extend(
        [
            Path.home()
            / "SynologyDrive"
            / "TUB"
            / "3_Projekte"
            / "MG_Farm"
            / "5_Data"
            / "01_LFP"
            / "00_Data"
            / "Versuch_18650_standart"
            / "MGFarm_18650_FE",
            Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"),
        ]
    )
    required = ("df_FE_C11.parquet", "df_FE_C23.parquet", "df_FE_C29.parquet")
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if all((candidate / name).is_file() for name in required):
            return candidate
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not locate all BATMM test cells. Checked:\n{checked}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def install_tensorflow_import_stub() -> None:
    """Allow importing BATMM's PyTorch classes when TensorFlow is unavailable.

    BATMM keeps PyTorch and TensorFlow conversion helpers in the same modules.
    The comparison only calls the PyTorch constructors, so a minimal type stub
    is sufficient and avoids adding TensorFlow as an unrelated dependency.
    """

    if importlib.util.find_spec("tensorflow") is not None:
        return

    tensorflow = types.ModuleType("tensorflow")
    keras = types.ModuleType("tensorflow.keras")
    layers = types.ModuleType("tensorflow.keras.layers")

    class Layer:
        pass

    class Model:
        pass

    layers.Layer = Layer
    keras.layers = layers
    keras.Model = Model
    tensorflow.keras = keras
    sys.modules["tensorflow"] = tensorflow
    sys.modules["tensorflow.keras"] = keras
    sys.modules["tensorflow.keras.layers"] = layers


def import_batmm():
    if not BATMM_ROOT.is_dir():
        raise FileNotFoundError(f"BATMM folder is missing: {BATMM_ROOT}")
    install_tensorflow_import_stub()
    sys.path.insert(0, str(BATMM_ROOT))
    config = importlib.import_module("config")
    data_utils = importlib.import_module("src.utils.data_utils")
    modules = {
        name: importlib.import_module(f"src.utils.{name}_utils")
        for name in MODEL_ORDER
    }
    return config, data_utils, modules


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hourly_cells(data_root: Path, test_cells: list[str], data_utils) -> dict[str, pd.DataFrame]:
    cells = {}
    required = data_utils.BASE_FEATURES + [data_utils.TARGET, "Testtime[s]"]
    for cell in test_cells:
        path = data_root / f"{cell}.parquet"
        print(f"Aggregating {cell} from {path}")
        raw = pd.read_parquet(path, columns=required)
        hourly = data_utils.aggregate_hourly(raw)
        del raw
        gc.collect()
        hourly = hourly.replace([np.inf, -np.inf], np.nan).dropna(
            subset=data_utils.FEATURES + [data_utils.TARGET]
        )
        cells[cell] = hourly.reset_index(drop=True)
        print(f"  {len(hourly)} hourly samples")
    return cells


def load_model(model_name: str, module, device: torch.device):
    create_fn = getattr(module, f"create_{model_name}")
    model = create_fn()
    checkpoint_path = BATMM_ROOT / "models" / model_name.upper() / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), checkpoint_path


@torch.inference_mode()
def predict_stateful(
    model: torch.nn.Module,
    x_scaled: np.ndarray,
    device: torch.device,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    state = None
    for start in range(0, len(x_scaled), chunk):
        stop = min(start + chunk, len(x_scaled))
        inputs = torch.from_numpy(x_scaled[start:stop]).unsqueeze(0).to(device)
        output, state = model(inputs, state=state, return_state=True)
        if isinstance(state, tuple):
            state = tuple(item.detach() for item in state)
        else:
            state = state.detach()
        predictions.append(output.squeeze(0).detach().cpu().numpy())
    return np.concatenate(predictions), np.arange(len(x_scaled), dtype=np.int64)


@torch.inference_mode()
def predict_batmm_windows(
    model: torch.nn.Module,
    x_scaled: np.ndarray,
    device: torch.device,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match BATMM SeqDataset: non-overlapping windows and no partial tail."""

    complete_length = (len(x_scaled) // chunk) * chunk
    predictions = []
    for start in range(0, complete_length, chunk):
        inputs = torch.from_numpy(x_scaled[start : start + chunk]).unsqueeze(0).to(device)
        output = model(inputs)
        predictions.append(output.squeeze(0).detach().cpu().numpy())
    if not predictions:
        raise ValueError(f"Input contains fewer than {chunk} samples.")
    return np.concatenate(predictions), np.arange(complete_length, dtype=np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    residual = y_pred - y_true
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    total = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - np.sum(np.square(residual)) / total) if total > 0.0 else float("nan")
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "max_error": float(np.max(np.abs(residual))),
    }


def summarize_metrics(per_cell: pd.DataFrame, prediction_frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        model_rows = per_cell[per_cell["model"] == model_name]
        macro = {
            metric: float(model_rows[metric].mean())
            for metric in ("mse", "mae", "rmse", "r2")
        }
        macro["max_error"] = float(model_rows["max_error"].max())
        rows.append({"model": model_name, "aggregation": "cell_macro", **macro})

        frames = [prediction_frames[(model_name, cell)] for cell in model_rows["cell"]]
        y_true = np.concatenate([frame["soh_reference"].to_numpy() for frame in frames])
        y_pred = np.concatenate([frame["soh_prediction"].to_numpy() for frame in frames])
        rows.append(
            {
                "model": model_name,
                "aggregation": "sample_weighted",
                **compute_metrics(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def read_reference_metrics() -> pd.DataFrame:
    if not CURRENT_OVERVIEW.is_file():
        return pd.DataFrame()
    pattern = re.compile(
        r"^\|\s*(CNN|GRU|LSTM|TCN)\s*\|\s*([^|]+)\|\s*([0-9.]+)\s*\|[^|]+\|[^|]+\|\s*([0-9.]+)\s*\|"
    )
    rows = []
    for line in CURRENT_OVERVIEW.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            rows.append(
                {
                    "model": match.group(1).lower(),
                    "reference_version": match.group(2).strip(),
                    "reference_mae": float(match.group(3)),
                    "reference_rmse": float(match.group(4)),
                }
            )
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax) -> None:
    ax.grid(True, color="#D7DBDE", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_baseline_summary(summary: pd.DataFrame, inventory: pd.DataFrame, output: Path) -> None:
    selected = summary[summary["aggregation"] == "cell_macro"].set_index("model").loc[list(MODEL_ORDER)]
    sizes = inventory.set_index("model").loc[list(MODEL_ORDER), "fp32_weights_mib"]
    x = np.arange(len(MODEL_ORDER))
    width = 0.27
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax = axes[0]
    for index, model_name in enumerate(MODEL_ORDER):
        color = COLORS[model_name]
        mae = selected.loc[model_name, "mae"]
        rmse = selected.loc[model_name, "rmse"]
        ax.bar(index - width / 2, mae, width, color=color, edgecolor=color, alpha=0.48, linewidth=1.4)
        ax.bar(index + width / 2, rmse, width, color=color, edgecolor=color, alpha=0.95, linewidth=1.4)
        ax.text(index - width / 2, mae + 0.00045, f"{mae:.4f}", ha="center", va="bottom", fontsize=9)
        ax.text(index + width / 2, rmse + 0.00045, f"{rmse:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [MODEL_LABELS[name] for name in MODEL_ORDER], fontweight="bold")
    ax.set_xlabel("Architecture")
    ax.set_ylabel("SOH error [0-1]")
    ax.set_ylim(0.0, max(selected["rmse"].max() * 1.28, 0.02))
    legend_handles = [
        Patch(facecolor="#C7C7C7", edgecolor="#777777", label="MAE"),
        Patch(facecolor="#666666", edgecolor="#555555", label="RMSE"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    ax.text(-0.16, 1.06, "(a)", transform=ax.transAxes, fontsize=15, fontweight="bold")
    style_axis(ax)

    ax = axes[1]
    for index, model_name in enumerate(MODEL_ORDER):
        value = sizes.loc[model_name]
        ax.bar(index, value, width=0.54, color=COLORS[model_name], edgecolor=COLORS[model_name], alpha=0.72, linewidth=1.4)
        ax.text(index, value + 0.08, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [MODEL_LABELS[name] for name in MODEL_ORDER], fontweight="bold")
    ax.set_xlabel("Architecture")
    ax.set_ylabel("FP32 weights [MiB]")
    ax.set_ylim(0.0, max(sizes.max() * 1.22, 4.0))
    ax.text(-0.16, 1.06, "(b)", transform=ax.transAxes, fontsize=15, fontweight="bold")
    style_axis(ax)

    fig.tight_layout(w_pad=3.0)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_cell_trajectories(
    cells: list[str],
    prediction_frames: dict[tuple[str, str], pd.DataFrame],
    output: Path,
) -> None:
    fig, axes = plt.subplots(len(cells), 1, figsize=(13, 9.2), sharex=False)
    for panel, (ax, cell) in enumerate(zip(axes, cells)):
        reference = prediction_frames[("lstm", cell)]
        ax.plot(
            reference["time_h"],
            reference["soh_reference"],
            color=REFERENCE_COLOR,
            linewidth=2.2,
            label="Reference SOH",
            zorder=5,
        )
        for model_name in MODEL_ORDER:
            frame = prediction_frames[(model_name, cell)]
            ax.plot(
                frame["time_h"],
                frame["soh_prediction"],
                color=COLORS[model_name],
                linewidth=1.25,
                alpha=0.95,
                label=MODEL_LABELS[model_name],
            )
        ax.set_ylabel("SOH [0-1]")
        ax.set_title(cell.replace("df_FE_", "Cell "), loc="left", fontweight="bold")
        ax.text(-0.07, 1.02, f"({chr(97 + panel)})", transform=ax.transAxes, fontsize=13, fontweight="bold")
        style_axis(ax)
    axes[-1].set_xlabel("Time [h]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.6)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_single_cell_trajectory(
    cell: str,
    prediction_frames: dict[tuple[str, str], pd.DataFrame],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    reference = prediction_frames[("lstm", cell)]
    ax.plot(reference["time_h"], reference["soh_reference"], color=REFERENCE_COLOR, linewidth=2.5, label="Reference SOH", zorder=5)
    for model_name in MODEL_ORDER:
        frame = prediction_frames[(model_name, cell)]
        ax.plot(frame["time_h"], frame["soh_prediction"], color=COLORS[model_name], linewidth=1.35, label=MODEL_LABELS[model_name])
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("SOH [0-1]")
    ax.legend(loc="lower left", frameon=True, framealpha=0.94)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_per_cell_errors(per_cell: pd.DataFrame, output: Path) -> None:
    cells = list(dict.fromkeys(per_cell["cell"]))
    x = np.arange(len(cells))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    for ax, metric, panel in zip(axes, ("mae", "rmse"), ("a", "b")):
        for index, model_name in enumerate(MODEL_ORDER):
            model_rows = per_cell[per_cell["model"] == model_name].set_index("cell").loc[cells]
            positions = x + (index - 1.5) * width
            ax.bar(
                positions,
                model_rows[metric],
                width,
                color=COLORS[model_name],
                edgecolor=COLORS[model_name],
                alpha=0.68,
                linewidth=1.2,
                label=MODEL_LABELS[model_name],
            )
        ax.set_xticks(x, [cell.replace("df_FE_", "") for cell in cells], fontweight="bold")
        ax.set_xlabel("Held-out test cell")
        ax.set_ylabel(f"{metric.upper()} [0-1]")
        ax.text(-0.13, 1.04, f"({panel})", transform=ax.transAxes, fontsize=14, fontweight="bold")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.5)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_aggregation_comparison(summary: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(MODEL_ORDER))
    width = 0.28
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    for ax, metric, panel in zip(axes, ("mae", "rmse"), ("a", "b")):
        macro = summary[summary["aggregation"] == "cell_macro"].set_index("model").loc[list(MODEL_ORDER), metric]
        weighted = summary[summary["aggregation"] == "sample_weighted"].set_index("model").loc[list(MODEL_ORDER), metric]
        ax.bar(x - width / 2, macro, width, color=[COLORS[name] for name in MODEL_ORDER], edgecolor=[COLORS[name] for name in MODEL_ORDER], alpha=0.45, linewidth=1.3, label="Cell-macro")
        ax.bar(x + width / 2, weighted, width, color=[COLORS[name] for name in MODEL_ORDER], edgecolor=[COLORS[name] for name in MODEL_ORDER], alpha=0.95, linewidth=1.3, label="Sample-weighted")
        ax.set_xticks(x, [MODEL_LABELS[name] for name in MODEL_ORDER], fontweight="bold")
        ax.set_xlabel("Architecture")
        ax.set_ylabel(f"{metric.upper()} [0-1]")
        ax.text(-0.13, 1.04, f"({panel})", transform=ax.transAxes, fontsize=14, fontweight="bold")
        style_axis(ax)
    axes[0].legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(w_pad=2.5)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def write_audit(out_dir: Path, config, model_inventory: pd.DataFrame) -> None:
    audit = {
        "purpose": "Reproduce BATMM_for_Florian exactly enough to explain the differing base-model metrics.",
        "batmm_protocol": {
            "test_cells": list(config.TEST_CELLS),
            "hourly_features": "20 values: five base quantities times mean, standard deviation, minimum, and maximum",
            "target_aggregation_used_by_batmm_data_utils": "last",
            "recurrent_inference": "State is carried causally through each complete test-cell trajectory.",
            "convolutional_inference": "Non-overlapping windows are evaluated independently; incomplete final windows are omitted.",
            "reported_batmm_average": "Cell-macro mean: each test cell contributes equally.",
        },
        "configuration_findings": [
            "BATMM config.py uses CNN/TCN window lengths 128/96, while their YAML files state 96/120.",
            "BATMM data_utils.py uses target aggregation 'last' for every architecture, while the TCN YAML states 'mean'.",
            "BATMM config.py lists C15 as validation cell, while the model YAML files list C15 and C21.",
            "These differences are documented here but intentionally not corrected in this reproduction.",
        ],
        "models": model_inventory.to_dict(orient="records"),
    }
    (out_dir / "BATMM_PROTOCOL_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def write_results_summary(
    out_dir: Path,
    summary: pd.DataFrame,
    per_cell: pd.DataFrame,
) -> None:
    macro = summary[summary["aggregation"] == "cell_macro"].set_index("model")
    weighted = summary[summary["aggregation"] == "sample_weighted"].set_index("model")
    lines = [
        "# BATMM Base Model Comparison",
        "",
        "This run uses the BATMM_for_Florian checkpoints, scalers, feature construction, test-cell split, and inference behavior without changing the BATMM source files.",
        "",
        "## Cell-macro results",
        "",
        "| Model | MAE | RMSE | R2 |",
        "|---|---:|---:|---:|",
    ]
    for model_name in MODEL_ORDER:
        row = macro.loc[model_name]
        lines.append(
            f"| {MODEL_LABELS[model_name]} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['r2']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Sample-weighted results",
            "",
            "| Model | MAE | Global RMSE | R2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for model_name in MODEL_ORDER:
        row = weighted.loc[model_name]
        lines.append(
            f"| {MODEL_LABELS[model_name]} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['r2']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation of the differences",
            "",
            "- BATMM reports a cell-macro average, so C11, C23, and C29 contribute equally despite their different trajectory lengths.",
            "- The recurrent GRU and LSTM keep their hidden state throughout each cell. Their sample-weighted MAE values closely reproduce the previous comparison.",
            "- BATMM evaluates CNN and TCN in independent, non-overlapping windows. Each new window therefore starts without preceding context, which creates repeated boundary transients and particularly affects RMSE.",
            "- The TCN YAML declares a mean SOH target per hour, while BATMM data_utils.py applies the last SOH value to every model. This target mismatch additionally changes the TCN result.",
            "- The model footprints are unchanged. The differences originate from target construction, window handling, and metric aggregation rather than from different parameter counts.",
            "",
            "See `BATMM_PROTOCOL_AUDIT.json` for the exact configuration findings and `metrics_by_cell.csv` for all cell-level values.",
            "",
            "## Evaluated samples",
            "",
            "| Model | Cell | Source | Evaluated | Omitted tail |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in per_cell.iterrows():
        lines.append(
            f"| {MODEL_LABELS[row['model']]} | {row['cell'].replace('df_FE_', '')} | "
            f"{int(row['source_samples'])} | {int(row['evaluated_samples'])} | {int(row['omitted_tail_samples'])} |"
        )
    (out_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir or RESULTS_ROOT / f"BATMM_RESULTS_{timestamp}").resolve()
    plots_dir = out_dir / "plots"
    predictions_dir = out_dir / "predictions"
    plots_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir.mkdir(parents=True, exist_ok=False)

    print(f"BATMM root: {BATMM_ROOT}")
    print(f"Data root:  {data_root}")
    print(f"Device:     {device}")
    print(f"Output:     {out_dir}")

    previous_cwd = Path.cwd()
    os.chdir(BATMM_ROOT)
    try:
        config, data_utils, modules = import_batmm()
        hourly_cells = load_hourly_cells(data_root, list(config.TEST_CELLS), data_utils)
        per_cell_rows = []
        inventory_rows = []
        prediction_frames: dict[tuple[str, str], pd.DataFrame] = {}

        for model_name in MODEL_ORDER:
            print(f"\nEvaluating {MODEL_LABELS[model_name]}")
            module = modules[model_name]
            model, checkpoint_path = load_model(model_name, module, device)
            model_dir = checkpoint_path.parent
            scaler_path = model_dir / "scaler_robust.joblib"
            yaml_path = model_dir / "train_soh.yaml"
            scaler = joblib.load(scaler_path)
            with yaml_path.open("r", encoding="utf-8") as handle:
                yaml_config = yaml.safe_load(handle)

            batmm_chunk = int(config.CHUNK_SIZES[model_name])
            yaml_chunk = int(yaml_config["training"]["seq_chunk_size"])
            parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
            buffer_bytes = int(sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()))
            fp32_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())) + buffer_bytes
            inventory_rows.append(
                {
                    "model": model_name,
                    "yaml_model_type": yaml_config["model"]["type"],
                    "batmm_chunk_size": batmm_chunk,
                    "yaml_chunk_size": yaml_chunk,
                    "yaml_target_aggregation": yaml_config["sampling"]["target_agg"],
                    "parameters": parameter_count,
                    "fp32_weights_mib": fp32_bytes / (1024**2),
                    "checkpoint_bytes": checkpoint_path.stat().st_size,
                    "checkpoint_sha256": sha256(checkpoint_path),
                    "scaler_sha256": sha256(scaler_path),
                }
            )

            for cell, hourly in hourly_cells.items():
                x_scaled = scaler.transform(
                    hourly[data_utils.FEATURES].to_numpy(dtype=np.float32)
                ).astype(np.float32)
                if model_name in ("gru", "lstm"):
                    y_pred, indices = predict_stateful(model, x_scaled, device, args.rnn_chunk)
                    inference_mode = "causal_stateful"
                else:
                    y_pred, indices = predict_batmm_windows(model, x_scaled, device, batmm_chunk)
                    inference_mode = "nonoverlapping_batmm_windows"
                y_true = hourly[data_utils.TARGET].to_numpy(dtype=np.float32)[indices]
                metrics = compute_metrics(y_true, y_pred)
                row = {
                    "model": model_name,
                    "cell": cell,
                    "inference_mode": inference_mode,
                    "source_samples": len(hourly),
                    "evaluated_samples": len(indices),
                    "omitted_tail_samples": len(hourly) - len(indices),
                    **metrics,
                }
                per_cell_rows.append(row)
                print(
                    f"  {cell}: n={len(indices)}, MAE={metrics['mae']:.6f}, "
                    f"RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.6f}"
                )

                prediction_frame = pd.DataFrame(
                    {
                        "sample_index": indices,
                        "time_h": indices.astype(np.float64),
                        "soh_reference": y_true,
                        "soh_prediction": y_pred,
                        "signed_error": y_pred - y_true,
                        "absolute_error": np.abs(y_pred - y_true),
                    }
                )
                prediction_frames[(model_name, cell)] = prediction_frame
                short_cell = cell.replace("df_FE_", "")
                prediction_frame.to_csv(
                    predictions_dir / f"{model_name}_{short_cell}_predictions.csv",
                    index=False,
                )

            del model, scaler
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        per_cell = pd.DataFrame(per_cell_rows)
        inventory = pd.DataFrame(inventory_rows)
        summary = summarize_metrics(per_cell, prediction_frames)
        per_cell.to_csv(out_dir / "metrics_by_cell.csv", index=False)
        summary.to_csv(out_dir / "metrics_summary.csv", index=False)
        inventory.to_csv(out_dir / "model_inventory.csv", index=False)
        write_results_summary(out_dir, summary, per_cell)

        reference = read_reference_metrics()
        if not reference.empty:
            comparison = summary.merge(reference, on="model", how="left")
            comparison["mae_delta_vs_current"] = comparison["mae"] - comparison["reference_mae"]
            comparison["rmse_delta_vs_current"] = comparison["rmse"] - comparison["reference_rmse"]
            comparison.to_csv(out_dir / "comparison_to_current_reference.csv", index=False)

        write_audit(out_dir, config, inventory)

        metadata = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "script": str(Path(__file__).resolve()),
            "batmm_root": str(BATMM_ROOT),
            "data_root": str(data_root),
            "device": str(device),
            "python": sys.version,
            "torch": torch.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "test_cells": list(config.TEST_CELLS),
            "rnn_processing_chunk": args.rnn_chunk,
            "plots_rendered_during_run": not args.skip_plots,
            "note": "The RNN processing chunk changes execution granularity only; hidden state is carried across chunks.",
        }
        (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if not args.skip_plots:
            configure_plot_style()
            plot_baseline_summary(summary, inventory, plots_dir / "batmm_baseline_model_comparison.png")
            plot_cell_trajectories(list(config.TEST_CELLS), prediction_frames, plots_dir / "batmm_all_test_cell_trajectories.png")
            plot_single_cell_trajectory(config.TEST_CELLS[0], prediction_frames, plots_dir / "batmm_C11_soh_trajectory.png")
            plot_per_cell_errors(per_cell, plots_dir / "batmm_per_cell_errors.png")
            plot_aggregation_comparison(summary, plots_dir / "batmm_macro_vs_sample_weighted.png")
        print(f"\nCompleted. Results saved to {out_dir}")
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=UserWarning)
        main()
