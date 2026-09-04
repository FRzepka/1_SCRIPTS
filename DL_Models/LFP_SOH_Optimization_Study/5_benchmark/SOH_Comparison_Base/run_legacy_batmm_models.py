#!/usr/bin/env python3
"""Evaluate the legacy BATMM SOH checkpoints without modifying their clone."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


HERE = Path(__file__).resolve().parent
LEGACY_STUDY = HERE.parent / "batmm" / "LFP_SOH_Optimization_Study"
RESULTS_ROOT = HERE / "results"

MODEL_SPECS = {
    "cnn": ("CNN", "0.4.1.1"),
    "gru": ("GRU", "0.3.1.1"),
    "lstm": ("LSTM", "0.1.2.3"),
    "tcn": ("TCN", "0.2.2.1"),
}
MODEL_ORDER = ("cnn", "gru", "lstm", "tcn")
COLORS = {
    "cnn": "#59C7C2",
    "gru": "#59E83A",
    "lstm": "#E76B91",
    "tcn": "#294862",
}
COLLEAGUE_REPORTED_MAE = {
    "cnn": 0.0175,
    "gru": 0.0168,
    "lstm": 0.0188,
    "tcn": 0.0182,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--processing-chunk", type=int, default=168)
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
    required = ("df_FE_C11.parquet", "df_FE_C19.parquet", "df_FE_C23.parquet")
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if all((candidate / name).is_file() for name in required):
            return candidate
    raise FileNotFoundError(f"Could not find the legacy test cells below {candidates}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_directory(model_name: str) -> Path:
    family, version = MODEL_SPECS[model_name]
    return LEGACY_STUDY / "2_models" / family / "Base" / version


def read_config(model_name: str) -> dict:
    with (model_directory(model_name) / "config" / "train_soh.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        return yaml.safe_load(stream)


def build_model(model_name: str, config: dict, module, in_features: int):
    model = config["model"]
    if model_name == "cnn":
        return module.SOH_CNN_Seq2Seq(
            in_features=in_features,
            hidden_size=int(model["hidden_size"]),
            mlp_hidden=int(model["mlp_hidden"]),
            kernel_size=int(model.get("kernel_size", 5)),
            dilations=model.get("dilations"),
            num_blocks=int(model.get("num_blocks", 4)),
            dropout=float(model.get("dropout", 0.15)),
        )
    if model_name == "gru":
        return module.SOH_GRU_Seq2Seq(
            in_features=in_features,
            embed_size=int(model["embed_size"]),
            hidden_size=int(model["hidden_size"]),
            mlp_hidden=int(model["mlp_hidden"]),
            num_layers=int(model.get("num_layers", 2)),
            res_blocks=int(model.get("res_blocks", 2)),
            bidirectional=bool(model.get("bidirectional", False)),
            dropout=float(model.get("dropout", 0.15)),
        )
    if model_name == "lstm":
        return module.SOH_LSTM_Seq2Seq(
            in_features=in_features,
            embed_size=int(model["embed_size"]),
            hidden_size=int(model["hidden_size"]),
            mlp_hidden=int(model["mlp_hidden"]),
            num_layers=int(model.get("num_layers", 2)),
            res_blocks=int(model.get("res_blocks", 2)),
            bidirectional=bool(model.get("bidirectional", False)),
            dropout=float(model.get("dropout", 0.15)),
        )
    if model_name == "tcn":
        return module.CausalTCN_SOH(
            in_features=in_features,
            hidden_size=int(model["hidden_size"]),
            mlp_hidden=int(model["mlp_hidden"]),
            kernel_size=int(model["kernel_size"]),
            dilations=[int(value) for value in model["dilations"]],
            dropout=float(model.get("dropout", 0.05)),
        )
    raise ValueError(model_name)


def find_checkpoint(directory: Path) -> Path:
    checkpoints = list((directory / "checkpoints").glob("best_epoch*_rmse*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one best checkpoint in {directory}, found {checkpoints}")
    return checkpoints[0]


def aggregate_hourly(
    parquet_path: Path, base_features: list[str], target: str
) -> pd.DataFrame:
    columns = list(dict.fromkeys(base_features + [target, "Testtime[s]"]))
    raw = pd.read_parquet(parquet_path, columns=columns)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    raw = raw.sort_values("Testtime[s]")
    raw["_hour"] = (raw["Testtime[s]"] // 3600).astype(np.int64)
    specification = {feature: ["mean", "std", "min", "max"] for feature in base_features}
    specification[target] = ["last"]
    hourly = raw.groupby("_hour", sort=True).agg(specification)
    hourly.columns = [
        target if column[0] == target else f"{column[0]}_{column[1]}"
        for column in hourly.columns
    ]
    del raw
    gc.collect()
    return hourly.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


@torch.inference_mode()
def predict_independent_windows(
    model: torch.nn.Module, features: np.ndarray, chunk: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    complete_length = (len(features) // chunk) * chunk
    predictions = []
    for start in range(0, complete_length, chunk):
        batch = torch.from_numpy(features[start : start + chunk]).unsqueeze(0).to(device)
        predictions.append(model(batch).squeeze(0).detach().cpu().numpy())
    return np.concatenate(predictions), np.arange(complete_length, dtype=np.int64)


@torch.inference_mode()
def predict_continuous_rnn(
    model: torch.nn.Module, features: np.ndarray, chunk: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    state = None
    for start in range(0, len(features), chunk):
        batch = torch.from_numpy(features[start : start + chunk]).unsqueeze(0).to(device)
        output, state = model(batch, state=state, return_state=True)
        if isinstance(state, tuple):
            state = tuple(value.detach() for value in state)
        else:
            state = state.detach()
        predictions.append(output.squeeze(0).detach().cpu().numpy())
    return np.concatenate(predictions), np.arange(len(features), dtype=np.int64)


@torch.inference_mode()
def predict_continuous_convolution(
    model: torch.nn.Module, features: np.ndarray, chunk: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    context = None
    receptive_field = int(getattr(model, "receptive_field", 1))
    for start in range(0, len(features), chunk):
        current = features[start : start + chunk]
        model_input = current if context is None else np.concatenate((context, current), axis=0)
        output = model(torch.from_numpy(model_input).unsqueeze(0).to(device))
        predictions.append(output.squeeze(0).detach().cpu().numpy()[-len(current) :])
        context = model_input[-max(receptive_field - 1, 1) :].copy()
    return np.concatenate(predictions), np.arange(len(features), dtype=np.int64)


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    mse = float(np.mean(np.square(error)))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 17,
            "axes.labelsize": 21,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_trajectory(
    frames: dict[str, pd.DataFrame], output: Path, smoothed: bool, title: str
) -> None:
    fig, axis = plt.subplots(figsize=(16, 7.5))
    reference = frames["cnn"]
    axis.plot(
        reference["time_h"],
        reference["soh_reference"],
        color="#1A1A1A",
        linewidth=2.8,
        label="Reference SOH",
        zorder=5,
    )
    for model_name in MODEL_ORDER:
        frame = frames[model_name]
        prediction = frame["soh_prediction"]
        if smoothed:
            prediction = prediction.rolling(7, center=True, min_periods=1).median()
            indices = np.arange(0, len(frame), 3)
        else:
            indices = np.arange(len(frame))
        axis.plot(
            frame["time_h"].to_numpy()[indices],
            prediction.to_numpy()[indices],
            color=COLORS[model_name],
            linewidth=1.8,
            label=model_name.upper(),
        )
    axis.set_xlabel("Time [h]")
    axis.set_ylabel("SOH [0-1]")
    axis.set_ylim(0.62, 1.0)
    axis.set_title(title)
    axis.grid(color="#DADDE0", linewidth=1.0)
    axis.legend(loc="lower left", frameon=True, fancybox=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_mae(summary: pd.DataFrame, output: Path) -> None:
    c11 = summary[summary["scope"] == "C11"].copy()
    positions = np.arange(len(MODEL_ORDER))
    width = 0.34
    fig, axis = plt.subplots(figsize=(11.5, 6.4))
    for offset, mode, alpha in (
        (-width / 2, "independent_windows", 0.45),
        (width / 2, "continuous_context", 0.95),
    ):
        values = c11[c11["mode"] == mode].set_index("model").loc[list(MODEL_ORDER), "mae"]
        bars = axis.bar(
            positions + offset,
            values,
            width,
            color=[COLORS[name] for name in MODEL_ORDER],
            edgecolor=[COLORS[name] for name in MODEL_ORDER],
            alpha=alpha,
            label=mode.replace("_", " "),
        )
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.0005, f"{value:.4f}", ha="center")
    axis.set_xticks(positions, [name.upper() for name in MODEL_ORDER], fontweight="bold")
    axis.set_ylabel("MAE [0-1]")
    axis.set_xlabel("Architecture")
    axis.grid(axis="y", color="#DADDE0", linewidth=1.0)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def summarize(per_cell: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in per_cell["mode"].unique():
        for model_name in MODEL_ORDER:
            selected = per_cell[(per_cell["mode"] == mode) & (per_cell["model"] == model_name)]
            c11 = selected[selected["cell"] == "C11"].iloc[0]
            rows.append(
                {
                    "scope": "C11",
                    "mode": mode,
                    "model": model_name,
                    "mae": c11["mae"],
                    "rmse": c11["rmse"],
                }
            )
            rows.append(
                {
                    "scope": "legacy_test_cell_macro_C11_C19_C23",
                    "mode": mode,
                    "model": model_name,
                    "mae": float(selected["mae"].mean()),
                    "rmse": float(selected["rmse"].mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    predictions_dir = output / "predictions"
    plots_dir = output / "plots"
    predictions_dir.mkdir()
    plots_dir.mkdir()

    data_root = resolve_data_root(args.data_root)
    device = resolve_device(args.device)
    reference_config = read_config("lstm")
    base_features = list(reference_config["model"]["features"])
    target = str(reference_config["training"].get("target", "SOH"))
    expanded_features = [
        f"{feature}_{aggregation}"
        for feature in base_features
        for aggregation in ("mean", "std", "min", "max")
    ]
    test_cells = [value.replace("MGFarm_18650_", "") for value in reference_config["cells"]["test"]]

    hourly_cells = {}
    for cell in test_cells:
        print(f"Loading and aggregating {cell}")
        hourly_cells[cell] = aggregate_hourly(
            data_root / f"df_FE_{cell}.parquet", base_features, target
        )
        print(f"  {len(hourly_cells[cell])} hourly samples")

    metric_rows = []
    inventory_rows = []
    trajectory_frames: dict[str, dict[str, pd.DataFrame]] = {
        "independent_windows": {},
        "continuous_context": {},
    }

    for model_name in MODEL_ORDER:
        directory = model_directory(model_name)
        yaml_config = read_config(model_name)
        module = load_module(directory / "scripts" / "train_soh.py", f"legacy_{model_name}")
        checkpoint = find_checkpoint(directory)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        config = payload.get("config", yaml_config)
        model = build_model(model_name, config, module, len(expanded_features)).to(device)
        state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        scaler_path = directory / "scaler_robust.joblib"
        scaler = joblib.load(scaler_path)
        window = int(config["training"]["seq_chunk_size"])
        family, version = MODEL_SPECS[model_name]
        inventory_rows.append(
            {
                "model": model_name,
                "version": version,
                "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                "window_h": window,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "scaler_sha256": sha256(scaler_path),
                "architecture_source": "checkpoint_config",
                "yaml_matches_checkpoint_model_config": yaml_config["model"] == config["model"],
            }
        )
        print(f"Evaluating {family} {version}")

        for cell, hourly in hourly_cells.items():
            scaled = scaler.transform(hourly[expanded_features].to_numpy(dtype=np.float32)).astype(np.float32)
            reference_all = hourly[target].to_numpy(dtype=np.float32)
            for mode in ("independent_windows", "continuous_context"):
                if mode == "independent_windows":
                    prediction, indices = predict_independent_windows(model, scaled, window, device)
                elif model_name in ("gru", "lstm"):
                    prediction, indices = predict_continuous_rnn(
                        model, scaled, args.processing_chunk, device
                    )
                else:
                    prediction, indices = predict_continuous_convolution(
                        model, scaled, args.processing_chunk, device
                    )
                reference = reference_all[indices]
                values = metrics(reference, prediction)
                metric_rows.append(
                    {
                        "model": model_name,
                        "version": version,
                        "cell": cell,
                        "mode": mode,
                        "source_samples": len(hourly),
                        "evaluated_samples": len(indices),
                        "omitted_tail_samples": len(hourly) - len(indices),
                        **values,
                    }
                )
                frame = pd.DataFrame(
                    {
                        "sample_index": indices,
                        "time_h": indices.astype(np.float64),
                        "soh_reference": reference,
                        "soh_prediction": prediction,
                        "absolute_error": np.abs(prediction - reference),
                    }
                )
                frame.to_csv(
                    predictions_dir / f"{model_name}_{cell}_{mode}.csv", index=False
                )
                if mode == "continuous_context":
                    smoothed_prediction = (
                        pd.Series(prediction)
                        .rolling(7, center=True, min_periods=1)
                        .median()
                        .to_numpy(dtype=np.float64)
                    )
                    smoothed_values = metrics(reference, smoothed_prediction)
                    metric_rows.append(
                        {
                            "model": model_name,
                            "version": version,
                            "cell": cell,
                            "mode": "continuous_context_smoothed_display",
                            "source_samples": len(hourly),
                            "evaluated_samples": len(indices),
                            "omitted_tail_samples": len(hourly) - len(indices),
                            **smoothed_values,
                        }
                    )
                    smoothed_frame = frame.copy()
                    smoothed_frame["soh_prediction"] = smoothed_prediction
                    smoothed_frame["absolute_error"] = np.abs(
                        smoothed_prediction - reference
                    )
                    smoothed_frame.to_csv(
                        predictions_dir
                        / f"{model_name}_{cell}_continuous_context_smoothed_display.csv",
                        index=False,
                    )
                if cell == "C11":
                    trajectory_frames[mode][model_name] = frame
                print(
                    f"  {cell} {mode}: MAE={values['mae']:.6f}, "
                    f"RMSE={values['rmse']:.6f}, n={len(indices)}"
                )
        del model, scaler, module
        gc.collect()

    per_cell = pd.DataFrame(metric_rows)
    summary = summarize(per_cell)
    inventory = pd.DataFrame(inventory_rows)
    per_cell.to_csv(output / "metrics_by_cell_and_mode.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    inventory.to_csv(output / "model_inventory.csv", index=False)

    comparison_rows = []
    for model_name in MODEL_ORDER:
        row = {"model": model_name, "colleague_reported_mae": COLLEAGUE_REPORTED_MAE[model_name]}
        for scope, label in (
            ("C11", "c11"),
            ("legacy_test_cell_macro_C11_C19_C23", "legacy_macro"),
        ):
            for mode in ("independent_windows", "continuous_context"):
                value = summary[
                    (summary["scope"] == scope)
                    & (summary["mode"] == mode)
                    & (summary["model"] == model_name)
                ]["mae"].iloc[0]
                row[f"{label}_{mode}_mae"] = value
                row[f"{label}_{mode}_delta_vs_colleague"] = value - COLLEAGUE_REPORTED_MAE[model_name]
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(output / "comparison_to_colleague.csv", index=False)

    configure_style()
    plot_mae(summary, plots_dir / "legacy_C11_mae_by_inference_mode.png")
    plot_trajectory(
        trajectory_frames["independent_windows"],
        plots_dir / "legacy_C11_soh_independent_windows.png",
        smoothed=False,
        title="Legacy BATMM models: independent 168-hour windows",
    )
    plot_trajectory(
        trajectory_frames["continuous_context"],
        plots_dir / "legacy_C11_soh_continuous_context.png",
        smoothed=False,
        title="Legacy BATMM models: continuous context",
    )
    plot_trajectory(
        trajectory_frames["continuous_context"],
        plots_dir / "legacy_C11_soh_continuous_context_smoothed.png",
        smoothed=True,
        title="Legacy BATMM models: continuous context, smoothed display",
    )

    metadata = {
        "legacy_repository": str(LEGACY_STUDY.parent),
        "legacy_commit": "0ba6381a9b11545ae98b763edee051a6066d0d1d",
        "data_root": str(data_root),
        "device": str(device),
        "test_cells": test_cells,
        "common_comparison_cell": "C11",
        "target_aggregation": "last value per hour for all architectures",
        "configuration_precedence": "Checkpoint training configuration overrides the adjacent YAML.",
        "configuration_finding": "The LSTM YAML is stale and describes a smaller network than the checkpoint.",
        "independent_window_length_h": 168,
        "continuous_context": "RNN state carried; causal convolution context retained",
        "smoothing": "Centered 7-hour rolling median and every third plotting point; display only",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
