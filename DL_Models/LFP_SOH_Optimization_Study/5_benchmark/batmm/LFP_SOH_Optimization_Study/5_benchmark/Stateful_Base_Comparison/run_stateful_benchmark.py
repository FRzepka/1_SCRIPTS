#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from models import MODEL_CLASSES


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parents[1]
MODEL_SPECS = {
    "cnn": ("CNN", "0.4.2.1"),
    "gru": ("GRU", "0.3.1.2"),
    "lstm": ("LSTM", "0.1.2.4"),
    "tcn": ("TCN", "0.2.2.2"),
}
MODEL_ORDER = ("cnn", "gru", "lstm", "tcn")
TEST_CELLS = ("C11", "C23", "C29")
BASE_FEATURES = (
    "Voltage[V]",
    "Current[A]",
    "Temperature[\N{DEGREE SIGN}C]",
    "EFC",
    "Q_c",
)
AGGREGATIONS = ("mean", "std", "min", "max")
FEATURES = tuple(
    f"{feature}_{aggregation}"
    for feature in BASE_FEATURES
    for aggregation in AGGREGATIONS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="auto")
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
    required = [f"df_FE_{cell}.parquet" for cell in TEST_CELLS]
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if all((candidate / filename).is_file() for filename in required):
            return candidate
    raise FileNotFoundError(
        "Data directory not found. Pass it with --data-root or "
        "MGFARM_FE_DATA_ROOT."
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def model_directory(model_name: str) -> Path:
    family, version = MODEL_SPECS[model_name]
    return STUDY_ROOT / "2_models" / family / "Base" / version


def load_config(model_name: str) -> dict:
    path = model_directory(model_name) / "config" / "train_soh.yaml"
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_model(model_name: str, config: dict) -> torch.nn.Module:
    parameters = dict(config["model"])
    parameters.pop("type", None)
    parameters.pop("features", None)
    if model_name == "tcn":
        parameters.pop("output_kernel_size", None)
    return MODEL_CLASSES[model_name](in_features=len(FEATURES), **parameters)


def aggregate_hourly(path: Path) -> pd.DataFrame:
    columns = [*BASE_FEATURES, "SOH", "Testtime[s]"]
    data = pd.read_parquet(path, columns=columns)
    for column in data.select_dtypes(include=["float64"]).columns:
        data[column] = data[column].astype(np.float32)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    data = data.sort_values("Testtime[s]")
    data["_hour"] = (data["Testtime[s]"] // 3600).astype(np.int32)
    specification = {
        feature: list(AGGREGATIONS) for feature in BASE_FEATURES
    }
    specification["SOH"] = ["last"]
    hourly = data.groupby("_hour", sort=False).agg(specification)
    hourly.columns = [
        "SOH" if column[0] == "SOH" else f"{column[0]}_{column[1]}"
        for column in hourly.columns
    ]
    return (
        hourly.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[*FEATURES, "SOH"])
        .reset_index(drop=True)
        .astype(np.float32)
    )


@torch.inference_mode()
def predict_recurrent(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    predictions = []
    state = None
    for start in range(0, len(features), chunk_size):
        inputs = torch.from_numpy(features[start : start + chunk_size]).unsqueeze(0)
        output, state = model(inputs.to(device), state=state, return_state=True)
        if isinstance(state, tuple):
            state = tuple(value.detach() for value in state)
        else:
            state = state.detach()
        predictions.append(output.squeeze(0).cpu().numpy())
    return np.concatenate(predictions)


@torch.inference_mode()
def predict_convolution(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    history_length = int(model.receptive_field) - 1
    history = None
    predictions = []
    for start in range(0, len(features), chunk_size):
        current = features[start : start + chunk_size]
        model_input = (
            current if history is None else np.concatenate((history, current), axis=0)
        )
        inputs = torch.from_numpy(model_input).unsqueeze(0).to(device)
        output = model(inputs).squeeze(0).cpu().numpy()
        predictions.append(output[-len(current) :])
        history = model_input[-history_length:].copy() if history_length else None
    return np.concatenate(predictions)


def calculate_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(np.square(error)))
    denominator = float(np.sum(np.square(reference - np.mean(reference))))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - np.sum(np.square(error)) / denominator),
        "max_error": float(np.max(np.abs(error))),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_trajectory(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    combined = frames["cnn"][["time_h", "soh_reference"]].copy()
    for model_name in MODEL_ORDER:
        combined[f"soh_{model_name}"] = frames[model_name]["soh_prediction"]
    combined.to_csv(output_dir / "soh_trajectory_C11.csv", index=False)


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    hourly_cells = {
        cell: aggregate_hourly(data_root / f"df_FE_{cell}.parquet")
        for cell in TEST_CELLS
    }

    rows = []
    predictions = {}
    c11_frames = {}
    inventory = []
    for model_name in MODEL_ORDER:
        directory = model_directory(model_name)
        config = load_config(model_name)
        checkpoint_path = directory / "checkpoints" / "best_model.pt"
        scaler_path = directory / "scaler_robust.joblib"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(model_name, config).to(device)
        state_dict = checkpoint.get(
            "model_state_dict", checkpoint.get("state_dict", checkpoint)
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        scaler = joblib.load(scaler_path)
        chunk_size = int(config["training"]["seq_chunk_size"])
        inventory.append(
            {
                "model": model_name,
                "version": MODEL_SPECS[model_name][1],
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "scaler_sha256": file_sha256(scaler_path),
            }
        )

        for cell, hourly in hourly_cells.items():
            scaled = scaler.transform(
                hourly[list(FEATURES)].to_numpy(dtype=np.float32)
            ).astype(np.float32)
            if model_name in ("gru", "lstm"):
                prediction = predict_recurrent(model, scaled, device, chunk_size)
                mode = "continuous_recurrent_state"
            else:
                prediction = predict_convolution(model, scaled, device, chunk_size)
                mode = "continuous_convolution_context"
            reference = hourly["SOH"].to_numpy(dtype=np.float32)
            metrics = calculate_metrics(reference, prediction)
            rows.append(
                {
                    "model": model_name,
                    "cell": cell,
                    "inference_mode": mode,
                    "samples": len(reference),
                    **metrics,
                }
            )
            predictions[(model_name, cell)] = (reference, prediction)
            if cell == "C11":
                c11_frames[model_name] = pd.DataFrame(
                    {
                        "time_h": np.arange(len(reference), dtype=np.int64),
                        "soh_reference": reference,
                        "soh_prediction": prediction,
                    }
                )
            print(
                f"{model_name.upper()} {cell}: "
                f"MAE={metrics['mae']:.8f}, RMSE={metrics['rmse']:.8f}"
            )

    per_cell = pd.DataFrame(rows)
    summary_rows = []
    mae_rows = []
    for model_name in MODEL_ORDER:
        model_cells = per_cell[per_cell["model"] == model_name]
        macro = {
            metric: float(model_cells[metric].mean())
            for metric in ("mse", "mae", "rmse", "r2", "max_error")
        }
        references = np.concatenate(
            [predictions[(model_name, cell)][0] for cell in TEST_CELLS]
        )
        estimates = np.concatenate(
            [predictions[(model_name, cell)][1] for cell in TEST_CELLS]
        )
        weighted = calculate_metrics(references, estimates)
        summary_rows.extend(
            [
                {"model": model_name, "aggregation": "cell_macro", **macro},
                {
                    "model": model_name,
                    "aggregation": "sample_weighted",
                    **weighted,
                },
            ]
        )
        mae_rows.append(
            {
                "model": model_name.upper(),
                **{
                    f"mae_{cell}": float(
                        model_cells.loc[model_cells["cell"] == cell, "mae"].iloc[0]
                    )
                    for cell in TEST_CELLS
                },
                "mae_cell_macro": macro["mae"],
                "mae_sample_weighted": weighted["mae"],
            }
        )

    per_cell.to_csv(output_dir / "metrics_by_cell.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(mae_rows).to_csv(output_dir / "mae_results.csv", index=False)
    pd.DataFrame(inventory).to_csv(output_dir / "model_inventory.csv", index=False)
    save_trajectory(c11_frames, output_dir)
    metadata = {
        "data_directory": data_root.name,
        "data_files": [f"df_FE_{cell}.parquet" for cell in TEST_CELLS],
        "device": str(device),
        "test_cells": list(TEST_CELLS),
        "sampling_interval_seconds": 3600,
        "feature_aggregation": list(AGGREGATIONS),
        "target_aggregation": "last",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
