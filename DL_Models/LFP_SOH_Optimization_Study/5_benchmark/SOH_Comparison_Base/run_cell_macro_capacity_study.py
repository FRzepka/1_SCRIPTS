#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parents[1]
SPECS_PATH = STUDY_ROOT / "base_size_study_specs.json"
DEFAULT_OUTPUT = HERE / "results" / "CURRENT_MODELS_CELL_MACRO"
TEST_CELLS = ("C11", "C23", "C29")
SELECTED_ROLES = {
    "CNN": "base",
    "GRU": "larger_1",
    "LSTM": "larger_1",
    "TCN": "base",
}
SELECTED_MODEL_VERSIONS = {
    "CNN": "0.4.2.1_hp",
    "GRU": "0.3.1.2",
    "LSTM": "0.1.2.4",
    "TCN": "0.2.2.2",
}
BATMM_MODEL_VERSIONS = {
    "CNN": "0.4.2.1",
    "GRU": "0.3.1.2",
    "LSTM": "0.1.2.4",
    "TCN": "0.2.2.2",
}
REFERENCE_RESULTS = (
    STUDY_ROOT
    / "5_benchmark"
    / "batmm"
    / "LFP_SOH_Optimization_Study"
    / "5_benchmark"
    / "Stateful_Base_Comparison"
    / "results"
    / "metrics_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every trained base-size variant with the same continuous, "
            "cell-macro protocol used for the selected-model comparison."
        )
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cells", nargs="+", default=list(TEST_CELLS))
    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument("--skip-reference-check", action="store_true")
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
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if all((candidate / f"df_FE_{cell}.parquet").is_file() for cell in TEST_CELLS):
            return candidate
    raise FileNotFoundError(
        "Feature-data directory not found. Pass --data-root or set "
        "MGFARM_FE_DATA_ROOT."
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_specs() -> dict:
    with SPECS_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def variant_model_directory(family: str, variant: dict) -> Path:
    return (
        STUDY_ROOT
        / "2_models"
        / family
        / "Base"
        / f"{variant['version']}_{variant['tag']}"
    )


def selected_model_candidates(family: str) -> tuple[Path, Path]:
    canonical = (
        STUDY_ROOT
        / "2_models"
        / family
        / "Base"
        / SELECTED_MODEL_VERSIONS[family]
    )
    batmm_copy = (
        STUDY_ROOT
        / "5_benchmark"
        / "batmm"
        / "LFP_SOH_Optimization_Study"
        / "2_models"
        / family
        / "Base"
        / BATMM_MODEL_VERSIONS[family]
    )
    return canonical, batmm_copy


def required_model_files(directory: Path) -> tuple[Path, Path, Path]:
    return (
        directory / "config" / "train_soh.yaml",
        directory / "checkpoints" / "best_model.pt",
        directory / "scaler_robust.joblib",
    )


def model_directory(family: str, variant: dict) -> Path:
    if variant["role"] == SELECTED_ROLES[family]:
        for candidate in selected_model_candidates(family):
            if all(path.is_file() for path in required_model_files(candidate)):
                return candidate
        return selected_model_candidates(family)[0]
    return variant_model_directory(family, variant)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_module(family: str, selected_variant: dict):
    script_path = (
        variant_model_directory(family, selected_variant)
        / "scripts"
        / "train_soh.py"
    )
    if not script_path.is_file():
        raise FileNotFoundError(f"Model definition missing: {script_path}")
    specification = importlib.util.spec_from_file_location(
        f"capacity_{family.lower()}_model", script_path
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def build_model(
    family: str, config: dict, module, in_features: int
) -> torch.nn.Module:
    model_config = config["model"]
    if family == "LSTM":
        return module.SOH_LSTM_Seq2Seq(
            in_features=in_features,
            embed_size=int(model_config.get("embed_size", 96)),
            hidden_size=int(model_config["hidden_size"]),
            mlp_hidden=int(model_config["mlp_hidden"]),
            num_layers=int(model_config.get("num_layers", 2)),
            res_blocks=int(model_config.get("res_blocks", 2)),
            bidirectional=bool(model_config.get("bidirectional", False)),
            dropout=float(model_config.get("dropout", 0.15)),
        )
    if family == "GRU":
        return module.SOH_GRU_Seq2Seq(
            in_features=in_features,
            embed_size=int(model_config.get("embed_size", 96)),
            hidden_size=int(model_config["hidden_size"]),
            mlp_hidden=int(model_config["mlp_hidden"]),
            num_layers=int(model_config.get("num_layers", 2)),
            res_blocks=int(model_config.get("res_blocks", 2)),
            bidirectional=bool(model_config.get("bidirectional", False)),
            dropout=float(model_config.get("dropout", 0.15)),
        )
    if family == "TCN":
        return module.CausalTCN_SOH(
            in_features=in_features,
            hidden_size=int(model_config["hidden_size"]),
            mlp_hidden=int(model_config["mlp_hidden"]),
            kernel_size=int(model_config.get("kernel_size", 3)),
            dilations=model_config.get("dilations") or [1, 2, 4, 8],
            dropout=float(model_config.get("dropout", 0.05)),
        )
    if family == "CNN":
        model_class = module.SOH_CNN_Seq2Seq
        parameters = {
            "in_features": in_features,
            "hidden_size": int(model_config.get("hidden_size", 128)),
            "mlp_hidden": int(model_config.get("mlp_hidden", 96)),
            "kernel_size": int(model_config.get("kernel_size", 5)),
            "dilations": model_config.get("dilations"),
            "num_blocks": int(model_config.get("num_blocks", 4)),
            "dropout": float(model_config.get("dropout", 0.15)),
        }
        if "output_kernel_size" in inspect.signature(model_class).parameters:
            parameters["output_kernel_size"] = int(
                model_config.get("output_kernel_size", 1)
            )
        return model_class(**parameters)
    raise ValueError(f"Unsupported architecture: {family}")


@torch.inference_mode()
def predict_stateful_rnn(
    model: torch.nn.Module,
    inputs: np.ndarray,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions = []
    state = None
    for start in range(0, len(inputs), chunk_size):
        tensor = torch.from_numpy(inputs[start : start + chunk_size]).unsqueeze(0)
        output, state = model(tensor.to(device), state=state, return_state=True)
        if isinstance(state, tuple):
            state = tuple(value.detach() for value in state)
        else:
            state = state.detach()
        predictions.append(output.squeeze(0).cpu().numpy())
    return np.concatenate(predictions)


@torch.inference_mode()
def predict_causal_buffer(
    model: torch.nn.Module,
    inputs: np.ndarray,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    history_length = int(model.receptive_field) - 1
    history = None
    predictions = []
    for start in range(0, len(inputs), chunk_size):
        current = inputs[start : start + chunk_size]
        model_input = (
            current if history is None else np.concatenate((history, current), axis=0)
        )
        tensor = torch.from_numpy(model_input).unsqueeze(0).to(device)
        output = model(tensor).squeeze(0).cpu().numpy()
        predictions.append(output[-len(current) :])
        history = model_input[-history_length:].copy() if history_length else None
    return np.concatenate(predictions)


def compute_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    error = estimate.astype(np.float64) - reference.astype(np.float64)
    squared_error = np.square(error)
    denominator = float(np.sum(np.square(reference - np.mean(reference))))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "r2": float(1.0 - np.sum(squared_error) / denominator),
    }


def normalized_cell(cell: str) -> str:
    return cell.rsplit("_", 1)[-1]


def feature_columns(config: dict) -> tuple[list[str], list[str]]:
    base_features = list(config["model"]["features"])
    sampling = config.get("sampling", {})
    aggregations = list(sampling.get("feature_aggs", ["mean"]))
    features = [
        f"{feature}_{aggregation}"
        for feature in base_features
        for aggregation in aggregations
    ]
    return base_features, features


def aggregate_hourly(path: Path, config: dict) -> pd.DataFrame:
    base_features, features = feature_columns(config)
    target = str(config.get("training", {}).get("target", "SOH"))
    sampling = config.get("sampling", {})
    interval = int(sampling.get("interval_seconds", 3600))
    aggregations = list(sampling.get("feature_aggs", ["mean"]))
    columns = [*base_features, target, "Testtime[s]"]

    data = pd.read_parquet(path, columns=columns)
    for column in data.select_dtypes(include=["float64"]).columns:
        data[column] = data[column].astype(np.float32)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    data = data.sort_values("Testtime[s]")
    data["_interval"] = (data["Testtime[s]"] // interval).astype(np.int32)
    specification = {feature: aggregations for feature in base_features}
    # Figure 3 uses the final SOH observation in every hourly interval for all
    # architecture families. Keep that target definition fixed here as well.
    specification[target] = ["last"]
    hourly = data.groupby("_interval", sort=False).agg(specification)
    hourly.columns = [
        target if column[0] == target else f"{column[0]}_{column[1]}"
        for column in hourly.columns
    ]
    return (
        hourly.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[*features, target])
        .reset_index(drop=True)
        .astype(np.float32)
    )


def preflight(specs: dict, selected_only: bool = False) -> list[str]:
    missing = []
    for family_spec in specs["families"]:
        family = family_spec["family"]
        for variant in family_spec["variants"]:
            if selected_only and variant["role"] != SELECTED_ROLES[family]:
                continue
            directory = model_directory(family, variant)
            required = required_model_files(directory)
            missing.extend(str(path) for path in required if not path.is_file())
    return missing


def cell_macro_summary(
    per_cell: pd.DataFrame,
    predictions: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    metrics = ("mae", "rmse", "r2")
    for (family, tag), group in per_cell.groupby(["architecture", "variant"], sort=False):
        base = group.iloc[0]
        macro = {metric: float(group[metric].mean()) for metric in metrics}
        references = np.concatenate(
            [predictions[(family, tag, cell)][0] for cell in group["cell"]]
        )
        estimates = np.concatenate(
            [predictions[(family, tag, cell)][1] for cell in group["cell"]]
        )
        weighted = compute_metrics(references, estimates)
        common = {
            "architecture": family,
            "variant": tag,
            "version": base["version"],
            "role": base["role"],
            "parameters": int(base["parameters"]),
            "selected_for_figure_3": bool(base["selected_for_figure_3"]),
        }
        rows.append({**common, "aggregation": "cell_macro", **macro})
        rows.append({**common, "aggregation": "sample_weighted", **weighted})
    return pd.DataFrame(rows)


def reference_consistency(summary: pd.DataFrame) -> pd.DataFrame:
    if not REFERENCE_RESULTS.is_file():
        return pd.DataFrame()
    reference = pd.read_csv(REFERENCE_RESULTS)
    reference = reference[reference["aggregation"] == "cell_macro"].copy()
    reference["architecture"] = reference["model"].str.upper()
    selected = summary[
        (summary["aggregation"] == "cell_macro")
        & summary["selected_for_figure_3"]
    ].copy()
    comparison = selected.merge(
        reference[["architecture", "mae", "rmse"]],
        on="architecture",
        suffixes=("_capacity_study", "_figure_3"),
        validate="one_to_one",
    )
    comparison["mae_difference"] = (
        comparison["mae_capacity_study"] - comparison["mae_figure_3"]
    )
    comparison["rmse_difference"] = (
        comparison["rmse_capacity_study"] - comparison["rmse_figure_3"]
    )
    return comparison[
        [
            "architecture",
            "variant",
            "mae_capacity_study",
            "mae_figure_3",
            "mae_difference",
            "rmse_capacity_study",
            "rmse_figure_3",
            "rmse_difference",
        ]
    ]


def main() -> None:
    args = parse_args()
    specs = load_specs()
    missing = preflight(specs, selected_only=args.selected_only)
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The capacity study cannot be evaluated because trained artifacts "
            f"are missing:\n{lines}"
        )

    cells = tuple(normalized_cell(cell) for cell in args.cells)
    data_root = resolve_data_root(args.data_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model_modules = {}
    for family_spec in specs["families"]:
        family = family_spec["family"]
        selected_variant = next(
            variant
            for variant in family_spec["variants"]
            if variant["role"] == SELECTED_ROLES[family]
        )
        model_modules[family] = load_training_module(family, selected_variant)

    first_family = specs["families"][0]
    first_variant = first_family["variants"][0]
    first_config_path = (
        model_directory(first_family["family"], first_variant)
        / "config"
        / "train_soh.yaml"
    )
    with first_config_path.open("r", encoding="utf-8") as stream:
        reference_config = yaml.safe_load(stream)
    _, reference_features = feature_columns(reference_config)
    target = str(reference_config.get("training", {}).get("target", "SOH"))
    hourly_cells = {
        cell: aggregate_hourly(data_root / f"df_FE_{cell}.parquet", reference_config)
        for cell in cells
    }

    rows = []
    inventory = []
    predictions = {}
    for family_spec in specs["families"]:
        family = family_spec["family"]
        for variant in family_spec["variants"]:
            if args.selected_only and variant["role"] != SELECTED_ROLES[family]:
                continue
            directory = model_directory(family, variant)
            config_path = directory / "config" / "train_soh.yaml"
            checkpoint_path = directory / "checkpoints" / "best_model.pt"
            scaler_path = directory / "scaler_robust.joblib"
            with config_path.open("r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            _, features = feature_columns(config)
            if features != reference_features:
                raise ValueError(f"Feature mismatch in {directory}: {features}")

            model = build_model(
                family, config, model_modules[family], len(features)
            ).to(device)
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            state_dict = checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            )
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            scaler = joblib.load(scaler_path)
            chunk_size = int(config["training"]["seq_chunk_size"])
            parameters = int(sum(parameter.numel() for parameter in model.parameters()))
            selected = variant["role"] == SELECTED_ROLES[family]

            inventory.append(
                {
                    "architecture": family,
                    "variant": variant["tag"],
                    "version": variant["version"],
                    "role": variant["role"],
                    "parameters": parameters,
                    "selected_for_figure_3": selected,
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "scaler_sha256": file_sha256(scaler_path),
                    "model_directory": str(directory.relative_to(STUDY_ROOT)),
                }
            )

            for cell, hourly in hourly_cells.items():
                inputs = scaler.transform(
                    hourly[features].to_numpy(dtype=np.float32)
                ).astype(np.float32)
                if family in ("GRU", "LSTM"):
                    estimates = predict_stateful_rnn(
                        model, inputs, chunk_size, device
                    )
                    inference_mode = "continuous_recurrent_state"
                else:
                    estimates = predict_causal_buffer(
                        model, inputs, chunk_size, device
                    )
                    inference_mode = "continuous_convolution_context"
                reference = hourly[target].to_numpy(dtype=np.float32)
                metrics = compute_metrics(reference, estimates)
                predictions[(family, variant["tag"], cell)] = (
                    reference,
                    estimates,
                )
                rows.append(
                    {
                        "architecture": family,
                        "variant": variant["tag"],
                        "version": variant["version"],
                        "role": variant["role"],
                        "cell": cell,
                        "samples": len(reference),
                        "parameters": parameters,
                        "selected_for_figure_3": selected,
                        "inference_mode": inference_mode,
                        **metrics,
                    }
                )
                print(
                    f"{family:4s} {variant['tag']:10s} {cell}: "
                    f"MAE={metrics['mae']:.8f}, RMSE={metrics['rmse']:.8f}"
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    per_cell = pd.DataFrame(rows)
    summary = cell_macro_summary(per_cell, predictions)
    capacity = summary[summary["aggregation"] == "cell_macro"].copy()
    capacity["source"] = "current models, continuous three-cell evaluation"
    comparison = reference_consistency(summary)

    per_cell.to_csv(output_dir / "metrics_by_cell.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    capacity.to_csv(output_dir / "capacity_sensitivity_data.csv", index=False)
    pd.DataFrame(inventory).to_csv(output_dir / "model_inventory.csv", index=False)
    if not comparison.empty:
        comparison.to_csv(output_dir / "reference_consistency.csv", index=False)
        maximum_difference = float(
            comparison[["mae_difference", "rmse_difference"]].abs().to_numpy().max()
        )
        if maximum_difference > 1e-7 and not args.skip_reference_check:
            raise RuntimeError(
                "Selected-model results do not reproduce Figure 3. Maximum "
                f"absolute metric difference: {maximum_difference:.3e}."
            )

    metadata = {
        "data_root": str(data_root),
        "test_cells": list(cells),
        "device": str(device),
        "sampling_interval_seconds": 3600,
        "feature_aggregations": ["mean", "std", "min", "max"],
        "target_aggregation": "last",
        "aggregation_for_figure": "cell_macro",
        "inference": "continuous causal context",
        "reference_results": str(REFERENCE_RESULTS),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
