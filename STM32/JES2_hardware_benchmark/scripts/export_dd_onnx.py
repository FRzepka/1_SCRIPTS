#!/usr/bin/env python3
"""Export the exact JES2 pruned/fine-tuned DD checkpoint for STM32 tooling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from joblib import load


class GRUMLP(nn.Module):
    def __init__(self, features: int, hidden: int, mlp_hidden: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, sequence, state=None, return_state: bool = False):
        recurrent, new_state = self.gru(sequence, state)
        prediction = self.mlp(recurrent[:, -1, :]).squeeze(-1)
        if return_state:
            return prediction, new_state
        return prediction


class StatefulWrapper(nn.Module):
    def __init__(self, model: GRUMLP):
        super().__init__()
        self.model = model

    def forward(self, sample, state):
        return self.model(sample, state=state, return_state=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_scaler_header(path: Path, center, scale) -> None:
    center_values = ", ".join(f"{float(value):.9g}f" for value in center)
    scale_values = ", ".join(f"{float(value):.9g}f" for value in scale)
    path.write_text(
        "#ifndef JES2_DD_SCALER_H\n"
        "#define JES2_DD_SCALER_H\n\n"
        f"#define JES2_DD_FEATURE_COUNT {len(center)}U\n"
        f"static const float JES2_DD_SCALER_CENTER[JES2_DD_FEATURE_COUNT] = {{{center_values}}};\n"
        f"static const float JES2_DD_SCALER_SCALE[JES2_DD_FEATURE_COUNT] = {{{scale_values}}};\n\n"
        "#endif\n",
        encoding="ascii",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    default_model = root / "DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/PrunedFT_1.7.0.0_s30_struct"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_model / "config/train_soc.yaml")
    parser.add_argument("--checkpoint", type=Path, default=default_model / "checkpoints/best_model_finetuned.pt")
    parser.add_argument("--scaler", type=Path, default=default_model / "scaler_robust.joblib")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "exports/DD")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    features = list(model_cfg["features"])
    sequence_length = int(config["training"]["seq_chunk_size"])
    hidden = int(model_cfg["hidden_size"])
    layers = int(model_cfg.get("num_layers", 1))
    model = GRUMLP(
        features=len(features),
        hidden=hidden,
        mlp_hidden=int(model_cfg["mlp_hidden"]),
        layers=layers,
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scaler = load(args.scaler)
    if len(scaler.center_) != len(features) or len(scaler.scale_) != len(features):
        raise ValueError("Scaler dimensions do not match model features")
    scaler_header = args.out_dir / "jes2_dd_scaler.h"
    write_scaler_header(scaler_header, scaler.center_, scaler.scale_)

    fixed_path = args.out_dir / "jes2_dd_window2024.onnx"
    stateful_path = args.out_dir / "jes2_dd_stateful_step.onnx"
    dummy_window = torch.zeros((1, sequence_length, len(features)), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_window,
        fixed_path,
        input_names=["scaled_input_window"],
        output_names=["soc"],
        dynamic_axes=None,
        opset_version=int(config.get("export", {}).get("onnx_opset", 17)),
    )

    validation = {}
    try:
        import numpy as np
        import onnxruntime as ort

        generator = torch.Generator().manual_seed(42)
        validation_window = torch.randn(dummy_window.shape, generator=generator)
        with torch.no_grad():
            torch_output = model(validation_window).numpy()
        session = ort.InferenceSession(str(fixed_path), providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {"scaled_input_window": validation_window.numpy()})[0]
        validation["fixed_window_max_abs_difference"] = float(np.max(np.abs(torch_output - onnx_output)))
    except ImportError:
        validation["fixed_window_max_abs_difference"] = None
        validation["note"] = "Install onnxruntime to run numerical export validation"
    wrapper = StatefulWrapper(model)
    dummy_sample = torch.zeros((1, 1, len(features)), dtype=torch.float32)
    dummy_state = torch.zeros((layers, 1, hidden), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy_sample, dummy_state),
        stateful_path,
        input_names=["scaled_input_sample", "hidden_in"],
        output_names=["soc", "hidden_out"],
        dynamic_axes=None,
        opset_version=int(config.get("export", {}).get("onnx_opset", 17)),
    )

    manifest = {
        "schema_version": 1,
        "model": "JES2 DD pruned/fine-tuned GRU-MLP",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "scaler": str(args.scaler.resolve()),
        "scaler_sha256": sha256(args.scaler),
        "scaler_header": scaler_header.name,
        "scaler_header_sha256": sha256(scaler_header),
        "scaler_center": [float(value) for value in scaler.center_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "features": features,
        "sequence_length": sequence_length,
        "hidden_size": hidden,
        "num_layers": layers,
        "fixed_window_onnx": fixed_path.name,
        "fixed_window_onnx_sha256": sha256(fixed_path),
        "stateful_onnx": stateful_path.name,
        "stateful_onnx_sha256": sha256(stateful_path),
        "primary_hardware_comparison": "fixed_window_2024",
        "stateful_role": "separate deployment optimization; validate against rolling-window reference",
        "input_scaling": "RobustScaler is external to ONNX and must be reproduced in firmware",
        "validation": validation,
    }
    manifest_path = args.out_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
