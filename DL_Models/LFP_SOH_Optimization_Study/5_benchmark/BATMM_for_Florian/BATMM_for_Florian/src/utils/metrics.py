import csv
import io
import os
import time
import typing as T
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from src.utils.data_utils import FEATURES, TARGET
from config import RESULTS_DIR


# ============================================================
# EVALUATION METRICS
# ============================================================

def save_pairs_csv(y_true, y_pred, path=f"{RESULTS_DIR}/pruning/predictions.csv"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    file_exists = p.exists()

    data = np.column_stack((y_true, y_pred))

    with open(p, "ab") as f:
        if not file_exists:
            np.savetxt(f, data, delimiter=",", header="y_true,y_pred", comments="")
        else:
            np.savetxt(f, data, delimiter=",", comments="")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics over flattened target/prediction arrays."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.size == 0:
        raise ValueError("Cannot compute metrics for empty arrays.")
    if y_true.size != y_pred.size:
        raise ValueError(
            "y_true and y_pred must contain the same number of values; "
            f"got {y_true.size} and {y_pred.size}."
        )
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("y_true and y_pred must contain only finite values.")

    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)

    mse = float(np.mean(np.square(residuals)))
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(mse))
    max_error = float(np.max(abs_errors))

    ss_res = float(np.sum(np.square(residuals)))
    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(np.square(centered)))

    # Match the finite behavior commonly used for R2 on a constant target:
    # perfect predictions score 1.0; all other predictions score 0.0.
    scale = max(1.0, float(np.sum(np.square(y_true))))
    tolerance = np.finfo(np.float64).eps * scale
    if ss_tot <= tolerance:
        r2 = 1.0 if ss_res <= tolerance else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "max_error": max_error,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thread_state: bool = False, # Thread the state to the next batch if requested
) -> dict:
    """Evaluate a sequence-to-sequence regressor on every target timestep.

    When ``thread_state`` is true, hidden state is carried only for models whose
    forward method supports ``state`` and ``return_state`` (the supplied LSTM
    and GRU).  CNN/TCN models are evaluated normally.  State is initialized once
    per call, so calling this function separately for each battery cell resets it.
    """
    if loader is None:
        raise ValueError("loader must not be None.")

    model.to(device)
    was_training = model.training
    model.eval()

    floating_param = next((p for p in model.parameters() if p.is_floating_point()), None)
    model_dtype = floating_param.dtype if floating_param is not None else torch.float32
    non_blocking = device.type == "cuda"

    base_model = model.module if hasattr(model, "module") else model
    supports_state = hasattr(base_model, "lstm") or hasattr(base_model, "gru")
    if not supports_state:
        try:
            forward_parameters = inspect.signature(base_model.forward).parameters
            supports_state = (
                "state" in forward_parameters and "return_state" in forward_parameters
            )
        except (TypeError, ValueError):
            supports_state = False

    state = None
    y_true_parts: T.List[np.ndarray] = []
    y_pred_parts: T.List[np.ndarray] = []

    def _state_batch_size(current_state) -> T.Optional[int]:
        if current_state is None:
            return None
        state_tensor = current_state[0] if isinstance(current_state, (tuple, list)) else current_state
        if not torch.is_tensor(state_tensor) or state_tensor.ndim < 2:
            return None
        # PyTorch LSTM/GRU states have shape (layers * directions, batch, hidden).
        return int(state_tensor.shape[1])

    def _detach_state(current_state):
        if torch.is_tensor(current_state):
            return current_state.detach()
        if isinstance(current_state, tuple):
            return tuple(_detach_state(item) for item in current_state)
        if isinstance(current_state, list):
            return [_detach_state(item) for item in current_state]
        return current_state

    try:
        for inputs, targets in loader:
            inputs = inputs.to(
                device=device,
                dtype=model_dtype,
                non_blocking=non_blocking,
            )
            targets = targets.to(
                device=device,
                dtype=model_dtype,
                non_blocking=non_blocking,
            )

            if thread_state and supports_state:
                if state is not None and _state_batch_size(state) != inputs.shape[0]:
                    # A changed batch size cannot reuse a recurrent state safely.
                    state = None
                output = model(inputs, state=state, return_state=True)
                if not isinstance(output, (tuple, list)) or len(output) != 2:
                    raise RuntimeError(
                        "A stateful model must return (predictions, state) when "
                        "called with return_state=True."
                    )
                predictions, state = output
                state = _detach_state(state)
            else:
                predictions = model(inputs)
                if isinstance(predictions, (tuple, list)):
                    predictions = predictions[0]

            if predictions.ndim == targets.ndim + 1 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            if predictions.numel() != targets.numel():
                raise ValueError(
                    "Model output and target contain different numbers of values: "
                    f"prediction shape={tuple(predictions.shape)}, "
                    f"target shape={tuple(targets.shape)}."
                )
            predictions = predictions.reshape_as(targets)

            y_true_parts.append(targets.detach().cpu().reshape(-1).numpy())
            y_pred_parts.append(predictions.detach().cpu().reshape(-1).numpy())
    finally:
        model.train(was_training)

    if not y_true_parts:
        raise ValueError("The evaluation loader produced no target values.")

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    return compute_metrics(y_true, y_pred)


# ============================================================
# MODEL SIZE & INFERENCE TIME
# ============================================================

def get_model_size_mb(model: nn.Module, quantized: bool = False) -> float:
    """Return model size in megabytes."""
    if quantized:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.tell() / (1024 ** 2)
    param_size  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def measure_inference_time(
    model:          nn.Module,
    example_inputs: torch.Tensor,
    device:         torch.device,
    num_runs:       int = 100,
) -> float:
    """Return mean inference latency in milliseconds (measured on CPU)."""
    model          = model.to(torch.device("cpu"))
    example_inputs = example_inputs.to(torch.device("cpu"))
    model.eval()

    with torch.no_grad():                       # warm-up
        for _ in range(10):
            model(example_inputs)

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(example_inputs)
    elapsed_ms = (time.time() - start) / num_runs * 1000

    model.to(device)
    return elapsed_ms


# ============================================================
# CSV LOGGING
# ============================================================

_CSV_HEADER = [
    "architecture", "pruning_style", "quantization", "actual_ratio",
    "params_before", "params_after",
    "size_ratio", "size_before", "size_after",
    "macs_ratio", "macs_before", "macs_after",
    "inference_ratio", "inference_before", "inference_after",
    "test_mae", "test_rmse", "test_r2", "test_max_err",
]


def append_metrics_to_csv(
    csv_path:         str,
    model_name:       str,
    pruning_style:    str,
    quantization:     bool,
    actual_ratio:     float,
    params_before:    int,
    params_after:     int,
    size_before:      float,
    size_after:       float,
    macs_before:      int,
    macs_after:       int,
    inference_before: float,
    inference_after:  float,
    test_mae:         float,
    test_rmse:        float,
    test_r2:          float,
    test_max_err:     float,
) -> None:
    """Append one row of compression/evaluation metrics to a CSV file.

    The header is written automatically if the file is new or empty.
    """
    safe_div = lambda a, b: a / b if b > 0 else 0.0

    row = [
        model_name, pruning_style, quantization, actual_ratio,
        params_before, params_after,
        safe_div(size_after,      size_before),      size_before,      size_after,
        safe_div(macs_after,      macs_before),      macs_before,      macs_after,
        safe_div(inference_after, inference_before),  inference_before,  inference_after,
        test_mae, test_rmse, test_r2, test_max_err,
    ]

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(_CSV_HEADER)
        writer.writerow(row)

    print(f"\nMetrics appended to {csv_path}")
