import csv
import io
import os
import time
import typing as T
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
    save_pairs_csv(y_true, y_pred)
    err    = y_true - y_pred
    mse    = float((err ** 2).mean())      # <-- ADDED: Calculate MSE
    mae    = float(np.abs(err).mean())
    rmse   = float(np.sqrt(mse))           # <-- UPDATED: Reuse MSE for RMSE
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "max_error": float(np.abs(err).max())}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thread_state: bool = False,
) -> dict:

    model.eval()
    preds, targets = [], []
    state = None

    is_stateful = any(isinstance(m, (nn.LSTM, nn.GRU)) for m in model.modules())

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        if is_stateful:
            # Pass the state through the batch (requires batch_size=1 for test)
            pred, new_state = model(xb, state=state, return_state=True)

            # Only thread the state to the next batch if explicitly requested
            if thread_state:
                state = tuple(s.detach() for s in new_state) if isinstance(new_state, tuple) else new_state.detach()
            else:
                state = None # Reset state for the next batch
        else:
            pred = model(xb)

        preds.append(pred.cpu().reshape(-1))
        targets.append(yb.cpu().reshape(-1))

    return compute_metrics(torch.cat(targets).numpy(), torch.cat(preds).numpy())


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

