"""Evaluate the four uncompressed PyTorch base models on the PC."""

import argparse
import csv
import gc
from pathlib import Path

import joblib
import torch

from config import CHUNK_SIZES, DATA_DIR, RESULTS_DIR, TEST_CELLS
from src.utils.data_utils import build_dataloaders
from src.utils.metrics import evaluate
import src.utils.cnn_utils as cnn_u
import src.utils.gru_utils as gru_u
import src.utils.lstm_utils as lstm_u
import src.utils.tcn_utils as tcn_u


MODEL_CONFIGS = {
    "cnn": (cnn_u.create_cnn, cnn_u.CHECKPOINT_PATH, cnn_u.SCALER_PATH, False),
    "tcn": (tcn_u.create_tcn, tcn_u.CHECKPOINT_PATH, tcn_u.SCALER_PATH, False),
    "lstm": (lstm_u.create_lstm, lstm_u.CHECKPOINT_PATH, lstm_u.SCALER_PATH, True),
    "gru": (gru_u.create_gru, gru_u.CHECKPOINT_PATH, gru_u.SCALER_PATH, True),
}
METRIC_NAMES = ("mse", "mae", "rmse", "r2", "max_error")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the CNN, TCN, LSTM and GRU base checkpoints on the PC."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CONFIGS,
        default=list(MODEL_CONFIGS),
        help="Models to evaluate (default: all four).",
    )
    parser.add_argument("--data_root", default=DATA_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--csv_output",
        default=str(Path(RESULTS_DIR) / "pc_base_model_metrics.csv"),
        help="Output CSV path; pass an empty string to disable CSV output.",
    )
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_base_model(create_fn, checkpoint_path: str, device: torch.device):
    model = create_fn()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def evaluate_model(model_name: str, device: torch.device, args) -> list[dict]:
    create_fn, checkpoint_path, scaler_path, thread_state = MODEL_CONFIGS[model_name]
    model = load_base_model(create_fn, checkpoint_path, device)
    scaler = joblib.load(scaler_path)

    # Convolutional models receive their trained sequence length. Recurrent
    # models receive one sample at a time and carry state between samples.
    sequence_length = 1 if thread_state else CHUNK_SIZES[model_name]
    _, _, test_loaders = build_dataloaders(
        data_root=args.data_root,
        scaler=scaler,
        chunk=sequence_length,
        num_workers=args.num_workers,
        load_train=False,
        load_val=False,
        load_test=True,
    )
    if not test_loaders:
        raise RuntimeError(f"No test data was loaded for {model_name}.")

    rows = []
    print(f"\n{model_name.upper()} ({checkpoint_path})")
    for cell_name, loader in zip(TEST_CELLS, test_loaders):
        metrics = evaluate(
            model,
            loader,
            device=device,
            thread_state=thread_state,
        )
        row = {"model": model_name, "scope": cell_name, **metrics}
        rows.append(row)
        print(
            f"  {cell_name:10s} | MSE={metrics['mse']:.6f} "
            f"MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} "
            f"R2={metrics['r2']:.6f} MaxErr={metrics['max_error']:.6f}"
        )

    # This is a macro average: every test cell contributes equally. MaxErr is
    # the worst error across cells rather than the mean of their maxima.
    summary = {
        name: sum(row[name] for row in rows) / len(rows)
        for name in METRIC_NAMES
        if name != "max_error"
    }
    summary["max_error"] = max(row["max_error"] for row in rows)
    summary_row = {"model": model_name, "scope": "macro_average", **summary}
    rows.append(summary_row)
    print(
        f"  {'AVERAGE':10s} | MSE={summary['mse']:.6f} "
        f"MAE={summary['mae']:.6f} RMSE={summary['rmse']:.6f} "
        f"R2={summary['r2']:.6f} MaxErr={summary['max_error']:.6f}"
    )

    del model, scaler, test_loaders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def write_csv(rows: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "scope", *METRIC_NAMES))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nMetrics saved to {path}")


def main(args_list=None) -> list[dict]:
    args = get_parser().parse_args(args_list)
    device = resolve_device(args.device)
    print(f"Evaluating base models on {device}.")

    rows = []
    for model_name in args.models:
        rows.extend(evaluate_model(model_name, device, args))

    if args.csv_output:
        write_csv(rows, args.csv_output)
    return rows


if __name__ == "__main__":
    main()
