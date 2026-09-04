#!/usr/bin/env python3
"""Render plots from a completed BATMM numerical result folder."""

from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def import_plot_functions():
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_stub = types.ModuleType("torch")

        def inference_mode():
            return lambda function: function

        torch_stub.inference_mode = inference_mode
        sys.modules["torch"] = torch_stub

    from run_batmm_base_comparison import (
        MODEL_ORDER,
        configure_plot_style,
        plot_aggregation_comparison,
        plot_baseline_summary,
        plot_cell_trajectories,
        plot_per_cell_errors,
        plot_single_cell_trajectory,
        write_results_summary,
    )

    return {
        "model_order": MODEL_ORDER,
        "configure": configure_plot_style,
        "summary": plot_baseline_summary,
        "trajectories": plot_cell_trajectories,
        "single": plot_single_cell_trajectory,
        "per_cell": plot_per_cell_errors,
        "aggregation": plot_aggregation_comparison,
        "write_summary": write_results_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render plots from BATMM CSV results.")
    parser.add_argument(
        "result_dir",
        type=Path,
        nargs="?",
        default=None,
        help="BATMM_RESULTS_* directory. Defaults to the latest complete result.",
    )
    return parser.parse_args()


def find_result_dir(requested: Path | None) -> Path:
    if requested is not None:
        result_dir = requested.resolve()
    else:
        candidates = sorted(
            (HERE / "results").glob("BATMM_RESULTS_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        result_dir = next(
            (
                path
                for path in candidates
                if (path / "metrics_summary.csv").is_file()
                and (path / "model_inventory.csv").is_file()
                and (path / "run_metadata.json").is_file()
            ),
            None,
        )
        if result_dir is None:
            raise FileNotFoundError("No complete BATMM_RESULTS_* folder was found.")
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)
    return result_dir


def load_predictions(result_dir: Path, model_order) -> tuple[list[str], dict]:
    metadata = json.loads((result_dir / "run_metadata.json").read_text(encoding="utf-8"))
    cells = metadata["test_cells"]
    frames = {}
    for model_name in model_order:
        for cell in cells:
            short_cell = cell.replace("df_FE_", "")
            path = result_dir / "predictions" / f"{model_name}_{short_cell}_predictions.csv"
            frames[(model_name, cell)] = pd.read_csv(path)
    return cells, frames


def main() -> None:
    args = parse_args()
    result_dir = find_result_dir(args.result_dir)
    functions = import_plot_functions()
    summary = pd.read_csv(result_dir / "metrics_summary.csv")
    inventory = pd.read_csv(result_dir / "model_inventory.csv")
    per_cell = pd.read_csv(result_dir / "metrics_by_cell.csv")
    cells, prediction_frames = load_predictions(result_dir, functions["model_order"])
    plots_dir = result_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    functions["configure"]()
    functions["summary"](summary, inventory, plots_dir / "batmm_baseline_model_comparison.png")
    functions["trajectories"](cells, prediction_frames, plots_dir / "batmm_all_test_cell_trajectories.png")
    functions["single"](cells[0], prediction_frames, plots_dir / "batmm_C11_soh_trajectory.png")
    functions["per_cell"](per_cell, plots_dir / "batmm_per_cell_errors.png")
    functions["aggregation"](summary, plots_dir / "batmm_macro_vs_sample_weighted.png")
    functions["write_summary"](result_dir, summary, per_cell)

    render_metadata = {
        "rendered_at": datetime.now().astimezone().isoformat(),
        "result_dir": str(result_dir),
        "pandas": pd.__version__,
    }
    (result_dir / "render_metadata.json").write_text(
        json.dumps(render_metadata, indent=2), encoding="utf-8"
    )
    print(f"Rendered plots in {plots_dir}")


if __name__ == "__main__":
    main()
