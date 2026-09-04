#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an existing SOH training module with portable data paths."
    )
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_training_module(path: Path):
    specification = importlib.util.spec_from_file_location("portable_train_soh", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def main() -> None:
    args = parse_args()
    module = load_training_module(args.script.resolve())
    original_data_loader = module.DataLoader

    def portable_data_loader(*loader_args, **loader_kwargs):
        # The original lambda-based collate function cannot be serialized by
        # Windows worker processes. Default collation produces the same tensors.
        loader_kwargs.pop("collate_fn", None)
        loader_kwargs["num_workers"] = 0
        loader_kwargs.pop("prefetch_factor", None)
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs["pin_memory"] = False
        return original_data_loader(*loader_args, **loader_kwargs)

    module.DataLoader = portable_data_loader
    forwarded = [
        str(args.script),
        "--config",
        str(args.config.resolve()),
        "--data-root",
        str(args.data_root.resolve()),
        "--out-root",
        str(args.output_root.resolve()),
        "--run-id",
        args.run_id,
    ]
    if args.smoke_test:
        forwarded.append("--smoke-test")
    sys.argv = forwarded
    module.main()


if __name__ == "__main__":
    main()
