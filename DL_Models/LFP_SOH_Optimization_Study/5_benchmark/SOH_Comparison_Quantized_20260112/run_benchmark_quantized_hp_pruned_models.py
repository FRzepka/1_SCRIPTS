#!/usr/bin/env python3
"""Run quantized benchmark for HP base models against pruned+quantized variants."""
import argparse
import importlib.util
import sys
from pathlib import Path


BASE_SPECS = [
    ("LSTM", "0.1.2.3"),
    ("TCN", "0.2.2.1"),
    ("GRU", "0.3.1.1"),
    ("CNN", "0.4.1.1"),
]


def load_base_benchmark_module(base_script: Path):
    spec = importlib.util.spec_from_file_location("benchmark_quantized_base", str(base_script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def build_models(prune_pct: int):
    models = []
    for family, version in BASE_SPECS:
        models.append(
            {
                "key": f"{family}_{version}_hp",
                "label": f"{family} {version} hp (p{prune_pct}+int8)",
                "family": family,
                "base_dir": f"DL_Models/LFP_SOH_Optimization_Study/2_models/{family}/Base/{version}_hp",
                "quant_dir": f"DL_Models/LFP_SOH_Optimization_Study/2_models/{family}/Quantized/{version}_hp_p{prune_pct}_int8",
            }
        )
    return models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--cells", type=str, default=None)
    ap.add_argument("--prune-pct", type=int, default=50)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_script = script_dir / "run_benchmark_quantized_models.py"
    mod = load_base_benchmark_module(base_script)
    mod.MODELS = build_models(int(args.prune_pct))

    argv = [str(base_script)]
    if args.out_dir:
        argv += ["--out-dir", args.out_dir]
    if args.device:
        argv += ["--device", args.device]
    if args.cells:
        argv += ["--cells", args.cells]

    old_argv = list(sys.argv)
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
