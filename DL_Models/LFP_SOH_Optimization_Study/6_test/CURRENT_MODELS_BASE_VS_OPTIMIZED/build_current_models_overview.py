#!/usr/bin/env python3
"""Build one current overview for selected final models (Base vs Optimized).

Optimized = structured pruning + finetune + int8 quantization
"""
import math
import os
import re
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import load as joblib_load
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study")
OUT_ROOT = ROOT / "6_test" / "CURRENT_MODELS_BASE_VS_OPTIMIZED"
PLOT_DIR = OUT_ROOT / "base_vs_optimized"

MODEL_SPECS = [
    {
        "family": "LSTM",
        "version": "0.1.2.4",
        "base_dir": ROOT / "2_models" / "LSTM" / "Base" / "0.1.2.4",
        "opt_dir": ROOT / "2_models" / "LSTM" / "PrunedFT" / "0.1.2.4_s20_struct_ft",
        "quant_file": ROOT / "2_models" / "LSTM" / "Quantized" / "0.1.2.4_s20_struct_ft_int8" / "quantized_state_dict.pt",
        "base_ckpt_name": "best_model.pt",
        "opt_ckpt_name": "best_model_finetuned.pt",
    },
    {
        "family": "GRU",
        "version": "0.3.1.2",
        "base_dir": ROOT / "2_models" / "GRU" / "Base" / "0.3.1.2",
        "opt_dir": ROOT / "2_models" / "GRU" / "PrunedFT" / "0.3.1.2_s20_struct_ft",
        "quant_file": ROOT / "2_models" / "GRU" / "Quantized" / "0.3.1.2_s20_struct_ft_int8" / "quantized_state_dict.pt",
        "base_ckpt_name": "best_model.pt",
        "opt_ckpt_name": "best_model_finetuned.pt",
    },
    {
        "family": "TCN",
        "version": "0.2.2.2",
        "base_dir": ROOT / "2_models" / "TCN" / "Base" / "0.2.2.2",
        "opt_dir": ROOT / "2_models" / "TCN" / "PrunedFT" / "0.2.2.2_s20_struct_ft",
        "quant_file": ROOT / "2_models" / "TCN" / "Quantized" / "0.2.2.2_s20_struct_ft_int8" / "quantized_state_dict.pt",
        "base_ckpt_name": "best_model.pt",
        "opt_ckpt_name": "best_model_finetuned.pt",
    },
    {
        "family": "CNN",
        "version": "0.4.2.1_hp",
        "base_dir": ROOT / "2_models" / "CNN" / "Base" / "0.4.2.1_hp",
        "opt_dir": ROOT / "2_models" / "CNN" / "PrunedFT" / "0.4.2.1_hp_s20_struct_ft",
        "quant_file": ROOT / "2_models" / "CNN" / "Quantized" / "0.4.2.1_hp_s20_struct_ft_int8" / "quantized_state_dict.pt",
        "base_ckpt_name": "best_model.pt",
        "opt_ckpt_name": "best_model_finetuned.pt",
    },
]


def expand_env_with_defaults(val: str) -> str:
    pattern = re.compile(r"\$\{([^:}]+):-([^}]+)\}")

    def repl(match):
        var, default = match.group(1), match.group(2)
        env_val = os.getenv(var)
        return env_val if env_val not in (None, "") else default

    if not isinstance(val, str):
        return val
    return os.path.expandvars(pattern.sub(repl, val))


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_module(train_py: Path):
    spec = importlib.util.spec_from_file_location("train_soh_module", str(train_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def load_cell_dataframe(data_root: Path, cell: str) -> pd.DataFrame:
    path = data_root / f"df_FE_{cell.split('_')[-1]}.parquet"
    if not path.exists():
        path = data_root / f"df_FE_{cell}.parquet"
    if not path.exists():
        cid = cell[-3:]
        alt = data_root / f"df_FE_C{cid}.parquet"
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(f"Could not locate parquet for cell {cell} in {data_root}")
    return pd.read_parquet(path)


def build_model(cfg: dict, train_mod):
    mtype = str(cfg["model"].get("type", "")).lower()
    base_features = cfg["model"]["features"]
    sampling_cfg = cfg.get("sampling", {})
    if hasattr(train_mod, "expand_features_for_sampling"):
        features = train_mod.expand_features_for_sampling(base_features, sampling_cfg)
    else:
        features = base_features

    if "lstm" in mtype:
        model = train_mod.SOH_LSTM_Seq2Seq(
            in_features=len(features),
            embed_size=int(cfg["model"].get("embed_size", 96)),
            hidden_size=int(cfg["model"].get("hidden_size", 160)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 128)),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            res_blocks=int(cfg["model"].get("res_blocks", 2)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.15)),
        )
        mode = "lstm"
    elif "gru" in mtype:
        model = train_mod.SOH_GRU_Seq2Seq(
            in_features=len(features),
            embed_size=int(cfg["model"].get("embed_size", 96)),
            hidden_size=int(cfg["model"].get("hidden_size", 160)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 128)),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            res_blocks=int(cfg["model"].get("res_blocks", 2)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.15)),
        )
        mode = "gru"
    elif "tcn" in mtype:
        dilations = cfg["model"].get("dilations")
        if not dilations:
            num_layers = int(cfg["model"].get("num_layers", 4))
            dilations = [2 ** i for i in range(num_layers)]
        model = train_mod.CausalTCN_SOH(
            in_features=len(features),
            hidden_size=int(cfg["model"].get("hidden_size", 128)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 96)),
            kernel_size=int(cfg["model"].get("kernel_size", 3)),
            dilations=[int(d) for d in dilations],
            dropout=float(cfg["model"].get("dropout", 0.05)),
        )
        mode = "tcn"
    elif "cnn" in mtype:
        dilations = cfg["model"].get("dilations")
        model = train_mod.SOH_CNN_Seq2Seq(
            in_features=len(features),
            hidden_size=int(cfg["model"].get("hidden_size", 128)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 96)),
            kernel_size=int(cfg["model"].get("kernel_size", 5)),
            dilations=[int(d) for d in dilations] if dilations is not None else None,
            num_blocks=int(cfg["model"].get("num_blocks", 4)),
            dropout=float(cfg["model"].get("dropout", 0.15)),
        )
        mode = "cnn"
    else:
        raise ValueError(f"Unsupported model type: {cfg['model'].get('type')}")

    return model, features, mode


def predict_stateful(model, mode: str, X_scaled: np.ndarray, seq_len: int, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        if mode == "lstm":
            preds = []
            h = None
            c = None
            for start in range(0, len(X_scaled), seq_len):
                end = min(start + seq_len, len(X_scaled))
                xb = torch.from_numpy(X_scaled[start:end]).unsqueeze(0).to(device)
                if h is None:
                    y_seq, (h, c) = model(xb, state=None, return_state=True)
                else:
                    y_seq, (h, c) = model(xb, state=(h, c), return_state=True)
                preds.append(y_seq.squeeze(0).detach().cpu().numpy())
            return np.concatenate(preds, axis=0)

        if mode == "gru":
            preds = []
            h = None
            for start in range(0, len(X_scaled), seq_len):
                end = min(start + seq_len, len(X_scaled))
                xb = torch.from_numpy(X_scaled[start:end]).unsqueeze(0).to(device)
                if h is None:
                    y_seq, h = model(xb, state=None, return_state=True)
                else:
                    y_seq, h = model(xb, state=h, return_state=True)
                preds.append(y_seq.squeeze(0).detach().cpu().numpy())
            return np.concatenate(preds, axis=0)

        if mode in ("tcn", "cnn"):
            preds = []
            rf = int(getattr(model, "receptive_field", 1))
            context = None
            for start in range(0, len(X_scaled), seq_len):
                end = min(start + seq_len, len(X_scaled))
                xb = X_scaled[start:end]
                if rf > 1:
                    if context is not None:
                        xb_in = np.concatenate([context, xb], axis=0)
                    else:
                        xb_in = xb
                    context = xb_in[-(rf - 1):].copy()
                else:
                    xb_in = xb
                xb_t = torch.from_numpy(xb_in).unsqueeze(0).to(device)
                y_seq = model(xb_t)
                y_np = y_seq.squeeze(0).detach().cpu().numpy()
                if rf > 1:
                    y_np = y_np[-len(xb):]
                preds.append(y_np)
            return np.concatenate(preds, axis=0)

        raise ValueError(f"Unsupported mode: {mode}")


def weighted_mae_rmse(rows: List[Dict[str, float]]) -> Tuple[float, float]:
    df = pd.DataFrame(rows)
    w = df["samples"].astype(float)
    mae = float((df["mae"] * w).sum() / w.sum())
    rmse = float((df["rmse"] * w).sum() / w.sum())
    return mae, rmse


def eval_variant(model_dir: Path, ckpt: Path, device: torch.device):
    cfg = load_yaml(model_dir / "config" / "train_soh.yaml")
    train_mod = load_train_module(model_dir / "scripts" / "train_soh.py")
    model, features, mode = build_model(cfg, train_mod)
    model.to(device)

    state = torch.load(ckpt, map_location=device)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)

    scaler = joblib_load(model_dir / "scaler_robust.joblib")
    data_root = Path(expand_env_with_defaults(cfg["paths"]["data_root"]))
    target = cfg.get("training", {}).get("target", "SOH")
    base_features = cfg["model"]["features"]
    sampling_cfg = cfg.get("sampling", {})
    seq_len = int(cfg["training"]["seq_chunk_size"])
    test_cells = list(cfg.get("cells", {}).get("test", []))

    rows = []
    preds = {}
    for cell in test_cells:
        df = load_cell_dataframe(data_root, cell)
        if hasattr(train_mod, "maybe_aggregate_hourly"):
            df = train_mod.maybe_aggregate_hourly(df, base_features, target, sampling_cfg)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target]).reset_index(drop=True)
        X = df[features].to_numpy(dtype=np.float32)
        X_scaled = scaler.transform(X).astype(np.float32)
        y_true = df[target].to_numpy(dtype=np.float32)
        y_pred = predict_stateful(model, mode, X_scaled, seq_len, device)
        n = min(len(y_true), len(y_pred))
        y_true = y_true[:n]
        y_pred = y_pred[:n]
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        rows.append({"cell": cell, "mae": mae, "rmse": rmse, "samples": int(n)})
        preds[cell] = {"y_true": y_true, "y_pred": y_pred}

    params = int(sum(p.numel() for p in model.parameters()))
    return rows, preds, params


def plot_compare(
    cell: str,
    family: str,
    version: str,
    y_true: np.ndarray,
    y_base: np.ndarray,
    y_opt: np.ndarray,
    out_path: Path,
    hours_per_step: float = 1.0,
):
    n = min(len(y_true), len(y_base), len(y_opt))
    y_true = y_true[:n]
    y_base = y_base[:n]
    y_opt = y_opt[:n]

    step = max(1, n // 6000)
    x_hours = np.arange(0, n, step, dtype=np.float32) * float(hours_per_step)
    y_true_d = y_true[::step]
    y_base_d = y_base[::step]
    y_opt_d = y_opt[::step]

    plt.figure(figsize=(14, 5))
    plt.plot(y_true_d, label="SOH true", linewidth=1.2, color="black", alpha=0.8)
    plt.plot(y_base_d, label="Base pred", linewidth=1.0, color="#1f77b4", alpha=0.85)
    plt.plot(y_opt_d, label="Optimized pred", linewidth=1.0, color="#d62728", alpha=0.85)
    duration_h = float(n) * float(hours_per_step)
    plt.title(f"{family} {version} | {cell} | Base vs Optimized | {duration_h:.0f} h")
    plt.xlabel("Time in hours")
    plt.ylabel("SOH")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for old in PLOT_DIR.glob("*.png"):
        old.unlink()

    per_model_rows = []
    per_cell_rows = []

    for spec in MODEL_SPECS:
        base_ckpt = spec["base_dir"] / "checkpoints" / spec["base_ckpt_name"]
        opt_ckpt = spec["opt_dir"] / "checkpoints" / spec["opt_ckpt_name"]
        quant_file = spec["quant_file"]
        base_cfg = load_yaml(spec["base_dir"] / "config" / "train_soh.yaml")
        interval_seconds = int(base_cfg.get("sampling", {}).get("interval_seconds", 3600))
        hours_per_step = float(interval_seconds) / 3600.0 if interval_seconds > 0 else 1.0

        if not base_ckpt.exists():
            raise FileNotFoundError(f"Missing base checkpoint: {base_ckpt}")
        if not opt_ckpt.exists():
            raise FileNotFoundError(f"Missing optimized checkpoint: {opt_ckpt}")
        if not quant_file.exists():
            raise FileNotFoundError(f"Missing quantized file: {quant_file}")

        base_rows, base_preds, base_params = eval_variant(spec["base_dir"], base_ckpt, device)
        opt_rows, opt_preds, opt_params = eval_variant(spec["opt_dir"], opt_ckpt, device)

        base_mae, base_rmse = weighted_mae_rmse(base_rows)
        opt_mae, opt_rmse = weighted_mae_rmse(opt_rows)

        base_kb = base_params * 4 / 1024.0
        pruned_kb = opt_params * 4 / 1024.0
        quant_kb = quant_file.stat().st_size / 1024.0

        prune_red = (1.0 - (pruned_kb / base_kb)) * 100.0
        quant_red = (1.0 - (quant_kb / pruned_kb)) * 100.0
        total_red = (1.0 - (quant_kb / base_kb)) * 100.0
        mae_delta = opt_mae - base_mae
        mae_delta_pct = (mae_delta / base_mae) * 100.0 if base_mae != 0 else 0.0

        per_model_rows.append(
            {
                "model": spec["family"],
                "version": spec["version"],
                "base_params": base_params,
                "pruned_params": opt_params,
                "base_mae": base_mae,
                "base_rmse": base_rmse,
                "opt_mae": opt_mae,
                "opt_rmse": opt_rmse,
                "mae_delta": mae_delta,
                "mae_delta_percent": mae_delta_pct,
                "base_kb": base_kb,
                "pruned_kb": pruned_kb,
                "quant_kb": quant_kb,
                "pruning_reduction_percent": prune_red,
                "quantization_reduction_percent": quant_red,
                "total_reduction_percent": total_red,
                "optimized_version": f"{spec['version']}_optimized",
            }
        )

        base_by_cell = {r["cell"]: r for r in base_rows}
        opt_by_cell = {r["cell"]: r for r in opt_rows}
        common_cells = sorted(set(base_by_cell.keys()) & set(opt_by_cell.keys()))
        for cell in common_cells:
            b = base_by_cell[cell]
            o = opt_by_cell[cell]
            per_cell_rows.append(
                {
                    "model": spec["family"],
                    "version": spec["version"],
                    "cell": cell,
                    "base_mae": b["mae"],
                    "base_rmse": b["rmse"],
                    "opt_mae": o["mae"],
                    "opt_rmse": o["rmse"],
                    "samples": b["samples"],
                }
            )

            y_true = base_preds[cell]["y_true"]
            y_base = base_preds[cell]["y_pred"]
            y_opt = opt_preds[cell]["y_pred"]
            n = min(len(y_true), len(y_base), len(y_opt))
            out_plot = PLOT_DIR / f"{spec['family']}_{spec['version']}_{cell}_base_vs_optimized.png"
            plot_compare(
                cell,
                spec["family"],
                spec["version"],
                y_true[:n],
                y_base[:n],
                y_opt[:n],
                out_plot,
                hours_per_step=hours_per_step,
            )

    model_df = pd.DataFrame(per_model_rows).sort_values("model")
    cell_df = pd.DataFrame(per_cell_rows).sort_values(["model", "cell"])

    model_csv = OUT_ROOT / "CURRENT_MODELS_OVERVIEW.csv"
    cell_csv = OUT_ROOT / "CURRENT_MODELS_CELL_METRICS.csv"
    model_df.to_csv(model_csv, index=False)
    cell_df.to_csv(cell_csv, index=False)

    md_path = OUT_ROOT / "CURRENT_MODELS_OVERVIEW.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Current Models Overview (Base vs Optimized)\n\n")
        f.write("Optimized = **structured pruning + finetune + int8 quantization**.\n\n")
        f.write("Plots: `6_test/CURRENT_MODELS_BASE_VS_OPTIMIZED/base_vs_optimized`\n\n")
        f.write("| Model | Version | Base MAE | Opt MAE | MAE Delta | Base RMSE | Opt RMSE | Base kB | Pruned kB | Optimized kB (quant) | Pruning Reduktion | Quantization Reduktion | Gesamt Reduktion |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in model_df.iterrows():
            f.write(
                f"| {r['model']} | {r['version']} | {r['base_mae']:.6f} | {r['opt_mae']:.6f} | "
                f"{r['mae_delta']:+.6f} ({r['mae_delta_percent']:+.1f}%) | {r['base_rmse']:.6f} | {r['opt_rmse']:.6f} | "
                f"{r['base_kb']:.1f} | {r['pruned_kb']:.1f} | {r['quant_kb']:.1f} | "
                f"{r['pruning_reduction_percent']:.1f}% | {r['quantization_reduction_percent']:.1f}% | {r['total_reduction_percent']:.1f}% |\n"
            )

    print("Wrote:", md_path)
    print("Wrote:", model_csv)
    print("Wrote:", cell_csv)
    print("Plots:", PLOT_DIR)


if __name__ == "__main__":
    main()
