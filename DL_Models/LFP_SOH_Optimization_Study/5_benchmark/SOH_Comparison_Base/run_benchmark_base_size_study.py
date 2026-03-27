#!/usr/bin/env python3
import argparse
import importlib.util
import inspect
import json
import math
import os
import time
import typing as T
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import load as joblib_load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path("/home/florianr/MG_Farm/1_Scripts")
STUDY_ROOT = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study"
SPECS_PATH = STUDY_ROOT / "base_size_study_specs.json"


def load_specs() -> dict:
    with open(SPECS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_train_module(train_py: Path):
    spec = importlib.util.spec_from_file_location("train_soh_module", str(train_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def expand_env_with_defaults(val: str) -> str:
    if not isinstance(val, str):
        return val
    out = val
    while "${" in out and ":-" in out:
        start = out.find("${")
        end = out.find("}", start)
        if start == -1 or end == -1:
            break
        expr = out[start + 2:end]
        if ":-" in expr:
            var, default = expr.split(":-", 1)
            env_val = os.getenv(var)
            repl = env_val if env_val not in (None, "") else default
        else:
            repl = os.getenv(expr, "")
        out = out[:start] + repl + out[end + 1:]
    return os.path.expandvars(out)


def expand_features_for_sampling(base_features: T.List[str], sampling_cfg: dict) -> T.List[str]:
    if not sampling_cfg or not sampling_cfg.get("enabled", False):
        return list(base_features)
    feature_aggs = sampling_cfg.get("feature_aggs", ["mean"])
    return [f"{feat}_{agg}" for feat in base_features for agg in feature_aggs]


def load_cell_parquet(data_root: str, cell: str) -> pd.DataFrame:
    path = os.path.join(data_root, f"df_FE_{cell.split('_')[-1]}.parquet")
    if not os.path.exists(path):
        path = os.path.join(data_root, f"df_FE_{cell}.parquet")
    if not os.path.exists(path):
        cid = cell[-3:]
        alt = os.path.join(data_root, f"df_FE_C{cid}.parquet")
        if os.path.exists(alt):
            path = alt
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not locate parquet for cell {cell} in {data_root}")
    return pd.read_parquet(path)


def aggregate_hourly(df: pd.DataFrame, base_features: T.List[str], target: str, sampling_cfg: dict) -> pd.DataFrame:
    if not sampling_cfg or not sampling_cfg.get("enabled", False):
        return df
    if "Testtime[s]" not in df.columns:
        return df
    interval = int(sampling_cfg.get("interval_seconds", 3600))
    feature_aggs = sampling_cfg.get("feature_aggs", ["mean"])
    target_agg = sampling_cfg.get("target_agg", "last")

    cols = list(dict.fromkeys(base_features + [target, "Testtime[s]"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=base_features + [target, "Testtime[s]"]).copy()
    if work.empty:
        return work
    work = work.sort_values("Testtime[s]")
    work["_bin"] = (work["Testtime[s]"] // interval).astype(np.int64)
    agg_spec = {feat: feature_aggs for feat in base_features}
    agg_spec[target] = [target_agg]
    out = work.groupby("_bin", sort=True).agg(agg_spec)
    out.columns = [
        target if col[0] == target else f"{col[0]}_{col[1]}"
        for col in out.columns
    ]
    return out.reset_index(drop=True)


def build_model(cfg: dict, train_mod, in_features: int) -> torch.nn.Module:
    mtype = str(cfg["model"].get("type", "")).lower()
    if "lstm" in mtype:
        return train_mod.SOH_LSTM_Seq2Seq(
            in_features=in_features,
            embed_size=int(cfg["model"].get("embed_size", 96)),
            hidden_size=int(cfg["model"]["hidden_size"]),
            mlp_hidden=int(cfg["model"]["mlp_hidden"]),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            res_blocks=int(cfg["model"].get("res_blocks", 2)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.15)),
        )
    if "gru" in mtype:
        return train_mod.SOH_GRU_Seq2Seq(
            in_features=in_features,
            embed_size=int(cfg["model"].get("embed_size", 96)),
            hidden_size=int(cfg["model"]["hidden_size"]),
            mlp_hidden=int(cfg["model"]["mlp_hidden"]),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            res_blocks=int(cfg["model"].get("res_blocks", 2)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.15)),
        )
    if "tcn" in mtype:
        dilations = cfg["model"].get("dilations") or [1, 2, 4, 8]
        return train_mod.CausalTCN_SOH(
            in_features=in_features,
            hidden_size=int(cfg["model"]["hidden_size"]),
            mlp_hidden=int(cfg["model"]["mlp_hidden"]),
            kernel_size=int(cfg["model"].get("kernel_size", 3)),
            dilations=dilations,
            dropout=float(cfg["model"].get("dropout", 0.05)),
        )
    if "cnn" in mtype:
        cls = train_mod.SOH_CNN_Seq2Seq
        kwargs = dict(
            in_features=in_features,
            hidden_size=int(cfg["model"].get("hidden_size", 128)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 96)),
            kernel_size=int(cfg["model"].get("kernel_size", 5)),
            dilations=cfg["model"].get("dilations"),
            num_blocks=int(cfg["model"].get("num_blocks", 4)),
            dropout=float(cfg["model"].get("dropout", 0.15)),
        )
        if "output_kernel_size" in inspect.signature(cls).parameters:
            kwargs["output_kernel_size"] = int(cfg["model"].get("output_kernel_size", 1))
        return cls(**kwargs)
    raise ValueError(f"Unsupported model type: {cfg['model'].get('type')}")


def predict_stateful_rnn(model, x_scaled: np.ndarray, chunk: int, device: torch.device) -> np.ndarray:
    preds = []
    state = None
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_scaled), chunk):
            end = min(start + chunk, len(x_scaled))
            xb = torch.from_numpy(x_scaled[start:end]).unsqueeze(0).to(device)
            y_seq, state = model(xb, state=state, return_state=True)
            preds.append(y_seq.squeeze(0).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_causal_buffer(model, x_scaled: np.ndarray, chunk: int, device: torch.device) -> np.ndarray:
    preds = []
    context = None
    rf = int(getattr(model, "receptive_field", 1))
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_scaled), chunk):
            end = min(start + chunk, len(x_scaled))
            xb = x_scaled[start:end]
            if rf > 1:
                xb_in = xb if context is None else np.concatenate([context, xb], axis=0)
                context = xb_in[-(rf - 1):].copy()
            else:
                xb_in = xb
            xb_t = torch.from_numpy(xb_in).unsqueeze(0).to(device)
            y_seq = model(xb_t).squeeze(0).detach().cpu().numpy()
            if rf > 1:
                y_seq = y_seq[-len(xb):]
            preds.append(y_seq)
    return np.concatenate(preds, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def plot_family_predictions(cell: str, y_true: np.ndarray, preds: dict, out_path: Path) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, color="black", linewidth=1.2, alpha=0.9, label="true")
    for label, y_pred in preds.items():
        plt.plot(y_pred, linewidth=0.9, alpha=0.8, label=label)
    plt.title(f"{cell} prediction comparison")
    plt.xlabel("Sample index")
    plt.ylabel("SOH")
    plt.ylim(0.0, 1.0)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_family_tradeoff(df: pd.DataFrame, family: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for metric, ax in (("mae", axes[0]), ("rmse", axes[1])):
        ax.plot(df["params"], df[metric], marker="o", linewidth=1.2)
        for _, row in df.iterrows():
            ax.annotate(row["tag"], (row["params"], row[metric]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xlabel("Parameters")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{family} {metric.upper()} vs params")
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def family_model_dir(family: str, version: str, tag: str) -> Path:
    return STUDY_ROOT / "2_models" / family / "Base" / f"{version}_{tag}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the SOH base-size-study models on one shared test cell.")
    ap.add_argument("--study-ts", type=str, default=None, help="Optional training timestamp to store in benchmark metadata.")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--cell", type=str, default=None, help="Override test cell.")
    args = ap.parse_args()

    specs = load_specs()
    cell = args.cell or specs["test_cell"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    default_root = STUDY_ROOT / "5_benchmark" / "SOH_Comparison_Base" / "results"
    out_dir = Path(args.out_dir) if args.out_dir else default_root / f"RESULTS_{ts}"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_rows = []
    prediction_rows = []

    for family_spec in specs["families"]:
        family = family_spec["family"]
        family_preds = {}
        family_rows = []

        first_model_dir = family_model_dir(family, family_spec["variants"][0]["version"], family_spec["variants"][0]["tag"])
        with open(first_model_dir / "config" / "train_soh.yaml", "r", encoding="utf-8") as f:
            ref_cfg = yaml.safe_load(f)
        data_root = expand_env_with_defaults(ref_cfg["paths"]["data_root"])
        base_features = ref_cfg["model"]["features"]
        target = ref_cfg.get("training", {}).get("target", "SOH")
        sampling_cfg = ref_cfg.get("sampling", {})
        features = expand_features_for_sampling(base_features, sampling_cfg)
        df = load_cell_parquet(data_root, cell)
        df = aggregate_hourly(df, base_features, target, sampling_cfg)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target]).reset_index(drop=True)
        x_raw = df[features].to_numpy(dtype=np.float32)
        y_true = df[target].to_numpy(dtype=np.float32)

        for variant in family_spec["variants"]:
            model_dir = family_model_dir(family, variant["version"], variant["tag"])
            cfg_path = model_dir / "config" / "train_soh.yaml"
            train_py = model_dir / "scripts" / "train_soh.py"
            ckpt_path = model_dir / "checkpoints" / "best_model.pt"
            scaler_path = model_dir / "scaler_robust.joblib"

            if not all(path.exists() for path in (cfg_path, train_py, ckpt_path, scaler_path)):
                raise FileNotFoundError(f"Incomplete model folder: {model_dir}")

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            train_mod = load_train_module(train_py)
            sampling_cfg = cfg.get("sampling", {})
            features = expand_features_for_sampling(cfg["model"]["features"], sampling_cfg)
            chunk = int(cfg["training"].get("seq_chunk_size", 168))
            scaler = joblib_load(scaler_path)
            x_scaled = scaler.transform(x_raw).astype(np.float32)

            model = build_model(cfg, train_mod, in_features=len(features)).to(device)
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("model_state_dict", state)
            model.load_state_dict(state_dict)

            if family in ("LSTM", "GRU"):
                y_pred = predict_stateful_rnn(model, x_scaled, chunk, device)
            else:
                y_pred = predict_causal_buffer(model, x_scaled, chunk, device)

            metrics = compute_metrics(y_true, y_pred)
            params = int(sum(p.numel() for p in model.parameters()))
            hidden_size = int(cfg["model"].get("hidden_size", 0))
            family_rows.append({
                "family": family,
                "version": variant["version"],
                "tag": variant["tag"],
                "role": variant["role"],
                "hidden_size": hidden_size,
                "params": params,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "model_dir": str(model_dir.relative_to(ROOT)),
            })
            family_preds[f"{variant['version']} {variant['tag']}"] = y_pred

            prediction_rows.append({
                "family": family,
                "version": variant["version"],
                "tag": variant["tag"],
                "role": variant["role"],
                "cell": cell,
                "samples": len(y_true),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "params": params,
                "hidden_size": hidden_size,
                "model_dir": str(model_dir.relative_to(ROOT)),
            })

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        family_df = pd.DataFrame(family_rows).sort_values("hidden_size")
        family_df.to_csv(out_dir / f"{family.lower()}_summary.csv", index=False)
        summary_rows.extend(family_rows)
        plot_family_predictions(cell, y_true, family_preds, plots_dir / f"{family.lower()}_{cell}_predictions.png")
        plot_family_tradeoff(family_df, family, plots_dir / f"{family.lower()}_tradeoff.png")

    summary_df = pd.DataFrame(summary_rows).sort_values(["family", "hidden_size"])
    predictions_df = pd.DataFrame(prediction_rows).sort_values(["family", "hidden_size"])
    summary_df.to_csv(out_dir / "all_models_summary.csv", index=False)
    predictions_df.to_csv(out_dir / "metrics_by_model.csv", index=False)

    meta = {
        "generated_at": ts,
        "study_timestamp": args.study_ts,
        "device": str(device),
        "specs_path": str(SPECS_PATH.relative_to(ROOT)),
        "test_cell": cell,
        "notes": "Each family is benchmarked on the same test cell using its copied 2_models/Base size-study folders.",
    }
    with open(out_dir / "benchmark_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
