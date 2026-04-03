#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/florianr/MG_Farm/1_Scripts")
SIM_ENV_DIR = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "4_simulation_environment"
sys.path.append(str(SIM_ENV_DIR))
from robustness_common import apply_measurement_scenario, build_online_aux_features, load_cell_dataframe  # noqa: E402


MODELS = {
    "SOC_1.7.0.0": {
        "config": ROOT / "DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/train_soc.yaml",
        "ckpt": ROOT / "DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/soc_epoch0002_rmse0.01488.pt",
        "scaler": ROOT / "DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/scaler_robust.joblib",
        "train_script": ROOT / "DL_Models/LFP_SOC_SOH_Model/1_training/1.7.0.0/scripts/train_soc.py",
        "class_name": "GRUMLP",
        "color": "#08bdba",
    },
    "SOC_1.7.1.0_intermediate": {
        "config": ROOT / "DL_Models/LFP_SOC_SOH_Model/1_training/1.7.1.0/config/train_soc_best_from_hpt.yaml",
        "ckpt": ROOT / "DL_Models/LFP_SOC_SOH_Model/1_training/1.7.1.0/outputs/checkpoints/soc_epoch0012_rmse0.01371.pt",
        "scaler": ROOT / "DL_Models/LFP_SOC_SOH_Model/1_training/1.7.1.0/outputs/scaler_robust.joblib",
        "train_script": ROOT / "DL_Models/LFP_SOC_SOH_Model/1_training/1.7.1.0/scripts/train_soc.py",
        "class_name": "GRUMLP",
        "color": "#d95f02",
    },
}


SCENARIOS = {
    "baseline": {"scenario": "baseline"},
    "current_bias_3pct": {"scenario": "current_offset", "current_offset_pct": 0.03},
    "current_noise_high": {"scenario": "current_noise", "current_noise_std": 0.15},
    "voltage_noise": {"scenario": "voltage_noise", "voltage_noise_std": 0.01},
}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_feature_dataframe(cell: str, scenario_name: str, scenario_cfg: dict) -> pd.DataFrame:
    data_root = "/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE"
    df = load_cell_dataframe(data_root, cell)
    required = ["Testtime[s]", "Voltage[V]", "Current[A]", "Temperature[°C]", "SOC", "SOH"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)

    args = SimpleNamespace(
        seed=42,
        current_offset_a=None,
        current_offset_pct=None,
        voltage_offset_v=None,
        temp_offset_c=None,
        current_noise_std=None,
        voltage_noise_std=None,
        temp_noise_std=None,
        temp_constant=None,
        quantize_current_a=None,
        quantize_voltage_v=None,
        quantize_temp_c=None,
        spike_channel="Voltage[V]",
        spike_magnitude=None,
        spike_period=None,
        spike_prob=None,
        soc_init_error=0.0,
        missing_gap_seconds=0.0,
        missing_samples_every=None,
        missing_samples_pct=None,
        irregular_dt_jitter=None,
        downsample_hz=None,
        drop_pct=None,
        drop_segment_len=None,
    )
    for k, v in scenario_cfg.items():
        setattr(args, k, v)

    df_s, meta = apply_measurement_scenario(df, scenario_cfg["scenario"], args)
    freeze_mask = np.asarray(meta["freeze_mask"], dtype=bool)
    soc_delta = float(meta.get("soc_init_delta", 0.0))

    df_s = build_online_aux_features(
        df=df_s,
        freeze_mask=freeze_mask,
        current_sign=1.0,
        v_max=3.65,
        v_tol=0.02,
        cv_seconds=300.0,
        nominal_capacity_ah=1.8,
        initial_soc_delta=soc_delta,
    )
    if "_dt_s_online" in df_s.columns:
        df_s["dt_s"] = df_s["_dt_s_online"].astype(np.float32)
    return df_s


def stateful_chunk_predict(model, Xs: np.ndarray, chunk: int, device: torch.device):
    preds = np.empty(len(Xs), dtype=np.float32)
    state = None
    with torch.inference_mode():
        for start in range(0, len(Xs), chunk):
            end = min(start + chunk, len(Xs))
            xb = torch.from_numpy(Xs[start:end]).unsqueeze(0).to(device)
            out, state = model.gru(xb, state)
            flat = out.reshape(-1, out.shape[-1])
            y_all = model.mlp(flat).reshape(out.shape[0], out.shape[1]).detach().cpu().numpy()
            preds[start:end] = y_all.squeeze(0)
    return preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "p95_error": float(np.percentile(abs_err, 95)),
        "max_abs_error": float(np.max(abs_err)),
        "bias": float(np.mean(err)),
    }


def load_model(spec: dict, device: torch.device):
    cfg = yaml.safe_load(Path(spec["config"]).read_text())
    mod = load_module(Path(spec["train_script"]), f"{Path(spec['train_script']).stem}_{Path(spec['ckpt']).stem}")
    model_cls = getattr(mod, spec["class_name"])
    model = model_cls(
        in_features=len(cfg["model"]["features"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.05)),
    ).to(device)
    state = torch.load(spec["ckpt"], map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return cfg, model


def plot_summary(out_dir: Path, scenario_order: list[str], summary: dict):
    models = list(MODELS.keys())
    x = np.arange(len(scenario_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model_name in enumerate(models):
        vals = [summary["results"][sc][model_name]["mae"] for sc in scenario_order]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=model_name, color=MODELS[model_name]["color"])
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_order, rotation=15, ha="right")
    ax.set_ylabel("MAE [-]")
    ax.set_title("SOC intermediate comparison on full-cell scenarios")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mae_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model_name in enumerate(models):
        vals = [summary["delta_vs_baseline"][sc][model_name]["delta_mae"] for sc in scenario_order if sc != "baseline"]
        xpos = np.arange(len(scenario_order) - 1)
        ax.bar(xpos + (i - 0.5) * width, vals, width=width, label=model_name, color=MODELS[model_name]["color"])
    ax.set_xticks(np.arange(len(scenario_order) - 1))
    ax.set_xticklabels([sc for sc in scenario_order if sc != "baseline"], rotation=15, ha="right")
    ax.set_ylabel(r"$\Delta$MAE [-]")
    ax.set_title("Scenario penalty relative to baseline")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "delta_mae_comparison.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluate current SOC_1.7.1.0 checkpoint against SOC_1.7.0.0 on full-cell scenarios.")
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()))
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--out_dir", default=str(ROOT / "DL_Models/LFP_SOC_SOH_Model/3_test/2026-03-31_soc1710_intermediate_check"))
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    active_models = {name: MODELS[name] for name in args.models}

    summary = {
        "cell": args.cell,
        "device": args.device,
        "checkpoint_under_test": str(MODELS["SOC_1.7.1.0_intermediate"]["ckpt"]),
        "models": list(active_models.keys()),
        "scenarios": list(args.scenarios),
        "results": {},
        "delta_vs_baseline": {},
    }

    loaded = {name: load_model(spec, device) for name, spec in active_models.items()}

    active_scenarios = {name: SCENARIOS[name] for name in args.scenarios}

    for scenario_name, scenario_cfg in active_scenarios.items():
        print(f"[SCENARIO] {scenario_name}", flush=True)
        df = build_feature_dataframe(args.cell, scenario_name, scenario_cfg)
        y_true = df["SOC"].to_numpy(dtype=np.float32)
        summary["results"][scenario_name] = {}

        for model_name, spec in active_models.items():
            print(f"  [MODEL] {model_name}", flush=True)
            cfg, model = loaded[model_name]
            features = list(cfg["model"]["features"])
            work = df.copy()
            for col in features:
                work[col] = work[col].ffill().bfill()
            X = work[features].to_numpy(dtype=np.float32)
            Xs = joblib.load(spec["scaler"]).transform(X).astype(np.float32)
            chunk = int(cfg["training"]["seq_chunk_size"])
            pred = stateful_chunk_predict(model, Xs, chunk, device)
            metrics = compute_metrics(y_true, pred)
            summary["results"][scenario_name][model_name] = metrics

    baseline = summary["results"]["baseline"]
    for scenario_name in active_scenarios:
        if scenario_name == "baseline":
            continue
        summary["delta_vs_baseline"][scenario_name] = {}
        for model_name in active_models:
            mae = summary["results"][scenario_name][model_name]["mae"]
            base_mae = baseline[model_name]["mae"]
            rmse = summary["results"][scenario_name][model_name]["rmse"]
            base_rmse = baseline[model_name]["rmse"]
            summary["delta_vs_baseline"][scenario_name][model_name] = {
                "delta_mae": float(mae - base_mae),
                "delta_rmse": float(rmse - base_rmse),
            }

    # Direct pairwise comparison of old vs new
    pairwise = {}
    if {"SOC_1.7.0.0", "SOC_1.7.1.0_intermediate"}.issubset(active_models.keys()):
        for scenario_name in active_scenarios:
            old = summary["results"][scenario_name]["SOC_1.7.0.0"]
            new = summary["results"][scenario_name]["SOC_1.7.1.0_intermediate"]
            pairwise[scenario_name] = {
                "mae_improvement": float(old["mae"] - new["mae"]),
                "rmse_improvement": float(old["rmse"] - new["rmse"]),
                "p95_improvement": float(old["p95_error"] - new["p95_error"]),
            }
        summary["pairwise_old_minus_new"] = pairwise

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if len(active_models) == 2:
        plot_summary(out_dir, list(active_scenarios.keys()), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
