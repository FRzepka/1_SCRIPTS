#!/usr/bin/env python3
"""Finetune structured-pruned paper models, requantize them, and run full-cell compares."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import typing as T
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.preprocessing import RobustScaler


ROOT = Path("/home/florianr/MG_Farm/1_Scripts")
CELL = "MGFarm_18650_C07"

SOC_ROOT = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "2_models" / "SOC_1.7.0.0" / "Pruned_Quantized_1.7.0.0_s30_struct_int8"
SOH_ROOT = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / "LSTM" / "Base" / "0.1.2.5_base_h160" / "Pruned_Quantized_0.1.2.5_base_h160_s30_struct_int8"

SOH_FINETUNE_PY = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "3_pruning" / "finetune_pruned_soh_model.py"
SOH_QUANT_PY = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "4_quantize" / "quantize_soh_model.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def write_json(path: Path, payload: dict) -> None:
    def _safe(v):
        if isinstance(v, dict):
            return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_best_checkpoint(ckpt_dir: Path) -> Path:
    best = None
    best_rmse = None
    for path in ckpt_dir.glob("*.pt"):
        import re
        match = re.search(r"rmse([0-9]+(?:\.[0-9]+)?)", path.name)
        if not match:
            continue
        rmse = float(match.group(1))
        if best is None or rmse < best_rmse:
            best = path
            best_rmse = rmse
    if best is None:
        pts = sorted(ckpt_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        best = pts[0]
    return best


def finetune_soc() -> dict:
    model_dir = SOC_ROOT / "pruned_model"
    out_dir = SOC_ROOT / "finetuned_model"
    cfg = load_yaml(model_dir / "config" / "train_soc.yaml")
    train_mod = load_module("soc_train_mod_ft", model_dir / "scripts" / "train_soc.py")

    cfg["paths"]["out_root"] = str(out_dir)
    cfg.setdefault("dataloader", {})
    cfg["dataloader"]["num_workers"] = 2
    cfg["dataloader"]["prefetch_factor"] = 2
    cfg["dataloader"]["persistent_workers"] = True
    cfg["dataloader"]["pin_memory"] = True

    features = cfg["model"]["features"]
    hidden_size = int(cfg["model"]["hidden_size"])
    mlp_hidden = int(cfg["model"]["mlp_hidden"])
    num_layers = int(cfg["model"].get("num_layers", 1))
    dropout = float(cfg["model"].get("dropout", 0.05))
    chunk = int(cfg["training"]["seq_chunk_size"])
    batch_size = int(cfg["training"].get("batch_size", 64))
    accum_steps = int(cfg["training"].get("accum_steps", 1))
    lr = 2e-4
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    epochs = 8
    val_interval = 1
    early_stopping = 4

    scaler = RobustScaler()
    train_loader, val_loader = train_mod.create_dataloaders(cfg, features, chunk, scaler, batch_size=batch_size)
    model = train_mod.GRUMLP(
        in_features=len(features),
        hidden_size=hidden_size,
        mlp_hidden=mlp_hidden,
        num_layers=num_layers,
        dropout=dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ckpt_path = model_dir / "checkpoints" / "soc_best_model_pruned.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    optimizer = train_mod.make_optimizer(model, lr=lr, weight_decay=weight_decay)
    amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config").mkdir(parents=True, exist_ok=True)
    (out_dir / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_dir / "config" / "train_soc.yaml", out_dir / "config" / "train_soc.yaml")
    shutil.copy2(model_dir / "scripts" / "train_soc.py", out_dir / "scripts" / "train_soc.py")

    base_metrics, _, _ = train_mod.eval_model(model, val_loader, device)
    best_rmse = float(base_metrics["rmse"])
    best_epoch = 0
    best_path = out_dir / "checkpoints" / "best_model_finetuned.pt"
    torch.save({"epoch": 0, "model_state_dict": model.state_dict(), "metrics": base_metrics}, best_path)
    patience = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_mod.train_one_epoch(
            model, train_loader, device, optimizer, amp_scaler, max_grad_norm, epoch, epochs, accum_steps=accum_steps
        )
        metrics, _, _ = train_mod.eval_model(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(train_loss), **metrics})
        print(f"[SOC FT {epoch:03d}/{epochs}] train_loss={train_loss:.6f} val_rmse={metrics['rmse']:.6f} val_mae={metrics['mae']:.6f}", flush=True)
        if metrics["rmse"] < best_rmse:
            best_rmse = float(metrics["rmse"])
            best_epoch = epoch
            patience = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, best_path)
        else:
            patience += 1
            if patience >= early_stopping:
                print(f"[SOC FT] early stopping at epoch {epoch}", flush=True)
                break

    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    final_metrics, _, _ = train_mod.eval_model(model, val_loader, device)
    final_path = out_dir / "checkpoints" / "final_model_finetuned.pt"
    torch.save({"epoch": best_epoch, "model_state_dict": model.state_dict(), "metrics": final_metrics, "history": history}, final_path)
    joblib_dump(scaler, out_dir / "scaler_robust.joblib")
    meta = {
        "source_model_dir": str(model_dir),
        "source_checkpoint": str(ckpt_path),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "baseline_metrics": base_metrics,
        "final_metrics": final_metrics,
    }
    write_json(out_dir / "finetune_meta.json", meta)
    return meta


def requantize_soc() -> dict:
    model_dir = SOC_ROOT / "finetuned_model"
    out_dir = SOC_ROOT / "quantized_finetuned_model"
    cfg = load_yaml(model_dir / "config" / "train_soc.yaml")
    train_mod = load_module("soc_train_mod_q", model_dir / "scripts" / "train_soc.py")

    model = train_mod.GRUMLP(
        in_features=len(cfg["model"]["features"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.05)),
    ).cpu().eval()
    ckpt_path = model_dir / "checkpoints" / "best_model_finetuned.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    qmodel = torch.ao.quantization.quantize_dynamic(model, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config").mkdir(parents=True, exist_ok=True)
    (out_dir / "scripts").mkdir(parents=True, exist_ok=True)
    qpath = out_dir / "quantized_state_dict.pt"
    torch.save(qmodel.state_dict(), qpath)
    shutil.copy2(model_dir / "config" / "train_soc.yaml", out_dir / "config" / "train_soc.yaml")
    shutil.copy2(model_dir / "scripts" / "train_soc.py", out_dir / "scripts" / "train_soc.py")
    shutil.copy2(model_dir / "scaler_robust.joblib", out_dir / "scaler_robust.joblib")
    meta = {
        "model_dir": str(model_dir),
        "checkpoint": str(ckpt_path),
        "quantized_modules": ["GRU", "Linear"],
        "quant_mode": "dynamic",
        "quant_scope": "full",
        "quantized_state_dict_bytes": int(qpath.stat().st_size),
    }
    write_json(out_dir / "quantize_meta.json", meta)
    return meta


def compare_soc_fullcell() -> dict:
    base_dir = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "2_models" / "SOC_1.7.0.0"
    finetuned_dir = SOC_ROOT / "finetuned_model"
    out_dir = SOC_ROOT / "fullcell_compare_C07"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_mod = load_module("soc_train_mod_cmp", finetuned_dir / "scripts" / "train_soc.py")
    cfg = load_yaml(finetuned_dir / "config" / "train_soc.yaml")
    df = train_mod.load_cell_dataframe(cfg["paths"]["data_root"], CELL)
    work = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Testtime[s]", "Voltage[V]", "Current[A]", "Temperature[°C]", "SOH", "SOC", "Q_c"]).reset_index(drop=True)
    times_h = work["Testtime[s]"].to_numpy(dtype=np.float64) / 3600.0

    def run_one(model_dir: Path, ckpt_path: Path, scaler_path: Path) -> np.ndarray:
        cfg_path = model_dir / "config" / "train_soc.yaml"
        if not cfg_path.exists():
            cfg_path = model_dir / "train_soc.yaml"
        cfg_local = load_yaml(cfg_path)
        features = cfg_local["model"]["features"]
        feat_df = train_mod.engineer_frame_for_scaler(work, features)
        scaler = joblib_load(scaler_path)
        X = scaler.transform(feat_df.to_numpy(dtype=np.float32)).astype(np.float32)
        model = train_mod.GRUMLP(
            in_features=len(features),
            hidden_size=int(cfg_local["model"]["hidden_size"]),
            mlp_hidden=int(cfg_local["model"]["mlp_hidden"]),
            num_layers=int(cfg_local["model"].get("num_layers", 1)),
            dropout=float(cfg_local["model"].get("dropout", 0.05)),
        ).cpu().eval()
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        preds = []
        h = None
        with torch.inference_mode():
            chunk = int(cfg_local["training"]["seq_chunk_size"])
            for start in range(0, len(X), chunk):
                end = min(start + chunk, len(X))
                xb = torch.from_numpy(X[start:end]).unsqueeze(0)
                y_seq, h = model.gru(xb, h)
                y = model.mlp(y_seq.reshape(-1, y_seq.shape[-1])).squeeze(-1).numpy()
                preds.append(y)
        return np.concatenate(preds, axis=0)

    base_ckpt = find_best_checkpoint(base_dir)
    base_pred = run_one(base_dir, base_ckpt, base_dir / "scaler_robust.joblib")
    ft_pred = run_one(finetuned_dir, finetuned_dir / "checkpoints" / "best_model_finetuned.pt", finetuned_dir / "scaler_robust.joblib")
    y_true = work["SOC"].to_numpy(dtype=np.float32)

    metrics = {
        "base_mae": float(np.mean(np.abs(base_pred - y_true))),
        "ft_mae": float(np.mean(np.abs(ft_pred - y_true))),
        "mae_vs_base": float(np.mean(np.abs(ft_pred - base_pred))),
        "max_abs_diff_vs_base": float(np.max(np.abs(ft_pred - base_pred))),
    }
    write_json(out_dir / "summary.json", metrics)

    plt.figure(figsize=(14, 6))
    plt.plot(times_h, y_true, label="Reference", lw=1.2)
    plt.plot(times_h, base_pred, label="Base", lw=1.0)
    plt.plot(times_h, ft_pred, label="Pruned+FT", lw=1.0)
    plt.xlabel("Time [h]")
    plt.ylabel("SOC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "soc_fullcell_compare.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(times_h, np.abs(base_pred - y_true), label="|Base-Ref|", lw=0.9)
    plt.plot(times_h, np.abs(ft_pred - y_true), label="|FT-Ref|", lw=0.9)
    plt.xlabel("Time [h]")
    plt.ylabel("Absolute error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "soc_fullcell_abs_error.png", dpi=180)
    plt.close()
    return metrics


def compare_soh_fullcell() -> dict:
    base_dir = SOH_MODEL_DIR = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / "LSTM" / "Base" / "0.1.2.5_base_h160"
    finetuned_dir = SOH_ROOT / "finetuned_model"
    out_dir = SOH_ROOT / "fullcell_compare_C07"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_mod = load_module("soh_train_mod_cmp", finetuned_dir / "scripts" / "train_soh.py")
    cfg = load_yaml(finetuned_dir / "config" / "train_soh.yaml")

    def load_cell_dataframe(data_root: str, cell: str) -> pd.DataFrame:
        path = Path(data_root) / f"df_FE_{cell.split('_')[-1]}.parquet"
        if not path.exists():
            path = Path(data_root) / f"df_FE_{cell}.parquet"
        if not path.exists():
            cid = cell[-3:]
            alt = Path(data_root) / f"df_FE_C{cid}.parquet"
            if alt.exists():
                path = alt
        return pd.read_parquet(path)

    def aggregate_hourly(df: pd.DataFrame, base_features: list[str], target: str, sampling_cfg: dict) -> pd.DataFrame:
        interval = int(sampling_cfg.get("interval_seconds", 3600))
        feature_aggs = sampling_cfg.get("feature_aggs", ["mean"])
        target_agg = sampling_cfg.get("target_agg", "last")
        cols = list(dict.fromkeys(base_features + [target, "Testtime[s]"]))
        work = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=base_features + [target, "Testtime[s]"]).copy()
        work = work.sort_values("Testtime[s]")
        bins = (work["Testtime[s]"] // interval).astype(np.int64)
        work["_bin"] = bins
        agg_spec = {feat: feature_aggs for feat in base_features}
        agg_spec[target] = [target_agg]
        out = work.groupby("_bin", sort=True).agg(agg_spec)
        out.columns = [target if col[0] == target else f"{col[0]}_{col[1]}" for col in out.columns]
        return out.reset_index(drop=True)

    df = load_cell_dataframe(cfg["paths"]["data_root"], CELL)
    base_features = cfg["model"]["features"]
    sampling_cfg = cfg.get("sampling", {})
    target = cfg.get("training", {}).get("target", "SOH")
    hourly = aggregate_hourly(df, base_features, target, sampling_cfg)
    if hasattr(train_mod, "expand_features_for_sampling"):
        features = train_mod.expand_features_for_sampling(base_features, sampling_cfg)
    else:
        features = [f"{feat}_{agg}" for feat in base_features for agg in sampling_cfg.get("feature_aggs", ["mean"])]
    times_h = np.arange(len(hourly), dtype=np.float32)

    def run_one(model_dir: Path, ckpt_path: Path, scaler_path: Path) -> np.ndarray:
        cfg_local = load_yaml(model_dir / "config" / "train_soh.yaml")
        model = train_mod.SOH_LSTM_Seq2Seq(
            in_features=len(features),
            embed_size=int(cfg_local["model"].get("embed_size", 128)),
            hidden_size=int(cfg_local["model"].get("hidden_size", 160)),
            mlp_hidden=int(cfg_local["model"].get("mlp_hidden", 160)),
            num_layers=int(cfg_local["model"].get("num_layers", 2)),
            res_blocks=int(cfg_local["model"].get("res_blocks", 3)),
            bidirectional=bool(cfg_local["model"].get("bidirectional", False)),
            dropout=float(cfg_local["model"].get("dropout", 0.1)),
        ).cpu().eval()
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        scaler = joblib_load(scaler_path)
        X = scaler.transform(hourly[features].to_numpy(dtype=np.float32)).astype(np.float32)
        preds = []
        state_rnn = None
        with torch.inference_mode():
            for i in range(len(X)):
                xb = torch.from_numpy(X[i:i+1]).unsqueeze(0)
                y_seq, state_rnn = model(xb, state=state_rnn, return_state=True)
                preds.append(float(y_seq.squeeze().cpu().item()))
        return np.asarray(preds, dtype=np.float32)

    base_ckpt = find_best_checkpoint(base_dir / "checkpoints")
    base_pred = run_one(base_dir, base_ckpt, base_dir / "scaler_robust.joblib")
    ft_pred = run_one(finetuned_dir, finetuned_dir / "checkpoints" / "best_model_finetuned.pt", finetuned_dir / "scaler_robust.joblib")
    y_true = hourly[target].to_numpy(dtype=np.float32)
    metrics = {
        "base_mae": float(np.mean(np.abs(base_pred - y_true))),
        "ft_mae": float(np.mean(np.abs(ft_pred - y_true))),
        "mae_vs_base": float(np.mean(np.abs(ft_pred - base_pred))),
        "max_abs_diff_vs_base": float(np.max(np.abs(ft_pred - base_pred))),
    }
    write_json(out_dir / "summary.json", metrics)

    plt.figure(figsize=(14, 6))
    plt.plot(times_h, y_true, label="Reference", lw=1.2)
    plt.plot(times_h, base_pred, label="Base", lw=1.0)
    plt.plot(times_h, ft_pred, label="Pruned+FT", lw=1.0)
    plt.xlabel("Hour index")
    plt.ylabel("SOH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "soh_fullcell_compare.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(times_h, np.abs(base_pred - y_true), label="|Base-Ref|", lw=0.9)
    plt.plot(times_h, np.abs(ft_pred - y_true), label="|FT-Ref|", lw=0.9)
    plt.xlabel("Hour index")
    plt.ylabel("Absolute error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "soh_fullcell_abs_error.png", dpi=180)
    plt.close()
    return metrics


def main() -> None:
    soh_ft_mod = load_module("soh_finetune_mod_runtime", SOH_FINETUNE_PY)
    soh_quant_mod = load_module("soh_quant_mod_runtime", SOH_QUANT_PY)

    soc_ft = finetune_soc()
    soc_q = requantize_soc()

    soh_pruned_dir = SOH_ROOT / "pruned_model"
    soh_ft_dir = SOH_ROOT / "finetuned_model"
    soh_ft_mod.run_finetune(
        model_dir=soh_pruned_dir,
        out_dir=soh_ft_dir,
        ckpt_path=soh_pruned_dir / "checkpoints" / "best_model_pruned.pt",
        epochs=8,
        lr=2e-4,
        weight_decay=0.0,
        head_only=False,
        device_str="cuda",
        val_interval=1,
        early_stopping=4,
        max_batches=None,
        num_workers=2,
        prefetch_factor=2,
    )
    soh_q_dir = SOH_ROOT / "quantized_finetuned_model"
    soh_quant_mod.run_quantize(
        model_dir=soh_ft_dir,
        out_dir=soh_q_dir,
        ckpt_path=soh_ft_dir / "checkpoints" / "best_model_finetuned.pt",
    )

    soc_cmp = compare_soc_fullcell()
    soh_cmp = compare_soh_fullcell()
    summary = {
        "soc_finetune": soc_ft,
        "soc_quantized_finetuned": soc_q,
        "soc_fullcell_compare": soc_cmp,
        "soh_fullcell_compare": soh_cmp,
    }
    write_json(ROOT / "tools" / "model_optimization" / "latest_struct30_ftq_fullcell_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
