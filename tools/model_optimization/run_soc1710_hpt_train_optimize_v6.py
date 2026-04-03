#!/usr/bin/env python3
"""Automated workflow for SOC 1.7.1.0:

1. Optuna HPT
2. Final training
3. Structured 30% hidden-size pruning
4. Post-pruning finetuning
5. Dynamic int8 quantization
6. Full-cell validation artifacts under 3_test
7. Wait for shared SOH-LSTM deployment folder
8. Launch benchmark v6
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

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
PYTHON = Path("/home/florianr/anaconda3/envs/ml1/bin/python")
CELL = "MGFarm_18650_C07"
PRUNE_AMOUNT = 0.30

TRAIN_DIR = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "1_training" / "1.7.1.0"
HPT_DIR = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "1_training" / "1.7.1.0_HPT"
MODEL_DIR = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "2_models" / "SOC_1.7.1.0"
TRAIN_PY = TRAIN_DIR / "scripts" / "train_soc.py"
HPT_PY = HPT_DIR / "scripts" / "hpt_soc.py"
HPT_CFG = HPT_DIR / "config" / "hpt_soc.yaml"
BEST_CFG = TRAIN_DIR / "config" / "train_soc_best_from_hpt.yaml"

WORK_ROOT = MODEL_DIR / "Pruned_Quantized_1.7.1.0_s30_struct_int8"
FINAL_PRUNED_DST = MODEL_DIR / "PrunedFT_1.7.1.0_s30_struct"
FINAL_QUANT_DST = MODEL_DIR / "Quantized_1.7.1.0_s30_struct_ft_int8"
TEST_DST = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "3_test" / "2026-03-27_soc1710_pruned_quantized_fullcell_validation_v6"

SHARED_SOH_DST = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / "LSTM" / "Pruned" / "0.1.2.5_base_h160_s30_struct_ft"
V6_PIPELINE = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "4_simulation_environment" / "run_soc171_benchmark_v6_pipeline.py"
V6_PIPELINE_LOG = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "4_simulation_environment" / "campaigns" / "2026-03-27_extended_matrix_fullc07_v6_soc171_s30ft_soh0125_s30ft" / "pipeline.log"
SUMMARY_OUT = ROOT / "tools" / "model_optimization" / "latest_soc1710_struct30_ftq_fullcell_summary.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def screen_exists(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], capture_output=True, text=True, check=False)
    return name in result.stdout


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run(cmd: list[str]) -> None:
    log(f"[RUN] {' '.join(str(x) for x in cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def find_best_rmse_checkpoint(ckpt_dir: Path) -> Path:
    best: Path | None = None
    best_rmse: float | None = None
    for path in ckpt_dir.glob("*.pt"):
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


def _gate_row_indices(hidden_old: int, keep_idx: torch.Tensor, gates: int) -> list[int]:
    keep = [int(i) for i in keep_idx.tolist()]
    out: list[int] = []
    for g in range(gates):
        off = g * hidden_old
        out.extend([off + i for i in keep])
    return out


def _pick_gru_keep_indices(model: torch.nn.Module, keep_hidden: int) -> torch.Tensor:
    gru = model.gru
    hidden_old = int(gru.hidden_size)
    num_layers = int(gru.num_layers)
    scores = torch.zeros(hidden_old, dtype=torch.float32)

    for layer in range(num_layers):
        w_hh = getattr(gru, f"weight_hh_l{layer}").detach()
        for g in range(3):
            rows = w_hh[g * hidden_old:(g + 1) * hidden_old, :]
            scores += rows.abs().mean(dim=1).to(scores.dtype)
            scores += rows.abs().mean(dim=0).to(scores.dtype)
        if layer + 1 < num_layers:
            w_ih_next = getattr(gru, f"weight_ih_l{layer+1}").detach()
            for g in range(3):
                rows = w_ih_next[g * hidden_old:(g + 1) * hidden_old, :]
                scores += rows.abs().mean(dim=0).to(scores.dtype)

    first_linear = model.mlp[0]
    if isinstance(first_linear, torch.nn.Linear):
        scores += first_linear.weight.detach().abs().mean(dim=0).to(scores.dtype)

    keep = torch.topk(scores, k=keep_hidden, largest=True).indices
    keep, _ = torch.sort(keep)
    return keep.long()


def _copy_gru_hidden_shrink_weights(old_model: torch.nn.Module, new_model: torch.nn.Module, keep_idx: torch.Tensor) -> None:
    old_gru = old_model.gru
    new_gru = new_model.gru
    hidden_old = int(old_gru.hidden_size)
    num_layers = int(old_gru.num_layers)
    keep = [int(i) for i in keep_idx.tolist()]
    rows = _gate_row_indices(hidden_old, keep_idx, 3)

    for layer in range(num_layers):
        old_w_ih = getattr(old_gru, f"weight_ih_l{layer}").data
        new_w_ih = getattr(new_gru, f"weight_ih_l{layer}").data
        if layer == 0:
            in_cols = list(range(old_w_ih.shape[1]))
        else:
            in_cols = keep
        new_w_ih.copy_(old_w_ih[rows][:, in_cols])

        old_w_hh = getattr(old_gru, f"weight_hh_l{layer}").data
        new_w_hh = getattr(new_gru, f"weight_hh_l{layer}").data
        new_w_hh.copy_(old_w_hh[rows][:, keep])

        old_b_ih = getattr(old_gru, f"bias_ih_l{layer}").data
        new_b_ih = getattr(new_gru, f"bias_ih_l{layer}").data
        new_b_ih.copy_(old_b_ih[rows])

        old_b_hh = getattr(old_gru, f"bias_hh_l{layer}").data
        new_b_hh = getattr(new_gru, f"bias_hh_l{layer}").data
        new_b_hh.copy_(old_b_hh[rows])

    old_fc1 = old_model.mlp[0]
    new_fc1 = new_model.mlp[0]
    new_fc1.weight.data.copy_(old_fc1.weight.data[:, keep])
    new_fc1.bias.data.copy_(old_fc1.bias.data)

    for idx in (3,):
        old_fc = old_model.mlp[idx]
        new_fc = new_model.mlp[idx]
        if isinstance(old_fc, torch.nn.Linear) and isinstance(new_fc, torch.nn.Linear):
            new_fc.weight.data.copy_(old_fc.weight.data)
            new_fc.bias.data.copy_(old_fc.bias.data)


def run_hpt_and_train() -> None:
    run([str(PYTHON), str(HPT_PY), "--config", str(HPT_CFG)])
    run([str(PYTHON), str(TRAIN_PY), "--config", str(BEST_CFG)])


def prune_soc() -> dict[str, Any]:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    pruned_dir = WORK_ROOT / "pruned_model"

    train_mod = load_module("soc_train_mod_171", TRAIN_PY)
    cfg = load_yaml(MODEL_DIR / "train_soc.yaml")
    ckpt_path = find_best_rmse_checkpoint(MODEL_DIR)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)

    old_hidden = int(cfg["model"]["hidden_size"])
    new_hidden = max(1, int(round(old_hidden * (1.0 - PRUNE_AMOUNT))))
    model = train_mod.GRUMLP(
        in_features=len(cfg["model"]["features"]),
        hidden_size=old_hidden,
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.05)),
    ).cpu().eval()
    model.load_state_dict(state_dict)

    keep_idx = _pick_gru_keep_indices(model, keep_hidden=new_hidden)
    pruned_cfg = copy.deepcopy(cfg)
    pruned_cfg["model"]["hidden_size"] = int(new_hidden)
    pruned_model = train_mod.GRUMLP(
        in_features=len(pruned_cfg["model"]["features"]),
        hidden_size=new_hidden,
        mlp_hidden=int(pruned_cfg["model"]["mlp_hidden"]),
        num_layers=int(pruned_cfg["model"].get("num_layers", 1)),
        dropout=float(pruned_cfg["model"].get("dropout", 0.05)),
    ).cpu().eval()
    _copy_gru_hidden_shrink_weights(model, pruned_model, keep_idx)

    (pruned_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (pruned_dir / "config").mkdir(parents=True, exist_ok=True)
    (pruned_dir / "scripts").mkdir(parents=True, exist_ok=True)
    pruned_ckpt = pruned_dir / "checkpoints" / "soc_best_model_pruned.pt"
    torch.save({"model_state_dict": pruned_model.state_dict(), "config": pruned_cfg}, pruned_ckpt)
    with open(pruned_dir / "config" / "train_soc.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(pruned_cfg, f, sort_keys=False)
    shutil.copy2(TRAIN_PY, pruned_dir / "scripts" / "train_soc.py")
    shutil.copy2(MODEL_DIR / "scaler_robust.joblib", pruned_dir / "scaler_robust.joblib")
    if (MODEL_DIR / "export_manifest.json").exists():
        shutil.copy2(MODEL_DIR / "export_manifest.json", pruned_dir / "export_manifest.json")

    meta = {
        "model_dir": str(MODEL_DIR),
        "source_checkpoint": str(ckpt_path),
        "pruned_checkpoint": str(pruned_ckpt),
        "amount": PRUNE_AMOUNT,
        "mode": "structured",
        "structured_kind": "gru_hidden_shrink",
        "old_hidden_size": old_hidden,
        "new_hidden_size": new_hidden,
        "keep_indices": [int(x) for x in keep_idx.tolist()],
    }
    write_json(pruned_dir / "prune_meta.json", meta)
    return meta


def finetune_soc() -> dict[str, Any]:
    model_dir = WORK_ROOT / "pruned_model"
    out_dir = WORK_ROOT / "finetuned_model"
    cfg = load_yaml(model_dir / "config" / "train_soc.yaml")
    train_mod = load_module("soc_train_mod_ft171", model_dir / "scripts" / "train_soc.py")

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
    epochs = 10
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
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_mod.train_one_epoch(
            model, train_loader, device, optimizer, amp_scaler, max_grad_norm, epoch, epochs, accum_steps=accum_steps
        )
        metrics, _, _ = train_mod.eval_model(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(train_loss), **metrics})
        log(f"[SOC171 FT {epoch:03d}/{epochs}] train_loss={train_loss:.6f} val_rmse={metrics['rmse']:.6f} val_mae={metrics['mae']:.6f}")
        if metrics["rmse"] < best_rmse:
            best_rmse = float(metrics["rmse"])
            best_epoch = epoch
            patience = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, best_path)
        else:
            patience += 1
            if patience >= early_stopping:
                log(f"[SOC171 FT] early stopping at epoch {epoch}")
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


def quantize_soc() -> dict[str, Any]:
    model_dir = WORK_ROOT / "finetuned_model"
    out_dir = WORK_ROOT / "quantized_finetuned_model"
    cfg = load_yaml(model_dir / "config" / "train_soc.yaml")
    train_mod = load_module("soc_train_mod_q171", model_dir / "scripts" / "train_soc.py")

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


def compare_soc_fullcell() -> dict[str, Any]:
    base_dir = MODEL_DIR
    finetuned_dir = WORK_ROOT / "finetuned_model"
    out_dir = WORK_ROOT / "fullcell_compare_C07"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_mod = load_module("soc_train_mod_cmp171", finetuned_dir / "scripts" / "train_soc.py")
    cfg = load_yaml(finetuned_dir / "config" / "train_soc.yaml")
    df = train_mod.load_cell_dataframe(cfg["paths"]["data_root"], CELL)
    work = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Testtime[s]", "Voltage[V]", "Current[A]", "Temperature[°C]", "SOH", "SOC", "Q_c"]).reset_index(drop=True)
    times_h = work["Testtime[s]"].to_numpy(dtype=np.float64) / 3600.0

    def run_one(model_dir: Path, ckpt_path: Path, scaler_path: Path) -> np.ndarray:
        cfg_local = load_yaml(model_dir / "config" / "train_soc.yaml")
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

    base_ckpt = find_best_rmse_checkpoint(base_dir)
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
    plt.plot(times_h, base_pred, label="Base 1.7.1.0", lw=1.0)
    plt.plot(times_h, ft_pred, label="1.7.1.0 pruned+FT", lw=1.0)
    plt.xlabel("Time [h]")
    plt.ylabel("SOC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "soc_fullcell_compare.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(times_h, np.abs(base_pred - y_true), label="|Base-Ref|", lw=0.9)
    plt.plot(times_h, np.abs(ft_pred - y_true), label="|Pruned+FT-Ref|", lw=0.9)
    plt.xlabel("Time [h]")
    plt.ylabel("Absolute error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "soc_fullcell_abs_error.png", dpi=180)
    plt.close()
    return metrics


def copy_outputs() -> None:
    copytree_replace(WORK_ROOT / "finetuned_model", FINAL_PRUNED_DST)
    copytree_replace(WORK_ROOT / "quantized_finetuned_model", FINAL_QUANT_DST)
    TEST_DST.mkdir(parents=True, exist_ok=True)
    copytree_replace(WORK_ROOT / "fullcell_compare_C07", TEST_DST / "soc1710_fullcell_compare_C07")
    shutil.copy2(SUMMARY_OUT, TEST_DST / "summary_soc1710_ftq_fullcell.json")


def wait_for_soh() -> None:
    while not SHARED_SOH_DST.exists():
        log("[WAIT] shared SOH deployment folder still missing")
        time.sleep(60)


def launch_v6() -> None:
    if screen_exists("benchv6_pipeline"):
        log("[SKIP] benchv6_pipeline already exists")
        return
    V6_PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "screen",
            "-dmS",
            "benchv6_pipeline",
            "bash",
            "-lc",
            f"cd {ROOT} && PYTHONUNBUFFERED=1 {PYTHON} {V6_PIPELINE} >> {V6_PIPELINE_LOG} 2>&1",
        ],
        check=True,
    )
    log("[LAUNCH] benchv6_pipeline")


def main() -> None:
    log("[START] SOC 1.7.1.0 workflow")
    run_hpt_and_train()
    prune_meta = prune_soc()
    ft_meta = finetune_soc()
    q_meta = quantize_soc()
    compare_meta = compare_soc_fullcell()

    summary = {
        "soc_version": "1.7.1.0",
        "cell": CELL,
        "prune": prune_meta,
        "finetune": ft_meta,
        "quantize": q_meta,
        "fullcell_compare": compare_meta,
    }
    write_json(SUMMARY_OUT, summary)
    copy_outputs()
    log("[DONE] SOC outputs copied to final model/test folders")
    wait_for_soh()
    log("[DONE] shared SOH deployment folder ready")
    launch_v6()
    log("[DONE] workflow complete")


if __name__ == "__main__":
    main()
