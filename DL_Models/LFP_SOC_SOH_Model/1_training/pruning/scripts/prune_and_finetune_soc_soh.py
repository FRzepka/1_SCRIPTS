import argparse
import importlib.util
import json
import math
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.prune as prune
import yaml
from joblib import dump
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[3]
WORKDIR = ROOT.parents[1]
SOC_TRAIN = ROOT / "1_training" / "1.6.0.0"
SOH_TRAIN = ROOT / "1_training" / "0.1.2.3"
SOC_MODEL_DIR = ROOT / "2_models" / "SOC_1.6.0.0"
SOH_MODEL_DIR = ROOT / "2_models" / "SOH_0.1.2.3"
COMBINED_MODEL_DIR = ROOT / "2_models" / "SOC_SOH_1.6.0.0_0.1.2.3"
TEST_ROOT = ROOT / "3_test" / "SOC_SOH_1.6.0.0_0.1.2.3"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def model_sparsity(model: torch.nn.Module) -> dict:
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += int((p == 0).sum().item())
    return {"total_params": int(total), "zero_params": int(zeros), "global_sparsity": float(zeros / max(total, 1))}


def collect_prunable_parameters(model: torch.nn.Module) -> list:
    params = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            params.append((module, "weight"))
            if module.bias is not None:
                params.append((module, "bias"))
        elif isinstance(module, torch.nn.LSTM):
            for name, _ in module.named_parameters():
                if name.startswith("weight_ih") or name.startswith("weight_hh") or name.startswith("bias_ih") or name.startswith("bias_hh"):
                    params.append((module, name))
    return params


def remove_pruning_reparam(model: torch.nn.Module) -> None:
    for module, name in collect_prunable_parameters(model):
        if hasattr(module, f"{name}_orig"):
            prune.remove(module, name)


def _apply_linear_structured_pruning(module: torch.nn.Linear, amount: float) -> None:
    out_features = module.weight.shape[0]
    n_prune = max(1, int(round(out_features * amount)))
    weight = module.weight.detach()
    row_score = torch.norm(weight, p=2, dim=1)
    prune_idx = torch.argsort(row_score)[:n_prune]
    weight_mask = torch.ones_like(weight)
    weight_mask[prune_idx, :] = 0.0
    prune.custom_from_mask(module, "weight", weight_mask)
    if module.bias is not None:
        bias_mask = torch.ones_like(module.bias.detach())
        bias_mask[prune_idx] = 0.0
        prune.custom_from_mask(module, "bias", bias_mask)


def _lstm_gate_rows(hidden_size: int, unit_idx: int) -> list[int]:
    return [unit_idx, hidden_size + unit_idx, 2 * hidden_size + unit_idx, 3 * hidden_size + unit_idx]


def _apply_lstm_structured_pruning(module: torch.nn.LSTM, amount: float) -> None:
    hidden_size = module.hidden_size
    for layer in range(module.num_layers):
        suffixes = [""]
        if module.bidirectional:
            suffixes = ["", "_reverse"]
        for suffix in suffixes:
            w_ih_name = f"weight_ih_l{layer}{suffix}"
            w_hh_name = f"weight_hh_l{layer}{suffix}"
            b_ih_name = f"bias_ih_l{layer}{suffix}"
            b_hh_name = f"bias_hh_l{layer}{suffix}"
            w_ih = getattr(module, w_ih_name)
            w_hh = getattr(module, w_hh_name)
            unit_scores = []
            for unit_idx in range(hidden_size):
                rows = _lstm_gate_rows(hidden_size, unit_idx)
                score = torch.norm(w_ih.detach()[rows, :], p=2).item() + torch.norm(w_hh.detach()[rows, :], p=2).item()
                unit_scores.append(score)
            n_prune = max(1, int(round(hidden_size * amount)))
            prune_units = torch.argsort(torch.tensor(unit_scores))[:n_prune].tolist()

            w_ih_mask = torch.ones_like(w_ih.detach())
            w_hh_mask = torch.ones_like(w_hh.detach())
            b_ih_mask = torch.ones_like(getattr(module, b_ih_name).detach())
            b_hh_mask = torch.ones_like(getattr(module, b_hh_name).detach())
            for unit_idx in prune_units:
                rows = _lstm_gate_rows(hidden_size, unit_idx)
                w_ih_mask[rows, :] = 0.0
                w_hh_mask[rows, :] = 0.0
                w_hh_mask[:, unit_idx] = 0.0
                b_ih_mask[rows] = 0.0
                b_hh_mask[rows] = 0.0
            prune.custom_from_mask(module, w_ih_name, w_ih_mask)
            prune.custom_from_mask(module, w_hh_name, w_hh_mask)
            prune.custom_from_mask(module, b_ih_name, b_ih_mask)
            prune.custom_from_mask(module, b_hh_name, b_hh_mask)


def apply_structured_pruning(model: torch.nn.Module, amount: float) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            _apply_linear_structured_pruning(module, amount)
        elif isinstance(module, torch.nn.LSTM):
            _apply_lstm_structured_pruning(module, amount)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_safe(payload), indent=2))


def _to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def finetune_soc(device: torch.device, prune_amount: float, epochs: int) -> dict:
    mod = load_module("train_soc_mod", SOC_TRAIN / "scripts" / "train_soc.py")
    cfg = yaml.safe_load((SOC_TRAIN / "config" / "train_soc.yaml").read_text())
    mod.set_seed(int(cfg.get("seed", 42)))

    features = cfg["model"]["features"]
    chunk = int(cfg["training"]["seq_chunk_size"])
    batch_size = int(cfg["training"].get("batch_size", 512))
    lr = float(cfg["training"]["lr"]) * 0.35
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))

    scaler = RobustScaler()
    train_loader, val_loader = mod.create_dataloaders(cfg, features, chunk, scaler, batch_size=batch_size)

    model = mod.LSTMMLP(
        in_features=len(features),
        hidden_size=int(cfg["model"]["hidden_size"]),
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.05)),
    ).to(device)
    ckpt_path = SOC_TRAIN / "outputs" / "checkpoints" / "soc_epoch0005_rmse0.01393.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    base_metrics, _, _ = mod.eval_model(model, val_loader, device)
    apply_structured_pruning(model, prune_amount)
    pruned_metrics, _, _ = mod.eval_model(model, val_loader, device)

    optimizer = mod.make_optimizer(model, lr=lr, weight_decay=weight_decay)
    amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best = {"rmse": float("inf"), "epoch": 0, "state": None, "metrics": None}
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = mod.train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            amp_scaler,
            max_grad_norm,
            epoch_idx=epoch,
            total_epochs=epochs,
            accum_steps=1,
        )
        metrics, _, _ = mod.eval_model(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(train_loss), **metrics})
        print(f"[SOC prune-ft] epoch={epoch}/{epochs} train_loss={train_loss:.6f} val_rmse={metrics['rmse']:.6f} val_mae={metrics['mae']:.6f}", flush=True)
        if metrics["rmse"] < best["rmse"]:
            best = {
                "rmse": float(metrics["rmse"]),
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "metrics": metrics,
            }

    model.load_state_dict(best["state"])
    remove_pruning_reparam(model)
    final_metrics, _, _ = mod.eval_model(model, val_loader, device)
    scaler_out = SOC_MODEL_DIR / "pruned_model" / "scaler_robust.joblib"
    dump(scaler, scaler_out)

    out_ckpt = SOC_MODEL_DIR / "pruned_model" / f"soc_pruned20_finetuned_epoch{best['epoch']:04d}_rmse{best['rmse']:.5f}.pt"
    save_checkpoint(
        out_ckpt,
        {
            "epoch": best["epoch"],
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "features": features,
            "chunk": chunk,
            "prune_amount": prune_amount,
            "base_metrics": base_metrics,
            "pruned_metrics_before_ft": pruned_metrics,
            "final_metrics": final_metrics,
            "history": history,
        },
    )
    shutil.copy2(SOC_MODEL_DIR / "train_soc.yaml", SOC_MODEL_DIR / "pruned_model" / "train_soc.yaml")
    summary = {
        "checkpoint": str(out_ckpt),
        "scaler": str(scaler_out),
        "config": str(SOC_MODEL_DIR / "pruned_model" / "train_soc.yaml"),
        "prune_amount": prune_amount,
        "baseline_val_metrics": base_metrics,
        "pruned_pre_finetune_val_metrics": pruned_metrics,
        "final_val_metrics": final_metrics,
        **model_sparsity(model),
    }
    write_json(SOC_MODEL_DIR / "pruned_model" / "summary.json", summary)
    return summary


def finetune_soh(device: torch.device, prune_amount: float, epochs: int) -> dict:
    mod = load_module("train_soh_mod", SOH_TRAIN / "scripts" / "train_soh.py")
    cfg = yaml.safe_load((SOH_TRAIN / "config" / "train_soh.yaml").read_text())
    cfg["paths"]["data_root"] = mod.expand_env_with_defaults(cfg["paths"]["data_root"])
    mod.set_seed(int(cfg.get("seed", 42)))

    base_features = cfg["model"]["features"]
    sampling_cfg = cfg.get("sampling", {})
    features = mod.expand_features_for_sampling(base_features, sampling_cfg)
    target = cfg.get("training", {}).get("target", "SOH")
    chunk = int(cfg["training"]["seq_chunk_size"])
    batch_size = int(cfg["training"].get("batch_size", 256))
    lr = float(cfg["training"]["lr"]) * 0.35
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    warmup_steps = int(cfg["training"].get("warmup_steps", 0))
    loss_mode = str(cfg["training"].get("loss_mode", "seq2seq")).lower()
    smooth_loss_weight = float(cfg["training"].get("smooth_loss_weight", 0.0))
    smooth_loss_type = str(cfg["training"].get("smooth_loss_type", "l1")).lower()

    scaler = RobustScaler()
    train_loader, val_loader = mod.create_dataloaders(cfg, base_features, features, target, chunk, scaler, batch_size=batch_size)

    model = mod.SOH_LSTM_Seq2Seq(
        in_features=len(features),
        embed_size=int(cfg["model"]["embed_size"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        mlp_hidden=int(cfg["model"]["mlp_hidden"]),
        num_layers=int(cfg["model"].get("num_layers", 2)),
        res_blocks=int(cfg["model"].get("res_blocks", 2)),
        bidirectional=bool(cfg["model"].get("bidirectional", False)),
        dropout=float(cfg["model"].get("dropout", 0.15)),
    ).to(device)
    ckpt_path = SOH_TRAIN / "outputs" / "soh" / "run_20260109_163030" / "checkpoints" / "best_epoch0093_rmse0.02165.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    base_metrics, _, _ = mod.eval_model(model, val_loader, device, loss_mode=loss_mode, warmup_steps=warmup_steps)
    apply_structured_pruning(model, prune_amount)
    pruned_metrics, _, _ = mod.eval_model(model, val_loader, device, loss_mode=loss_mode, warmup_steps=warmup_steps)

    optimizer = mod.make_optimizer(model, lr=lr, weight_decay=weight_decay)
    amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best = {"rmse": float("inf"), "epoch": 0, "state": None, "metrics": None}
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = mod.train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            amp_scaler,
            max_grad_norm,
            epoch,
            epochs,
            accum_steps=1,
            loss_mode=loss_mode,
            warmup_steps=warmup_steps,
            smooth_loss_weight=smooth_loss_weight,
            smooth_loss_type=smooth_loss_type,
            max_batches=None,
        )
        metrics, _, _ = mod.eval_model(model, val_loader, device, loss_mode=loss_mode, warmup_steps=warmup_steps)
        history.append({"epoch": epoch, "train_loss": float(train_loss), **metrics})
        print(f"[SOH prune-ft] epoch={epoch}/{epochs} train_loss={train_loss:.6f} val_rmse={metrics['rmse']:.6f} val_mae={metrics['mae']:.6f}", flush=True)
        if metrics["rmse"] < best["rmse"]:
            best = {
                "rmse": float(metrics["rmse"]),
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "metrics": metrics,
            }

    model.load_state_dict(best["state"])
    remove_pruning_reparam(model)
    final_metrics, _, _ = mod.eval_model(model, val_loader, device, loss_mode=loss_mode, warmup_steps=warmup_steps)
    scaler_out = SOH_MODEL_DIR / "pruned_model" / "scaler_robust.joblib"
    dump(scaler, scaler_out)

    out_ckpt = SOH_MODEL_DIR / "pruned_model" / f"best_pruned20_finetuned_epoch{best['epoch']:04d}_rmse{best['rmse']:.5f}.pt"
    save_checkpoint(
        out_ckpt,
        {
            "epoch": best["epoch"],
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "metrics": final_metrics,
            "prune_amount": prune_amount,
            "base_metrics": base_metrics,
            "pruned_metrics_before_ft": pruned_metrics,
            "history": history,
        },
    )
    shutil.copy2(SOH_MODEL_DIR / "train_soh.yaml", SOH_MODEL_DIR / "pruned_model" / "train_soh.yaml")
    summary = {
        "checkpoint": str(out_ckpt),
        "scaler": str(scaler_out),
        "config": str(SOH_MODEL_DIR / "pruned_model" / "train_soh.yaml"),
        "prune_amount": prune_amount,
        "baseline_val_metrics": base_metrics,
        "pruned_pre_finetune_val_metrics": pruned_metrics,
        "final_val_metrics": final_metrics,
        **model_sparsity(model),
    }
    write_json(SOH_MODEL_DIR / "pruned_model" / "summary.json", summary)
    return summary


def write_combined_wrapper(soc_summary: dict, soh_summary: dict) -> Path:
    out_dir = COMBINED_MODEL_DIR / "pruned_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = COMBINED_MODEL_DIR / "simulate_soh_soc.py"
    dst = out_dir / "simulate_soh_soc.py"
    text = src.read_text()
    replacements = {
        str(SOC_MODEL_DIR / "train_soc.yaml"): str(SOC_MODEL_DIR / "pruned_model" / "train_soc.yaml"),
        str(SOC_MODEL_DIR / "soc_epoch0005_rmse0.01393.pt"): soc_summary["checkpoint"],
        str(SOC_MODEL_DIR / "scaler_robust.joblib"): soc_summary["scaler"],
        str(SOH_MODEL_DIR / "train_soh.yaml"): str(SOH_MODEL_DIR / "pruned_model" / "train_soh.yaml"),
        str(SOH_MODEL_DIR / "best_epoch0093_rmse0.02165.pt"): soh_summary["checkpoint"],
        str(SOH_MODEL_DIR / "scaler_robust.joblib"): soh_summary["scaler"],
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    dst.write_text(text)
    readme = out_dir / "README.md"
    readme.write_text(
        "# Pruned SOC+SOH chain\n\n"
        f"- SOC checkpoint: `{soc_summary['checkpoint']}`\n"
        f"- SOH checkpoint: `{soh_summary['checkpoint']}`\n"
        "- Use `simulate_soh_soc.py` in this folder for pruned-chain inference.\n"
    )
    write_json(out_dir / "summary.json", {"soc": soc_summary, "soh": soh_summary})
    return dst


def run_combined_test(sim_script: Path, soc_batch: int) -> Path:
    out_dir = TEST_ROOT / time.strftime("%Y-%m-%d_%H%M_pruned")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sim_script, out_dir / "simulate_soh_soc.py")
    cmd = [
        sys.executable,
        str(out_dir / "simulate_soh_soc.py"),
        "--out_dir",
        str(out_dir),
        "--cell",
        "MGFarm_18650_C07",
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
        "--soc_batch",
        str(soc_batch),
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--soc_epochs", type=int, default=8)
    ap.add_argument("--soh_epochs", type=int, default=8)
    ap.add_argument("--prune_amount", type=float, default=0.20)
    ap.add_argument("--soc_batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[SETUP] device={device} prune_amount={args.prune_amount:.2f}", flush=True)

    (SOC_MODEL_DIR / "pruned_model").mkdir(parents=True, exist_ok=True)
    (SOH_MODEL_DIR / "pruned_model").mkdir(parents=True, exist_ok=True)
    (COMBINED_MODEL_DIR / "pruned_model").mkdir(parents=True, exist_ok=True)

    soc_summary = finetune_soc(device=device, prune_amount=args.prune_amount, epochs=args.soc_epochs)
    soh_summary = finetune_soh(device=device, prune_amount=args.prune_amount, epochs=args.soh_epochs)
    sim_script = write_combined_wrapper(soc_summary, soh_summary)
    test_dir = run_combined_test(sim_script, soc_batch=args.soc_batch)
    write_json(
        COMBINED_MODEL_DIR / "pruned_model" / "combined_test_summary.json",
        {
            "test_dir": str(test_dir),
            "soc_summary": soc_summary,
            "soh_summary": soh_summary,
        },
    )
    print(f"[DONE] combined test dir: {test_dir}", flush=True)


if __name__ == "__main__":
    main()
