#!/usr/bin/env python3
"""Optuna hyperparameter search for compact SOH base models (LSTM/TCN/GRU/CNN)."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import re
import shutil
import time
import typing as t
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from joblib import dump as joblib_dump
from sklearn.preprocessing import RobustScaler


MODEL_SPECS = {
    "LSTM_0.1.3.1": {"family": "LSTM", "version": "0.1.3.1"},
    "TCN_0.2.3.1": {"family": "TCN", "version": "0.2.3.1"},
    "GRU_0.3.2.1": {"family": "GRU", "version": "0.3.2.1"},
    "CNN_0.4.2.1": {"family": "CNN", "version": "0.4.2.1"},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expand_env_with_defaults(val: str) -> str:
    pattern = re.compile(r"\$\{([^:}]+):-([^}]+)\}")

    def repl(match: re.Match) -> str:
        var, default = match.group(1), match.group(2)
        env_val = os.getenv(var)
        return env_val if env_val not in (None, "") else default

    if not isinstance(val, str):
        return val
    return os.path.expandvars(pattern.sub(repl, val))


def load_train_module(path: Path):
    spec = importlib.util.spec_from_file_location("train_soh_module", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def find_best_ckpt_from_dir(ckpt_dir: Path) -> t.Optional[Path]:
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob("best_epoch*_rmse*.pt"))
    if not candidates:
        return None

    def _score(p: Path) -> float:
        m = re.search(r"rmse([0-9]+(?:\.[0-9]+)?)", p.name)
        if not m:
            return float("inf")
        return float(m.group(1))

    candidates.sort(key=_score)
    return candidates[0]


def model_dirs(workspace_root: Path, model_key: str) -> dict:
    spec = MODEL_SPECS[model_key]
    fam = spec["family"]
    ver = spec["version"]
    base_dir = workspace_root / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / fam / "Base" / ver
    return {
        "base_dir": base_dir,
        "config": base_dir / "config" / "train_soh.yaml",
        "script": base_dir / "scripts" / "train_soh.py",
        "dest_hp": workspace_root / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / fam / "Base" / f"{ver}_hp",
        "family": fam,
        "version": ver,
    }


def build_model(cfg: dict, mod, family: str, in_features: int) -> nn.Module:
    if family == "LSTM":
        return mod.SOH_LSTM_Seq2Seq(
            in_features=in_features,
            embed_size=int(cfg["model"].get("embed_size", 96)),
            hidden_size=int(cfg["model"]["hidden_size"]),
            mlp_hidden=int(cfg["model"]["mlp_hidden"]),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            res_blocks=int(cfg["model"].get("res_blocks", 2)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.15)),
        )
    if family == "GRU":
        return mod.SOH_GRU_Seq2Seq(
            in_features=in_features,
            embed_size=int(cfg["model"].get("embed_size", 96)),
            hidden_size=int(cfg["model"]["hidden_size"]),
            mlp_hidden=int(cfg["model"]["mlp_hidden"]),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            res_blocks=int(cfg["model"].get("res_blocks", 2)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.15)),
        )
    if family == "TCN":
        dilations = cfg["model"].get("dilations")
        if not dilations:
            num_layers = int(cfg["model"].get("num_layers", 4))
            dilations = [2 ** i for i in range(num_layers)]
        return mod.CausalTCN_SOH(
            in_features=in_features,
            hidden_size=int(cfg["model"]["hidden_size"]),
            mlp_hidden=int(cfg["model"]["mlp_hidden"]),
            kernel_size=int(cfg["model"].get("kernel_size", 3)),
            dilations=[int(d) for d in dilations],
            dropout=float(cfg["model"].get("dropout", 0.05)),
        )
    if family == "CNN":
        dilations = cfg["model"].get("dilations")
        if dilations is not None:
            dilations = [int(d) for d in dilations]
        return mod.SOH_CNN_Seq2Seq(
            in_features=in_features,
            hidden_size=int(cfg["model"].get("hidden_size", 128)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 96)),
            kernel_size=int(cfg["model"].get("kernel_size", 5)),
            dilations=dilations,
            num_blocks=int(cfg["model"].get("num_blocks", 4)),
            dropout=float(cfg["model"].get("dropout", 0.15)),
        )
    raise ValueError(f"Unsupported family: {family}")


def trial_cfg_update(trial: optuna.Trial, cfg: dict, family: str, args) -> dict:
    c = copy.deepcopy(cfg)
    c["training"]["compile"] = False
    c["training"]["epochs"] = int(args.max_epochs)
    c["training"]["early_stopping"] = int(args.early_stopping)
    c["training"]["val_interval"] = int(args.val_interval)
    c["training"]["accum_steps"] = 1
    c["training"]["loss_mode"] = "seq2seq"
    c["training"]["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.6, 1.3)
    c["training"]["seq_chunk_size"] = trial.suggest_categorical("seq_chunk_size", [72, 96, 120, 144, 168])
    window_choices = [1, 6, 12, 24, 48]
    window_choices = [w for w in window_choices if w <= int(c["training"]["seq_chunk_size"])]
    c["training"]["window_stride"] = trial.suggest_categorical("window_stride", window_choices)
    c["training"]["lr"] = trial.suggest_float("lr", 5e-5, 6e-4, log=True)
    c["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)
    c["training"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 48, 64, 96])
    c["training"]["smooth_loss_weight"] = trial.suggest_float("smooth_loss_weight", 0.0, 0.20)
    c["training"]["smooth_loss_type"] = trial.suggest_categorical("smooth_loss_type", ["l1", "l2"])
    c["training"]["warmup_steps"] = trial.suggest_categorical("warmup_steps", [0, 4, 8, 12])
    c["training"]["warmup_epochs"] = trial.suggest_int("warmup_epochs", 0, 3)

    if family == "LSTM":
        c["model"]["embed_size"] = trial.suggest_categorical("embed_size", [32, 48, 64, 80])
        c["model"]["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 96, 128, 160])
        c["model"]["mlp_hidden"] = trial.suggest_categorical("mlp_hidden", [32, 48, 64, 96])
        c["model"]["num_layers"] = trial.suggest_int("num_layers", 1, 3)
        c["model"]["res_blocks"] = trial.suggest_int("res_blocks", 0, 2)
        c["model"]["dropout"] = trial.suggest_float("dropout", 0.05, 0.25)
        c["model"]["bidirectional"] = False
    elif family == "GRU":
        c["model"]["embed_size"] = trial.suggest_categorical("embed_size", [32, 48, 64, 80, 96])
        c["model"]["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 96, 128, 160, 192])
        c["model"]["mlp_hidden"] = trial.suggest_categorical("mlp_hidden", [32, 48, 64, 96])
        c["model"]["num_layers"] = trial.suggest_int("num_layers", 1, 3)
        c["model"]["res_blocks"] = trial.suggest_int("res_blocks", 0, 2)
        c["model"]["dropout"] = trial.suggest_float("dropout", 0.05, 0.25)
        c["model"]["bidirectional"] = False
    elif family == "TCN":
        c["model"]["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 48, 64, 80, 96])
        c["model"]["mlp_hidden"] = trial.suggest_categorical("mlp_hidden", [32, 48, 64, 80, 96])
        c["model"]["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5])
        c["model"]["dropout"] = trial.suggest_float("dropout", 0.02, 0.18)
        dilations_label = trial.suggest_categorical(
            "dilations",
            ["1,2,4", "1,1,2,4", "1,2,4,8", "1,2,2,4", "1,2,4,4,8"],
        )
        c["model"]["dilations"] = [int(x) for x in dilations_label.split(",")]
        c["training"]["warmup_steps"] = trial.suggest_categorical("warmup_steps_tcn", [-1, 0, 8, 16, 24])
    elif family == "CNN":
        c["model"]["hidden_size"] = trial.suggest_categorical("hidden_size", [48, 64, 80, 96, 128])
        c["model"]["mlp_hidden"] = trial.suggest_categorical("mlp_hidden", [32, 48, 64, 80, 96])
        c["model"]["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5])
        c["model"]["dropout"] = trial.suggest_float("dropout", 0.05, 0.22)
        dilations_label = trial.suggest_categorical(
            "dilations",
            ["1,2,4", "1,1,2,4", "1,2,2,4", "1,2,4,8"],
        )
        c["model"]["dilations"] = [int(x) for x in dilations_label.split(",")]
        c["model"]["num_blocks"] = len(c["model"]["dilations"])
    else:
        raise ValueError(f"Unsupported family: {family}")

    dl = c.get("dataloader", {})
    dl["num_workers"] = int(args.num_workers)
    dl["prefetch_factor"] = int(args.prefetch_factor)
    dl["persistent_workers"] = bool(args.num_workers > 0)
    dl["pin_memory"] = True
    c["dataloader"] = dl
    return c


def select_loss_targets(pred_seq: torch.Tensor, target_seq: torch.Tensor, loss_mode: str, warmup_steps: int):
    if loss_mode == "last":
        return pred_seq[:, -1], target_seq[:, -1]
    if warmup_steps > 0:
        return pred_seq[:, warmup_steps:], target_seq[:, warmup_steps:]
    return pred_seq, target_seq


def train_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer,
    amp_scaler: t.Optional[torch.amp.GradScaler],
    max_grad_norm: float,
    loss_mode: str,
    warmup_steps: int,
    smooth_loss_weight: float,
    smooth_loss_type: str,
    max_batches: t.Optional[int],
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)

    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    for batch_idx, (xb, yb) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=amp_device_type, enabled=amp_scaler is not None):
            pred_seq = model(xb)
            pred_sel, target_sel = select_loss_targets(pred_seq, yb, loss_mode, warmup_steps)
            loss = loss_fn(pred_sel, target_sel)
            if smooth_loss_weight > 0.0 and pred_sel.ndim == 2 and pred_sel.size(1) > 1:
                diffs = pred_sel[:, 1:] - pred_sel[:, :-1]
                smooth = (diffs ** 2).mean() if smooth_loss_type == "l2" else diffs.abs().mean()
                loss = loss + smooth_loss_weight * smooth

        if amp_scaler is not None:
            amp_scaler.scale(loss).backward()
            if max_grad_norm > 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        count += 1
    return total_loss / max(1, count)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"rmse": None, "mae": None, "r2": None, "max_error": None}
    diff = y_true - y_pred
    rmse = float(math.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    var = float(np.var(y_true))
    r2 = float(1.0 - (np.sum(diff ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))) if var > 0 else float("nan")
    max_error = float(np.max(np.abs(diff)))
    return {"rmse": rmse, "mae": mae, "r2": r2, "max_error": max_error}


def eval_model(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_mode: str,
    warmup_steps: int,
    max_batches: t.Optional[int],
) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred_seq = model(xb)
            pred_sel, tgt_sel = select_loss_targets(pred_seq, yb, loss_mode, warmup_steps)
            preds.append(pred_sel.detach().cpu().reshape(-1))
            targets.append(tgt_sel.detach().cpu().reshape(-1))

    if not preds:
        return {"rmse": None, "mae": None, "r2": None, "max_error": None}
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    return compute_metrics(y_true, y_pred)


def objective_score(metrics: dict, target_mae: t.Optional[float], mae_penalty_weight: float) -> float:
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    if rmse is None or mae is None:
        return float("inf")
    if target_mae is None:
        return float(rmse)
    return float(rmse + mae_penalty_weight * abs(mae - target_mae))


def arch_string(family: str, cfg: dict) -> str:
    m = cfg["model"]
    if family == "LSTM":
        return (
            f"LSTM(embed={m.get('embed_size')}, hidden={m.get('hidden_size')}, "
            f"layers={m.get('num_layers')}, res={m.get('res_blocks')}, drop={m.get('dropout'):.3f})"
        )
    if family == "GRU":
        return (
            f"GRU(embed={m.get('embed_size')}, hidden={m.get('hidden_size')}, "
            f"layers={m.get('num_layers')}, res={m.get('res_blocks')}, drop={m.get('dropout'):.3f})"
        )
    if family == "TCN":
        return (
            f"TCN(hidden={m.get('hidden_size')}, mlp={m.get('mlp_hidden')}, "
            f"k={m.get('kernel_size')}, dil={m.get('dilations')}, drop={m.get('dropout'):.3f})"
        )
    if family == "CNN":
        return (
            f"CNN(hidden={m.get('hidden_size')}, mlp={m.get('mlp_hidden')}, "
            f"k={m.get('kernel_size')}, dil={m.get('dilations')}, drop={m.get('dropout'):.3f})"
        )
    return family


def export_best_to_hp_dir(
    workspace_root: Path,
    dirs: dict,
    run_dir: Path,
    best_trial: optuna.trial.FrozenTrial,
    best_cfg: dict,
    best_ckpt: Path,
    scaler_path: Path,
    study: optuna.Study,
    target_mae: t.Optional[float],
    mae_penalty_weight: float,
) -> Path:
    hp_dir = dirs["dest_hp"]
    hp_dir.mkdir(parents=True, exist_ok=True)
    (hp_dir / "checkpoints").mkdir(exist_ok=True)
    (hp_dir / "config").mkdir(exist_ok=True)
    (hp_dir / "scripts").mkdir(exist_ok=True)

    src_scripts = dirs["base_dir"] / "scripts"
    for item in src_scripts.glob("*"):
        if item.is_file():
            shutil.copy2(item, hp_dir / "scripts" / item.name)

    with open(hp_dir / "config" / "train_soh.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    ckpt_dst = hp_dir / "checkpoints" / "best_model.pt"
    shutil.copy2(best_ckpt, ckpt_dst)
    scaler_dst = hp_dir / "scaler_robust.joblib"
    shutil.copy2(scaler_path, scaler_dst)

    params = int(best_trial.user_attrs.get("model_params", -1))
    param_size_kb = float(best_trial.user_attrs.get("param_size_kb", np.nan))
    ckpt_size_kb = float(ckpt_dst.stat().st_size / 1024.0)
    family = dirs["family"]

    summary = {
        "model_key": f"{family}_{dirs['version']}",
        "family": family,
        "version_base": dirs["version"],
        "version_hp": f"{dirs['version']}_hp",
        "best_trial_number": int(best_trial.number),
        "best_objective": float(study.best_value),
        "best_val_rmse": float(best_trial.user_attrs.get("best_val_rmse", np.nan)),
        "best_val_mae": float(best_trial.user_attrs.get("best_val_mae", np.nan)),
        "target_mae": target_mae,
        "mae_penalty_weight": mae_penalty_weight,
        "architecture": arch_string(family, best_cfg),
        "params": params,
        "param_size_kb_float32": param_size_kb,
        "checkpoint_size_kb": ckpt_size_kb,
        "best_params": best_trial.params,
        "best_cfg": best_cfg,
        "run_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials_total": len(study.trials),
    }
    save_json(hp_dir / "hpo_summary.json", summary)

    trial_rows = []
    for t0 in study.trials:
        trial_rows.append(
            {
                "number": t0.number,
                "state": str(t0.state),
                "value": None if t0.value is None else float(t0.value),
                "best_val_rmse": t0.user_attrs.get("best_val_rmse"),
                "best_val_mae": t0.user_attrs.get("best_val_mae"),
                "duration_sec": t0.user_attrs.get("duration_sec"),
                "params": t0.params,
            }
        )
    save_json(hp_dir / "trials.json", {"trials": trial_rows})
    save_json(run_dir / "export_summary.json", summary)

    # Mirror the final best HP model under 1_training for traceability.
    run_snapshot = (
        workspace_root
        / "DL_Models"
        / "LFP_SOH_Optimization_Study"
        / "1_training"
        / "hp_search"
        / "exports"
        / f"{dirs['family']}_{dirs['version']}_hp"
    )
    run_snapshot.mkdir(parents=True, exist_ok=True)
    save_json(run_snapshot / "hpo_summary.json", summary)
    return hp_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Optuna HPO for one compact SOH base model.")
    ap.add_argument("--model-key", choices=sorted(MODEL_SPECS.keys()), required=True)
    ap.add_argument("--workspace-root", default="/home/florianr/MG_Farm/1_Scripts")
    ap.add_argument("--study-root", default=None, help="Default: <workspace>/DL_Models/LFP_SOH_Optimization_Study/1_training/hp_search")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--early-stopping", type=int, default=10)
    ap.add_argument("--val-interval", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--max-val-batches", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-mae", type=float, default=None)
    ap.add_argument("--mae-penalty-weight", type=float, default=0.5)
    ap.add_argument("--storage", default=None, help="Optuna storage URI. Default sqlite in study-root.")
    ap.add_argument("--study-name", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    study_root = (
        Path(args.study_root).resolve()
        if args.study_root
        else workspace_root / "DL_Models" / "LFP_SOH_Optimization_Study" / "1_training" / "hp_search"
    )
    dirs = model_dirs(workspace_root, args.model_key)
    family = dirs["family"]
    version = dirs["version"]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = study_root / "runs" / f"{args.model_key}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(dirs["config"], "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["paths"]["data_root"] = expand_env_with_defaults(base_cfg["paths"]["data_root"])
    base_cfg["paths"]["out_root"] = expand_env_with_defaults(base_cfg["paths"]["out_root"])
    train_mod = load_train_module(dirs["script"])

    storage = args.storage or f"sqlite:///{(study_root / 'studies' / f'{args.model_key}.db').resolve()}"
    (study_root / "studies").mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"hpo_{args.model_key}"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=max(2, args.val_interval))
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print("=" * 100, flush=True)
    print(f"HPO start: {args.model_key} | family={family} | base={version} | device={device}", flush=True)
    print(f"study_name={study_name}", flush=True)
    print(f"storage={storage}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"target_mae={args.target_mae} | mae_penalty_weight={args.mae_penalty_weight}", flush=True)
    print("=" * 100, flush=True)

    progress_path = run_dir / "progress.json"

    def _objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        train_loader = val_loader = model = optimizer = scheduler = amp_scaler = None
        try:
            cfg = trial_cfg_update(trial, base_cfg, family, args)
            trial_dir = run_dir / "trials" / f"trial_{trial.number:04d}"
            ckpt_dir = trial_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cfg["paths"]["out_root"] = str(trial_dir)
            with open(trial_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            seed = int(args.seed + trial.number)
            set_seed(seed)

            base_features = cfg["model"]["features"]
            sampling_cfg = cfg.get("sampling", {})
            features = train_mod.expand_features_for_sampling(base_features, sampling_cfg)
            target = cfg.get("training", {}).get("target", "SOH")
            chunk = int(cfg["training"]["seq_chunk_size"])
            batch_size = int(cfg["training"]["batch_size"])

            scaler = RobustScaler()
            train_loader, val_loader = train_mod.create_dataloaders(
                cfg,
                base_features,
                features,
                target,
                chunk,
                scaler,
                batch_size=batch_size,
            )
            scaler_path = trial_dir / "scaler_robust.joblib"
            joblib_dump(scaler, scaler_path)

            model = build_model(cfg, train_mod, family, in_features=len(features)).to(device)
            model_params = int(sum(p.numel() for p in model.parameters()))
            param_size_kb = float(sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0)
            trial.set_user_attr("model_params", model_params)
            trial.set_user_attr("param_size_kb", param_size_kb)

            optimizer = train_mod.make_optimizer(
                model,
                lr=float(cfg["training"]["lr"]),
                weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
            )
            scheduler = train_mod.make_scheduler(optimizer, cfg["training"])
            amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

            epochs = int(cfg["training"]["epochs"])
            val_interval = int(cfg["training"].get("val_interval", 3))
            early_stopping = int(cfg["training"].get("early_stopping", 12))
            warmup_epochs = int(cfg["training"].get("warmup_epochs", 0))
            base_lr = float(cfg["training"]["lr"])
            loss_mode = str(cfg["training"].get("loss_mode", "seq2seq")).lower()
            warmup_steps = int(cfg["training"].get("warmup_steps", 0))
            smooth_loss_weight = float(cfg["training"].get("smooth_loss_weight", 0.0))
            smooth_loss_type = str(cfg["training"].get("smooth_loss_type", "l1")).lower()
            if warmup_steps < 0 and hasattr(model, "receptive_field"):
                warmup_steps = max(0, int(getattr(model, "receptive_field")) - 1)

            best = {"objective": float("inf"), "rmse": float("inf"), "mae": float("inf"), "epoch": -1, "path": None}
            patience = 0

            for epoch in range(1, epochs + 1):
                if warmup_epochs > 0 and epoch <= warmup_epochs:
                    warmup_lr = base_lr * (epoch / warmup_epochs)
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr

                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    device=device,
                    optimizer=optimizer,
                    amp_scaler=amp_scaler,
                    max_grad_norm=float(cfg["training"].get("max_grad_norm", 1.0)),
                    loss_mode=loss_mode,
                    warmup_steps=warmup_steps,
                    smooth_loss_weight=smooth_loss_weight,
                    smooth_loss_type=smooth_loss_type,
                    max_batches=args.max_train_batches,
                )

                if scheduler is not None and epoch > warmup_epochs:
                    scheduler.step(epoch)

                if epoch % val_interval == 0 or epoch == 1 or epoch == epochs:
                    metrics = eval_model(
                        model=model,
                        loader=val_loader,
                        device=device,
                        loss_mode=loss_mode,
                        warmup_steps=warmup_steps,
                        max_batches=args.max_val_batches,
                    )
                    score = objective_score(metrics, args.target_mae, args.mae_penalty_weight)
                    rmse = metrics.get("rmse")
                    mae = metrics.get("mae")
                    rmse_str = "nan" if rmse is None else f"{rmse:.6f}"
                    mae_str = "nan" if mae is None else f"{mae:.6f}"
                    print(
                        f"[{args.model_key}] trial={trial.number:04d}/{args.n_trials} "
                        f"epoch={epoch:03d}/{epochs} loss={train_loss:.6f} rmse={rmse_str} mae={mae_str} score={score:.6f}",
                        flush=True,
                    )

                    if score < best["objective"]:
                        best.update(
                            {
                                "objective": float(score),
                                "rmse": float(rmse if rmse is not None else np.nan),
                                "mae": float(mae if mae is not None else np.nan),
                                "epoch": int(epoch),
                            }
                        )
                        ckpt_path = ckpt_dir / f"best_epoch{epoch:04d}.pt"
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "metrics": metrics,
                                "config": cfg,
                                "trial_number": trial.number,
                            },
                            ckpt_path,
                        )
                        best["path"] = str(ckpt_path)
                        patience = 0
                    else:
                        patience += val_interval

                    trial.report(float(score), step=epoch)
                    if trial.should_prune():
                        trial.set_user_attr("duration_sec", float(time.time() - t0))
                        raise optuna.TrialPruned()

                if patience >= early_stopping:
                    break

            trial.set_user_attr("best_val_rmse", best["rmse"])
            trial.set_user_attr("best_val_mae", best["mae"])
            trial.set_user_attr("best_epoch", best["epoch"])
            trial.set_user_attr("best_ckpt", best["path"])
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("duration_sec", float(time.time() - t0))
            save_json(
                trial_dir / "trial_summary.json",
                {
                    "trial_number": trial.number,
                    "objective": best["objective"],
                    "best_val_rmse": best["rmse"],
                    "best_val_mae": best["mae"],
                    "best_epoch": best["epoch"],
                    "best_ckpt": best["path"],
                    "params": trial.params,
                    "duration_sec": float(time.time() - t0),
                },
            )
            return float(best["objective"])
        except RuntimeError as e:
            emsg = str(e).lower()
            if "out of memory" in emsg or "cuda error" in emsg:
                trial.set_user_attr("runtime_error", str(e))
                raise optuna.TrialPruned()
            raise
        finally:
            del train_loader, val_loader, model, optimizer, scheduler, amp_scaler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _callback(study_: optuna.Study, trial_: optuna.trial.FrozenTrial) -> None:
        try:
            best = study_.best_trial
        except ValueError:
            best = None
        save_json(
            progress_path,
            {
                "model_key": args.model_key,
                "family": family,
                "version_base": version,
                "run_dir": str(run_dir),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_trials_requested": args.n_trials,
                "n_trials_total": len(study_.trials),
                "last_trial_number": trial_.number,
                "last_trial_state": str(trial_.state),
                "last_trial_value": trial_.value,
                "best_trial_number": None if best is None else best.number,
                "best_value": None if best is None else best.value,
                "best_val_rmse": None if best is None else best.user_attrs.get("best_val_rmse"),
                "best_val_mae": None if best is None else best.user_attrs.get("best_val_mae"),
            },
        )

    study.optimize(_objective, n_trials=args.n_trials, callbacks=[_callback], gc_after_trial=True)

    best_trial = study.best_trial
    best_ckpt = Path(str(best_trial.user_attrs["best_ckpt"]))
    best_trial_dir = Path(str(best_trial.user_attrs["trial_dir"]))
    best_cfg_path = best_trial_dir / "config_resolved.yaml"
    if not best_cfg_path.exists():
        # Recover best cfg from checkpoint if helper file does not exist.
        state = torch.load(best_ckpt, map_location="cpu")
        best_cfg = state.get("config")
    else:
        with open(best_cfg_path, "r", encoding="utf-8") as f:
            best_cfg = yaml.safe_load(f)

    if best_cfg is None:
        # fallback
        best_cfg = trial_cfg_update(optuna.trial.FixedTrial(best_trial.params), base_cfg, family, args)

    scaler_path = best_trial_dir / "scaler_robust.joblib"
    hp_dir = export_best_to_hp_dir(
        workspace_root=workspace_root,
        dirs=dirs,
        run_dir=run_dir,
        best_trial=best_trial,
        best_cfg=best_cfg,
        best_ckpt=best_ckpt,
        scaler_path=scaler_path,
        study=study,
        target_mae=args.target_mae,
        mae_penalty_weight=args.mae_penalty_weight,
    )

    save_json(
        run_dir / "final_summary.json",
        {
            "model_key": args.model_key,
            "best_trial_number": best_trial.number,
            "best_value": study.best_value,
            "best_val_rmse": best_trial.user_attrs.get("best_val_rmse"),
            "best_val_mae": best_trial.user_attrs.get("best_val_mae"),
            "hp_dir": str(hp_dir),
            "run_dir": str(run_dir),
        },
    )

    print("=" * 100, flush=True)
    print(
        f"HPO done: {args.model_key} | best_trial={best_trial.number} | score={study.best_value:.6f} "
        f"| rmse={best_trial.user_attrs.get('best_val_rmse')} | mae={best_trial.user_attrs.get('best_val_mae')}",
        flush=True,
    )
    print(f"Exported HP model -> {hp_dir}", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
