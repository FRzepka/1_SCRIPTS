#!/usr/bin/env python3
"""Fine-tune a pruned SOH model checkpoint to recover accuracy/offset."""
import argparse
import importlib.util
import inspect
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def expand_env_with_defaults(val: str) -> str:
    pattern = re.compile(r"\$\{([^:}]+):-([^}]+)\}")

    def repl(match):
        var, default = match.group(1), match.group(2)
        env_val = os.getenv(var)
        return env_val if env_val not in (None, "") else default

    if not isinstance(val, str):
        return val
    return os.path.expandvars(pattern.sub(repl, val))


def load_train_module(train_py: Path):
    spec = importlib.util.spec_from_file_location("train_soh_module", str(train_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def build_model(cfg: dict, train_mod):
    base_features = cfg["model"]["features"]
    sampling_cfg = cfg.get("sampling", {})
    if hasattr(train_mod, "expand_features_for_sampling"):
        features = train_mod.expand_features_for_sampling(base_features, sampling_cfg)
    else:
        features = base_features

    mtype = str(cfg["model"].get("type", "")).lower()
    if "lstm" in mtype:
        cls = train_mod.SOH_LSTM_Seq2Seq
        model = cls(
            in_features=len(features),
            embed_size=int(cfg["model"].get("embed_size", 96)),
            hidden_size=int(cfg["model"].get("hidden_size", 160)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 128)),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            res_blocks=int(cfg["model"].get("res_blocks", 0)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
    elif "gru" in mtype:
        cls = train_mod.SOH_GRU_Seq2Seq
        model = cls(
            in_features=len(features),
            embed_size=int(cfg["model"].get("embed_size", 96)),
            hidden_size=int(cfg["model"].get("hidden_size", 160)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 128)),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            res_blocks=int(cfg["model"].get("res_blocks", 0)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
    elif "tcn" in mtype:
        cls = train_mod.CausalTCN_SOH
        dilations = cfg["model"].get("dilations") or [1, 2, 4, 8]
        model = cls(
            in_features=len(features),
            hidden_size=int(cfg["model"].get("hidden_size", 128)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 96)),
            kernel_size=int(cfg["model"].get("kernel_size", 3)),
            dilations=[int(d) for d in dilations],
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
    elif "cnn" in mtype:
        cls = train_mod.SOH_CNN_Seq2Seq
        dilations = cfg["model"].get("dilations")
        kwargs = dict(
            in_features=len(features),
            hidden_size=int(cfg["model"].get("hidden_size", 128)),
            mlp_hidden=int(cfg["model"].get("mlp_hidden", 96)),
            kernel_size=int(cfg["model"].get("kernel_size", 5)),
            dilations=[int(d) for d in dilations] if dilations is not None else None,
            num_blocks=int(cfg["model"].get("num_blocks", 4)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
        if "output_kernel_size" in inspect.signature(cls).parameters:
            kwargs["output_kernel_size"] = int(cfg["model"].get("output_kernel_size", 1))
        model = cls(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {cfg['model'].get('type')}")

    return model, features


def parse_state(raw_state):
    if isinstance(raw_state, dict) and "model_state_dict" in raw_state:
        return raw_state["model_state_dict"]
    return raw_state


def freeze_for_head_only(model: torch.nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        keep = name.startswith("head.")
        param.requires_grad = keep
        if keep:
            trainable += param.numel()
    return trainable


def compute_offset_diagnostics(y_true: np.ndarray, y_base: np.ndarray, y_hat: np.ndarray):
    n = min(len(y_true), len(y_base), len(y_hat))
    if n == 0:
        return {"n": 0, "mean_diff": float("nan"), "std_diff": float("nan"), "corr": float("nan")}
    y_true = y_true[:n]
    y_base = y_base[:n]
    y_hat = y_hat[:n]
    diff = y_hat - y_base
    corr = float(np.corrcoef(y_base, y_hat)[0, 1]) if n > 2 else float("nan")
    return {
        "n": n,
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "mae_vs_base": float(np.mean(np.abs(diff))),
        "mae": float(mean_absolute_error(y_true, y_hat)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_hat))),
        "r2": float(r2_score(y_true, y_hat)) if np.var(y_true) > 0 else float("nan"),
        "corr": corr,
    }


def run_finetune(
    model_dir: Path,
    out_dir: Path,
    ckpt_path: Optional[Path],
    epochs: int,
    lr: float,
    weight_decay: float,
    head_only: bool,
    device_str: str,
    val_interval: int,
    early_stopping: int,
    max_batches: Optional[int],
    num_workers: int,
    prefetch_factor: int,
) -> None:
    cfg_path = model_dir / "config" / "train_soh.yaml"
    train_py = model_dir / "scripts" / "train_soh.py"
    scaler_path = model_dir / "scaler_robust.joblib"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not train_py.exists():
        raise FileNotFoundError(f"Missing train script: {train_py}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["data_root"] = expand_env_with_defaults(cfg["paths"]["data_root"])
    cfg["paths"]["out_root"] = str(out_dir)
    cfg.setdefault("dataloader", {})
    cfg["dataloader"]["num_workers"] = int(num_workers)
    cfg["dataloader"]["prefetch_factor"] = int(prefetch_factor)
    cfg["dataloader"]["persistent_workers"] = bool(num_workers > 0)
    cfg["dataloader"]["pin_memory"] = True

    train_mod = load_train_module(train_py)
    model, features = build_model(cfg, train_mod)

    base_features = cfg["model"]["features"]
    target = cfg.get("training", {}).get("target", "SOH")
    chunk = int(cfg["training"].get("seq_chunk_size", 168))
    batch_size = int(cfg["training"].get("batch_size", 96))
    accum_steps = int(cfg["training"].get("accum_steps", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    warmup_steps = int(cfg["training"].get("warmup_steps", 0))
    loss_mode = str(cfg["training"].get("loss_mode", "seq2seq")).lower()
    smooth_loss_weight = float(cfg["training"].get("smooth_loss_weight", 0.0))
    smooth_loss_type = str(cfg["training"].get("smooth_loss_type", "l1")).lower()

    if ckpt_path is None:
        candidates = [
            model_dir / "checkpoints" / "best_model_pruned.pt",
            model_dir / "checkpoints" / "best_model.pt",
            model_dir / "checkpoints" / "final_model.pt",
        ]
        ckpt_path = next((p for p in candidates if p.exists()), None)
        if ckpt_path is None:
            pts = sorted((model_dir / "checkpoints").glob("*.pt"))
            if not pts:
                raise FileNotFoundError(f"No checkpoints in {model_dir / 'checkpoints'}")
            ckpt_path = pts[0]

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(parse_state(state))

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if head_only:
        trainable_params = freeze_for_head_only(model)
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    scaler = joblib_load(scaler_path)

    train_loader, val_loader = train_mod.create_dataloaders(
        cfg,
        base_features,
        features,
        target,
        chunk,
        scaler,
        batch_size=batch_size,
    )

    optimizer = train_mod.make_optimizer(model, lr=lr, weight_decay=weight_decay)
    amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    for sub in ("config", "scripts", "test"):
        src = model_dir / sub
        dst = out_dir / sub
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    best_rmse = float("inf")
    best_epoch = -1
    best_ckpt = out_dir / "checkpoints" / "best_model_finetuned.pt"
    patience = 0
    history = []

    # Baseline diagnostics (before fine-tuning) on validation split.
    base_metrics, y_true_base, y_pred_base = train_mod.eval_model(
        model, val_loader, device, loss_mode=loss_mode, warmup_steps=warmup_steps, max_batches=max_batches
    )

    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_mod.train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler_amp=amp_scaler,
            max_grad_norm=max_grad_norm,
            epoch_idx=epoch,
            total_epochs=epochs,
            accum_steps=accum_steps,
            loss_mode=loss_mode,
            warmup_steps=warmup_steps,
            smooth_loss_weight=smooth_loss_weight,
            smooth_loss_type=smooth_loss_type,
            max_batches=max_batches,
        )

        if (epoch % val_interval == 0) or (epoch == epochs):
            metrics, y_true, y_pred = train_mod.eval_model(
                model, val_loader, device, loss_mode=loss_mode, warmup_steps=warmup_steps, max_batches=max_batches
            )
            rmse = float(metrics.get("rmse", float("inf")))
            mae = float(metrics.get("mae", float("nan")))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_rmse": rmse,
                    "val_mae": mae,
                }
            )
            print(
                f"[FT {epoch:03d}/{epochs}] train_loss={float(train_loss):.6f} "
                f"val_rmse={rmse:.6f} val_mae={mae:.6f}"
            )
            if rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch
                patience = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                    },
                    best_ckpt,
                )
            else:
                patience += val_interval
                if patience >= early_stopping:
                    print(f"[FT] Early stopping at epoch {epoch}.")
                    break

    elapsed = time.time() - start

    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(parse_state(best_state))

    final_metrics, y_true_final, y_pred_final = train_mod.eval_model(
        model, val_loader, device, loss_mode=loss_mode, warmup_steps=warmup_steps, max_batches=max_batches
    )

    offset_before = compute_offset_diagnostics(np.asarray(y_true_base), np.asarray(y_pred_base), np.asarray(y_pred_base))
    offset_after = compute_offset_diagnostics(np.asarray(y_true_base), np.asarray(y_pred_base), np.asarray(y_pred_final))

    # Save scaler actually used by datasets (refit inside create_dataloaders).
    joblib_dump(scaler, out_dir / "scaler_robust.joblib")

    final_ckpt = out_dir / "checkpoints" / "final_model_finetuned.pt"
    torch.save(
        {
            "epoch": best_epoch if best_epoch > 0 else epochs,
            "model_state_dict": model.state_dict(),
            "metrics": final_metrics,
            "history": history,
        },
        final_ckpt,
    )

    meta = {
        "source_model_dir": str(model_dir),
        "source_checkpoint": str(ckpt_path),
        "best_checkpoint": str(best_ckpt),
        "final_checkpoint": str(final_ckpt),
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "head_only": head_only,
        "trainable_params": int(trainable_params),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_rmse),
        "final_metrics": final_metrics,
        "offset_before_vs_base": offset_before,
        "offset_after_vs_base": offset_after,
        "elapsed_sec": float(elapsed),
    }
    with open(out_dir / "finetune_meta.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(meta), f, indent=2)

    print(f"[FT] done: {out_dir}")
    print(f"[FT] best_epoch={best_epoch}, best_rmse={best_rmse:.6f}, trainable={trainable_params:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune pruned SOH checkpoint.")
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--val-interval", type=int, default=1)
    ap.add_argument("--early-stopping", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--head-only", action="store_true", default=False)
    args = ap.parse_args()

    run_finetune(
        model_dir=Path(args.model_dir),
        out_dir=Path(args.out_dir),
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        head_only=bool(args.head_only),
        device_str=args.device,
        val_interval=int(args.val_interval),
        early_stopping=int(args.early_stopping),
        max_batches=args.max_batches,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
    )


if __name__ == "__main__":
    main()
