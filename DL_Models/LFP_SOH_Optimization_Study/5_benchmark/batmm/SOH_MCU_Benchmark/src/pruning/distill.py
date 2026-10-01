"""
distill.py — Unified knowledge distillation entry-point for all architectures.

Mirrors the structure of prune.py and quantize.py: this is the script the
user calls directly.  Architecture-specific model factories and pruning
metadata are dispatched via --model_name.

Usage examples
--------------
# Short recovery distillation (finetune mode):
python distill.py --model_name lstm \\
    --teacher_checkpoint experiments/lstm_best.pt \\
    --student_checkpoint  experiments/pruning/lstm_pruned_oneshot_0.7123.pt \\
    --mode finetune

# Full retraining distillation:
python distill.py --model_name gru \\
    --teacher_checkpoint experiments/gru_best.pt \\
    --student_checkpoint  experiments/pruning/gru_pruned_iterative_0.6541.pt \\
    --mode full --full_epochs 100 --full_lr 3e-4

# Run both modes independently and save each:
python distill.py --model_name tcn \\
    --teacher_checkpoint experiments/tcn_best.pt \\
    --student_checkpoint  experiments/pruning/tcn_pruned_oneshot_0.6812.pt \\
    --mode both
"""

import os
import argparse
import joblib
import torch
import torch.nn as nn
import torch_pruning as tp

from config import DATA_DIR, RESULTS_DIR

from src.utils.data_utils import (
    FEATURES, build_dataloaders
)

from src.pruning.distillation_backend import DistillConfig, run_distillation_pipeline

import src.utils.lstm_utils as lstm_u
import src.utils.gru_utils  as gru_u
import src.utils.tcn_utils  as tcn_u
import src.utils.cnn_utils  as cnn_u


# ============================================================
# ARGUMENT PARSER
# ============================================================

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Knowledge distillation: teacher (full) → student (pruned)."
    )

    # ── Identity ──────────────────────────────────────────────────────────────
    p.add_argument("--model_name", type=str, required=True,
                   choices=["lstm", "gru", "cnn", "tcn"],
                   help="Architecture shared by teacher and student.")

    # ── Checkpoints ───────────────────────────────────────────────────────────
    p.add_argument("--teacher_checkpoint", type=str, default=None,
                   help="Path to the full (unpruned) teacher .pt file. "
                        "Falls back to the architecture default checkpoint.")
    p.add_argument("--student_checkpoint", type=str, required=True,
                   help="Path to the pruned student .pt file "
                        "(output of prune.py).")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data_root",      default=DATA_DIR)
    p.add_argument("--scaler",         default=None,
                   help="Path to scaler .pkl. Falls back to arch default.")
    p.add_argument("--batch_size",     type=int, default=128)
    p.add_argument("--seq_chunk_size", type=int, default=1)

    # ── Distillation mode ─────────────────────────────────────────────────────
    p.add_argument("--mode", choices=["finetune", "full", "both"], default="finetune",
                   help=(
                       "finetune : short recovery run (mirrors post-pruning fine-tuning). "
                       "full     : full retraining run with warm-up + cosine schedule. "
                       "both     : run finetune then full independently; saves each."
                   ))

    # ── Loss ──────────────────────────────────────────────────────────────────
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight on soft (teacher) loss. "
                        "L = alpha*L_soft + (1-alpha)*L_hard.  Default: 0.5")

    # ── Finetune-mode hyperparameters ─────────────────────────────────────────
    p.add_argument("--finetune_epochs", type=int,   default=15,
                   help="Epochs for 'finetune' mode.")
    p.add_argument("--lr",             type=float, default=1e-4,
                   help="Learning rate for 'finetune' mode.")

    # ── Full-mode hyperparameters ─────────────────────────────────────────────
    p.add_argument("--full_epochs",       type=int,   default=80,
                   help="Epochs for 'full' mode.")
    p.add_argument("--full_lr",           type=float, default=3e-4,
                   help="Peak learning rate for 'full' mode.")
    p.add_argument("--warmup_epoch_frac", type=float, default=0.05,
                   help="Fraction of full_epochs used for linear LR warm-up. Default: 0.05")

    # ── Shared regularisation ─────────────────────────────────────────────────
    p.add_argument("--weight_decay",       type=float, default=1e-4)
    p.add_argument("--max_grad_norm",      type=float, default=1.0)
    p.add_argument("--smooth_loss_weight", type=float, default=0.05,
                   help="Weight of smoothness regulariser on student predictions.")
    p.add_argument("--smooth_loss_type",   choices=["l1", "l2"], default="l1")
    p.add_argument("--warmup_steps",       type=int, default=8,
                   help="Leading time-steps excluded from loss (same as pruning).")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--out_dir",    default=os.path.join(RESULTS_DIR, "distillation"))
    p.add_argument("--csv_output", default=os.path.join(RESULTS_DIR, "distillation/distillation_results.csv"))
    p.add_argument("--device",     default="auto")

    return p


# ============================================================
# ARCHITECTURE REGISTRY
# ============================================================

# Full (teacher) model factories
_TEACHER_FACTORIES = {
    "lstm": lstm_u.create_lstm,
    "gru":  gru_u.create_gru,
    "tcn":  tcn_u.create_tcn,
    "cnn":  cnn_u.create_cnn,
}

# Pruned (student) model factories — same architecture, same hyperparams;
# structural resizing to the pruned checkpoint is handled by the backend.
_STUDENT_FACTORIES = {
    "lstm": lstm_u.create_lstm,
    "gru":  gru_u.create_gru,
    "tcn":  tcn_u.create_tcn,
    "cnn":  cnn_u.create_cnn,
}

_DEFAULT_CHECKPOINTS = {
    "lstm": lstm_u.CHECKPOINT_PATH,
    "gru":  gru_u.CHECKPOINT_PATH,
    "tcn":  tcn_u.CHECKPOINT_PATH,
    "cnn":  cnn_u.CHECKPOINT_PATH,
}

_DEFAULT_SCALERS = {
    "lstm": lstm_u.SCALER_PATH,
    "gru":  gru_u.SCALER_PATH,
    "tcn":  tcn_u.SCALER_PATH,
    "cnn":  cnn_u.SCALER_PATH,
}

# Sequence lengths per architecture (must match training / quantize.py)
_SEQ_LENS = {
    "lstm": 192,
    "gru":  168,
    "tcn":  120,
    "cnn":   96,
}

# Layer / pruning metadata needed to sync a pruned student checkpoint
_LAYER_META = {
    "lstm": (nn.Linear,  tp.prune_linear_out_channels,  "out_features"),
    "gru":  (nn.Linear,  tp.prune_linear_out_channels,  "out_features"),
    "tcn":  (nn.Conv1d,  tp.prune_conv_out_channels,    "out_channels"),
    "cnn":  (nn.Conv1d,  tp.prune_conv_out_channels,    "out_channels"),
}


# ============================================================
# MAIN
# ============================================================

def main():
    parser = get_parser()
    args   = parser.parse_args()

    # ── Resolve defaults ───────────────────────────────────────────────────────
    if not args.teacher_checkpoint:
        args.teacher_checkpoint = _DEFAULT_CHECKPOINTS[args.model_name]
    if not args.scaler:
        args.scaler = _DEFAULT_SCALERS[args.model_name]

    print(f"Starting {args.model_name.upper()} distillation pipeline  [mode={args.mode}]")
    print(f"  Teacher  : {args.teacher_checkpoint}")
    print(f"  Student  : {args.student_checkpoint}")
    print(f"  Alpha    : {args.alpha}  (soft={args.alpha:.2f}, hard={1-args.alpha:.2f})")

    # ── Data ──────────────────────────────────────────────────────────────────
    scaler = joblib.load(args.scaler)
    train_loader, val_loader, test_loaders = build_dataloaders(
        args.data_root, scaler,
        chunk      = args.seq_chunk_size,
        batch_size = args.batch_size,
    )

    # ── Attach data loaders to args so the backend can access them ────────────
    # (mirrors how prune.py passes loaders directly into prune_model)
    args._train_loader  = train_loader
    args._val_loader    = val_loader
    args._test_loaders  = test_loaders

    # ── Attach pruning metadata to args (used by backend for arch sync) ───────
    layer_type, prune_fn, size_attr = _LAYER_META[args.model_name]
    args._layer_type = layer_type
    args._prune_fn   = prune_fn
    args._size_attr  = size_attr

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = DistillConfig(
        model_name      = args.model_name,
        teacher         = _TEACHER_FACTORIES[args.model_name](),
        student         = _STUDENT_FACTORIES[args.model_name](),
        in_features     = len(FEATURES),
        example_seq_len = _SEQ_LENS[args.model_name],
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    run_distillation_pipeline(cfg, args)


if __name__ == "__main__":
    main()
