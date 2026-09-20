import torch
import torch.nn as nn
import joblib
import os
import argparse

from src.pruning.pruning_backend import prune_model
from src.utils.data_utils import FEATURES, build_dataloaders, load_finetune_hyperparameters
from config import RESULTS_DIR, DATA_DIR

import src.utils.lstm_utils as lstm_u
import src.utils.gru_utils as gru_u
import src.utils.tcn_utils as tcn_u
import src.utils.cnn_utils as cnn_u


# ===========
#   CONFIGS
# ===========

def get_parser():
    """Shared argument parser with architecture-specific defaults."""
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",       type=str, required=True,
                   choices=["lstm", "gru", "cnn", "tcn"])
    p.add_argument("--data_root",        default=DATA_DIR)
    p.add_argument("--checkpoint",       default=None)
    p.add_argument("--scaler",           default=None)
    p.add_argument("--pruning_ratio",    type=float, default=0.30,
                   help="Total pruning ratio (for one-shot) or per-iteration ratio (for iterative)")
    p.add_argument("--pruning_mode",     choices=["oneshot", "iterative"], default="oneshot")
    p.add_argument("--num_iterations",   type=int, default=5)
    p.add_argument("--finetune_epochs",  type=int,   default=15)
    p.add_argument("--finetune_lr",      type=float, default=1e-4)
    p.add_argument("--distill_teacher", type=str, default=None,
               help="Path to the full (unpruned) teacher checkpoint for KD-assisted "
                    "fine-tuning. If omitted, plain MSE fine-tuning is used.")
    p.add_argument("--distill_alpha",   type=float, default=0.5,
               help="Weight on soft (teacher) loss when --distill_teacher is set. "
                    "L = alpha*L_soft + (1-alpha)*L_hard.  Default: 0.5")
    p.add_argument("--batch_size",       type=int,   default=128)
    p.add_argument("--seq_chunk_size",   type=int,   default=1)
    p.add_argument("--out_dir",          default=os.path.join(RESULTS_DIR, "pruning/"))
    p.add_argument("--csv_output",       default=os.path.join(RESULTS_DIR, "pruning/pruning_results.csv"))
    p.add_argument("--device",           default="auto")
    return p


def get_arch_config(model_name: str):
    """Returns architecture-specific creation functions and default paths."""
    configs = {
        "lstm": (lstm_u.create_lstm, lstm_u.SCALER_PATH, lstm_u.CHECKPOINT_PATH),
        "gru":  (gru_u.create_gru,   gru_u.SCALER_PATH, gru_u.CHECKPOINT_PATH),
        "tcn":  (tcn_u.create_tcn,   tcn_u.SCALER_PATH, tcn_u.CHECKPOINT_PATH),
        "cnn":  (cnn_u.create_cnn,   cnn_u.SCALER_PATH, cnn_u.CHECKPOINT_PATH),
    }
    if model_name not in configs:
        raise ValueError(f"Unknown model architecture: {model_name}")
    return configs[model_name]


def get_stateful_configs(submodule, model):
    # 1. Protect the recurrent module itself (LSTM/GRU)
    ignored = list(submodule.modules())
    # 2. Protect the specific layers shared by both architectures
    ignored.append(model.feature_proj[4])
    for blk in model.res_blocks:
        ignored.extend([blk.fc2, blk.norm])
    ignored.append(model.head[6])

    # 3. Clean up duplicates and return
    final_ignored = list({id(m): m for m in ignored}.values())

    # 4. Create the lambda functions for the submodule
    f_fn = lambda: [p.requires_grad_(False) for p in submodule.parameters()]
    u_fn = lambda: [p.requires_grad_(True) for p in submodule.parameters()]

    return final_ignored, f_fn, u_fn


# =================================
#   ARCHITECTURE-SPECIFIC PRUNING
# =================================

def get_configs(model_name, model):
    if model_name == "lstm":
        return get_stateful_configs(model.lstm, model)

    elif model_name == "gru":
        return get_stateful_configs(model.gru, model)

    elif model_name == "tcn":
        ignored_layers = []
        first_block = model.tcn[0]
        ignored_layers.append(first_block.conv1.conv)
        if first_block.downsample is not None: ignored_layers.append(first_block.downsample)
        for m in model.modules():
            if isinstance(m, nn.Conv1d) and m.out_channels == 1: ignored_layers.append(m)
        return ignored_layers, None, None

    elif model_name == "cnn":
        return [m for m in model.modules() if isinstance(m, nn.Conv1d) and m.out_channels == 1], None, None


# ========
#   MAIN
# ========

def main(args_list=None):
    # 1. Setup Parser
    parser = get_parser()
    args = parser.parse_args(args_list)

    # 2. Get Model-Specific Factory and Defaults
    create_fn, def_scaler, def_ckpt = get_arch_config(args.model_name)

    # Override defaults if not provided in CLI
    scaler_path = args.scaler if args.scaler else def_scaler
    ckpt_path = args.checkpoint if args.checkpoint else def_ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    print(f"Starting {args.model_name.upper()} pruning pipeline on {device}...")

    # 3. Initialize Model and Load Weights
    model = create_fn()

    print(model)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model = model.to(device).eval()

    # 4. Load Data
    scaler = joblib.load(scaler_path)
    train_loader, val_loader, test_loaders = build_dataloaders(
        args.data_root, scaler, chunk=args.seq_chunk_size, batch_size=args.batch_size
    )
    example_inputs = torch.randn(1, args.seq_chunk_size, len(FEATURES)).to(device)

    # 5. Architecture-Specific Pruning Logic
    ignored_layers, freeze_fn, unfreeze_fn = get_configs(args.model_name, model)

    teacher = None
    if args.distill_teacher:
        print(f"\nLoading teacher for distillation-assisted pruning: {args.distill_teacher}")
        teacher_ckpt = torch.load(args.distill_teacher, map_location=device, weights_only=False)
        teacher_sd   = teacher_ckpt.get("model_state_dict", teacher_ckpt.get("state_dict", teacher_ckpt))
        teacher = create_fn()
        teacher.load_state_dict(teacher_sd, strict=True)
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"  Teacher loaded and frozen. Alpha={args.distill_alpha} (soft={args.distill_alpha:.2f}, hard={1-args.distill_alpha:.2f})")

    # 6. Run Unified Pruning Pipeline
    pruned_model, pruned_model_name = prune_model(
        model_name=args.model_name,
        model=model,
        args=args,
        device=device,
        example_inputs=example_inputs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loaders=test_loaders,
        ignored_layers=ignored_layers,
        freeze_fn=freeze_fn,
        unfreeze_fn=unfreeze_fn,
        teacher=teacher,
        alpha=args.distill_alpha,
    )

    return pruned_model, pruned_model_name

if __name__ == "__main__":
    main()
