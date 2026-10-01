import os
import re
import json
import typing as T

import torch
import torch.nn as nn
import torch_pruning as tp

from config import TEST_CELLS
from src.utils.metrics import (
    evaluate, measure_inference_time, get_model_size_mb, append_metrics_to_csv
)


# ============================================================
# DISTILLATION CONFIG
# ============================================================

class DistillConfig(T.NamedTuple):
    model_name:     str
    teacher:        nn.Module   # full-size, frozen
    student:        nn.Module   # pruned, to be trained
    in_features:    int
    example_seq_len: int


# ============================================================
# HELPERS
# ============================================================

def _parse_pruning_info(
    checkpoint_path: str,
    model_name: str,
) -> T.Tuple[T.Optional[str], T.Optional[float]]:
    """Extracts pruning style and ratio from a pruned checkpoint filename."""
    filename = os.path.basename(checkpoint_path)
    match = re.search(
        rf'{re.escape(model_name)}_pruned_(oneshot|iterative)_([\d.]+)\.pt',
        filename,
    )
    if match:
        return match.group(1), float(match.group(2))
    return None, None


def _avg(results: T.List[dict], key: str) -> float:
    return sum(r[key] for r in results) / len(results)


# ============================================================
# DISTILLATION LOSS
# ============================================================

def distillation_loss(
    student_pred:  torch.Tensor,
    teacher_pred:  torch.Tensor,
    targets:       torch.Tensor,
    alpha:         float,
    warmup_steps:  int = 0,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the combined distillation loss.

    Parameters
    ----------
    student_pred  : (B, T, 1)  raw student output
    teacher_pred  : (B, T, 1)  frozen teacher output (no grad)
    targets       : (B, T, 1)  ground-truth labels
    alpha         : weight on L_soft  (1-alpha weights L_hard)
    warmup_steps  : leading time-steps excluded from loss (same as pruning)

    Returns
    -------
    total_loss, l_soft, l_hard
    """
    mse = nn.MSELoss()

    s = student_pred[:, warmup_steps:]
    t = teacher_pred[:, warmup_steps:]
    y = targets[:,      warmup_steps:]

    l_soft = mse(s, t.detach())
    l_hard = mse(s, y)
    total  = alpha * l_soft + (1.0 - alpha) * l_hard

    return total, l_soft, l_hard


# ============================================================
# TRAINING LOOPS
# ============================================================

def train_one_epoch(
    student:       nn.Module,
    teacher:       nn.Module,
    loader:        torch.utils.data.DataLoader,
    optimizer:     torch.optim.Optimizer,
    device:        torch.device,
    alpha:         float,
    warmup_steps:  int,
    max_grad_norm: float,
    smooth_loss_weight: float,
    smooth_loss_type:   str,
) -> T.Tuple[float, float, float]:
    """Single epoch of distillation training. Returns (total, soft, hard) losses."""
    student.train()
    teacher.eval()

    total_loss_sum = soft_loss_sum = hard_loss_sum = 0.0
    n = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        with torch.no_grad():
            teacher_pred = teacher(xb)

        optimizer.zero_grad(set_to_none=True)
        student_pred = student(xb)

        loss, l_soft, l_hard = distillation_loss(
            student_pred, teacher_pred, yb, alpha, warmup_steps
        )

        # Optional smoothness regulariser (same as pruning fine-tune)
        if smooth_loss_weight > 0:
            p_sel = student_pred[:, warmup_steps:]
            if p_sel.size(1) > 1:
                diffs  = p_sel[:, 1:] - p_sel[:, :-1]
                smooth = (diffs ** 2).mean() if smooth_loss_type == "l2" else diffs.abs().mean()
                loss   = loss + smooth_loss_weight * smooth

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
        optimizer.step()

        total_loss_sum += loss.item()
        soft_loss_sum  += l_soft.item()
        hard_loss_sum  += l_hard.item()
        n += 1

    denom = max(1, n)
    return total_loss_sum / denom, soft_loss_sum / denom, hard_loss_sum / denom


def distill_finetune(
    student:            nn.Module,
    teacher:            nn.Module,
    train_loader:       torch.utils.data.DataLoader,
    val_loader:         torch.utils.data.DataLoader,
    device:             torch.device,
    epochs:             int,
    lr:                 float,
    alpha:              float,
    weight_decay:       float       = 1e-4,
    max_grad_norm:      float       = 1.0,
    smooth_loss_weight: float       = 0.05,
    smooth_loss_type:   str         = "l1",
    warmup_steps:       int         = 8,
) -> T.List[dict]:
    """
    Short recovery distillation — mirrors post-pruning fine-tuning length.
    Cosine-annealed LR from `lr` down to lr*0.01.
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    history = []

    for epoch in range(1, epochs + 1):
        total, l_soft, l_hard = train_one_epoch(
            student, teacher, train_loader, optimizer, device,
            alpha, warmup_steps, max_grad_norm, smooth_loss_weight, smooth_loss_type,
        )
        scheduler.step()
        val_metrics = evaluate(student, val_loader, device, warmup_steps)

        row = {
            "epoch":      epoch,
            "train_loss": total,
            "soft_loss":  l_soft,
            "hard_loss":  l_hard,
            **val_metrics,
        }
        history.append(row)
        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"loss={total:.5f}  soft={l_soft:.5f}  hard={l_hard:.5f}  "
            f"val_mae={val_metrics['mae']:.4f}  val_rmse={val_metrics['rmse']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    return history


def distill_full(
    student:            nn.Module,
    teacher:            nn.Module,
    train_loader:       torch.utils.data.DataLoader,
    val_loader:         torch.utils.data.DataLoader,
    device:             torch.device,
    epochs:             int,
    lr:                 float,
    alpha:              float,
    weight_decay:       float       = 1e-4,
    max_grad_norm:      float       = 1.0,
    smooth_loss_weight: float       = 0.05,
    smooth_loss_type:   str         = "l1",
    warmup_steps:       int         = 8,
    warmup_epoch_frac:  float       = 0.05,
) -> T.List[dict]:
    """
    Full retraining distillation run.

    LR schedule: linear warm-up for the first `warmup_epoch_frac` fraction of
    epochs, then cosine annealing to lr*0.001.  This is more aggressive than
    the finetune schedule to allow the student to fully reorganise its weights.
    """
    warmup_epochs = max(1, int(epochs * warmup_epoch_frac))
    cosine_epochs = epochs - warmup_epochs

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_sched  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cosine_epochs), eta_min=lr * 0.001
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    history = []

    for epoch in range(1, epochs + 1):
        phase = "warm-up" if epoch <= warmup_epochs else "cosine"
        total, l_soft, l_hard = train_one_epoch(
            student, teacher, train_loader, optimizer, device,
            alpha, warmup_steps, max_grad_norm, smooth_loss_weight, smooth_loss_type,
        )
        scheduler.step()
        val_metrics = evaluate(student, val_loader, device, warmup_steps)

        row = {
            "epoch":      epoch,
            "phase":      phase,
            "train_loss": total,
            "soft_loss":  l_soft,
            "hard_loss":  l_hard,
            **val_metrics,
        }
        history.append(row)
        print(
            f"  Epoch {epoch:3d}/{epochs} [{phase:7s}]  "
            f"loss={total:.5f}  soft={l_soft:.5f}  hard={l_hard:.5f}  "
            f"val_mae={val_metrics['mae']:.4f}  val_rmse={val_metrics['rmse']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    return history


# ============================================================
# ARCHITECTURE SYNC  (same helper as quant_backend)
# ============================================================

def _match_student_to_checkpoint(
    student:     nn.Module,
    state_dict:  dict,
    layer_type:  type,
    prune_fn:    T.Callable,
    size_attr:   str,
    seq_len:     int,
    in_features: int,
) -> bool:
    """
    Structurally resizes *student* layers to match the shapes stored in
    *state_dict*, using torch-pruning.  No-op when the checkpoint was not
    pruned (shapes already match).
    """
    student.eval()
    example_inputs = torch.randn(1, seq_len, in_features)
    dg = tp.DependencyGraph().build_dependency(student, example_inputs=example_inputs)

    print("Syncing student architecture to checkpoint shapes...")
    pruning_applied = False

    for name, module in student.named_modules():
        if not isinstance(module, layer_type):
            continue
        weight_key = f"{name}.weight"
        if weight_key not in state_dict:
            continue

        saved_out   = state_dict[weight_key].shape[0]
        current_out = getattr(module, size_attr)

        if saved_out < current_out:
            indices = list(range(current_out - saved_out))
            try:
                pruning_group = dg.get_pruning_group(module, prune_fn, idxs=indices)
                pruning_group.exec()
                print(f"  Adjusted {name}: {current_out} -> {saved_out}")
                pruning_applied = True
            except Exception as exc:
                print(f"  Warning: could not adjust {name} — {exc}. Skipping.")

    if not pruning_applied:
        print("  No architectural resizing was necessary.")

    return pruning_applied


# ============================================================
# SHARED PIPELINE
# ============================================================

def run_distillation_pipeline(cfg: DistillConfig, args) -> None:
    """
    1.  Load & freeze teacher checkpoint.
    2.  Load student checkpoint (pruned) and sync its architecture.
    3.  Evaluate teacher and pruned-student baselines.
    4.  Run distillation (finetune or full mode, or both).
    5.  Evaluate distilled student on test cells.
    6.  Print comparison table.
    7.  Save distilled student weights.
    8.  Append metrics to CSV.

    ``args`` is the Namespace produced by distill.py's get_parser().
    """
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )

    # ── Load Teacher ─────────────────────────────────────────────────────────
    print(f"\nLoading teacher checkpoint: {args.teacher_checkpoint}")
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher_sd   = teacher_ckpt.get("model_state_dict", teacher_ckpt.get("state_dict", teacher_ckpt))

    teacher = cfg.teacher
    teacher.load_state_dict(teacher_sd, strict=True)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"  Teacher size : {get_model_size_mb(teacher):.2f} MB")

    # ── Load Student ─────────────────────────────────────────────────────────
    print(f"\nLoading student (pruned) checkpoint: {args.student_checkpoint}")
    student_ckpt = torch.load(args.student_checkpoint, map_location=device, weights_only=False)
    student_sd   = student_ckpt.get("model_state_dict", student_ckpt.get("state_dict", student_ckpt))

    student = cfg.student
    _match_student_to_checkpoint(
        student      = student,
        state_dict   = student_sd,
        layer_type   = args._layer_type,
        prune_fn     = args._prune_fn,
        size_attr    = args._size_attr,
        seq_len      = cfg.example_seq_len,
        in_features  = cfg.in_features,
    )
    student.load_state_dict(student_sd, strict=False)
    student.to(device)

    pruning_style, pruning_ratio = _parse_pruning_info(
        args.student_checkpoint, cfg.model_name
    )

    example_input = torch.randn(1, cfg.example_seq_len, cfg.in_features).to(device)

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BASELINES (before distillation)")
    print("=" * 65)

    teacher_size = get_model_size_mb(teacher)
    student_size = get_model_size_mb(student)
    teacher_time = measure_inference_time(teacher, example_input, device)
    student_time = measure_inference_time(student, example_input, device)

    print(f"  Teacher — size: {teacher_size:.2f} MB  |  inference: {teacher_time:.2f} ms")
    print(f"  Student — size: {student_size:.2f} MB  |  inference: {student_time:.2f} ms")
    print(f"  Size reduction vs teacher: {(1 - student_size / teacher_size) * 100:.1f}%")

    teacher_val = evaluate(teacher, args._val_loader, device)
    student_val = evaluate(student, args._val_loader, device)
    print(f"\n  Teacher val — MAE: {teacher_val['mae']:.4f}  RMSE: {teacher_val['rmse']:.4f}  R²: {teacher_val.get('r2', 0):.4f}")
    print(f"  Student val — MAE: {student_val['mae']:.4f}  RMSE: {student_val['rmse']:.4f}  R²: {student_val.get('r2', 0):.4f}")

    # ── Distillation ─────────────────────────────────────────────────────────
    history_all = {}

    modes_to_run = (
        ["finetune", "full"] if args.mode == "both"
        else [args.mode]
    )

    # When running both modes we need to restore the original student weights
    # before the second run so the two modes are independent.
    original_student_sd = {k: v.clone() for k, v in student.state_dict().items()}

    for mode in modes_to_run:
        if len(modes_to_run) > 1:
            print(f"\n{'=' * 65}\nDISTILLATION MODE: {mode.upper()}\n{'=' * 65}")
            # Reset student to post-pruning weights for each independent run
            student.load_state_dict(original_student_sd)
            student.to(device)

        if mode == "finetune":
            print(f"\nRunning FINETUNE distillation ({args.finetune_epochs} epochs, lr={args.lr}) …\n")
            history = distill_finetune(
                student             = student,
                teacher             = teacher,
                train_loader        = args._train_loader,
                val_loader          = args._val_loader,
                device              = device,
                epochs              = args.finetune_epochs,
                lr                  = args.lr,
                alpha               = args.alpha,
                weight_decay        = args.weight_decay,
                max_grad_norm       = args.max_grad_norm,
                smooth_loss_weight  = args.smooth_loss_weight,
                smooth_loss_type    = args.smooth_loss_type,
                warmup_steps        = args.warmup_steps,
            )
        else:  # full
            print(f"\nRunning FULL distillation ({args.full_epochs} epochs, lr={args.full_lr}) …\n")
            history = distill_full(
                student             = student,
                teacher             = teacher,
                train_loader        = args._train_loader,
                val_loader          = args._val_loader,
                device              = device,
                epochs              = args.full_epochs,
                lr                  = args.full_lr,
                alpha               = args.alpha,
                weight_decay        = args.weight_decay,
                max_grad_norm       = args.max_grad_norm,
                smooth_loss_weight  = args.smooth_loss_weight,
                smooth_loss_type    = args.smooth_loss_type,
                warmup_steps        = args.warmup_steps,
                warmup_epoch_frac   = args.warmup_epoch_frac,
            )

        history_all[mode] = history

        # ── Post-distillation validation ──────────────────────────────────────
        post_val = evaluate(student, args._val_loader, device)
        print(
            f"\n  Post-distillation val — "
            f"MAE: {post_val['mae']:.4f}  RMSE: {post_val['rmse']:.4f}  R²: {post_val.get('r2', 0):.4f}"
        )

        # ── Test Cell Evaluation ──────────────────────────────────────────────
        print("\n" + "=" * 65 + "\nTEST CELL EVALUATION\n" + "=" * 65)
        test_results = []
        for cell_id, loader in zip(TEST_CELLS, args._test_loaders):
            m = evaluate(student, loader, device=torch.device("cpu"), warmup_steps=0, thread_state=True)
            test_results.append(m)
            print(
                f"  {cell_id:10s} | MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                f"R²={m.get('r2', 0):.4f}  MaxErr={m.get('max_error', 0):.4f}"
            )

        n = len(test_results)
        test_mae      = _avg(test_results, 'mae')
        test_rmse     = _avg(test_results, 'rmse')
        test_r2       = _avg(test_results, 'r2')
        test_max_err  = _avg(test_results, 'max_error')

        print("-" * 65)
        print(f"  AVERAGE    | MAE={test_mae:.4f}  RMSE={test_rmse:.4f}  R²={test_r2:.4f}")

        # ── Comparison Table ──────────────────────────────────────────────────
        _print_comparison_table(
            teacher_val  = teacher_val,
            student_pre  = student_val,
            student_post = post_val,
            mode         = mode,
        )

        # ── Save ──────────────────────────────────────────────────────────────
        student_base = os.path.splitext(os.path.basename(args.student_checkpoint))[0]
        save_name    = f"{student_base}_distilled_{mode}.pt"
        out_path     = os.path.join(args.out_dir, save_name)

        save_dict = {
            "model_state_dict": student.state_dict(),
            "distillation_mode":  mode,
            "alpha":              args.alpha,
            "pruning_style":      pruning_style or "none",
            "pruning_ratio":      pruning_ratio or 1.0,
        }
        torch.save(save_dict, out_path)
        print(f"\n  Saved distilled student → {out_path}")

        # ── Training history ──────────────────────────────────────────────────
        history_path = os.path.join(args.out_dir, f"{student_base}_distill_{mode}_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"  Training history    → {history_path}")

        # ── CSV Logging ───────────────────────────────────────────────────────
        if args.csv_output:
            append_metrics_to_csv(
                csv_path         = args.csv_output,
                model_name       = cfg.model_name,
                pruning_style    = pruning_style or "none",
                quantization     = False,
                actual_ratio     = pruning_ratio or 1.0,
                params_before    = 0,
                params_after     = 0,
                size_before      = teacher_size,
                size_after       = student_size,
                macs_before      = 0,
                macs_after       = 0,
                inference_before = teacher_time,
                inference_after  = student_time,
                test_mae         = test_mae,
                test_rmse        = test_rmse,
                test_r2          = test_r2,
                test_max_err     = test_max_err,
            )


# ============================================================
# COMPARISON TABLE
# ============================================================

def _print_comparison_table(
    teacher_val:  dict,
    student_pre:  dict,
    student_post: dict,
    mode:         str,
) -> None:
    print("\n" + "=" * 65)
    print(f"SUMMARY  [{mode.upper()} distillation]")
    print(f"{'':25s} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8}")
    print("-" * 65)
    print(f"  {'Teacher (baseline)':23s} | {teacher_val['mae']:8.4f} | {teacher_val['rmse']:8.4f} | {teacher_val.get('r2', 0):8.4f}")
    print(f"  {'Student (pruned)':23s} | {student_pre['mae']:8.4f} | {student_pre['rmse']:8.4f} | {student_pre.get('r2', 0):8.4f}")
    print(f"  {'Student (distilled)':23s} | {student_post['mae']:8.4f} | {student_post['rmse']:8.4f} | {student_post.get('r2', 0):8.4f}")

    recovery = (
        (student_pre['mae'] - student_post['mae']) /
        max(abs(student_pre['mae'] - teacher_val['mae']), 1e-9)
    ) * 100
    print(f"\n  MAE gap closed by distillation: {recovery:+.1f}%")
    print("=" * 65)
