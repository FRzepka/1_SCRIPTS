import os
import json
import typing as T
import torch
import torch.nn as nn
import torch_pruning as tp
import argparse
import copy
import math

from config import TEST_CELLS, EARLY_STOPPING
from src.utils.metrics import (
    evaluate, measure_inference_time, get_model_size_mb, append_metrics_to_csv
)
from src.pruning.distillation_backend import train_one_epoch, distillation_loss


# ----- FINETUNE -----
def finetune(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    smooth_loss_weight: float = 0.05,
    smooth_loss_type: str = "l1",
    teacher: nn.Module = None,
    alpha: float = 0.5,
) -> T.List[dict]:
    """Fine-tune ``model`` and restore its best validation checkpoint in-place.

    The returned list contains one metrics dictionary per completed epoch.  The
    model itself is restored to the checkpoint with the lowest validation MSE,
    including the pre-fine-tuning checkpoint when no epoch improves it.

    Recurrent state is deliberately *not* threaded between training batches:
    the training loader is shuffled and may mix windows from different battery
    cells.  LSTM/GRU state is still used normally inside each input chunk.
    """
    if teacher is not None:
        raise NotImplementedError(
            "Distillation not implemented yet."
        )
    if train_loader is None or val_loader is None:
        raise ValueError("Both train_loader and val_loader are required for fine-tuning.")
    if epochs <= 0:
        return []
    if smooth_loss_weight < 0:
        raise ValueError("smooth_loss_weight must be non-negative.")
    if smooth_loss_weight > 0 and smooth_loss_type.strip().lower() not in {"l1", "mae", "l2", "mse"}:
        raise ValueError(
            f"Unsupported smooth_loss_type={smooth_loss_type!r}; "
            "expected one of 'l1', 'mae', 'l2', or 'mse'."
        )

    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]  # only trainable parameters to be passed to optimizer
    if not trainable:
        raise ValueError("The model has no trainable parameters after pruning/freezing.")

    floating_param = next((p for p in model.parameters() if p.is_floating_point()), None)   # detect dtype of model (should be FP32)
    model_dtype = floating_param.dtype if floating_param is not None else torch.float32

    non_blocking = device.type == "cuda"    # Helps CPU to GPU transfer when we are on CUDA

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),   # max epochs (prevent 0)
        eta_min=lr * 0.01,
    )
    loss_fn = nn.MSELoss()
    history: T.List[dict] = []

    # Helper to prevent output shape mismatches (LSTM and GRU return hidden states on top of prediction)
    def _match_prediction_shape(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        if prediction.ndim == target.ndim + 1 and prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)
        if prediction.numel() != target.numel():
            raise ValueError(
                "Model output and target contain different numbers of values: "
                f"prediction shape={tuple(prediction.shape)}, target shape={tuple(target.shape)}."
            )
        return prediction.reshape_as(target)

    # Treat the pre-fine-tuning checkpoint as the first candidate best model.
    initial_val = evaluate(model, val_loader, device)
    best_val_mse = float(initial_val["mse"])
    best_val_metrics = dict(initial_val)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    patience = max(0, int(EARLY_STOPPING))

    # Training
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_smooth = 0.0
        total_values = 0

        for inputs, targets in train_loader:
            # Make sure to match inputs and targets with the model dtype
            inputs = inputs.to(
                device=device,
                dtype=model_dtype,
                non_blocking=non_blocking,
            )
            targets = targets.to(
                device=device,
                dtype=model_dtype,
                non_blocking=non_blocking,
            )

            optimizer.zero_grad(set_to_none=True)

            # Do not carry hidden state across batches.  The loader shuffles and
            # batches windows from potentially different cells, so doing so would
            # leak state between unrelated time-series segments.
            predictions = _match_prediction_shape(model(inputs), targets)
            mse_loss = loss_fn(predictions, targets)

            smooth_loss = predictions.new_zeros(())
            if (
                smooth_loss_weight > 0
                and predictions.ndim >= 2
                and predictions.shape[-1] > 1
            ):
                pred_delta = torch.diff(predictions, dim=-1)
                target_delta = torch.diff(targets, dim=-1)
                if smooth_kind in {"l1", "mae"}:
                    smooth_loss = nn.functional.l1_loss(pred_delta, target_delta)
                else:
                    smooth_loss = nn.functional.mse_loss(pred_delta, target_delta)

            loss = mse_loss + smooth_loss_weight * smooth_loss

            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()

            value_count = targets.numel()
            total_loss += float(loss.detach()) * value_count
            total_mse += float(mse_loss.detach()) * value_count
            total_smooth += float(smooth_loss.detach()) * value_count
            total_values += value_count

        if total_values == 0:
            raise ValueError("The training loader produced no target values.")

        train_loss = total_loss / total_values
        train_mse = total_mse / total_values
        train_smooth = total_smooth / total_values
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Validation windows are evaluated independently.  This matches the
        # batched validation loader and prevents state leakage across cells.
        val_metrics = evaluate(model, val_loader, device, thread_state=False)
        val_mse = float(val_metrics["mse"])
        if not math.isfinite(val_mse):
            raise RuntimeError(f"Validation MSE became non-finite in epoch {epoch}.")

        improved = val_mse < best_val_mse
        if improved:
            best_val_mse = val_mse
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_mse": float(train_mse),
            "train_smooth_loss": float(train_smooth),
            "val_mse": float(val_metrics["mse"]),
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_r2": float(val_metrics["r2"]),
            "val_max_error": float(val_metrics["max_error"]),
            "lr": current_lr,
            "improved": bool(improved),
        })

        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train={train_loss:.6f} | val_mse={val_metrics['mse']:.6f} | "
            f"val_mae={val_metrics['mae']:.6f} | "
            f"val_rmse={val_metrics['rmse']:.6f} | lr={current_lr:.3e}{marker}"
        )

        scheduler.step()

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping after {epoch} epochs: validation MSE did not "
                f"improve for {patience} consecutive epoch(s)."
            )
            break

    # Keep the existing return contract (history) while making the passed model
    # itself the best validation model used by the rest of the pruning pipeline.
    model.load_state_dict(best_state)
    model.eval()
    print(
        f"Restored best fine-tuning checkpoint from epoch {best_epoch} "
        f"(val MSE={best_val_metrics['mse']:.6f}, "
        f"MAE={best_val_metrics['mae']:.6f}, "
        f"RMSE={best_val_metrics['rmse']:.6f})."
    )
    return history


def apply_pruning(model: nn.Module, example_inputs: torch.Tensor, pruning_ratio: float, ignored_layers: T.List[nn.Module]) -> None:
    imp = tp.importance.GroupMagnitudeImportance(p=2)
    pruner = tp.pruner.BasePruner(
        model, example_inputs, importance=imp, pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers, round_to=4,
    )
    pruner.step()


def calculate_effective_ratio(target_ratio: float, num_iterations: int) -> float:
    """
    Calculate per-iteration pruning ratio to achieve target total ratio.

    For iterative pruning: (1 - r_iter)^n = (1 - r_total)
    So: r_iter = 1 - (1 - r_total)^(1/n)
    """
    return 1.0 - (1.0 - target_ratio) ** (1.0 / num_iterations)


# ----- PRUNING -----

def prune_model(
    model_name: str,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    example_inputs: torch.Tensor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loaders: list,
    ignored_layers: list,
    model_cfg: dict = None,
    freeze_fn: T.Callable = None,
    unfreeze_fn: T.Callable = None,
    finetune_kwargs: dict = None,
    teacher: nn.Module = None,
    alpha: float = 0.5,
):
    if finetune_kwargs is None: finetune_kwargs = {}
    if args.pruning_ratio != 0.0:
        print(f"\nDevice : {device}\nPruning Mode: {args.pruning_mode.upper()}")
        os.makedirs(args.out_dir, exist_ok=True)

        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        base_val = evaluate(model, val_loader, device)
        base_size = get_model_size_mb(model)
        base_time = measure_inference_time(model, example_inputs, device)

        print("=" * 60)
        print("BASELINE MODEL (BEFORE PRUNING)")
        print(f"  Params : {base_params / 1e6:.4f} M\n  Size   : {base_size:.2f} MB")
        print(f"  MACs   : {base_macs  / 1e9:.4f} G\n  Inference: {base_time:.2f} ms")
        print(f"  Val MAE : {base_val['mae']:.4f}   RMSE: {base_val['rmse']:.4f}   R²: {base_val.get('r2', 0):.4f}")
        print("=" * 60)

        if args.pruning_mode == "oneshot":
            print(f"\nPerforming ONE-SHOT pruning with ratio={args.pruning_ratio:.4f} …\n")
            tp.utils.print_tool.before_pruning(model)
            apply_pruning(model, example_inputs, args.pruning_ratio, ignored_layers)
            tp.utils.print_tool.after_pruning(model)

            pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
            pruned_size = get_model_size_mb(model)
            pruned_time = measure_inference_time(model, example_inputs, device)
            pruned_val = evaluate(model, val_loader, device)

            print("\n" + "=" * 60)
            print("AFTER ONE-SHOT PRUNING (before fine-tuning)")
            print(f"  Params : {base_params/1e6:.4f} M  →  {pruned_params/1e6:.4f} M")
            print(f"  Size   : {base_size:.2f} MB  →  {pruned_size:.2f} MB")
            print(f"  MACs   : {base_macs/1e9:.4f} G  →  {pruned_macs/1e9:.4f} G")
            print(f"  Inference: {base_time:.2f} ms  →  {pruned_time:.2f} ms")
            print(f"  Val MAE : {pruned_val['mae']:.4f}   RMSE: {pruned_val['rmse']:.4f}")
            print("=" * 60)

            if freeze_fn: freeze_fn()
            ft_val = pruned_val
            if args.finetune_epochs > 0:
                print(f"\nFine-tuning for {args.finetune_epochs} epochs …\n")
                finetune(model, train_loader, val_loader, device, epochs=args.finetune_epochs, lr=args.finetune_lr, teacher=teacher, alpha=alpha, **finetune_kwargs)
                ft_val = evaluate(model, val_loader, device)
                print("\n" + "=" * 60)
                print(f"AFTER FINE-TUNING\n  Val MAE : {ft_val['mae']:.4f}   RMSE: {ft_val['rmse']:.4f}")
                print("=" * 60)

        else:
            per_iter_ratio = calculate_effective_ratio(args.pruning_ratio, args.num_iterations)
            print(f"\nPerforming ITERATIVE pruning:\n  Target: {args.pruning_ratio:.4f} | Iterations: {args.num_iterations} | Per-iter: {per_iter_ratio:.4f}\n")

            if freeze_fn: freeze_fn()
            iteration_history = []

            for iteration in range(1, args.num_iterations + 1):
                print("\n" + "=" * 60 + f"\nITERATION {iteration}/{args.num_iterations}\n" + "=" * 60)
                apply_pruning(model, example_inputs, per_iter_ratio, ignored_layers)

                cur_macs, cur_params = tp.utils.count_ops_and_params(model, example_inputs)
                pruned_val = evaluate(model, val_loader, device)

                ft_val = pruned_val
                if args.finetune_epochs > 0:
                    print(f"\nFine-tuning for {args.finetune_epochs} epochs …")
                    finetune(model, train_loader, val_loader, device, epochs=args.finetune_epochs, lr=args.finetune_lr, teacher=teacher, alpha=alpha, **finetune_kwargs)
                    ft_val = evaluate(model, val_loader, device)

                iteration_history.append({
                    'iteration': int(iteration), 'params': int(cur_params), 'macs': int(cur_macs),
                    'param_ratio': float(cur_params / base_params),
                    'mae_before_ft': float(pruned_val['mae']), 'rmse_before_ft': float(pruned_val['rmse']),
                    'mae_after_ft': float(ft_val['mae']), 'rmse_after_ft': float(ft_val['rmse']),
                })

            history_path = os.path.join(args.out_dir, f"iterative_history_{args.pruning_ratio:.4f}.json")
            with open(history_path, 'w') as f:
                json.dump(iteration_history, f, indent=2)
            print(f"\nIteration history saved → {history_path}")

        if unfreeze_fn: unfreeze_fn()

        # Final evaluation on test cells
        test_mae_avg, test_rmse_avg, test_r2_avg, test_max_err_avg = 0.0, 0.0, 0.0, 0.0
        if test_loaders:
            print("\n" + "=" * 60 + "\nTEST CELL EVALUATION (SEQUENTIAL)\n" + "=" * 60)
            test_results = []

            model.to(device)

            # Iterate through the individual loaders per cell
            for cell_id, loader in zip(TEST_CELLS, test_loaders):
                m = evaluate(model, loader, device=device, thread_state=True)
                test_results.append(m)
                print(f"  {cell_id:10s} | MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                    f"R²={m.get('r2', 0):.4f}  MaxErr={m.get('max_error', 0):.4f}")

            n = len(test_results)
            test_mae_avg     = sum(r['mae'] for r in test_results) / n
            test_rmse_avg    = sum(r['rmse'] for r in test_results) / n
            test_r2_avg      = sum(r.get('r2', 0) for r in test_results) / n
            test_max_err_avg = sum(r.get('max_error', 0) for r in test_results) / n

            print("-" * 60)
            print(f"  AVERAGE    | MAE={test_mae_avg:.4f}  RMSE={test_rmse_avg:.4f}  R²={test_r2_avg:.4f}")

        # Save final PyTorch model
        final_macs, final_params = tp.utils.count_ops_and_params(model, example_inputs)
        final_size = get_model_size_mb(model)
        final_time = measure_inference_time(model, example_inputs, device)
        final_ratio = final_params / base_params

        base_name = f"{model_name}_pruned_{args.pruning_mode}{'_kd' if teacher is not None else ''}_{final_ratio:.4f}"
        out_path = os.path.join(args.out_dir, f"{base_name}.pt")

        save_dict = {
            "model_state_dict": model.state_dict(),
            "pruning_mode": args.pruning_mode,
            "pruning_ratio": args.pruning_ratio,
            "actual_param_ratio": final_ratio,
            "num_iterations": args.num_iterations if args.pruning_mode == "iterative" else 1,
        }
        if model_cfg: save_dict["model_cfg"] = model_cfg
        torch.save(save_dict, out_path)
        print(f"\nPruned model saved → {out_path}")

        return model, base_name


    # Return model to original device if needed for further processing
    model.to(device)

    # ==========================

    if args.csv_output and args.pruning_ratio != 0.0:
        append_metrics_to_csv(
            csv_path=args.csv_output,
            model_name=model_name,
            pruning_style=args.pruning_mode,
            quantization=False,
            actual_ratio=final_ratio,
            params_before=base_params,
            params_after=final_params,
            size_before=base_size,
            size_after=final_size,
            macs_before=base_macs,
            macs_after=final_macs,
            inference_before=base_time,
            inference_after=final_time,
            test_mae=test_mae_avg,
            test_rmse=test_rmse_avg,
            test_r2=test_r2_avg,
            test_max_err=test_max_err_avg,
        )
