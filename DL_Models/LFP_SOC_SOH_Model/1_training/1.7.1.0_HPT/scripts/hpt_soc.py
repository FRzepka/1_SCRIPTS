import copy
import gc
import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optuna.distributions import CategoricalDistribution
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[3]
TRAIN17 = ROOT / '1_training' / '1.7.1.0' / 'scripts' / 'train_soc.py'
BEST_CFG_OUT = ROOT / '1_training' / '1.7.1.0' / 'config' / 'train_soc_best_from_hpt.yaml'
BEST_JSON_OUT = ROOT / '1_training' / '1.7.1.0_HPT' / 'outputs' / 'best_trial.json'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def save_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    def _safe(v):
        if isinstance(v, dict):
            return {k: _safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v
    path.write_text(json.dumps(_safe(payload), indent=2))


def plot_trials(path: Path, trials_df: pd.DataFrame):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(trials_df['trial'], trials_df['best_val_rmse'], marker='o')
        plt.xlabel('Trial')
        plt.ylabel('Best val RMSE')
        plt.title('Optuna trials - SOC 1.7.1.0')
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
    except Exception as exc:
        print(f'Plot warning: {exc}')


def estimate_compact_flash_kib(mod, trial_cfg: dict, prune_amount: float = 0.30, overhead_factor: float = 1.08) -> float:
    hidden = int(trial_cfg['model']['hidden_size'])
    pruned_hidden = max(1, int(round(hidden * (1.0 - prune_amount))))
    proxy_model = mod.GRUMLP(
        in_features=len(trial_cfg['model']['features']),
        hidden_size=pruned_hidden,
        mlp_hidden=int(trial_cfg['model']['mlp_hidden']),
        num_layers=int(trial_cfg['model'].get('num_layers', 1)),
        dropout=float(trial_cfg['model'].get('dropout', 0.05)),
    )
    n_params = sum(p.numel() for p in proxy_model.parameters())
    estimated_bytes = float(n_params) * overhead_factor
    return estimated_bytes / 1024.0


def trial_distributions(search_space: dict) -> dict[str, CategoricalDistribution]:
    return {
        'hidden_size': CategoricalDistribution(search_space.get('hidden_size', [64, 96])),
        'mlp_hidden': CategoricalDistribution(search_space.get('mlp_hidden', [64, 96])),
        'num_layers': CategoricalDistribution(search_space.get('num_layers', [1, 2])),
        'dropout': CategoricalDistribution(search_space.get('dropout', [0.05, 0.08, 0.12])),
        'lr': CategoricalDistribution(search_space.get('lr', [1e-4, 1.2e-4, 1.5e-4])),
        'weight_decay': CategoricalDistribution(search_space.get('weight_decay', [0.0, 1e-5, 1e-4])),
        'batch_size': CategoricalDistribution(search_space.get('batch_size', [256, 512])),
        'accum_steps': CategoricalDistribution(search_space.get('accum_steps', [1, 2])),
        'dt_jitter_std_s': CategoricalDistribution(search_space.get('dt_jitter_std_s', [0.01, 0.02, 0.03])),
        'current_channel_dropout_prob': CategoricalDistribution(search_space.get('current_channel_dropout_prob', [0.10, 0.15, 0.20])),
    }


def write_study_artifacts(study: optuna.Study, out_root: Path):
    rows = []
    for t in study.trials:
        rows.append(
            {
                'trial': t.number,
                'state': str(t.state),
                'objective_value': t.value,
                'raw_best_val_rmse': t.user_attrs.get('raw_best_val_rmse'),
                'estimated_compact_flash_kib': t.user_attrs.get('estimated_compact_flash_kib'),
                'size_penalty': t.user_attrs.get('size_penalty'),
                **t.params,
            }
        )
    trials_df = pd.DataFrame(rows)
    trials_df.to_csv(out_root / 'hpt_trials.csv', index=False)
    valid = trials_df[trials_df['raw_best_val_rmse'].notna()].copy()
    if not valid.empty:
        plot_trials(out_root / 'hpt_trials.png', valid.rename(columns={'raw_best_val_rmse': 'best_val_rmse'}))

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if complete:
        best_trial = study.best_trial
        save_json(
            {
                'trial': best_trial.number,
                'value': best_trial.value,
                'params': best_trial.params,
                'raw_best_val_rmse': best_trial.user_attrs.get('raw_best_val_rmse'),
                'estimated_compact_flash_kib': best_trial.user_attrs.get('estimated_compact_flash_kib'),
                'size_penalty': best_trial.user_attrs.get('size_penalty'),
            },
            BEST_JSON_OUT,
        )


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Optuna HPT for SOC 1.7.0.0')
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    with open(args.config, 'r') as handle:
        cfg = yaml.safe_load(handle)

    mod = load_module('train_soc17_mod', TRAIN17)
    mod.set_seed(int(cfg.get('seed', 42)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_root = Path(cfg['paths']['out_root'])
    out_root.mkdir(parents=True, exist_ok=True)

    features = cfg['model']['features']
    epochs = int(cfg['training'].get('epochs', 30))
    val_interval = int(cfg['training'].get('val_interval', 1))
    early_stopping = int(cfg['training'].get('early_stopping', 6))
    max_grad_norm = float(cfg['training'].get('max_grad_norm', 1.0))

    hpt = cfg.get('hpt', {})
    n_trials = int(hpt.get('n_trials', 20))
    metric = hpt.get('metric', 'val_rmse')
    max_train_batches = int(hpt.get('max_train_batches', 0)) or None
    max_val_batches = int(hpt.get('max_val_batches', 0)) or None
    ss = cfg.get('search_space', {})
    study_name = hpt.get('study_name', 'soc1710_v5_hpt')
    storage_path = Path(hpt.get('storage_path', out_root / 'optuna_study.sqlite3'))
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    def objective(trial: optuna.Trial):
        trial_cfg = copy.deepcopy(cfg['base_train_config'])
        trial_cfg['model']['hidden_size'] = trial.suggest_categorical('hidden_size', ss.get('hidden_size', [64, 96]))
        trial_cfg['model']['mlp_hidden'] = trial.suggest_categorical('mlp_hidden', ss.get('mlp_hidden', [64, 96]))
        trial_cfg['model']['num_layers'] = trial.suggest_categorical('num_layers', ss.get('num_layers', [1, 2]))
        trial_cfg['model']['dropout'] = trial.suggest_categorical('dropout', ss.get('dropout', [0.05, 0.08, 0.12]))
        trial_cfg['training']['lr'] = trial.suggest_categorical('lr', ss.get('lr', [1e-4, 1.2e-4, 1.5e-4]))
        trial_cfg['training']['weight_decay'] = trial.suggest_categorical('weight_decay', ss.get('weight_decay', [0.0, 1e-5, 1e-4]))
        trial_cfg['training']['batch_size'] = trial.suggest_categorical('batch_size', ss.get('batch_size', [256, 512]))
        trial_cfg['training']['accum_steps'] = trial.suggest_categorical('accum_steps', ss.get('accum_steps', [1, 2]))
        trial_cfg['augmentation']['dt_jitter_std_s'] = trial.suggest_categorical('dt_jitter_std_s', ss.get('dt_jitter_std_s', [0.01, 0.02, 0.03]))
        trial_cfg['augmentation']['current_channel_dropout_prob'] = trial.suggest_categorical('current_channel_dropout_prob', ss.get('current_channel_dropout_prob', [0.10, 0.15, 0.20]))

        scaler = None
        train_loader = None
        val_loader = None
        model = None
        optimizer = None
        amp_scaler = None
        history = []
        best_val = float('inf')
        try:
            scaler = RobustScaler()
            chunk = int(trial_cfg['training']['seq_chunk_size'])
            batch_size = int(trial_cfg['training']['batch_size'])
            train_loader, val_loader = mod.create_dataloaders(trial_cfg, features, chunk, scaler, batch_size=batch_size)

            model = mod.GRUMLP(
                in_features=len(features),
                hidden_size=int(trial_cfg['model']['hidden_size']),
                mlp_hidden=int(trial_cfg['model']['mlp_hidden']),
                num_layers=int(trial_cfg['model']['num_layers']),
                dropout=float(trial_cfg['model']['dropout']),
            ).to(device)
            optimizer = mod.make_optimizer(model, float(trial_cfg['training']['lr']), float(trial_cfg['training']['weight_decay']))
            amp_scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

            patience = 0
            for epoch in range(1, epochs + 1):
                train_loss = mod.train_one_epoch(model, train_loader, device, optimizer, amp_scaler, max_grad_norm, epoch, epochs, accum_steps=int(trial_cfg['training'].get('accum_steps', 1)), max_batches=max_train_batches)
                if (epoch % val_interval == 0) or (epoch == 1):
                    val_metrics, _, _ = mod.eval_model(model, val_loader, device, max_batches=max_val_batches)
                    val_rmse = float(val_metrics['rmse'])
                    val_mae = float(val_metrics['mae'])
                    history.append({'epoch': epoch, 'train_loss': train_loss, 'val_rmse': val_rmse, 'val_mae': val_mae})
                    trial.report(val_rmse if metric == 'val_rmse' else val_mae, step=epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    if val_rmse < best_val:
                        best_val = val_rmse
                        patience = 0
                    else:
                        patience += 1
                    if patience >= early_stopping:
                        break
        except torch.OutOfMemoryError as exc:
            trial.set_user_attr('oom', True)
            trial.set_user_attr('oom_message', str(exc))
            raise optuna.TrialPruned() from exc
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                trial.set_user_attr('oom', True)
                trial.set_user_attr('oom_message', str(exc))
                raise optuna.TrialPruned() from exc
            raise
        finally:
            del optimizer
            del model
            del train_loader
            del val_loader
            del scaler
            del amp_scaler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        est_flash_kib = estimate_compact_flash_kib(mod, trial_cfg)
        target_flash_kib = float(hpt.get('target_compact_flash_kib', 200.0))
        flash_tol_kib = float(hpt.get('flash_tolerance_kib', 50.0))
        penalty_weight = float(hpt.get('size_penalty_weight', 0.0))
        flash_gap = max(0.0, abs(est_flash_kib - target_flash_kib) - flash_tol_kib)
        size_penalty = flash_gap / max(target_flash_kib, 1.0)
        objective_value = float(best_val) + penalty_weight * size_penalty

        trial.set_user_attr('estimated_compact_flash_kib', est_flash_kib)
        trial.set_user_attr('size_penalty', size_penalty)
        trial.set_user_attr('raw_best_val_rmse', float(best_val))

        trial_dir = out_root / 'trials' / f'trial_{trial.number:04d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                'params': trial.params,
                'history': history,
                'best_val_rmse': float(best_val),
                'estimated_compact_flash_kib': est_flash_kib,
                'size_penalty': size_penalty,
                'objective_value': objective_value,
            },
            trial_dir / 'history.json',
        )
        return objective_value

    study = optuna.create_study(
        direction=cfg['hpt'].get('direction', 'minimize'),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        for item in hpt.get('bootstrap_completed_trials', []):
            params = item['params']
            raw_best_val_rmse = float(item['raw_best_val_rmse'])
            trial_cfg = copy.deepcopy(cfg['base_train_config'])
            trial_cfg['model']['hidden_size'] = int(params['hidden_size'])
            trial_cfg['model']['mlp_hidden'] = int(params['mlp_hidden'])
            trial_cfg['model']['num_layers'] = int(params['num_layers'])
            trial_cfg['model']['dropout'] = float(params['dropout'])
            trial_cfg['training']['lr'] = float(params['lr'])
            trial_cfg['training']['weight_decay'] = float(params['weight_decay'])
            trial_cfg['training']['batch_size'] = int(params['batch_size'])
            trial_cfg['training']['accum_steps'] = int(params['accum_steps'])
            trial_cfg['augmentation']['dt_jitter_std_s'] = float(params['dt_jitter_std_s'])
            trial_cfg['augmentation']['current_channel_dropout_prob'] = float(params['current_channel_dropout_prob'])

            est_flash_kib = estimate_compact_flash_kib(mod, trial_cfg)
            target_flash_kib = float(hpt.get('target_compact_flash_kib', 200.0))
            flash_tol_kib = float(hpt.get('flash_tolerance_kib', 50.0))
            penalty_weight = float(hpt.get('size_penalty_weight', 0.0))
            flash_gap = max(0.0, abs(est_flash_kib - target_flash_kib) - flash_tol_kib)
            size_penalty = flash_gap / max(target_flash_kib, 1.0)
            objective_value = raw_best_val_rmse + penalty_weight * size_penalty

            frozen = optuna.trial.create_trial(
                params=params,
                distributions=trial_distributions(ss),
                value=objective_value,
                user_attrs={
                    'raw_best_val_rmse': raw_best_val_rmse,
                    'estimated_compact_flash_kib': est_flash_kib,
                    'size_penalty': size_penalty,
                    'bootstrap': True,
                },
            )
            study.add_trial(frozen)

    def _after_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        write_study_artifacts(study, out_root)

    remaining_trials = max(0, n_trials - len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL)]))
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, callbacks=[_after_trial])
    else:
        print(f'No remaining HPT trials to run; existing study already has {len(study.trials)} trials.')

    write_study_artifacts(study, out_root)
    best_trial = study.best_trial

    best_cfg = copy.deepcopy(cfg['base_train_config'])
    best_cfg['model']['hidden_size'] = int(best_trial.params['hidden_size'])
    best_cfg['model']['mlp_hidden'] = int(best_trial.params['mlp_hidden'])
    best_cfg['model']['num_layers'] = int(best_trial.params['num_layers'])
    best_cfg['model']['dropout'] = float(best_trial.params['dropout'])
    best_cfg['training']['lr'] = float(best_trial.params['lr'])
    best_cfg['training']['weight_decay'] = float(best_trial.params['weight_decay'])
    best_cfg['training']['batch_size'] = int(best_trial.params['batch_size'])
    best_cfg['training']['accum_steps'] = int(best_trial.params['accum_steps'])
    best_cfg['augmentation']['dt_jitter_std_s'] = float(best_trial.params['dt_jitter_std_s'])
    best_cfg['augmentation']['current_channel_dropout_prob'] = float(best_trial.params['current_channel_dropout_prob'])
    BEST_CFG_OUT.write_text(yaml.safe_dump(best_cfg, sort_keys=False))
    print(
        f"Best trial={best_trial.number} objective={best_trial.value:.6f} "
        f"raw_rmse={best_trial.user_attrs.get('raw_best_val_rmse', float('nan')):.6f} "
        f"est_flash_kib={best_trial.user_attrs.get('estimated_compact_flash_kib', float('nan')):.1f}"
    )
    print(f'Wrote best config to {BEST_CFG_OUT}')


if __name__ == '__main__':
    main()
