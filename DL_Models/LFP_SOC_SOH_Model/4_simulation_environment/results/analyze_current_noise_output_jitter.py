import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
OUT_DIR = RESULTS_DIR / 'noise_detail'
ARCHIVE_ROOT = ROOT / 'archive'

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_DIR))

from build_paper_results import MODEL_DIRS
from build_curated_paper_results import MODEL_ORDER, MODEL_META, _load_run_series

CELL = 'MGFarm_18650_C07'
LEVELS = [0.02, 0.10, 0.15, 0.20]
WARMUP_S = 600.0


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / 'summary.json').read_text())


def _iter_model_roots(model: str):
    active_root = MODEL_DIRS[model]
    root_names = {model, active_root.parent.name}
    roots = [active_root]
    if ARCHIVE_ROOT.exists():
        for name in sorted(root_names):
            roots.extend(sorted(ARCHIVE_ROOT.glob(f'*/runs/{name}')))
            roots.extend(sorted(ARCHIVE_ROOT.glob(f'*/{name}')))
    uniq = []
    seen = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(root)
    return uniq


def _latest_run(model: str, scenario: str, level=None):
    scenario_candidates = [scenario]
    if scenario == 'current_noise' and level is not None:
        if abs(float(level) - 0.02) < 1e-9:
            scenario_candidates = ['current_noise', 'current_noise_low']
        elif abs(float(level) - 0.10) < 1e-9:
            scenario_candidates = ['current_noise', 'current_noise_high']
    hits = []
    for root in _iter_model_roots(model):
        if not root.exists():
            continue
        for scenario_name in scenario_candidates:
            base = root / scenario_name
            if not base.exists():
                continue
            for run_dir in sorted(base.iterdir()):
                if not run_dir.is_dir() or not (run_dir / 'summary.json').exists():
                    continue
                s = _summary(run_dir)
                if s.get('cell') != CELL:
                    continue
                if level is not None:
                    std = s.get('scenario_meta', {}).get('current_noise_std')
                    if std is None or abs(float(std) - float(level)) > 1e-9:
                        continue
                hits.append((run_dir.stat().st_mtime, run_dir, s))
    if not hits:
        raise FileNotFoundError(f'Missing run for {model} / {scenario} / {level}')
    _, run_dir, s = max(hits, key=lambda x: x[0])
    return run_dir, s


def _step_stats(series: pd.DataFrame) -> dict:
    s = series[['time_s', 'soc_pred']].copy().sort_values('time_s')
    s = s[s['time_s'] >= WARMUP_S].copy()
    step = s['soc_pred'].diff().abs().dropna()
    if step.empty:
        return {
            'mean_step_abs': float('nan'),
            'p95_step_abs': float('nan'),
            'p99_step_abs': float('nan'),
            'max_step_abs': float('nan'),
            'rms_step': float('nan'),
        }
    vals = step.to_numpy(dtype=float)
    return {
        'mean_step_abs': float(np.mean(vals)),
        'p95_step_abs': float(np.quantile(vals, 0.95)),
        'p99_step_abs': float(np.quantile(vals, 0.99)),
        'max_step_abs': float(np.max(vals)),
        'rms_step': float(np.sqrt(np.mean(vals ** 2))),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODEL_ORDER:
        base_dir, _ = _latest_run(model, 'baseline')
        base_series = _load_run_series(base_dir, model)
        base_stats = _step_stats(base_series)
        for level in LEVELS:
            noise_dir, _ = _latest_run(model, 'current_noise', level)
            noise_series = _load_run_series(noise_dir, model)
            noise_stats = _step_stats(noise_series)
            row = {
                'model': model,
                'class': MODEL_META[model]['short'],
                'current_noise_std': level,
                'run_dir': str(noise_dir),
                **{f'base_{k}': v for k, v in base_stats.items()},
                **noise_stats,
            }
            row['delta_p95_step_abs'] = row['p95_step_abs'] - row['base_p95_step_abs']
            row['delta_mean_step_abs'] = row['mean_step_abs'] - row['base_mean_step_abs']
            row['delta_rms_step'] = row['rms_step'] - row['base_rms_step']
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(['model', 'current_noise_std']).reset_index(drop=True)
    df.to_csv(OUT_DIR / 'current_noise_output_jitter.csv', index=False)
    try:
        (OUT_DIR / 'current_noise_output_jitter.md').write_text(df.to_markdown(index=False))
    except Exception:
        pass

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f7f7f7',
        'axes.edgecolor': '#444444',
        'axes.grid': True,
        'grid.color': '#d9d9d9',
        'grid.alpha': 0.7,
        'grid.linewidth': 0.8,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'savefig.bbox': 'tight',
        'savefig.dpi': 240,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), sharex=True)
    for model in MODEL_ORDER:
        sub = df[df['model'] == model].sort_values('current_noise_std')
        color = MODEL_META[model]['color']
        label = MODEL_META[model]['short']
        axes[0].plot(sub['current_noise_std'], sub['p95_step_abs'], marker='o', lw=2.2, color=color, label=label)
        axes[1].plot(sub['current_noise_std'], sub['delta_p95_step_abs'], marker='o', lw=2.2, color=color, label=label)
    axes[0].set_title('Output jitter under current noise')
    axes[0].set_ylabel(r'p95 $|\Delta \hat{SOC}|$ per 1 s step')
    axes[1].set_title('Additional output jitter vs baseline')
    axes[1].set_ylabel(r'$\Delta$ p95 $|\Delta \hat{SOC}|$ per 1 s step')
    for ax in axes:
        ax.set_xlabel(r'Current-noise std $\sigma_I$ [A]')
        ax.set_xticks(LEVELS)
        ax.legend(frameon=True, ncol=2, loc='upper left')
    fig.suptitle('Current-noise output volatility probe', fontsize=17, y=1.02)
    fig.savefig(OUT_DIR / 'Figure_noise_output_jitter_probe.png')
    plt.close(fig)

    print(df[['model','current_noise_std','p95_step_abs','delta_p95_step_abs']].to_string(index=False))


if __name__ == '__main__':
    main()
