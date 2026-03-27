import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_paper_results import load_campaign_rows
from build_curated_paper_results import MODEL_META, MODEL_ORDER, _load_run_series, _thin, _write_md

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
OUT_ROOT = RESULTS_DIR / 'scenario_detail'


def _rolling_mae(series: pd.DataFrame, window_s: int = 900) -> pd.Series:
    dt = series['time_s'].diff().median()
    if pd.isna(dt) or dt <= 0:
        dt = 1.0
    window = max(1, int(round(window_s / dt)))
    return series['abs_err'].rolling(window=window, min_periods=window).mean()


def write_scenario_folder(df: pd.DataFrame, alias: str) -> None:
    out_dir = OUT_ROOT / alias
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df['alias'] == alias].copy().sort_values('model')
    sub.to_csv(out_dir / 'summary.csv', index=False)
    _write_md(sub.round(5), out_dir / 'summary.md')

    series = []
    for model in MODEL_ORDER:
        row = sub[sub['model'] == model]
        if row.empty:
            continue
        run_dir = Path(row.iloc[0]['run_dir'])
        try:
            s = _load_run_series(run_dir, model)
        except Exception:
            continue
        series.append(_thin(s, max_points=5000))

    if not series:
        (out_dir / 'README.md').write_text(f'# {alias}\n\nNo time-series data available.\n')
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series[0]['time_s'] / 3600.0, series[0]['soc_true'], color='#111111', lw=1.2, label='SOC true')
    for s in series:
        model = s['model'].iloc[0]
        meta = MODEL_META[model]
        ax.plot(s['time_s'] / 3600.0, s['soc_pred'], lw=1.0, color=meta['color'], alpha=0.9, label=meta['short'])
    ax.set_title(f'{alias}: full-run SOC overview')
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('SOC [-]')
    ax.legend(frameon=True, ncol=3)
    fig.savefig(out_dir / 'soc_overview.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for s in series:
        model = s['model'].iloc[0]
        meta = MODEL_META[model]
        rolling = _rolling_mae(s)
        mask = np.isfinite(rolling.to_numpy())
        if not mask.any():
            continue
        ax.plot(
            s.loc[mask, 'time_s'] / 3600.0,
            rolling.loc[mask],
            lw=1.0,
            color=meta['color'],
            alpha=0.9,
            label=meta['short'],
        )
    ax.set_title(f'{alias}: rolling MAE overview (15 min)')
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Rolling MAE [-]')
    ax.legend(frameon=True, ncol=3)
    fig.savefig(out_dir / 'rolling_mae_overview.png', dpi=220)
    plt.close(fig)

    (out_dir / 'README.md').write_text(
        f'# {alias}\n\n'
        '- `summary.csv` and `summary.md` contain scenario-level metrics.\n'
        '- `soc_overview.png` is a quick full-run comparison for inspection, not a paper figure.\n'
        '- `rolling_mae_overview.png` shows a 15 min rolling MAE trend over the full run.\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign_tag', required=True)
    ap.add_argument('--out_root', default=None)
    args = ap.parse_args()

    global OUT_ROOT
    if args.out_root:
        OUT_ROOT = Path(args.out_root)
    campaign_dir = ROOT / 'campaigns' / args.campaign_tag
    df = load_campaign_rows(campaign_dir)
    if df.empty:
        raise SystemExit(f'No rows found for {args.campaign_tag}')
    for alias in sorted(df['alias'].unique()):
        write_scenario_folder(df, alias)
    print(f'Wrote scenario detail folders under {OUT_ROOT}')


if __name__ == '__main__':
    main()
