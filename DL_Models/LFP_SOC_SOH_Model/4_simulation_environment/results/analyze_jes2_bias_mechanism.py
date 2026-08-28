from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("DM", "HDM", "HECM", "DD")
STATES = ("fresh", "mid_life", "aged")
RESET_MODES = ("natural", "no_full_charge_reset")
PREDICTIONS = {"DM": "soc_cc", "HDM": "soc_cc", "HECM": "soc_ecm", "DD": "soc_pred"}
PATTERNS = {
    "DM": "soc_cc_fullcell_*.csv",
    "HDM": "soc_cc_soh_fullcell_*.csv",
    "HECM": "ecm_soc_fullcell_*.csv",
    "DD": "soc_pred_fullcell_*.csv",
}
HORIZONS_H = (1, 3, 6, 12, 24)


def load_run(root: Path, state: str, reset_mode: str, alias: str, model: str) -> pd.DataFrame:
    directory = root / "runs/C29" / f"C29_{state}" / reset_mode / alias / model
    paths = list(directory.glob(PATTERNS[model]))
    if not paths:
        raise FileNotFoundError(f"Missing trajectory for {state} {reset_mode} {alias} {model}")
    frame = pd.read_csv(paths[0])
    return frame.rename(columns={PREDICTIONS[model]: "soc_pred"})


def paired_divergence(baseline: pd.DataFrame, biased: pd.DataFrame) -> pd.DataFrame:
    pair = baseline[["index", "time_s", "soc_pred"]].merge(
        biased[["index", "soc_pred"]], on="index", suffixes=("_baseline", "_biased"), how="inner"
    )
    pair["divergence"] = np.abs(pair.soc_pred_biased - pair.soc_pred_baseline)
    return pair


def adverse_pair(root: Path, state: str, reset_mode: str, model: str) -> tuple[str, pd.DataFrame]:
    baseline = load_run(root, state, reset_mode, "baseline", model)
    candidates = {
        "positive_3pct": paired_divergence(baseline, load_run(root, state, reset_mode, "positive_3pct", model)),
        "negative_3pct": paired_divergence(baseline, load_run(root, state, reset_mode, "negative_3pct", model)),
    }
    alias = max(candidates, key=lambda name: float(candidates[name].divergence.mean()))
    return alias, candidates[alias]


def find_reset_events(dm_baseline: pd.DataFrame) -> np.ndarray:
    charge = dm_baseline.q_m_new.to_numpy(dtype=float)
    transitions = np.flatnonzero((charge[:-1] < -1e-4) & (np.abs(charge[1:]) <= 1e-9)) + 1
    if not len(transitions):
        return transitions
    keep = np.r_[True, np.diff(dm_baseline.time_s.to_numpy(dtype=float)[transitions]) > 1800]
    return transitions[keep]


def main() -> None:
    simulation = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Analyze the focused JES2 bias accumulation/reset trajectories.")
    parser.add_argument(
        "--campaign", type=Path,
        default=simulation / "campaigns/jes2_bias_mechanism_C29_20260828",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.out_dir or args.campaign / "results"
    out.mkdir(parents=True, exist_ok=True)

    horizon_rows = []
    selected: dict[tuple[str, str, str], tuple[str, pd.DataFrame]] = {}
    for state in STATES:
        window_start = float(load_run(args.campaign, state, "natural", "baseline", "DM").time_s.iloc[0])
        for reset_mode in RESET_MODES:
            for model in MODELS:
                alias, pair = adverse_pair(args.campaign, state, reset_mode, model)
                pair = pair.copy()
                pair["elapsed_h"] = (pair.time_s - window_start) / 3600.0
                dt_h = pair.time_s.diff().fillna(0).clip(lower=0).to_numpy(dtype=float) / 3600.0
                pair["cumulative_burden_soc_h"] = np.cumsum(pair.divergence.to_numpy(dtype=float) * dt_h)
                selected[(state, reset_mode, model)] = (alias, pair)
                for horizon in HORIZONS_H:
                    part = pair[(pair.elapsed_h >= 0) & (pair.elapsed_h <= horizon)]
                    tail = pair[(pair.elapsed_h > max(0, horizon - 1)) & (pair.elapsed_h <= horizon)]
                    horizon_rows.append({
                        "cell": "C29", "soh_state": state, "reset_mode": reset_mode,
                        "model": model, "adverse_alias": alias, "horizon_h": horizon,
                        "mean_divergence": float(part.divergence.mean()),
                        "terminal_1h_mean_divergence": float(tail.divergence.mean()),
                        "cumulative_burden_soc_h": float(part.cumulative_burden_soc_h.iloc[-1]),
                    })
    horizons = pd.DataFrame(horizon_rows)
    horizons.to_csv(out / "bias_mechanism_horizons.csv", index=False)

    event_rows = []
    for state in STATES:
        dm = load_run(args.campaign, state, "natural", "baseline", "DM")
        event_indices = find_reset_events(dm)
        event_times = dm.time_s.to_numpy(dtype=float)[event_indices]
        for event_number, event_time in enumerate(event_times, start=1):
            for reset_mode in RESET_MODES:
                for model in MODELS:
                    alias, pair = selected[(state, reset_mode, model)]
                    relative = pair.time_s.to_numpy(dtype=float) - event_time
                    pre = pair.divergence[(relative >= -900) & (relative < 0)]
                    post = pair.divergence[(relative >= 60) & (relative <= 960)]
                    if pre.empty or post.empty:
                        continue
                    pre_mean, post_mean = float(pre.mean()), float(post.mean())
                    event_rows.append({
                        "cell": "C29", "soh_state": state, "event": event_number,
                        "event_time_s": event_time, "reset_mode": reset_mode, "model": model,
                        "adverse_alias": alias, "pre_15min_mean_divergence": pre_mean,
                        "post_15min_mean_divergence": post_mean,
                        "post_minus_pre": post_mean - pre_mean,
                        "relative_reduction": (pre_mean - post_mean) / pre_mean if pre_mean > 0 else np.nan,
                    })
    events = pd.DataFrame(event_rows)
    events.to_csv(out / "bias_mechanism_reset_events.csv", index=False)

    final = horizons[horizons.horizon_h == 24].pivot_table(
        index=["soh_state", "model"], columns="reset_mode", values="cumulative_burden_soc_h"
    ).reset_index()
    final["no_reset_to_natural_ratio"] = final.no_full_charge_reset / final.natural
    final.to_csv(out / "bias_mechanism_24h_summary.csv", index=False)
    print(final.to_string(index=False))
    print(f"reset events: {len(events)} rows; output: {out}")


if __name__ == "__main__":
    main()
