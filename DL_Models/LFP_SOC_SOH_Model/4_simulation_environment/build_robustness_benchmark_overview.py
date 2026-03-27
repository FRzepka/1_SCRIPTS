import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parent
MODELS = [
    "CC_1.0.0",
    "CC_SOH_1.0.0",
    "ECM_0.0.1",
    "SOC_SOH_1.6.0.0_0.1.2.3",
]
SCENARIO_ORDER = [
    "baseline",
    "current_offset",
    "voltage_offset",
    "temp_offset",
    "current_noise",
    "voltage_noise",
    "temp_noise",
    "adc_quantization",
    "spikes",
    "initial_soc_error",
    "missing_samples",
    "irregular_sampling",
    "missing_gap",
    "temp_mask",
    "downsample",
    "missing_segments",
]


def latest_summary(model: str, scenario: str) -> Optional[Path]:
    base = ROOT / model / "runs" / scenario
    if not base.exists():
        return None
    candidates = sorted(base.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_summary(path: Path) -> Dict:
    data = json.loads(path.read_text())
    data["run_dir"] = str(path.parent.relative_to(ROOT))
    return data


def build_rows() -> List[Dict]:
    rows: List[Dict] = []
    for model in MODELS:
        baseline_path = latest_summary(model, "baseline")
        baseline = load_summary(baseline_path) if baseline_path else {}
        base_mae = float(baseline.get("mae")) if baseline.get("mae") is not None else None
        base_rmse = float(baseline.get("rmse")) if baseline.get("rmse") is not None else None
        for scenario in SCENARIO_ORDER:
            summary_path = latest_summary(model, scenario)
            if not summary_path:
                continue
            data = load_summary(summary_path)
            mae = data.get("mae")
            rmse = data.get("rmse")
            row = {
                "model": model,
                "cell": data.get("cell", ""),
                "scenario": scenario,
                "mae": mae,
                "rmse": rmse,
                "p95_error": data.get("p95_error"),
                "max_error": data.get("max_error"),
                "jump_count_gt_5pct": data.get("jump_count_gt_5pct"),
                "drift_rate_abs_err_per_h": data.get("drift_rate_abs_err_per_h"),
                "disturbed_mae": data.get("disturbed_mae"),
                "post_disturbance_mae": data.get("post_disturbance_mae"),
                "recovery_time_h": data.get("recovery_time_h"),
                "delta_mae": None if base_mae is None or mae is None else float(mae) - base_mae,
                "delta_rmse": None if base_rmse is None or rmse is None else float(rmse) - base_rmse,
                "run_dir": data.get("run_dir", ""),
            }
            rows.append(row)
    return rows


def main() -> None:
    rows = build_rows()
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No summary files found.")

    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    df = df.sort_values(["model", "scenario"]).reset_index(drop=True)

    csv_path = ROOT / "benchmark_overview_latest.csv"
    md_path = ROOT / "benchmark_overview_latest.md"
    short_path = ROOT / "benchmark_overview_short.csv"

    df.to_csv(csv_path, index=False)
    df[
        [
            "model",
            "scenario",
            "mae",
            "rmse",
            "p95_error",
            "max_error",
            "jump_count_gt_5pct",
            "delta_mae",
            "delta_rmse",
            "recovery_time_h",
            "run_dir",
        ]
    ].to_csv(short_path, index=False)
    md_path.write_text(df.to_markdown(index=False))

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {short_path}")


if __name__ == "__main__":
    main()
