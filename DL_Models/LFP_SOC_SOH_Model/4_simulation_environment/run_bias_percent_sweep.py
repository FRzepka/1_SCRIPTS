import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNNERS = {
    "CC_1.0.0": ROOT / "CC_1.0.0" / "run_cc_scenario.py",
    "CC_SOH_1.0.0": ROOT / "CC_SOH_1.0.0" / "run_cc_soh_scenario.py",
    "ECM_0.0.1": ROOT / "ECM_0.0.1" / "run_ecm_scenario.py",
    "ECM_0.0.3": ROOT / "ECM_0.0.3" / "run_ecm_scenario.py",
    "SOC_SOH_1.6.0.0_0.1.2.3": ROOT / "SOC_SOH_1.6.0.0_0.1.2.3" / "run_soc_soh_scenario.py",
    "SOC_SOH_1.7.0.0_0.1.2.3": ROOT / "SOC_SOH_1.7.0.0_0.1.2.3" / "run_soc_soh_scenario.py",
}
DATA_ROOT = Path("/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
BIAS_LEVELS_PCT = [0.5, 1.5, 3.0]


def cell_path(cell: str) -> Path:
    suffix = cell.split("_")[-1]
    path = DATA_ROOT / f"df_FE_{suffix}.parquet"
    if path.exists():
        return path
    path = DATA_ROOT / f"df_FE_{cell}.parquet"
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not locate parquet for {cell} in {DATA_ROOT}")


def max_abs_current(cell: str) -> float:
    df = pd.read_parquet(cell_path(cell), columns=["Current[A]"])
    return float(df["Current[A]"].abs().max())


def scenario_rows(i_max: float):
    rows = [("baseline", "baseline", [], {"label": "0.0%", "offset_a": 0.0})]
    for pct in BIAS_LEVELS_PCT:
        amp = i_max * pct / 100.0
        alias = f"current_bias_{str(pct).replace('.', 'p')}pct"
        rows.append(
            (
                alias,
                "current_offset",
                ["--current_offset_a", f"{amp:.8f}"],
                {"label": f"{pct:.1f}%", "offset_a": float(amp)},
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description="Run baseline + percent-based current-bias sweep for one model.")
    ap.add_argument("--model", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    runner = RUNNERS[args.model]
    i_max = max_abs_current(args.cell)
    tag = args.tag or f"bias_0p5_1p5_3p0_percent_{args.cell.lower()}"
    campaign_dir = ROOT / "campaigns" / tag
    campaign_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tag": tag,
        "model": args.model,
        "cell": args.cell,
        "runner": str(runner),
        "python": sys.executable,
        "max_abs_current_a": i_max,
        "bias_levels_pct": BIAS_LEVELS_PCT,
        "runs": [],
        "started_utc": datetime.utcnow().isoformat(),
    }

    for alias, scenario, scenario_args, meta in scenario_rows(i_max):
        run_name = f"{datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')}_{tag}_{alias}"
        out_dir = ROOT / args.model / "runs" / alias / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(runner),
            "--cell",
            args.cell,
            "--scenario",
            scenario,
            "--out_dir",
            str(out_dir),
            *scenario_args,
            *args.extra,
        ]
        if args.model in {"ECM_0.0.1", "ECM_0.0.3"}:
            cmd.extend(["--soh_mode", "gru"])
        print("RUN", " ".join(cmd), flush=True)
        rec = {
            "alias": alias,
            "scenario": scenario,
            "meta": meta,
            "cmd": cmd,
            "out_dir": str(out_dir),
            "started_utc": datetime.utcnow().isoformat(),
        }
        subprocess.run(cmd, check=True)
        rec["finished_utc"] = datetime.utcnow().isoformat()
        manifest["runs"].append(rec)
        (campaign_dir / f"{args.model}_manifest.json").write_text(json.dumps(manifest, indent=2))

    manifest["finished_utc"] = datetime.utcnow().isoformat()
    (campaign_dir / f"{args.model}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
