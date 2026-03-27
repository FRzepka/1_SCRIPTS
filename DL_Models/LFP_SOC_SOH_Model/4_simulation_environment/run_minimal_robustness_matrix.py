import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNERS = {
    "CC_1.0.0": ROOT / "CC_1.0.0" / "run_cc_scenario.py",
    "CC_SOH_1.0.0": ROOT / "CC_SOH_1.0.0" / "run_cc_soh_scenario.py",
    "ECM_0.0.1": ROOT / "ECM_0.0.1" / "run_ecm_scenario.py",
    "ECM_0.0.3": ROOT / "ECM_0.0.3" / "run_ecm_scenario.py",
    "SOC_SOH_1.6.0.0_0.1.2.3": ROOT / "SOC_SOH_1.6.0.0_0.1.2.3" / "run_soc_soh_scenario.py",
}


def scenario_matrix(model: str):
    rows = [
        ("baseline", []),
        ("current_noise", ["--current_noise_std", "0.02"]),
        ("current_offset", ["--current_offset_a", "0.05"]),
        ("missing_samples", ["--missing_samples_every", "50"]),
        ("spikes", ["--spike_channel", "Voltage[V]", "--spike_magnitude", "0.05", "--spike_period", "1000"]),
    ]
    if model != "SOC_SOH_1.6.0.0_0.1.2.3":
        # C07 starts with a saturated SOC initialization for the current CC/ECM setup.
        # A positive delta clips away and does not stress the estimator. Use a negative
        # initialization error and remove warmup masking so the trend is visible.
        rows.insert(3, ("initial_soc_error", ["--soc_init_error", "-0.10", "--warmup_seconds", "0"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the minimal robustness benchmark matrix for one model.")
    ap.add_argument("--model", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    tag = args.tag or datetime.utcnow().strftime("%Y-%m-%d_%H%M_minimal_matrix")
    runner = RUNNERS[args.model]
    campaign_dir = ROOT / "campaigns" / tag
    campaign_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tag": tag,
        "model": args.model,
        "cell": args.cell,
        "runner": str(runner),
        "python": sys.executable,
        "started_utc": datetime.utcnow().isoformat(),
        "runs": [],
    }

    for scenario, scenario_args in scenario_matrix(args.model):
        run_name = f"{datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')}_{tag}_{scenario}"
        out_dir = ROOT / args.model / "runs" / scenario / run_name
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
        rec = {
            "scenario": scenario,
            "cmd": cmd,
            "out_dir": str(out_dir),
            "status": "running",
            "started_utc": datetime.utcnow().isoformat(),
        }
        print("RUN", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
            rec["status"] = "completed"
        except subprocess.CalledProcessError as exc:
            rec["status"] = "failed"
            rec["returncode"] = exc.returncode
            manifest["runs"].append(rec)
            (campaign_dir / f"{args.model}_manifest.json").write_text(json.dumps(manifest, indent=2))
            raise
        rec["finished_utc"] = datetime.utcnow().isoformat()
        manifest["runs"].append(rec)
        (campaign_dir / f"{args.model}_manifest.json").write_text(json.dumps(manifest, indent=2))

    manifest["finished_utc"] = datetime.utcnow().isoformat()
    (campaign_dir / f"{args.model}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
