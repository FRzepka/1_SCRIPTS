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
    "SOC_SOH_1.7.0.0_0.1.2.3": ROOT / "SOC_SOH_1.7.0.0_0.1.2.3" / "run_soc_soh_scenario.py",
}


def scenario_matrix(model: str):
    rows = [
        ("baseline", "baseline", []),
        ("current_noise_low", "current_noise", ["--current_noise_std", "0.02"]),
        ("current_noise_high", "current_noise", ["--current_noise_std", "0.10"]),
        ("voltage_noise", "voltage_noise", ["--voltage_noise_std", "0.01"]),
        ("temp_noise", "temp_noise", ["--temp_noise_std", "1.0"]),
        ("current_offset", "current_offset", ["--current_offset_a", "0.05"]),
        ("voltage_offset", "voltage_offset", ["--voltage_offset_v", "0.02"]),
        ("temp_offset", "temp_offset", ["--temp_offset_c", "3.0"]),
        ("adc_quantization", "adc_quantization", []),
        ("missing_samples", "missing_samples", ["--missing_samples_every", "50"]),
        ("irregular_sampling", "irregular_sampling", ["--irregular_dt_jitter", "0.1"]),
        ("missing_gap", "missing_gap", ["--missing_gap_seconds", "3600"]),
        ("spikes_high", "spikes", ["--spike_channel", "Voltage[V]", "--spike_magnitude", "0.20", "--spike_period", "1000"]),
    ]
    if model != "SOC_SOH_1.6.0.0_0.1.2.3":
        rows.insert(9, ("initial_soc_error", "initial_soc_error", ["--soc_init_error", "-0.10", "--warmup_seconds", "0"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the extended robustness benchmark matrix for one model.")
    ap.add_argument("--model", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--aliases", nargs="*", default=None, help="Optional subset of scenario aliases to run.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip runs whose summary.json already exists.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    tag = args.tag or datetime.utcnow().strftime("%Y-%m-%d_%H%M_extended_matrix")
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

    selected_aliases = set(args.aliases) if args.aliases else None

    for alias, scenario, scenario_args in scenario_matrix(args.model):
        if selected_aliases and alias not in selected_aliases:
            continue
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
        rec = {
            "alias": alias,
            "scenario": scenario,
            "cmd": cmd,
            "out_dir": str(out_dir),
            "status": "running",
            "started_utc": datetime.utcnow().isoformat(),
        }
        if args.skip_existing:
            latest_completed = None
            for prev in reversed(manifest["runs"]):
                if prev.get("alias") != alias or prev.get("status") != "completed":
                    continue
                prev_summary = Path(prev["out_dir"]) / "summary.json"
                if prev_summary.exists():
                    latest_completed = prev
                    break
            if latest_completed is not None:
                rec["status"] = "skipped_existing"
                rec["finished_utc"] = datetime.utcnow().isoformat()
                rec["out_dir"] = latest_completed["out_dir"]
                manifest["runs"].append(rec)
                (campaign_dir / f"{args.model}_manifest.json").write_text(json.dumps(manifest, indent=2))
                print(f"SKIP {alias} existing={latest_completed['out_dir']}", flush=True)
                continue
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
