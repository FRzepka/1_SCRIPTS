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
    "ECM_0.0.3": ROOT / "ECM_0.0.3" / "run_ecm_scenario.py",
    "SOC_SOH_1.7.0.0_0.1.2.3": ROOT / "SOC_SOH_1.7.0.0_0.1.2.3" / "run_soc_soh_scenario.py",
}
LEVELS = [("0p02", 0.02), ("0p10", 0.10), ("0p15", 0.15), ("0p20", 0.20)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the current-noise detail sweep for one model.")
    ap.add_argument("--model", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    runner = RUNNERS[args.model]
    campaign_dir = ROOT / "campaigns" / args.tag
    campaign_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tag": args.tag,
        "model": args.model,
        "cell": args.cell,
        "runner": str(runner),
        "python": sys.executable,
        "levels": [std for _, std in LEVELS],
        "started_utc": datetime.utcnow().isoformat(),
        "runs": [],
    }

    for suffix, std in LEVELS:
        run_name = f"{datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')}_{args.tag}_std_{suffix}"
        out_dir = ROOT / args.model / "runs" / "current_noise" / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(runner),
            "--cell",
            args.cell,
            "--scenario",
            "current_noise",
            "--current_noise_std",
            f"{std:.2f}",
            "--out_dir",
            str(out_dir),
            *args.extra,
        ]
        if args.model == "ECM_0.0.3":
            cmd.extend(["--soh_mode", "gru"])
        print("RUN", " ".join(cmd), flush=True)
        rec = {
            "std": std,
            "cmd": cmd,
            "out_dir": str(out_dir),
            "started_utc": datetime.utcnow().isoformat(),
        }
        subprocess.run(cmd, check=True)
        rec["finished_utc"] = datetime.utcnow().isoformat()
        manifest["runs"].append(rec)
        (campaign_dir / f"{args.model}_current_noise_manifest.json").write_text(json.dumps(manifest, indent=2))

    manifest["finished_utc"] = datetime.utcnow().isoformat()
    (campaign_dir / f"{args.model}_current_noise_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
