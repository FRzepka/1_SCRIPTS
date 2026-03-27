import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT.parents[2]
RESULTS_DIR = ROOT / "results"
LEVELS = [
    ("0p02", 0.02),
    ("0p10", 0.10),
    ("0p15", 0.15),
    ("0p20", 0.20),
]


def screen_exists(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], cwd=WORKDIR, capture_output=True, text=True, check=False)
    return name in result.stdout


def launch_run(cell: str, tag: str, suffix: str, std: float) -> str:
    screen_name = f"ecm_noise_{suffix}"
    if screen_exists(screen_name):
        return screen_name

    out_dir = ROOT / "ECM_0.0.3" / "runs" / "current_noise" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{tag}_std_{suffix}"
    log_path = ROOT / "campaigns" / f"{tag}_{suffix}.log"
    cmd = (
        f"cd {shlex.quote(str(WORKDIR))} && "
        f"PYTHONUNBUFFERED=1 {shlex.quote(sys.executable)} "
        f"{shlex.quote(str(ROOT / 'ECM_0.0.3' / 'run_ecm_scenario.py'))} "
        f"--cell {shlex.quote(cell)} "
        f"--scenario current_noise "
        f"--current_noise_std {std:.2f} "
        f"--soh_mode gru "
        f"--out_dir {shlex.quote(str(out_dir))} > {shlex.quote(str(log_path))} 2>&1"
    )
    subprocess.run(["screen", "-dmS", screen_name, "bash", "-lc", cmd], cwd=WORKDIR, check=True)
    return screen_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--tag", default="2026-03-16_noise_sweep_ecm003")
    args = ap.parse_args()

    screens = []
    for suffix, std in LEVELS:
        screen_name = launch_run(args.cell, args.tag, suffix, std)
        screens.append(screen_name)
        print(f"[LAUNCH] {screen_name}: sigma_I={std:.2f} A", flush=True)

    while True:
        alive = [name for name in screens if screen_exists(name)]
        if not alive:
            break
        print(f"[WAIT] active screens: {', '.join(alive)}", flush=True)
        time.sleep(30)

    print("[POST] rebuilding noise detail analysis", flush=True)
    subprocess.run(
        [sys.executable, str(RESULTS_DIR / "analyze_current_noise_detail.py")],
        cwd=WORKDIR,
        check=True,
    )

    print("[POST] rebuilding curated paper figures", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(RESULTS_DIR / "build_curated_paper_results.py"),
            "--campaign_tag",
            "2026-03-12_extended_matrix_fullc07",
        ],
        cwd=WORKDIR,
        check=True,
    )
    print("[DONE] ECM_0.0.3 noise sweep and rebuild complete", flush=True)


if __name__ == "__main__":
    main()
