import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT.parents[2]

ALIAS_GROUPS = {
    "g1": ["baseline", "current_noise_low", "current_noise_high", "voltage_noise"],
    "g2": ["temp_noise", "current_offset", "voltage_offset", "temp_offset"],
    "g3": ["adc_quantization", "initial_soc_error", "missing_samples"],
    "g4": ["irregular_sampling", "missing_gap", "spikes_high"],
}


def _screen_exists(name: str) -> bool:
    result = subprocess.run(
        ["screen", "-ls"],
        cwd=WORKDIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return name in result.stdout


def _launch_group(group_name: str, aliases: list[str], tag_prefix: str, cell: str) -> str:
    screen_name = f"ecm003_{group_name}"
    if _screen_exists(screen_name):
        return screen_name

    tag = f"{tag_prefix}_ecm003_{group_name}"
    log_path = ROOT / "campaigns" / f"{tag}.log"
    alias_args = " ".join(shlex.quote(a) for a in aliases)
    cmd = (
        f"cd {shlex.quote(str(WORKDIR))} && "
        f"PYTHONUNBUFFERED=1 {shlex.quote(sys.executable)} "
        f"{shlex.quote(str(ROOT / 'run_extended_robustness_matrix.py'))} "
        f"--model ECM_0.0.3 --cell {shlex.quote(cell)} --tag {shlex.quote(tag)} "
        f"--aliases {alias_args} > {shlex.quote(str(log_path))} 2>&1"
    )
    subprocess.run(
        ["screen", "-dmS", screen_name, "bash", "-lc", cmd],
        cwd=WORKDIR,
        check=True,
    )
    return screen_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="MGFarm_18650_C07")
    ap.add_argument("--tag_prefix", default="2026-03-12_extended_matrix_fullc07")
    args = ap.parse_args()

    screens = []
    for group_name, aliases in ALIAS_GROUPS.items():
        screen_name = _launch_group(group_name, aliases, args.tag_prefix, args.cell)
        screens.append(screen_name)
        print(f"[LAUNCH] {screen_name}: {', '.join(aliases)}", flush=True)

    while True:
        alive = [name for name in screens if _screen_exists(name)]
        if not alive:
            break
        print(f"[WAIT] active screens: {', '.join(alive)}", flush=True)
        time.sleep(30)

    print("[DONE] all ECM_0.0.3 extended groups finished", flush=True)


if __name__ == "__main__":
    main()
