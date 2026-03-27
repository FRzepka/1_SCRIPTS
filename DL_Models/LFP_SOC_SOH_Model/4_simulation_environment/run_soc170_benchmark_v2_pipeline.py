import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT.parents[2]
PYTHON = Path("/home/florianr/anaconda3/envs/ml1/bin/python")
EXT_TAG = "2026-03-24_extended_matrix_fullc07_v2_soc170"
BIAS_TAG = "bias_0p5_1p5_3p0_percent_fullc07_v2_soc170"
NOISE_TAG = "2026-03-24_current_noise_detail_fullc07_v2_soc170"
CELL = "MGFarm_18650_C07"


def screen_exists(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], cwd=WORKDIR, capture_output=True, text=True, check=False)
    return name in result.stdout


def launch(name: str, cmd: str) -> None:
    if screen_exists(name):
        return
    subprocess.run(["screen", "-dmS", name, "bash", "-lc", cmd], cwd=WORKDIR, check=True)


def py(script: str) -> str:
    return shlex.quote(str(ROOT / script))


def main() -> None:
    campaign_dir = ROOT / "campaigns" / EXT_TAG
    campaign_dir.mkdir(parents=True, exist_ok=True)

    jobs = {
        "benchv2_cc": (
            f"cd {shlex.quote(str(WORKDIR))} && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_extended_robustness_matrix.py')} --model CC_1.0.0 --cell {CELL} --tag {EXT_TAG} "
            f"> {shlex.quote(str(campaign_dir / 'CC_1.0.0_extended.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_bias_percent_sweep.py')} --model CC_1.0.0 --cell {CELL} --tag {BIAS_TAG} "
            f"> {shlex.quote(str(campaign_dir / 'CC_1.0.0_bias.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_current_noise_detail_sweep.py')} --model CC_1.0.0 --cell {CELL} --tag {NOISE_TAG} "
            f"> {shlex.quote(str(campaign_dir / 'CC_1.0.0_noise.log'))} 2>&1"
        ),
        "benchv2_ccsoh": (
            f"cd {shlex.quote(str(WORKDIR))} && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_extended_robustness_matrix.py')} --model CC_SOH_1.0.0 --cell {CELL} --tag {EXT_TAG} --extra --device cpu "
            f"> {shlex.quote(str(campaign_dir / 'CC_SOH_1.0.0_extended.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_bias_percent_sweep.py')} --model CC_SOH_1.0.0 --cell {CELL} --tag {BIAS_TAG} --extra --device cpu "
            f"> {shlex.quote(str(campaign_dir / 'CC_SOH_1.0.0_bias.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_current_noise_detail_sweep.py')} --model CC_SOH_1.0.0 --cell {CELL} --tag {NOISE_TAG} --extra --device cpu "
            f"> {shlex.quote(str(campaign_dir / 'CC_SOH_1.0.0_noise.log'))} 2>&1"
        ),
        "benchv2_ecm": (
            f"cd {shlex.quote(str(WORKDIR))} && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_extended_robustness_matrix.py')} --model ECM_0.0.3 --cell {CELL} --tag {EXT_TAG} --extra --device cpu "
            f"> {shlex.quote(str(campaign_dir / 'ECM_0.0.3_extended.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_bias_percent_sweep.py')} --model ECM_0.0.3 --cell {CELL} --tag {BIAS_TAG} --extra --device cpu "
            f"> {shlex.quote(str(campaign_dir / 'ECM_0.0.3_bias.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_current_noise_detail_sweep.py')} --model ECM_0.0.3 --cell {CELL} --tag {NOISE_TAG} --extra --device cpu "
            f"> {shlex.quote(str(campaign_dir / 'ECM_0.0.3_noise.log'))} 2>&1"
        ),
        "benchv2_soc170": (
            f"cd {shlex.quote(str(WORKDIR))} && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_extended_robustness_matrix.py')} --model SOC_SOH_1.7.0.0_0.1.2.3 --cell {CELL} --tag {EXT_TAG} "
            f"> {shlex.quote(str(campaign_dir / 'SOC_SOH_1.7.0.0_0.1.2.3_extended.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_bias_percent_sweep.py')} --model SOC_SOH_1.7.0.0_0.1.2.3 --cell {CELL} --tag {BIAS_TAG} "
            f"> {shlex.quote(str(campaign_dir / 'SOC_SOH_1.7.0.0_0.1.2.3_bias.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_current_noise_detail_sweep.py')} --model SOC_SOH_1.7.0.0_0.1.2.3 --cell {CELL} --tag {NOISE_TAG} "
            f"> {shlex.quote(str(campaign_dir / 'SOC_SOH_1.7.0.0_0.1.2.3_noise.log'))} 2>&1"
        ),
    }

    for name, cmd in jobs.items():
        launch(name, cmd)
        print(f"[LAUNCH] {name}", flush=True)

    while True:
        alive = [name for name in jobs if screen_exists(name)]
        if not alive:
            break
        print(f"[WAIT] active screens: {', '.join(alive)}", flush=True)
        time.sleep(60)

    print("[POST] local recovery analysis v2", flush=True)
    subprocess.run([
        str(PYTHON),
        str(ROOT / 'analyze_local_recovery_focus_v2.py'),
        '--campaign_tag', EXT_TAG,
    ], cwd=WORKDIR, check=True)

    print("[POST] current-noise detail v2", flush=True)
    subprocess.run([str(PYTHON), str(ROOT / 'results' / 'analyze_current_noise_detail_v2.py')], cwd=WORKDIR, check=True)
    subprocess.run([str(PYTHON), str(ROOT / 'results' / 'analyze_current_noise_output_jitter_v2.py')], cwd=WORKDIR, check=True)

    print("[POST] paper outputs v2", flush=True)
    subprocess.run([str(PYTHON), str(ROOT / 'results' / 'build_paper_results_v2.py'), '--campaign_tag', EXT_TAG], cwd=WORKDIR, check=True)
    subprocess.run([str(PYTHON), str(ROOT / 'results' / 'build_bias_percent_results_v2.py'), '--campaign_tag', BIAS_TAG, '--cell', CELL], cwd=WORKDIR, check=True)
    subprocess.run([str(PYTHON), str(ROOT / 'results' / 'build_curated_paper_results_v2.py'), '--campaign_tag', EXT_TAG, '--cell', CELL], cwd=WORKDIR, check=True)
    print("[DONE] benchmark v2 complete", flush=True)


if __name__ == '__main__':
    main()
