import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT.parents[2]
BIAS_TAG = 'bias_0p5_1p5_3p0_percent_fullc07_ecm003'
EXT_TAG = '2026-03-12_extended_matrix_fullc07'


def ecm_bias_running() -> bool:
    cmd = "ps -ef | grep 'run_bias_percent_sweep.py --model ECM_0.0.3' | grep -v grep >/dev/null"
    return subprocess.run(cmd, shell=True).returncode == 0


def run_cmd(cmd):
    print('RUN', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=WORKDIR)


def main() -> None:
    while ecm_bias_running():
        print('[PIPELINE] waiting for ECM_0.0.3 bias sweep to finish...', flush=True)
        time.sleep(60)

    print('[PIPELINE] ECM_0.0.3 bias sweep finished; launching parallel extended ECM_0.0.3 suite', flush=True)
    run_cmd([
        sys.executable,
        str(ROOT / 'run_ecm003_extended_parallel.py'),
        '--cell', 'MGFarm_18650_C07',
        '--tag_prefix', EXT_TAG,
    ])

    print('[PIPELINE] rebuilding curated paper figures', flush=True)
    run_cmd([
        sys.executable,
        str(ROOT / 'results' / 'build_curated_paper_results.py'),
        '--campaign_tag', EXT_TAG,
    ])

    print('[PIPELINE] rebuilding bias-percent results with ECM_0.0.3 override', flush=True)
    run_cmd([
        sys.executable,
        str(ROOT / 'results' / 'build_bias_percent_results.py'),
        '--campaign_tag', 'bias_0p5_1p5_3p0_percent_fullc07',
        '--ecm_campaign_tag', BIAS_TAG,
    ])

    print('[PIPELINE] building scenario detail folders', flush=True)
    run_cmd([
        sys.executable,
        str(ROOT / 'results' / 'build_scenario_detail_outputs.py'),
        '--campaign_tag', EXT_TAG,
        '--out_root',
        str(ROOT / 'results' / 'noise_detail'),
    ])

    print('[PIPELINE] done', flush=True)


if __name__ == '__main__':
    main()
