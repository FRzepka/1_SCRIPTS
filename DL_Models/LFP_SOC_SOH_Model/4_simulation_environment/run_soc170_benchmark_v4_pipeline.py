import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT.parents[2]
PYTHON = Path("/home/florianr/anaconda3/envs/ml1/bin/python")

EXT_TAG = "2026-03-27_extended_matrix_fullc07_v4_soc170_s30ft_soh0125_s30ft"
BIAS_TAG = "bias_0p5_1p5_3p0_percent_fullc07_v4_soc170_s30ft_soh0125_s30ft"
NOISE_TAG = "2026-03-27_current_noise_detail_fullc07_v4_soc170_s30ft_soh0125_s30ft"
LOCAL_OUT = ROOT / "analysis_local_focus" / "2026-03-27_local_recovery_v4"
NOISE_OUT = ROOT / "results" / "noise_detail_v4"
PAPER_TABLES_OUT = ROOT / "results" / "paper_tables_v4"
PAPER_FIGURES_OUT = ROOT / "results" / "paper_figures_v4"
PAPER_MANIFEST_OUT = ROOT / "results" / "paper_results_manifest_v4.json"
CELL = "MGFarm_18650_C07"

SOC_CONFIG = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/PrunedFT_1.7.0.0_s30_struct/config/train_soc.yaml")
SOC_CKPT = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/PrunedFT_1.7.0.0_s30_struct/checkpoints/best_model_finetuned.pt")
SOC_SCALER = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/PrunedFT_1.7.0.0_s30_struct/scaler_robust.joblib")

SOH_CONFIG = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Pruned/0.1.2.5_base_h160_s30_struct_ft/config/train_soh.yaml")
SOH_CKPT = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Pruned/0.1.2.5_base_h160_s30_struct_ft/checkpoints/best_model_finetuned.pt")
SOH_SCALER = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Pruned/0.1.2.5_base_h160_s30_struct_ft/scaler_robust.joblib")


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
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    NOISE_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    common_soh_args = " ".join(
        [
            "--soh_config", shlex.quote(str(SOH_CONFIG)),
            "--soh_ckpt", shlex.quote(str(SOH_CKPT)),
            "--soh_scaler", shlex.quote(str(SOH_SCALER)),
        ]
    )
    dd_model_args = " ".join(
        [
            "--soc_config", shlex.quote(str(SOC_CONFIG)),
            "--soc_ckpt", shlex.quote(str(SOC_CKPT)),
            "--soc_scaler", shlex.quote(str(SOC_SCALER)),
            "--soh_config", shlex.quote(str(SOH_CONFIG)),
            "--soh_ckpt", shlex.quote(str(SOH_CKPT)),
            "--soh_scaler", shlex.quote(str(SOH_SCALER)),
        ]
    )

    jobs = {
        "benchv4_cc": (
            f"cd {shlex.quote(str(WORKDIR))} && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_extended_robustness_matrix.py')} --model CC_1.0.0 --cell {CELL} --tag {EXT_TAG} "
            f"> {shlex.quote(str(campaign_dir / 'CC_1.0.0_extended.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_bias_percent_sweep.py')} --model CC_1.0.0 --cell {CELL} --tag {BIAS_TAG} "
            f"> {shlex.quote(str(campaign_dir / 'CC_1.0.0_bias.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_current_noise_detail_sweep.py')} --model CC_1.0.0 --cell {CELL} --tag {NOISE_TAG} "
            f"> {shlex.quote(str(campaign_dir / 'CC_1.0.0_noise.log'))} 2>&1"
        ),
        "benchv4_ccsoh": (
            f"cd {shlex.quote(str(WORKDIR))} && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_extended_robustness_matrix.py')} --model CC_SOH_1.0.0 --cell {CELL} --tag {EXT_TAG} --extra --device cpu {common_soh_args} "
            f"> {shlex.quote(str(campaign_dir / 'CC_SOH_1.0.0_extended.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_bias_percent_sweep.py')} --model CC_SOH_1.0.0 --cell {CELL} --tag {BIAS_TAG} --extra --device cpu {common_soh_args} "
            f"> {shlex.quote(str(campaign_dir / 'CC_SOH_1.0.0_bias.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_current_noise_detail_sweep.py')} --model CC_SOH_1.0.0 --cell {CELL} --tag {NOISE_TAG} --extra --device cpu {common_soh_args} "
            f"> {shlex.quote(str(campaign_dir / 'CC_SOH_1.0.0_noise.log'))} 2>&1"
        ),
        "benchv4_ecm": (
            f"cd {shlex.quote(str(WORKDIR))} && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_extended_robustness_matrix.py')} --model ECM_0.0.3 --cell {CELL} --tag {EXT_TAG} --extra --device cpu {common_soh_args} "
            f"> {shlex.quote(str(campaign_dir / 'ECM_0.0.3_extended.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_bias_percent_sweep.py')} --model ECM_0.0.3 --cell {CELL} --tag {BIAS_TAG} --extra --device cpu {common_soh_args} "
            f"> {shlex.quote(str(campaign_dir / 'ECM_0.0.3_bias.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_current_noise_detail_sweep.py')} --model ECM_0.0.3 --cell {CELL} --tag {NOISE_TAG} --extra --device cpu {common_soh_args} "
            f"> {shlex.quote(str(campaign_dir / 'ECM_0.0.3_noise.log'))} 2>&1"
        ),
        "benchv4_soc170": (
            f"cd {shlex.quote(str(WORKDIR))} && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_extended_robustness_matrix.py')} --model SOC_SOH_1.7.0.0_0.1.2.3 --cell {CELL} --tag {EXT_TAG} --extra {dd_model_args} "
            f"> {shlex.quote(str(campaign_dir / 'SOC_SOH_1.7.0.0_0.1.2.3_extended.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_bias_percent_sweep.py')} --model SOC_SOH_1.7.0.0_0.1.2.3 --cell {CELL} --tag {BIAS_TAG} --extra {dd_model_args} "
            f"> {shlex.quote(str(campaign_dir / 'SOC_SOH_1.7.0.0_0.1.2.3_bias.log'))} 2>&1 && "
            f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_current_noise_detail_sweep.py')} --model SOC_SOH_1.7.0.0_0.1.2.3 --cell {CELL} --tag {NOISE_TAG} --extra {dd_model_args} "
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

    print("[POST] local recovery analysis v4", flush=True)
    subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "analyze_local_recovery_focus_v2.py"),
            "--campaign_tag",
            EXT_TAG,
            "--out_dir",
            str(LOCAL_OUT),
        ],
        cwd=WORKDIR,
        check=True,
    )

    print("[POST] current-noise detail v4", flush=True)
    subprocess.run(
        [str(PYTHON), str(ROOT / "results" / "analyze_current_noise_detail_v2.py"), "--out_dir", str(NOISE_OUT)],
        cwd=WORKDIR,
        check=True,
    )
    subprocess.run(
        [str(PYTHON), str(ROOT / "results" / "analyze_current_noise_output_jitter_v2.py"), "--out_dir", str(NOISE_OUT)],
        cwd=WORKDIR,
        check=True,
    )

    print("[POST] curated paper outputs v4", flush=True)
    subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "results" / "build_curated_paper_results_v2.py"),
            "--campaign_tag",
            EXT_TAG,
            "--cell",
            CELL,
            "--paper_tables_dir",
            str(PAPER_TABLES_OUT),
            "--paper_figures_dir",
            str(PAPER_FIGURES_OUT),
            "--local_analysis_dir",
            str(LOCAL_OUT),
            "--noise_detail_dir",
            str(NOISE_OUT),
            "--bias_campaign_tag",
            BIAS_TAG,
            "--bias_ecm_campaign_tag",
            BIAS_TAG,
            "--manifest_out",
            str(PAPER_MANIFEST_OUT),
        ],
        cwd=WORKDIR,
        check=True,
    )
    print("[DONE] benchmark v4 complete", flush=True)


if __name__ == "__main__":
    main()
