import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT.parents[2]
PYTHON = Path("/home/florianr/anaconda3/envs/ml1/bin/python")

V4_EXT_TAG = "2026-03-27_extended_matrix_fullc07_v4_soc170_s30ft_soh0125_s30ft"
V4_BIAS_TAG = "bias_0p5_1p5_3p0_percent_fullc07_v4_soc170_s30ft_soh0125_s30ft"
V4_NOISE_TAG = "2026-03-27_current_noise_detail_fullc07_v4_soc170_s30ft_soh0125_s30ft"

EXT_TAG = "2026-03-31_extended_matrix_fullc07_v5_soc171_interim_soh0125_s30ft"
BIAS_TAG = "bias_0p5_1p5_3p0_percent_fullc07_v5_soc171_interim_soh0125_s30ft"
NOISE_TAG = "2026-03-31_current_noise_detail_fullc07_v5_soc171_interim_soh0125_s30ft"
LOCAL_OUT = ROOT / "analysis_local_focus" / "2026-03-31_local_recovery_v5"
NOISE_OUT = ROOT / "results" / "noise_detail_v5"
PAPER_TABLES_OUT = ROOT / "results" / "paper_tables_v5"
PAPER_FIGURES_OUT = ROOT / "results" / "paper_figures_v5"
PAPER_MANIFEST_OUT = ROOT / "results" / "paper_results_manifest_v5.json"
CELL = "MGFarm_18650_C07"

SOC_CONFIG = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/1_training/1.7.1.0/config/train_soc_best_from_hpt.yaml")
SOC_SCALER = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/1_training/1.7.1.0/outputs/scaler_robust.joblib")
SOC_CKPT_DIR = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/1_training/1.7.1.0/outputs/checkpoints")

SOH_CONFIG = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Pruned/0.1.2.5_base_h160_s30_struct_ft/config/train_soh.yaml")
SOH_CKPT = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Pruned/0.1.2.5_base_h160_s30_struct_ft/checkpoints/best_model_finetuned.pt")
SOH_SCALER = Path("/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Pruned/0.1.2.5_base_h160_s30_struct_ft/scaler_robust.joblib")

NON_SOC_MODELS = ["CC_1.0.0", "CC_SOH_1.0.0", "ECM_0.0.3"]
RMSE_RE = re.compile(r"rmse([0-9.]+)\.pt$")


def screen_exists(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], cwd=WORKDIR, capture_output=True, text=True, check=False)
    return name in result.stdout


def launch(name: str, cmd: str) -> None:
    if screen_exists(name):
        return
    subprocess.run(["screen", "-dmS", name, "bash", "-lc", cmd], cwd=WORKDIR, check=True)


def py(script: str) -> str:
    return shlex.quote(str(ROOT / script))


def best_soc_ckpt() -> Path:
    best = None
    for path in sorted(SOC_CKPT_DIR.glob("soc_epoch*_rmse*.pt")):
        m = RMSE_RE.search(path.name)
        if not m:
            continue
        rmse = float(m.group(1))
        if best is None or rmse < best[0]:
            best = (rmse, path)
    if best is None:
        raise FileNotFoundError(f"No SOC checkpoints found in {SOC_CKPT_DIR}")
    return best[1]


def latest_tagged_run(alias_dir: Path, source_tag: str) -> Optional[Path]:
    tagged = sorted(
        [p for p in alias_dir.iterdir() if p.is_dir() and source_tag in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    return tagged[-1] if tagged else None


def ensure_symlink_clone(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst, target_is_directory=True)


def clone_v4_runs_for_tag(source_tag: str, target_tag: str) -> None:
    for model in NON_SOC_MODELS:
        runs_root = ROOT / model / "runs"
        if not runs_root.exists():
            continue
        for alias_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            src = latest_tagged_run(alias_dir, source_tag)
            if src is None:
                continue
            dst_name = src.name.replace(source_tag, target_tag)
            if dst_name == src.name:
                dst_name = f"{src.name}_{target_tag}"
            dst = alias_dir / dst_name
            ensure_symlink_clone(src, dst)


def main() -> None:
    campaign_dir = ROOT / "campaigns" / EXT_TAG
    campaign_dir.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    NOISE_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    soc_ckpt = best_soc_ckpt()
    print(f"[INFO] Using SOC checkpoint: {soc_ckpt}", flush=True)

    print("[PREP] cloning non-SOC v4 runs into v5 tags", flush=True)
    clone_v4_runs_for_tag(V4_EXT_TAG, EXT_TAG)
    clone_v4_runs_for_tag(V4_BIAS_TAG, BIAS_TAG)
    clone_v4_runs_for_tag(V4_NOISE_TAG, NOISE_TAG)

    common_soh_args = " ".join(
        [
            "--soh_config", shlex.quote(str(SOH_CONFIG)),
            "--soh_ckpt", shlex.quote(str(SOH_CKPT)),
            "--soh_scaler", shlex.quote(str(SOH_SCALER)),
        ]
    )
    dd_model_args = " ".join(
        [
            "--device", "cuda",
            "--require_gpu",
            "--soc_config", shlex.quote(str(SOC_CONFIG)),
            "--soc_ckpt", shlex.quote(str(soc_ckpt)),
            "--soc_scaler", shlex.quote(str(SOC_SCALER)),
            "--soh_config", shlex.quote(str(SOH_CONFIG)),
            "--soh_ckpt", shlex.quote(str(SOH_CKPT)),
            "--soh_scaler", shlex.quote(str(SOH_SCALER)),
        ]
    )

    soc_job = (
        f"cd {shlex.quote(str(WORKDIR))} && "
        f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_extended_robustness_matrix.py')} "
        f"--model SOC_SOH_1.7.0.0_0.1.2.3 --cell {CELL} --tag {EXT_TAG} --extra {dd_model_args} "
        f"> {shlex.quote(str(campaign_dir / 'SOC_SOH_1.7.1.0_interim_0.1.2.5_extended.log'))} 2>&1 && "
        f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_bias_percent_sweep.py')} "
        f"--model SOC_SOH_1.7.0.0_0.1.2.3 --cell {CELL} --tag {BIAS_TAG} --extra {dd_model_args} "
        f"> {shlex.quote(str(campaign_dir / 'SOC_SOH_1.7.1.0_interim_0.1.2.5_bias.log'))} 2>&1 && "
        f"PYTHONUNBUFFERED=1 {shlex.quote(str(PYTHON))} {py('run_current_noise_detail_sweep.py')} "
        f"--model SOC_SOH_1.7.0.0_0.1.2.3 --cell {CELL} --tag {NOISE_TAG} --extra {dd_model_args} "
        f"> {shlex.quote(str(campaign_dir / 'SOC_SOH_1.7.1.0_interim_0.1.2.5_noise.log'))} 2>&1"
    )

    launch("benchv5_soc171", soc_job)
    print("[LAUNCH] benchv5_soc171", flush=True)

    while screen_exists("benchv5_soc171"):
        print("[WAIT] active screens: benchv5_soc171", flush=True)
        time.sleep(60)

    print("[POST] local recovery analysis v5", flush=True)
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

    print("[POST] current-noise detail v5", flush=True)
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

    print("[POST] curated paper outputs v5", flush=True)
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
    print("[DONE] benchmark v5 complete", flush=True)


if __name__ == "__main__":
    main()
