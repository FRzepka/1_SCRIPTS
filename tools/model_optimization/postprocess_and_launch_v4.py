#!/usr/bin/env python3
"""Wait for finetune to finish, copy artifacts to final folders, and launch benchmark v4."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/florianr/MG_Farm/1_Scripts")
SCREEN_FT = "ft_struct30_paper"
SCREEN_V3_PIPE = "benchv3_pipeline"
SCREEN_V3_SOC = "benchv3_soc170"

SOC_SRC = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "2_models" / "SOC_1.7.0.0" / "Pruned_Quantized_1.7.0.0_s30_struct_int8"
SOH_SRC = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / "LSTM" / "Base" / "0.1.2.5_base_h160" / "Pruned_Quantized_0.1.2.5_base_h160_s30_struct_int8"

SOC_DST_PRUNED = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "2_models" / "SOC_1.7.0.0" / "PrunedFT_1.7.0.0_s30_struct"
SOC_DST_QUANT = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "2_models" / "SOC_1.7.0.0" / "Quantized_1.7.0.0_s30_struct_ft_int8"
SOH_DST_PRUNED = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / "LSTM" / "Pruned" / "0.1.2.5_base_h160_s30_struct_ft"
SOH_DST_QUANT = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study" / "2_models" / "LSTM" / "Quantized" / "0.1.2.5_base_h160_s30_struct_ft_int8"

TEST_DST = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "3_test" / "2026-03-27_pruned_quantized_fullcell_validation_v4"
LOG = ROOT / "tools" / "model_optimization" / "logs" / "postprocess_and_launch_v4_2026-03-27.log"
V4_SCRIPT = ROOT / "DL_Models" / "LFP_SOC_SOH_Model" / "4_simulation_environment" / "run_soc170_benchmark_v4_pipeline.py"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def screen_exists(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], capture_output=True, text=True, check=False)
    return name in result.stdout


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def wait_for_ft() -> None:
    summary = ROOT / "tools" / "model_optimization" / "latest_struct30_ftq_fullcell_summary.json"
    while screen_exists(SCREEN_FT) or not summary.exists():
        log("[WAIT] finetune still running or summary missing")
        time.sleep(60)


def wait_for_v3() -> None:
    while screen_exists(SCREEN_V3_PIPE) or screen_exists(SCREEN_V3_SOC):
        log("[WAIT] v3 benchmark still active")
        time.sleep(60)


def copy_outputs() -> None:
    copytree_replace(SOC_SRC / "finetuned_model", SOC_DST_PRUNED)
    copytree_replace(SOC_SRC / "quantized_finetuned_model", SOC_DST_QUANT)
    copytree_replace(SOH_SRC / "finetuned_model", SOH_DST_PRUNED)
    copytree_replace(SOH_SRC / "quantized_finetuned_model", SOH_DST_QUANT)

    TEST_DST.mkdir(parents=True, exist_ok=True)
    copytree_replace(SOC_SRC / "fullcell_compare_C07", TEST_DST / "soc_fullcell_compare_C07")
    copytree_replace(SOH_SRC / "fullcell_compare_C07", TEST_DST / "soh_fullcell_compare_C07")
    shutil.copy2(ROOT / "tools" / "model_optimization" / "latest_struct30_ftq_fullcell_summary.json", TEST_DST / "summary_ftq_fullcell.json")


def launch_v4() -> None:
    if screen_exists("benchv4_pipeline"):
        log("[SKIP] benchv4_pipeline already exists")
        return
    subprocess.run(
        [
            "screen",
            "-dmS",
            "benchv4_pipeline",
            "bash",
            "-lc",
            f"cd {ROOT} && PYTHONUNBUFFERED=1 /home/florianr/anaconda3/envs/ml1/bin/python {V4_SCRIPT} >> {ROOT}/DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/2026-03-27_extended_matrix_fullc07_v4_soc170_s30ft_soh0125_s30ft/pipeline.log 2>&1",
        ],
        check=True,
    )
    log("[LAUNCH] benchv4_pipeline")


def main() -> None:
    log("[START] waiting for finetune")
    wait_for_ft()
    log("[DONE] finetune complete")
    copy_outputs()
    log("[DONE] copied outputs to final model/test folders")
    wait_for_v3()
    log("[DONE] v3 complete")
    launch_v4()
    log("[DONE] v4 launched")


if __name__ == "__main__":
    main()
