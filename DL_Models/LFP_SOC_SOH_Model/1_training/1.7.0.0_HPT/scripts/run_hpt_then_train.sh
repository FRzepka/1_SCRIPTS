#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/florianr/MG_Farm/1_Scripts"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate ml1
cd "$ROOT"
python DL_Models/LFP_SOC_SOH_Model/1_training/1.7.0.0_HPT/scripts/hpt_soc.py \
  --config DL_Models/LFP_SOC_SOH_Model/1_training/1.7.0.0_HPT/config/hpt_soc.yaml
python DL_Models/LFP_SOC_SOH_Model/1_training/1.7.0.0/scripts/train_soc.py \
  --config DL_Models/LFP_SOC_SOH_Model/1_training/1.7.0.0/config/train_soc_best_from_hpt.yaml
