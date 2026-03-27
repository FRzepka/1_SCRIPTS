# Test Run 2026-01-30_02 – Full-cell SOC plot

This run generates a full-length SOC prediction plot for a single test cell using the 1.6.0.0 model.

## How to run (GPU)
```bash
source /home/florianr/anaconda3/etc/profile.d/conda.sh
conda activate ml1
python run_test_soc_first_10000.py \
  --config /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/1_training/1.6.0.0/config/train_soc.yaml \
  --out_dir /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/3_test/1.6.0.0/2026-01-30_02 \
  --cell MGFarm_18650_C07 \
  --full_cell \
  --device cuda \
  --batch_size 128
```

Adjust `--cell` as needed. If GPU is busy, add `--device cpu`.
