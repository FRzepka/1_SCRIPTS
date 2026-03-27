# 1.6.0.0 Test: SOC prediction on first 10,000 rows

This folder contains a small evaluation that predicts SOC over the first 10,000 rows of a selected cell using the 1.6.0.0 model (with SOH as input).

## Files
- `run_test_soc_first_10000.py` – script that loads the model + scaler and predicts SOC on the first N rows.
- `soc_pred_first_10000_rows_<CELL>.csv` – per-sample predictions (true vs pred).
- `test_summary.json` – summary of settings + metrics.

## How to run
```bash
source /home/florianr/anaconda3/etc/profile.d/conda.sh
conda activate ml1
python run_test_soc_first_10000.py \
  --config /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/1_training/1.6.0.0/config/train_soc.yaml \
  --out_dir /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/3_test/1.6.0.0 \
  --cell MGFarm_18650_C07 \
  --n_rows 10000
```

If `--cell` is omitted, the first validation cell from the config is used.
