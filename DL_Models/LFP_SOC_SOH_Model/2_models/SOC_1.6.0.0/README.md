# LFP SOC+SOH Model 1.6.0.0 (Standalone)

This folder contains a standalone 1.6.0.0 SOC model that uses SOH as an input feature.

## Contents
- `soc_epoch0005_rmse0.01393.pt` – trained checkpoint (best RMSE among available checkpoints)
- `scaler_robust.joblib` – feature scaler used during training
- `train_soc.yaml` – config (features, chunk size, model size)
- `predict_soc.py` – standalone inference script

## Quick start
```bash
source /home/florianr/anaconda3/etc/profile.d/conda.sh
conda activate ml1

python predict_soc.py \
  --config /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/train_soc.yaml \
  --checkpoint /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/soc_epoch0005_rmse0.01393.pt \
  --scaler /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.6.0.0/scaler_robust.joblib \
  --data_root /home/florianr/MG_Farm/0_Data/MGFarm_18650_FE \
  --cell MGFarm_18650_C07 \
  --out_csv /tmp/soc_pred_1.6.0.0.csv \
  --n_rows 10000
```

The output CSV contains `index` and `soc_pred` for the rolling window predictions.
