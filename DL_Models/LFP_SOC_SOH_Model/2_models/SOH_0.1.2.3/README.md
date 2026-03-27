# SOH Model 0.1.2.3 (Standalone)

Hourly SOH model (seq2seq) with hourly feature aggregation.

## Contents
- `best_epoch0093_rmse0.02165.pt` – checkpoint
- `scaler_robust.joblib` – feature scaler
- `train_soh.yaml` – config
- `predict_soh.py` – standalone hourly inference script

## Quick start
```bash
source /home/florianr/anaconda3/etc/profile.d/conda.sh
conda activate ml1

python predict_soh.py \
  --config /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/train_soh.yaml \
  --checkpoint /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/best_epoch0093_rmse0.02165.pt \
  --scaler /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOC_SOH_Model/2_models/SOH_0.1.2.3/scaler_robust.joblib \
  --data_root /home/florianr/MG_Farm/0_Data/MGFarm_18650_FE \
  --cell MGFarm_18650_C07 \
  --out_csv /tmp/soh_hourly_c07.csv \
  --device cuda
```
