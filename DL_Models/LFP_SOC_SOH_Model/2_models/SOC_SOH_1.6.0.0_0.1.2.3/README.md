# SOC+SOH Model Chain (SOC_1.6.0.0 + SOH_0.1.2.3)

This folder provides a standalone simulation script that computes SOH hourly using the SOH model (0.1.2.3) and feeds the held SOH value into the SOC model (1.6.0.0).

## Contents
- `simulate_soh_soc.py` – end-to-end SOH→SOC simulation

## Quick start
```bash
source /home/florianr/anaconda3/etc/profile.d/conda.sh
conda activate ml1

python simulate_soh_soc.py \
  --out_dir /tmp/soc_soh_sim \
  --cell MGFarm_18650_C07 \
  --device cuda \
  --soc_batch 32
```

Outputs:
- `soh_hourly_<CELL>.csv` + `.png`
- `soc_pred_fullcell_<CELL>.csv` + `.png`
- `summary.json`
