# Run 2026-01-30_01 – SOH→SOC Simulation

Simulates SOH using model 0.1.2.3 (hourly updates, hold for 1 hour) and feeds it into SOC model 1.6.0.0.

## How to run (GPU)
```bash
source /home/florianr/anaconda3/etc/profile.d/conda.sh
conda activate ml1
python simulate_soh_soc.py --out_dir . --cell MGFarm_18650_C07 --device cuda --soc_batch 64
```

Outputs:
- `soh_hourly_<CELL>.csv` + `soh_hourly_<CELL>.png`
- `soc_pred_fullcell_<CELL>.csv` + `soc_pred_fullcell_<CELL>.png`
- `summary.json`
