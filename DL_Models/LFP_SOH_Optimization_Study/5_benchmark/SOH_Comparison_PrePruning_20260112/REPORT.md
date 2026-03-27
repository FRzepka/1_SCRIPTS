# SOH Pre-Pruning Benchmark Report

Generated results: `/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_SOH_Optimization_Study/5_benchmark/SOH_Comparison_PrePruning_20260112/results_base/RESULTS_20260112_134339`

Notes: All metrics/plots use hourly aggregation with target = last-of-hour for comparability.

## Summary (Test Cells)

| Model | Params | Param Size (KB) | Checkpoint Size (KB) | Test MAE | Test RMSE | Test R² |
|------|--------:|----------------:|---------------------:|--------:|----------:|--------:|
| TCN 0.2.2.1 | 266,977 | 1042.9 | 3164.1 | 0.03778 | 0.04436 | -0.3743 |
| GRU 0.3.1.1 | 1,919,617 | 7498.5 | 22554.7 | 0.04071 | 0.04501 | -1.1215 |
| LSTM 0.1.2.3 | 1,041,185 | 4067.1 | 12249.7 | 0.05119 | 0.05881 | -2.9105 |
| CNN 0.4.1.1 | 2,056,929 | 8034.9 | 24136.3 | 0.11657 | 0.13330 | -6.8343 |

## Plots

- `RESULTS_20260112_134339/plots/MGFarm_18650_C11_comparison.png`
- `RESULTS_20260112_134339/plots/MGFarm_18650_C19_comparison.png`
- `RESULTS_20260112_134339/plots/MGFarm_18650_C23_comparison.png`
- `RESULTS_20260112_134339/plots/test_mae_bar.png`
- `RESULTS_20260112_134339/plots/test_rmse_bar.png`
- `RESULTS_20260112_134339/plots/test_r2_bar.png`
- `RESULTS_20260112_134339/plots/params_bar.png`

## Raw Outputs

- `RESULTS_20260112_134339/metrics_summary.csv`
- `RESULTS_20260112_134339/metrics_by_cell.csv`
- `RESULTS_20260112_134339/benchmark_meta.json`
