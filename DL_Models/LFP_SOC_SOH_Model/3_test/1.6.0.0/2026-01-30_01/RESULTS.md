# Test Results – 1.6.0.0 (SOC first 10,000 rows)

- Date: 2026-01-30
- Cell: `MGFarm_18650_C07`
- Rows used: 10,000 (rolling window)
- Chunk size: 2024
- Predictions: 7,977
- Device: CPU (GPU OOM during test)
- Checkpoint: `soc_epoch0005_rmse0.01393.pt`
- Scaler: `scaler_robust.joblib`

## Metrics
- RMSE: 0.0103362832
- MAE: 0.0091929678

## Artifacts
- `soc_pred_first_10000_rows_MGFarm_18650_C07.csv`
- `soc_pred_first_10000_rows_MGFarm_18650_C07.png`
- `test_summary.json`
