# SOH JES2 0.1.0

Frozen SOH model for the JES 2.0 benchmark. It uses exactly the same cell
exposure as the SOC 1.7.x models.

- Train: C01, C03, C05, C11, C17, C23
- Validation: C07, C19, C21
- Holdout: C09, C13, C15, C25, C27, C29
- Architecture: LSTM hybrid seq2seq, embed 128, hidden 112, two layers,
  three residual MLP blocks
- Parameters: 382,353
- Scaler fit: 22,986 hourly samples from training cells only
- Best checkpoint: epoch 21 of 33 (early stopping)

Validation metrics:

| RMSE | MAE | R2 |
|---:|---:|---:|
| 0.021920 | 0.015769 | 0.729124 |

Unweighted mean across the six holdout cells:

| RMSE | MAE | R2 |
|---:|---:|---:|
| 0.031557 | 0.024945 | 0.887467 |

Pooled across all 13,038 holdout samples, RMSE is 0.032326 and MAE is
0.021792.

The holdout was evaluated once after model selection. Do not use these cells
for further checkpoint selection, retraining, pruning, or benchmark tuning.
Exact provenance and hashes are recorded in `training_meta.json`; per-cell
results are in `holdout_metrics.csv`.
