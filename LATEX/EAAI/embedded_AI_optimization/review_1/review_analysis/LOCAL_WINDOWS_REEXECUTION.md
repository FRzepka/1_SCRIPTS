# Local Windows SOH C-model re-execution

## Purpose

The original SOH benchmark archive contains filtered Base, Pruned, and Quantized
outputs, but no raw Quantized output. To evaluate how compression interacts with
the filter, all three exported C models were therefore executed again locally on
the same input sequence.

## Re-execution

- Dataset: complete C07 feature Parquet, 14,499,038 valid rows.
- Features: test time, voltage, current, temperature, EFC, and cumulative charge.
- Ordering: the original default pandas quicksort order by test time.
- Alignment: the first 2,048 recurrent warm-up samples were removed.
- Compared samples: 14,496,990 for every model.
- Models: exported Base, Pruned, and recurrent-weight Quantized SOH C models.
- Runtime: local Windows ARM64 DLLs compiled with Zig 0.15.2.
- Calibration: the same first-point scaling used by the saved benchmark pipeline.
- Post-processing: both the code filter (`alpha=0.02`, symmetric limiter) and the
  conflicting manuscript-text filter (`alpha=1e-6`, downward-only limiter) were
  applied to every raw model output.

No model was trained, no HPC job was run, and no STM32 firmware was executed. The
copied Quantized scaler header needed a build-only signature correction so that its
two-buffer call matches the existing model source and the Base/Pruned scaler API;
no scaler constants or arithmetic were changed.

## Validation and scope

For the first 10,000 aligned Quantized samples, applying the code filter to the
local raw output differs from the archived filtered output by only about `7.5e-6`
SOH MAE. Across the complete recurrent sequence, the difference grows to about
`0.019` SOH MAE. This indicates that the local Windows trajectory does not exactly
reproduce the historical Linux run over the full horizon, whether because of
floating-point, ordering, or other platform-sensitive recurrent-state effects.

Consequently, all raw-to-filter comparisons use Base, Pruned, and Quantized outputs
from the same local execution. These results should be labelled as a local C-model
re-execution and should not be mixed silently with the archived manuscript metrics.

## Main outputs

- `results/filter/soh_filter_compression_local_windows.csv`
- `results/filter/soh_filter_compression_local_trajectory.csv`
- `../figures/Review_1_Additional/review_soh_filter_compression_all_models_mae.png`
- `../figures/Review_1_Additional/review_soh_filter_local_raw_all_models.png`
- `../figures/Review_1_Additional/review_soh_filter_local_benchmark_all_models.png`
- `../figures/Review_1_Additional/review_soh_filter_quantized_raw_vs_filtered.png`
