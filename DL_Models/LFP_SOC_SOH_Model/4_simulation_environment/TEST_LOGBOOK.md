# Test Logbook

This logbook tracks what has been executed in the simulation environment, what the current benchmark methodology is, and which gaps still need to be closed before the paper-level robustness study is complete.

## Scope

- Root: `DL_Models/LFP_SOC_SOH_Model/4_simulation_environment`
- Current benchmark focus: realistic robustness tests based on **measurement manipulation only**
- Cell used for current benchmark campaign: `MGFarm_18650_C07`

## Benchmark Principle

The robustness benchmark currently follows these constraints:

- only manipulate measured inputs (`Current[A]`, `Voltage[V]`, `Temperature[°C]`, sampling/time)
- no artificial change of model parameters during testing
- keep online feature generation realistic
- if measurements are frozen/missing, derived online quantities must also freeze accordingly

## Active Model Set

- `CC_1.0.0`
- `CC_SOH_1.0.0`
- `ECM_0.0.1`
- `SOC_SOH_1.6.0.0_0.1.2.3`

## Campaigns

### `2026-03-12_minimal_matrix_fullc07`

Purpose:

- first full-cell C07 execution of the compact robustness matrix

Planned scenarios:

1. `baseline`
2. `current_noise`
3. `current_offset`
4. `initial_soc_error`
5. `missing_samples`
6. `spikes`

Status snapshot:

- `CC_1.0.0`: baseline, current_noise, current_offset, initial_soc_error, missing_samples completed
- `CC_SOH_1.0.0`: baseline, current_noise, current_offset completed
- `ECM_0.0.1`: baseline, current_noise, current_offset completed
- `SOC_SOH_1.6.0.0_0.1.2.3`: long-running full-cell runs still in progress / incomplete in first analysis pass

Analysis output:

- `campaigns/2026-03-12_minimal_matrix_fullc07/analysis/minimal_matrix_summary.csv`
- `campaigns/2026-03-12_minimal_matrix_fullc07/analysis/minimal_matrix_summary.md`
- `campaigns/2026-03-12_minimal_matrix_fullc07/analysis/01_mae_grouped.png`
- `campaigns/2026-03-12_minimal_matrix_fullc07/analysis/02_delta_mae_grouped.png`
- `campaigns/2026-03-12_minimal_matrix_fullc07/analysis/03_p95_heatmap.png`
- `campaigns/2026-03-12_minimal_matrix_fullc07/analysis/FINDINGS.md`

## What Has Been Learned So Far

### Confirmed

- `CC_SOH_1.0.0` currently has the best baseline on C07.
- `current_offset` clearly exposes the weakness of Coulomb Counting.
- `ECM_0.0.1` is substantially more robust than plain CC under current bias.
- the shared disturbance infrastructure in `robustness_common.py` is working and already supports a larger benchmark space than the current minimal matrix uses.

### Initial SOC Error Status

- `initial_soc_error` in the first minimal-matrix run was not meaningful for the current setup.
- reason: the compact campaign used `+10%`, but the current CC/ECM initialization baseline is already at a saturated upper bound (`soc_init=1.0`), so a positive error can clip away and fail to stress the model.
- this was corrected in the campaign runner:
  - use `-10%` instead of `+10%`
  - use `--warmup_seconds 0` so the effect is not masked
- `SOC_SOH_1.6.0.0_0.1.2.3` now also has an `initial_soc_error` variant for local analysis:
  - implementation: apply the initial offset to the online `Q_c` feature before rolling SOC inference
  - run: `SOC_SOH_1.6.0.0_0.1.2.3/runs/initial_soc_error/2026-03-12_1110_initial_soc_error_proxy_m10_fix`

### Still Missing for the Paper Matrix

- extended scenarios:
  - `voltage_noise`
  - `temp_noise`
  - `voltage_offset`
  - `temp_offset`
  - `adc_quantization`
- a refined `initial_soc_error` protocol for `SOC_SOH` if a true initial-state perturbation is desired there

## Important Methodological Notes

### Initial SOC Error

For this dataset and current online setup:

- a **positive** initial error is not a good stress test because initialization can saturate
- for C07, a **negative** initial error is currently the better choice to expose recovery behavior

This is not a cosmetic change. It is necessary to make the test identifiable on this cell.

## Follow-up Local Analysis

### `2026-03-12_local_recovery`

Purpose:

- move beyond global MAE/RMSE and inspect local behavior for:
  - `initial_soc_error`
  - `spikes`
  - `current_noise`

Artifacts:

- `analysis_local_focus/2026-03-12_local_recovery/local_metrics.csv`
- `analysis_local_focus/2026-03-12_local_recovery/local_metrics.md`
- `analysis_local_focus/2026-03-12_local_recovery/01_initial_soc_error_local_recovery.png`
- `analysis_local_focus/2026-03-12_local_recovery/02_spikes_local_recovery.png`
- `analysis_local_focus/2026-03-12_local_recovery/03_current_noise_local_trend.png`

Additional stronger-noise runs:

- `current_noise_std = 0.10 A`
- stored in each model's `runs/current_noise/2026-03-12_0940_current_noise_high_0p10`

Main observations:

- `initial_soc_error` is now evaluated in a CV-free window, so recovery is not attributed to an early CV reset.
- `initial_soc_error` now recovers locally for:
  - `CC_1.0.0` after about `1.166 h` under the strict band and `0.250 h` under the fair band
  - `ECM_0.0.1` after about `1.157 h` under the strict band and `0.250 h` under the fair band
  - `SOC_SOH_1.6.0.0_0.1.2.3` after about `1.180 h` under the strict band and `0.562 h` under the fair band
- `CC_SOH_1.0.0` does not re-enter even the wider fair band in the inspected CV-free window, so this is currently treated as a genuine non-recovery result for this cell/perturbation rather than only a threshold artifact.
- stronger current noise (`0.10 A`) still does not collapse the models globally, but the rolling local error trend increases for all models; the strongest increase is currently seen for `ECM` and `CC`.
- stronger voltage spikes (`0.20 V`) were added to make the effect more visible:
  - `CC_1.0.0`: almost unchanged globally
  - `CC_SOH_1.0.0`: slightly better than baseline in global MAE due to noise-like cancellation, no sustained local failure
  - `ECM_0.0.1`: effectively unchanged globally
  - `SOC_SOH_1.6.0.0_0.1.2.3`: clear visible degradation (`MAE 0.01907 -> 0.02552`), but still no long-lasting spike recovery time under the current single-spike metric

### SOC_SOH Runtime

`SOC_SOH_1.6.0.0_0.1.2.3` full-cell runs are significantly slower than the other models because:

- hourly SOH must be inferred online
- SOC is computed via rolling sequence inference over the full trajectory

The current direct runs were started as background Python processes, not as `screen` sessions.

## Open Actions

1. regenerate the minimal-matrix campaign analysis if the new `SOC_SOH initial_soc_error` and strong-spike variants should be folded into the main table
2. decide whether `SOC_SOH initial_soc_error` should remain a `Q_c`-proxy test or be elevated to an official benchmark scenario
3. strengthen the spike protocol further if a non-zero local recovery time is needed:
   - clustered spikes
   - burst spikes
   - longer spike windows
4. start the extended robustness block:
   - voltage noise
   - temperature noise
   - voltage offset
   - temperature offset
   - ADC quantization

## Relevant Files

- benchmark definition: `ROBUSTNESS_BENCHMARK.md`
- shared disturbance code: `robustness_common.py`
- campaign runner: `run_minimal_robustness_matrix.py`
- campaign analysis: `analyze_minimal_robustness_matrix.py`
