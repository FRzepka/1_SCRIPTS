# Robustness Benchmark

This benchmark manipulates only measured signals and timing. It does not inject artificial model-parameter mismatch and it does not require new datasets.

## Robustness definition

A SOC estimator is treated as robust if, under realistic sensor and signal faults, it still:

- keeps the estimate numerically stable
- avoids large outliers and jumps
- limits long-term drift
- recovers after the disturbance
- preserves low error increase relative to baseline

## Scenario groups

### A. Input disturbances

- `current_offset`: constant current sensor bias
- `voltage_offset`: constant voltage sensor bias
- `temp_offset`: constant temperature sensor bias
- `current_noise`: additive Gaussian current noise
- `voltage_noise`: additive Gaussian voltage noise
- `temp_noise`: additive Gaussian temperature noise
- `adc_quantization`: finite ADC resolution on current, voltage, temperature
- `spikes`: sparse outliers on a selected input channel

### B. Initialization errors

- `initial_soc_error`: additive SOC initialization error for models that explicitly use a SOC state

Notes:
- For data-driven SOC models without explicit SOC state input, this scenario is reported but may be structurally non-applicable.

### C. Signal integrity problems

- `missing_samples`: isolated frozen samples, either periodic or random
- `irregular_sampling`: variable sample spacing by jittering `Testtime[s]`
- `missing_gap`: one central burst dropout with frozen prediction and zero effective integration time

Legacy compatibility scenarios kept in the runners:

- `temp_mask`
- `downsample`
- `missing_segments`

## Metrics

Every run now stores at least:

- `mae`
- `rmse`
- `bias`
- `max_error`
- `p95_error`
- `jump_count_gt_5pct`
- `output_variance`
- `abs_error_variance`
- `drift_rate_soc_per_h`
- `drift_rate_abs_err_per_h`

If a disturbance mask exists, the run also stores:

- `disturbed_mae`
- `disturbed_rmse`
- `disturbed_max_error`
- `pre_disturbance_mae`
- `post_disturbance_mae`
- `post_disturbance_rmse`
- `recovery_threshold_abs_err`
- `recovery_time_s`
- `recovery_time_h`
- `residual_error_after_recovery`

## Benchmark philosophy

The comparison is between estimation philosophies under realistic measurement corruption:

- `CC_1.0.0`: integration-based
- `CC_SOH_1.0.0`: integration-based with learned SOH support
- `ECM_0.0.3`: physics-based / hybrid
- `SOC_SOH_1.6.0.0_0.1.2.3`: data-driven

## Current-noise interpretation

The current-noise benchmark needs two separate readings:

- global penalty: `Delta MAE = MAE_noise - MAE_baseline`
- local behaviour: short-term output spikes and jitter under the same noise level

The current-noise injection itself is simple:

- `I_noisy(t) = I(t) + n(t)`
- `n(t) ~ N(0, sigma_I^2)`

Important observations from the current benchmark:

- `DM` and `HDM` stay visually smooth under current noise because they integrate current directly
  - per-step SOC perturbation is small: `Delta SOC ~= I_noise * dt / (3600 * C_Ah)`
  - with `sigma_I = 0.20 A`, `dt = 1 s`, `C = 1.8 Ah`, this is only about `3.1e-5` SOC per sample
- `DD` becomes visibly noisy because it does not only see `Current[A]`
  - it also uses `Q_c`, `dI_dt[A/s]`, and `dU_dt[V/s]`
  - the derivative channel amplifies current noise strongly
- `HECM` is sensitive for a different reason
  - it is not a CC-type integrator
  - the EKF uses current directly in the measurement/state update and switches parameter tables by current sign
  - near low-current phases, current noise can flip the sign and therefore switch charge/discharge parameter sets

### Why `DD` looks much noisier than `DM` / `HDM`

The `DD` model uses the SOC feature set:

- `Voltage[V]`
- `Current[A]`
- `Temperature[°C]`
- `SOH`
- `Q_c`
- `dU_dt[V/s]`
- `dI_dt[A/s]`

This matters because `dI_dt` is computed from the already corrupted current signal. In the strong-noise run (`sigma_I = 0.20 A`), the measured amplification is approximately:

- `std(Current)` baseline -> noise: `1.511 -> 1.524`
- `std(dI/dt)` baseline -> noise: `0.0716 -> 0.2923`

So the raw current changes only moderately, but the derivative feature becomes about four times noisier.

### Why `HECM` is noise-sensitive

`HECM` uses weekly cached GRU-SOH, but this is not the dominant reason for the current-noise sensitivity. The more important mechanisms are:

- current-sign dependent parameter selection
  - charge and discharge tables are switched by `current >= 0`
- direct current term in the EKF measurement equation
  - `y_p = ... + Ri * current`
- current-dependent interpolation of ECM parameters

Measured sign-flip rates under `sigma_I = 0.20 A`:

- full run: about `2.04 %`
- when `|I| < 0.2 A`: about `9.51 %`
- when `|I| >= 0.2 A`: about `0.078 %`

This explains why `HECM` becomes especially fragile in low-current, CV, and transition phases.

### Suggested local current-noise metrics

The following local metrics are useful and should not be conflated:

- error spikes:
  - `p95(|Delta e(t)|)`, with `Delta e(t) = |y_noise(t)-y_true(t)| - |y_base(t)-y_true(t)|`
  - this measures strong additional local error peaks caused by noise
- output jitter:
  - `p95(|Delta y_hat(t)|)`, with `Delta y_hat(t) = y_hat(t) - y_hat(t-1)`
  - this measures how noisy the output trajectory itself becomes

For the current setup, the exploratory output-jitter probe shows:

- `DM`: `p95(|Delta y_hat|)` at `0.20 A` is about `0.000315`
- `HDM`: about `0.000342`
- `HECM`: about `0.000340`
- `DD`: about `0.049635`

This is useful because it captures exactly the visual impression that the `DD` trajectory becomes much more jagged, even when its global MAE penalty remains moderate.

## Overview builder

Use:

```bash
python DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/build_robustness_benchmark_overview.py
```

It writes:

- `benchmark_overview_latest.csv`
- `benchmark_overview_latest.md`
- `benchmark_overview_short.csv`

## Campaign runners

- compact matrix: `run_minimal_robustness_matrix.py`
- extended matrix: `run_extended_robustness_matrix.py`

The extended matrix adds:

- `voltage_noise`
- `temp_noise`
- `voltage_offset`
- `temp_offset`
- `adc_quantization`
- `irregular_sampling`
- `missing_gap`
- stronger `spikes_high`
- stronger `current_noise_high`
