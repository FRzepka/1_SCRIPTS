| class                     | focus_scenario     | local_metric                            |    value |   threshold |
|:--------------------------|:-------------------|:----------------------------------------|---------:|------------:|
| Direct measurement        | initial_soc_error  | recovery_time_to_baseline_band_strict_h |   1.1658 |      0.08   |
| Direct measurement        | initial_soc_error  | recovery_time_to_baseline_band_fair_h   |   0.25   |      0.1599 |
| Hybrid direct measurement | initial_soc_error  | recovery_time_to_baseline_band_strict_h | nan      |      0.0094 |
| Hybrid direct measurement | initial_soc_error  | recovery_time_to_baseline_band_fair_h   | nan      |      0.02   |
| Hybrid ECM                | initial_soc_error  | recovery_time_to_baseline_band_strict_h |   2.3693 |      0.0241 |
| Hybrid ECM                | initial_soc_error  | recovery_time_to_baseline_band_fair_h   |   1.1564 |      0.0482 |
| Data-driven               | initial_soc_error  | recovery_time_to_baseline_band_strict_h |   1.0464 |      0.0336 |
| Data-driven               | initial_soc_error  | recovery_time_to_baseline_band_fair_h   |   1.0014 |      0.0672 |
| Direct measurement        | missing_gap        | recovery_time_h                         |  27.3511 |      0.0216 |
| Hybrid direct measurement | missing_gap        | recovery_time_h                         |  27.4093 |      0.0008 |
| Hybrid ECM                | missing_gap        | recovery_time_h                         |  28.5611 |      0.0095 |
| Data-driven               | missing_gap        | recovery_time_h                         |  27.4351 |      0.007  |
| Direct measurement        | spikes             | median_spike_recovery_time_s            |   0      |      0.08   |
| Hybrid direct measurement | spikes             | median_spike_recovery_time_s            |   0      |      0.0094 |
| Hybrid ECM                | spikes             | median_spike_recovery_time_s            |   0      |      0.0241 |
| Data-driven               | spikes             | median_spike_recovery_time_s            |   0      |      0.0336 |
| Direct measurement        | current_noise_high | late_minus_early_excess_rolling_mae     |  -0.0013 |    nan      |
| Direct measurement        | current_noise_high | mean_excess_rolling_mae_high_noise      |  -0.0001 |    nan      |
| Hybrid direct measurement | current_noise_high | late_minus_early_excess_rolling_mae     |  -0.0011 |    nan      |
| Hybrid direct measurement | current_noise_high | mean_excess_rolling_mae_high_noise      |   0.0019 |    nan      |
| Hybrid ECM                | current_noise_high | late_minus_early_excess_rolling_mae     |   0.0038 |    nan      |
| Hybrid ECM                | current_noise_high | mean_excess_rolling_mae_high_noise      |   0.0073 |    nan      |
| Data-driven               | current_noise_high | late_minus_early_excess_rolling_mae     |  -0.0002 |    nan      |
| Data-driven               | current_noise_high | mean_excess_rolling_mae_high_noise      |   0.0009 |    nan      |