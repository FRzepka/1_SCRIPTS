# Decision Diagram Notes (v4)

These diagrams are exploratory decision aids built from the existing `paper_tables_v4` results.
They do not replace the raw benchmark results and are not meant as a universal ranking.

## Inputs

- Baseline table: `table_baseline.csv`
- Disturbed-scenario table: `table_key_results.csv`
- Local behaviour table: `table_local_behaviour.csv`

## Meta-scores

All scores are normalized to `[0, 1]` across the four estimator classes, with higher being better.

### Accuracy

Average of min-max normalized baseline `MAE`, `RMSE`, and `P95`.

### Robustness

Average of min-max normalized disturbed-scenario `delta_MAE` over:
- Current noise (high)
- Current bias
- Irregular sampling
- Burst dropout
- Missing samples
- Voltage spikes
- Temperature noise
- Voltage noise

### Recovery

Average of penalized lower-is-better scores for:
- `recovery_time_to_baseline_band_strict_h`
- `recovery_time_to_baseline_band_fair_h`

Missing recovery times are treated as non-recovery and mapped to a zero score using a penalty larger than the slowest finite recovery time.

## Decision profiles

These are not presented as a single universal truth. They are only alternative weighting views:
- `Accuracy-first`: Accuracy=0.60, Robustness=0.25, Recovery=0.15
- `Robustness-first`: Accuracy=0.20, Robustness=0.65, Recovery=0.15
- `Recovery-first`: Accuracy=0.20, Robustness=0.20, Recovery=0.60

## Final normalized scores

| Model   | Class                     |   Accuracy |   Robustness |   Recovery |
|:--------|:--------------------------|-----------:|-------------:|-----------:|
| DM      | Direct measurement        |     0      |       0.654  |     0.9688 |
| HDM     | Hybrid direct measurement |     1      |       0.2049 |     0      |
| HECM    | Hybrid ECM                |     0.788  |       0.7303 |     0.2755 |
| DD      | Data-driven               |     0.7155 |       0.3163 |     0.6857 |

## Robustness raw inputs (`delta_MAE`)

| Class                     |   Burst dropout |   Current bias |   Current noise (high) |   Irregular sampling |   Missing samples |   Temperature noise |   Voltage noise |   Voltage spikes |
|:--------------------------|----------------:|---------------:|-----------------------:|---------------------:|------------------:|--------------------:|----------------:|-----------------:|
| Direct measurement        |        0.002159 |       0.283259 |              -6.8e-05  |            -0.000136 |          0.007558 |            0        |        0.021319 |         0.000103 |
| Hybrid direct measurement |        0.002664 |       0.304326 |               0.001943 |             0.000289 |          0.007867 |            0.004276 |        0.072701 |         0.003769 |
| Hybrid ECM                |        0.001411 |       0.084199 |               0.007472 |             0.000443 |          0.002878 |            0        |       -5e-06    |         0        |
| Data-driven               |        0.00223  |       0.275867 |               0.000862 |             0.000468 |          0.001641 |            0.008061 |        0.059741 |         0.004379 |

## Recovery raw inputs

| Class                     |   recovery_time_to_baseline_band_fair_h |   recovery_time_to_baseline_band_strict_h |
|:--------------------------|----------------------------------------:|------------------------------------------:|
| Direct measurement        |                                 0.25    |                                   1.16583 |
| Hybrid direct measurement |                               nan       |                                 nan       |
| Hybrid ECM                |                                 1.15639 |                                   2.36928 |
| Data-driven               |                                 1.00139 |                                   1.04639 |