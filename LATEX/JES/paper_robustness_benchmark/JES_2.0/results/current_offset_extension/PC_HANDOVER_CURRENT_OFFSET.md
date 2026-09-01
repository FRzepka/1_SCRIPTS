# JES2 current-offset extension: PC handover

## Frozen execution result

- Disturbance: additive measured-current offset, `I_disturbed = I_measured +/- 0.050 A`.
- This is separate from the existing multiplicative current-gain error.
- Coverage: DM, HDM, HECM, and DD on all six holdout cells and all 16 frozen windows.
- Main extension: 128/128 completed runs, zero failures.
- Lookup extension: 224/224 completed HECM runs, zero failures.
- Evaluation mask: source sample 2023 onward, 84,377 matched samples per 24-h run.
- Aggregation: windows averaged within cell, equal-weight six-cell macro, 10,000 cell-bootstrap draws.
- Adverse result: the larger signed delta MAE is selected separately within each cell.

| Model | Delta MAE at -50 mA | Delta MAE at +50 mA | Adverse delta MAE [95% CI] | Adverse total MAE |
|---|---:|---:|---:|---:|
| DM | 0.19335 | 0.23245 | 0.24675 [0.22526, 0.26926] | 0.31572 |
| HDM | 0.25848 | 0.24725 | 0.27273 [0.25464, 0.29107] | 0.31395 |
| HECM | 0.17611 | 0.07355 | 0.17632 [0.13287, 0.21419] | 0.21006 |
| DD | 0.02363 | 0.04009 | 0.04009 [0.02262, 0.05558] | 0.06593 |

DD has the smallest adverse additive-offset penalty. HDM has the largest. The offset sign is
not uniformly adverse across models or cells, which is why the paired signed result must be
retained rather than publishing only the more convenient direction.

## HECM lookup consequence

The largest lookup interaction for the additive current-offset case is `0.006116 MAE`. It
occurs for the -50 mA subcase under the +10% time-constant scaling. Its cell-bootstrap interval
includes zero. Several smaller positive-offset interactions have intervals that exclude zero.

This changes the interpretation of the earlier lookup result. The previous statement that all
measurement-disturbance interactions remain below `0.00192 MAE` is no longer valid after
adding current offset. For the extended set, the observed maximum is `0.006116 MAE`.
Absolute HECM offset penalties remain much larger than these lookup interactions.

## Required paper integration on the PC

1. Add `Current offset (+/-50 mA)` as one scenario definition with two matched sign subcases.
2. Increase the declared scenario count from 19 to 20.
3. Increase the traceable public execution build from 6,912 to 7,040 runs.
4. Use the visible taxonomy `Random sensor noise` and `Systematic sensor errors`.
5. Place current noise, voltage noise, and temperature noise under random sensor noise.
6. Place current gain, current offset, voltage offset, and temperature offset under systematic sensor errors.
7. Update the protocol table, test matrix, cross-scenario heatmap, numerical Appendix table, and reviewer response.
8. Recompute the illustrative robustness synthesis before updating Figure 15. Do not insert the new result into the old score without rerunning its family weighting.
9. Update Figure 25 or its table with the HECM current-offset lookup interaction. The previous `0.00192` global maximum must be replaced or explicitly limited to the earlier disturbance subset.
10. Keep the scope statement that the benchmark uses representative sensor-error mechanisms and does not cross every channel, gain, offset, drift, and combined-fault possibility.

## Files to use

- `JES2_CURRENT_OFFSET_RESULTS.md`: short model result.
- `jes2_current_offset_protocol.json`: frozen main protocol.
- `jes2_current_offset_runs.csv`: 128 run-level values and matched baselines.
- `jes2_current_offset_cells.csv`: window-averaged cell values.
- `jes2_current_offset_statistics.csv`: signed macro statistics.
- `jes2_current_offset_adverse_statistics.csv`: adverse-direction result used above.
- `hecm_lookup_current_offset_protocol.json`: lookup extension protocol.
- `hecm_lookup_current_offset_runs.csv`: 224 HECM run-level values.
- `hecm_lookup_current_offset_cells.csv`: HECM cell interactions.
- `hecm_lookup_current_offset_statistics.csv`: signed lookup interactions and intervals.

## PDF state

`JES_2.0/Robustness_Benchmark_Manuscript_JES2_Updated.pdf` was restored to the repository
version intentionally. No PDF was rebuilt after the current-offset extension. The protected
original `Robustness_Benchmark_Manuscript.pdf` was not changed.
