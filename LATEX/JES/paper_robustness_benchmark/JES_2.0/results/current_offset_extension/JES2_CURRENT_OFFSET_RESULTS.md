# JES2 additive current-offset results

Generated from the completed six-cell, 16-window JES2 extension. The measured current is
changed additively by -50 mA or +50 mA. This is distinct from the multiplicative
current-gain error already present in the benchmark. All four estimators use the same
sign-matched causal SOH trace, common sample mask, and matched nominal baseline.

## Cell-macro result

| Model | Delta MAE at -50 mA | Delta MAE at +50 mA | Adverse Delta MAE [95% CI] | Adverse sign by cell (+/-) |
|---|---:|---:|---:|---:|
| DM | 0.19335 | 0.23245 | 0.24675 [0.22526, 0.26926] | 5/1 |
| HDM | 0.25848 | 0.24725 | 0.27273 [0.25464, 0.29107] | 3/3 |
| HECM | 0.17611 | 0.07355 | 0.17632 [0.13287, 0.21419] | 1/5 |
| DD | 0.02363 | 0.04009 | 0.04009 [0.02262, 0.05558] | 6/0 |

## Interpretation boundary

The smallest adverse additive-offset penalty is observed for DD, while HDM has the largest.
The result represents one matched signed stress level. It does not claim complete coverage of
all current-sensor offset magnitudes, time variation, calibration drift, or combined sensor errors.
The manuscript PDF was intentionally not rebuilt from these results.
