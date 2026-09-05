# Three-cell Capacity Sensitivity Check

The 28 trained size variants were evaluated without retraining on C11, C23, and C29. Metrics were calculated per cell and then macro-averaged so that every test cell has equal weight.

## Figure 3 consistency

| Architecture | New MAE | Figure 3 MAE | New RMSE | Figure 3 RMSE | Match at 4 decimals |
|---|---:|---:|---:|---:|:---:|
| LSTM | 0.016663 | 0.016700 | 0.019937 | 0.019900 | yes |
| TCN | 0.015190 | 0.015200 | 0.019592 | 0.019600 | yes |
| GRU | 0.014587 | 0.014600 | 0.017973 | 0.018000 | yes |
| CNN | 0.017198 | 0.017200 | 0.022383 | 0.022400 | yes |

## Capacity result

| Architecture | Displayed Figure 3 point | Internal model ID | C11 MAE | Three-cell MAE | Strict local minimum | Lowest tested point | Lowest MAE |
|---|---|---|---:|---:|:---:|---|---:|
| CNN | base_h128 | base_h128 | 0.014958 | 0.017198 | yes | l3_h224 | 0.015463 |
| GRU | base_h160 | l1_h160 | 0.022258 | 0.014587 | no | s2_h96 | 0.013878 |
| LSTM | base_h192 | l1_h192 | 0.022995 | 0.016663 | no | s1_h160 | 0.014918 |
| TCN | base_h96 | base_h96 | 0.012554 | 0.015190 | no | l1_h112 | 0.013875 |

## Interpretation

The three-cell result confirms a non-monotonic relationship between parameter count and test MAE. It does not support the stronger claim that every baseline used in Figure 3 is a local test-MAE minimum. Only the selected CNN is a strict local minimum in this sweep.

The single-cell C11 figure should not be retained merely because it produces a cleaner minimum. The defensible interpretation is that the reference configurations were fixed through the model-development and validation pipeline before this test-set sweep. The three-cell result is a post-selection sensitivity analysis, not a second model-selection step. Since every variant represents one trained initialization, the curves must not be interpreted as a seed-averaged causal effect of model capacity.
