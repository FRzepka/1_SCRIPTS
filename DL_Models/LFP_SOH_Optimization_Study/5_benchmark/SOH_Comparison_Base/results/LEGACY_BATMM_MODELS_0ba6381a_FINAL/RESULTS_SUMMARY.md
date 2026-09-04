# Legacy BATMM Model Evaluation

This evaluation uses the unmodified checkpoints and scalers from BATMM commit
`0ba6381a9b11545ae98b763edee051a6066d0d1d`. The shared comparison cell is C11.
It is a held-out test cell for both the legacy and current model generations.

## C11 MAE

| Model | Version | Independent 168 h windows | Continuous context | Smoothed continuous display |
|---|---|---:|---:|---:|
| CNN | 0.4.1.1 | 0.111977 | 0.108395 | 0.103767 |
| GRU | 0.3.1.1 | 0.025580 | 0.022415 | 0.022413 |
| LSTM | 0.1.2.3 | 0.024809 | 0.020873 | 0.020861 |
| TCN | 0.2.2.1 | 0.042598 | 0.026485 | 0.025672 |

The smoothed column uses a centered seven-hour rolling median. It is included
only to quantify the displayed curve and is not a causal estimator result.
Keeping every third point for plotting does not change any reported metric.

## Legacy Test-Cell Macro MAE

The native legacy split contains C11, C19, and C23.

| Model | Independent 168 h windows | Continuous context |
|---|---:|---:|
| CNN | 0.123032 | 0.116589 |
| GRU | 0.038953 | 0.040713 |
| LSTM | 0.048576 | 0.051184 |
| TCN | 0.058471 | 0.037779 |

The continuous-context macro results reproduce the benchmark report included
in the cloned repository to the displayed precision. This validates the data
aggregation, checkpoint loading, scaler application, and inference logic.

## Comparison With the Reported Values

The reported MAE values of CNN 0.0175, GRU 0.0168, LSTM 0.0188, and TCN 0.0182
are not produced by these legacy checkpoints under either tested inference
mode. The difference is especially conclusive for CNN. Those values must use a
different model generation, evaluation set, metric aggregation, or inference
protocol.

## Repository Finding

The adjacent LSTM YAML does not describe the included checkpoint. The YAML
specifies a smaller two-layer model, whereas the checkpoint metadata specifies
the actual three-layer LSTM with hidden size 192. The evaluation therefore uses
the training configuration embedded in each checkpoint. The other three YAML
model configurations agree with their checkpoints.
