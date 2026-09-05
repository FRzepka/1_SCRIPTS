# Key local results

All errors below are percentage points of the normalized full-scale target unless
stated otherwise.

## Long-horizon streaming outputs

| Task | Base MAE | Pruned MAE | Quantized MAE |
|---|---:|---:|---:|
| SOC | 2.6845 | 2.3380 | 2.7912 |
| SOH | 0.8523 | 1.4572 | 1.4103 |

The SOC error increases for all three models toward later sequence segments. Base
MAE changes from 1.116 in the first tenth to 4.538 in the final tenth; Quantized
changes similarly from 1.234 to 4.600. Its mean absolute deviation from Base is
0.332 in the first tenth and 0.271 in the final tenth, so the saved output does not
show a quantization-specific monotonic drift. Pruned-to-Base deviation grows from
0.924 to 2.592, even though Pruned has the lowest target MAE.

SOH window errors fluctuate strongly with the ageing phase. Neither compressed
variant shows a monotonic deviation from Base across all ten windows. This supports
a bounded statement about the saved streaming replay, not universal recurrent-state
stability.

## SOH filtering

The exported Base, Pruned, and Quantized C models were re-executed on the same
14,496,990-sample C07 sequence in one local Windows environment. The resulting
comparison is:

| Model | Raw MAE | Code filter MAE | Manuscript-text filter MAE |
|---|---:|---:|---:|
| Base | 1.8496 | 1.6771 | 1.4435 |
| Pruned | 1.9847 | 1.6436 | 1.6842 |
| Quantized | 2.1064 | 1.9259 | 1.0422 |

The code filter (`alpha=0.02`, symmetric limiter) changes all three model outputs
and reduces their local MAE by 0.173, 0.341, and 0.180 percentage points,
respectively. Its limiter activates on more than 99.8% of samples for every model,
so filtering is not a negligible final cosmetic step.

The apparently lower error of the manuscript filter comes with extremely slow
dynamics: at 1 Hz, `alpha=1e-6` reaches 90% of a step after about 26.65 days. The
saved-code value `alpha=0.02` reaches 90% after about 114 seconds. Accuracy alone is
therefore not enough to choose between them; the actual submitted pipeline must be
identified and its delay discussed. The local Windows values are a separate
re-execution and must not replace archived Linux benchmark values without explicit
labelling.

## Utility sensitivity

Across all 1,771 non-negative 5%-spaced combinations of accuracy, flash, RAM, and
the existing energy estimate:

| Task | Base wins | Pruned wins | Quantized wins |
|---|---:|---:|---:|
| SOC | 0.00% | 94.64% | 5.36% |
| SOH | 28.85% | 53.36% | 17.79% |

For SOH, Base becomes preferable when accuracy receives sufficiently high weight;
Quantized becomes preferable when flash dominates. Pruned is the most frequent but
not universal winner. SOC Pruned is robust over most tested priority combinations.

## Mixed-precision quantization

| Task/matrix | Relative reconstruction RMSE | P95 absolute weight error |
|---|---:|---:|
| SOC input-to-hidden | 0.452% | 0.00551 |
| SOC hidden-to-hidden | 0.542% | 0.00114 |
| SOH input-to-hidden | 0.381% | 0.00868 |
| SOH hidden-to-hidden | 0.653% | 0.01049 |

Exported model storage falls from 87.5 to 37.0 KiB for SOC and from 335.0 to
138.0 KiB for SOH. The persistent FP32 hidden and cell states remain 512 bytes for
SOC and 1,024 bytes for SOH. These values describe exported tensors, not the full
linked firmware footprint.

## Static operation accounting

SOC Base and Quantized both execute 22,080 model MACs, but the current quantized
kernel adds 17,920 per-weight FP32 scale multiplications. SOH adds 68,608 such scale
multiplications to 85,120 MACs. This is consistent with the observed runtime penalty
but is not a substitute for cycle-level or memory-bandwidth profiling.

## Limited software fault checks

Holding the previous output for random missing updates changes aggregate MAE only
minimally because adjacent predictions are highly correlated. This does not test
missing input samples or state evolution under interrupted inference.

Random one-bit corruption of the FP32 output frequently causes catastrophic errors.
A physical-range check plus hold-last rejects many exponent/sign corruptions, but
in-range corruptions remain; for arbitrary bit locations the mitigated P95 error is
still roughly 72--73% for SOC and about 94% for SOH. A range check is therefore only
a basic guard, not a complete safety mechanism.
