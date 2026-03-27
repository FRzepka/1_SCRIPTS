# STM32 SOC and SOH Benchmark Summary (10 000 samples)

Date: 2025-11-28 (full-stream replay, 10 000 inferences per model)

All values are averages over the full 10 000-sample benchmark.  
Latency is measured on the host (end-to-end UART round-trip), inference time is measured on-device using the DWT
cycle counter. RAM corresponds to static data plus peak stack usage observed during the run.

## SOC models on STM32H753ZI

| Model        | Latency [ms] | Inference [ms] | Flash [KB] | Static RAM [KB] | Stack RAM [KB] | Total RAM [KB] | Energy [µJ] |
|:------------|-------------:|---------------:|-----------:|----------------:|---------------:|---------------:|------------:|
| Base FP32   |        12.70 |           1.40 |     105.32 |            4.36 |           0.57 |           4.93 |     700.83  |
| Pruned FP32 |        12.20 |           0.80 |      62.27 |            3.46 |           0.56 |           4.03 |     400.38  |
| Quant INT8  |        18.48 |           6.99 |      52.48 |            0.46 |           3.50 |           3.96 |   3 494.66  |

## SOH models on STM32H753ZI

| Model        | Latency [ms] | Inference [ms] | Flash [KB] | Static RAM [KB] | Stack RAM [KB] | Total RAM [KB] | Energy [µJ] |
|:------------|-------------:|---------------:|-----------:|----------------:|---------------:|---------------:|------------:|
| Base FP32   |        34.88 |          22.73 |     335.00 |            6.96 |           1.73 |           8.69 |  11 366.93  |
| Pruned FP32 |        24.69 |          12.72 |     182.41 |            5.18 |           1.78 |           6.96 |   6 362.23  |
| Quant INT8  |        41.23 |          29.21 |     138.00 |            0.46 |           6.24 |           6.70 |  14 604.27  |

Notes:

- Flash sizes are taken from the final firmware binaries (map files) and converted to kilobytes using 1 KB = 1024 bytes.
- Static RAM is the sum of `.data` and `.bss` segments; stack RAM is the peak stack usage reported by the STM32
  benchmark firmware; total RAM is Static + peak Stack.
- Energy is approximated from the measured inference time assuming a constant device power of 0.5 W:
  \(E \approx t_{\text{inf}} \cdot 0.5\,\text{W}\), reported in microjoules.
