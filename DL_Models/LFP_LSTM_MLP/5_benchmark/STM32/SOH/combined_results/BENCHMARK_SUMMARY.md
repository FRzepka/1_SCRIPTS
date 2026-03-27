# STM32 SOH Model Benchmark

Date: 2025-11-26 18:43:55

| Model     |   Latency_us |   Energy_uJ |   Static_RAM_Bytes |   Stack_RAM_Bytes |   Total_RAM_Bytes | Flash_Usage_Bytes   |
|:----------|-------------:|------------:|-------------------:|------------------:|------------------:|:--------------------|
| Base      |      5032.19 |     2516.09 |               8308 |               592 |              8900 | N/A                 |
| Pruned    |      2796.55 |     1398.27 |               8308 |               592 |              8900 | N/A                 |
| Quantized |      5033.67 |     2516.83 |               8308 |               592 |              8900 | N/A                 |

*RAM measured on-device (Static + Peak Stack). Latency averaged over 50 runs.*