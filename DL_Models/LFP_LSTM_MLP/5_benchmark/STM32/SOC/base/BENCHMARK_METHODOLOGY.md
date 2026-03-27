# STM32 SOC Benchmark Methodology

## Overview
This benchmark evaluates the performance of the LSTM-based SOC estimator running on the STM32H753 microcontroller. The benchmark is designed to be non-invasive, using the existing UART communication interface to trigger inferences and collect metrics.

## Experimental Setup
- **Device**: STM32H753ZI (Nucleo-144)
- **Clock Speed**: 480 MHz (Core), 240 MHz (Bus)
- **Communication**: UART over USB (ST-Link Virtual COM Port) at 115200 baud.
- **Firmware**: X-CUBE-AI generated C-model (INT8/FP32 hybrid or FP32 base).

## Metrics Collected

### 1. Host-Side Latency (Round-Trip Time)
- **Definition**: Time elapsed between sending a sample (6 float features) and receiving the SOC prediction.
- **Measurement**: Python `time.perf_counter()` on the host PC.
- **Components**: Serial transmission time + Device parsing time + **Inference Time** + Serial response time.
- **Relevance**: Represents the real-world responsiveness of the system when queried by a master controller.

### 2. Device-Side Inference Time
- **Definition**: Pure computation time for the neural network inference (excluding UART overhead).
- **Measurement**: On-chip DWT Cycle Counter (Data Watchpoint and Trace unit).
- **Precision**: Single clock cycle precision (1 cycle ≈ 2.08 ns @ 480 MHz).
- **Reporting**: The firmware aggregates and reports this periodically via `METRICS:` lines.

### 3. Energy Estimation
- **Definition**: Estimated energy consumption per inference.
- **Calculation**: $E = P_{busy} \times t_{inference}$
- **Assumptions**: $P_{busy} \approx 350 \text{ mW}$ (Active run mode), $P_{idle} \approx 250 \text{ mW}$.
- **Note**: This is a software estimate. For precise electrical validation, the firmware toggles pin **PE1** (Blue LED) during inference, allowing for oscilloscope-based power measurement.

### 4. Memory Footprint
- **Flash (ROM)**: Extracted from the `.elf` file size or `.text` section in the map file. Represents code size + constant weights.
- **RAM**: Extracted from `.data` + `.bss` sections. Represents static variables and buffers. Dynamic heap usage (activations) is managed by the X-CUBE-AI runtime.

## Benchmark Procedure
1. **Connection**: Host connects to the STM32 via USB Serial (`COM7`).
2. **Warmup**: A few initial samples are sent to stabilize the connection and caches.
3. **Execution**: The host sends `N=5000` synthetic samples.
   - Samples are generated using the same RobustScaler distribution as the training data to ensure realistic activation patterns.
4. **Data Collection**:
   - Each response is timestamped.
   - Periodic `METRICS` packets are parsed to retrieve internal device stats.
5. **Analysis**:
   - Outliers are filtered (though rare in wired serial).
   - Mean, Median, and P95 latency are computed.
   - Throughput (inferences per second) is calculated.

## How to Run
Ensure the STM32 is flashed and connected to `COM7`.

```bash
python STM32/benchmark/SOC_base/benchmark_stm32_soc.py --samples 5000
```

## Output
The script generates:
- `stm32_soc_bench.csv`: Raw data for every sample.
- `paper_summary.md`: A Markdown table ready for copy-pasting into the paper.
- `latency_boxplot.png`: Distribution of response times.
- `device_metrics.png`: Internal timing and energy stability.
