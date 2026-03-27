Benchmarking the flashed SOC base model on an STM32

Overview
- This host script connects to the flashed STM32 SOC firmware via serial (UART) and performs a brief benchmark based on realistic synthetic inputs.
- It collects per-sample round-trip latencies (host send -> device SOC reply) and parses periodic `METRICS:` lines emitted by the firmware (cycles, us, energy in uJ).
- Outputs: CSV results, latency histogram and time-series PNGs, and a short summary JSON.

Quick usage (Windows cmd):

```cmd
conda activate ml1
pip install -r STM32\bench_host\requirements.txt
python STM32\bench_host\benchmark_stm32_soc.py --port COM3 --baud 115200 --samples 5000
```

Notes
- The script does not modify the device firmware and only opens the serial port.
- Energy estimates come from the device `METRICS:` lines (firmware computes an approximate microjoule value). For accurate electrical energy measurements use an external power meter or scope while the script toggles the GPIO (firmware toggles PE1 on each sample).
- If your device prints a `BOOT` line at startup, the script will wait for it before starting.
