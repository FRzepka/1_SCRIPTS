# JES2 DD firmware

STM32H753ZI firmware for the isolated JES2 data-driven SOC hardware benchmark.

- Implements `JES2_HW_V1` over USART3/ST-Link VCP at 115200 baud.
- Runs the pruned GRU-MLP directly as handwritten float32 C code.
- Uses the supplied reference SOH as one of eight input features. No SOH model runs on the board.
- Recomputes each 2024-sample rolling window from a zero GRU state.
- Measures estimator execution with the Cortex-M7 DWT cycle counter.
- Uses no ONNX or X-CUBE-AI runtime.
- Green LD1 blinks at 2 Hz while idle and pulses rapidly while samples are processed.
- Red LD3 signals malformed UART commands or a fatal firmware error.

The first 2023 samples after reset return `WARMUP`, matching the JES2 software-reference contract.

Build and flash:

```powershell
./build.ps1 -Configuration Release
./flash.ps1 -Configuration Release
```
