# JES2 DM firmware

Initial STM32H753ZI firmware for the isolated JES2 SOC hardware benchmark.

- Implements `JES2_HW_V1` over USART3/ST-Link VCP at 115200 baud.
- Runs the software-equivalent direct-measurement Coulomb counter.
- Measures estimator execution with the Cortex-M7 DWT cycle counter.
- Green LD1 blinks at 2 Hz while idle and rapidly while samples are processed.
- Red LD3 signals malformed UART commands or a fatal firmware error.
- Orange LD2 is used only in the startup sequence.

Build and flash:

```powershell
./build.ps1 -Configuration Release
./flash.ps1 -Configuration Release
```

