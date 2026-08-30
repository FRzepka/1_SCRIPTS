# JES2 STM32 Serial Protocol

## Transport

- UART over ST-Link Virtual COM Port
- 115200 baud, 8 data bits, no parity, 1 stop bit
- UTF-8/ASCII text, one message per line, `\n` terminator
- Decimal separator is always `.`

## Startup

Firmware prints exactly one identification line:

```text
READY,JES2_HW_V1,DD,firmware_git_sha,480000000
```

Fields are protocol, active model, firmware revision and CPU clock in Hz.
The host may request the same line again with:

```text
HELLO
```

## Commands

Reset all estimator state and the DD circular input window:

```text
RESET
```

Response:

```text
ACK,RESET
```

Process one ordered nominal input sample:

```text
STEP,sample_id,voltage_v,current_a,temperature_c,soh,q_c_ah,dv_dt_v_s,di_dt_a_s,dt_s
```

The firmware selects the fields needed by its active estimator. No disturbed
measurement values are generated on the board.

Successful response:

```text
RESULT,sample_id,model,soc,cycles,OK
```

DD response before the 2024-sample window is full:

```text
RESULT,sample_id,DD,nan,0,WARMUP
```

Error response:

```text
ERROR,sample_id,error_code
```

Read the accumulated on-device runtime-memory profile:

```text
MEMORY
```

Response fields are `.data`, `.bss`, static RAM, peak heap, peak stack, and
their combined peak in bytes:

```text
MEMORY,model,data_bytes,bss_bytes,static_bytes,heap_peak_bytes,stack_peak_bytes,total_peak_bytes
```

The stack watermark starts in `Reset_Handler` and therefore covers startup,
protocol handling, and estimator execution. Heap growth is recorded in
`_sbrk()`. The `MEMORY` command does not reset either high-water mark.

## Timing boundary

The DWT cycle counter starts immediately before estimator execution and stops
immediately after the SOC output is available. UART parsing, input copying,
formatting and transmission are excluded. For DD, window management is reported
separately if possible; the primary `cycles` value must include all computation
required to produce one SOC from the filled 2024-sample window.

## Trigger pin

If energy is measured externally, set the selected GPIO high at the same point
as the DWT start and low at the DWT stop. Record board supply voltage, instrument,
sample rate, shunt and trigger pin in the result metadata.

## Ordering

The host sends only one outstanding `STEP`. Firmware must echo `sample_id`.
Mismatch, timeout or duplicate ID invalidates the measurement round.
