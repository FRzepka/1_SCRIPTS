# ECM (Qinnan) – Python EKF Model

This folder provides a Python implementation of the embedded ECM EKF from `echo/CM7/Core/Src/battery_ekf.c` using the lookup tables in `simulator/ECM_parameter.c`.

## Inputs
- Current `Current[A]`
- Voltage `Voltage[V]`
- SoH (optional; if missing, defaults to 1.0)
- Time base via `Testtime[s]` (or fixed `dt_s`)

## Outputs
- SOC estimate (`soc_ecm`)
- Estimated terminal voltage (`u_ecm`)

## How it works
The model ports the EKF math from the C code. It supports:
- charge/discharge parameter sets (based on current sign)
- SoH scaling of capacity
- configurable `delta_t_s` (fixed or from data)
- optional covariance update (off by default to mirror C implementation)

## Run a test
Example (uses FE parquet data):
```
python run_ecm_test.py \
  --cell MGFarm_18650_C07 \
  --out_dir /path/to/out \
  --dt_mode data \
  --warmup_seconds 600
```

Outputs:
- `ecm_soc_fullcell_<cell>.csv`
- `ecm_soc_fullcell_<cell>.png`
- `summary.json`

## Notes
- The embedded code uses a fixed `deltaT=60s`. If you want exact firmware behavior, run with `--dt_mode fixed --dt_s 60`.
- Default covariance update is disabled to match the C file as-is.

## Native C (exact firmware math)
If you want to execute the original C EKF on the host, build the native shared library and use the ctypes wrapper:

Build:
```
python_model/native/build_native.sh
```

Usage:
```
from ecm_native import ECMNativeEKF

ekf = ECMNativeEKF(soc_init=1.0)
soc, u_hat = ekf.step(current_a, voltage_v, soh=1.0)
```

This uses the unmodified `battery_ekf.c` and the same parameter tables.
