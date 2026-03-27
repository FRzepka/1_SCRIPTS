# STM32 Base SOH Testing

Scripts for testing SOH (State of Health) inference on STM32H753ZI with FP32 LSTM+MLP model.

## Hardware Requirements
- **Board**: NUCLEO-H753ZI
- **Firmware**: `AI_Project_LSTM_SOH_base` (in `STM32/workspace_1.17.0/`)
- **COM Port**: Typically COM7 (STLink Virtual COM Port)

## Model Configuration
- **Features (6)**: Testtime[s], Voltage[V], Current[A], Temperature[°C], EFC, Q_c
- **Target**: SOH (State of Health)
- **Architecture**: 128 hidden LSTM + 128 hidden MLP
- **Precision**: FP32
- **Post-processing**: PIN-before-filter + strict step limiting + EMA smoothing

## Scripts

### `run_soh_stream_and_plot.py`
Streams N samples from a parquet file to the STM32, collects SOH predictions, and saves plots + metrics.

**Usage:**
```bash
python run_soh_stream_and_plot.py \
  --port COM7 \
  --baud 115200 \
  --parquet "path/to/df_FE_C07.parquet" \
  --start 0 \
  --n 3000 \
  --prime 0 \
  --delay 0.01
```

**Important Parameters:**
- `--prime N`: Number of priming samples (default 2047). Set to 0 to capture from first sample.
- `--strict-filter`: Enable strict step filtering (if you want tighter predictions)

**Outputs** (in `results/STM32_SOH_STREAM_<timestamp>/`):
- `overlay_full.png` - Full predictions vs ground truth
- `overlay_firstN.png` - First 500 samples zoomed
- `diff_hist.png` - Error distribution histogram
- `metrics.json` - MAE, RMSE, MAX error
- `log.txt` - Complete UART communication log

## Data Format
Parquet files must contain:
- **Feature columns**: `Testtime[s]`, `Voltage[V]`, `Current[A]`, `Temperature[°C]`, `EFC`, `Q_c`
- **Ground truth**: `SOH` column (for validation)

Typical data source: `MGFarm_18650_FE/df_FE_C*.parquet` (Feature-Engineered files)

## Validated Results
- **Sample Rate**: ~31.4 samples/s (slower due to larger model)
- **MAE**: 0.00048 (0.048% error)
- **RMSE**: 0.00064 (0.064% error)
- **MAX Error**: 0.00142 (0.14% max deviation)
- **Timeouts**: 0 (stable communication)

## Common Issues
1. **Port blocked**: Close CubeIDE debug sessions before running scripts
2. **Wrong features**: Ensure you send **6 features** in correct order (Testtime, V, I, T, EFC, Q_c)
3. **No results**: Check `--prime` parameter - default 2047 means first 2047 samples are discarded
