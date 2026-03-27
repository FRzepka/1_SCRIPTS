# STM32 Base SOC Testing

Scripts for testing SOC (State of Charge) inference on STM32H753ZI with FP32 LSTM+MLP model.

## Hardware Requirements
- **Board**: NUCLEO-H753ZI
- **Firmware**: `AI_Project_LSTM_SOC_base` (in `STM32/workspace_1.17.0/`)
- **COM Port**: Typically COM7 (STLink Virtual COM Port)

## Model Configuration
- **Features (6)**: Voltage[V], Current[A], Temperature[°C], Q_c, dU_dt[V/s], dI_dt[A/s]
- **Target**: SOC (State of Charge)
- **Architecture**: 64 hidden LSTM + 64 hidden MLP
- **Precision**: FP32

## Scripts

### `run_soc_stream_and_plot.py`
Streams N samples from a parquet file to the STM32, collects SOC predictions, and saves plots + metrics.

**Usage:**
```bash
python run_soc_stream_and_plot.py \
  --port COM7 \
  --baud 115200 \
  --parquet "path/to/df_FE_C07.parquet" \
  --start 0 \
  --n 3000 \
  --delay 0.01
```

**Outputs** (in `results/STM32_SOC_STREAM_<timestamp>/`):
- `overlay_full.png` - Full predictions vs ground truth
- `overlay_firstN.png` - First 500 samples zoomed
- `diff_hist.png` - Error distribution histogram
- `metrics.json` - MAE, RMSE, MAX error
- `log.txt` - Complete UART communication log

### `run_soc_live_plot.py`
Real-time rolling window plot of SOC predictions (for interactive debugging).

**Usage:**
```bash
python run_soc_live_plot.py \
  --port COM7 \
  --baud 115200 \
  --parquet "path/to/df_FE_C07.parquet" \
  --window 2000 \
  --start 0 \
  --n 5000
```

## Data Format
Parquet files must contain:
- **Feature columns**: `Voltage[V]`, `Current[A]`, `Temperature[°C]`, `Q_c`, `dU_dt[V/s]`, `dI_dt[A/s]`
- **Ground truth**: `SOC` or `soc` column (for validation)

Typical data source: `MGFarm_18650_FE/df_FE_C*.parquet` (Feature-Engineered files)

## Validated Results
- **Sample Rate**: ~87.5 samples/s
- **MAE**: Typically < 1% for validation datasets
- **Timeouts**: 0 (stable communication)
