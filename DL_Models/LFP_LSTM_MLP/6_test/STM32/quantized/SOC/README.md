# STM32 Quantized SOC Testing

Scripts for testing SOC (State of Charge) inference on STM32H753ZI with INT8 LSTM + FP32 MLP model.

## Hardware Requirements
- **Board**: NUCLEO-H753ZI
- **Firmware**: `AI_Project_LSTM_SOC_quantized` (in `STM32/workspace_1.17.0/`)
- **COM Port**: Typically COM7 (STLink Virtual COM Port)

## Model Configuration
- **Features (6)**: Voltage[V], Current[A], Temperature[°C], Q_c, dU_dt[V/s], dI_dt[A/s]
- **Target**: SOC (State of Charge)
- **Architecture**: 64 hidden LSTM (INT8) + 64 hidden MLP (FP32)
- **Quantization**: Per-row symmetric INT8 for LSTM weights
- **Memory Savings**: ~70% reduction vs FP32 (LSTM weights only)

## Scripts

### `run_soc_quantized_stream_and_plot.py`
Streams N samples from a parquet file to the STM32, collects SOC predictions, and saves plots + metrics.

**Usage:**
```bash
python run_soc_quantized_stream_and_plot.py \
  --port COM7 \
  --baud 115200 \
  --parquet "c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" \
  --start 0 \
  --n 3000 \
  --delay 0.01
```

**Outputs** (in `results/STM32_SOC_QUANTIZED_STREAM_<timestamp>/`):
- `overlay_full.png` - Full predictions vs ground truth
- `overlay_firstN.png` - First 500 samples zoomed
- `diff_hist.png` - Error distribution histogram
- `metrics.json` - MAE, RMSE, MAX error, inference timing
- `log.txt` - Complete UART communication log

### `run_soc_quantized_live_plot.py`
Real-time rolling window plot of SOC predictions (for interactive debugging).

**Usage:**
```bash
python run_soc_quantized_live_plot.py \
  --port COM7 \
  --baud 115200 \
  --parquet "c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" \
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
- **Sample Rate**: ~57.2 samples/s (faster than FP32 base at 86.8 s/s due to INT8 efficiency)
- **Accuracy**: Comparable to FP32 baseline (< 1% MAE)
- **Timeouts**: Minimal (2 per 1000 samples)

## Quantization Details
- **LSTM Weights**: INT8 per-row symmetric quantization
- **LSTM Activations**: FP32 (for numerical stability)
- **MLP**: Full FP32 (no quantization)
- **Scales**: Per-row scales for dequantization
- **Generated from**: `DL_Models/LFP_LSTM_MLP/4_quantize/manual_lstm_int8_from_pt.py`

## Common Issues
1. **Port blocked**: Close CubeIDE debug sessions before running scripts
2. **Wrong features**: Ensure you send **6 features** in correct order (V, I, T, Q_c, dU_dt, dI_dt)
3. **Firmware mismatch**: Verify `AI_Project_LSTM_SOC_quantized` is flashed (not SOH variant)
