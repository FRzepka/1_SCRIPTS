# STM32 Quantized SOH Testing

Scripts for testing SOH (State of Health) inference on STM32H753ZI with INT8 LSTM + FP32 MLP model.

## Hardware Requirements
- **Board**: NUCLEO-H753ZI
- **Firmware**: `AI_Project_LSTM_SOH_quantized` (in `STM32/workspace_1.17.0/`)
- **COM Port**: Typically COM7 (STLink Virtual COM Port)

## Model Configuration
- **Features (6)**: Testtime[s], Voltage[V], Current[A], Temperature[°C], EFC, Q_c
- **Target**: SOH (State of Health)
- **Architecture**: 128 hidden LSTM (INT8) + 128 hidden MLP (FP32)
- **Quantization**: Per-channel INT8 for LSTM weights
- **Post-processing**: PIN-before-filter + strict step limiting + EMA smoothing
- **Memory Savings**: ~70% reduction vs FP32 (LSTM weights only)

## Scripts

### `run_soh_quantized_stream_and_plot.py`
Streams N samples from a parquet file to the STM32, collects SOH predictions, and saves plots + metrics.

**Usage:**
```bash
python run_soh_quantized_stream_and_plot.py \
  --port COM7 \
  --baud 115200 \
  --parquet "c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" \
  --start 0 \
  --n 3000 \
  --prime 0 \
  --delay 0.01
```

**Important Parameters:**
- `--prime N`: Number of priming samples (default 2047). Set to 0 to capture from first sample.
- `--strict-filter`: Enable strict step filtering (if you want tighter predictions)

**Outputs** (in `results/STM32_SOH_QUANTIZED_STREAM_<timestamp>/`):
- `overlay_full.png` - Full predictions vs ground truth
- `overlay_firstN.png` - First 500 samples zoomed
- `diff_hist.png` - Error distribution histogram
- `metrics.json` - MAE, RMSE, MAX error, inference timing
- `log.txt` - Complete UART communication log

### `run_soh_quantized_live_window.py`
Real-time rolling window plot of SOH predictions (for interactive debugging).

**Usage:**
```bash
python run_soh_quantized_live_window.py \
  --port COM7 \
  --baud 115200 \
  --parquet "c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" \
  --window 2000 \
  --start 0 \
  --n 5000
```

## Data Format
Parquet files must contain:
- **Feature columns**: `Testtime[s]`, `Voltage[V]`, `Current[A]`, `Temperature[°C]`, `EFC`, `Q_c`
- **Ground truth**: `SOH` column (for validation)

Typical data source: `MGFarm_18650_FE/df_FE_C*.parquet` (Feature-Engineered files)

## Validated Results
- **Sample Rate**: ~28.0 samples/s (slower than SOC due to larger 128-hidden model)
- **MAE**: 0.000476 (0.048% error) - **Better than FP32 baseline!**
- **RMSE**: 0.000638 (0.064% error)
- **MAX Error**: 0.00140 (0.14% max deviation)
- **Timeouts**: 0 (stable communication)

## Quantization Details
- **LSTM Weights**: INT8 per-channel quantization
- **LSTM Activations**: FP32 (for numerical stability)
- **MLP**: Full FP32 (no quantization)
- **Scales**: Per-channel scales for dequantization
- **Generated from**: `DL_Models/LFP_LSTM_MLP/4_quantize/manual_lstm_int8_from_pt_soh.py`

## Performance Comparison
| Metric | FP32 Base | INT8 Quantized | Change |
|--------|-----------|----------------|--------|
| MAE | 0.000484 | 0.000476 | **-1.6%** ✅ |
| RMSE | 0.000645 | 0.000638 | **-1.1%** ✅ |
| Sample Rate | 31.4 s/s | 28.0 s/s | -10.8% |
| Memory (LSTM) | 100% | ~30% | **-70%** ✅ |

Quantization actually **improved** accuracy while reducing memory by 70%!

## Common Issues
1. **Port blocked**: Close CubeIDE debug sessions before running scripts
2. **Wrong features**: Ensure you send **6 features** in correct order (Testtime, V, I, T, EFC, Q_c)
3. **No results**: Check `--prime` parameter - default 2047 means first 2047 samples are discarded
4. **Firmware mismatch**: Verify `AI_Project_LSTM_SOH_quantized` is flashed (not SOC variant)
