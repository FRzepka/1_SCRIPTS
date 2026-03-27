# Quantized Models

INT8/INT16 quantized LSTM models for STM32 deployment.

## Active Models

### SOC (State of Charge) - v1.5.0.0
**Location**: `soc_1.5.0.0_quantized/`

- **Source**: `base/1.5.0.0_soc_epoch0001_rmse0.02897.pt`
- **Architecture**: 64 hidden LSTM (INT8) + 64 MLP (FP32)
- **Features (6)**: Voltage, Current, Temperature, Q_c, dU_dt, dI_dt
- **Performance**: 57 samples/s, MAE < 0.3%
- **STM32 Firmware**: `AI_Project_LSTM_SOC_quantized`

### SOH (State of Health) - v2.1.0.0
**Location**: `soh_2.1.0.0_quantized/`

- **Source**: `base/2.1.0.0_soh_epoch0120_rmse0.03359.pt`
- **Architecture**: 128 hidden LSTM (INT8/INT16) + 128 MLP (FP32)
- **Features (6)**: Testtime, Voltage, Current, Temperature, EFC, Q_c
- **Performance**: 28 samples/s, MAE=0.048% (better than FP32!)
- **STM32 Firmware**: `AI_Project_LSTM_SOH_quantized`

## Quantization Strategy
- **LSTM Weights**: INT8 per-row/channel quantization
- **LSTM Activations**: FP32 (for numerical stability)
- **MLP**: Full FP32 (no quantization)
- **Memory Savings**: ~70% reduction for LSTM weights
- **Accuracy**: Comparable or better than FP32 baseline

## Archive
Old experimental quantized models and intermediate exports are stored in `archive/` for reference:
- `manual_int8_lstm/` - Original SOC quantization
- `SOH_Quantized/` - Original SOH quantization
- `manual_int8_lstm_soh/` - Early SOH experiments
- `manual_int8_lstm_soh_export*/` - Various export attempts
- `manual_int16hh_lstm_soh_export_p99_5/` - INT16 HH experiments

## STM32 Integration
Quantized models are deployed to STM32H753ZI via CubeIDE projects:
1. Generate quantization with Python scripts in `4_quantize/`
2. Copy C headers to firmware `Core/Inc/`
3. Build and flash firmware
4. Test with scripts in `6_test/STM32/quantized/{SOC|SOH}/`

## Validation
Each active model folder contains:
- Model weights (NPZ or C headers)
- Metadata (JSON manifest)
- Validation scripts (Python)
- Performance metrics
- README with generation commands


