# Base Models

FP32 baseline LSTM models for battery state estimation.

## Active Models

### SOC (State of Charge) - v1.5.0.0
**Location**: `soc_1.5.0.0_base/`

- **Checkpoint**: `1.5.0.0_soc_epoch0001_rmse0.02897.pt`
- **Architecture**: 64 hidden LSTM + 64 hidden MLP (FP32)
- **Features (6)**: Voltage, Current, Temperature, Q_c, dU_dt, dI_dt
- **Performance**: 87 samples/s, RMSE=0.02897
- **STM32 Firmware**: `AI_Project_LSTM_SOC_base`
- **Quantized**: `quantized/soc_1.5.0.0_quantized/`

### SOH (State of Health) - v2.1.0.0
**Location**: `soh_2.1.0.0_base/`

- **Checkpoint**: `2.1.0.0_soh_epoch0120_rmse0.03359.pt`
- **Architecture**: 128 hidden LSTM + 128 hidden MLP (FP32)
- **Features (6)**: Testtime, Voltage, Current, Temperature, EFC, Q_c
- **Performance**: 31 samples/s, MAE=0.048%, RMSE=0.03359
- **STM32 Firmware**: `AI_Project_LSTM_SOH_base`
- **Quantized**: `quantized/soh_2.1.0.0_quantized/`

## Model Structure
Each model folder contains:
- **Checkpoint** (`.pt`): PyTorch trained model with weights, config, optimizer state
- **C Implementation**: Reference FP32 C code for validation
- **README**: Model details, features, performance metrics

## Training
Models trained with:
- **Framework**: PyTorch LSTM + MLP
- **Optimizer**: Adam with weight decay
- **Scheduler**: Cosine warm restarts
- **Data**: MGFarm 18650 LFP cells (feature-engineered)
- **Validation**: Hold-out cells for generalization testing

## Deployment Pipeline
1. **Train**: PyTorch model with feature engineering
2. **Export**: Save checkpoint with config and features
3. **Quantize**: INT8 quantization for embedded deployment (optional)
4. **STM32**: Deploy via X-CUBE-AI or manual C implementation
5. **Validate**: Test with hardware-in-the-loop scripts in `6_test/STM32/`

## Archive
Old C implementations and experimental exports stored in `archive/`:
- `c_implementation_20251024/` - Original SOC C reference
- `SOH_c_implementation/` - Original SOH C reference

## Version Naming
- **x.y.z.w_target_epoch####_metric#.#####.pt**
  - `x.y.z.w`: Training script version
  - `target`: SOC or SOH
  - `epoch####`: Training epoch number
  - `metric#.#####`: Best validation metric value
