# SOC Base Model (v1.5.0.0)

**Active FP32 baseline model for State of Charge (SOC) prediction.**

## Model Checkpoint
- **File**: `1.5.0.0_soc_epoch0001_rmse0.02897.pt`
- **Version**: 1.5.0.0
- **Target**: SOC (State of Charge)
- **Architecture**: 64 hidden LSTM + 64 hidden MLP
- **Precision**: FP32
- **Training Metric**: RMSE = 0.02897

## Features (6)
1. `Voltage[V]`
2. `Current[A]`
3. `Temperature[°C]`
4. `Q_c` (Charge capacity)
5. `dU_dt[V/s]` (Voltage rate of change)
6. `dI_dt[A/s]` (Current rate of change)

## C Implementation
**Location**: `c_implementation/`

C reference implementation for validation (not for production):
- `lstm_model_soc.c/h` - FP32 LSTM implementation
- `test_main.c` - Validation test harness
- Compiled with: `gcc -O3 -march=native`

Used for:
- Validating Python→C conversion accuracy
- Benchmarking against quantized versions
- Reference for STM32 firmware development

## STM32 Firmware
**Target Project**: `STM32/workspace_1.17.0/AI_Project_LSTM_SOC_base`

Uses X-CUBE-AI generated code from this checkpoint.

## Performance
- **Sample Rate**: ~87 samples/s on STM32H753ZI
- **MAE**: Typically < 1% on validation datasets
- **Memory**: Full FP32 precision (~256 KB for weights)

## Training Configuration
- **Training Version**: 1.5.0.0
- **Epochs**: 1 (early stop or single epoch run)
- **Optimizer**: Adam with weight decay
- **Loss**: MSE (reported as RMSE)
- **Data**: MGFarm 18650 cells with feature engineering

## Usage
Load checkpoint in PyTorch:
```python
import torch
ckpt = torch.load('1.5.0.0_soc_epoch0001_rmse0.02897.pt')
model_state = ckpt['model_state_dict']
config = ckpt['config']
features = ckpt['features']
```

## Quantized Version
INT8 quantized version: `2_models/quantized/soc_1.5.0.0_quantized/`
