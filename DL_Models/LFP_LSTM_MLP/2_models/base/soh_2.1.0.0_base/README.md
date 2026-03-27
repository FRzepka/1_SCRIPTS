# SOH Base Model (v2.1.0.0)

**Active FP32 baseline model for State of Health (SOH) prediction.**

## Model Checkpoint
- **File**: `2.1.0.0_soh_epoch0120_rmse0.03359.pt`
- **Version**: 2.1.0.0
- **Target**: SOH (State of Health)
- **Architecture**: 128 hidden LSTM + 128 hidden MLP
- **Precision**: FP32
- **Training Metric**: RMSE = 0.03359 (at epoch 120)

## Features (6)
1. `Testtime[s]` (Cumulative test time)
2. `Voltage[V]`
3. `Current[A]`
4. `Temperature[°C]`
5. `EFC` (Equivalent Full Cycles)
6. `Q_c` (Charge capacity)

## C Implementation
**Location**: `c_implementation/`

C reference implementation for validation:
- `lstm_model_soh.c/h` - FP32 LSTM implementation with post-processing
- `post_process.c/h` - PIN calibration + strict filtering + EMA smoothing
- `test_main.c` - Validation test harness
- Compiled with: `gcc -O3 -march=native`

Post-processing pipeline:
- **PIN-before-filter**: Calibration at startup
- **Strict step limiting**: REL_CAP=1e-4, ABS_CAP=2e-6, 5 passes
- **EMA smoothing**: alpha=2e-4

## STM32 Firmware
**Target Project**: `STM32/workspace_1.17.0/AI_Project_LSTM_SOH_base`

Uses X-CUBE-AI generated code from this checkpoint with custom post-processing.

## Performance
- **Sample Rate**: ~31 samples/s on STM32H753ZI (slower due to 128 hidden size)
- **MAE**: 0.000484 (0.048% error)
- **RMSE**: 0.000645 (0.064% error)
- **MAX Error**: 0.00142 (0.14% max deviation)
- **Memory**: Full FP32 precision (~1 MB for weights)

## Training Configuration
- **Training Version**: 2.1.0.0
- **Epochs**: 120 (with early stopping)
- **Optimizer**: Adam with weight decay
- **Scheduler**: Cosine warm restarts
- **Loss**: MSE (reported as RMSE)
- **Data**: MGFarm 18650 cells (train: C01,C03,C05,C11,C17,C23 | val: C07,C19,C21)
- **Chunk Size**: 2048 samples

## Usage
Load checkpoint in PyTorch:
```python
import torch
ckpt = torch.load('2.1.0.0_soh_epoch0120_rmse0.03359.pt')
model_state = ckpt['model_state_dict']
config = ckpt['config']
features = ckpt['features']  # ['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'EFC', 'Q_c']
```

## Quantized Version
INT8/INT16 quantized version: `2_models/quantized/soh_2.1.0.0_quantized/`

Quantized version actually **outperforms** FP32 baseline (MAE: 0.000476 vs 0.000484)!

