# AI_Project_LSTM_hybrid_int8

## Hybrid LSTM Model for SOC Prediction on STM32H753ZI

### Model Architecture
- **LSTM Layer**: FP32 (256 gates, 64 hidden units)
- **MLP Layers**: INT8 quantized (64→64→1 with ReLU)

### Performance
- **Accuracy**: ~1% MAE (50k test samples on PC)
- **Model Size**: 35 KB (vs 90 KB FP32 baseline)
- **Size Reduction**: 61%

### Key Implementation Details

#### CRITICAL: ONNX LSTM Gate Order
ONNX uses gate order **[i, o, f, c]**, NOT PyTorch **[i, f, g, o]**!
- `i`: Input gate (0-63)
- `o`: Output gate (64-127)  
- `f`: Forget gate (128-191)
- `c`: Cell gate (192-255)

This was the main bug causing 19% error initially!

#### Quantization Scheme
- **LSTM**: FP32 (no quantization)
- **MLP Input**: UINT8, scale=0.007797, zero_point=127
- **MLP Layer 0**: INT8 weights, per-channel scales
- **MLP Layer 0→3**: UINT8, scale=0.009019, zero_point=0 (ReLU output)
- **MLP Layer 3**: INT8 weights, final sigmoid output

### Files

#### Core Implementation
- `lstm_model_hybrid_int8.h/c`: Main model implementation
- `model_weights_hybrid_int8.h`: Exported weights from ONNX (210 KB header)
- `ai_wrapper.h/c`: Wrapper interface for main application
- `scaler_params.h`: RobustScaler preprocessing parameters

#### Support Files
- `main.c`: STM32 application with UART interface
- `gpio.c/h`: GPIO configuration

### Usage

```c
#include "ai_wrapper.h"

// Initialize model
ai_wrapper_init();

// Prepare input [V, I, T, SOC_prev, Ah, Time]
float input[6] = {3.2f, 0.5f, 25.0f, 0.5f, 0.1f, 10.0f};
float soc;

// Run prediction (streaming, stateful)
ai_wrapper_run(input, &soc);

// Reset states if needed
ai_wrapper_reset_state();
```

### Testing

Use Python script to test over UART:
```bash
cd ../../Arduino/LFP_SOC_SOH/bin
python stm32_uart_test_50k.py --port COM3
```

### Compilation
- **IDE**: STM32CubeIDE 1.17.0
- **Target**: STM32H753ZI (480 MHz Cortex-M7)
- **Optimization**: -O2 or -O3
- **Float ABI**: Hardware FP (FPU enabled)

### Performance Expectations
- **FP32 baseline**: ~500 tokens/sec
- **Hybrid INT8**: ~800-1000 tokens/sec (estimated)
- **Speedup**: ~1.6-2× vs FP32

### Next Steps
1. Compile in STM32CubeIDE
2. Flash to STM32H753ZI
3. Test with `stm32_uart_test_50k.py`
4. Compare accuracy and throughput vs FP32 baseline
5. Consider full INT8 LSTM for further speedup
