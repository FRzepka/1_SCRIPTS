/*
 * STATIC INT8 Model Weights
 * Exported from ONNX with STATIC quantization
 * Input scale is FIXED (pre-calibrated)
 */

#ifndef MODEL_WEIGHTS_INT8_STATIC_H
#define MODEL_WEIGHTS_INT8_STATIC_H

#include <stdint.h>

// Model dimensions: INPUT_SIZE=256, HIDDEN_SIZE=64
// LSTM_GATES=256, LSTM_WEIGHT_SIZE=1536, LSTM_RECURRENT_SIZE=16384
// These constants are defined in lstm_model_hybrid_int8.h

#define LSTM_USE_FP32 1

extern const float lstm_W[1536];
extern const float lstm_R[16384];
extern const float lstm_B[256];

// MLP Quantization Parameters
// No input scale found - using dynamic quantization
#define INPUT_SCALE (1.0f / 127.0f)
#define INPUT_ZERO_POINT 0

// MLP Quantization Parameters
extern const float mlp_input_scale;
extern const int8_t mlp_input_zero_point;
extern const float mlp0_relu_output_scale;
extern const int8_t mlp0_relu_output_zero_point;

// MLP Layer 0 (64 hidden units)
extern const int8_t mlp0_weight_q[4096];
extern const float mlp0_weight_scale[64];
extern const int32_t mlp0_bias_q[64];
extern const float mlp0_bias_scale[64];

// MLP Layer 3 (output layer)
extern const int8_t mlp3_weight_q[64];
extern const float mlp3_weight_scale[1];
extern const int32_t mlp3_bias_q[1];
extern const float mlp3_bias_scale[1];

#endif // MODEL_WEIGHTS_INT8_STATIC_H
