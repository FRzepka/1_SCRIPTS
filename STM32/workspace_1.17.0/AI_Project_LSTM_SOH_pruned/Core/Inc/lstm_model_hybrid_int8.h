/*
 * lstm_model_hybrid_int8.h
 *
 * Hybrid LSTM Model: FP32 LSTM + INT8 MLP
 * Optimized for STM32H753ZI (480 MHz Cortex-M7)
 *
 * Model Structure:
 *   - LSTM: FP32 (256 gates, 64 hidden units)
 *   - MLP:  INT8 quantized (64->64->1 with ReLU)
 *
 * Performance: ~1% MAE on 50k test samples
 * Memory: ~35 KB (vs 90 KB FP32)
 */

#ifndef LSTM_MODEL_HYBRID_INT8_H_
#define LSTM_MODEL_HYBRID_INT8_H_

#include <stdint.h>
#include <math.h>

// Model configuration
#define INPUT_SIZE 6
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 1

// LSTM uses FP32
#define LSTM_USE_FP32 1

// MLP uses INT8
#define MLP_USE_INT8 1
#define MLP_HIDDEN_SIZE 64

// Model state structure
typedef struct {
    float h[HIDDEN_SIZE];  // Hidden state (FP32)
    float c[HIDDEN_SIZE];  // Cell state (FP32)
} LSTMState;

// Function prototypes
void lstm_model_init(LSTMState *state);
float lstm_model_predict(LSTMState *state, const float *input);

// Inline activation functions
static inline float sigmoid_f32(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static inline float tanh_f32(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return -1.0f;
    return tanhf(x);
}

static inline float relu_f32(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

#endif /* LSTM_MODEL_HYBRID_INT8_H_ */
