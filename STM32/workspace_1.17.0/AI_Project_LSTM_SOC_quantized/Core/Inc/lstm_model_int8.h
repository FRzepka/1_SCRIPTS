/*
 * INT8 Quantized LSTM-MLP Model Implementation
 * Single-step stateful inference for SOC prediction
 * 
 * Quantization: Dynamic INT8 with per-channel scales
 * Dequantization: real_value = scale * (int8_value - zero_point)
 */

#ifndef LSTM_MODEL_INT8_H
#define LSTM_MODEL_INT8_H

#include <stdint.h>
#include <math.h>

/* Model configuration
 * Do not define INPUT_SIZE/HIDDEN_SIZE here to avoid redefinition warnings.
 * The manual weight header provides INPUT_SIZE/HIDDEN_SIZE and channel dims.
 */
#ifndef LSTM_INPUT_SIZE
#define LSTM_INPUT_SIZE 6
#endif
#ifndef LSTM_HIDDEN_SIZE
#define LSTM_HIDDEN_SIZE 64
#endif

/* Model state structure (FP32 states for numerical stability) */
typedef struct {
    float h[LSTM_HIDDEN_SIZE];  /* Hidden state */
    float c[LSTM_HIDDEN_SIZE];  /* Cell state */
} LSTMState_INT8;

/* Model instance */
typedef struct {
    LSTMState_INT8 state;
    int initialized;
} LSTMModel_INT8;

/* ========== Function Declarations ========== */

/**
 * Stack Measurement (Global)
 */
extern uint32_t g_min_stack_ptr;
void lstm_reset_stack_measure(void);

/**
 * Initialize model (reset states to zero)
 */
void lstm_model_int8_init(LSTMModel_INT8* model);

/**
 * Reset states to zero
 */
void lstm_model_int8_reset(LSTMModel_INT8* model);

/**
 * Single-step inference with INT8 quantized weights
 * 
 * @param model     Model instance
 * @param input     Input features [INPUT_SIZE] (FP32)
 * @param output    Output SOC prediction (pointer to single float)
 */
void lstm_model_int8_inference(LSTMModel_INT8* model, const float* input, float* output);

/* ========== INT8 Helper Functions ========== */

/**
 * INT8 Matrix-Vector Multiplication with dequantization
 * W_int8: [out_size, in_size] (quantized weights)
 * x: [in_size] (FP32 input)
 * scales: [out_size] (per-channel scales)
 * zero_points: [out_size] (per-channel zero points)
 * y: [out_size] (FP32 output)
 * 
 * Formula: y[i] = sum_j(scale[i] * (W_int8[i,j] - zp[i]) * x[j])
 */
void mat_vec_mul_int8(const int8_t* W_int8, const float* x, 
                      const float* scales, const int8_t* zero_points,
                      float* y, int out_size, int in_size);

/**
 * INT8 Matrix-Vector Multiplication with bias
 * Same as above but adds FP32 bias
 */
void mat_vec_mul_int8_bias(const int8_t* W_int8, const float* x,
                           const float* scales, const int8_t* zero_points,
                           const float* bias,
                           float* y, int out_size, int in_size);

/* ========== Activation Functions ========== */

/**
 * Sigmoid activation: 1 / (1 + exp(-x))
 */
static inline float sigmoid_int8(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Tanh activation
 */
static inline float tanh_int8(float x) {
    return tanhf(x);
}

/**
 * ReLU activation: max(0, x)
 */
static inline float relu_int8(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

/* ========== Vector Operations ========== */

void vec_add_int8(float* a, const float* b, int size);
void vec_mul_int8(float* a, const float* b, int size);
void vec_copy_int8(float* dst, const float* src, int size);

/* ======== Minimal API used by main.c (manual INT8 LSTM + FP32 MLP) ======== */
/**
 * Predict one step using manual INT8 weights and FP32 math for stability.
 * input: [LSTM_INPUT_SIZE] floats (raw features),
 * h_state/c_state: [LSTM_HIDDEN_SIZE] floats (updated in-place),
 * prediction: output float in [0,1].
 */
void lstm_model_predict_int8(
    const float *input,
    float *h_state,
    float *c_state,
    float *prediction);

/**
 * Zero-initialize hidden and cell states.
 */
void lstm_model_reset_states(float *h_state, float *c_state);

#endif /* LSTM_MODEL_INT8_H */
