/*
 * LSTM-MLP Model Implementation in C
 * Single-step stateful inference for SOC prediction
 */

#ifndef LSTM_MODEL_H
#define LSTM_MODEL_H

#include <stdint.h>
#include <math.h>

/* Model configuration (will be defined in model_weights.h) */
#ifndef INPUT_SIZE
#define INPUT_SIZE 6
#endif

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 64
#endif

#ifndef MLP_HIDDEN
#define MLP_HIDDEN 64
#endif

#ifndef NUM_LAYERS
#define NUM_LAYERS 1
#endif

/* Model state structure */
typedef struct {
    float h[HIDDEN_SIZE];  /* Hidden state */
    float c[HIDDEN_SIZE];  /* Cell state */
} LSTMState;

/* Model instance */
typedef struct {
    LSTMState state;
    int initialized;
} LSTMModel;

/* ========== Function Declarations ========== */

/**
 * Initialize model (reset states to zero) - FP32 version
 */
void lstm_model_fp32_init(LSTMModel* model);

/**
 * Reset states to zero
 */
void lstm_model_reset(LSTMModel* model);

/**
 * Single-step inference
 * 
 * @param model     Model instance
 * @param input     Input features [INPUT_SIZE]
 * @param output    Output SOC prediction (pointer to single float)
 */
void lstm_model_inference(LSTMModel* model, const float* input, float* output);

/* ========== Helper Functions ========== */

/**
 * Sigmoid activation: 1 / (1 + exp(-x))
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Tanh activation
 */
static inline float tanh_activation(float x) {
    return tanhf(x);
}

/**
 * ReLU activation: max(0, x)
 */
static inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

/**
 * Matrix-vector multiplication: y = W * x + b
 * W: [out_size, in_size] (row-major)
 * x: [in_size]
 * b: [out_size]
 * y: [out_size]
 */
void mat_vec_mul(const float* W, const float* x, const float* b, 
                 float* y, int out_size, int in_size);

/**
 * Element-wise vector operations
 */
void vec_add(float* a, const float* b, int size);
void vec_mul(float* a, const float* b, int size);
void vec_copy(float* dst, const float* src, int size);

#endif /* LSTM_MODEL_H */
