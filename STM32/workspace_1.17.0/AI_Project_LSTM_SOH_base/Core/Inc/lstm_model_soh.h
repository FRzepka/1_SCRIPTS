/*
 * LSTM-MLP SOH Model Implementation in C (FP32)
 * Single-step stateful inference for SOH prediction (linear head)
 */

#ifndef LSTM_MODEL_SOH_H
#define LSTM_MODEL_SOH_H

#include <stdint.h>
#include <math.h>

/* Model configuration (keep in sync with exported weights) */
#ifndef INPUT_SIZE
#define INPUT_SIZE 6
#endif
#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 128
#endif
#ifndef MLP_HIDDEN
#define MLP_HIDDEN 128
#endif
#ifndef NUM_LAYERS
#define NUM_LAYERS 1
#endif

/* Model state structure */
typedef struct {
    float h[HIDDEN_SIZE];  /* Hidden state */
    float c[HIDDEN_SIZE];  /* Cell state */
} LSTMStateSOH;

/* Model instance */
typedef struct {
    LSTMStateSOH state;
    int initialized;
} LSTMModelSOH;

/* ========== Function Declarations ========== */

void lstm_model_soh_init(LSTMModelSOH* model);
void lstm_model_soh_reset(LSTMModelSOH* model);
void lstm_model_soh_inference(LSTMModelSOH* model, const float* input, float* output);

/* Helpers */
static inline float sigmoidf_fast(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float tanhf_fast(float x) { return tanhf(x); }
static inline float reluf(float x) { return (x > 0.0f) ? x : 0.0f; }

/* Stack Measurement Globals */
extern volatile uint32_t g_min_stack_ptr;
static inline void lstm_reset_stack_measure(void) { g_min_stack_ptr = 0xFFFFFFFF; }

#endif /* LSTM_MODEL_SOH_H */


