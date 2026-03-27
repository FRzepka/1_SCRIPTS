/*
 * LSTM-MLP SOH Model Implementation in C (FP32)
 * Single-step stateful inference for SOH prediction (linear head)
 */

// Include weights first so size macros (INPUT_SIZE, HIDDEN_SIZE, ...) apply
#include "model_weights_soh.h"
#include "lstm_model_soh.h"
#include <string.h>

/* Use global buffers to avoid large stack usage */
static float g_gates[4 * HIDDEN_SIZE];
static float g_h_contrib[4 * HIDDEN_SIZE];
static float g_i_gate[HIDDEN_SIZE];
static float g_f_gate[HIDDEN_SIZE];
static float g_g_gate[HIDDEN_SIZE];
static float g_o_gate[HIDDEN_SIZE];
static float g_mlp_hidden[MLP_HIDDEN];

// Stack Measurement
uint32_t g_min_stack_ptr = 0xFFFFFFFF;

void lstm_reset_stack_measure(void) {
    g_min_stack_ptr = 0xFFFFFFFF;
}

static void update_stack_measure(void) {
    uint32_t sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));
    if (sp < g_min_stack_ptr) {
        g_min_stack_ptr = sp;
    }
}

void mat_vec_mul(const float* W, const float* x, const float* b,
                 float* y, int out_size, int in_size)
{
    for (int i = 0; i < out_size; i++) {
        float sum = (b ? b[i] : 0.0f);
        for (int j = 0; j < in_size; j++) {
            sum += W[i * in_size + j] * x[j];
        }
        y[i] = sum;
    }
}

void vec_add(float* a, const float* b, int size)
{
    for (int i = 0; i < size; i++) a[i] += b[i];
}

void vec_copy(float* dst, const float* src, int size)
{
    memcpy(dst, src, size * sizeof(float));
}

static void lstm_cell_forward_soh(const float* input, LSTMStateSOH* state)
{
    update_stack_measure(); // Measure stack at entry
    /* gates = W_ih * x + W_hh * h + (b_ih + b_hh) */
    mat_vec_mul(LSTM_WEIGHT_IH, input, LSTM_BIAS, g_gates, 4 * HIDDEN_SIZE, INPUT_SIZE);
    mat_vec_mul(LSTM_WEIGHT_HH, state->h, NULL, g_h_contrib, 4 * HIDDEN_SIZE, HIDDEN_SIZE);
    vec_add(g_gates, g_h_contrib, 4 * HIDDEN_SIZE);

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        g_i_gate[i] = sigmoidf_fast(g_gates[i]);
        g_f_gate[i] = sigmoidf_fast(g_gates[HIDDEN_SIZE + i]);
        g_g_gate[i] = tanhf_fast(g_gates[2 * HIDDEN_SIZE + i]);
        g_o_gate[i] = sigmoidf_fast(g_gates[3 * HIDDEN_SIZE + i]);
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        state->c[i] = g_f_gate[i] * state->c[i] + g_i_gate[i] * g_g_gate[i];
        state->h[i] = g_o_gate[i] * tanhf_fast(state->c[i]);
    }
}

/* Linear SOH head: y = W2 * relu(W1*h + b1) + b2 (no sigmoid) */
static float mlp_forward_soh(const float* h)
{
    update_stack_measure(); // Measure stack at entry
    mat_vec_mul(MLP_FC1_WEIGHT, h, MLP_FC1_BIAS, g_mlp_hidden, MLP_HIDDEN, HIDDEN_SIZE);
    for (int i = 0; i < MLP_HIDDEN; i++) g_mlp_hidden[i] = reluf(g_mlp_hidden[i]);
    float y=0.0f;
    mat_vec_mul(MLP_FC2_WEIGHT, g_mlp_hidden, MLP_FC2_BIAS, &y, 1, MLP_HIDDEN);
    return y;
}

void lstm_model_soh_init(LSTMModelSOH* model)
{
    memset(model->state.h, 0, sizeof(model->state.h));
    memset(model->state.c, 0, sizeof(model->state.c));
    model->initialized = 1;
}

void lstm_model_soh_reset(LSTMModelSOH* model)
{
    lstm_model_soh_init(model);
}

void lstm_model_soh_inference(LSTMModelSOH* model, const float* input, float* output)
{
    if (!model->initialized) lstm_model_soh_init(model);
    lstm_cell_forward_soh(input, &model->state);
    *output = mlp_forward_soh(model->state.h);
}
