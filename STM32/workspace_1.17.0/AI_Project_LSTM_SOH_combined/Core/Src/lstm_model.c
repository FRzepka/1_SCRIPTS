/*
 * LSTM-MLP Model Implementation in C
 * Single-step stateful inference for SOC prediction
 */

#include "lstm_model.h"
// #include "model_weights.h" // Removed, using specific headers
#include "weights_base.h"
#include "weights_pruned.h"
#include "weights_quantized.h"
#include <string.h>

/* ========== Helper Functions ========== */

/* Global working buffers to avoid stack overflow (placed in BSS/RAM) */
/* Using MAX_HIDDEN_SIZE (128) to accommodate all models */
static float g_gates[4 * MAX_HIDDEN_SIZE];
static float g_i_gate[MAX_HIDDEN_SIZE];
static float g_f_gate[MAX_HIDDEN_SIZE];
static float g_g_gate[MAX_HIDDEN_SIZE];
static float g_o_gate[MAX_HIDDEN_SIZE];
static float g_h_contrib[4 * MAX_HIDDEN_SIZE];
static float g_mlp_hidden[MLP_HIDDEN];
static const float g_zero_bias[4 * MAX_HIDDEN_SIZE] = {0};  /* Zero bias for hidden contribution */

/* Current Model State */
static ModelType s_current_model = MODEL_BASE;
static int s_hidden_size = 64; // Default to Base

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

void lstm_model_set_type(ModelType type) {
    s_current_model = type;
    switch (type) {
        case MODEL_BASE:
            s_hidden_size = 128;
            break;
        case MODEL_PRUNED:
            s_hidden_size = 90;
            break;
        case MODEL_QUANTIZED:
            s_hidden_size = 128;
            break;
        default:
            s_hidden_size = 128;
    }
}

ModelType lstm_model_get_type(void) {
    return s_current_model;
}

void mat_vec_mul(const float* W, const float* x, const float* b, 
                 float* y, int out_size, int in_size) {
    for (int i = 0; i < out_size; i++) {
        float sum = (b ? b[i] : 0.0f);  /* Start with bias */
        for (int j = 0; j < in_size; j++) {
            sum += W[i * in_size + j] * x[j];
        }
        y[i] = sum;
    }
}

void vec_add(float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void vec_mul(float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] *= b[i];
    }
}

void vec_copy(float* dst, const float* src, int size) {
    memcpy(dst, src, size * sizeof(float));
}

/* ========== Model Functions ========== */

void lstm_model_init(LSTMModel* model) {
    /* Zero-initialize states */
    memset(model->state.h, 0, MAX_HIDDEN_SIZE * sizeof(float));
    memset(model->state.c, 0, MAX_HIDDEN_SIZE * sizeof(float));
    model->initialized = 1;
}

void lstm_model_reset(LSTMModel* model) {
    lstm_model_init(model);
}

/**
 * LSTM Cell Forward Pass
 */
void lstm_cell_forward(const float* input, LSTMState* state) {
    update_stack_measure(); // Measure stack at entry
    /* Use global buffers to avoid stack overflow */
    int H = s_hidden_size;
    
    /* Select Weights based on current model */
    const float* W_ih;
    const float* W_hh;
    const float* B_lstm;
    
    switch (s_current_model) {
        case MODEL_BASE:
            W_ih = BASE_LSTM_WEIGHT_IH;
            W_hh = BASE_LSTM_WEIGHT_HH;
            B_lstm = BASE_LSTM_BIAS;
            break;
        case MODEL_PRUNED:
            W_ih = PRUNED_LSTM_WEIGHT_IH;
            W_hh = PRUNED_LSTM_WEIGHT_HH;
            B_lstm = PRUNED_LSTM_BIAS;
            break;
        case MODEL_QUANTIZED:
            W_ih = QUANT_LSTM_WEIGHT_IH;
            W_hh = QUANT_LSTM_WEIGHT_HH;
            B_lstm = QUANT_LSTM_BIAS;
            break;
        default:
            return;
    }

    /* CRITICAL: Clear buffers to ensure no corruption between calls */
    for (int i = 0; i < 4 * H; i++) {
        g_gates[i] = 0.0f;
        g_h_contrib[i] = 0.0f;
    }
    
    /* Compute all gates: gates = W_ih * input + W_hh * h + bias */
    /* Note: LSTM weights are stored as [4*H, I] and [4*H, H] */
    
    /* Input contribution: W_ih * input */
    mat_vec_mul(W_ih, input, B_lstm, g_gates, 4 * H, INPUT_SIZE);
    
    /* Hidden contribution: W_hh * h (add to gates) */
    mat_vec_mul(W_hh, state->h, g_zero_bias, g_h_contrib, 4 * H, H);
    vec_add(g_gates, g_h_contrib, 4 * H);
    
    /* Split gates and apply activations */
    for (int i = 0; i < H; i++) {
        g_i_gate[i] = sigmoid(g_gates[i]);                           /* Input gate */
        g_f_gate[i] = sigmoid(g_gates[H + i]);             /* Forget gate */
        g_g_gate[i] = tanh_activation(g_gates[2 * H + i]); /* Cell gate */
        g_o_gate[i] = sigmoid(g_gates[3 * H + i]);         /* Output gate */
    }
    
    /* Update cell state: c_t = f_t * c_{t-1} + i_t * g_t */
    for (int i = 0; i < H; i++) {
        state->c[i] = g_f_gate[i] * state->c[i] + g_i_gate[i] * g_g_gate[i];
    }
    
    /* Update hidden state: h_t = o_t * tanh(c_t) */
    for (int i = 0; i < H; i++) {
        state->h[i] = g_o_gate[i] * tanh_activation(state->c[i]);
    }
}

/**
 * MLP Forward Pass
 */
float mlp_forward(const float* input) {
    update_stack_measure(); // Measure stack at entry
    float output;
    int H = s_hidden_size;
    
    const float* W_fc1;
    const float* B_fc1;
    const float* W_fc2;
    const float* B_fc2;
    
    switch (s_current_model) {
        case MODEL_BASE:
            W_fc1 = BASE_MLP_FC1_WEIGHT;
            B_fc1 = BASE_MLP_FC1_BIAS;
            W_fc2 = BASE_MLP_FC2_WEIGHT;
            B_fc2 = BASE_MLP_FC2_BIAS;
            break;
        case MODEL_PRUNED:
            W_fc1 = PRUNED_MLP_FC1_WEIGHT;
            B_fc1 = PRUNED_MLP_FC1_BIAS;
            W_fc2 = PRUNED_MLP_FC2_WEIGHT;
            B_fc2 = PRUNED_MLP_FC2_BIAS;
            break;
        case MODEL_QUANTIZED:
            W_fc1 = QUANT_MLP_FC1_WEIGHT;
            B_fc1 = QUANT_MLP_FC1_BIAS;
            W_fc2 = QUANT_MLP_FC2_WEIGHT;
            B_fc2 = QUANT_MLP_FC2_BIAS;
            break;
        default:
            return 0.0f;
    }
    
    /* First layer: hidden = ReLU(W1 * input + b1) */
    mat_vec_mul(W_fc1, input, B_fc1, g_mlp_hidden, MLP_HIDDEN, H);
    
    /* Apply ReLU */
    for (int i = 0; i < MLP_HIDDEN; i++) {
        g_mlp_hidden[i] = relu(g_mlp_hidden[i]);
    }
    
    /* Second layer: output = W2 * hidden + b2 */
    mat_vec_mul(W_fc2, g_mlp_hidden, B_fc2, &output, 1, MLP_HIDDEN);
    
    /* Apply Sigmoid - REMOVED for SOH (Linear Output) */
    // output = sigmoid(output);
    
    return output;
}

/**
 * Full model inference: LSTM + MLP
 */
void lstm_model_inference(LSTMModel* model, const float* input, float* output) {
    /* Initialize if needed */
    if (!model->initialized) {
        lstm_model_init(model);
    }
    
    /* LSTM forward pass (updates state) */
    lstm_cell_forward(input, &model->state);
    
    /* MLP forward pass (uses updated hidden state) */
    *output = mlp_forward(model->state.h);
}

