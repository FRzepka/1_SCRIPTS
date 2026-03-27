/*
 * LSTM-MLP Model Implementation in C
 * Single-step stateful inference for SOC prediction
 */

#include "lstm_model.h"
#include "model_weights.h"
#include <string.h>

/* ========== Helper Functions ========== */

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
    memset(model->state.h, 0, HIDDEN_SIZE * sizeof(float));
    memset(model->state.c, 0, HIDDEN_SIZE * sizeof(float));
    model->initialized = 1;
}

void lstm_model_reset(LSTMModel* model) {
    lstm_model_init(model);
}

/**
 * LSTM Cell Forward Pass
 * 
 * Implements LSTM equations:
 * i_t = sigmoid(W_ii * x_t + W_hi * h_{t-1} + b_i)  [input gate]
 * f_t = sigmoid(W_if * x_t + W_hf * h_{t-1} + b_f)  [forget gate]
 * g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_g)     [cell gate]
 * o_t = sigmoid(W_io * x_t + W_ho * h_{t-1} + b_o)  [output gate]
 * c_t = f_t * c_{t-1} + i_t * g_t                   [cell state]
 * h_t = o_t * tanh(c_t)                              [hidden state]
 */
void lstm_cell_forward(const float* input, LSTMState* state) {
    float gates[4 * HIDDEN_SIZE];  /* i, f, g, o gates concatenated */
    float i_gate[HIDDEN_SIZE];
    float f_gate[HIDDEN_SIZE];
    float g_gate[HIDDEN_SIZE];
    float o_gate[HIDDEN_SIZE];
    
    /* Compute all gates: gates = W_ih * input + W_hh * h + bias */
    /* Note: LSTM weights are stored as [4*H, I] and [4*H, H] */
    
    /* Input contribution: W_ih * input */
    mat_vec_mul(LSTM_WEIGHT_IH, input, LSTM_BIAS, gates, 4 * HIDDEN_SIZE, INPUT_SIZE);
    
    /* Hidden contribution: W_hh * h (add to gates) */
    float h_contrib[4 * HIDDEN_SIZE];
    mat_vec_mul(LSTM_WEIGHT_HH, state->h, NULL, h_contrib, 4 * HIDDEN_SIZE, HIDDEN_SIZE);
    vec_add(gates, h_contrib, 4 * HIDDEN_SIZE);
    
    /* Split gates and apply activations */
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        i_gate[i] = sigmoid(gates[i]);                           /* Input gate */
        f_gate[i] = sigmoid(gates[HIDDEN_SIZE + i]);             /* Forget gate */
        g_gate[i] = tanh_activation(gates[2 * HIDDEN_SIZE + i]); /* Cell gate */
        o_gate[i] = sigmoid(gates[3 * HIDDEN_SIZE + i]);         /* Output gate */
    }
    
    /* Update cell state: c_t = f_t * c_{t-1} + i_t * g_t */
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        state->c[i] = f_gate[i] * state->c[i] + i_gate[i] * g_gate[i];
    }
    
    /* Update hidden state: h_t = o_t * tanh(c_t) */
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        state->h[i] = o_gate[i] * tanh_activation(state->c[i]);
    }
}

/**
 * MLP Forward Pass
 * 
 * Two-layer MLP with ReLU activation and Sigmoid output:
 * hidden = ReLU(FC1(input))
 * output = Sigmoid(FC2(hidden))
 */
float mlp_forward(const float* input) {
    float hidden[MLP_HIDDEN];
    float output;
    
    /* First layer: hidden = ReLU(W1 * input + b1) */
    mat_vec_mul(MLP_FC1_WEIGHT, input, MLP_FC1_BIAS, hidden, MLP_HIDDEN, HIDDEN_SIZE);
    
    /* Apply ReLU */
    for (int i = 0; i < MLP_HIDDEN; i++) {
        hidden[i] = relu(hidden[i]);
    }
    
    /* Second layer: output = W2 * hidden + b2 */
    mat_vec_mul(MLP_FC2_WEIGHT, hidden, MLP_FC2_BIAS, &output, 1, MLP_HIDDEN);
    
    /* Apply Sigmoid */
    output = sigmoid(output);
    
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


