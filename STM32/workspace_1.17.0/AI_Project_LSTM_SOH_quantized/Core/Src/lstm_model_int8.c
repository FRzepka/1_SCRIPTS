/**
 ******************************************************************************
 * @file    lstm_model_int8.c
 * @brief   Manual INT8 (weights-only) LSTM + FP32 MLP inference
 * @note    Matches the PC reference produced by manual_lstm_int8_from_pt.py
 ******************************************************************************
 */

#include "lstm_model_int8.h"
#include "scaler_params.h"
// Manual INT8 LSTM weights (per-row scales) + FP32 MLP weights
// Copy these headers into Core/Inc via copy_manual_int8_headers.ps1
#include "model_weights_lstm_int8_manual.h"
#include "mlp_weights_fp32.h"
#include <string.h>
#include <math.h>

// Helper: sigmoid
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// Predict one step using manual INT8 LSTM weights (per-row scales) and FP32 MLP
void lstm_model_predict_int8(
    const float *input,
    float *h_state,
    float *c_state,
    float *prediction
) {
    // 1) Scale input like on PC
    float x_scaled[INPUT_SIZE];
    scaler_transform(input, x_scaled);

    // 2) LSTM gates (i,f,g,o) using INT8 weights + per-row scales; FP32 accum
    const int gate_size = HIDDEN_SIZE * 4;
    float gates[HIDDEN_SIZE * 4];
    for (int ch = 0; ch < gate_size; ++ch) {
        float acc = 0.0f;
        const float s_ih = LSTM_W_IH_SCALE[ch];
        const float s_hh = LSTM_W_HH_SCALE[ch];
        // x @ W_ih.T
        for (int j = 0; j < INPUT_SIZE; ++j)
            acc += x_scaled[j] * ((float)LSTM_W_IH[ch][j]) * s_ih;
        // h @ W_hh.T
        for (int k = 0; k < HIDDEN_SIZE; ++k)
            acc += h_state[k] * ((float)LSTM_W_HH[ch][k]) * s_hh;
        // + bias
        acc += LSTM_B[ch];
        gates[ch] = acc;
    }

    // 3) Split + activations
    float i_gate[HIDDEN_SIZE], f_gate[HIDDEN_SIZE], g_gate[HIDDEN_SIZE], o_gate[HIDDEN_SIZE];
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        i_gate[h] = sigmoidf(gates[0*HIDDEN_SIZE + h]);
        f_gate[h] = sigmoidf(gates[1*HIDDEN_SIZE + h]);
        g_gate[h] = tanhf   (gates[2*HIDDEN_SIZE + h]);
        o_gate[h] = sigmoidf(gates[3*HIDDEN_SIZE + h]);
    }

    // 4) Update c,h (float math)
    for (int h = 0; h < HIDDEN_SIZE; ++h)
        c_state[h] = f_gate[h] * c_state[h] + i_gate[h] * g_gate[h];
    for (int h = 0; h < HIDDEN_SIZE; ++h)
        h_state[h] = o_gate[h] * tanhf(c_state[h]);

    // 5) FP32 MLP head: [64]->[64](ReLU)->[1](Sigmoid)
    float x0[MLP0_OUT_DIM];
    for (int o = 0; o < MLP0_OUT_DIM; ++o) {
        float sum = MLP0_BIAS[o];
        for (int i = 0; i < MLP0_IN_DIM; ++i) sum += h_state[i] * MLP0_WEIGHT[o][i];
        x0[o] = (sum > 0.0f) ? sum : 0.0f;
    }
    float y_lin = MLP1_BIAS[0];
    for (int i = 0; i < MLP1_IN_DIM; ++i) y_lin += x0[i] * MLP1_WEIGHT[0][i];
    *prediction = sigmoidf(y_lin);
}

// Zero states
void lstm_model_reset_states(float *h_state, float *c_state) {
    memset(h_state, 0, HIDDEN_SIZE * sizeof(float));
    memset(c_state, 0, HIDDEN_SIZE * sizeof(float));
}

