#include <math.h>
#include <string.h>
#include "model_weights_lstm_int8_manual_soh.h"
#include "mlp_weights_fp32_soh.h"
#include "lstm_model_lstm_int8_fp32mlp_soh.h"
#include "scaler_params_soh.h"

static inline float sigmoidf_fast(float x) { return 1.0f / (1.0f + expf(-x)); }

void lstm_model_soh_int8_init(LSTMStateSOH* state) {
  memset(state->h, 0, sizeof(state->h));
  memset(state->c, 0, sizeof(state->c));
}

float lstm_model_soh_int8_forward(float input[INPUT_SIZE], LSTMStateSOH* state) {
  // Preprocess inputs using scaler header
  float x_scaled[INPUT_SIZE];
  scaler_soh_transform(input, x_scaled);

  float gates[LSTM_CHANNELS];
  // Compute gates using int8 weights + per-row scales
  for (int ch = 0; ch < LSTM_CHANNELS; ++ch) {
    float acc = 0.0f;
    const float s_ih = LSTM_W_IH_SCALE[ch];
    const float s_hh = LSTM_W_HH_SCALE[ch];
    // x * W_ih
    for (int j = 0; j < INPUT_SIZE; ++j) {
      acc += x_scaled[j] * ((float)LSTM_W_IH[ch][j]) * s_ih;
    }
    // h * W_hh
    for (int k = 0; k < HIDDEN_SIZE; ++k) {
      acc += state->h[k] * ((float)LSTM_W_HH[ch][k]) * s_hh;
    }
    acc += LSTM_B[ch];
    gates[ch] = acc;
  }

  // Split gates [i, f, g, o]
  float i_gate[HIDDEN_SIZE], f_gate[HIDDEN_SIZE], g_gate[HIDDEN_SIZE], o_gate[HIDDEN_SIZE];
  for (int h = 0; h < HIDDEN_SIZE; ++h) {
    i_gate[h] = sigmoidf_fast(gates[0*HIDDEN_SIZE + h]);
    f_gate[h] = sigmoidf_fast(gates[1*HIDDEN_SIZE + h]);
    g_gate[h] = tanhf(gates[2*HIDDEN_SIZE + h]);
    o_gate[h] = sigmoidf_fast(gates[3*HIDDEN_SIZE + h]);
  }

  // Cell/Hidden update (float math)
  for (int h = 0; h < HIDDEN_SIZE; ++h) {
    state->c[h] = f_gate[h] * state->c[h] + i_gate[h] * g_gate[h];
  }
  for (int h = 0; h < HIDDEN_SIZE; ++h) {
    state->h[h] = o_gate[h] * tanhf(state->c[h]);
  }

  // MLP head (FP32), linear output for SOH
  float x0[MLP0_OUT_DIM];
  for (int o = 0; o < MLP0_OUT_DIM; ++o) {
    float sum = MLP0_BIAS[o];
    for (int i = 0; i < MLP0_IN_DIM; ++i) {
      sum += state->h[i] * MLP0_WEIGHT[o][i];
    }
    x0[o] = fmaxf(0.0f, sum); // ReLU
  }
  float y_lin = MLP1_BIAS[0];
  for (int i = 0; i < MLP1_IN_DIM; ++i) {
    y_lin += x0[i] * MLP1_WEIGHT[0][i];
  }
  return y_lin; // linear SOH estimate
}
