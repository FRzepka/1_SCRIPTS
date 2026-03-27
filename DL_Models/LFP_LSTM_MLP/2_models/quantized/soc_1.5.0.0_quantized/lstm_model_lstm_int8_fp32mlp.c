#include <math.h>
#include <string.h>
#include "model_weights_lstm_int8_manual.h"
#include "mlp_weights_fp32.h"
#include "lstm_model_lstm_int8_fp32mlp.h"
#include "scaler_params.h" // provides scaler_transform(float *x)

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

void lstm_model_init(LSTMState* state) {
  memset(state->h, 0, sizeof(state->h));
  memset(state->c, 0, sizeof(state->c));
}

float lstm_model_forward(float input[INPUT_SIZE], LSTMState* state) {
  // Preprocess inputs
  float scaled[INPUT_SIZE];
  scaler_transform(input, scaled);
  for (int i = 0; i < INPUT_SIZE; ++i) {
    input[i] = scaled[i];
  }

  float gates[LSTM_CHANNELS];
  // Compute gates: sum_j x[j] * W_ih[q] * scale + sum_k h[k] * W_hh[q] * scale + bias
  for (int ch = 0; ch < LSTM_CHANNELS; ++ch) {
    float acc = 0.0f;
    const float s_ih = LSTM_W_IH_SCALE[ch];
    const float s_hh = LSTM_W_HH_SCALE[ch];
    for (int j = 0; j < INPUT_SIZE; ++j) {
      acc += input[j] * ((float)LSTM_W_IH[ch][j]) * s_ih;
    }
    for (int k = 0; k < HIDDEN_SIZE; ++k) {
      acc += state->h[k] * ((float)LSTM_W_HH[ch][k]) * s_hh;
    }
    acc += LSTM_B[ch];
    gates[ch] = acc;
  }

  // Split gates [i, f, g, o]
  float i_gate[HIDDEN_SIZE], f_gate[HIDDEN_SIZE], g_gate[HIDDEN_SIZE], o_gate[HIDDEN_SIZE];
  for (int h = 0; h < HIDDEN_SIZE; ++h) {
    i_gate[h] = sigmoidf(gates[0*HIDDEN_SIZE + h]);
    f_gate[h] = sigmoidf(gates[1*HIDDEN_SIZE + h]);
    g_gate[h] = tanhf   (gates[2*HIDDEN_SIZE + h]);
    o_gate[h] = sigmoidf(gates[3*HIDDEN_SIZE + h]);
  }

  // Cell/Hidden update (float math)
  for (int h = 0; h < HIDDEN_SIZE; ++h) {
    state->c[h] = f_gate[h] * state->c[h] + i_gate[h] * g_gate[h];
  }
  for (int h = 0; h < HIDDEN_SIZE; ++h) {
    state->h[h] = o_gate[h] * tanhf(state->c[h]);
  }

  // MLP head (FP32)
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
  float y = sigmoidf(y_lin);
  return y;
}
