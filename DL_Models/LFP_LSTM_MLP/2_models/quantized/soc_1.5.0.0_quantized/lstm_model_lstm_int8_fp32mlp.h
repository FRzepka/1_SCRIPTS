#ifndef LSTM_MODEL_LSTM_INT8_FP32MLP_H
#define LSTM_MODEL_LSTM_INT8_FP32MLP_H

#include <stdint.h>

typedef struct {
  float h[HIDDEN_SIZE];
  float c[HIDDEN_SIZE];
} LSTMState;

void lstm_model_init(LSTMState* state);
float lstm_model_forward(float input[INPUT_SIZE], LSTMState* state);

#endif

