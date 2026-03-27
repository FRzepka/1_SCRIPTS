#ifndef LSTM_MODEL_LSTM_INT8_FP32MLP_SOH_H
#define LSTM_MODEL_LSTM_INT8_FP32MLP_SOH_H

#include <stdint.h>

typedef struct {
  float h[HIDDEN_SIZE];
  float c[HIDDEN_SIZE];
} LSTMStateSOH;

/* Stack Measurement Globals */
extern volatile uint32_t g_min_stack_ptr;
static inline void lstm_reset_stack_measure(void) { g_min_stack_ptr = 0xFFFFFFFF; }

void lstm_model_soh_int8_init(LSTMStateSOH* state);
float lstm_model_soh_int8_forward(float input[INPUT_SIZE], LSTMStateSOH* state);

#endif /* LSTM_MODEL_LSTM_INT8_FP32MLP_SOH_H */

