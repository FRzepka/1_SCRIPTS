/**
 * Wrapper for manual LSTM-MLP SOC prediction model
 * Provides simple init, run, and deinit APIs for float inputs/outputs.
 */
#ifndef AI_WRAPPER_H
#define AI_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Contract:
// - Input: 6 float features [Voltage, Current, Temperature, Q_c, dU_dt, dI_dt]
// - Output: 1 float prediction (SOC 0.0 to 1.0)
// - Return 0 on success, negative on error

int ai_wrapper_init(void);
int ai_wrapper_run(const float in[6], float* out);
void ai_wrapper_reset_state(void);  // Reset LSTM hidden states
void ai_wrapper_deinit(void);
int ai_wrapper_get_error(int* type, int* code);

#ifdef __cplusplus
}
#endif

#endif // AI_WRAPPER_H
