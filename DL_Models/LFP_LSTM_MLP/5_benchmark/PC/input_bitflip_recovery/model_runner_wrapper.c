#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#if defined(RUNNER_SOC_FP32)

#include "lstm_model.h"
#include "scaler_params.h"
typedef LSTMModel RunnerState;

static void model_init(RunnerState *state) {
    lstm_model_init(state);
}

static float model_step(RunnerState *state, const float *raw_input) {
    float scaled[INPUT_SIZE];
    float output = 0.0f;
    scaler_transform(raw_input, scaled);
    lstm_model_inference(state, scaled, &output);
    return output;
}

#elif defined(RUNNER_SOC_INT8)

#include "lstm_model_lstm_int8_fp32mlp.h"
typedef LSTMState RunnerState;

static void model_init(RunnerState *state) {
    lstm_model_init(state);
}

static float model_step(RunnerState *state, const float *raw_input) {
    float input_copy[INPUT_SIZE];
    memcpy(input_copy, raw_input, sizeof(input_copy));
    return lstm_model_forward(input_copy, state);
}

#elif defined(RUNNER_SOH_FP32)

#include "lstm_model_soh.h"
typedef LSTMModelSOH RunnerState;

static void model_init(RunnerState *state) {
    lstm_model_soh_init(state);
}

static float model_step(RunnerState *state, const float *raw_input) {
    float output = 0.0f;
    lstm_model_soh_inference(state, raw_input, &output);
    return output;
}

#elif defined(RUNNER_SOH_INT8)

#include "lstm_model_lstm_int8_fp32mlp_soh.h"
typedef LSTMStateSOH RunnerState;

static void model_init(RunnerState *state) {
    lstm_model_soh_int8_init(state);
}

static float model_step(RunnerState *state, const float *raw_input) {
    float input_copy[INPUT_SIZE];
    memcpy(input_copy, raw_input, sizeof(input_copy));
    return lstm_model_soh_int8_forward(input_copy, state);
}

#else
#error "Select one RUNNER_* model interface"
#endif

EXPORT size_t runner_state_size(void) {
    return sizeof(RunnerState);
}

EXPORT int runner_input_size(void) {
    return INPUT_SIZE;
}

EXPORT int runner_hidden_size(void) {
    return HIDDEN_SIZE;
}

EXPORT void runner_init(void *state) {
    model_init((RunnerState *)state);
}

EXPORT float runner_step(void *state, const float *raw_input) {
    return model_step((RunnerState *)state, raw_input);
}
