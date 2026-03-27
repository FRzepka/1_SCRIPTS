// Wrapper for manual LSTM-MLP SOC prediction model
#include "ai_wrapper.h"
#include "lstm_model.h"
#include "scaler_params.h"

// Static LSTM model instance (holds hidden states h and c)
static LSTMModel s_lstm_model;
static int s_initialized = 0;

// Error tracking
static struct {
    int type;
    int code;
} s_last_error = {0, 0};

int ai_wrapper_init(void)
{
    if (s_initialized) {
        return 0;  // Already initialized
    }

    // Initialize the LSTM model (sets states to zero)
    lstm_model_init(&s_lstm_model);
    
    s_initialized = 1;
    s_last_error.type = 0;
    s_last_error.code = 0;
    
    return 0;
}

int ai_wrapper_run(const float in[6], float* out)
{
    if (!s_initialized) {
        s_last_error.type = 1;
        s_last_error.code = 1;  // Not initialized
        return -1;
    }
    
    if (!in || !out) {
        s_last_error.type = 2;
        s_last_error.code = 2;  // Invalid arguments
        return -2;
    }

    // Apply RobustScaler preprocessing
    float scaled_input[6];
    scaler_transform(in, scaled_input);

    // Run LSTM inference (updates internal states)
    lstm_model_inference(&s_lstm_model, scaled_input, out);

    s_last_error.type = 0;
    s_last_error.code = 0;
    
    return 0;
}

void ai_wrapper_reset_state(void)
{
    if (s_initialized) {
        lstm_model_reset(&s_lstm_model);
    }
}

void ai_wrapper_deinit(void)
{
    if (s_initialized) {
        // Reset states to clean up
        lstm_model_reset(&s_lstm_model);
        s_initialized = 0;
    }
}

int ai_wrapper_get_error(int* type, int* code)
{
    if (!type || !code) {
        return -1;
    }
    
    *type = s_last_error.type;
    *code = s_last_error.code;
    
    return 0;
}
