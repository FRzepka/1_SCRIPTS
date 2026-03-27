// Wrapper for manual LSTM-MLP SOC prediction model
#include "ai_wrapper.h"
#include "lstm_model.h"
#include "scaler_params.h"

// Static LSTM model instance (holds hidden states h and c)
static LSTMModel s_lstm_model;
static int s_initialized = 0;

// --- SOH Real-Time Filter ---
// Implements EMA + Slew Rate Limiter for "brutal" smoothing
typedef struct {
    float current_ema;
    float last_output;
} SOH_Filter_t;

static SOH_Filter_t s_soh_filter;

static void soh_filter_init(SOH_Filter_t* filter) {
    filter->current_ema = 1.0f; // Start at 100% SOH
    filter->last_output = 1.0f;
}

static float soh_filter_update(SOH_Filter_t* filter, float raw_val) {
    // Parameters tuned via PC Benchmark (simulate_stm32_filter.py)
    // Alpha = 1e-6 (~2M samples smoothing)
    // Max Drop = 2e-8 (Max drop per sample)
    const float alpha = 1.0e-6f;
    const float max_drop = 2.0e-8f;
    
    // 1. EMA Filter
    filter->current_ema = (alpha * raw_val) + ((1.0f - alpha) * filter->current_ema);
    
    // 2. Drop Limiter
    float proposed = filter->current_ema;
    
    // Prevent falling too fast
    if (proposed < filter->last_output - max_drop) {
        proposed = filter->last_output - max_drop;
    }
    
    // Optional: Prevent rising? (SOH usually doesn't rise)
    // For now, we allow rising if the EMA rises (e.g. after calibration), 
    // but the EMA is very slow anyway.
    
    filter->last_output = proposed;
    return proposed;
}
// ----------------------------

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
    
    // Initialize the SOH Filter
    soh_filter_init(&s_soh_filter);
    
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
    float raw_prediction;
    lstm_model_inference(&s_lstm_model, scaled_input, &raw_prediction);
    
    // Apply Real-Time SOH Filter
    *out = soh_filter_update(&s_soh_filter, raw_prediction);

    s_last_error.type = 0;
    s_last_error.code = 0;
    
    return 0;
}

void ai_wrapper_reset_state(void)
{
    if (s_initialized) {
        lstm_model_reset(&s_lstm_model);
        soh_filter_init(&s_soh_filter); // Reset filter to 100%
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
