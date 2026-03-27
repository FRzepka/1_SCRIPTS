/*
 * lstm_model_hybrid_int8.c
 *
 * Hybrid LSTM Model Implementation: FP32 LSTM + INT8 MLP
 *
 * CRITICAL: ONNX LSTM gate order is [i, o, f, c] NOT PyTorch [i, f, g, o]!
 */

#include "lstm_model_hybrid_int8.h"
#include "model_weights_hybrid_int8.h"
#include <string.h>

// Stack Measurement
extern uint32_t g_min_stack_ptr;

static void update_stack_measure(void) {
    uint32_t sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));
    if (sp < g_min_stack_ptr) {
        g_min_stack_ptr = sp;
    }
}

// Initialize model state
void lstm_model_init(LSTMState *state) {
    memset(state->h, 0, sizeof(state->h));
    memset(state->c, 0, sizeof(state->c));
}

// LSTM cell forward pass (FP32)
static void lstm_cell_forward(const float *x, LSTMState *state) {
    update_stack_measure(); // Measure stack at entry
    float gates[256];  // 4 * HIDDEN_SIZE
    
    // Compute gates: W*x + R*h + B
    // lstm_B already contains Wb + Rb (256 elements)
    for (int i = 0; i < 256; i++) {
        float sum = 0.0f;
        
        // W * x (input projection)
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += lstm_W[i * INPUT_SIZE + j] * x[j];
        }
        
        // R * h (recurrent projection)
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += lstm_R[i * HIDDEN_SIZE + j] * state->h[j];
        }
        
        // Add bias (already summed Wb + Rb)
        sum += lstm_B[i];
        
        gates[i] = sum;
    }
    
    // Apply activation functions
    // CRITICAL: ONNX gate order is [i, o, f, c] NOT [i, f, g, o]!
    float i_gate[HIDDEN_SIZE];  // Input gate
    float o_gate[HIDDEN_SIZE];  // Output gate
    float f_gate[HIDDEN_SIZE];  // Forget gate
    float g_gate[HIDDEN_SIZE];  // Cell gate (new content)
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        i_gate[j] = sigmoid_f32(gates[j]);           // i: 0-63
        o_gate[j] = sigmoid_f32(gates[64 + j]);      // o: 64-127
        f_gate[j] = sigmoid_f32(gates[128 + j]);     // f: 128-191
        g_gate[j] = tanh_f32(gates[192 + j]);        // c: 192-255
    }
    
    // Update cell state: c = f * c_prev + i * g
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        state->c[j] = f_gate[j] * state->c[j] + i_gate[j] * g_gate[j];
    }
    
    // Update hidden state: h = o * tanh(c)
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        state->h[j] = o_gate[j] * tanh_f32(state->c[j]);
    }
}

// MLP forward pass (INT8 quantized)
static float mlp_forward_int8(const float *h) {
    update_stack_measure(); // Measure stack at entry
    // Quantize LSTM hidden state to UINT8
    uint8_t h_q[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float val = h[j] / mlp_input_scale + (float)mlp_input_zero_point;
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        h_q[j] = (uint8_t)(val + 0.5f);
    }
    
    // Layer 0: INT8 matmul [64x64] @ [64] -> [64]
    int32_t acc0[MLP_HIDDEN_SIZE];
    for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
        acc0[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            // INT8 weight × (UINT8 input - zero_point)
            acc0[i] += (int32_t)mlp0_weight_q[i * HIDDEN_SIZE + j] * 
                       ((int32_t)h_q[j] - mlp_input_zero_point);
        }
    }
    
    // Dequantize layer 0: output = (input_scale * weight_scale) * acc + bias
    float hidden[MLP_HIDDEN_SIZE];
    for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
        float out_scale = mlp_input_scale * mlp0_weight_scale[i];
        hidden[i] = out_scale * (float)acc0[i] + 
                    (float)mlp0_bias_q[i] * mlp0_bias_scale[i];
        // ReLU activation
        hidden[i] = relu_f32(hidden[i]);
    }
    
    // Quantize hidden for layer 3 (use ReLU output scale!)
    uint8_t hidden_q[MLP_HIDDEN_SIZE];
    for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
        float val = hidden[j] / mlp0_relu_output_scale + (float)mlp0_relu_output_zero_point;
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        hidden_q[j] = (uint8_t)(val + 0.5f);
    }
    
    // Layer 3: INT8 matmul [1x64] @ [64] -> [1]
    int32_t acc3 = 0;
    for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
        acc3 += (int32_t)mlp3_weight_q[j] * 
                ((int32_t)hidden_q[j] - mlp0_relu_output_zero_point);
    }
    
    // Dequantize layer 3
    float out_scale = mlp0_relu_output_scale * mlp3_weight_scale[0];
    float output = out_scale * (float)acc3 + 
                   (float)mlp3_bias_q[0] * mlp3_bias_scale[0];
    
    // Sigmoid activation
    output = sigmoid_f32(output);
    
    return output;
}

// Main prediction function
float lstm_model_predict(LSTMState *state, const float *input) {
    // 1. LSTM cell forward (FP32)
    lstm_cell_forward(input, state);
    
    // 2. MLP forward (INT8)
    float soc = mlp_forward_int8(state->h);
    
    return soc;
}
