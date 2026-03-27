/*
ARDUINO LSTM SOC PREDICTOR - LIVE EXACT VERSION WITH JSON I/O
IDENTISCH zu live_test_soc.py - VOLLE 32 Hidden Units!
*/

#include <ArduinoJson.h>
#include "lstm_weights_exact.h"
#include <math.h>

// Exakte Konstanten wie live_test_soc.py
#define INPUT_SIZE 4
#define HIDDEN_SIZE ARDUINO_HIDDEN_SIZE    // Use size from weights header (8)
#define OUTPUT_SIZE 1
#define BUFFER_SIZE 256

// LSTM Hidden States (32 Units!)
float h_state[HIDDEN_SIZE] = {0};
float c_state[HIDDEN_SIZE] = {0};

// Temporäre Arrays für LSTM Berechnungen
float input_gate[HIDDEN_SIZE];
float forget_gate[HIDDEN_SIZE];
float candidate_gate[HIDDEN_SIZE];
float output_gate[HIDDEN_SIZE];
float new_c_state[HIDDEN_SIZE];
float new_h_state[HIDDEN_SIZE];

// MLP Zwischenergebnisse
float mlp_layer0[32];
float mlp_layer3[32];
float mlp_output;

// Timing
unsigned long lastPrediction = 0;
unsigned long predictionCount = 0;

// Function declarations
float predictSOC(float voltage, float current, float soh, float q_c);
void computeGate(float* input, float* hidden, float* gate_output,
                 const float weight_ih[][INPUT_SIZE], const float weight_hh[][32],
                 const float* bias_ih, const float* bias_hh);
void computeCandidateGate(float* input, float* hidden, float* gate_output,
                          const float weight_ih[][INPUT_SIZE], const float weight_hh[][32], 
                          const float* bias_ih, const float* bias_hh);
void resetLSTMState();

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  
  // Initialisierung
  resetLSTMState();
  
  // JSON ready message
  StaticJsonDocument<128> readyMsg;
  readyMsg["status"] = "ready";
  readyMsg["model"] = "LSTM_live_exact";
  readyMsg["hidden_size"] = HIDDEN_SIZE;
  readyMsg["parameters"] = 6881;
  serializeJson(readyMsg, Serial);
  Serial.println();
}

void processInput(char* input) {
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, input);
  if (err) {
    StaticJsonDocument<64> resp;
    resp["status"] = "error";
    resp["message"] = "Invalid JSON";
    serializeJson(resp, Serial);
    Serial.println();
    return;
  }
  float voltage = doc["v"];
  float current = doc["i"];
  float soh     = doc["s"];
  float q_c     = doc["q"];
  float true_soc = doc["t"];

  unsigned long startTime = micros();
  float soc_prediction = predictSOC(voltage, current, soh, q_c);
  unsigned long inferenceTime = micros() - startTime;

  StaticJsonDocument<256> resp;
  resp["pred_soc"] = soc_prediction;
  resp["true_soc"] = true_soc;
  resp["voltage"] = voltage;
  resp["current"] = current;
  resp["inference_time_us"] = inferenceTime;
  resp["inference_time_ms"] = inferenceTime / 1000.0;
  resp["model_type"] = "LSTM";
  resp["status"] = "OK";
  serializeJson(resp, Serial);
  Serial.println();
}

float predictSOC(float voltage, float current, float soh, float q_c) {
  // Input Array (wie PyTorch: [voltage, current, soh, q_c])
  float input[INPUT_SIZE] = {voltage, current, soh, q_c};
  
  // ==================== LSTM FORWARD PASS ====================
  // EXAKT wie PyTorch LSTM Implementation!
    // 1. Input Gate: i_t = σ(W_ii * x_t + b_ii + W_hi * h_t + b_hi)
  computeGate(input, h_state, input_gate, 
              lstm_input_ih_weights, lstm_input_hh_weights, 
              lstm_input_bias, lstm_input_bias);
  
  // 2. Forget Gate: f_t = σ(W_if * x_t + b_if + W_hf * h_t + b_hf)  
  computeGate(input, h_state, forget_gate,
              lstm_forget_ih_weights, lstm_forget_hh_weights,
              lstm_forget_bias, lstm_forget_bias);
  
  // 3. Candidate Gate: g_t = tanh(W_ig * x_t + b_ig + W_hg * h_t + b_hg)
  computeCandidateGate(input, h_state, candidate_gate,
                       lstm_candidate_ih_weights, lstm_candidate_hh_weights,
                       lstm_candidate_bias, lstm_candidate_bias);
  
  // 4. Output Gate: o_t = σ(W_io * x_t + b_io + W_ho * h_t + b_ho)
  computeGate(input, h_state, output_gate,
              lstm_output_ih_weights, lstm_output_hh_weights,
              lstm_output_bias, lstm_output_bias);
  
  // 5. Cell State Update: C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    new_c_state[i] = forget_gate[i] * c_state[i] + input_gate[i] * candidate_gate[i];
  }
  
  // 6. Hidden State Update: h_t = o_t ⊙ tanh(C_t)
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    new_h_state[i] = output_gate[i] * tanh(new_c_state[i]);
  }
  
  // Update States für nächsten Zeitschritt
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    h_state[i] = new_h_state[i];
    c_state[i] = new_c_state[i];
  }
  
  // ==================== MLP FORWARD PASS ====================
  // EXAKT wie PyTorch MLP Implementation!
    // Layer 0: Linear(32, 32) + ReLU + Dropout
  for (int i = 0; i < 32; i++) {
    float sum = mlp0_bias[i];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += mlp0_weights[i][j] * h_state[j];
    }
    mlp_layer0[i] = fmax(0.0f, sum); // ReLU
  }
  
  // Layer 3: Linear(32, 32) + ReLU + Dropout  
  for (int i = 0; i < 32; i++) {
    float sum = mlp3_bias[i];
    for (int j = 0; j < 32; j++) {
      sum += mlp3_weights[i][j] * mlp_layer0[j];
    }
    mlp_layer3[i] = fmax(0.0f, sum); // ReLU
  }
  
  // Layer 6: Linear(32, 1) + Sigmoid
  float sum = mlp6_bias[0];
  for (int j = 0; j < 32; j++) {
    sum += mlp6_weights[0][j] * mlp_layer3[j];
  }
  mlp_output = 1.0f / (1.0f + exp(-sum)); // Sigmoid
  
  return mlp_output;
}

void computeGate(float* input, float* hidden, float* gate_output,
                 const float weight_ih[][INPUT_SIZE], const float weight_hh[][32],
                 const float* bias_ih, const float* bias_hh) {
  
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = bias_ih[i] + bias_hh[i];
    
    // Input contribution: W_ih * x
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += weight_ih[i][j] * input[j];
    }
    
    // Hidden contribution: W_hh * h  
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += weight_hh[i][j] * hidden[j];
    }
    
    // Sigmoid activation
    gate_output[i] = 1.0f / (1.0f + exp(-sum));
  }
}

void computeCandidateGate(float* input, float* hidden, float* gate_output,
                          const float weight_ih[][INPUT_SIZE], const float weight_hh[][32], 
                          const float* bias_ih, const float* bias_hh) {
  
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = bias_ih[i] + bias_hh[i];
    
    // Input contribution
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += weight_ih[i][j] * input[j];
    }
    
    // Hidden contribution
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += weight_hh[i][j] * hidden[j];
    }
    
    // Tanh activation
    gate_output[i] = tanh(sum);
  }
}

void resetLSTMState() {
  // Initialisiere Hidden und Cell States auf 0 (wie PyTorch)
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    h_state[i] = 0.0f;
    c_state[i] = 0.0f;
  }
  predictionCount = 0;
}

void loop() {
  static const int bufferSize = 128;
  static char buffer[bufferSize];
  static size_t idx = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || idx >= bufferSize-1) {
      buffer[idx] = '\0';
      processInput(buffer);
      idx = 0;
    } else {
      buffer[idx++] = c;
    }
  }
}
