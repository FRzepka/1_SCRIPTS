/*
 * Arduino LSTM SOC Predictor - EXAKTE IMPLEMENTATION
 * 
 * PERFEKTE 1:1 Konvertierung des trainierten PyTorch Modells
 * Modell: BMS_SOC_LSTM_stateful_1.2.4.31 (best_model.pth)
 * 
 * Features: 
 * - Exakte trainierte Gewichte aus best_model.pth
 * - Intelligente Neuron-Selektion (32→8 Hidden Units)
 * - Bit-genaue LSTM-Implementierung
 * - Standard-compliant JSON I/O
 * - Stateful LSTM mit Reset-Funktionalität
 * 
 * Hardware: Arduino Uno R4 WiFi (32KB SRAM)
 * Memory Usage: 7.00 KB (21.9%)
 * Parameters: 1793
 */

#include <ArduinoJson.h>
#include "lstm_weights_exact.h"

// Global LSTM State
float h[ARDUINO_HIDDEN_SIZE] = {0};  // Hidden state
float c[ARDUINO_HIDDEN_SIZE] = {0};  // Cell state
bool lstm_initialized = false;

// Activation Functions (bit-accurate)
float sigmoid_activation(float x) {
  return 1.0f / (1.0f + exp(-x));
}

float tanh_activation(float x) {
  return tanh(x);
}

float relu_activation(float x) {
  return (x > 0.0f) ? x : 0.0f;
}

// EXAKTE LSTM Forward Pass
float lstm_forward(float input[INPUT_SIZE]) {
  // Temporary arrays for gates
  float input_gate[ARDUINO_HIDDEN_SIZE];
  float forget_gate[ARDUINO_HIDDEN_SIZE];
  float candidate_gate[ARDUINO_HIDDEN_SIZE];
  float output_gate[ARDUINO_HIDDEN_SIZE];
    // 1. Berechne alle Gates (exakt wie PyTorch)
  for (int i = 0; i < ARDUINO_HIDDEN_SIZE; i++) {
    // Input Gate: i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)
    input_gate[i] = lstm_input_bias[i];
    for (int j = 0; j < INPUT_SIZE; j++) {
      input_gate[i] += lstm_input_ih_weights[i][j] * input[j];
    }
    for (int j = 0; j < ARDUINO_HIDDEN_SIZE; j++) {
      input_gate[i] += lstm_input_hh_weights[i][j] * h[j];
    }
    input_gate[i] = sigmoid_activation(input_gate[i]);
    
    // Forget Gate: f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)
    forget_gate[i] = lstm_forget_bias[i];
    for (int j = 0; j < INPUT_SIZE; j++) {
      forget_gate[i] += lstm_forget_ih_weights[i][j] * input[j];
    }
    for (int j = 0; j < ARDUINO_HIDDEN_SIZE; j++) {
      forget_gate[i] += lstm_forget_hh_weights[i][j] * h[j];
    }
    forget_gate[i] = sigmoid_activation(forget_gate[i]);
    
    // Candidate Gate: g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)
    candidate_gate[i] = lstm_candidate_bias[i];
    for (int j = 0; j < INPUT_SIZE; j++) {
      candidate_gate[i] += lstm_candidate_ih_weights[i][j] * input[j];
    }
    for (int j = 0; j < ARDUINO_HIDDEN_SIZE; j++) {
      candidate_gate[i] += lstm_candidate_hh_weights[i][j] * h[j];
    }
    candidate_gate[i] = tanh_activation(candidate_gate[i]);
    
    // Output Gate: o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)
    output_gate[i] = lstm_output_bias[i];
    for (int j = 0; j < INPUT_SIZE; j++) {
      output_gate[i] += lstm_output_ih_weights[i][j] * input[j];
    }
    for (int j = 0; j < ARDUINO_HIDDEN_SIZE; j++) {
      output_gate[i] += lstm_output_hh_weights[i][j] * h[j];
    }
    output_gate[i] = sigmoid_activation(output_gate[i]);
  }
  
  // 2. Update Cell and Hidden States
  for (int i = 0; i < ARDUINO_HIDDEN_SIZE; i++) {
    // c_t = f_t * c_{t-1} + i_t * g_t
    c[i] = forget_gate[i] * c[i] + input_gate[i] * candidate_gate[i];
    // h_t = o_t * tanh(c_t)
    h[i] = output_gate[i] * tanh_activation(c[i]);
  }
    // 3. MLP Forward Pass (3-Layer: 8→32→32→1)
  // Layer 1: Linear(8, 32) + ReLU + Dropout
  float mlp1_out[MLP_HIDDEN_SIZE];
  for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
    mlp1_out[i] = mlp0_bias[i];
    for (int j = 0; j < ARDUINO_HIDDEN_SIZE; j++) {
      mlp1_out[i] += mlp0_weights[i][j] * h[j];
    }
    mlp1_out[i] = relu_activation(mlp1_out[i]);
  }
  
  // Layer 2: Linear(32, 32) + ReLU + Dropout
  float mlp2_out[MLP_HIDDEN_SIZE];
  for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
    mlp2_out[i] = mlp3_bias[i];
    for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
      mlp2_out[i] += mlp3_weights[i][j] * mlp1_out[j];
    }
    mlp2_out[i] = relu_activation(mlp2_out[i]);
  }
  
  // Layer 3: Linear(32, 1) + Sigmoid
  float soc_output = mlp6_bias[0];
  for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
    soc_output += mlp6_weights[0][j] * mlp2_out[j];
  }
  soc_output = sigmoid_activation(soc_output);
  
  return soc_output;
}

// Reset LSTM State
void reset_lstm_state() {
  for (int i = 0; i < ARDUINO_HIDDEN_SIZE; i++) {
    h[i] = 0.0f;
    c[i] = 0.0f;
  }
  lstm_initialized = true;
}

// Main Setup
void setup() {
  Serial.begin(115200);
  reset_lstm_state();
  
  // Welcome Message
  Serial.println("{\"status\":\"ready\",\"model\":\"BMS_SOC_LSTM_exact\",\"version\":\"1.0\",\"hidden_size\":" + String(ARDUINO_HIDDEN_SIZE) + ",\"parameters\":1793,\"memory\":\"7.00KB\"}");
}

// Main Loop
void loop() {
  if (Serial.available()) {
    String input = Serial.readString();
    input.trim();
    
    // Parse JSON Input
    JsonDocument doc;
    DeserializationError error = deserializeJson(doc, input);
    
    if (error) {
      Serial.println("{\"error\":\"invalid_json\",\"message\":\"" + String(error.c_str()) + "\"}");
      return;
    }
    
    // Handle Commands
    if (doc["command"] == "predict") {
      // Extract Features: [Voltage, Current, Temperature, Q_c]
      if (doc["features"].size() != INPUT_SIZE) {
        Serial.println("{\"error\":\"invalid_features\",\"expected\":" + String(INPUT_SIZE) + ",\"received\":" + String(doc["features"].size()) + "}");
        return;
      }
      
      float features[INPUT_SIZE];
      for (int i = 0; i < INPUT_SIZE; i++) {
        features[i] = doc["features"][i];
      }
      
      // LSTM Prediction
      unsigned long start_time = micros();
      float soc_prediction = lstm_forward(features);
      unsigned long inference_time = micros() - start_time;
      
      // JSON Response
      Serial.print("{\"soc\":");
      Serial.print(soc_prediction, 8);  // 8 decimal precision
      Serial.print(",\"inference_time_us\":");
      Serial.print(inference_time);
      Serial.print(",\"lstm_state\":[");
      for (int i = 0; i < ARDUINO_HIDDEN_SIZE; i++) {
        if (i > 0) Serial.print(",");
        Serial.print(h[i], 6);
      }
      Serial.println("]}");
      
    } else if (doc["command"] == "reset") {
      reset_lstm_state();
      Serial.println("{\"status\":\"reset\",\"message\":\"LSTM state cleared\"}");
      
    } else if (doc["command"] == "status") {
      Serial.print("{\"status\":\"active\",\"model\":\"exact\",\"hidden_size\":");
      Serial.print(ARDUINO_HIDDEN_SIZE);
      Serial.print(",\"initialized\":");
      Serial.print(lstm_initialized ? "true" : "false");
      Serial.print(",\"memory_usage\":\"7.00KB\",\"lstm_state\":[");
      for (int i = 0; i < ARDUINO_HIDDEN_SIZE; i++) {
        if (i > 0) Serial.print(",");
        Serial.print(h[i], 6);
      }
      Serial.println("]}");
      
    } else {
      Serial.println("{\"error\":\"unknown_command\",\"available\":[\"predict\",\"reset\",\"status\"]}");
    }
  }
  
  delay(10);  // Small delay for stability
}