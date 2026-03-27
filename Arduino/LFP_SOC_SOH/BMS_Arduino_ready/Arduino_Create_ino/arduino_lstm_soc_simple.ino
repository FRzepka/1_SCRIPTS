/*
 * Arduino LSTM SOC Prediction - EINFACH & FUNKTIONIEREND
 * 
 * Verwendet die Gewichte aus lstm_weights_exact.h
 * Einfache, manuelle Implementierung ohne überkomplizierte Generierung
 */

#include "lstm_weights_exact.h"
#include <ArduinoJson.h>

// LSTM State Arrays - 32 Hidden Units
float h_state[32];
float c_state[32];

// Temporary arrays für LSTM gates
float input_gate[32];
float forget_gate[32];
float candidate_gate[32];
float output_gate[32];

// MLP intermediate arrays
float mlp_layer1[32];
float mlp_layer2[32];

// Input scaling parameters - EXAKTE Werte aus dem Training!
// Extrahiert mit extract_scaler_params.py - StandardScaler 1:1
float voltage_mean = 3.314710f;
float voltage_std = 0.152665f;
float current_mean = -0.000235f;
float current_std = 1.521030f;
float soh_mean = 0.925386f;
float soh_std = 0.059232f;
float qc_mean = -0.584225f;
float qc_std = 0.395014f;

void setup() {
  Serial.begin(115200);
  
  // Initialize LSTM states to zero
  for (int i = 0; i < 32; i++) {
    h_state[i] = 0.0;
    c_state[i] = 0.0;
  }
  
  Serial.println("Arduino LSTM SOC ready!");
  Serial.println("Send: {\"voltage\":3.7,\"current\":0.5,\"soh\":0.85,\"qc\":2800}");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    // Parse JSON
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, input);
    
    if (error) {
      Serial.println("{\"error\":\"Invalid JSON\"}");
      return;
    }
    
    // Extract values
    float voltage = doc["voltage"];
    float current = doc["current"];
    float soh = doc["soh"];
    float qc = doc["qc"];
    
    // Scale inputs (StandardScaler)
    voltage = (voltage - voltage_mean) / voltage_std;
    current = (current - current_mean) / current_std;
    soh = (soh - soh_mean) / soh_std;
    qc = (qc - qc_mean) / qc_std;
    
    // LSTM Forward Pass
    unsigned long start_time = micros();
    float soc_prediction = lstm_forward(voltage, current, soh, qc);
    unsigned long inference_time = micros() - start_time;
    
    // Send result
    Serial.print("{\"soc\":");
    Serial.print(soc_prediction, 6);
    Serial.print(",\"inference_time_us\":");
    Serial.print(inference_time);
    Serial.println("}");
  }
}

float lstm_forward(float voltage, float current, float soh, float qc) {
  // Input vector
  float input[4] = {voltage, current, soh, qc};
  
  // LSTM Cell Computation für alle 32 Hidden Units
  for (int h = 0; h < 32; h++) {
    
    // Input Gate
    float i_gate = 0.0;
    for (int i = 0; i < 4; i++) {
      i_gate += input[i] * lstm_input_ih_weights[h][i];
    }
    for (int i = 0; i < 32; i++) {
      i_gate += h_state[i] * lstm_input_hh_weights[h][i];
    }
    i_gate += lstm_input_ih_bias[h] + lstm_input_hh_bias[h];
    input_gate[h] = sigmoid(i_gate);
    
    // Forget Gate
    float f_gate = 0.0;
    for (int i = 0; i < 4; i++) {
      f_gate += input[i] * lstm_forget_ih_weights[h][i];
    }
    for (int i = 0; i < 32; i++) {
      f_gate += h_state[i] * lstm_forget_hh_weights[h][i];
    }
    f_gate += lstm_forget_ih_bias[h] + lstm_forget_hh_bias[h];
    forget_gate[h] = sigmoid(f_gate);
    
    // Candidate Gate
    float g_gate = 0.0;
    for (int i = 0; i < 4; i++) {
      g_gate += input[i] * lstm_candidate_ih_weights[h][i];
    }
    for (int i = 0; i < 32; i++) {
      g_gate += h_state[i] * lstm_candidate_hh_weights[h][i];
    }
    g_gate += lstm_candidate_ih_bias[h] + lstm_candidate_hh_bias[h];
    candidate_gate[h] = tanh_func(g_gate);
    
    // Output Gate
    float o_gate = 0.0;
    for (int i = 0; i < 4; i++) {
      o_gate += input[i] * lstm_output_ih_weights[h][i];
    }
    for (int i = 0; i < 32; i++) {
      o_gate += h_state[i] * lstm_output_hh_weights[h][i];
    }
    o_gate += lstm_output_ih_bias[h] + lstm_output_hh_bias[h];
    output_gate[h] = sigmoid(o_gate);
    
    // Update cell and hidden state
    c_state[h] = forget_gate[h] * c_state[h] + input_gate[h] * candidate_gate[h];
    h_state[h] = output_gate[h] * tanh_func(c_state[h]);
  }
  
  // MLP Forward Pass
  // Layer 1: 32 -> 32
  for (int i = 0; i < 32; i++) {
    mlp_layer1[i] = 0.0;
    for (int j = 0; j < 32; j++) {
      mlp_layer1[i] += h_state[j] * mlp0_weights[i][j];
    }
    mlp_layer1[i] += mlp0_bias[i];
    mlp_layer1[i] = relu(mlp_layer1[i]);
  }
  
  // Layer 2: 32 -> 32
  for (int i = 0; i < 32; i++) {
    mlp_layer2[i] = 0.0;
    for (int j = 0; j < 32; j++) {
      mlp_layer2[i] += mlp_layer1[j] * mlp3_weights[i][j];
    }
    mlp_layer2[i] += mlp3_bias[i];
    mlp_layer2[i] = relu(mlp_layer2[i]);
  }
  
  // Final layer: 32 -> 1
  float output = 0.0;
  for (int j = 0; j < 32; j++) {
    output += mlp_layer2[j] * mlp6_weights[0][j];
  }
  output += mlp6_bias[0];
  output = sigmoid(output);
  
  return output;
}

// Activation functions
float sigmoid(float x) {
  if (x > 10) return 1.0;
  if (x < -10) return 0.0;
  return 1.0 / (1.0 + exp(-x));
}

float tanh_func(float x) {
  if (x > 10) return 1.0;
  if (x < -10) return -1.0;
  return tanh(x);
}

float relu(float x) {
  return (x > 0) ? x : 0.0;
}
