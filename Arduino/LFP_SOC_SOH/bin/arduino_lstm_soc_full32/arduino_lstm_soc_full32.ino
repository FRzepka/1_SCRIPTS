/*
ARDUINO LSTM SOC PREDICTOR - VOLLSTÄNDIGE 32 HIDDEN UNITS
IDENTISCH zu live_test_soc.py - KEINE Komprimierung!
*/

#include "lstm_weights_exact.h"
#include <math.h>

// Vollständige Konstanten wie live_test_soc.py
#define INPUT_SIZE 4
#define HIDDEN_SIZE 32    // VOLLSTÄNDIGE 32 wie PyTorch!
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

// Eingabepuffer
char inputBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// Aktivierungsfunktionen
float sigmoid_activation(float x) {
  return 1.0f / (1.0f + exp(-x));
}

float tanh_activation(float x) {
  return tanh(x);
}

float relu_activation(float x) {
  return fmax(0.0f, x);
}

// VOLLSTÄNDIGE LSTM Forward Pass (32 Hidden Units)
float predictSOC(float voltage, float current, float soh, float q_c) {
  float input[INPUT_SIZE] = {voltage, current, soh, q_c};
  
  // ==================== LSTM FORWARD PASS ====================
  // EXAKT wie PyTorch mit VOLLSTÄNDIGEN 32 Hidden Units!
  
  // 1. Input Gate: i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = lstm_input_ih_bias[i] + lstm_input_hh_bias[i];
    
    // Input contribution
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += lstm_input_ih_weights[i][j] * input[j];
    }
    
    // Hidden contribution
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += lstm_input_hh_weights[i][j] * h_state[j];
    }
    
    input_gate[i] = sigmoid_activation(sum);
  }
  
  // 2. Forget Gate: f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = lstm_forget_ih_bias[i] + lstm_forget_hh_bias[i];
    
    // Input contribution
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += lstm_forget_ih_weights[i][j] * input[j];
    }
    
    // Hidden contribution
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += lstm_forget_hh_weights[i][j] * h_state[j];
    }
    
    forget_gate[i] = sigmoid_activation(sum);
  }
  
  // 3. Candidate Gate: g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = lstm_candidate_ih_bias[i] + lstm_candidate_hh_bias[i];
    
    // Input contribution
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += lstm_candidate_ih_weights[i][j] * input[j];
    }
    
    // Hidden contribution
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += lstm_candidate_hh_weights[i][j] * h_state[j];
    }
    
    candidate_gate[i] = tanh_activation(sum);
  }
  
  // 4. Output Gate: o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = lstm_output_ih_bias[i] + lstm_output_hh_bias[i];
    
    // Input contribution
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += lstm_output_ih_weights[i][j] * input[j];
    }
    
    // Hidden contribution
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += lstm_output_hh_weights[i][j] * h_state[j];
    }
    
    output_gate[i] = sigmoid_activation(sum);
  }
  
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

void resetLSTMStates() {
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    h_state[i] = 0.0f;
    c_state[i] = 0.0f;
  }
}

void processJSON(String jsonString) {
  // Simple JSON parsing for "V,I,SOH,Q_c" format
  int commaIndex1 = jsonString.indexOf(',');
  int commaIndex2 = jsonString.indexOf(',', commaIndex1 + 1);
  int commaIndex3 = jsonString.indexOf(',', commaIndex2 + 1);
  
  if (commaIndex1 > 0 && commaIndex2 > 0 && commaIndex3 > 0) {
    float voltage = jsonString.substring(0, commaIndex1).toFloat();
    float current = jsonString.substring(commaIndex1 + 1, commaIndex2).toFloat();
    float soh = jsonString.substring(commaIndex2 + 1, commaIndex3).toFloat();
    float q_c = jsonString.substring(commaIndex3 + 1).toFloat();
    
    // SOC Vorhersage
    float soc_pred = predictSOC(voltage, current, soh, q_c);
    
    // Antwort senden
    Serial.print(soc_pred, 6);
    Serial.println();
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  
  // Initialisierung
  resetLSTMStates();
  
  // Start-Info
  printModelInfo();
  Serial.println("🎯 VOLLSTÄNDIGER ARDUINO LSTM SOC PREDICTOR BEREIT!");
  Serial.println("💾 Speicher: 27.38 KB (85.6% von 32KB)");
  Serial.println("🔧 Hidden Units: 32 (vollständig)");
  Serial.println("📊 Parameter: 7009");
  Serial.println("Warte auf Daten...");
}

void loop() {
  if (Serial.available()) {
    char incomingByte = Serial.read();
    
    if (incomingByte == '\n' || incomingByte == '\r') {
      if (bufferIndex > 0) {
        inputBuffer[bufferIndex] = '\0';
        String inputString = String(inputBuffer);
        
        // Spezielle Kommandos
        if (inputString.equals("RESET")) {
          resetLSTMStates();
          Serial.println("LSTM States reset");
        } else if (inputString.equals("INFO")) {
          printModelInfo();
        } else {
          processJSON(inputString);
        }
        
        bufferIndex = 0;
      }
    } else if (bufferIndex < BUFFER_SIZE - 1) {
      inputBuffer[bufferIndex++] = incomingByte;
    }
  }
}

// Debug Funktionen
void printModelInfo() {
  Serial.println("🎯 ARDUINO LSTM MODEL INFO (VOLLSTÄNDIG)");
  Serial.println("========================================");
  Serial.print("Input Size: "); Serial.println(INPUT_SIZE);
  Serial.print("Hidden Size: "); Serial.println(HIDDEN_SIZE);
  Serial.print("Output Size: "); Serial.println(OUTPUT_SIZE);
  Serial.println();
  Serial.println("Architecture: LSTM(4→32) + MLP(32→32→32→1)");
  Serial.println("Activation: Sigmoid gates, Tanh candidate, ReLU MLP, Sigmoid output");
  Serial.println("Weights: Loaded from best_model.pth (VOLLSTÄNDIG)");
  Serial.println("Memory: 27.38 KB (85.6% von 32KB)");
  Serial.println("Parameters: 7009");
  Serial.println("========================================");
}
