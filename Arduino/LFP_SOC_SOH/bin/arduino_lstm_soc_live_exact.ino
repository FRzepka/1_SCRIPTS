/*
ARDUINO LSTM SOC PREDICTOR - LIVE EXACT VERSION
IDENTISCH zu live_test_soc.py - VOLLE 32 Hidden Units!
*/

#include "lstm_weights_exact.h"
#include <math.h>

// Exakte Konstanten wie live_test_soc.py
#define INPUT_SIZE 4
#define HIDDEN_SIZE 32    // VOLLE 32 wie PyTorch!
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

// Timing
unsigned long lastPrediction = 0;
unsigned long predictionCount = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  
  // Initialisierung
  resetLSTMState();
  
  Serial.println("🚀 ARDUINO LSTM SOC PREDICTOR - LIVE EXACT");
  Serial.println("🎯 32 Hidden Units - IDENTISCH zu live_test_soc.py");
  Serial.println("📊 6881 Parameters, 26.88 KB Memory");
  Serial.println("✅ Ready for SOC Prediction!");
  Serial.println();
  Serial.println("Input format: voltage,current,soh,q_c");
  Serial.println("Example: 3.2,-2.5,0.95,2.8");
  Serial.println();
}

void loop() {
  // Lese Serial Input
  if (Serial.available()) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (bufferIndex > 0) {
        inputBuffer[bufferIndex] = '\0';
        processInput(inputBuffer);
        bufferIndex = 0;
      }
    } else if (bufferIndex < BUFFER_SIZE - 1) {
      inputBuffer[bufferIndex++] = c;
    }
  }
  
  // Status-Updates alle 5 Sekunden
  if (millis() - lastPrediction > 5000 && predictionCount > 0) {
    printStatus();
    lastPrediction = millis();
  }
}

void processInput(char* input) {
  float voltage, current, soh, q_c;
  
  // Parse Input (CSV format)
  if (sscanf(input, "%f,%f,%f,%f", &voltage, &current, &soh, &q_c) == 4) {
    
    unsigned long startTime = micros();
    
    // LSTM Vorhersage (EXAKT wie PyTorch!)
    float soc_prediction = predictSOC(voltage, current, soh, q_c);
    
    unsigned long inferenceTime = micros() - startTime;
    predictionCount++;
    
    // Ausgabe
    Serial.print("SOC: ");
    Serial.print(soc_prediction, 6);
    Serial.print(" | Time: ");
    Serial.print(inferenceTime);
    Serial.print(" µs | Count: ");
    Serial.println(predictionCount);
    
  } else {
    Serial.println("❌ Invalid input format. Expected: voltage,current,soh,q_c");
  }
}

float predictSOC(float voltage, float current, float soh, float q_c) {
  // Input Array (wie PyTorch: [voltage, current, soh, q_c])
  float input[INPUT_SIZE] = {voltage, current, soh, q_c};
  
  // ==================== LSTM FORWARD PASS ====================
  // EXAKT wie PyTorch LSTM Implementation!
  
  // 1. Input Gate: i_t = σ(W_ii * x_t + b_ii + W_hi * h_t + b_hi)
  computeGate(input, h_state, input_gate, 
              lstm_weight_ih_input, lstm_weight_hh_input, 
              lstm_bias_ih_input, lstm_bias_hh_input);
  
  // 2. Forget Gate: f_t = σ(W_if * x_t + b_if + W_hf * h_t + b_hf)  
  computeGate(input, h_state, forget_gate,
              lstm_weight_ih_forget, lstm_weight_hh_forget,
              lstm_bias_ih_forget, lstm_bias_hh_forget);
  
  // 3. Candidate Gate: g_t = tanh(W_ig * x_t + b_ig + W_hg * h_t + b_hg)
  computeCandidateGate(input, h_state, candidate_gate,
                       lstm_weight_ih_candidate, lstm_weight_hh_candidate,
                       lstm_bias_ih_candidate, lstm_bias_hh_candidate);
  
  // 4. Output Gate: o_t = σ(W_io * x_t + b_io + W_ho * h_t + b_ho)
  computeGate(input, h_state, output_gate,
              lstm_weight_ih_output, lstm_weight_hh_output,
              lstm_bias_ih_output, lstm_bias_hh_output);
  
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
      sum += mlp0_weight[i][j] * h_state[j];
    }
    mlp_layer0[i] = fmax(0.0f, sum); // ReLU
  }
  
  // Layer 3: Linear(32, 32) + ReLU + Dropout  
  for (int i = 0; i < 32; i++) {
    float sum = mlp3_bias[i];
    for (int j = 0; j < 32; j++) {
      sum += mlp3_weight[i][j] * mlp_layer0[j];
    }
    mlp_layer3[i] = fmax(0.0f, sum); // ReLU
  }
  
  // Layer 6: Linear(32, 1) + Sigmoid
  float sum = mlp6_bias[0];
  for (int j = 0; j < 32; j++) {
    sum += mlp6_weight[0][j] * mlp_layer3[j];
  }
  mlp_output = 1.0f / (1.0f + exp(-sum)); // Sigmoid
  
  return mlp_output;
}

void computeGate(float* input, float* hidden, float* gate_output,
                 const float weight_ih[][INPUT_SIZE], const float weight_hh[][HIDDEN_SIZE],
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
                          const float weight_ih[][INPUT_SIZE], const float weight_hh[][HIDDEN_SIZE], 
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
  Serial.println("🔄 LSTM State Reset - Ready for new sequence");
}

void printStatus() {
  Serial.println("📊 === ARDUINO LSTM STATUS ===");
  Serial.print("🔢 Predictions: "); Serial.println(predictionCount);
  Serial.print("🧠 Hidden Size: "); Serial.println(HIDDEN_SIZE);
  Serial.print("📝 Parameters: 6881");
  Serial.print("💾 Memory: 26.88 KB");
  Serial.println();
  
  // Sample Hidden State (erste 5 Werte)
  Serial.print("🎯 Hidden State [0-4]: ");
  for (int i = 0; i < 5; i++) {
    Serial.print(h_state[i], 4);
    if (i < 4) Serial.print(", ");
  }
  Serial.println();
  Serial.println("========================");
}

// Debug Funktionen
void printModelInfo() {
  Serial.println("🎯 ARDUINO LSTM MODEL INFO");
  Serial.println("==========================");
  Serial.print("Input Size: "); Serial.println(INPUT_SIZE);
  Serial.print("Hidden Size: "); Serial.println(HIDDEN_SIZE);
  Serial.print("Output Size: "); Serial.println(OUTPUT_SIZE);
  Serial.println();
  Serial.println("Architecture: LSTM(4→32) + MLP(32→32→32→1)");
  Serial.println("Activation: Sigmoid gates, Tanh candidate, ReLU MLP, Sigmoid output");
  Serial.println("Weights: Loaded from best_model.pth");
  Serial.println("==========================");
}
