/*
ARDUINO LSTM SOC PREDICTOR - VOLLSTÄNDIGE 32 HIDDEN UNITS + HARDWARE MONITORING
Basiert auf arduino_lstm_soc_full32.ino mit zusätzlichen Hardware-Performance-Metriken
IDENTISCH zu live_test_soc.py - KEINE Komprimierung!
+ Hardware Monitoring: RAM, CPU Load, Inference Time, Temperature
*/

#include "lstm_weights.h"
#include <math.h>

// Vollständige Konstanten wie live_test_soc.py
#define INPUT_SIZE 4
#define HIDDEN_SIZE 32    // VOLLSTÄNDIGE 32 wie PyTorch!
#define OUTPUT_SIZE 1
#define BUFFER_SIZE 256

// ============ HARDWARE MONITORING VARIABLEN ============
unsigned long inference_start_micros = 0;
unsigned long inference_end_micros = 0;
unsigned long last_inference_time_us = 0;
unsigned long total_inferences = 0;
unsigned long total_inference_time_us = 0;

// Performance Statistics
float avg_inference_time_us = 0.0;
float min_inference_time_us = 999999.0;
float max_inference_time_us = 0.0;

// RAM Monitoring - Arduino Uno R4 WiFi (RA4M1) korrekte Adressen
extern char _end;
extern "C" char *sbrk(int i);
char *ramstart = (char *)0x20000000;  // RA4M1 SRAM Start
char *ramend   = (char *)0x20008000;  // RA4M1 SRAM End (32KB)

// CPU Load Estimation
unsigned long loop_start_time = 0;
unsigned long active_time_us = 0;
unsigned long total_time_us = 0;
float estimated_cpu_load = 0.0;

// Temperature (falls MCU Sensor verfügbar)
float mcu_temperature = 0.0;

// ============ ORIGINAL LSTM VARIABLEN ============
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

// ============ HARDWARE MONITORING FUNKTIONEN ============

int getFreeRam() {
  char *heapend = sbrk(0);
  register char * stack_ptr asm ("sp");
  return stack_ptr - heapend;
}

int getTotalRam() {
  return ramend - ramstart;
}

int getUsedRam() {
  return getTotalRam() - getFreeRam();
}

float getRamFragmentation() {
  int total = getTotalRam();
  int used = getUsedRam();
  return (float)used / (float)total * 100.0;
}

void updatePerformanceStats(unsigned long inference_time_us) {
  total_inferences++;
  total_inference_time_us += inference_time_us;
  
  // Update Statistics
  avg_inference_time_us = (float)total_inference_time_us / (float)total_inferences;
  
  if (inference_time_us < min_inference_time_us) {
    min_inference_time_us = inference_time_us;
  }
  
  if (inference_time_us > max_inference_time_us) {
    max_inference_time_us = inference_time_us;
  }
}

void updateCpuLoad(unsigned long active_us, unsigned long total_us) {
  if (total_us > 0) {
    estimated_cpu_load = ((float)active_us / (float)total_us) * 100.0;
  }
}

// Vereinfachte MCU Temperatur (falls verfügbar)
float readMcuTemperature() {
  // Placeholder - abhängig vom MCU Typ
  // Für Arduino Uno/Nano: kein interner Temp Sensor
  // Für STM32/ESP32: analogRead von internem Temp Channel
  return 25.0; // Dummy Wert
}

void printHardwareStats() {
  int free_ram = getFreeRam();
  int used_ram = getUsedRam();
  int total_ram = getTotalRam();
  float fragmentation = getRamFragmentation();
  
  Serial.print("STATS:");
  Serial.print(last_inference_time_us); Serial.print(",");
  Serial.print(free_ram); Serial.print(",");
  Serial.print(used_ram); Serial.print(",");
  Serial.print(estimated_cpu_load, 1); Serial.print(",");
  Serial.print(mcu_temperature, 1);
  Serial.println();
}

void printBenchmarkResults() {
  Serial.print("BENCHMARK:");
  Serial.print(avg_inference_time_us, 0); Serial.print(",");
  Serial.print(min_inference_time_us, 0); Serial.print(",");
  Serial.print(max_inference_time_us, 0); Serial.print(",");
  Serial.print(getTotalRam()); Serial.print(",");
  Serial.print(getFreeRam()); Serial.print(",");
  Serial.print(F_CPU / 1000000); // MHz
  Serial.println();
}

void performBenchmark() {
  Serial.println("Starting benchmark...");
  
  // Reset Statistics
  total_inferences = 0;
  total_inference_time_us = 0;
  min_inference_time_us = 999999.0;
  max_inference_time_us = 0.0;
  
  // Benchmark mit Test-Daten
  float test_voltages[] = {3.2, 3.4, 3.6, 3.8, 4.0};
  float test_currents[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  float test_soh = 0.95;
  float test_qc = 0.8;
  
  // 25 Test-Durchläufe (5x5 Kombinationen)
  for (int v = 0; v < 5; v++) {
    for (int c = 0; c < 5; c++) {
      unsigned long start_us = micros();
      float soc = predictSOC(test_voltages[v], test_currents[c], test_soh, test_qc);
      unsigned long end_us = micros();
      
      updatePerformanceStats(end_us - start_us);
      
      // Kurze Pause zwischen Tests
      delayMicroseconds(100);
    }
  }
  
  printBenchmarkResults();
}

// ============ ORIGINAL LSTM FUNKTIONEN ============

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

// VOLLSTÄNDIGE LSTM Forward Pass (32 Hidden Units) + Hardware Monitoring
float predictSOC(float voltage, float current, float soh, float q_c) {
  // ============ HARDWARE MONITORING START ============
  inference_start_micros = micros();
  
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
  
  // ============ HARDWARE MONITORING END ============
  inference_end_micros = micros();
  last_inference_time_us = inference_end_micros - inference_start_micros;
  updatePerformanceStats(last_inference_time_us);
  
  return mlp_output;
}

void resetLSTMStates() {
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    h_state[i] = 0.0f;
    c_state[i] = 0.0f;
  }
  
  // Reset auch Performance Stats
  total_inferences = 0;
  total_inference_time_us = 0;
  min_inference_time_us = 999999.0;
  max_inference_time_us = 0.0;
  avg_inference_time_us = 0.0;
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
    
    // SOC Vorhersage mit Hardware Monitoring
    float soc_pred = predictSOC(voltage, current, soh, q_c);
    
    // Erweiterte Antwort mit Hardware-Metriken
    Serial.print("DATA:");
    Serial.print(soc_pred, 6); Serial.print(",");
    Serial.print(last_inference_time_us); Serial.print(",");
    Serial.print(getFreeRam()); Serial.print(",");
    Serial.print(getUsedRam()); Serial.print(",");
    Serial.print(estimated_cpu_load, 1); Serial.print(",");
    Serial.print(mcu_temperature, 1);
    Serial.println();
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  
  // Initialisierung
  resetLSTMStates();
  
  // Hardware Monitoring Setup
  mcu_temperature = readMcuTemperature();
  loop_start_time = micros();
  
  // QUIET START - Keine automatischen Ausgaben für saubere Kommunikation
  Serial.println("READY_WITH_MONITORING");
}

void loop() {
  unsigned long loop_current_time = micros();
  
  if (Serial.available()) {
    unsigned long cmd_start = micros();
    
    char incomingByte = Serial.read();
    
    if (incomingByte == '\n' || incomingByte == '\r') {
      if (bufferIndex > 0) {
        inputBuffer[bufferIndex] = '\0';
        String inputString = String(inputBuffer);
        
        // Erweiterte Kommandos für Hardware Monitoring
        if (inputString.equals("RESET")) {
          resetLSTMStates();
          Serial.println("LSTM States and stats reset");
        } else if (inputString.equals("INFO")) {
          printModelInfo();
        } else if (inputString.equals("STATS")) {
          printHardwareStats();
        } else if (inputString.equals("RAM")) {
          Serial.print("RAM:");
          Serial.print(getFreeRam()); Serial.print(",");
          Serial.print(getUsedRam()); Serial.print(",");
          Serial.print(getTotalRam()); Serial.print(",");
          Serial.print(getRamFragmentation(), 1);
          Serial.println();
        } else if (inputString.equals("BENCHMARK")) {
          performBenchmark();
        } else {
          processJSON(inputString);
        }
        
        bufferIndex = 0;
      }
    } else if (bufferIndex < BUFFER_SIZE - 1) {
      inputBuffer[bufferIndex++] = incomingByte;
    }
    
    // CPU Load Berechnung
    unsigned long cmd_end = micros();
    active_time_us += (cmd_end - cmd_start);
  }
  
  // CPU Load Update alle 1000 Loops
  if ((loop_current_time - loop_start_time) > 100000) { // 100ms
    total_time_us = loop_current_time - loop_start_time;
    updateCpuLoad(active_time_us, total_time_us);
    
    // Reset für nächste Messung
    loop_start_time = loop_current_time;
    active_time_us = 0;
    
    // Temperature Update
    mcu_temperature = readMcuTemperature();
  }
}

// Debug Funktionen
void printModelInfo() {
  Serial.println("🎯 ARDUINO LSTM MODEL INFO + HARDWARE MONITORING");
  Serial.println("===============================================");
  Serial.print("Input Size: "); Serial.println(INPUT_SIZE);
  Serial.print("Hidden Size: "); Serial.println(HIDDEN_SIZE);
  Serial.print("Output Size: "); Serial.println(OUTPUT_SIZE);
  Serial.println();
  Serial.println("Architecture: LSTM(4→32) + MLP(32→32→32→1)");
  Serial.println("Activation: Sigmoid gates, Tanh candidate, ReLU MLP, Sigmoid output");
  Serial.println("Weights: Loaded from best_model.pth (VOLLSTÄNDIG)");
  Serial.println();
  Serial.println("🔧 HARDWARE MONITORING:");
  Serial.print("Free RAM: "); Serial.print(getFreeRam()); Serial.println(" bytes");
  Serial.print("Used RAM: "); Serial.print(getUsedRam()); Serial.println(" bytes");
  Serial.print("Total RAM: "); Serial.print(getTotalRam()); Serial.println(" bytes");
  Serial.print("RAM Usage: "); Serial.print(getRamFragmentation(), 1); Serial.println("%");
  Serial.print("CPU Frequency: "); Serial.print(F_CPU / 1000000); Serial.println(" MHz");
  Serial.print("Total Inferences: "); Serial.println(total_inferences);
  Serial.print("Avg Inference Time: "); Serial.print(avg_inference_time_us, 0); Serial.println(" μs");
  Serial.print("Min Inference Time: "); Serial.print(min_inference_time_us, 0); Serial.println(" μs");
  Serial.print("Max Inference Time: "); Serial.print(max_inference_time_us, 0); Serial.println(" μs");
  Serial.print("Estimated CPU Load: "); Serial.print(estimated_cpu_load, 1); Serial.println("%");
  Serial.print("MCU Temperature: "); Serial.print(mcu_temperature, 1); Serial.println("°C");
  Serial.println("===============================================");
}
