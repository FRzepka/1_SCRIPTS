/*
ARDUINO LSTM SOC PREDICTOR - VOLLSTÄNDIGE 64 HIDDEN UNITS + HARDWARE MONITORING
Basiert auf arduino_lstm_soc_full32.ino mit 64x64 Architektur
IDENTISCH zu Training - KEINE Komprimierung!
+ Hardware Monitoring: RAM, CPU Load, Inference Time, Temperature
WARNUNG: Benötigt >100KB RAM - für Arduino Mega 2560 oder ESP32!
*/

#include "lstm_weights.h"
#include <math.h>

// Vollständige Konstanten wie Training
#define INPUT_SIZE 4
#define HIDDEN_SIZE 64    // VOLLSTÄNDIGE 64 wie PyTorch!
#define OUTPUT_SIZE 1
#define MLP_HIDDEN_SIZE 64
#define BUFFER_SIZE 256

// Model Information for Monitoring
#define TOTAL_PARAMETERS 26305
// #define MEMORY_USAGE_KB 103  // Estimated 103KB - Commented out, will use definition from lstm_weights.h

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

// RAM Monitoring - Arduino Mega 2560 / ESP32 kompatibel
extern char _end;
extern "C" char *sbrk(int i);

// CPU Load Estimation
unsigned long loop_start_time = 0;
unsigned long active_time_us = 0;
unsigned long total_time_us = 0;
float estimated_cpu_load = 0.0;

// Temperature (falls MCU Sensor verfügbar)
float mcu_temperature = 0.0;

// ============ ORIGINAL LSTM VARIABLEN (64 Units!) ============
// LSTM Hidden States (64 Units!)
float h_state[HIDDEN_SIZE] = {0};
float c_state[HIDDEN_SIZE] = {0};

// Temporäre Arrays für LSTM Berechnungen
float input_gate[HIDDEN_SIZE];
float forget_gate[HIDDEN_SIZE];
float candidate_gate[HIDDEN_SIZE];
float output_gate[HIDDEN_SIZE];
float new_c_state[HIDDEN_SIZE];
float new_h_state[HIDDEN_SIZE];

// MLP Zwischenergebnisse (64x64 Architecture)
float mlp_layer0[MLP_HIDDEN_SIZE];
float mlp_layer3[MLP_HIDDEN_SIZE];
float mlp_output;

// Eingabepuffer
char inputBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// ============ HARDWARE MONITORING FUNKTIONEN ============

// ============ REAL-TIME RAM MEASUREMENT FUNCTIONS ============
// Diese Funktionen messen ECHTE RAM-Nutzung zur Laufzeit

int getFreeRam() {
  char *heapend = sbrk(0);           
  register char * stack_ptr asm ("sp");  
  return stack_ptr - heapend;        
}

int getTotalRam() {
  // Detect Arduino type and return appropriate RAM size
  #ifdef ARDUINO_AVR_MEGA2560
    return 8192;  // Arduino Mega 2560: 8KB SRAM
  #elif defined(ESP32)
    return 327680; // ESP32: ~320KB SRAM
  #else
    return 32768;  // Arduino UNO R4: 32KB SRAM (insufficient!)
  #endif
}

int getUsedRam() {
  return getTotalRam() - getFreeRam();
}

float getRamFragmentation() {
  int total = getTotalRam();
  int used = getUsedRam();
  return (float)used / (float)total * 100.0;
}

// ============ ENHANCED FLASH MEMORY FUNCTIONS ============
int getTotalFlash() {
  #ifdef ARDUINO_AVR_MEGA2560
    return 262144;  // Arduino Mega 2560: 256KB Flash
  #elif defined(ESP32)
    return 4194304; // ESP32: 4MB Flash
  #else
    return 262144;  // Arduino UNO R4: 256KB Flash
  #endif
}

int getUsedFlash() {
  // Estimation based on model size and Arduino code
  return MEMORY_USAGE_KB * 1024 + 20480; // Model + Arduino code (~20KB)
}

int getFreeFlash() {
  return getTotalFlash() - getUsedFlash();
}

float getFlashUtilization() {
  int total = getTotalFlash();
  int used = getUsedFlash();
  return (float)used / (float)total * 100.0;
}

// ============ PERFORMANCE MONITORING FUNCTIONS ============
void updatePerformanceStats() {
  total_inferences++;
  total_inference_time_us += last_inference_time_us;
  avg_inference_time_us = (float)total_inference_time_us / (float)total_inferences;
  
  if (last_inference_time_us < min_inference_time_us) {
    min_inference_time_us = last_inference_time_us;
  }
  if (last_inference_time_us > max_inference_time_us) {
    max_inference_time_us = last_inference_time_us;
  }
}

void updateCpuLoad() {
  unsigned long current_time = micros();
  if (loop_start_time > 0) {
    unsigned long loop_duration = current_time - loop_start_time;
    total_time_us += loop_duration;
    active_time_us += last_inference_time_us;
    
    if (total_time_us > 0) {
      estimated_cpu_load = (float)active_time_us / (float)total_time_us * 100.0;
    }
  }
  loop_start_time = current_time;
}

float readMcuTemperature() {
  // Placeholder für MCU-Temperatursensor
  // Arduino Mega/UNO haben keinen eingebauten Temperatursensor
  return 25.0 + (float)(analogRead(A0)) * 0.01; // Mock-Wert
}

// ============ LSTM NEURAL NETWORK FUNCTIONS (64x64) ============

// Sigmoid Aktivierungsfunktion
float sigmoid(float x) {
  if (x > 10) return 1.0;
  if (x < -10) return 0.0;
  return 1.0 / (1.0 + exp(-x));
}

// Tanh Aktivierungsfunktion
float tanh_activation(float x) {
  if (x > 10) return 1.0;
  if (x < -10) return -1.0;
  return tanh(x);
}

// ReLU Aktivierungsfunktion für MLP
float relu(float x) {
  return (x > 0) ? x : 0;
}

// LSTM Forward Pass (64 Hidden Units)
void lstm_forward(float input[INPUT_SIZE]) {
  // Input Gate
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    input_gate[i] = lstm_input_ih_bias[i] + lstm_input_hh_bias[i];
    for (int j = 0; j < INPUT_SIZE; j++) {
      input_gate[i] += input[j] * lstm_input_ih_weights[i][j];
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      input_gate[i] += h_state[j] * lstm_input_hh_weights[i][j];
    }
    input_gate[i] = sigmoid(input_gate[i]);
  }

  // Forget Gate
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    forget_gate[i] = lstm_forget_ih_bias[i] + lstm_forget_hh_bias[i];
    for (int j = 0; j < INPUT_SIZE; j++) {
      forget_gate[i] += input[j] * lstm_forget_ih_weights[i][j];
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      forget_gate[i] += h_state[j] * lstm_forget_hh_weights[i][j];
    }
    forget_gate[i] = sigmoid(forget_gate[i]);
  }

  // Candidate Gate
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    candidate_gate[i] = lstm_candidate_ih_bias[i] + lstm_candidate_hh_bias[i];
    for (int j = 0; j < INPUT_SIZE; j++) {
      candidate_gate[i] += input[j] * lstm_candidate_ih_weights[i][j];
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      candidate_gate[i] += h_state[j] * lstm_candidate_hh_weights[i][j];
    }
    candidate_gate[i] = tanh_activation(candidate_gate[i]);
  }

  // Output Gate
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    output_gate[i] = lstm_output_ih_bias[i] + lstm_output_hh_bias[i];
    for (int j = 0; j < INPUT_SIZE; j++) {
      output_gate[i] += input[j] * lstm_output_ih_weights[i][j];
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      output_gate[i] += h_state[j] * lstm_output_hh_weights[i][j];
    }
    output_gate[i] = sigmoid(output_gate[i]);
  }

  // Cell State Update
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    new_c_state[i] = forget_gate[i] * c_state[i] + input_gate[i] * candidate_gate[i];
    c_state[i] = new_c_state[i];
  }

  // Hidden State Update
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    new_h_state[i] = output_gate[i] * tanh_activation(c_state[i]);
    h_state[i] = new_h_state[i];
  }
}

// MLP Forward Pass (64->64->64->1)
float mlp_forward() {
  // Layer 0: (MLP_HIDDEN_SIZE)
  for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
    mlp_layer0[i] = mlp0_bias[i]; // Changed from mlp_layer0_bias
    for (int j = 0; j < ARDUINO_HIDDEN_SIZE; j++) {
      mlp_layer0[i] += h_state[j] * mlp0_weights[i][j]; // Changed from mlp_layer0_weights
    }
    mlp_layer0[i] = tanh(mlp_layer0[i]); // Activation
  }

  // Layer 3: (MLP_HIDDEN_SIZE)
  for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
    mlp_layer3[i] = mlp3_bias[i]; // Changed from mlp_layer3_bias
    for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
      mlp_layer3[i] += mlp_layer0[j] * mlp3_weights[i][j]; // Changed from mlp_layer3_weights
    }
    mlp_layer3[i] = tanh(mlp_layer3[i]); // Activation
  }

  // Output Layer (Layer 6): (1)
  float mlp_output = mlp6_bias[0]; // Changed from mlp_layer6_bias
  for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
    mlp_output += mlp_layer3[j] * mlp6_weights[0][j]; // Changed from mlp_layer6_weights
  }
  // No activation for regression output, or sigmoid if SOC is 0-1 and model trained with it
  return mlp_output; 
}

// Vollständige SOC Prediction (64x64)
float predict_soc(float voltage, float current, float soh, float q_c) { // MODIFIED: soh, q_c
  float input[INPUT_SIZE] = {voltage, current, soh, q_c}; // MODIFIED: soh, q_c
  
  // Inference Time Measurement
  inference_start_micros = micros();
  
  // LSTM Forward Pass
  lstm_forward(input);
  
  // MLP Forward Pass
  float soc = mlp_forward();
  
  // End Timing
  inference_end_micros = micros();
  last_inference_time_us = inference_end_micros - inference_start_micros;
  
  // Update Performance Statistics
  updatePerformanceStats();
  
  return soc;
}

// ============ DATA PROCESSING FUNCTIONS ============
void processData(float voltage, float current, float soh, float q_c) { // MODIFIED: soh, q_c
  float soc = predict_soc(voltage, current, soh, q_c); // MODIFIED: soh, q_c
  
  // Human-readable output format (exactly like 32-bit version)
  Serial.print("📊 SOC: ");
  Serial.print(soc, 4);
  Serial.print(" (");
  Serial.print(soc * 100.0, 2);
  Serial.print("%) | Time: ");
  Serial.print(last_inference_time_us);
  Serial.println(" μs");
  
  // Update CPU Load
  updateCpuLoad();
}

// ============ HARDWARE STATS COMMANDS ============
void printHardwareStats() {
  // Model Information
  Serial.println("🔧 ARDUINO LSTM MODEL (64x64):");
  Serial.print("📊 Parameters: ");
  Serial.println(TOTAL_PARAMETERS);
  Serial.print("💾 Est. Memory: ");
  Serial.print(MEMORY_USAGE_KB);
  Serial.println(" KB");
  Serial.print("⚡ Hidden Units: ");
  Serial.println(HIDDEN_SIZE);
  Serial.print("🧠 MLP Hidden: ");
  Serial.println(MLP_HIDDEN_SIZE);
  
  // Performance Stats
  Serial.println("📈 PERFORMANCE STATS:");
  Serial.print("🕐 Last Inference: ");
  Serial.print(last_inference_time_us);
  Serial.println(" μs");
  Serial.print("📊 Avg Inference: ");
  Serial.print(avg_inference_time_us, 1);
  Serial.println(" μs");
  Serial.print("⚡ Min/Max: ");
  Serial.print(min_inference_time_us);
  Serial.print("/");
  Serial.print(max_inference_time_us);
  Serial.println(" μs");
  Serial.print("🔄 Total Inferences: ");
  Serial.println(total_inferences);
  Serial.print("🎯 CPU Load: ");
  Serial.print(estimated_cpu_load, 1);
  Serial.println("%");
  
  // Temperature
  mcu_temperature = readMcuTemperature();
  Serial.print("🌡️ MCU Temp: ");
  Serial.print(mcu_temperature, 1);
  Serial.println("°C");
}

void printRamStats() {
  // int freeRam = getFreeRam(); // Unused variable
  int totalRam = getTotalRam();
  int usedRam = getUsedRam();
  float ramFragmentation = getRamFragmentation();
  
  // int freeFlash = getFreeFlash(); // Unused variable
  int totalFlash = getTotalFlash();
  int usedFlash = getUsedFlash();
  float flashUtilization = getFlashUtilization();
  
  Serial.println("💾 MEMORY STATUS (64x64 Model):");
  
  // RAM Status mit deutschem Format "Verbrauch/Gesamt"
  Serial.print("🐏 RAM: ");
  Serial.print(usedRam / 1024.0, 1);
  Serial.print("/");
  Serial.print(totalRam / 1024.0, 1);
  Serial.print(" KB | Fragmentierung: ");
  Serial.print(ramFragmentation, 1);
  Serial.println("%");
  
  // Flash Status mit deutschem Format "Verbrauch/Gesamt"
  Serial.print("📂 Flash: ");
  Serial.print(usedFlash / 1024.0, 1);
  Serial.print("/");
  Serial.print(totalFlash / 1024.0, 1);
  Serial.print(" KB | Nutzung: ");
  Serial.print(flashUtilization, 1);
  Serial.println("%");
}

// ============ SETUP & LOOP ============
void setup() {
  Serial.begin(115200);
  long entryTime = millis();
  while (!Serial && (millis() - entryTime < 4000)) { // Wait up to 4 seconds for serial connection
    delay(10); 
  }
  
  Serial.println("---ARDUINO BOOTED MINIMAL---"); // Added for minimal boot signal
  delay(100);                                   // Added delay

  // Initialisierungsnachricht
  // Serial.println("⚡ Ready for predictions..."); // Commented out for minimal boot test
  // delay(100); // Kurze Pause, um sicherzustellen, dass die Nachricht gesendet wird

  // Initiale RAM/Flash-Info senden
  // printRamStats(); // Commented out for minimal boot test
  // delay(100);
}

void loop() {
  // Beispielwerte für die Eingabe (könnten auch von Sensoren kommen)
  float voltage = 12.0;
  float current = 0.5;
  float soh = 0.8; // State of Health (80%)
  float q_c = 1.5; // Ladungszustand (z.B. 1.5 Ah)

  // Daten verarbeiten und SOC vorhersagen
  processData(voltage, current, soh, q_c);
  
  // Kurze Pause zwischen den Vorhersagen
  delay(1000);
}
