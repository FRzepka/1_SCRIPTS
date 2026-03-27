/*
ARDUINO LSTM SOC PREDICTOR - VOLLSTÄNDIGE 16 HIDDEN UNITS + HARDWARE MONITORING
Basiert auf arduino_lstm_soc_full32_with_monitoring.ino - Angepasst für 16x16 Architecture
IDENTISCH zu live_test_soc.py - KEINE Komprimierung!
+ Hardware Monitoring: RAM, CPU Load, Inference Time, Temperature
*/

#include "lstm_weights.h"
#include <math.h>

// Vollständige Konstanten für 16x16 Architecture
#define INPUT_SIZE 4
#define HIDDEN_SIZE 16    // VOLLSTÄNDIGE 16 wie PyTorch!
#define MLP_HIDDEN_SIZE 16
#define OUTPUT_SIZE 1
#define BUFFER_SIZE 256

// Model Information
#define TOTAL_PARAMETERS 3200    // Approximate for 16x16 LSTM + MLP
#define MEMORY_USAGE_KB 8        // Approximate memory usage in KB

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

// RAM Monitoring
extern char _end;
extern "C" char *sbrk(int i);
char *ramstart = (char *)0x20000000;  // Arduino UNO R4 SRAM Start
char *ramend   = (char *)0x20008000;  // Arduino UNO R4 SRAM End (32KB)

// Flash Memory Detection - Arduino UNO R4 WiFi (RA4M1)
char *flashstart = (char *)0x00000000;  // RA4M1 Flash Start
char *flashend   = (char *)0x00040000;  // RA4M1 Flash End (256KB)

// CPU Load Estimation
unsigned long loop_start_time = 0;
unsigned long active_time_us = 0;
unsigned long total_time_us = 0;
float estimated_cpu_load = 0.0;

// Temperature (falls MCU Sensor verfügbar)
float mcu_temperature = 0.0;

// ============ ORIGINAL LSTM VARIABLEN ============
// LSTM Hidden States (16 Units!)
float h_state[HIDDEN_SIZE] = {0};
float c_state[HIDDEN_SIZE] = {0};

// Temporäre Arrays für LSTM Gates (16 Units!)
float input_gate[HIDDEN_SIZE];
float forget_gate[HIDDEN_SIZE];
float candidate_gate[HIDDEN_SIZE];
float output_gate[HIDDEN_SIZE];
float new_c_state[HIDDEN_SIZE];
float new_h_state[HIDDEN_SIZE];

// MLP Layers (16 Units!)
float mlp_layer0[MLP_HIDDEN_SIZE];
float mlp_layer3[MLP_HIDDEN_SIZE];
float mlp_output = 0.0;

// ============ HARDWARE MONITORING FUNKTIONEN ============
// ECHTE RAM-Messung zur Laufzeit (nicht Compiler-Schätzungen)
int getFreeRam() {
  // ARM Cortex-M basierte dynamische RAM-Messung
  // Stack pointer minus heap end = available RAM
  char *heapend = sbrk(0);           // Aktuelle Heap-Grenze
  register char * stack_ptr asm ("sp");     // Aktueller Stack-Pointer  
  return stack_ptr - heapend;        // Freier RAM zwischen Stack und Heap
}

int getTotalRam() {
  // Arduino UNO R4 WiFi: 32KB SRAM
  return ramend - ramstart;  // 0x20008000 - 0x20000000 = 32768 Bytes
}

int getUsedRam() {
  // Verwendeter RAM = Gesamt - Verfügbar
  return getTotalRam() - getFreeRam();
}

float getRamUsagePercent() {
  // RAM-Auslastung in Prozent
  int total = getTotalRam();
  int used = getUsedRam();
  return (float)used / (float)total * 100.0;
}

float getRamFragmentation() {
  // Alias für getRamUsagePercent für Kompatibilität
  return getRamUsagePercent();
}

// Flash Memory Functions - Arduino UNO R4 WiFi (RA4M1) mit echten Messungen
int getTotalFlash() {
  return flashend - flashstart;  // 256KB Flash total
}

int getUsedFlash() {
  // Echte Flash-Nutzung für Arduino UNO R4
  #if defined(ARDUINO_UNOR4_WIFI) || defined(ARDUINO_UNOR4_MINIMA)
    // Echte Flash-Nutzung aus Linker-Symbolen - Arduino UNO R4 WiFi (RA4M1)
    extern char __etext[];        // Ende des Textbereichs (Code)
    extern char __data_start__[]; // Beginn der initialisierten Daten im RAM
    extern char __data_end__[];   // Ende der initialisierten Daten im RAM
    
    // Flash beginnt bei 0x00000000
    uint32_t flash_start = 0x00000000;
    uint32_t code_size = (uint32_t)__etext - flash_start;
    uint32_t data_size = (uint32_t)__data_end__ - (uint32_t)__data_start__; // Datenkopie in Flash
    uint32_t total_used_flash = code_size + data_size;
    return (int)total_used_flash;
  #else
    // Fallback-Schätzungen für andere Arduino-Boards
    int total_flash = getTotalFlash();
    
    if (total_flash <= 32768) {
      // Arduino Uno/Nano (32KB Flash) - konservative Schätzung
      return 28 * 1024; // 28KB verwendet
    } else if (total_flash <= 262144) {
      // Arduino Mega (256KB Flash) - größere Programmschätzung
      return 180 * 1024; // 180KB verwendet
    } else {
      // Größere Boards (Due, ESP32, etc.) - skaliert mit verfügbarem Flash
      return total_flash / 4; // 25% als Schätzung verwenden
    }
  #endif
}

int getFreeFlash() {
  return getTotalFlash() - getUsedFlash();
}

float getFlashUsagePercent() {
  int total = getTotalFlash();
  int used = getUsedFlash();
  return (float)used / (float)total * 100.0;
}

float getFlashUtilization() {
  // Alias für getFlashUsagePercent für Kompatibilität
  return getFlashUsagePercent();
}

void updatePerformanceStats() {
  // Berechne Inferenzzeit
  last_inference_time_us = micros() - inference_start_micros;
  
  total_inferences++;
  total_inference_time_us += last_inference_time_us;
  
  // Aktualisiere Statistiken
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
  if (total_time_us > 0) {
    estimated_cpu_load = ((float)active_time_us / (float)total_time_us) * 100.0;
  }
}

// Einfache MCU Temperatur (falls verfügbar)
float readMcuTemperature() {
  // Platzhalter - hängt vom MCU-Typ ab
  // Für Arduino Uno/Nano: kein interner Temperatursensor
  // Für STM32/ESP32: analogRead vom internen Temperaturkanal
  return 25.0; // Dummy-Wert
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

void printRamInfo() {
  Serial.print("RAM:");
  Serial.print(getFreeRam()); Serial.print(",");
  Serial.print(getUsedRam()); Serial.print(",");
  Serial.print(getTotalRam()); Serial.print(",");
  Serial.print(getRamUsagePercent(), 1);
  Serial.println();
}

void printBenchmarkResults() {
  Serial.print("BENCHMARK:");
  Serial.print(avg_inference_time_us, 0); Serial.print(",");
  Serial.print(min_inference_time_us, 0); Serial.print(",");
  Serial.print(max_inference_time_us, 0); Serial.print(",");
  Serial.print(getTotalRam()); Serial.print(",");
  Serial.print(getFreeRam()); Serial.print(",");
  Serial.print(getTotalFlash()); Serial.print(",");
  Serial.print(getFreeFlash()); Serial.print(",");
  
  // Erzwinge die Erkennung der Arduino UNO R4 WiFi durch Senden von 48MHz
  #if defined(ARDUINO_UNOR4_WIFI) || defined(ARDUINO_UNOR4_MINIMA)
    Serial.print(48); // Arduino UNO R4 WiFi/Minima: 48 MHz
  #else
    Serial.print(F_CPU / 1000000); // Andere Boards: tatsächliche Frequenz verwenden
  #endif
  
  Serial.println();
}

void performBenchmark() {
  Serial.println("Starte Benchmark...");
  
  // Führe mehrere Vorhersagen für den Benchmark durch
  float test_voltage = 3.7;
  float test_current = 1.0;
  float test_soh = 0.95;
  float test_q_c = 0.8;
  
  unsigned long benchmark_start = micros();
  
  // Führe 10 Vorhersagen für den Benchmark durch
  for (int i = 0; i < 10; i++) {
    predictSOC(test_voltage, test_current, test_soh, test_q_c);
  }
  
  unsigned long benchmark_end = micros();
  unsigned long benchmark_time = benchmark_end - benchmark_start;
  
  Serial.print("Benchmark abgeschlossen: ");
  Serial.print(benchmark_time / 10);
  Serial.println(" μs durchschnittlich pro Vorhersage");
  
  printBenchmarkResults();
}

// ============ AKTIVIERUNGSFUNKTIONEN ============
float sigmoid_activation(float x) {
  return 1.0f / (1.0f + exp(-x));
}

float tanh_activation(float x) {
  return tanh(x);
}

float relu_activation(float x) {
  return fmax(0.0f, x);
}

// ============ VOLLSTÄNDIGE LSTM FORWARD PASS (16 Hidden Units) ============
float predictSOC(float voltage, float current, float soh, float q_c) {
  // Start Timing
  inference_start_micros = micros();
  
  float input[INPUT_SIZE] = {voltage, current, soh, q_c};
  
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
  
  // Layer 0: Linear(16, 16) + ReLU + Dropout
  for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
    float sum = mlp0_bias[i];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += mlp0_weights[i][j] * h_state[j];
    }
    mlp_layer0[i] = fmax(0.0f, sum); // ReLU
  }
  
  // Layer 3: Linear(16, 16) + ReLU + Dropout  
  for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
    float sum = mlp3_bias[i];
    for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
      sum += mlp3_weights[i][j] * mlp_layer0[j];
    }
    mlp_layer3[i] = fmax(0.0f, sum); // ReLU
  }
  
  // Layer 6: Linear(16, 1) + Sigmoid
  float sum = mlp6_bias[0];
  for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
    sum += mlp6_weights[0][j] * mlp_layer3[j];
  }
  mlp_output = 1.0f / (1.0f + exp(-sum)); // Sigmoid
  
  // End Timing
  inference_end_micros = micros();
  last_inference_time_us = inference_end_micros - inference_start_micros;
  
  return mlp_output;
}

void resetLSTMStates() {
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    h_state[i] = 0.0;
    c_state[i] = 0.0;
  }
  Serial.println("🔄 LSTM States zurückgesetzt");
}

void printModelInfo() {
  Serial.println("🎯 ARDUINO LSTM MODEL INFO (16x16)");
  Serial.println("=======================================");
  Serial.print("Input Size: "); Serial.println(INPUT_SIZE);
  Serial.print("Hidden Size: "); Serial.println(HIDDEN_SIZE);
  Serial.print("MLP Hidden Size: "); Serial.println(MLP_HIDDEN_SIZE);
  Serial.print("Output Size: "); Serial.println(OUTPUT_SIZE);
  Serial.println();
  Serial.println("Architecture: LSTM(4→16) + MLP(16→16→16→1)");
  Serial.println("Activation: Sigmoid gates, Tanh candidate, ReLU MLP, Sigmoid output");
  Serial.println("Weights: Loaded from best_model.pth (16x16)");
  Serial.print("Total Parameters: "); Serial.println(TOTAL_PARAMETERS);
  Serial.print("Memory Usage: "); Serial.print(MEMORY_USAGE_KB); Serial.println(" KB");
  Serial.println("=======================================");
}

void processData(String dataString) {
  // Parse CSV: voltage,current,soh,q_c
  int commaIndex1 = dataString.indexOf(',');
  int commaIndex2 = dataString.indexOf(',', commaIndex1 + 1);
  int commaIndex3 = dataString.indexOf(',', commaIndex2 + 1);
  
  if (commaIndex1 == -1 || commaIndex2 == -1 || commaIndex3 == -1) {
    Serial.println("❌ Error: Invalid data format. Expected: voltage,current,soh,q_c");
    return;
  }
  
  float voltage = dataString.substring(0, commaIndex1).toFloat();
  float current = dataString.substring(commaIndex1 + 1, commaIndex2).toFloat();
  float soh = dataString.substring(commaIndex2 + 1, commaIndex3).toFloat();
  float q_c = dataString.substring(commaIndex3 + 1).toFloat();
  
  // SOC Prediction
  float predicted_soc = predictSOC(voltage, current, soh, q_c);
    // Update Performance Stats
  updatePerformanceStats();
  updateCpuLoad();
  
  // Original format that Python script expects: "📊 SOC: X.XXXX (XX.XX%) | Time: XXXX μs"
  Serial.print("📊 SOC: ");
  Serial.print(predicted_soc, 4);
  Serial.print(" (");
  Serial.print(predicted_soc * 100, 2);
  Serial.print("%) | Time: ");
  Serial.print(last_inference_time_us);
  Serial.println(" μs");
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  
  // Initialization
  resetLSTMStates();
  
  // Hardware Monitoring Setup
  mcu_temperature = readMcuTemperature();
  loop_start_time = micros();
  
  Serial.println("\n🎯 ARDUINO LSTM SOC PREDICTOR - 16x16 + MONITORING");
  Serial.println("====================================================");
  
  printModelInfo();
  
  Serial.println("\n📝 Commands:");
  Serial.println("- Send data: voltage,current,soh,q_c");
  Serial.println("- RESET: Reset LSTM states");
  Serial.println("- STATS: Show performance statistics"); 
  Serial.println("- RAM: Show RAM usage");
  Serial.println("- BENCHMARK: Run performance benchmark");
  Serial.println("- INFO: Show model information");
  Serial.println("\n✅ Ready for predictions...");
}

void loop() {
  unsigned long loop_current_time = micros();
  
  if (Serial.available()) {
    unsigned long cmd_start = micros();
    String inputString = Serial.readStringUntil('\n');
    inputString.trim();
    
    // Extended commands for hardware monitoring - matching 32-bit version
    if (inputString.equalsIgnoreCase("RESET")) {
      resetLSTMStates();
      Serial.println("LSTM States and stats reset");
    } else if (inputString.equalsIgnoreCase("INFO")) {
      printModelInfo();
    } else if (inputString.equalsIgnoreCase("STATS")) {
      printHardwareStats();
    } else if (inputString.equalsIgnoreCase("RAM")) {
      printRamInfo();
    } else if (inputString.equalsIgnoreCase("BENCHMARK")) {
      performBenchmark();
    } else if (inputString.length() > 0) {
      processData(inputString);
    }
    
    // CPU Load calculation
    unsigned long cmd_end = micros();
    active_time_us += (cmd_end - cmd_start);
  }
  
  // CPU Load Update every 100ms
  if ((loop_current_time - loop_start_time) > 100000) { // 100ms
    total_time_us = loop_current_time - loop_start_time;
    updateCpuLoad();
    
    // Reset for next measurement
    loop_start_time = loop_current_time;
    active_time_us = 0;
    
    // Temperature Update
    mcu_temperature = readMcuTemperature();
  }
}
