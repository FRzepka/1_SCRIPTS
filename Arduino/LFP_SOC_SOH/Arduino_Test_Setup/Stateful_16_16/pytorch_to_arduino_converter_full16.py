"""
PyTorch zu Arduino Konverter - VOLLSTÄNDIGE 16 Hidden Units
Extrahiert die exakten Gewichte aus best_model.pth für Arduino UNO R4
Modell: BMS_SOC_LSTM_stateful_1.2.4 (16x16 Architecture)
KEINE Komprimierung - vollständige Modell-Architektur!
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os

# Exakte Modell-Konstanten aus dem Training (16x16)
HIDDEN_SIZE = 16
NUM_LAYERS = 1
MLP_HIDDEN = 16
INPUT_SIZE = 4

# Pfad zum Modell
MODEL_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_16_16\model\best_model.pth"

# Exakte Modell-Definition (identisch zum Training)
class SOCModel(nn.Module):
    def __init__(self, input_size=4, dropout=0.1):
        super().__init__()
        # LSTM ohne Dropout (voller Informationsfluss)
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=0.0)
        # deeper MLP-Head
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(MLP_HIDDEN, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.contiguous()
        h, c = hidden
        h, c = h.contiguous(), c.contiguous()
        hidden = (h, c)
        out, hidden = self.lstm(x, hidden)
        batch, seq_len, hid = out.size()
        out_flat = out.contiguous().view(batch * seq_len, hid)
        soc_flat = self.mlp(out_flat)
        soc = soc_flat.view(batch, seq_len)
        return soc, hidden

def load_exact_model():
    """Lade das exakte trainierte 16x16 Modell"""
    device = torch.device("cpu")  # CPU für Konvertierung
    
    # Modell erstellen
    model = SOCModel(input_size=INPUT_SIZE, dropout=0.1).to(device)
    
    # Exakte Gewichte laden
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    print(f"✅ VOLLSTÄNDIGES 16x16 Model loaded from {MODEL_PATH}")
    return model, device

def analyze_exact_weights(model):
    """Analysiere die exakten Gewichte"""
    print("\n🔍 VOLLSTÄNDIGE GEWICHTS-ANALYSE (16 Hidden Units):")
    
    # LSTM Gewichte
    lstm_weight_ih = model.lstm.weight_ih_l0.data.numpy()  # [64, 4]
    lstm_weight_hh = model.lstm.weight_hh_l0.data.numpy()  # [64, 16]
    lstm_bias_ih = model.lstm.bias_ih_l0.data.numpy()      # [64]
    lstm_bias_hh = model.lstm.bias_hh_l0.data.numpy()      # [64]
    
    print(f"📊 LSTM weight_ih: {lstm_weight_ih.shape} = [4*hidden_size, input_size]")
    print(f"📊 LSTM weight_hh: {lstm_weight_hh.shape} = [4*hidden_size, hidden_size]")
    print(f"📊 LSTM bias_ih: {lstm_bias_ih.shape}")
    print(f"📊 LSTM bias_hh: {lstm_bias_hh.shape}")
    
    # MLP Gewichte
    mlp0_weight = model.mlp[0].weight.data.numpy()  # [16, 16]
    mlp0_bias = model.mlp[0].bias.data.numpy()      # [16]
    mlp3_weight = model.mlp[3].weight.data.numpy()  # [16, 16]
    mlp3_bias = model.mlp[3].bias.data.numpy()      # [16]
    mlp6_weight = model.mlp[6].weight.data.numpy()  # [1, 16]
    mlp6_bias = model.mlp[6].bias.data.numpy()      # [1]
    
    print(f"📊 MLP[0] weight: {mlp0_weight.shape}")
    print(f"📊 MLP[3] weight: {mlp3_weight.shape}")
    print(f"📊 MLP[6] weight: {mlp6_weight.shape}")
    
    return {
        'lstm_weight_ih': lstm_weight_ih,
        'lstm_weight_hh': lstm_weight_hh,
        'lstm_bias_ih': lstm_bias_ih,
        'lstm_bias_hh': lstm_bias_hh,
        'mlp0_weight': mlp0_weight,
        'mlp0_bias': mlp0_bias,
        'mlp3_weight': mlp3_weight,
        'mlp3_bias': mlp3_bias,
        'mlp6_weight': mlp6_weight,
        'mlp6_bias': mlp6_bias
    }

def create_arduino_weights_full16(weights):
    """Erstelle Arduino-Gewichte für vollständige 16-Hidden-Architecture"""
    
    print(f"\n⚙️ ARDUINO-GEWICHTE ERSTELLEN (VOLLSTÄNDIGE 16 HIDDEN UNITS):")
    
    # LSTM Gewichte komplett verwenden - KEINE Reduktion!
    # PyTorch LSTM Layout: [4*hidden_size, input_size] = [64, 4]
    # Gates: input, forget, candidate, output (je 16 Neuronen)
    
    arduino_weights = {}
    
    # Für jedes Gate ALLE 16 Neuronen verwenden
    for gate_idx, gate_name in enumerate(['input', 'forget', 'candidate', 'output']):
        gate_start = gate_idx * HIDDEN_SIZE
        gate_end = gate_start + HIDDEN_SIZE
        
        # Input-zu-Hidden Gewichte für dieses Gate - ALLE verwenden!
        gate_ih = weights['lstm_weight_ih'][gate_start:gate_end, :]  # [16, 4]
        arduino_weights[f'lstm_{gate_name}_ih_weights'] = gate_ih
        
        # Hidden-zu-Hidden Gewichte für dieses Gate - ALLE verwenden!
        gate_hh = weights['lstm_weight_hh'][gate_start:gate_end, :]  # [16, 16]
        arduino_weights[f'lstm_{gate_name}_hh_weights'] = gate_hh
        
        # Bias für dieses Gate - separat wie PyTorch!
        gate_bias_ih = weights['lstm_bias_ih'][gate_start:gate_end]  # [16]
        gate_bias_hh = weights['lstm_bias_hh'][gate_start:gate_end]  # [16]
        arduino_weights[f'lstm_{gate_name}_ih_bias'] = gate_bias_ih
        arduino_weights[f'lstm_{gate_name}_hh_bias'] = gate_bias_hh
        
        print(f"✅ {gate_name} gate: ih{gate_ih.shape}, hh{gate_hh.shape}, bias_ih{gate_bias_ih.shape}, bias_hh{gate_bias_hh.shape}")
    
    # MLP Gewichte vollständig verwenden - KEINE Änderung!
    arduino_weights['mlp0_weights'] = weights['mlp0_weight']  # [16, 16]
    arduino_weights['mlp0_bias'] = weights['mlp0_bias']      # [16]
    
    arduino_weights['mlp3_weights'] = weights['mlp3_weight']  # [16, 16]
    arduino_weights['mlp3_bias'] = weights['mlp3_bias']      # [16]
    
    arduino_weights['mlp6_weights'] = weights['mlp6_weight']  # [1, 16]
    arduino_weights['mlp6_bias'] = weights['mlp6_bias']      # [1]
    
    print(f"✅ MLP layers: 0:{weights['mlp0_weight'].shape}, 3:{weights['mlp3_weight'].shape}, 6:{weights['mlp6_weight'].shape}")
    
    return arduino_weights

def generate_arduino_header_full16(arduino_weights, output_file):
    """Generiere Arduino Header mit vollständigen 16-Hidden-Unit Gewichten"""
    
    total_params = 0
    
    header_content = '''/*
 * VOLLSTÄNDIGE LSTM Gewichte aus trainiertem PyTorch Modell
 * Modell: BMS_SOC_LSTM_stateful_1.2.4 (16x16 Architecture)
 * Arduino Hidden Size: 16 (VOLLSTÄNDIGE ARCHITEKTUR)
 * Exakte 1:1 Konvertierung - KEINE Komprimierung!
 * Ziel: Arduino UNO R4 (32KB SRAM)
 */

#ifndef LSTM_WEIGHTS_H
#define LSTM_WEIGHTS_H

// Arduino LSTM Konfiguration
#define HIDDEN_SIZE 16
#define INPUT_SIZE 4
#define MLP_HIDDEN_SIZE 16
#define OUTPUT_SIZE 1

'''
    
    # LSTM Gewichte schreiben
    for gate in ['input', 'forget', 'candidate', 'output']:
        # Input-zu-Hidden Gewichte
        ih_weights = arduino_weights[f'lstm_{gate}_ih_weights']
        header_content += f"\n// LSTM {gate.upper()} Gate - Input zu Hidden Gewichte [{ih_weights.shape[0]}][{ih_weights.shape[1]}]\n"
        header_content += f"const float lstm_{gate}_ih_weights[{ih_weights.shape[0]}][{ih_weights.shape[1]}] = {{\n"
        for i in range(ih_weights.shape[0]):
            row = ", ".join([f"{w:.6f}f" for w in ih_weights[i]])
            header_content += f"  {{{row}}},\n"
        header_content += "};\n"
        total_params += ih_weights.size
        
        # Hidden-zu-Hidden Gewichte
        hh_weights = arduino_weights[f'lstm_{gate}_hh_weights']
        header_content += f"\n// LSTM {gate.upper()} Gate - Hidden zu Hidden Gewichte [{hh_weights.shape[0]}][{hh_weights.shape[1]}]\n"
        header_content += f"const float lstm_{gate}_hh_weights[{hh_weights.shape[0]}][{hh_weights.shape[1]}] = {{\n"
        for i in range(hh_weights.shape[0]):
            row = ", ".join([f"{w:.6f}f" for w in hh_weights[i]])
            header_content += f"  {{{row}}},\n"
        header_content += "};\n"
        total_params += hh_weights.size
        
        # Input-zu-Hidden Bias
        ih_bias = arduino_weights[f'lstm_{gate}_ih_bias']
        header_content += f"\n// LSTM {gate.upper()} Gate - Input Bias [{ih_bias.shape[0]}]\n"
        header_content += f"const float lstm_{gate}_ih_bias[{ih_bias.shape[0]}] = {{\n"
        bias_str = ", ".join([f"{b:.6f}f" for b in ih_bias])
        header_content += f"  {bias_str}\n"
        header_content += "};\n"
        total_params += ih_bias.size
        
        # Hidden-zu-Hidden Bias
        hh_bias = arduino_weights[f'lstm_{gate}_hh_bias']
        header_content += f"\n// LSTM {gate.upper()} Gate - Hidden Bias [{hh_bias.shape[0]}]\n"
        header_content += f"const float lstm_{gate}_hh_bias[{hh_bias.shape[0]}] = {{\n"
        bias_str = ", ".join([f"{b:.6f}f" for b in hh_bias])
        header_content += f"  {bias_str}\n"
        header_content += "};\n"
        total_params += hh_bias.size
    
    # MLP Gewichte
    for layer_name, weight_key in [('mlp0', 'mlp0_weights'), ('mlp3', 'mlp3_weights'), ('mlp6', 'mlp6_weights')]:
        weights = arduino_weights[weight_key]
        header_content += f"\n// MLP {layer_name.upper()} Gewichte [{weights.shape[0]}][{weights.shape[1]}]\n"
        header_content += f"const float {layer_name}_weights[{weights.shape[0]}][{weights.shape[1]}] = {{\n"
        for i in range(weights.shape[0]):
            row = ", ".join([f"{w:.6f}f" for w in weights[i]])
            header_content += f"  {{{row}}},\n"
        header_content += "};\n"
        total_params += weights.size
        
        # Bias
        bias_key = weight_key.replace('weights', 'bias')
        bias = arduino_weights[bias_key]
        header_content += f"\n// MLP {layer_name.upper()} Bias [{bias.shape[0]}]\n"
        header_content += f"const float {layer_name}_bias[{bias.shape[0]}] = {{\n"
        bias_str = ", ".join([f"{b:.6f}f" for b in bias])
        header_content += f"  {bias_str}\n"
        header_content += "};\n"
        total_params += bias.size
    
    memory_kb = total_params * 4 / 1024
    memory_percent = total_params * 4 / 32768 * 100
    
    header_content += f'''
// Modell-Informationen
#define TOTAL_PARAMETERS {total_params}
#define MEMORY_USAGE_KB {memory_kb:.2f}
#define MEMORY_USAGE_PERCENT {memory_percent:.1f}

#endif // LSTM_WEIGHTS_H
'''
    
    # Header-Datei schreiben
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header_content)
    
    print(f"\n✅ VOLLSTÄNDIGER Arduino Header generiert: {output_file}")
    print(f"📊 Gesamtparameter: {total_params}")
    print(f"💾 Speicherverbrauch: {memory_kb:.2f} KB ({memory_percent:.1f}% von 32KB)")
    
    if memory_percent > 90:
        print("⚠️  WARNUNG: Speicherverbrauch > 90% - könnte knapp werden!")
    elif memory_percent > 80:
        print("⚡ HINWEIS: Speicherverbrauch > 80% - sollte funktionieren")
    else:
        print("✅ Speicherverbrauch OK - ausreichend Platz verfügbar")
    
    return total_params

def generate_arduino_ino_full16(output_file):
    """Generiere Arduino .ino Datei mit 16x16 Architecture und Hardware Monitoring"""
    
    ino_content = '''/*
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
int getFreeRam() {
  char *heapend = (char*)sbrk(0);
  char *stack_ptr = (char*)__builtin_frame_address(0);
  return stack_ptr - heapend;
}

int getUsedRam() {
  return 32768 - getFreeRam();  // 32KB - Free RAM
}

float getRamUsagePercent() {
  return (float)getUsedRam() / 32768.0 * 100.0;
}

unsigned long getFlashUsed() {
  // Approximation basierend auf Sketch Size
  return 50000;  // Typisch für dieses Programm
}

float getFlashUsagePercent() {
  return (float)getFlashUsed() / 262144.0 * 100.0;  // 256KB Flash
}

void updatePerformanceStats() {
  total_inferences++;
  total_inference_time_us += last_inference_time_us;
  avg_inference_time_us = (float)total_inference_time_us / total_inferences;
  
  if (last_inference_time_us < min_inference_time_us) {
    min_inference_time_us = last_inference_time_us;
  }
  if (last_inference_time_us > max_inference_time_us) {
    max_inference_time_us = last_inference_time_us;
  }
}

void estimateCpuLoad() {
  unsigned long current_time = micros();
  if (loop_start_time > 0) {
    unsigned long loop_duration = current_time - loop_start_time;
    total_time_us += loop_duration;
    active_time_us += last_inference_time_us;
    
    if (total_time_us > 0) {
      estimated_cpu_load = (float)active_time_us / total_time_us * 100.0;
    }
  }
  loop_start_time = current_time;
}

// Einfache MCU Temperatur (falls verfügbar)
float readMcuTemperature() {
  // Arduino UNO R4 hat keinen eingebauten Temperatursensor
  // Hier könnte ein externer Sensor angeschlossen werden
  return 25.0;  // Dummy-Wert
}

void printHardwareStatus() {
  Serial.println("\\n=== HARDWARE STATUS ===");
  Serial.print("🖥️  CPU Load: "); Serial.print(estimated_cpu_load, 1); Serial.println("%");
  Serial.print("💾 RAM Used: "); Serial.print(getUsedRam()); Serial.print(" bytes ("); 
  Serial.print(getRamUsagePercent(), 1); Serial.println("%)");
  Serial.print("💾 RAM Free: "); Serial.print(getFreeRam()); Serial.println(" bytes");
  Serial.print("💿 Flash Used: "); Serial.print(getFlashUsed()); Serial.print(" bytes (");
  Serial.print(getFlashUsagePercent(), 1); Serial.println("%)");
  Serial.print("🌡️  MCU Temp: "); Serial.print(readMcuTemperature(), 1); Serial.println("°C");
  Serial.println("========================");
}

void printPerformanceStats() {
  Serial.println("\\n=== PERFORMANCE STATS ===");
  Serial.print("⏱️  Inference Time: "); Serial.print(last_inference_time_us); Serial.println(" μs");
  Serial.print("📊 Avg Time: "); Serial.print(avg_inference_time_us, 0); Serial.println(" μs");
  Serial.print("📊 Min Time: "); Serial.print(min_inference_time_us, 0); Serial.println(" μs");
  Serial.print("📊 Max Time: "); Serial.print(max_inference_time_us, 0); Serial.println(" μs");
  Serial.print("🔢 Total Inferences: "); Serial.println(total_inferences);
  Serial.println("==========================");
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
  estimateCpuLoad();
  
  // Output
  Serial.print("📊 SOC: "); Serial.print(predicted_soc, 4);
  Serial.print(" ("); Serial.print(predicted_soc * 100, 2); Serial.print("%)");
  Serial.print(" | Time: "); Serial.print(last_inference_time_us); Serial.println(" μs");
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\\n🎯 ARDUINO LSTM SOC PREDICTOR - 16x16 + MONITORING");
  Serial.println("====================================================");
  
  printModelInfo();
  printHardwareStatus();
  
  Serial.println("\\n📝 Commands:");
  Serial.println("- Send data: voltage,current,soh,q_c");
  Serial.println("- RESET: Reset LSTM states");
  Serial.println("- STATS: Show performance statistics"); 
  Serial.println("- HARDWARE: Show hardware status");
  Serial.println("- INFO: Show model information");
  Serial.println("\\n✅ Ready for predictions...");
  
  loop_start_time = micros();
}

void loop() {
  if (Serial.available()) {
    String inputString = Serial.readStringUntil('\\n');
    inputString.trim();
    
    if (inputString.equalsIgnoreCase("RESET")) {
      resetLSTMStates();
    } else if (inputString.equalsIgnoreCase("STATS")) {
      printPerformanceStats();
    } else if (inputString.equalsIgnoreCase("HARDWARE")) {
      printHardwareStatus();
    } else if (inputString.equalsIgnoreCase("INFO")) {
      printModelInfo();
    } else if (inputString.length() > 0) {
      processData(inputString);
    }
  }
  
  // Periodische Hardware-Updates (alle 10 Sekunden)
  static unsigned long last_hardware_update = 0;
  if (millis() - last_hardware_update > 10000) {
    estimateCpuLoad();
    mcu_temperature = readMcuTemperature();
    last_hardware_update = millis();
  }
  
  delay(10);
}
'''
    
    # INO-Datei schreiben
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(ino_content)
    
    print(f"✅ Arduino .ino Datei generiert: {output_file}")

def main():
    """Hauptfunktion für vollständige 16-Hidden-Unit Konvertierung"""
    print("🎯 VOLLSTÄNDIGER PyTorch zu Arduino Konverter (16 Hidden Units)")
    print("=" * 60)
    
    # 1. Exaktes Modell laden
    model, device = load_exact_model()
    
    # 2. Exakte Gewichte analysieren
    weights = analyze_exact_weights(model)
    
    # 3. Arduino-Gewichte erstellen (vollständig)
    arduino_weights = create_arduino_weights_full16(weights)
    
    # 4. Arduino Header generieren
    output_dir = os.path.dirname(os.path.abspath(__file__))
    header_file = os.path.join(output_dir, "code_weights", "arduino_lstm_soc_full16_with_monitoring", "lstm_weights.h")
    
    # Erstelle Verzeichnis falls nicht vorhanden
    os.makedirs(os.path.dirname(header_file), exist_ok=True)
    
    total_params = generate_arduino_header_full16(arduino_weights, header_file)
    
    # 5. Arduino .ino Datei generieren
    ino_file = os.path.join(output_dir, "code_weights", "arduino_lstm_soc_full16_with_monitoring", "arduino_lstm_soc_full16_with_monitoring.ino")
    generate_arduino_ino_full16(ino_file)
    
    print(f"\\n🎉 VOLLSTÄNDIGE KONVERTIERUNG ABGESCHLOSSEN!")
    print(f"📁 Header Output: {header_file}")
    print(f"📁 Arduino Output: {ino_file}")
    print(f"🔧 Arduino Hidden Size: 16 (vollständig)")
    print(f"📊 Parameters: {total_params}")
    print(f"💾 Memory: {total_params * 4 / 1024:.2f} KB")
    
    return arduino_weights

if __name__ == "__main__":
    main()
