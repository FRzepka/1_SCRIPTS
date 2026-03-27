"""
PyTorch zu Arduino Konverter - VOLLSTÄNDIGE 32 Hidden Units
Extrahiert die exakten Gewichte aus best_model.pth für Arduino UNO R4
KEINE Komprimierung - vollständige Modell-Architektur!
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Exakte Modell-Konstanten aus dem Training
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32
MODEL_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\training_run_1_2_4_31\best_model.pth"

# Exakte Modell-Definition (identisch zum Training)
class SOCModel(nn.Module):
    def __init__(self, input_size=4, dropout=0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=0.0)
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
    """Lade das exakte trainierte Modell"""
    device = torch.device("cpu")  # CPU für Konvertierung
    
    # Modell erstellen
    model = SOCModel(input_size=4, dropout=0.05).to(device)
    
    # Exakte Gewichte laden
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    print(f"✅ VOLLSTÄNDIGES Model loaded from {MODEL_PATH}")
    return model, device

def analyze_exact_weights(model):
    """Analysiere die exakten Gewichte"""
    print("\n🔍 VOLLSTÄNDIGE GEWICHTS-ANALYSE (32 Hidden Units):")
    
    # LSTM Gewichte
    lstm_weight_ih = model.lstm.weight_ih_l0.data.numpy()  # [128, 4]
    lstm_weight_hh = model.lstm.weight_hh_l0.data.numpy()  # [128, 32]
    lstm_bias_ih = model.lstm.bias_ih_l0.data.numpy()      # [128]
    lstm_bias_hh = model.lstm.bias_hh_l0.data.numpy()      # [128]
    
    print(f"📊 LSTM weight_ih: {lstm_weight_ih.shape} = [4*hidden_size, input_size]")
    print(f"📊 LSTM weight_hh: {lstm_weight_hh.shape} = [4*hidden_size, hidden_size]")
    print(f"📊 LSTM bias_ih: {lstm_bias_ih.shape}")
    print(f"📊 LSTM bias_hh: {lstm_bias_hh.shape}")
    
    # MLP Gewichte
    mlp0_weight = model.mlp[0].weight.data.numpy()  # [32, 32]
    mlp0_bias = model.mlp[0].bias.data.numpy()      # [32]
    mlp3_weight = model.mlp[3].weight.data.numpy()  # [32, 32]
    mlp3_bias = model.mlp[3].bias.data.numpy()      # [32]
    mlp6_weight = model.mlp[6].weight.data.numpy()  # [1, 32]
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

def create_arduino_weights_full32(weights):
    """Erstelle Arduino-Gewichte für vollständige 32-Hidden-Architecture"""
    
    print(f"\n⚙️ ARDUINO-GEWICHTE ERSTELLEN (VOLLSTÄNDIGE 32 HIDDEN UNITS):")
    
    # LSTM Gewichte komplett verwenden - KEINE Reduktion!
    # PyTorch LSTM Layout: [4*hidden_size, input_size] = [128, 4]
    # Gates: input, forget, candidate, output (je 32 Neuronen)
    
    arduino_weights = {}
    
    # Für jedes Gate ALLE 32 Neuronen verwenden
    for gate_idx, gate_name in enumerate(['input', 'forget', 'candidate', 'output']):
        gate_start = gate_idx * HIDDEN_SIZE
        gate_end = gate_start + HIDDEN_SIZE
        
        # Input-zu-Hidden Gewichte für dieses Gate - ALLE verwenden!
        gate_ih = weights['lstm_weight_ih'][gate_start:gate_end, :]  # [32, 4]
        arduino_weights[f'lstm_{gate_name}_ih_weights'] = gate_ih
        
        # Hidden-zu-Hidden Gewichte für dieses Gate - ALLE verwenden!
        gate_hh = weights['lstm_weight_hh'][gate_start:gate_end, :]  # [32, 32]
        arduino_weights[f'lstm_{gate_name}_hh_weights'] = gate_hh
        
        # Bias für dieses Gate - separat wie PyTorch!
        gate_bias_ih = weights['lstm_bias_ih'][gate_start:gate_end]  # [32]
        gate_bias_hh = weights['lstm_bias_hh'][gate_start:gate_end]  # [32]
        arduino_weights[f'lstm_{gate_name}_ih_bias'] = gate_bias_ih
        arduino_weights[f'lstm_{gate_name}_hh_bias'] = gate_bias_hh
        
        print(f"✅ {gate_name} gate: ih{gate_ih.shape}, hh{gate_hh.shape}, bias_ih{gate_bias_ih.shape}, bias_hh{gate_bias_hh.shape}")
    
    # MLP Gewichte vollständig verwenden - KEINE Änderung!
    arduino_weights['mlp0_weights'] = weights['mlp0_weight']  # [32, 32]
    arduino_weights['mlp0_bias'] = weights['mlp0_bias']      # [32]
    
    arduino_weights['mlp3_weights'] = weights['mlp3_weight']  # [32, 32]
    arduino_weights['mlp3_bias'] = weights['mlp3_bias']      # [32]
    
    arduino_weights['mlp6_weights'] = weights['mlp6_weight']  # [1, 32]
    arduino_weights['mlp6_bias'] = weights['mlp6_bias']      # [1]
    
    print(f"✅ MLP layers: 0:{weights['mlp0_weight'].shape}, 3:{weights['mlp3_weight'].shape}, 6:{weights['mlp6_weight'].shape}")
    
    return arduino_weights

def generate_arduino_header_full32(arduino_weights, output_file):
    """Generiere Arduino Header mit vollständigen 32-Hidden-Unit Gewichten"""
    
    total_params = 0
    
    header_content = '''/*
 * VOLLSTÄNDIGE LSTM Gewichte aus trainiertem PyTorch Modell
 * Modell: BMS_SOC_LSTM_stateful_1.2.4.31
 * Arduino Hidden Size: 32 (VOLLSTÄNDIGE ARCHITEKTUR)
 * Exakte 1:1 Konvertierung - KEINE Komprimierung!
 * Ziel: Arduino UNO R4 (32KB SRAM)
 */

#ifndef LSTM_WEIGHTS_EXACT_H
#define LSTM_WEIGHTS_EXACT_H

// Arduino LSTM Konfiguration
#define ARDUINO_HIDDEN_SIZE 32
#define INPUT_SIZE 4
#define MLP_HIDDEN_SIZE 32

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

#endif // LSTM_WEIGHTS_EXACT_H
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

def main():
    """Hauptfunktion für vollständige 32-Hidden-Unit Konvertierung"""
    print("🎯 VOLLSTÄNDIGER PyTorch zu Arduino Konverter (32 Hidden Units)")
    print("=" * 60)
    
    # 1. Exaktes Modell laden
    model, device = load_exact_model()
    
    # 2. Exakte Gewichte analysieren
    weights = analyze_exact_weights(model)
    
    # 3. Arduino-Gewichte erstellen (vollständig)
    arduino_weights = create_arduino_weights_full32(weights)
    
    # 4. Arduino Header generieren
    output_file = "lstm_weights_exact.h"
    total_params = generate_arduino_header_full32(arduino_weights, output_file)
    
    print(f"\n🎉 VOLLSTÄNDIGE KONVERTIERUNG ABGESCHLOSSEN!")
    print(f"📁 Output: {output_file}")
    print(f"🔧 Arduino Hidden Size: 32 (vollständig)")
    print(f"📊 Parameters: {total_params}")
    print(f"💾 Memory: {total_params * 4 / 1024:.2f} KB")
    
    return arduino_weights

if __name__ == "__main__":
    main()
