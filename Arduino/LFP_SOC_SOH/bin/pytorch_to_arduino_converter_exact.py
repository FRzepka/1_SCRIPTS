"""
PyTorch zu Arduino Konverter - EXAKT für dein trainiertes Modell
Extrahiert die exakten Gewichte aus best_model.pth und konvertiert sie 1:1
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

# LIVE EXACT - VOLLE ARCHITEKTUR!
ARDUINO_HIDDEN_SIZE = 32  # VOLLE 32 Hidden Units wie live_test_soc.py
TARGET_NEURONS = 32       # KEINE Reduktion - vollständige Architektur!

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
    
    print(f"✅ EXACT Model loaded from {MODEL_PATH}")
    return model, device

def analyze_exact_weights(model):
    """Analysiere die exakten Gewichte"""
    print("\n🔍 EXAKTE GEWICHTS-ANALYSE:")
    
    # LSTM Gewichte
    lstm_weight_ih = model.lstm.weight_ih_l0.data.numpy()  # [128, 4]
    lstm_weight_hh = model.lstm.weight_hh_l0.data.numpy()  # [128, 32]
    lstm_bias_ih = model.lstm.bias_ih_l0.data.numpy()      # [128]
    lstm_bias_hh = model.lstm.bias_hh_l0.data.numpy()      # [128]
    
    print(f"📊 LSTM weight_ih: {lstm_weight_ih.shape} = [4*hidden_size*4, input_size]")
    print(f"📊 LSTM weight_hh: {lstm_weight_hh.shape} = [4*hidden_size*4, hidden_size]")
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

def select_important_neurons_exact(weights):
    """Für vollständige 32-Neuronen Implementation - KEINE Selektion!"""
    
    print(f"\n🧠 VOLLSTÄNDIGE 32-NEURONEN ARCHITEKTUR:")
    print(f"🎯 Verwende ALLE 32 Hidden Units - keine Komprimierung!")
    
    # Alle Neuronen verwenden (0 bis 31)
    all_neurons = np.arange(32)
    print(f"🎯 Using all neurons: {list(all_neurons)}")
    
    return all_neurons

def create_arduino_weights_exact(weights):
    """Erstelle Arduino-Gewichte für vollständige 32-Hidden-Architecture"""
    
    print(f"\n⚙️ ARDUINO-GEWICHTE ERSTELLEN (VOLLE 32 HIDDEN UNITS):")
    
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
        arduino_weights[f'lstm_{gate_name}_ih'] = gate_ih
        
        # Hidden-zu-Hidden Gewichte für dieses Gate - ALLE verwenden!
        gate_hh = weights['lstm_weight_hh'][gate_start:gate_end, :]  # [32, 32]
        arduino_weights[f'lstm_{gate_name}_hh'] = gate_hh
        
        # Bias für dieses Gate - KOMBINIERT wie PyTorch!
        gate_bias_ih = weights['lstm_bias_ih'][gate_start:gate_end]  # [32]
        gate_bias_hh = weights['lstm_bias_hh'][gate_start:gate_end]  # [32]
        arduino_weights[f'lstm_{gate_name}_ih_bias'] = gate_bias_ih
        arduino_weights[f'lstm_{gate_name}_hh_bias'] = gate_bias_hh
        
        print(f"✅ {gate_name} gate: ih{gate_ih.shape}, hh{gate_hh.shape}, bias ih{gate_bias_ih.shape}, bias hh{gate_bias_hh.shape}")
    
    # MLP Gewichte vollständig verwenden - KEINE Änderung!
    arduino_weights['mlp0_weight'] = weights['mlp0_weight']  # [32, 32]
    arduino_weights['mlp0_bias'] = weights['mlp0_bias']      # [32]
    
    arduino_weights['mlp3_weight'] = weights['mlp3_weight']  # [32, 32]
    arduino_weights['mlp3_bias'] = weights['mlp3_bias']      # [32]
    
    arduino_weights['mlp6_weight'] = weights['mlp6_weight']  # [1, 32]
    arduino_weights['mlp6_bias'] = weights['mlp6_bias']      # [1]
    
    print(f"✅ MLP layers: 0:{weights['mlp0_weight'].shape}, 3:{weights['mlp3_weight'].shape}, 6:{weights['mlp6_weight'].shape}")
    
    return arduino_weights

def generate_arduino_header_exact(arduino_weights, output_file):
    """Generiere Arduino Header mit exakten Gewichten"""
    
    total_params = 0
    
    header_content = '''/*
 * EXAKTE LSTM Gewichte aus trainiertem PyTorch Modell
 * Modell: BMS_SOC_LSTM_stateful_1.2.4.31
 * Arduino Hidden Size: 8 (reduziert von 32)
 * Intelligente Neuron-Selektion basierend auf MLP-Wichtigkeit
 */

#ifndef LSTM_WEIGHTS_EXACT_H
#define LSTM_WEIGHTS_EXACT_H

// Arduino LSTM Konfiguration
#define ARDUINO_HIDDEN_SIZE 8
#define INPUT_SIZE 4
#define MLP_HIDDEN_SIZE 32

'''
    
    # LSTM Gewichte schreiben
    for gate in ['input', 'forget', 'candidate', 'output']:
        # Input-zu-Hidden Gewichte
        ih_weights = arduino_weights[f'lstm_{gate}_ih']
        header_content += f"\n// LSTM {gate.upper()} Gate - Input zu Hidden Gewichte [{ih_weights.shape[0]}][{ih_weights.shape[1]}]\n"
        header_content += f"const float lstm_{gate}_ih_weights[{ih_weights.shape[0]}][{ih_weights.shape[1]}] = {{\n"
        for i in range(ih_weights.shape[0]):
            row = ", ".join([f"{w:.6f}f" for w in ih_weights[i]])
            header_content += f"  {{{row}}},\n"
        header_content += "};\n"
        total_params += ih_weights.size
        
        # Hidden-zu-Hidden Gewichte
        hh_weights = arduino_weights[f'lstm_{gate}_hh']
        header_content += f"\n// LSTM {gate.upper()} Gate - Hidden zu Hidden Gewichte [{hh_weights.shape[0]}][{hh_weights.shape[1]}]\n"
        header_content += f"const float lstm_{gate}_hh_weights[{hh_weights.shape[0]}][{hh_weights.shape[1]}] = {{\n"
        for i in range(hh_weights.shape[0]):
            row = ", ".join([f"{w:.6f}f" for w in hh_weights[i]])
            header_content += f"  {{{row}}},\n"
        header_content += "};\n"
        total_params += hh_weights.size
        
        # Bias
        bias = arduino_weights[f'lstm_{gate}_bias']
        header_content += f"\n// LSTM {gate.upper()} Gate - Bias [{bias.shape[0]}]\n"
        header_content += f"const float lstm_{gate}_bias[{bias.shape[0]}] = {{\n"
        bias_str = ", ".join([f"{b:.6f}f" for b in bias])
        header_content += f"  {bias_str}\n"
        header_content += "};\n"
        total_params += bias.size
    
    # MLP Gewichte
    for layer_name, weight_key in [('mlp0', 'mlp0_weight'), ('mlp3', 'mlp3_weight'), ('mlp6', 'mlp6_weight')]:
        weights = arduino_weights[weight_key]
        header_content += f"\n// MLP {layer_name.upper()} Gewichte [{weights.shape[0]}][{weights.shape[1]}]\n"
        header_content += f"const float {layer_name}_weights[{weights.shape[0]}][{weights.shape[1]}] = {{\n"
        for i in range(weights.shape[0]):
            row = ", ".join([f"{w:.6f}f" for w in weights[i]])
            header_content += f"  {{{row}}},\n"
        header_content += "};\n"
        total_params += weights.size
        
        # Bias
        bias_key = weight_key.replace('weight', 'bias')
        bias = arduino_weights[bias_key]
        header_content += f"\n// MLP {layer_name.upper()} Bias [{bias.shape[0]}]\n"
        header_content += f"const float {layer_name}_bias[{bias.shape[0]}] = {{\n"
        bias_str = ", ".join([f"{b:.6f}f" for b in bias])
        header_content += f"  {bias_str}\n"
        header_content += "};\n"
        total_params += bias.size
    
    header_content += f'''
// Modell-Informationen
#define TOTAL_PARAMETERS {total_params}
#define MEMORY_USAGE_KB {total_params * 4 / 1024:.2f}
#define MEMORY_USAGE_PERCENT {total_params * 4 / 32768 * 100:.1f}

#endif // LSTM_WEIGHTS_EXACT_H
'''
    
    # Header-Datei schreiben
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header_content)
    
    print(f"\n✅ EXAKTE Arduino Header generiert: {output_file}")
    print(f"📊 Gesamtparameter: {total_params}")
    print(f"💾 Speicherverbrauch: {total_params * 4 / 1024:.2f} KB ({total_params * 4 / 32768 * 100:.1f}% von 32KB)")
    
    return total_params

def main():
    """Hauptfunktion für exakte Konvertierung"""
    print("🎯 EXAKTER PyTorch zu Arduino Konverter")
    print("=" * 50)
    
    # 1. Exaktes Modell laden
    model, device = load_exact_model()
    
    # 2. Exakte Gewichte analysieren
    weights = analyze_exact_weights(model)
    
    # 3. Wichtige Neuronen auswählen (basierend auf MLP6-Gewichten)
    selected_neurons = select_important_neurons_exact(weights)
    
    # 4. Arduino-Gewichte erstellen
    arduino_weights = create_arduino_weights_exact(weights, selected_neurons)
    
    # 5. Arduino Header generieren
    output_file = "lstm_weights_exact.h"
    total_params = generate_arduino_header_exact(arduino_weights, output_file)
    
    print(f"\n🎉 EXAKTE KONVERTIERUNG ABGESCHLOSSEN!")
    print(f"📁 Output: {output_file}")
    print(f"🔧 Arduino Hidden Size: {TARGET_NEURONS}")
    print(f"📊 Parameters: {total_params}")
    print(f"💾 Memory: {total_params * 4 / 1024:.2f} KB")
    
    return arduino_weights, selected_neurons

if __name__ == "__main__":
    main()
