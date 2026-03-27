"""
PyTorch zu Arduino Konverter - EXAKT wie live_test_soc.py
Lädt die IDENTISCHEN Gewichte und konvertiert sie 1:1 für Arduino
KEINE Komprimierung, KEINE Tricks - nur die pure Wahrheit!
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# EXAKTE Konstanten aus live_test_soc.py
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32
INPUT_SIZE = 4

MODEL_PATH = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\training_run_1_2_4_31\best_model.pth"

# EXAKTE SOCModel Klasse aus live_test_soc.py
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
    """Lade das EXAKTE Modell wie in live_test_soc.py"""
    device = torch.device("cpu")  # Immer CPU für Konsistenz
    
    # Modell erstellen - EXAKT wie live_test_soc.py
    model = SOCModel(input_size=4, dropout=0.05).to(device)
    
    # Gewichte laden
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ EXACT Model loaded from {MODEL_PATH}")
    return model

def extract_lstm_weights(model):
    """Extrahiere ALLE LSTM Gewichte ohne Komprimierung"""
    lstm = model.lstm
    
    # Alle LSTM Parameter
    weight_ih = lstm.weight_ih_l0.detach().numpy()  # (4*hidden, input) = (128, 4)
    weight_hh = lstm.weight_hh_l0.detach().numpy()  # (4*hidden, hidden) = (128, 32)
    bias_ih = lstm.bias_ih_l0.detach().numpy()      # (4*hidden,) = (128,)
    bias_hh = lstm.bias_hh_l0.detach().numpy()      # (4*hidden,) = (128,)
    
    print(f"🎯 LSTM Gewichte:")
    print(f"   weight_ih: {weight_ih.shape}")
    print(f"   weight_hh: {weight_hh.shape}")
    print(f"   bias_ih: {bias_ih.shape}")
    print(f"   bias_hh: {bias_hh.shape}")
    
    return weight_ih, weight_hh, bias_ih, bias_hh

def extract_mlp_weights(model):
    """Extrahiere ALLE MLP Gewichte"""
    mlp = model.mlp
    
    # Layer 0: Linear(32, 32)
    mlp0_weight = mlp[0].weight.detach().numpy()  # (32, 32)
    mlp0_bias = mlp[0].bias.detach().numpy()      # (32,)
    
    # Layer 3: Linear(32, 32) 
    mlp3_weight = mlp[3].weight.detach().numpy()  # (32, 32)
    mlp3_bias = mlp[3].bias.detach().numpy()      # (32,)
    
    # Layer 6: Linear(32, 1)
    mlp6_weight = mlp[6].weight.detach().numpy()  # (1, 32)
    mlp6_bias = mlp[6].bias.detach().numpy()      # (1,)
    
    print(f"🎯 MLP Gewichte:")
    print(f"   mlp0: weight {mlp0_weight.shape}, bias {mlp0_bias.shape}")
    print(f"   mlp3: weight {mlp3_weight.shape}, bias {mlp3_bias.shape}")
    print(f"   mlp6: weight {mlp6_weight.shape}, bias {mlp6_bias.shape}")
    
    return (mlp0_weight, mlp0_bias, mlp3_weight, mlp3_bias, mlp6_weight, mlp6_bias)

def generate_arduino_header(weight_ih, weight_hh, bias_ih, bias_hh, mlp_weights):
    """Generiere Arduino Header mit ALLEN Gewichten"""
    mlp0_weight, mlp0_bias, mlp3_weight, mlp3_bias, mlp6_weight, mlp6_bias = mlp_weights
    
    header = f"""// LSTM Gewichte - EXAKT aus live_test_soc.py best_model.pth
// Generiert am: {Path(__file__).name}
// KEINE Komprimierung - VOLLE 32 Hidden Units!

#ifndef LSTM_WEIGHTS_LIVE_EXACT_H
#define LSTM_WEIGHTS_LIVE_EXACT_H

// Modell-Konstanten (EXAKT wie PyTorch)
#define INPUT_SIZE 4
#define HIDDEN_SIZE 32
#define MLP_HIDDEN 32
#define OUTPUT_SIZE 1

// LSTM Gewichte (Input zu Hidden) - Shape: (4*32, 4) = (128, 4)
const float lstm_weight_ih[{weight_ih.shape[0]}][{weight_ih.shape[1]}] = {{
"""
    
    # LSTM weight_ih
    for i in range(weight_ih.shape[0]):
        header += "  {"
        for j in range(weight_ih.shape[1]):
            header += f"{weight_ih[i,j]:.8f}f"
            if j < weight_ih.shape[1] - 1:
                header += ", "
        header += "}"
        if i < weight_ih.shape[0] - 1:
            header += ","
        header += "\n"
    header += "};\n\n"
    
    # LSTM weight_hh
    header += f"// LSTM Gewichte (Hidden zu Hidden) - Shape: (4*32, 32) = (128, 32)\n"
    header += f"const float lstm_weight_hh[{weight_hh.shape[0]}][{weight_hh.shape[1]}] = {{\n"
    for i in range(weight_hh.shape[0]):
        header += "  {"
        for j in range(weight_hh.shape[1]):
            header += f"{weight_hh[i,j]:.8f}f"
            if j < weight_hh.shape[1] - 1:
                header += ", "
        header += "}"
        if i < weight_hh.shape[0] - 1:
            header += ","
        header += "\n"
    header += "};\n\n"
    
    # LSTM bias_ih
    header += f"// LSTM Bias (Input) - Shape: (128,)\n"
    header += f"const float lstm_bias_ih[{len(bias_ih)}] = {{\n  "
    for i, bias in enumerate(bias_ih):
        header += f"{bias:.8f}f"
        if i < len(bias_ih) - 1:
            header += ", "
        if (i + 1) % 8 == 0:
            header += "\n  "
    header += "\n};\n\n"
    
    # LSTM bias_hh
    header += f"// LSTM Bias (Hidden) - Shape: (128,)\n"
    header += f"const float lstm_bias_hh[{len(bias_hh)}] = {{\n  "
    for i, bias in enumerate(bias_hh):
        header += f"{bias:.8f}f"
        if i < len(bias_hh) - 1:
            header += ", "
        if (i + 1) % 8 == 0:
            header += "\n  "
    header += "\n};\n\n"
    
    # MLP Layer 0 (32->32)
    header += f"// MLP Layer 0 Gewichte (32->32)\n"
    header += f"const float mlp0_weight[{mlp0_weight.shape[0]}][{mlp0_weight.shape[1]}] = {{\n"
    for i in range(mlp0_weight.shape[0]):
        header += "  {"
        for j in range(mlp0_weight.shape[1]):
            header += f"{mlp0_weight[i,j]:.8f}f"
            if j < mlp0_weight.shape[1] - 1:
                header += ", "
        header += "}"
        if i < mlp0_weight.shape[0] - 1:
            header += ","
        header += "\n"
    header += "};\n\n"
    
    header += f"const float mlp0_bias[{len(mlp0_bias)}] = {{\n  "
    for i, bias in enumerate(mlp0_bias):
        header += f"{bias:.8f}f"
        if i < len(mlp0_bias) - 1:
            header += ", "
        if (i + 1) % 8 == 0:
            header += "\n  "
    header += "\n};\n\n"
    
    # MLP Layer 3 (32->32)
    header += f"// MLP Layer 3 Gewichte (32->32)\n"
    header += f"const float mlp3_weight[{mlp3_weight.shape[0]}][{mlp3_weight.shape[1]}] = {{\n"
    for i in range(mlp3_weight.shape[0]):
        header += "  {"
        for j in range(mlp3_weight.shape[1]):
            header += f"{mlp3_weight[i,j]:.8f}f"
            if j < mlp3_weight.shape[1] - 1:
                header += ", "
        header += "}"
        if i < mlp3_weight.shape[0] - 1:
            header += ","
        header += "\n"
    header += "};\n\n"
    
    header += f"const float mlp3_bias[{len(mlp3_bias)}] = {{\n  "
    for i, bias in enumerate(mlp3_bias):
        header += f"{bias:.8f}f"
        if i < len(mlp3_bias) - 1:
            header += ", "
        if (i + 1) % 8 == 0:
            header += "\n  "
    header += "\n};\n\n"
    
    # MLP Layer 6 (32->1)
    header += f"// MLP Layer 6 Gewichte (32->1)\n"
    header += f"const float mlp6_weight[{mlp6_weight.shape[0]}][{mlp6_weight.shape[1]}] = {{\n"
    for i in range(mlp6_weight.shape[0]):
        header += "  {"
        for j in range(mlp6_weight.shape[1]):
            header += f"{mlp6_weight[i,j]:.8f}f"
            if j < mlp6_weight.shape[1] - 1:
                header += ", "
        header += "}"
        if i < mlp6_weight.shape[0] - 1:
            header += ","
        header += "\n"
    header += "};\n\n"
    
    header += f"const float mlp6_bias[{len(mlp6_bias)}] = {{\n  "
    for i, bias in enumerate(mlp6_bias):
        header += f"{bias:.8f}f"
        if i < len(mlp6_bias) - 1:
            header += ", "
    header += "\n};\n\n"
    
    header += "#endif // LSTM_WEIGHTS_LIVE_EXACT_H\n"
    
    return header

def main():
    """Hauptfunktion - Lade Modell und generiere Arduino Code"""
    print("🚀 Starting PyTorch zu Arduino Konvertierung - LIVE EXACT VERSION")
    print("🎯 KEINE Komprimierung - VOLLE Power!")
    
    # Modell laden
    model = load_exact_model()
    
    # Gewichte extrahieren
    print("\n📊 Extrahiere LSTM Gewichte...")
    weight_ih, weight_hh, bias_ih, bias_hh = extract_lstm_weights(model)
    
    print("\n📊 Extrahiere MLP Gewichte...")
    mlp_weights = extract_mlp_weights(model)
    
    # Arduino Header generieren
    print("\n🔧 Generiere Arduino Header...")
    header_content = generate_arduino_header(weight_ih, weight_hh, bias_ih, bias_hh, mlp_weights)
    
    # Header speichern
    header_path = Path("lstm_weights_live_exact.h")
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    # Statistiken
    total_params = (weight_ih.size + weight_hh.size + bias_ih.size + bias_hh.size + 
                   mlp_weights[0].size + mlp_weights[1].size + 
                   mlp_weights[2].size + mlp_weights[3].size + 
                   mlp_weights[4].size + mlp_weights[5].size)
    
    memory_kb = total_params * 4 / 1024  # 4 bytes per float
    
    print(f"\n✅ Arduino Header generiert: {header_path}")
    print(f"📊 Modell Statistiken:")
    print(f"   🧠 LSTM Hidden Units: {HIDDEN_SIZE} (VOLL!)")
    print(f"   📝 Gesamt Parameter: {total_params:,}")
    print(f"   💾 Speicherbedarf: {memory_kb:.2f} KB")
    print(f"   🎯 Architektur: LSTM({INPUT_SIZE}→{HIDDEN_SIZE}) + MLP({HIDDEN_SIZE}→{MLP_HIDDEN}→{MLP_HIDDEN}→1)")
    print(f"\n🔥 READY für Arduino Upload - KEINE KOMPROMISSE!")

if __name__ == "__main__":
    main()
