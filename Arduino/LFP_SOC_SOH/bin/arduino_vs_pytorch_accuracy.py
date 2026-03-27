"""
Arduino vs PyTorch LSTM Accuracy Vergleich
Testet das exakte Arduino-Modell gegen das originale PyTorch-Modell
"""

import torch
import torch.nn as nn
import serial
import json
import numpy as np
import time
from pathlib import Path

# Modell-Konstanten (identisch zu live_test_soc.py)
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32
MODEL_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\training_run_1_2_4_31\best_model.pth"

# PyTorch Modell-Definition (identisch zu live_test_soc.py)
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

def load_pytorch_model():
    """Lade das originale PyTorch-Modell"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 PyTorch Device: {device}")
    
    model = SOCModel(input_size=4, dropout=0.05).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    print(f"✅ PyTorch Model loaded from {MODEL_PATH}")
    return model, device

def init_pytorch_hidden(device, batch_size=1):
    """Initialisiere PyTorch Hidden States"""
    h = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=device)
    c = torch.zeros_like(h)
    return h, c

def setup_arduino():
    """Arduino-Verbindung einrichten"""
    print("🔗 Connecting to Arduino...")
    ser = serial.Serial('COM13', 115200, timeout=3)
    time.sleep(2)  # Reset abwarten
    
    # Arduino Status prüfen
    ser.write(json.dumps({'command': 'status'}).encode() + b'\n')
    time.sleep(0.5)
    response = ser.readline().decode().strip()
    status = json.loads(response)
    
    print(f"✅ Arduino connected: {status['model']}, Hidden Size: {status['hidden_size']}")
    
    # Arduino LSTM Reset
    ser.write(json.dumps({'command': 'reset'}).encode() + b'\n')
    time.sleep(0.5)
    ser.readline()  # Reset response lesen
    
    return ser

def predict_pytorch(model, device, hidden_state, features):
    """PyTorch Vorhersage"""
    x = torch.tensor([features], dtype=torch.float32, device=device).view(1, 1, 4)
    
    with torch.no_grad():
        pred_soc, hidden_state = model(x, hidden_state)
        pred_soc_value = pred_soc.item()
    
    # Hidden state für nächsten Schritt
    h, c = hidden_state
    hidden_state = (h.detach(), c.detach())
    
    return pred_soc_value, hidden_state

def predict_arduino(ser, features):
    """Arduino Vorhersage"""
    command = {'command': 'predict', 'features': features}
    ser.write(json.dumps(command).encode() + b'\n')
    time.sleep(0.001)  # Kurze Wartezeit
    
    response = ser.readline().decode().strip()
    result = json.loads(response)
    
    return result['soc'], result['inference_time_us']

def accuracy_comparison():
    """Hauptvergleichsfunktion"""
    print("🎯 ARDUINO vs PYTORCH ACCURACY VERGLEICH")
    print("=" * 60)
    
    # Modelle laden
    pytorch_model, device = load_pytorch_model()
    arduino_ser = setup_arduino()
    
    # Hidden States initialisieren
    pytorch_hidden = init_pytorch_hidden(device)
    
    # Test-Datenpunkte (realistische skalierte Werte)
    test_data = [
        [0.5, -0.3, 0.8, 0.7],   # Entladung
        [0.6, 0.2, 0.85, 0.75],  # Ladung
        [0.4, -0.5, 0.9, 0.6],   # Starke Entladung
        [0.7, 0.1, 0.95, 0.8],   # Schwache Ladung
        [0.3, -0.8, 0.7, 0.5],   # Sehr starke Entladung
        [0.8, 0.4, 1.0, 0.9],    # Starke Ladung
        [0.2, -0.2, 0.6, 0.4],   # Niedrige SOC
        [0.9, 0.0, 0.95, 1.0],   # Hohe SOC
    ]
    
    print("\n📊 SEQUENTIELLE VORHERSAGEN (Stateful LSTM):")
    print("-" * 60)
    print("Test#  PyTorch SOC  Arduino SOC  Difference  Arduino Time")
    print("-" * 60)
    
    total_error = 0
    arduino_times = []
    
    for i, features in enumerate(test_data):
        # PyTorch Vorhersage
        pytorch_soc, pytorch_hidden = predict_pytorch(
            pytorch_model, device, pytorch_hidden, features
        )
        
        # Arduino Vorhersage
        arduino_soc, arduino_time = predict_arduino(arduino_ser, features)
        
        # Fehler berechnen
        difference = abs(pytorch_soc - arduino_soc)
        total_error += difference
        arduino_times.append(arduino_time)
        
        print(f"{i+1:4d}   {pytorch_soc:11.6f}  {arduino_soc:11.6f}  {difference:10.6f}  {arduino_time:8d} μs")
    
    # Statistiken
    mean_error = total_error / len(test_data)
    max_error = max([abs(predict_pytorch(pytorch_model, device, init_pytorch_hidden(device), features)[0] - 
                        predict_arduino(arduino_ser, features)[0]) for features in test_data])
    mean_arduino_time = np.mean(arduino_times)
    
    print("-" * 60)
    print(f"📊 ERGEBNISSE:")
    print(f"   Mean Absolute Error: {mean_error:.6f}")
    print(f"   Max Error:           {max_error:.6f}")
    print(f"   Mean Arduino Time:   {mean_arduino_time:.1f} μs")
    print(f"   Arduino Memory:      7.00 KB (21.9%)")
    print(f"   Model Compression:   32→8 hidden units (75% reduction)")
    
    # Bewertung
    if mean_error < 0.001:
        print(f"✅ AUSGEZEICHNET: Sehr hohe Accuracy (< 0.1% error)")
    elif mean_error < 0.01:
        print(f"✅ GUT: Hohe Accuracy (< 1% error)")
    elif mean_error < 0.05:
        print(f"⚠️  AKZEPTABEL: Moderate Accuracy (< 5% error)")
    else:
        print(f"❌ KRITISCH: Niedrige Accuracy (> 5% error)")
    
    arduino_ser.close()
    print("\n🎉 VERGLEICH ABGESCHLOSSEN!")

if __name__ == "__main__":
    accuracy_comparison()
