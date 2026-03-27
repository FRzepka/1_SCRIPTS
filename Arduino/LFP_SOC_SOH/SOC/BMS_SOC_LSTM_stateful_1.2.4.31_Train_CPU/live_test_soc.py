"""
BMS SOC LSTM Live Test Script
Lädt das trainierte Modell und macht stateful SOC-Vorhersagen
mit Live-Plot der eingehenden Daten
"""

import torch
import torch.nn as nn
import socket
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from pathlib import Path
from collections import deque
import time
import threading
import queue
import gc  # Added for garbage collection
import psutil  # Added for performance monitoring
import logging  # Added for logging

# Configure logging for performance monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modell-Konstanten (müssen mit Training übereinstimmen)
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32
MODEL_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\training_run_1_2_4_31\best_model.pth"

# Socket-Konstanten
HOST = 'localhost'
PORT = 12345

# Plot-Konstanten - OPTIMIZED
MAX_POINTS = 500  # Reduced from 2000 for better memory efficiency
UPDATE_INTERVAL = 100  # ms - Slightly increased for better performance

# Performance monitoring constants
PERFORMANCE_LOG_INTERVAL = 1000  # Log performance every N points
MEMORY_CLEANUP_INTERVAL = 5000   # Force garbage collection every N points

# Globale Datenstrukturen - OPTIMIZED
data_queue = queue.Queue(maxsize=200)  # BOUNDED QUEUE to prevent memory leaks
plot_data = {
    'timestamps': deque(maxlen=MAX_POINTS),
    'true_soc': deque(maxlen=MAX_POINTS),
    'pred_soc': deque(maxlen=MAX_POINTS),
    'voltage': deque(maxlen=MAX_POINTS),
    'current': deque(maxlen=MAX_POINTS),
    'error': deque(maxlen=MAX_POINTS)
}

# Performance tracking
performance_stats = {
    'processed_points': 0,
    'start_time': time.time(),
    'processing_times': deque(maxlen=100),
    'memory_usage': deque(maxlen=100)
}

# Modell-Definition (identisch zum Training)
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

def load_model():
    """Lade das trainierte Modell"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")
    
    # Modell erstellen
    model = SOCModel(input_size=4, dropout=0.05).to(device)
    
    # Gewichte laden
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    print(f"✅ Model loaded from {MODEL_PATH}")
    return model, device

def init_hidden(device, batch_size=1):
    """Initialisiere Hidden States"""
    h = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=device)
    c = torch.zeros_like(h)
    return h, c

def data_receiver_thread():
    """Thread für Datenempfang"""
    print("🔌 Connecting to data sender...")
    
    while True:
        try:
            # Verbindung zum Data Sender
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))
            print(f"✅ Connected to data sender at {HOST}:{PORT}")
            
            buffer = ""
            while True:
                # Daten empfangen
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                  # Verarbeite komplette JSON-Zeilen
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        data_packet = json.loads(line)
                        # NON-BLOCKING queue operation with fallback
                        try:
                            data_queue.put_nowait(data_packet)
                        except queue.Full:
                            # Queue is full, drop oldest data
                            try:
                                data_queue.get_nowait()  # Remove oldest
                                data_queue.put_nowait(data_packet)  # Add new
                            except queue.Empty:
                                pass  # Queue became empty, try again
                    except json.JSONDecodeError:
                        continue
                        
        except ConnectionRefusedError:
            print("❌ Connection refused. Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Data receiver error: {e}")
            time.sleep(2)
        finally:
            try:
                client_socket.close()
            except:
                pass

def process_data_point(model, device, hidden_state, data_packet):
    """Verarbeite einen Datenpunkt durch das Modell"""
    # Eingabedaten extrahieren
    voltage = data_packet['voltage']
    current = data_packet['current']
    soh = data_packet['soh']
    q_c = data_packet['q_c']
    true_soc = data_packet['true_soc']
    
    # Tensor erstellen (1, 1, 4) für single timestep
    x = torch.tensor([[voltage, current, soh, q_c]], 
                     dtype=torch.float32, device=device).view(1, 1, 4)
    
    # Vorhersage
    with torch.no_grad():
        pred_soc, hidden_state = model(x, hidden_state)
        pred_soc_value = pred_soc.item()
    
    # Fehler berechnen
    error = abs(pred_soc_value - true_soc)
    
    # Hidden state für nächsten Schritt vorbereiten
    h, c = hidden_state
    hidden_state = (h.detach(), c.detach())
    
    return pred_soc_value, error, hidden_state

def update_plot_data(data_packet, pred_soc, error):
    """Aktualisiere Plot-Daten"""
    timestamp = pd.to_datetime(data_packet['timestamp'])
    
    plot_data['timestamps'].append(timestamp)
    plot_data['true_soc'].append(data_packet['true_soc'])
    plot_data['pred_soc'].append(pred_soc)
    plot_data['voltage'].append(data_packet['voltage'])
    plot_data['current'].append(data_packet['current'])
    plot_data['error'].append(error)

def animate_plots(frame, model, device, hidden_state):
    """Animation-Funktion für Live-Plots - OPTIMIZED"""
    global plot_data
    # Verarbeite verfügbare Datenpunkte mit Begrenzung
    processed_count = 0
    BATCH_PROCESS_SIZE = 20  # Optimized batch size
    while not data_queue.empty() and processed_count < BATCH_PROCESS_SIZE:
        try:
            data_packet = data_queue.get_nowait()
            
            # Modell-Vorhersage
            pred_soc, error, hidden_state[0] = process_data_point(
                model, device, hidden_state[0], data_packet
            )
            
            # Plot-Daten aktualisieren
            update_plot_data(data_packet, pred_soc, error)
            
            processed_count += 1
              # Status ausgeben
            if data_packet['index'] % 500 == 0:  # Weniger häufige Ausgaben bei höherer Geschwindigkeit
                print(f"🚀 Point {data_packet['index']}/{data_packet['total_points']} - "
                      f"True SOC: {data_packet['true_soc']:.3f}, "
                      f"Pred SOC: {pred_soc:.3f}, "
                      f"Error: {error:.4f} - SPEED: 10ms")
                
        except queue.Empty:
            break
    
    # Plots aktualisieren nur wenn Daten vorhanden
    if len(plot_data['timestamps']) > 0:
        # SOC Vergleich Plot
        ax1.clear()
        times = list(plot_data['timestamps'])
        true_socs = list(plot_data['true_soc'])
        pred_socs = list(plot_data['pred_soc'])
        
        ax1.plot(times, true_socs, 'b-', label='True SOC', linewidth=2)
        ax1.plot(times, pred_socs, 'r--', label='Predicted SOC', linewidth=2)
        ax1.set_ylabel('SOC')
        ax1.set_title('Live SOC Prediction vs Ground Truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fehler Plot
        ax2.clear()
        errors = list(plot_data['error'])
        ax2.plot(times, errors, 'g-', label='Absolute Error', linewidth=1)
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Prediction Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Spannung und Strom Plot
        ax3.clear()
        voltages = list(plot_data['voltage'])
        ax3.plot(times, voltages, 'orange', label='Voltage (scaled)', linewidth=1)
        ax3.set_ylabel('Voltage (scaled)')
        ax3.set_xlabel('Time')
        ax3.set_title('Input Voltage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.clear()
        currents = list(plot_data['current'])
        ax4.plot(times, currents, 'purple', label='Current (scaled)', linewidth=1)
        ax4.set_ylabel('Current (scaled)')
        ax4.set_xlabel('Time')
        ax4.set_title('Input Current')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
          # Statistiken berechnen und anzeigen
        if len(errors) > 10:
            mean_error = np.mean(errors[-500:])  # Mehr Punkte für Statistiken
            rmse = np.sqrt(np.mean(np.array(errors[-500:])**2))
            fig.suptitle(f'Live SOC Prediction (10ms) - Mean Error: {mean_error:.4f}, RMSE: {rmse:.4f} - Points: {len(errors)}')

def main():
    """Hauptfunktion"""
    global fig, ax1, ax2, ax3, ax4
    
    print("🚀 Starting BMS SOC Live Test...")
    
    # Modell laden
    model, device = load_model()
    
    # Hidden State initialisieren (in Liste für Referenz-Übergabe)
    hidden_state = [init_hidden(device)]
    
    # Data Receiver Thread starten
    receiver_thread = threading.Thread(target=data_receiver_thread, daemon=True)
    receiver_thread.start()
    
    # Plot Setup
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Live BMS SOC Prediction')
    
    # Animation starten
    anim = FuncAnimation(
        fig, 
        lambda frame: animate_plots(frame, model, device, hidden_state),
        interval=UPDATE_INTERVAL,
        blit=False,
        cache_frame_data=False
    )
    
    print("📊 Live plot started. Waiting for data...")
    print("💡 Start the data_sender_C19.py script to begin receiving data")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("\n🛑 Stopping live test...")
    
    print("👋 Live test stopped")

if __name__ == "__main__":
    main()
