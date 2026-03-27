"""
BMS SOC LSTM Live Test Script - OPTIMIZED VERSION
Optimized for performance with bounded queues, adaptive timeouts, and memory management
Fixes memory leaks and performance degradation over time
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
import gc
import psutil
import logging

# Configure logging for performance monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modell-Konstanten (müssen mit Training übereinstimmen)
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32
MODEL_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\training_run_1_2_4_31\best_model.pth"

# Socket-Konstanten - OPTIMIZED
HOST = 'localhost'
PORT = 12345
SOCKET_TIMEOUT = 5.0  # Socket timeout to prevent hanging
BUFFER_SIZE = 8192    # Larger buffer for better performance
RECONNECT_DELAY = 1.0 # Faster reconnection

# Plot-Konstanten - OPTIMIZED
MAX_POINTS = 500      # Reduced from 2000 for better memory efficiency
UPDATE_INTERVAL = 100 # Slightly increased for better performance
BATCH_PROCESS_SIZE = 20  # Process fewer points per frame for smoother animation

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
    'last_log_time': time.time(),
    'processing_times': deque(maxlen=100),
    'queue_sizes': deque(maxlen=100),
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
    logger.info(f"🎯 Using device: {device}")
    
    # Modell erstellen
    model = SOCModel(input_size=4, dropout=0.05).to(device)
    
    # Gewichte laden
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    logger.info(f"✅ Model loaded from {MODEL_PATH}")
    return model, device

def init_hidden(device, batch_size=1):
    """Initialisiere Hidden States"""
    h = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=device)
    c = torch.zeros_like(h)
    return h, c

def log_performance_stats():
    """Log performance statistics"""
    current_time = time.time()
    elapsed = current_time - performance_stats['start_time']
    
    if elapsed > 0:
        throughput = performance_stats['processed_points'] / elapsed
        queue_size = data_queue.qsize()
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Store metrics
        performance_stats['queue_sizes'].append(queue_size)
        performance_stats['memory_usage'].append(memory_mb)
        
        # Average processing time
        avg_processing_time = np.mean(performance_stats['processing_times']) if performance_stats['processing_times'] else 0
        
        logger.info(f"📊 Performance - Points: {performance_stats['processed_points']}, "
                   f"Throughput: {throughput:.1f} pts/s, Queue: {queue_size}, "
                   f"Memory: {memory_mb:.1f}MB, Avg Process Time: {avg_processing_time*1000:.1f}ms")

def data_receiver_thread():
    """Thread für Datenempfang - OPTIMIZED"""
    logger.info("🔌 Connecting to data sender...")
    
    while True:
        client_socket = None
        try:
            # Verbindung zum Data Sender mit Timeout
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(SOCKET_TIMEOUT)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
            client_socket.connect((HOST, PORT))
            logger.info(f"✅ Connected to data sender at {HOST}:{PORT}")
            
            buffer = ""
            while True:
                try:
                    # Daten empfangen mit optimiertem Buffer
                    data = client_socket.recv(BUFFER_SIZE).decode('utf-8')
                    if not data:
                        break
                    
                    buffer += data
                    
                    # Verarbeite komplette JSON-Zeilen
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():  # Skip empty lines
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
                                        
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error: {e}")
                                continue
                                
                except socket.timeout:
                    logger.warning("Socket timeout, continuing...")
                    continue
                except ConnectionResetError:
                    logger.warning("Connection reset by peer")
                    break
                        
        except ConnectionRefusedError:
            logger.warning(f"Connection refused. Retrying in {RECONNECT_DELAY} seconds...")
            time.sleep(RECONNECT_DELAY)
        except Exception as e:
            logger.error(f"Data receiver error: {e}")
            time.sleep(RECONNECT_DELAY)
        finally:
            if client_socket:
                try:
                    client_socket.close()
                except:
                    pass

def process_data_point(model, device, hidden_state, data_packet):
    """Verarbeite einen Datenpunkt durch das Modell - OPTIMIZED"""
    start_time = time.time()
    
    try:
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
        
        # Performance tracking
        processing_time = time.time() - start_time
        performance_stats['processing_times'].append(processing_time)
        
        return pred_soc_value, error, hidden_state
        
    except Exception as e:
        logger.error(f"Error processing data point: {e}")
        return 0.0, 1.0, hidden_state

def update_plot_data(data_packet, pred_soc, error):
    """Aktualisiere Plot-Daten - OPTIMIZED"""
    try:
        timestamp = pd.to_datetime(data_packet['timestamp'])
        
        # Efficient data updates using deque maxlen
        plot_data['timestamps'].append(timestamp)
        plot_data['true_soc'].append(data_packet['true_soc'])
        plot_data['pred_soc'].append(pred_soc)
        plot_data['voltage'].append(data_packet['voltage'])
        plot_data['current'].append(data_packet['current'])
        plot_data['error'].append(error)
        
    except Exception as e:
        logger.error(f"Error updating plot data: {e}")

def animate_plots(frame, model, device, hidden_state):
    """Animation-Funktion für Live-Plots - OPTIMIZED"""
    global plot_data, performance_stats
    
    # Verarbeite verfügbare Datenpunkte mit Begrenzung
    processed_count = 0
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
            performance_stats['processed_points'] += 1
            
            # Performance logging
            if performance_stats['processed_points'] % PERFORMANCE_LOG_INTERVAL == 0:
                log_performance_stats()
            
            # Periodic memory cleanup
            if performance_stats['processed_points'] % MEMORY_CLEANUP_INTERVAL == 0:
                gc.collect()
                logger.info("🧹 Performed garbage collection")
                
            # Status ausgeben (weniger häufig)
            if data_packet.get('index', 0) % 1000 == 0:
                logger.info(f"🚀 Point {data_packet.get('index', 0)}/{data_packet.get('total_points', 0)} - "
                           f"True SOC: {data_packet['true_soc']:.3f}, "
                           f"Pred SOC: {pred_soc:.3f}, "
                           f"Error: {error:.4f}")
                
        except queue.Empty:
            break
        except Exception as e:
            logger.error(f"Error in animation loop: {e}")
            break
    
    # Plots aktualisieren nur wenn Daten vorhanden
    if len(plot_data['timestamps']) > 0:
        try:
            # Convert to lists once for efficiency
            times = list(plot_data['timestamps'])
            true_socs = list(plot_data['true_soc'])
            pred_socs = list(plot_data['pred_soc'])
            errors = list(plot_data['error'])
            voltages = list(plot_data['voltage'])
            currents = list(plot_data['current'])
            
            # SOC Vergleich Plot
            ax1.clear()
            ax1.plot(times, true_socs, 'b-', label='True SOC', linewidth=2, alpha=0.8)
            ax1.plot(times, pred_socs, 'r--', label='Predicted SOC', linewidth=2, alpha=0.8)
            ax1.set_ylabel('SOC')
            ax1.set_title('Live SOC Prediction vs Ground Truth')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Fehler Plot
            ax2.clear()
            ax2.plot(times, errors, 'g-', label='Absolute Error', linewidth=1, alpha=0.8)
            ax2.set_ylabel('Absolute Error')
            ax2.set_title('Prediction Error')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Spannung Plot
            ax3.clear()
            ax3.plot(times, voltages, 'orange', label='Voltage (scaled)', linewidth=1, alpha=0.8)
            ax3.set_ylabel('Voltage (scaled)')
            ax3.set_xlabel('Time')
            ax3.set_title('Input Voltage')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Strom Plot
            ax4.clear()
            ax4.plot(times, currents, 'purple', label='Current (scaled)', linewidth=1, alpha=0.8)
            ax4.set_ylabel('Current (scaled)')
            ax4.set_xlabel('Time')
            ax4.set_title('Input Current')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Statistiken berechnen und anzeigen
            if len(errors) > 10:
                recent_errors = errors[-min(100, len(errors)):]  # Use fewer points for stats
                mean_error = np.mean(recent_errors)
                rmse = np.sqrt(np.mean(np.array(recent_errors)**2))
                
                # Performance info
                elapsed = time.time() - performance_stats['start_time']
                throughput = performance_stats['processed_points'] / elapsed if elapsed > 0 else 0
                
                fig.suptitle(f'Live SOC Prediction (OPTIMIZED) - Mean Error: {mean_error:.4f}, '
                           f'RMSE: {rmse:.4f} - Points: {len(errors)} - Throughput: {throughput:.1f} pts/s')
                           
        except Exception as e:
            logger.error(f"Error updating plots: {e}")

def main():
    """Hauptfunktion - OPTIMIZED"""
    global fig, ax1, ax2, ax3, ax4
    
    logger.info("🚀 Starting BMS SOC Live Test (OPTIMIZED VERSION)...")
    
    try:
        # Modell laden
        model, device = load_model()
        
        # Hidden State initialisieren (in Liste für Referenz-Übergabe)
        hidden_state = [init_hidden(device)]
        
        # Data Receiver Thread starten
        receiver_thread = threading.Thread(target=data_receiver_thread, daemon=True)
        receiver_thread.start()
        
        # Plot Setup mit Performance-optimierten Einstellungen
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Live BMS SOC Prediction (OPTIMIZED)')
        
        # Animation starten mit optimierten Parametern
        anim = FuncAnimation(
            fig, 
            lambda frame: animate_plots(frame, model, device, hidden_state),
            interval=UPDATE_INTERVAL,
            blit=False,
            cache_frame_data=False
        )
        
        logger.info("📊 Live plot started. Waiting for data...")
        logger.info("💡 Start the data_sender_C19.py script to begin receiving data")
        logger.info("🛑 Press Ctrl+C to stop")
        
        plt.tight_layout()
        plt.show()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping live test...")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Performance summary
        elapsed = time.time() - performance_stats['start_time']
        if elapsed > 0:
            final_throughput = performance_stats['processed_points'] / elapsed
            logger.info(f"📊 Final Performance Summary:")
            logger.info(f"   - Total Points: {performance_stats['processed_points']}")
            logger.info(f"   - Runtime: {elapsed:.1f}s")
            logger.info(f"   - Average Throughput: {final_throughput:.1f} pts/s")
            logger.info(f"   - Max Memory: {max(performance_stats['memory_usage']) if performance_stats['memory_usage'] else 0:.1f}MB")
    
    logger.info("👋 Live test stopped")

if __name__ == "__main__":
    main()
