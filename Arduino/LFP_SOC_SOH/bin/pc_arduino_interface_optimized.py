"""
OPTIMIZED PC-Arduino Interface für BMS SOC Prediction
- Behebt Performance-Bottlenecks die über Zeit zunehmen
- Memory-effiziente Queue-Verwaltung
- Adaptive Timing für Arduino-Kommunikation
- Non-blocking Plot-Updates
"""

import serial
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from collections import deque
import threading
import queue
import gc
import warnings
warnings.filterwarnings('ignore')

# ======================== OPTIMIERTE KONFIGURATION ========================
DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
CELL_NAME = "MGFarm_18650_C19"
ARDUINO_PORT = "COM13"
BAUD_RATE = 115200

# PERFORMANCE OPTIMIERUNGEN
SEND_INTERVAL = 0.05      # 50ms = 20Hz (optimal für Arduino)
MAX_POINTS = 500          # Reduziert von 1000 - weniger Memory
UPDATE_INTERVAL = 50      # Häufigere Plot-Updates für smooth display
QUEUE_SIZE = 100          # Limitierte Queue-Größe verhindert Memory-Aufbau
TIMEOUT_BASE = 0.5        # Basis-Timeout
TIMEOUT_ADAPTIVE = True   # Adaptives Timeout basierend auf Arduino Response

# Optimierte Datenstrukturen mit fester Größe
arduino_data_queue = queue.Queue(maxsize=QUEUE_SIZE)
performance_monitor = {
    'total_sent': 0,
    'total_received': 0,
    'avg_response_time': deque(maxlen=100),
    'communication_errors': 0,
    'start_time': time.time(),
    'arduino_response_time': deque(maxlen=50),  # Für adaptives Timeout
}

plot_data = {
    'timestamps': deque(maxlen=MAX_POINTS),
    'true_soc': deque(maxlen=MAX_POINTS),
    'pred_soc': deque(maxlen=MAX_POINTS),
    'voltage': deque(maxlen=MAX_POINTS),
    'current': deque(maxlen=MAX_POINTS),
    'error': deque(maxlen=MAX_POINTS),
    'inference_time': deque(maxlen=MAX_POINTS),
    'communication_time': deque(maxlen=MAX_POINTS)
}

class OptimizedArduinoCommunicator:
    def __init__(self, port, baud_rate):
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None
        self.running = False
        self.adaptive_timeout = TIMEOUT_BASE
        
    def connect(self):
        """Verbindung zum Arduino mit Error-Handling"""
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Arduino Boot-Zeit
            
            # Teste Verbindung
            self.arduino.write(b'{"test": true}\n')
            time.sleep(0.5)
            if self.arduino.in_waiting > 0:
                response = self.arduino.readline().decode('utf-8', errors='ignore')
                print(f"✅ Arduino connected: {response.strip()}")
            else:
                print("✅ Arduino connected (no test response)")
            
            return True
        except Exception as e:
            print(f"❌ Arduino connection failed: {e}")
            return False
    
    def send_prediction_request(self, features, true_soc, idx):
        """Optimierte Arduino-Kommunikation mit adaptivem Timeout"""
        if not self.arduino:
            return None
            
        send_start = time.time()
        
        try:
            # Erstelle kompaktes JSON (weniger Overhead)
            data = {
                'v': float(features[0]),  # Voltage
                'i': float(features[1]),  # Current
                's': float(features[2]),  # SOH
                'q': float(features[3]),  # Q_c
                't': float(true_soc),     # True SOC
                'idx': int(idx)
            }
            
            # Sende JSON
            json_str = json.dumps(data, separators=(',', ':')) + '\n'  # Kompakt ohne Spaces
            self.arduino.write(json_str.encode('utf-8'))
            performance_monitor['total_sent'] += 1
            
            # Adaptive Timeout basierend auf vergangenen Arduino Response-Zeiten
            if performance_monitor['arduino_response_time']:
                avg_response = np.mean(performance_monitor['arduino_response_time'])
                self.adaptive_timeout = max(0.3, min(2.0, avg_response * 2))  # 2x avg, begrenzt
            
            # Empfange Response mit Timeout
            response_start = time.time()
            timeout_end = response_start + self.adaptive_timeout
            
            while time.time() < timeout_end:
                if self.arduino.in_waiting > 0:
                    response_line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    
                    if response_line:
                        try:
                            response = json.loads(response_line)
                            
                            # Berechne Timing-Statistiken
                            total_time = time.time() - send_start
                            arduino_time = time.time() - response_start
                            
                            performance_monitor['arduino_response_time'].append(arduino_time)
                            performance_monitor['avg_response_time'].append(total_time)
                            performance_monitor['total_received'] += 1
                            
                            # Füge Performance-Daten hinzu
                            response['communication_time'] = total_time * 1000  # ms
                            response['arduino_response_time'] = arduino_time * 1000
                            
                            return response
                            
                        except json.JSONDecodeError:
                            continue  # Ignoriere ungültige JSON-Zeilen
                
                time.sleep(0.001)  # Kleine Pause um CPU zu schonen
            
            # Timeout erreicht
            performance_monitor['communication_errors'] += 1
            return None
            
        except Exception as e:
            performance_monitor['communication_errors'] += 1
            print(f"❌ Communication error: {e}")
            return None
    
    def close(self):
        """Schließe Arduino-Verbindung"""
        if self.arduino:
            self.arduino.close()

def load_scaler_optimized():
    """Optimierte Scaler-Erstellung mit Memory-Management"""
    print("🔧 Loading scaler (optimized)...")
    
    base = Path(DATA_PATH)
    all_cells = [
        "MGFarm_18650_C01", "MGFarm_18650_C03", "MGFarm_18650_C05",
        "MGFarm_18650_C11", "MGFarm_18650_C17", "MGFarm_18650_C23",
        "MGFarm_18650_C07", "MGFarm_18650_C19", "MGFarm_18650_C21"
    ]
    
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
    scaler = StandardScaler()
    
    for cell_name in all_cells:
        folder = base / cell_name
        dfp = folder / "df.parquet"
        
        if dfp.exists():
            # Lade nur die benötigten Spalten
            df = pd.read_parquet(dfp, columns=feats)
            scaler.partial_fit(df[feats])
            
            # Memory-Management
            del df
            gc.collect()
            
    print("✅ Scaler loaded and optimized")
    return scaler

def load_test_data_optimized():
    """Optimiertes Laden der Test-Daten"""
    print("📊 Loading test data (optimized)...")
    
    base = Path(DATA_PATH)
    folder = base / CELL_NAME
    dfp = folder / "df.parquet"
    
    if not dfp.exists():
        raise FileNotFoundError(f"Data not found: {dfp}")
    
    # Lade nur notwendige Spalten
    required_cols = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c", "SOC_ZHU", "Absolute_Time[yyyy-mm-dd hh:mm:ss]"]
    df = pd.read_parquet(dfp, columns=required_cols)
    df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    
    # Verwende nur einen Subset für Tests (reduziert Memory und Zeit)
    if len(df) > 5000:
        df = df.iloc[::max(1, len(df)//5000)].reset_index(drop=True)  # Subsample
        
    print(f"✅ Loaded {len(df)} data points")
    return df

def arduino_communication_thread_optimized(communicator, df_scaled):
    """Optimierter Communication-Thread"""
    print("🚀 Starting optimized Arduino communication...")
    
    communicator.running = True
    
    for idx, row in df_scaled.iterrows():
        if not communicator.running:
            break
            
        features = [row['Voltage[V]'], row['Current[A]'], row['SOH_ZHU'], row['Q_c']]
        true_soc = row['SOC_ZHU']
        
        # Sende Request an Arduino
        response = communicator.send_prediction_request(features, true_soc, idx)
        
        if response:
            # Verarbeite Response
            processed_data = {
                'timestamp': time.time(),
                'pred_soc': response.get('pred_soc', 0),
                'true_soc': response.get('true_soc', true_soc),
                'voltage': features[0],
                'current': features[1],
                'error': abs(response.get('pred_soc', 0) - true_soc),
                'inference_time': response.get('inference_time_ms', 0),
                'communication_time': response.get('communication_time', 0),
                'idx': idx
            }
            
            # Non-blocking Queue Add
            try:
                arduino_data_queue.put_nowait(processed_data)
            except queue.Full:
                # Queue voll - entferne alte Daten
                try:
                    arduino_data_queue.get_nowait()
                    arduino_data_queue.put_nowait(processed_data)
                except queue.Empty:
                    pass
            
            # Progress Report
            if idx % 100 == 0:
                total_time = time.time() - performance_monitor['start_time']
                throughput = performance_monitor['total_received'] / total_time if total_time > 0 else 0
                avg_comm_time = np.mean(performance_monitor['avg_response_time']) * 1000 if performance_monitor['avg_response_time'] else 0
                
                print(f"🔄 Point {idx}: SOC {processed_data['pred_soc']:.3f} "
                      f"(True: {true_soc:.3f}), Error: {processed_data['error']:.4f}, "
                      f"Throughput: {throughput:.1f} Hz, Comm: {avg_comm_time:.1f}ms")
        
        # Adaptive Sleep basierend auf Arduino Performance
        if performance_monitor['arduino_response_time']:
            avg_arduino_time = np.mean(performance_monitor['arduino_response_time'])
            # Passe Send-Interval an Arduino Performance an
            sleep_time = max(SEND_INTERVAL, avg_arduino_time * 1.5)
        else:
            sleep_time = SEND_INTERVAL
            
        time.sleep(sleep_time)
    
    communicator.running = False
    print("✅ Arduino communication completed")

def update_plot_data_optimized():
    """Optimiertes Plot-Update mit Batch-Processing"""
    updates_processed = 0
    max_updates = 10  # Limitiere Updates pro Frame
    
    while not arduino_data_queue.empty() and updates_processed < max_updates:
        try:
            data = arduino_data_queue.get_nowait()
            
            # Füge Daten zu deques hinzu (automatisch bounded)
            plot_data['timestamps'].append(data['timestamp'])
            plot_data['true_soc'].append(data['true_soc'])
            plot_data['pred_soc'].append(data['pred_soc'])
            plot_data['voltage'].append(data['voltage'])
            plot_data['current'].append(data['current'])
            plot_data['error'].append(data['error'])
            plot_data['inference_time'].append(data['inference_time'])
            plot_data['communication_time'].append(data['communication_time'])
            
            updates_processed += 1
            
        except queue.Empty:
            break
    
    return updates_processed > 0

def animate_plots_optimized(frame):
    """Optimierte Plot-Animation mit Performance-Monitoring"""
    # Update Data
    data_updated = update_plot_data_optimized()
    
    if not data_updated or len(plot_data['timestamps']) < 5:
        return
    
    # Performance Statistics
    current_time = time.time()
    total_runtime = current_time - performance_monitor['start_time']
    success_rate = (performance_monitor['total_received'] / max(1, performance_monitor['total_sent'])) * 100
    throughput = performance_monitor['total_received'] / max(1, total_runtime)
    
    # Konvertiere deque zu lists für Plotting (effizient)
    times = list(plot_data['timestamps'])
    true_socs = list(plot_data['true_soc'])
    pred_socs = list(plot_data['pred_soc'])
    errors = list(plot_data['error'])
    inference_times = list(plot_data['inference_time'])
    comm_times = list(plot_data['communication_time'])
    
    # Clear und Plot (minimaler Overhead)
    ax1.clear()
    ax1.plot(times, true_socs, 'b-', label='True SOC', linewidth=1.5, alpha=0.8)
    ax1.plot(times, pred_socs, 'r--', label='Arduino Prediction', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('SOC')
    ax1.set_title('SOC Prediction vs Ground Truth')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.clear()
    ax2.plot(times, errors, 'g-', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Prediction Error')
    ax2.grid(True, alpha=0.3)
    
    ax3.clear()
    ax3.plot(times, inference_times, 'orange', label='Arduino Inference', linewidth=1, alpha=0.7)
    ax3.plot(times, comm_times, 'purple', label='Communication', linewidth=1, alpha=0.7)
    ax3.set_ylabel('Time [ms]')
    ax3.set_xlabel('Time')
    ax3.set_title('Performance Timing')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance Summary
    if len(errors) > 10:
        recent_errors = errors[-50:]  # Nur letzte 50 für Stats
        mean_error = np.mean(recent_errors)
        rmse = np.sqrt(np.mean(np.array(recent_errors)**2))
        mean_inference = np.mean(inference_times[-50:]) if len(inference_times) >= 50 else np.mean(inference_times)
        mean_comm = np.mean(comm_times[-50:]) if len(comm_times) >= 50 else np.mean(comm_times)
        
        title = (f'Arduino LSTM Monitor | Error: {mean_error:.4f} | RMSE: {rmse:.4f} | '
                f'Inference: {mean_inference:.1f}ms | Comm: {mean_comm:.1f}ms | '
                f'Throughput: {throughput:.1f}Hz | Success: {success_rate:.1f}%')
        
        fig.suptitle(title, fontsize=10)

def main():
    """Optimierte Hauptfunktion"""
    global fig, ax1, ax2, ax3
    
    print("🚀 Starting OPTIMIZED PC-Arduino Interface...")
    print("🔧 Performance optimizations enabled:")
    print(f"   - Adaptive timing: {TIMEOUT_ADAPTIVE}")
    print(f"   - Max plot points: {MAX_POINTS}")
    print(f"   - Queue size limit: {QUEUE_SIZE}")
    print(f"   - Send interval: {SEND_INTERVAL*1000:.0f}ms")
    
    # Daten laden (optimiert)
    scaler = load_scaler_optimized()
    df = load_test_data_optimized()
    
    # Skaliere Features
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
    df_scaled = df.copy()
    df_scaled[feats] = scaler.transform(df[feats])
    
    # Arduino Communicator
    communicator = OptimizedArduinoCommunicator(ARDUINO_PORT, BAUD_RATE)
    
    if not communicator.connect():
        print("❌ Failed to connect to Arduino")
        return
    
    # Communication Thread starten
    comm_thread = threading.Thread(
        target=arduino_communication_thread_optimized,
        args=(communicator, df_scaled),
        daemon=True
    )
    comm_thread.start()
    
    # Optimierte Plot-Setup
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    # Animation mit optimiertem Interval
    anim = FuncAnimation(
        fig,
        animate_plots_optimized,
        interval=UPDATE_INTERVAL,
        blit=False,
        cache_frame_data=False
    )
    
    print("📊 Optimized live monitoring started...")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("\n🛑 Stopping optimized interface...")
    finally:
        communicator.running = False
        communicator.close()
        
        # Final Performance Report
        total_time = time.time() - performance_monitor['start_time']
        print(f"\n📊 FINAL PERFORMANCE REPORT:")
        print(f"   Runtime: {total_time:.1f}s")
        print(f"   Messages sent: {performance_monitor['total_sent']}")
        print(f"   Messages received: {performance_monitor['total_received']}")
        print(f"   Success rate: {performance_monitor['total_received']/max(1, performance_monitor['total_sent'])*100:.1f}%")
        print(f"   Communication errors: {performance_monitor['communication_errors']}")
        print(f"   Average throughput: {performance_monitor['total_received']/total_time:.1f} Hz")
        if performance_monitor['avg_response_time']:
            print(f"   Average response time: {np.mean(performance_monitor['avg_response_time'])*1000:.1f}ms")
    
    print("👋 Optimized interface stopped")

if __name__ == "__main__":
    main()
