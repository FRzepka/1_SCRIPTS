"""
PC-Arduino Interface für BMS SOC Prediction
- Sendet skalierte Sensor-Daten an Arduino über Serial
- Empfängt SOC-Vorhersagen vom Arduino
- Live-Plot der Ergebnisse
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

# Konstanten
DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
CELL_NAME = "MGFarm_18650_C19"
ARDUINO_PORT = "COM13"  # Updated to your Arduino port
BAUD_RATE = 115200
SEND_INTERVAL = 0.1  # 100ms - nicht zu schnell für Arduino

# Plot-Konstanten
MAX_POINTS = 1000
UPDATE_INTERVAL = 100  # ms

# Globale Datenstrukturen
arduino_data_queue = queue.Queue()
plot_data = {
    'timestamps': deque(maxlen=MAX_POINTS),
    'true_soc': deque(maxlen=MAX_POINTS),
    'pred_soc': deque(maxlen=MAX_POINTS),
    'voltage': deque(maxlen=MAX_POINTS),
    'current': deque(maxlen=MAX_POINTS),
    'error': deque(maxlen=MAX_POINTS),
    'inference_time': deque(maxlen=MAX_POINTS)
}

def load_scaler():
    """Lade den Scaler vom Training (identisch zum ursprünglichen Script)"""
    base = Path(DATA_PATH)
    
    all_cells = [
        "MGFarm_18650_C01", "MGFarm_18650_C03", "MGFarm_18650_C05",
        "MGFarm_18650_C11", "MGFarm_18650_C17", "MGFarm_18650_C23",
        "MGFarm_18650_C07", "MGFarm_18650_C19", "MGFarm_18650_C21"
    ]
    
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
    scaler = StandardScaler()
    
    print("🔧 Recreating scaler from training data...")
    for cell_name in all_cells:
        folder = base / cell_name
        if folder.exists():
            dfp = folder / "df.parquet"
            if dfp.exists():
                df = pd.read_parquet(dfp)
                df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
                scaler.partial_fit(df[feats])
                print(f"✅ Partial fit für {cell_name}: {len(df)} Zeilen")
    
    return scaler

def load_test_data():
    """Lade C19 Test-Daten"""
    base = Path(DATA_PATH)
    folder = base / CELL_NAME
    dfp = folder / "df.parquet"
    
    if not dfp.exists():
        raise FileNotFoundError(f"C19 data not found: {dfp}")
    
    df = pd.read_parquet(dfp)
    df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    
    print(f"📊 Loaded C19 data: {len(df)} rows")
    return df

def prepare_data(df, scaler):
    """Bereite die Daten vor (skalieren)"""
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
    
    df_scaled = df.copy()
    df_scaled[feats] = scaler.transform(df[feats])
    
    print(f"🔧 Data prepared and scaled")
    return df_scaled

def arduino_communication_thread(arduino_serial, df_scaled):
    """Thread für Arduino-Kommunikation"""
    print("🤖 Starting Arduino communication thread...")
    
    for idx, row in df_scaled.iterrows():
        # Verwende bereits skalierte Daten aus df_scaled
        scaled_features = [row['Voltage[V]'], row['Current[A]'], row['SOH_ZHU'], row['Q_c']]
        try:
            # Erstelle Datenpaket für Arduino im erwarteten Format
            data_packet = {
                'command': 'predict',
                'features': [
                    float(scaled_features[0]),  # Voltage (scaled)
                    float(scaled_features[1]),  # Current (scaled) 
                    float(scaled_features[2]),  # SOH (scaled)
                    float(scaled_features[3])   # Q_c (scaled)
                ]
            }
            
            # Als JSON an Arduino senden
            json_data = json.dumps(data_packet) + '\n'
            arduino_serial.write(json_data.encode('utf-8'))
            # Debug send
            print(f"➡️ Sent idx={idx}, command=predict, features={data_packet['features']}")
            
            # Poll for JSON response with timeout
            start_time = time.time()
            timeout = 1.0
            response = None
            while time.time() - start_time < timeout:
                if arduino_serial.in_waiting > 0:
                    line = arduino_serial.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        try:
                            response = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
            
            if response:
                # Convert Arduino response to our format
                if 'soc' in response:
                    inference_time = response.get('inference_time_us', 0) / 1000.0  # Convert to ms
                    processed_response = {
                        'pred_soc': response['soc'],
                        'true_soc': float(row['SOC_ZHU']),
                        'voltage': float(scaled_features[0]),
                        'current': float(scaled_features[1]),
                        'inference_time': inference_time,
                        'timestamp': row['timestamp']
                    }
                    arduino_data_queue.put(processed_response)
                    if idx % 50 == 0:
                        print(f"🤖 Point {idx}: True {processed_response['true_soc']:.3f}, Pred {processed_response['pred_soc']:.3f}, Error {abs(processed_response['pred_soc']-processed_response['true_soc']):.4f}, Time {inference_time:.1f}ms")
                else:
                    print(f"❗ Invalid response format for idx={idx}: {response}")
            else:
                print(f"❗ No response for idx={idx} within timeout")
            
            time.sleep(SEND_INTERVAL)
            
        except Exception as e:
            print(f"❌ Arduino communication error: {e}")
            break

def update_plot_data(response):
    """Aktualisiere Plot-Daten"""
    plot_data['timestamps'].append(response['timestamp'])
    plot_data['true_soc'].append(response.get('true_soc', 0))
    plot_data['pred_soc'].append(response.get('pred_soc', 0))
    plot_data['voltage'].append(response.get('voltage', 0))
    plot_data['current'].append(response.get('current', 0))
    
    error = abs(response.get('pred_soc', 0) - response.get('true_soc', 0))
    plot_data['error'].append(error)
    plot_data['inference_time'].append(response.get('inference_time', 0))

def animate_plots(frame):
    """Animation-Funktion für Live-Plots"""
    # Verarbeite alle verfügbaren Arduino-Antworten
    processed_count = 0
    while not arduino_data_queue.empty() and processed_count < 20:
        try:
            response = arduino_data_queue.get_nowait()
            update_plot_data(response)
            processed_count += 1
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
        ax1.plot(times, pred_socs, 'r--', label='Arduino Predicted SOC', linewidth=2)
        ax1.set_ylabel('SOC')
        ax1.set_title('Arduino LSTM SOC Prediction vs Ground Truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fehler Plot
        ax2.clear()
        errors = list(plot_data['error'])
        ax2.plot(times, errors, 'g-', label='Absolute Error', linewidth=1)
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Arduino Prediction Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Inference Zeit Plot
        ax3.clear()
        inference_times = list(plot_data['inference_time'])
        ax3.plot(times, inference_times, 'orange', label='Inference Time [ms]', linewidth=1)
        ax3.set_ylabel('Time [ms]')
        ax3.set_xlabel('Time')
        ax3.set_title('Arduino Inference Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Input Voltage Plot
        ax4.clear()
        voltages = list(plot_data['voltage'])
        ax4.plot(times, voltages, 'purple', label='Voltage (scaled)', linewidth=1)
        ax4.set_ylabel('Voltage (scaled)')
        ax4.set_xlabel('Time')
        ax4.set_title('Input Voltage')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Statistiken
        if len(errors) > 10:
            mean_error = np.mean(errors[-100:])
            rmse = np.sqrt(np.mean(np.array(errors[-100:])**2))
            mean_inference = np.mean(inference_times[-100:]) if inference_times else 0
            fig.suptitle(f'Arduino LSTM Live Test - Error: {mean_error:.4f}, RMSE: {rmse:.4f}, Avg Inference: {mean_inference:.1f}ms')

def main():
    """Hauptfunktion"""
    global fig, ax1, ax2, ax3, ax4
    
    print("🚀 Starting PC-Arduino BMS Interface...")
    
    # Daten laden
    scaler = load_scaler()
    df = load_test_data()
    df_scaled = prepare_data(df, scaler)
    
    # Arduino-Verbindung
    try:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Arduino Setup-Zeit
        print(f"✅ Connected to Arduino on {ARDUINO_PORT}")
    except Exception as e:
        print(f"❌ Arduino connection failed: {e}")
        print(f"💡 Please check the port {ARDUINO_PORT} and make sure Arduino is connected")
        return
    
    # Arduino Communication Thread starten
    comm_thread = threading.Thread(
        target=arduino_communication_thread, 
        args=(arduino, df_scaled), 
        daemon=True
    )
    comm_thread.start()
    
    # Plot Setup
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Arduino LSTM SOC Prediction Live Test')
    
    # Animation starten
    anim = FuncAnimation(
        fig, 
        animate_plots,
        interval=UPDATE_INTERVAL,
        blit=False,
        cache_frame_data=False
    )
    
    print("📊 Live plot started. Arduino processing...")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("\n🛑 Stopping Arduino interface...")
    finally:
        arduino.close()
    
    print("👋 Arduino interface stopped")

if __name__ == "__main__":
    main()
