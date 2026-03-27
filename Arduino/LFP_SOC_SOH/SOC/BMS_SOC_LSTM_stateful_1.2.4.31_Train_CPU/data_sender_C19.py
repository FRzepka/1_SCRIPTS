"""
BMS SOC Data Sender - C19 Cell
Sendet C19 Zellendaten sekündlich über Socket
Für Test des trainierten LSTM-Modells
"""

import pandas as pd
import numpy as np
import time
import socket
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import sys

# Konstanten
DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
CELL_NAME = "MGFarm_18650_C19"
HOST = 'localhost'
PORT = 12345
SEND_INTERVAL = 0.01  # Sekunden - BESCHLEUNIGT auf 10ms!

def load_scaler():
    """Lade den Scaler vom Training"""
    # Wir recreaten den Scaler wie im Training
    base = Path(DATA_PATH)
    
    # Alle Zellen wie im Training
    all_cells = [
        "MGFarm_18650_C01", "MGFarm_18650_C03", "MGFarm_18650_C05",
        "MGFarm_18650_C11", "MGFarm_18650_C17", "MGFarm_18650_C23",
        "MGFarm_18650_C07", "MGFarm_18650_C19", "MGFarm_18650_C21"
    ]
    
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
    scaler = StandardScaler()
    
    print("Recreating scaler from training data...")
    for cell_name in all_cells:
        folder = base / cell_name
        if folder.exists():
            dfp = folder / "df.parquet"
            if dfp.exists():
                df = pd.read_parquet(dfp)
                df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
                scaler.partial_fit(df[feats])
                print(f"Partial fit für {cell_name}: {len(df)} Zeilen")
    
    return scaler

def load_c19_data():
    """Lade C19 Daten"""
    base = Path(DATA_PATH)
    folder = base / CELL_NAME
    dfp = folder / "df.parquet"
    
    if not dfp.exists():
        raise FileNotFoundError(f"C19 data not found: {dfp}")
    
    df = pd.read_parquet(dfp)
    df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    
    print(f"Loaded C19 data: {len(df)} rows")
    return df

def prepare_data(df, scaler):
    """Bereite die Daten vor (skalieren)"""
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
    
    # Skalierung anwenden
    df_scaled = df.copy()
    df_scaled[feats] = scaler.transform(df[feats])
    
    print(f"Data prepared and scaled")
    return df_scaled

def start_data_sender():
    """Startet den Daten-Sender"""
    print("🚀 Starting C19 Data Sender...")
    
    # Lade Scaler und Daten
    scaler = load_scaler()
    df = load_c19_data()
    df_scaled = prepare_data(df, scaler)
    
    # Socket erstellen
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"📡 Waiting for connection on {HOST}:{PORT}")
    print(f"📊 Ready to send {len(df_scaled)} data points from {CELL_NAME}")
    
    try:
        while True:
            # Warte auf Verbindung
            client_socket, address = server_socket.accept()
            print(f"✅ Connected to {address}")
            
            try:
                # Sende Daten sequenziell
                for idx, row in df_scaled.iterrows():
                    # Erstelle Datenpaket
                    data_packet = {
                        'timestamp': row['timestamp'].isoformat(),
                        'voltage': float(row['Voltage[V]']),
                        'current': float(row['Current[A]']),
                        'soh': float(row['SOH_ZHU']),
                        'q_c': float(row['Q_c']),
                        'true_soc': float(row['SOC_ZHU']),  # Ground truth für Vergleich
                        'index': int(idx),
                        'total_points': len(df_scaled)
                    }
                      # Als JSON senden
                    json_data = json.dumps(data_packet) + '\n'
                    client_socket.send(json_data.encode('utf-8'))
                    
                    if idx % 100 == 0:
                        print(f"📤 Sent data point {idx}/{len(df_scaled)} - SOC: {data_packet['true_soc']:.3f} - SPEED: {SEND_INTERVAL*1000:.1f}ms interval")
                    
                    # Warte SEND_INTERVAL Sekunden
                    time.sleep(SEND_INTERVAL)
                
                print("✅ All data sent successfully!")
                
            except (BrokenPipeError, ConnectionResetError):
                print("❌ Client disconnected")
            except KeyboardInterrupt:
                print("\n🛑 Stopping data sender...")
                break
            finally:
                client_socket.close()
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping data sender...")
    finally:
        server_socket.close()
        print("👋 Data sender stopped")

if __name__ == "__main__":
    start_data_sender()
