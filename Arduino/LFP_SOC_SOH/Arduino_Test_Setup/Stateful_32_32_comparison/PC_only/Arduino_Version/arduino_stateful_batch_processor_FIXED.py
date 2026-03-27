#!/usr/bin/env python3
"""
🔬 Arduino Stateful LSTM SOC Prediction - FIXED VERSION
Korrigierte Version mit richtigem Datenformat für Arduino-Kommunikation

PROBLEM GEFUNDEN UND BEHOBEN:
- Python sendete: "DATA:V,I,SOH,Q_c" 
- Arduino erwartete: "V,I,SOH,Q_c"
- "DATA:" Präfix entfernt!
"""

import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import os
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import logging
import warnings
import argparse
import joblib

# Warnings unterdrücken
warnings.filterwarnings("ignore", category=UserWarning)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === ZEITBEREICH KONFIGURATION & PERFORMANCE ===
START_TIME_SEC = 0      # Start in Sekunden
DURATION_SEC = 1000     # Reduziert für Tests - war 10000

# === PERFORMANCE OPTIMIERUNGEN ===
BATCH_SIZE = 5          # Kleinere Batches für bessere Übersicht
SERIAL_TIMEOUT = 0.2    # Etwas länger für stabile Kommunikation
MAX_RETRIES = 2         
ENABLE_SAMPLING = True  
SAMPLE_EVERY_N = 10     # Stärkeres Sampling für Tests

# === ARDUINO KONFIGURATION ===
ARDUINO_PORT = 'COM13'
BAUDRATE = 115200
DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
SCALER_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32\scaler.pkl"
ARDUINO_SKETCH_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32\code_weights\arduino_lstm_soc_full32_with_monitoring\arduino_lstm_soc_full32_with_monitoring.ino"
ARDUINO_CLI_PATH = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Arduino_CLI\arduino-cli.exe"
ARDUINO_FQBN = "arduino:renesas_uno:unor4wifi"

# === FARBSCHEMA ===
COLOR_SCHEME = {
    'ground_truth': '#2091C9',     # Blau für Ground Truth
    'arduino_pred': '#FF6B6B',     # Rot für Arduino
    'error_color': '#D9140E'       
}

print("🖥️  Arduino Processing Mode - FIXED COMMUNICATION")
print("🎯 Target Hardware: Arduino UNO R4 WiFi")
print("🔧 FIX: Entfernt 'DATA:' Präfix - Arduino erwartet nur 'V,I,SOH,Q_c'")
print(f"⏱️  Batch Processing Range: {START_TIME_SEC}s - {START_TIME_SEC + DURATION_SEC}s ({DURATION_SEC}s duration)")

class ArduinoStatefulLSTMProcessor:
    def __init__(self, port, baudrate, data_path, sketch_path, fqbn):
        self.port = port
        self.baudrate = baudrate
        self.data_path = data_path
        self.sketch_path = sketch_path
        self.fqbn = fqbn
        self.arduino = None
        self.arduino_connected = False
        
        if not self.check_arduino_cli():
            raise RuntimeError("Arduino CLI nicht gefunden")
        
        logger.info("Arduino Stateful LSTM Processor initialized")

    def check_arduino_cli(self):
        """Prüft ob Arduino CLI verfügbar ist"""
        try:
            result = subprocess.run([ARDUINO_CLI_PATH, "version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ Arduino CLI found at {ARDUINO_CLI_PATH}: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"❌ Arduino CLI error: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Arduino CLI check failed: {e}")
            return False

    def upload_arduino_sketch(self):
        """Kompiliert und lädt den Arduino Sketch hoch"""
        try:
            logger.info(f"📤 Uploading sketch: {self.sketch_path}")
            
            # Kompilieren
            compile_cmd = [ARDUINO_CLI_PATH, "compile", "--fqbn", self.fqbn, self.sketch_path]
            logger.info("🔨 Compiling...")
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"❌ Compilation failed: {result.stderr}")
                return False
            
            logger.info("✅ Compilation successful")
            
            # Upload
            upload_cmd = [ARDUINO_CLI_PATH, "upload", "-p", self.port, "--fqbn", self.fqbn, self.sketch_path]
            logger.info(f"📤 Uploading to {self.port}...")
            result = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"❌ Upload failed: {result.stderr}")
                return False
            
            logger.info("✅ Upload successful")
            time.sleep(2)  # Arduino Neustart abwarten
            return True
            
        except Exception as e:
            logger.error(f"❌ Sketch upload error: {e}")
            return False

    def connect_arduino(self):
        """Verbindet mit Arduino"""
        try:
            logger.info(f"🔌 Connecting to Arduino on {self.port}...")
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=SERIAL_TIMEOUT)
            time.sleep(2)  # Arduino Initialisierung abwarten
            
            # Teste Verbindung
            self.arduino.write(b"INFO\n")
            response = self.arduino.readline().decode().strip()
            
            if response:
                logger.info(f"✅ Arduino connected: {response}")
                self.arduino_connected = True
                return True
            else:
                logger.error("❌ No response from Arduino")
                return False
                
        except Exception as e:
            logger.error(f"Arduino connection failed: {e}")
            self.arduino_connected = False
            return False

    def send_single_prediction(self, scaled_features):
        """Sendet einzelne Vorhersage-Anfrage an Arduino"""
        try:
            # KORRIGIERTES FORMAT: Nur V,I,SOH,Q_c ohne "DATA:" Präfix
            command = f"{scaled_features[0]:.6f},{scaled_features[1]:.6f},{scaled_features[2]:.6f},{scaled_features[3]:.6f}\n"
            
            self.arduino.write(command.encode())
            self.arduino.flush()
            
            # Warte auf Antwort
            response = ""
            start_time = time.time()
            while time.time() - start_time < SERIAL_TIMEOUT:
                if self.arduino.in_waiting > 0:
                    response = self.arduino.readline().decode().strip()
                    break
            
            # Parse Arduino Antwort: "DATA:soc_pred,inference_time,ram_free,..."
            if response and response.startswith("DATA:"):
                data_parts = response.replace("DATA:", "").split(",")
                if len(data_parts) >= 1:
                    return float(data_parts[0])  # SOC Vorhersage
            
            return None
            
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            return None

    def load_and_process_data(self):
        """Lädt und verarbeitet die Daten"""
        try:
            logger.info(f"📊 Loading data from {self.data_path}")
            df = pd.read_parquet(self.data_path)
            
            # Features
            feats = ['V', 'I', 'SOH', 'Q_c']
            
            # Scaler laden oder erstellen
            if os.path.exists(SCALER_PATH):
                scaler = joblib.load(SCALER_PATH)
                logger.info("📊 Scaler loaded from file")
            else:
                scaler = RobustScaler()
                scaler.fit(df[feats])
                joblib.dump(scaler, SCALER_PATH)
                logger.info("📊 New scaler created and saved")
            
            return df, scaler, feats
            
        except Exception as e:
            logger.error(f"❌ Data loading error: {e}")
            return None, None, None

    def run_batch_processing(self, auto_upload=True, start_time_sec=START_TIME_SEC, 
                           duration_sec=DURATION_SEC, enable_sampling=ENABLE_SAMPLING,
                           sample_rate=SAMPLE_EVERY_N, batch_size=BATCH_SIZE):
        """Hauptfunktion für Batch Processing"""
        try:
            # Upload Sketch
            if auto_upload:
                if not self.upload_arduino_sketch():
                    return False
            
            # Verbinde mit Arduino
            if not self.connect_arduino():
                return False
            
            # Lade Daten
            df, scaler, feats = self.load_and_process_data()
            if df is None:
                return False
            
            # Reset Arduino LSTM State
            self.arduino.write(b"RESET\n")
            time.sleep(0.2)
            
            # Zeitbereich definieren
            end_sec = start_time_sec + duration_sec
            start_idx = max(0, start_time_sec)
            end_idx = min(len(df), end_sec)
            
            logger.info(f"📊 Processing time range: {start_time_sec}s - {end_sec}s")
            logger.info(f"📊 Data indices: {start_idx} - {end_idx} ({end_idx - start_idx} samples)")
            
            # Extrahiere Daten
            seq_data = df[feats].values[start_idx:end_idx]
            labels = df["SOC_ZHU"].values[start_idx:end_idx]
            timestamps = np.arange(start_idx, end_idx)
            
            # Sampling anwenden
            if enable_sampling and len(seq_data) > 200:
                original_length = len(seq_data)
                sample_indices = list(range(0, len(seq_data), sample_rate))
                if sample_indices[-1] != len(seq_data) - 1:
                    sample_indices.append(len(seq_data) - 1)
                
                seq_data = seq_data[sample_indices]
                labels = labels[sample_indices]
                timestamps = timestamps[sample_indices]
                
                logger.info(f"🔍 Sampling: {original_length:,} → {len(seq_data):,} samples (every {sample_rate}th)")
            
            # Skaliere Features
            scaled_features = scaler.transform(seq_data)
            
            # Arduino Vorhersagen
            logger.info(f"🧠 Starting Arduino predictions for {len(scaled_features)} samples...")
            
            predictions = []
            successful_preds = 0
            
            progress_bar = tqdm(scaled_features, desc="🧠 Arduino Inference", ncols=100)
            
            for i, features in enumerate(progress_bar):
                pred = self.send_single_prediction(features)
                
                if pred is not None:
                    predictions.append(pred)
                    successful_preds += 1
                else:
                    predictions.append(0.0)  # Fallback
                
                # Update Progress
                progress_bar.set_postfix({
                    'Success': f"{successful_preds}/{i+1}",
                    'Rate': f"{successful_preds/(i+1)*100:.1f}%"
                })
                
                # Kleine Pause für stabile Kommunikation
                time.sleep(0.01)
            
            predictions = np.array(predictions)
            
            # Ergebnisse
            mae = mean_absolute_error(labels, predictions)
            logger.info(f"📈 MAE: {mae:.4f}")
            logger.info(f"📈 Success Rate: {successful_preds}/{len(scaled_features)} ({successful_preds/len(scaled_features)*100:.1f}%)")
            
            # Plot erstellen
            self.create_plot(predictions, labels, timestamps, mae, start_time_sec)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            return False

    def create_plot(self, preds, labels, timestamps, mae_value, start_sec):
        """Erstellt Vergleichsplot"""
        try:
            plt.figure(figsize=(14, 8))
            
            # Zeitachse in Sekunden
            time_axis = timestamps
            
            # Plot
            plt.plot(time_axis, labels, 
                    color=COLOR_SCHEME['ground_truth'], linewidth=2.5, 
                    label='Ground Truth SOC', alpha=0.9)
            
            plt.plot(time_axis, preds, 
                    color=COLOR_SCHEME['arduino_pred'], linewidth=2.0, 
                    label='Arduino Stateful LSTM', alpha=0.8)
            
            # Formatierung
            plt.xlabel('Time [seconds]', fontsize=12, fontweight='bold')
            plt.ylabel('State of Charge (SOC)', fontsize=12, fontweight='bold')
            plt.title('Arduino Stateful LSTM vs Ground Truth - COMMUNICATION FIXED', 
                     fontsize=14, fontweight='bold', pad=20)
            
            # Info Box
            info_text = f"MAE: {mae_value:.4f}\nSamples: {len(preds):,}\nHardware: Arduino UNO R4 WiFi"
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.legend(fontsize=11, loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Speichern
            plot_filename = f"arduino_soc_prediction_FIXED_{start_sec}s_{DURATION_SEC}s.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Plot saved: {plot_filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Plot creation error: {e}")

    def cleanup(self):
        """Ressourcen aufräumen"""
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
                logger.info("🔌 Arduino connection closed")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description='Arduino LSTM SOC Prediction - FIXED VERSION')
    parser.add_argument('--port', '-p', default=ARDUINO_PORT, help='Serial port')
    parser.add_argument('--no-upload', action='store_true', help='Skip sketch upload')
    parser.add_argument('--start', type=int, default=START_TIME_SEC, help='Start time (seconds)')
    parser.add_argument('--duration', type=int, default=DURATION_SEC, help='Duration (seconds)')
    parser.add_argument('--sample-rate', type=int, default=SAMPLE_EVERY_N, help='Sampling rate')
    
    args = parser.parse_args()
    
    processor = ArduinoStatefulLSTMProcessor(
        args.port, BAUDRATE, DATA_PATH, 
        ARDUINO_SKETCH_PATH, ARDUINO_FQBN
    )
    
    try:
        success = processor.run_batch_processing(
            auto_upload=not args.no_upload,
            start_time_sec=args.start,
            duration_sec=args.duration,
            sample_rate=args.sample_rate
        )
        
        if success:
            print("✅ Processing completed successfully!")
        else:
            print("❌ Processing failed!")
            
    except KeyboardInterrupt:
        print("⏹️ Stopped by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
