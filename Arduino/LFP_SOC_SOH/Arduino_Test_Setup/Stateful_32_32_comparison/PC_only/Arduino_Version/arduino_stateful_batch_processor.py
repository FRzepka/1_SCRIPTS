#!/usr/bin/env python3
"""
🔬 Arduino Stateful LSTM SOC Prediction - Batch Processing Version
Microcontroller Deployment Analysis for Arduino UNO R4 WiFi

Features:
- Arduino LSTM Hardware Acceleration 
- Konfigurierbarer Zeitbereich (Start + Duration in Sekunden)
- Batch Processing ohne Live-Delays
- Ground Truth vs. Arduino Prediction Plot
- MAE Berechnung und Anzeige
- Automatisches Sketch Upload
- Publication-ready Scientific Plot

Target Hardware: Arduino UNO R4 WiFi (Cortex-M4, 32KB RAM, 256KB Flash)
Based on: stateful_only_plot.py but running on Arduino hardware
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
DURATION_SEC = 10000     # Dauer in Sekunden (reduziert für Tests)

# === PERFORMANCE OPTIMIERUNGEN ===
BATCH_SIZE = 10         # Verarbeite mehrere Samples gleichzeitig
SERIAL_TIMEOUT = 0.1    # Reduzierter Timeout für schnellere Verarbeitung
MAX_RETRIES = 2         # Weniger Wiederholungen bei Fehlern
ENABLE_SAMPLING = True  # Aktiviert intelligentes Sampling für Tests
SAMPLE_EVERY_N = 5      # Nur jedes N-te Sample verarbeiten (wenn Sampling aktiv)

# === ARDUINO KONFIGURATION ===
ARDUINO_PORT = 'COM13'
BAUDRATE = 115200
DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
SCALER_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32\scaler.pkl"
ARDUINO_SKETCH_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32\code_weights\arduino_lstm_soc_full32_with_monitoring\arduino_lstm_soc_full32_with_monitoring.ino"
ARDUINO_CLI_PATH = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Arduino_CLI\arduino-cli.exe"
ARDUINO_FQBN = "arduino:renesas_uno:unor4wifi"  # Arduino UNO R4 WiFi

# === DEFINIERTES FARBSCHEMA (wie stateful_only_plot.py) ===
COLOR_SCHEME = {
    # 🔴 Main Stateful LSTM colors - orange/red tones
    'stateful_lstm': '#FF6B6B',      # Red/Orange for Stateful LSTM
    'stateful_secondary': '#FFB3B3', # Soft Red/Pink for secondary elements
    
    # 🔷 Blue accent - professional accent color
    'blue_accent': '#2091C9',         # Lebendiges Blau für Linien, Punkte, Verbindungen
    'ground_truth': '#2091C9',        # Blau für Ground Truth
    
    # ⚡ Kräftige Akzent-Farben für Details
    'accent_blue': '#2091C9',        # Lebendiges Blau für Linien, Punkte, Verbindungen
    'accent_violet': '#BB76F7',      # Elegantes Violett für spezielle Highlights
    'error_color': '#D9140E'         # Signalfarbe Rot für Fehler
}

# === DEBUG & VALIDATION FEATURES ===
ENABLE_DEBUG = True     # Aktiviert Debug-Output
ENABLE_PC_COMPARISON = True  # Vergleich mit PC-Modell für Validierung
DEBUG_SAMPLE_COUNT = 10 # Anzahl Samples für detaillierte Debug-Ausgabe

# Device setup
print(f"🖥️  Arduino Processing Mode")
print(f"🎯 Target Hardware: Arduino UNO R4 WiFi")
print(f"⏱️  Batch Processing Range: {START_TIME_SEC}s - {START_TIME_SEC + DURATION_SEC}s ({DURATION_SEC}s duration)")

class ArduinoUploader:
    """Handles automatic Arduino sketch compilation and upload"""
    
    def __init__(self, sketch_path, fqbn, port, cli_path=None):
        self.sketch_path = sketch_path
        self.fqbn = fqbn
        self.port = port
        self.cli_path = cli_path or ARDUINO_CLI_PATH
        self.arduino_cli_available = self.check_arduino_cli()
    
    def check_arduino_cli(self):
        """Check if arduino-cli is available at the specified path"""
        try:
            # First try the specified path
            if self.cli_path and os.path.exists(self.cli_path):
                result = subprocess.run([self.cli_path, 'version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"✅ Arduino CLI found at {self.cli_path}: {result.stdout.strip()}")
                    return True
            
            # Fallback to system PATH
            result = subprocess.run(['arduino-cli', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ Arduino CLI found in system PATH: {result.stdout.strip()}")
                self.cli_path = 'arduino-cli'  # Use system PATH version
                return True
            else:
                logger.warning("❌ Arduino CLI not found")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("❌ Arduino CLI not found")
            return False
    
    def compile_and_upload(self):
        """Compile and upload the sketch"""
        if not self.arduino_cli_available:
            return False, "Arduino CLI not installed"
        
        try:
            logger.info(f"🔨 Compiling sketch: {self.sketch_path}")
            cmd = [self.cli_path, 'compile', '--fqbn', self.fqbn, self.sketch_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("✅ Compilation successful")
                # Parse flash usage from output
                output = result.stderr + result.stdout
                import re
                flash_match = re.search(r'Sketch uses (\d+) bytes.*Maximum is (\d+) bytes', output)
                if flash_match:
                    used = int(flash_match.group(1))
                    total = int(flash_match.group(2))
                    logger.info(f"📊 Flash usage: {used} bytes ({used/1024:.1f} KB) of {total} bytes ({total/1024:.1f} KB)")
                
                # Upload
                logger.info(f"⬆️ Uploading to {self.port}...")
                cmd = [self.cli_path, 'upload', '--fqbn', self.fqbn, '-p', self.port, self.sketch_path]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    logger.info("✅ Upload successful")
                    return True, "Compile and upload successful"
                else:
                    logger.error(f"❌ Upload failed: {result.stderr}")
                    return False, result.stderr
            else:
                logger.error(f"❌ Compilation failed: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Compilation/Upload timeout"
        except Exception as e:
            return False, str(e)

class ArduinoStatefulLSTMProcessor:
    """Arduino Stateful LSTM Processor - Batch Processing wie stateful_only_plot.py"""
    
    def __init__(self, port, baudrate, data_path, sketch_path, fqbn):
        self.port = port
        self.baudrate = baudrate
        self.data_path = data_path
        self.sketch_path = sketch_path
        self.fqbn = fqbn
        
        # Arduino Uploader
        self.uploader = ArduinoUploader(sketch_path, fqbn, port, ARDUINO_CLI_PATH)
        
        # Hardware Connection
        self.arduino = None
        self.arduino_connected = False
        
        # Data Storage
        self.ground_truth_data = None
        self.scaler = None
        
        logger.info("Arduino Stateful LSTM Processor initialized")
    
    def upload_arduino_sketch(self):
        """Upload Arduino sketch automatically"""
        logger.info("🚀 Starting automatic Arduino sketch upload...")
        
        if not os.path.exists(self.sketch_path):
            logger.error(f"❌ Arduino sketch not found: {self.sketch_path}")
            return False
        
        success, message = self.uploader.compile_and_upload()
        
        if success:
            logger.info("✅ Arduino sketch uploaded successfully!")
            time.sleep(3)  # Wait for Arduino to restart
            return True
        else:
            logger.error(f"❌ Arduino upload failed: {message}")
            return False
    
    def load_test_data(self):
        """Lädt die C19 Testdaten (wie stateful_only_plot.py)"""
        try:
            logger.info(f"📊 Loading test data from: {self.data_path}")
            df = pd.read_parquet(self.data_path)
            
            required_cols = ['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_c', 'SOC_ZHU']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in data: {missing_cols}")
                return False
            
            df = df.dropna(subset=required_cols)
            df = df[df['Voltage[V]'] > 0]
            
            # Prepare data like in training
            df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
            
            self.ground_truth_data = df
            logger.info(f"✅ Test data loaded: {len(self.ground_truth_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return False
    
    def create_stateful_scaler(self):
        """Erstellt den RobustScaler für Stateful LSTM (wie stateful_only_plot.py)"""
        base_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
        base = Path(base_path)
        
        # Alle Zellen wie im Training (train + val)
        train_cells = [
            "MGFarm_18650_C01", "MGFarm_18650_C03", "MGFarm_18650_C05",
            "MGFarm_18650_C11", "MGFarm_18650_C17", "MGFarm_18650_C23"
        ]
        val_cells = [
            "MGFarm_18650_C07", "MGFarm_18650_C19", "MGFarm_18650_C21"
        ]
        all_cells = train_cells + val_cells
        
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        
        print("🔧 Creating RobustScaler for Stateful LSTM...")
        
        # Sammle alle Daten für Scaler-Fitting
        combined_data_list = []
        for cell_name in all_cells:
            cell_path = base / cell_name / "df.parquet"
            if cell_path.exists():
                df_cell = pd.read_parquet(cell_path)
                nan_count = df_cell[feats].isna().sum().sum()
                if nan_count > 0:
                    print(f"   - WARNING: {cell_name} has {nan_count} NaNs - filling with median")
                    df_cell[feats] = df_cell[feats].fillna(df_cell[feats].median())
                combined_data_list.append(df_cell[feats])
            else:
                print(f"   ❌ WARNING: {cell_path} not found")
        
        if not combined_data_list:
            raise ValueError("No training data found for scaler!")
        else:
            combined_data = pd.concat(combined_data_list, ignore_index=True)
        
        # Fit RobustScaler
        self.scaler = RobustScaler()
        self.scaler.fit(combined_data)        
        print(f"✅ RobustScaler fitted on {len(combined_data):,} samples")
        return feats
    
    def connect_arduino(self):
        """Arduino connection mit optimierten Einstellungen"""
        try:
            logger.info(f"🔌 Connecting to Arduino on {self.port}...")
            # Optimierte Serial-Einstellungen für bessere Performance
            self.arduino = serial.Serial(
                port=self.port, 
                baudrate=self.baudrate, 
                timeout=SERIAL_TIMEOUT,
                write_timeout=SERIAL_TIMEOUT,
                # Optimierte Buffer-Größen
                inter_byte_timeout=0.01
            )
            time.sleep(2)  # Kürzere Stabilisierungszeit
            
            # Test connection with STATS command
            self.arduino.write(b'STATS\n')
            time.sleep(0.2)  # Kürzerer Timeout
            if self.arduino.in_waiting > 0:
                response = self.arduino.readline().decode().strip()
                if response and ("STATS:" in response or len(response) > 5):
                    self.arduino_connected = True
                    logger.info("✅ Arduino connected successfully with optimized settings")
                    return True
            
            logger.error("Arduino connection test failed")
            return False
                
        except Exception as e:
            logger.error(f"Arduino connection failed: {e}")
            return False    
    def predict_arduino_time_range(self, df, scaler, feats, start_sec, duration_sec, 
                                   enable_sampling=ENABLE_SAMPLING, sample_rate=SAMPLE_EVERY_N, batch_size=BATCH_SIZE):
        """Führt Arduino Batch Prediction für bestimmten Zeitbereich durch (wie stateful_only_plot.py)"""
        if not self.arduino_connected:
            logger.error("Arduino not connected")
            return None, None, None
        
        # Scale features
        df_scaled = df.copy()
        df_scaled[feats] = scaler.transform(df[feats])
        
        # Calculate time range indices (assuming 1Hz sampling rate)
        end_sec = start_sec + duration_sec
        start_idx = max(0, start_sec)
        end_idx = min(len(df), end_sec)
        
        print(f"📊 Time range: {start_sec}s - {end_sec}s")
        print(f"📊 Data indices: {start_idx} - {end_idx} ({end_idx - start_idx} samples)")
          # Extract time range data with optional sampling
        seq = df_scaled[feats].values[start_idx:end_idx]
        labels = df["SOC_ZHU"].values[start_idx:end_idx]
        timestamps = df['timestamp'].values[start_idx:end_idx] if 'timestamp' in df.columns else None
          # Apply intelligent sampling if enabled
        if enable_sampling and len(seq) > 1000:
            original_length = len(seq)
            # Sample every N-th element but keep start and end
            sample_indices = list(range(0, len(seq), sample_rate))
            if sample_indices[-1] != len(seq) - 1:
                sample_indices.append(len(seq) - 1)  # Always include last sample
            
            seq = seq[sample_indices]
            labels = labels[sample_indices]
            if timestamps is not None:
                timestamps = timestamps[sample_indices]
            
            print(f"🔍 Sampling enabled: {original_length:,} → {len(seq):,} samples (every {sample_rate}th)")
        
        total = len(seq)
        print(f"🧠 Arduino Stateful Prediction: {total:,} samples...")
        
        preds = []
        
        # Reset Arduino state for new sequence
        try:
            self.arduino.write(b'RESET\n')
            time.sleep(0.1)
        except:
            pass        # Process with optimized batch processing
        progress_bar = tqdm(range(0, total, batch_size), desc="🧠 Arduino Inference", 
                           unit="batch", ncols=100, 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}')
        
        successful_predictions = 0
        batch_count = 0
        
        for batch_start in progress_bar:
            batch_end = min(batch_start + batch_size, total)
            batch_size = batch_end - batch_start
            batch_count += 1
            
            # Process batch
            batch_predictions = []
            
            for i in range(batch_start, batch_end):
                retries = 0
                prediction_success = False
                
                while retries < MAX_RETRIES and not prediction_success:
                    try:                        # Scale features for current sample
                        scaled_features = seq[i]
                        
                        # Create command (OHNE "DATA:" Präfix - Arduino erwartet nur V,I,SOH,Q_c)
                        command = f"{scaled_features[0]:.6f},{scaled_features[1]:.6f},{scaled_features[2]:.6f},{scaled_features[3]:.6f}\n"
                        
                        # Send to Arduino
                        self.arduino.write(command.encode())
                        
                        # Read response with optimized timeout
                        start_time = time.time()
                        response = ""
                        while time.time() - start_time < SERIAL_TIMEOUT:
                            if self.arduino.in_waiting > 0:
                                response = self.arduino.readline().decode().strip()
                                break
                            # Keine time.sleep() für bessere Performance
                        
                        if response and response.startswith("DATA:"):
                            # Parse: "DATA:SOC,inference_time_us,ram_free,ram_used,cpu_load,temp"
                            data = response.replace("DATA:", "").split(",")
                            if len(data) >= 1:
                                soc_pred = float(data[0])
                                batch_predictions.append(soc_pred)
                                successful_predictions += 1
                                prediction_success = True
                            else:
                                retries += 1
                        else:
                            retries += 1
                            
                    except Exception as e:
                        retries += 1
                        if retries >= MAX_RETRIES:
                            logger.warning(f"Failed prediction at sample {i} after {MAX_RETRIES} retries: {e}")
                
                if not prediction_success:
                    batch_predictions.append(0.0)  # Fallback
            
            # Add batch predictions to results
            preds.extend(batch_predictions)
            
            # Update progress with performance stats
            samples_processed = batch_end
            success_rate = successful_predictions / samples_processed * 100
            throughput = successful_predictions / (time.time() - start_time + 0.001)  # Samples per second
            
            progress_bar.set_postfix({
                'Samples': f'{samples_processed:,}/{total:,}',
                'Success': f'{success_rate:.1f}%',
                'Rate': f'{throughput:.1f}Hz'
            })
        
        preds = np.array(preds)
        gts = labels[:len(preds)]
        
        success_rate = successful_predictions / total * 100
        print(f"✅ Arduino prediction completed! Generated {len(preds):,} predictions.")
        print(f"📊 Success rate: {success_rate:.1f}% ({successful_predictions}/{total})")
        
        return preds, gts, timestamps[:len(preds)] if timestamps is not None else None
    
    def create_arduino_plot(self, preds, gts, timestamps, mae_value, start_sec):
        """Erstellt den Arduino Stateful LSTM Plot (wie stateful_only_plot.py)"""
        
        # Erstelle Zeit-Achse in Sekunden relativ zum Start
        time_seconds = np.arange(len(preds)) + start_sec
        
        # === PLOT ERSTELLEN ===
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Plot Ground Truth und Prediction
        ax.plot(time_seconds, gts, 
               color=COLOR_SCHEME['ground_truth'], 
               linewidth=2.0, 
               label='Ground Truth SOC',
               alpha=0.8)
        
        ax.plot(time_seconds, preds, 
               color=COLOR_SCHEME['stateful_lstm'], 
               linewidth=1.8, 
               label='Arduino Stateful LSTM',
               alpha=0.9)
          # Styling
        ax.set_xlabel('Time [seconds]', fontsize=14, fontweight='bold')
        ax.set_ylabel('State of Charge (SOC)', fontsize=14, fontweight='bold')
        ax.set_title(f'🤖 Arduino Stateful LSTM SOC Prediction\n'
                    f'Time Range: {start_sec}s - {start_sec + (len(preds))}s | MAE: {mae_value:.4f}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Grid und Layout
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        
        # Y-Achse auf SOC-Bereich beschränken
        ax.set_ylim(0, 1)
        
        # MAE Text Box
        textstr = f'MAE: {mae_value:.4f}\nSamples: {len(preds):,}\nHardware: Arduino UNO R4 WiFi'
        props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
          # Speichern
        output_path = Path("arduino_stateful_soc_prediction.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {output_path.absolute()}")
        
        plt.show()    
    def run_batch_processing(self, auto_upload=True, start_time_sec=START_TIME_SEC, duration_sec=DURATION_SEC, 
                           enable_sampling=ENABLE_SAMPLING, sample_rate=SAMPLE_EVERY_N, batch_size=BATCH_SIZE):
        """Hauptfunktion für Arduino Batch Processing"""
        
        print("🔬 Starting Arduino Stateful LSTM SOC Prediction - Batch Processing")
        print("=" * 80)
        print(f"⏱️  Processing Range: {start_time_sec}s - {start_time_sec + duration_sec}s ({duration_sec}s duration)")
        if enable_sampling:
            print(f"🚀 Performance Mode: Sampling every {sample_rate}th point, batch size {batch_size}")
        
        if auto_upload:
            print("\n⬆️ Uploading Arduino sketch...")
            if not self.upload_arduino_sketch():
                print("❌ Upload failed. Continue with existing sketch? (y/n): ", end="")
                if input().lower() != 'y':
                    return False
        
        # === SCALER ERSTELLEN ===
        print("\n🔧 Creating Scaler...")
        feats = self.create_stateful_scaler()
        
        # === TESTDATEN LADEN ===
        print("\n📊 Loading Test Data...")
        if not self.load_test_data():
            return False
        
        # === ARDUINO VERBINDEN ===
        print("\n🔌 Connecting to Arduino...")
        if not self.connect_arduino():
            return False        # === VORHERSAGE FÜR ZEITBEREICH ===
        print(f"\n🧠 Running Arduino Stateful Prediction for time range {start_time_sec}s - {start_time_sec + duration_sec}s...")
        preds, gts, timestamps = self.predict_arduino_time_range(
            self.ground_truth_data, self.scaler, feats, 
            start_time_sec, duration_sec, enable_sampling, sample_rate, batch_size
        )
        
        if preds is None:
            print("❌ Prediction failed")
            return False
        
        # === MAE BERECHNUNG ===
        mae_value = mean_absolute_error(gts, preds)
        rmse_value = np.sqrt(mean_squared_error(gts, preds))
        r2_value = r2_score(gts, preds)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   MAE:  {mae_value:.6f}")
        print(f"   RMSE: {rmse_value:.6f}")
        print(f"   R²:   {r2_value:.6f}")
        
        # === PLOT ERSTELLEN ===
        print(f"\n📊 Creating Plot...")
        self.create_arduino_plot(preds, gts, timestamps, mae_value, start_time_sec)
        
        print("\n✅ Arduino Analysis completed successfully!")
        return True
    
    def debug_arduino_prediction(self, df, scaler, feats, num_samples=5):
        """Debug-Funktion um Arduino-Predictions zu analysieren"""
        if not self.arduino_connected:
            logger.error("Arduino not connected for debugging")
            return False
        
        print("\n🔍 === ARDUINO DEBUG ANALYSIS ===")
        
        # Scale features
        df_scaled = df.copy()
        df_scaled[feats] = scaler.transform(df[feats])
        
        # Take first few samples for detailed analysis
        test_samples = min(num_samples, len(df))
        
        for i in range(test_samples):
            try:
                # Original data
                orig_data = df.iloc[i]
                scaled_data = df_scaled[feats].iloc[i].values
                true_soc = orig_data['SOC_ZHU']
                
                print(f"\n📊 Sample {i+1}:")
                print(f"   Original: V={orig_data['Voltage[V]']:.3f}, I={orig_data['Current[A]']:.3f}, SOH={orig_data['SOH_ZHU']:.3f}, Q_c={orig_data['Q_c']:.3f}")
                print(f"   Scaled:   V={scaled_data[0]:.6f}, I={scaled_data[1]:.6f}, SOH={scaled_data[2]:.6f}, Q_c={scaled_data[3]:.6f}")
                print(f"   True SOC: {true_soc:.4f}")
                
                # Send to Arduino
                command = f"DATA:{scaled_data[0]:.6f},{scaled_data[1]:.6f},{scaled_data[2]:.6f},{scaled_data[3]:.6f}\n"
                print(f"   Command:  {command.strip()}")
                
                self.arduino.write(command.encode())
                
                # Read response with timeout
                start_time = time.time()
                response = ""
                while time.time() - start_time < 1.0:  # Longer timeout for debug
                    if self.arduino.in_waiting > 0:
                        response = self.arduino.readline().decode().strip()
                        break
                    time.sleep(0.01)
                
                print(f"   Response: {response}")
                
                if response and response.startswith("DATA:"):
                    data = response.replace("DATA:", "").split(",")
                    if len(data) >= 1:
                        arduino_soc = float(data[0])
                        error = abs(arduino_soc - true_soc)
                        print(f"   Arduino:  {arduino_soc:.4f}")
                        print(f"   Error:    {error:.4f}")
                        
                        if len(data) >= 6:
                            print(f"   Stats:    Time={data[1]}μs, RAM_free={data[2]}B, CPU={data[4]}%")
                    else:
                        print(f"   ❌ Invalid response format")
                else:
                    print(f"   ❌ No valid response received")
                
                time.sleep(0.1)  # Small delay between samples
                
            except Exception as e:
                print(f"   ❌ Error in sample {i+1}: {e}")
        
        print("\n🔍 === DEBUG ANALYSIS COMPLETE ===\n")
        return True

    def validate_arduino_vs_pc_model(self, df, scaler, feats, num_samples=10):
        """Validiert Arduino-Modell gegen PC-Modell"""
        if not self.arduino_connected:
            logger.error("Arduino not connected for validation")
            return False
        
        print("\n⚖️  === ARDUINO vs PC MODEL VALIDATION ===")
        
        try:
            # Lade PC-Modell für Vergleich
            from sklearn.preprocessing import RobustScaler
            import torch
            import torch.nn as nn
            
            # Definiere PC-Modell-Klasse (vereinfacht)
            class StatefulSOCModel(nn.Module):
                def __init__(self, input_size=4, dropout=0.03):
                    super().__init__()
                    self.hidden_size = 32
                    self.num_layers = 1
                    
                    self.lstm = nn.LSTM(
                        input_size, 32, 1,
                        batch_first=True, dropout=0.0
                    )
                    
                    self.mlp = nn.Sequential(
                        nn.Linear(32, 32),
                        nn.LayerNorm(32),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(32, 32),
                        nn.LayerNorm(32),
                        nn.ReLU(),
                        nn.Dropout(dropout * 0.5),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x, hidden):
                    out, new_hidden = self.lstm(x, hidden)
                    out = self.mlp(out)
                    return out, new_hidden
            
            # Versuche PC-Modell zu laden
            pc_model_path = Path(r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32_comparison\PC_only\models_for_PC_only\best_model_sf_32")
            
            if pc_model_path.exists():
                device = torch.device("cpu")
                pc_model = StatefulSOCModel(input_size=4, dropout=0.03)
                pc_model.load_state_dict(torch.load(pc_model_path, map_location=device, weights_only=False))
                pc_model.eval()
                
                print("✅ PC-Modell geladen für Vergleich")
                
                # Initialize hidden states
                h = torch.zeros(1, 1, 32, device=device)
                c = torch.zeros_like(h)
                
                # Scale features
                df_scaled = df.copy()
                df_scaled[feats] = scaler.transform(df[feats])
                
                print(f"\n📊 Vergleiche erste {num_samples} Samples:")
                
                total_arduino_error = 0
                total_pc_error = 0
                valid_comparisons = 0
                
                for i in range(min(num_samples, len(df))):
                    try:
                        # Get data
                        scaled_data = df_scaled[feats].iloc[i].values
                        true_soc = df.iloc[i]['SOC_ZHU']
                        
                        # PC Prediction
                        input_tensor = torch.tensor(scaled_data, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                        with torch.no_grad():
                            pc_output, (h, c) = pc_model(input_tensor, (h, c))
                            pc_soc = pc_output.squeeze().item()
                        
                        # Arduino Prediction
                        command = f"DATA:{scaled_data[0]:.6f},{scaled_data[1]:.6f},{scaled_data[2]:.6f},{scaled_data[3]:.6f}\n"
                        self.arduino.write(command.encode())
                        
                        start_time = time.time()
                        response = ""
                        while time.time() - start_time < 1.0:
                            if self.arduino.in_waiting > 0:
                                response = self.arduino.readline().decode().strip()
                                break
                            time.sleep(0.01)
                        
                        if response and response.startswith("DATA:"):
                            data = response.replace("DATA:", "").split(",")
                            if len(data) >= 1:
                                arduino_soc = float(data[0])
                                
                                # Calculate errors
                                arduino_error = abs(arduino_soc - true_soc)
                                pc_error = abs(pc_soc - true_soc)
                                model_diff = abs(arduino_soc - pc_soc)
                                
                                total_arduino_error += arduino_error
                                total_pc_error += pc_error
                                valid_comparisons += 1
                                
                                print(f"   Sample {i+1}: True={true_soc:.4f}, PC={pc_soc:.4f}, Arduino={arduino_soc:.4f}")
                                print(f"            PC_Err={pc_error:.4f}, Arduino_Err={arduino_error:.4f}, Diff={model_diff:.4f}")
                            else:
                                print(f"   Sample {i+1}: ❌ Invalid Arduino response")
                        else:
                            print(f"   Sample {i+1}: ❌ No Arduino response")
                        
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"   Sample {i+1}: ❌ Error: {e}")
                
                if valid_comparisons > 0:
                    avg_arduino_error = total_arduino_error / valid_comparisons
                    avg_pc_error = total_pc_error / valid_comparisons
                    
                    print(f"\n📊 VERGLEICHSERGEBNIS:")
                    print(f"   PC-Modell MAE:      {avg_pc_error:.6f}")
                    print(f"   Arduino-Modell MAE: {avg_arduino_error:.6f}")
                    print(f"   Verhältnis:         {avg_arduino_error/avg_pc_error:.2f}x")
                    
                    if avg_arduino_error > avg_pc_error * 2:
                        print(f"   ⚠️  Arduino-Modell ist deutlich schlechter!")
                        print(f"   🔧 Mögliche Ursachen:")
                        print(f"      - Unterschiedliche Gewichte")
                        print(f"      - Float-Precision-Unterschiede") 
                        print(f"      - Falsche Skalierung")
                        print(f"      - LSTM-State-Probleme")
                    else:
                        print(f"   ✅ Arduino-Modell Performance akzeptabel")
                
            else:
                print("❌ PC-Modell nicht gefunden - verwende nur Arduino-Debug")
                  except Exception as e:
            print(f"❌ Fehler bei PC-Modell-Vergleich: {e}")
        
        print("\n⚖️  === VALIDATION COMPLETE ===\n")
        return True
    
    def debug_single_prediction(self, test_data_point):
        """Debug-Funktion für einzelne Vorhersage"""
        if not self.arduino or not self.arduino.is_open:
            logger.error("❌ Arduino nicht verbunden")
            return None
            
        try:
            # Sende Testdaten im korrekten Format
            # Format: V,I,SOH,Q_c (ohne "DATA:" Prefix)
            data_str = f"{test_data_point[0]:.6f},{test_data_point[1]:.6f},{test_data_point[2]:.6f},{test_data_point[3]:.6f}\n"
            logger.info(f"🔍 Sende Debug-Daten: {data_str.strip()}")
            
            self.arduino.write(data_str.encode())
            self.arduino.flush()
            
            # Warte auf Antwort
            response = ""
            start_time = time.time()
            while time.time() - start_time < 2.0:  # 2s timeout
                if self.arduino.in_waiting > 0:
                    response += self.arduino.read(self.arduino.in_waiting).decode()
                    if '\n' in response:
                        break
                time.sleep(0.01)
            
            logger.info(f"🔍 Arduino Antwort: {response.strip()}")
            
            # Parse Antwort - Arduino antwortet mit "DATA:soc_pred,inference_time,..." 
            if response.strip() and response.startswith("DATA:"):
                try:
                    data_parts = response.replace("DATA:", "").split(",")
                    prediction = float(data_parts[0])
                    return prediction
                except (ValueError, IndexError):
                    logger.error(f"❌ Konnte Antwort nicht parsen: {response}")
                    return None
            else:
                logger.error("❌ Keine gültige Antwort vom Arduino")
                return None
                
        except Exception as e:
            logger.error(f"❌ Debug-Fehler: {e}")
            return None
    
    def compare_with_pc_model(self, test_samples=10):
        """Vergleiche Arduino mit PC-Modell"""
        logger.info(f"🔍 Vergleiche Arduino mit PC-Modell ({test_samples} Samples)")
        
        # Lade Testdaten
        try:
            data = pd.read_csv(DATA_PATH)
            scaler = joblib.load(SCALER_PATH)
            
            # Wähle zufällige Testpunkte
            test_indices = np.random.choice(len(data), min(test_samples, len(data)), replace=False)
            
            comparisons = []
            for i, idx in enumerate(test_indices):
                row = data.iloc[idx]
                input_data = np.array([row['V'], row['I'], row['SOH'], row['Q_c']])
                scaled_data = scaler.transform([input_data])[0]
                
                # Arduino Vorhersage
                arduino_pred = self.debug_single_prediction(scaled_data)
                
                if arduino_pred is not None:
                    # Ground Truth
                    true_soc = row['SOC']
                    
                    comparisons.append({
                        'index': idx,
                        'true_soc': true_soc,
                        'arduino_pred': arduino_pred,
                        'error': abs(true_soc - arduino_pred),
                        'input_data': input_data.tolist()
                    })
                    
                    logger.info(f"📊 Sample {i+1}: True={true_soc:.4f}, Arduino={arduino_pred:.4f}, Error={abs(true_soc - arduino_pred):.4f}")
                else:
                    logger.warning(f"⚠️ Sample {i+1}: Keine Arduino-Antwort")
                
                time.sleep(0.1)  # Kurze Pause zwischen Tests
            
            if comparisons:
                avg_error = np.mean([c['error'] for c in comparisons])
                max_error = np.max([c['error'] for c in comparisons])
                logger.info(f"📈 Durchschnittlicher Fehler: {avg_error:.4f}")
                logger.info(f"📈 Maximaler Fehler: {max_error:.4f}")
                return comparisons
            else:
                logger.error("❌ Keine erfolgreichen Vergleiche")
                return []
                
        except Exception as e:
            logger.error(f"❌ Fehler beim PC-Modell-Vergleich: {e}")
            return []
    
    def test_arduino_communication(self):
        """Teste grundlegende Arduino-Kommunikation"""
        if not self.arduino or not self.arduino.is_open:
            logger.error("❌ Arduino nicht verbunden")
            return False
            
        try:
            logger.info("🔍 Teste Arduino-Kommunikation...")
            
            # Sende einfachen Test-Befehl
            test_commands = [
                "0.5,0.1,0.8,100.0\n",
                "0.6,0.2,0.8,100.0\n", 
                "0.7,0.3,0.8,100.0\n"
            ]
            
            for i, cmd in enumerate(test_commands):
                logger.info(f"🔍 Test {i+1}: Sende '{cmd.strip()}'")
                
                self.arduino.write(cmd.encode())
                self.arduino.flush()
                
                # Warte auf Antwort
                response = ""
                start_time = time.time()
                while time.time() - start_time < 2.0:
                    if self.arduino.in_waiting > 0:
                        response += self.arduino.read(self.arduino.in_waiting).decode()
                        if '\n' in response:
                            break
                    time.sleep(0.01)
                
                logger.info(f"🔍 Antwort {i+1}: '{response.strip()}'")
                
                if not response.strip():
                    logger.error(f"❌ Keine Antwort bei Test {i+1}")
                    return False
                    
                time.sleep(0.5)  # Pause zwischen Tests
            
            logger.info("✅ Arduino-Kommunikation funktioniert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Kommunikationstest fehlgeschlagen: {e}")
            return False
    def upload_sketch(self, auto_upload=True):
        """Alias für upload_arduino_sketch für bessere Kompatibilität"""
        if auto_upload:
            return self.upload_arduino_sketch()
        else:
            logger.info("⏭️ Sketch upload übersprungen")
            return True
    
    def connect_to_arduino(self):
        """Alias für connect_arduino für bessere Kompatibilität"""
        return self.connect_arduino()
    
    def cleanup(self):
        """Ressourcen aufräumen"""
        try:
            if hasattr(self, 'arduino') and self.arduino and self.arduino.is_open:
                self.arduino.close()
                logger.info("🔌 Arduino connection closed")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")
def main():
    """Main function mit Konfiguration"""
    parser = argparse.ArgumentParser(description='Arduino Stateful LSTM SOC Prediction - Batch Processing')
    parser.add_argument('--port', '-p', default=ARDUINO_PORT, help=f'Serial port (default: {ARDUINO_PORT})')
    parser.add_argument('--baudrate', '-b', type=int, default=BAUDRATE, help=f'Baud rate (default: {BAUDRATE})')
    parser.add_argument('--no-upload', action='store_true', help='Skip automatic upload')
    parser.add_argument('--sketch', '-s', default=ARDUINO_SKETCH_PATH, help='Arduino sketch path')
    parser.add_argument('--fqbn', '-f', default=ARDUINO_FQBN, help='Arduino FQBN')
    parser.add_argument('--start', type=int, default=START_TIME_SEC, help=f'Start time in seconds (default: {START_TIME_SEC})')
    parser.add_argument('--duration', type=int, default=DURATION_SEC, help=f'Duration in seconds (default: {DURATION_SEC})')
    parser.add_argument('--fast', action='store_true', help='Enable fast mode with sampling')
    parser.add_argument('--sample-rate', type=int, default=SAMPLE_EVERY_N, help=f'Sample every N-th point (default: {SAMPLE_EVERY_N})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size for processing (default: {BATCH_SIZE})')
    parser.add_argument('--debug', action='store_true', help='Run debug tests only')
    parser.add_argument('--compare', action='store_true', help='Run Arduino vs PC comparison')
    
    args = parser.parse_args()
    
    # Performance settings from arguments
    enable_sampling = args.fast if hasattr(args, 'fast') else ENABLE_SAMPLING
    sample_rate = args.sample_rate if hasattr(args, 'sample_rate') else SAMPLE_EVERY_N
    batch_size = args.batch_size if hasattr(args, 'batch_size') else BATCH_SIZE
    
    if args.fast:
        print(f"🚀 Fast mode enabled: Sampling every {sample_rate}th point, batch size {batch_size}")
    
    processor = ArduinoStatefulLSTMProcessor(
        args.port, args.baudrate, DATA_PATH, 
        args.sketch, args.fqbn
    )
    
    try:
        # Debug-Modi
        if args.debug:
            logger.info("🔍 Starte Debug-Modus...")
            if processor.upload_sketch(auto_upload=not args.no_upload):
                if processor.connect_to_arduino():
                    processor.test_arduino_communication()
                    processor.cleanup()
            return
        
        if args.compare:
            logger.info("🔍 Starte Arduino vs PC Vergleich...")
            if processor.upload_sketch(auto_upload=not args.no_upload):
                if processor.connect_to_arduino():
                    processor.compare_with_pc_model(test_samples=5)
                    processor.cleanup()
            return
        
        # Normaler Batch-Processing-Modus
        success = processor.run_batch_processing(
            auto_upload=not args.no_upload, 
            start_time_sec=args.start, 
            duration_sec=args.duration,
            enable_sampling=enable_sampling,
            sample_rate=sample_rate,
            batch_size=batch_size
        )
        if not success:
            print("❌ Processing failed")
    except KeyboardInterrupt:
        print("⏹️ Program stopped by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
