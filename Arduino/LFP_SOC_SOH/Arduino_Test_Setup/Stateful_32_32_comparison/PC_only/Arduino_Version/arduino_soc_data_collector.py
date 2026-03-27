#!/usr/bin/env python3
"""
🔬 Arduino SOC Data Collector - Basierend auf arduino_soc_monitor_sf_32_32.py
===========================================================================

✨ FEATURES:
- Basiert auf dem funktionierenden arduino_soc_monitor_sf_32_32.py
- Sammelt Arduino SOC Vorhersagen von Sekunde 0 bis zu einer bestimmten Zeit
- Speichert alle Daten in CSV-Datei für spätere Analyse
- KEINE Live-Plots - nur Datensammlung
- Verwendet exakt die gleiche Arduino-Kommunikation wie das funktionierende Script

🎯 ZIEL: CSV-Datei mit Ground Truth vs Arduino Predictions erstellen
"""

import serial
import pandas as pd
import numpy as np
import time
import subprocess
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import warnings
import argparse
from tqdm import tqdm

# Warnings unterdrücken
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font", category=UserWarning)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== EINSTELLUNGEN (identisch zu arduino_soc_monitor_sf_32_32.py) =====
ARDUINO_PORT = 'COM13'
BAUDRATE = 115200
DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
ARDUINO_SKETCH_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32\code_weights\arduino_lstm_soc_full32_with_monitoring\arduino_lstm_soc_full32_with_monitoring.ino"
ARDUINO_CLI_PATH = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Arduino_CLI\arduino-cli.exe"
ARDUINO_FQBN = "arduino:renesas_uno:unor4wifi"

# Datensammlung Einstellungen
DEFAULT_END_TIME = 1000  # Bis Sekunde 1000 sammeln
PREDICTION_DELAY = 30   # 30ms zwischen Predictions (wie im Original)

print("📊 Arduino SOC Data Collector")
print("🎯 Basierend auf funktionierendem arduino_soc_monitor_sf_32_32.py")
print("💾 Sammelt Daten und speichert in CSV für spätere Analyse")

class ArduinoUploader:
    """Exakt identisch zu arduino_soc_monitor_sf_32_32.py"""
    
    def __init__(self, sketch_path, fqbn, port, cli_path=None):
        self.sketch_path = sketch_path
        self.fqbn = fqbn
        self.port = port
        self.cli_path = cli_path or ARDUINO_CLI_PATH
        self.arduino_cli_available = self.check_arduino_cli()
    
    def check_arduino_cli(self):
        """Check if arduino-cli is available at the specified path"""
        try:
            if self.cli_path and os.path.exists(self.cli_path):
                result = subprocess.run([self.cli_path, 'version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"✅ Arduino CLI found at {self.cli_path}: {result.stdout.strip()}")
                    return True
            
            result = subprocess.run(['arduino-cli', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ Arduino CLI found in system PATH: {result.stdout.strip()}")
                self.cli_path = 'arduino-cli'
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
                output = result.stderr + result.stdout
                import re
                flash_match = re.search(r'Sketch uses (\d+) bytes.*Maximum is (\d+) bytes', output)
                if flash_match:
                    used = int(flash_match.group(1))
                    total = int(flash_match.group(2))
                    logger.info(f"📊 Flash usage: {used} bytes ({used/1024:.1f} KB) of {total} bytes ({total/1024:.1f} KB)")
                
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

class ArduinoSOCDataCollector:
    """Arduino SOC Data Collector - sammelt Daten in CSV statt live plotting"""
    
    def __init__(self, port, baudrate, data_path, sketch_path, fqbn):
        self.port = port
        self.baudrate = baudrate
        self.data_path = data_path
        self.sketch_path = sketch_path
        self.fqbn = fqbn
        
        # Arduino Uploader (identisch zu arduino_soc_monitor_sf_32_32.py)
        self.uploader = ArduinoUploader(sketch_path, fqbn, port, ARDUINO_CLI_PATH)
        
        # Hardware Connection
        self.arduino = None
        self.arduino_connected = False
        
        # Data Storage für CSV
        self.collected_data = []
        
        # Ground Truth Data
        self.ground_truth_data = None
        self.scaler = None
        self.data_index = 0
        
        # Performance Tracking
        self.prediction_count = 0
        self.successful_predictions = 0
        self.communication_errors = 0
        self.start_time = time.time()
        
        logger.info("Arduino SOC Data Collector initialized")
    
    def upload_arduino_sketch(self):
        """Upload Arduino sketch automatically (identisch zu arduino_soc_monitor_sf_32_32.py)"""
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
    
    def load_ground_truth_data(self):
        """Lädt Ground Truth Data (identisch zu arduino_soc_monitor_sf_32_32.py)"""
        try:
            logger.info(f"Loading Ground Truth data from: {self.data_path}")
            df = pd.read_parquet(self.data_path)
            
            required_cols = ['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_c', 'SOC_ZHU']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in data: {missing_cols}")
                return False
            
            df = df.dropna(subset=required_cols)
            df = df[df['Voltage[V]'] > 0]
            
            df_clean = df.rename(columns={
                'Voltage[V]': 'voltage',
                'Current[A]': 'current', 
                'SOH_ZHU': 'soh',
                'Q_c': 'q_c',
                'SOC_ZHU': 'soc'
            })
                
            self.ground_truth_data = df_clean[['voltage', 'current', 'soh', 'q_c', 'soc']].copy()
            logger.info(f"✅ Ground Truth data loaded: {len(self.ground_truth_data)} samples")
            
            self.setup_scaler()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Ground Truth data: {e}")
            return False
    
    def setup_scaler(self):
        """Setup scaler (identisch zu arduino_soc_monitor_sf_32_32.py)"""
        try:
            scaler_path = Path(self.data_path).parent / "scaler.pkl"
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Loaded existing scaler")
            else:
                feature_cols = ['voltage', 'current', 'soh', 'q_c']
                features = self.ground_truth_data[feature_cols].values
                
                self.scaler = StandardScaler()
                self.scaler.fit(features)
                
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info("✅ Created and saved new scaler")
                
        except Exception as e:
            logger.error(f"Scaler setup failed: {e}")
            self.scaler = StandardScaler()
    
    def scale_features(self, voltage, current, soh, q_c):
        """Scale features (identisch zu arduino_soc_monitor_sf_32_32.py)"""
        features = np.array([[voltage, current, soh, q_c]])
        if self.scaler:
            return self.scaler.transform(features)[0]
        return features[0]
    
    def get_data_sample_by_index(self, index):
        """Get specific data sample by index"""
        if self.ground_truth_data is None or len(self.ground_truth_data) == 0:
            return None
            
        try:
            if index >= len(self.ground_truth_data):
                return None  # Ende der Daten erreicht
            
            sample = self.ground_truth_data.iloc[index]
            
            return {
                'voltage': float(sample['voltage']),
                'current': float(sample['current']),
                'soh': float(sample['soh']),
                'q_c': float(sample['q_c']),
                'soc_true': float(sample['soc'])
            }
            
        except Exception as e:
            logger.error(f"Error getting data sample at index {index}: {e}")
            return None
    
    def connect_arduino(self):
        """Arduino connection (identisch zu arduino_soc_monitor_sf_32_32.py)"""
        try:
            logger.info(f"Connecting to Arduino on {self.port}...")
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(3)  # Arduino connection stabilization
            
            # Test connection with STATS command
            self.arduino.write(b'STATS\n')
            time.sleep(0.5)
            response = self.arduino.readline().decode().strip()
            if response and ("STATS:" in response or len(response) > 5):
                self.arduino_connected = True
                logger.info("✅ Arduino connected successfully")
                return True
            else:
                logger.error(f"Arduino connection test failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Arduino connection failed: {e}")
            return False
    
    def predict_with_arduino(self, voltage, current, soh, q_c):
        """Arduino LSTM Prediction (identisch zu arduino_soc_monitor_sf_32_32.py)"""
        if not self.arduino_connected:
            return None, None
            
        try:
            # Scale features
            scaled_features = self.scale_features(voltage, current, soh, q_c)
            
            # Create command - WICHTIG: Verwendet das funktionierende Format!
            command = f"DATA:{scaled_features[0]:.6f},{scaled_features[1]:.6f},{scaled_features[2]:.6f},{scaled_features[3]:.6f}\n"
            
            # Send to Arduino
            self.arduino.write(command.encode())
            
            # Read response with timeout
            response = self.arduino.readline().decode().strip()
            
            if response and response.startswith("DATA:"):
                # Parse: "DATA:SOC,inference_time_us,ram_free,ram_used,cpu_load,temp"
                data = response.replace("DATA:", "").split(",")
                if len(data) >= 6:
                    soc_pred = float(data[0])
                    inference_us = float(data[1])
                    ram_free = int(data[2])
                    ram_used = int(data[3])
                    cpu_load = float(data[4])
                    temperature = float(data[5])
                    
                    self.successful_predictions += 1
                    return soc_pred, {'inference_us': inference_us, 'ram_free': ram_free, 
                                     'ram_used': ram_used, 'cpu_load': cpu_load, 'temp': temperature}
            
            self.communication_errors += 1
            return None, None
            
        except Exception as e:
            logger.warning(f"Arduino prediction failed: {e}")
            self.communication_errors += 1
            return None, None
    
    def collect_data(self, end_time_sec=DEFAULT_END_TIME, auto_upload=True):
        """Hauptfunktion: Sammelt Daten von Sekunde 0 bis end_time_sec"""
        try:
            print(f"🚀 Arduino SOC Data Collection: 0s - {end_time_sec}s")
            print("="*60)
            
            if auto_upload:
                print("⬆️ Uploading Arduino sketch...")
                if not self.upload_arduino_sketch():
                    print("❌ Upload failed. Continue with existing sketch? (y/n): ", end="")
                    if input().lower() != 'y':
                        return False
            
            print("📊 Loading ground truth data...")
            if not self.load_ground_truth_data():
                print("❌ Failed to load data")
                return False
            
            print("🔌 Connecting to Arduino...")
            if not self.connect_arduino():
                print("❌ Failed to connect to Arduino")
                return False
            
            # Reset Arduino LSTM State
            self.arduino.write(b"RESET\n")
            time.sleep(0.2)
            logger.info("🔄 Arduino LSTM state reset")
            
            # Bestimme Anzahl der Samples bis end_time_sec
            max_samples = min(end_time_sec, len(self.ground_truth_data))
            
            print(f"📊 Collecting {max_samples} samples (0s to {end_time_sec}s)...")
            print("💾 Data will be saved to CSV file")
            
            # Progress bar
            progress_bar = tqdm(range(max_samples), desc="🧠 Collecting Arduino Predictions", 
                              unit="sample", ncols=100)
            
            collection_start_time = time.time()
            
            for sample_idx in progress_bar:
                # Get ground truth data sample
                data_sample = self.get_data_sample_by_index(sample_idx)
                if data_sample is None:
                    logger.warning(f"No data sample at index {sample_idx}")
                    continue
                
                # Get Arduino prediction
                soc_pred, hardware_metrics = self.predict_with_arduino(
                    data_sample['voltage'], data_sample['current'], 
                    data_sample['soh'], data_sample['q_c']
                )
                
                # Collect data point
                data_point = {
                    'time_seconds': sample_idx,
                    'voltage': data_sample['voltage'],
                    'current': data_sample['current'],
                    'soh': data_sample['soh'],
                    'q_c': data_sample['q_c'],
                    'soc_ground_truth': data_sample['soc_true'],
                    'soc_arduino_prediction': soc_pred if soc_pred is not None else np.nan,
                    'mae_error': abs(soc_pred - data_sample['soc_true']) if soc_pred is not None else np.nan,
                    'arduino_inference_us': hardware_metrics['inference_us'] if hardware_metrics else np.nan,
                    'arduino_ram_free': hardware_metrics['ram_free'] if hardware_metrics else np.nan,
                    'arduino_ram_used': hardware_metrics['ram_used'] if hardware_metrics else np.nan,
                    'arduino_cpu_load': hardware_metrics['cpu_load'] if hardware_metrics else np.nan,
                    'arduino_temperature': hardware_metrics['temp'] if hardware_metrics else np.nan,
                    'prediction_successful': soc_pred is not None
                }
                
                self.collected_data.append(data_point)
                self.prediction_count += 1
                
                # Update progress bar
                success_rate = self.successful_predictions / self.prediction_count * 100
                progress_bar.set_postfix({
                    'Success': f"{success_rate:.1f}%",
                    'Current MAE': f"{data_point['mae_error']:.4f}" if not np.isnan(data_point['mae_error']) else "N/A"
                })
                
                # Delay zwischen Predictions (wie im Original)
                time.sleep(PREDICTION_DELAY / 1000.0)
            
            collection_end_time = time.time()
            collection_duration = collection_end_time - collection_start_time
            
            # Statistiken
            successful_preds = len([d for d in self.collected_data if d['prediction_successful']])
            avg_mae = np.nanmean([d['mae_error'] for d in self.collected_data])
            
            print(f"\n📊 Collection Summary:")
            print(f"   Total Samples: {len(self.collected_data)}")
            print(f"   Successful Predictions: {successful_preds} ({successful_preds/len(self.collected_data)*100:.1f}%)")
            print(f"   Average MAE: {avg_mae:.4f}")
            print(f"   Collection Time: {collection_duration:.1f}s")
            print(f"   Throughput: {len(self.collected_data)/collection_duration:.1f} samples/s")
            
            # Speichere in CSV
            self.save_to_csv(end_time_sec)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data collection error: {e}")
            return False
        
        finally:
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
                logger.info("🔌 Arduino connection closed")
    
    def plot_collected_data(self, df, end_time_sec):
        """Plottet die gesammelten Daten direkt nach der Sammlung"""
        try:
            import matplotlib.pyplot as plt
            
            print(f"\n📊 Creating plot for {len(df)} data points...")
            
            # Filter successful predictions for plotting
            df_success = df[df['prediction_successful'] == True].copy()
            
            if len(df_success) == 0:
                print("⚠️ No successful predictions to plot")
                return
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'Arduino SOC Data Collection Results (0s - {end_time_sec}s)', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: SOC Comparison
            axes[0,0].plot(df_success['time_seconds'], df_success['soc_ground_truth'], 
                          'b-', linewidth=2, label='Ground Truth SOC', alpha=0.8)
            axes[0,0].plot(df_success['time_seconds'], df_success['soc_arduino_prediction'], 
                          'r--', linewidth=2, label='Arduino LSTM Prediction', alpha=0.8)
            axes[0,0].set_title('SOC: Ground Truth vs Arduino Prediction', fontweight='bold')
            axes[0,0].set_xlabel('Time [seconds]')
            axes[0,0].set_ylabel('State of Charge')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_ylim(0, 1)
            
            # Plot 2: MAE Error over time
            axes[0,1].plot(df_success['time_seconds'], df_success['mae_error'], 
                          'orange', linewidth=1.5, alpha=0.7)
            axes[0,1].set_title('MAE Error over Time', fontweight='bold')
            axes[0,1].set_xlabel('Time [seconds]')
            axes[0,1].set_ylabel('Mean Absolute Error')
            axes[0,1].grid(True, alpha=0.3)
            
            # Add MAE statistics
            avg_mae = df_success['mae_error'].mean()
            max_mae = df_success['mae_error'].max()
            axes[0,1].axhline(y=avg_mae, color='red', linestyle='--', alpha=0.7, 
                             label=f'Avg MAE: {avg_mae:.4f}')
            axes[0,1].legend()
            
            # Plot 3: Arduino Hardware Performance
            axes[1,0].plot(df_success['time_seconds'], df_success['arduino_inference_us'], 
                          'green', linewidth=1.5, alpha=0.7)
            axes[1,0].set_title('Arduino Inference Time', fontweight='bold')
            axes[1,0].set_xlabel('Time [seconds]')
            axes[1,0].set_ylabel('Inference Time [μs]')
            axes[1,0].grid(True, alpha=0.3)
            
            # Add average inference time
            avg_inference = df_success['arduino_inference_us'].mean()
            axes[1,0].axhline(y=avg_inference, color='red', linestyle='--', alpha=0.7,
                             label=f'Avg: {avg_inference:.0f}μs')
            axes[1,0].legend()
            
            # Plot 4: Success Rate and Statistics
            axes[1,1].axis('off')  # Turn off axis for text
            
            # Calculate statistics
            total_samples = len(df)
            successful_samples = len(df_success)
            success_rate = successful_samples / total_samples * 100
            avg_ram_free = df_success['arduino_ram_free'].mean()
            avg_cpu_load = df_success['arduino_cpu_load'].mean()
            avg_temp = df_success['arduino_temperature'].mean()
            
            # Statistics text
            stats_text = f"""
📊 COLLECTION STATISTICS
─────────────────────────
Total Samples:     {total_samples:,}
Successful:        {successful_samples:,} ({success_rate:.1f}%)
Time Range:        0s - {end_time_sec}s

🎯 PREDICTION ACCURACY
─────────────────────────
Average MAE:       {avg_mae:.4f}
Maximum MAE:       {max_mae:.4f}
Min MAE:           {df_success['mae_error'].min():.4f}

⚡ ARDUINO PERFORMANCE
─────────────────────────
Avg Inference:     {avg_inference:.0f} μs
Min Inference:     {df_success['arduino_inference_us'].min():.0f} μs
Max Inference:     {df_success['arduino_inference_us'].max():.0f} μs

💾 HARDWARE STATUS
─────────────────────────
Avg RAM Free:      {avg_ram_free:.0f} bytes
Avg CPU Load:      {avg_cpu_load:.1f}%
Avg Temperature:   {avg_temp:.1f}°C
"""
            
            axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_filename = f"arduino_soc_plot_0s_to_{end_time_sec}s_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_filename}")
            
            # Show plot
            plt.show()
            
            print(f"✅ Plot created successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to create plot: {e}")
            print(f"⚠️ Plotting failed, but CSV data is still available: {e}")
    
    def save_to_csv(self, end_time_sec):
        """Speichert gesammelte Daten in CSV-Datei und erstellt Plot"""
        try:
            # Erstelle DataFrame
            df = pd.DataFrame(self.collected_data)
            
            # Filename mit Zeitstempel
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"arduino_soc_data_0s_to_{end_time_sec}s_{timestamp}.csv"
            
            # Speichere CSV
            df.to_csv(filename, index=False)
            
            print(f"💾 Data saved to: {filename}")
            print(f"📊 CSV contains {len(df)} rows and {len(df.columns)} columns")
            print(f"📋 Columns: {', '.join(df.columns.tolist())}")
            
            # Zeige erste paar Zeilen
            print(f"\n📄 First 3 rows preview:")
            print(df.head(3).to_string(index=False))
            
            logger.info(f"✅ Data successfully saved to {filename}")
            
            # 📊 PLOT ERSTELLEN
            print(f"\n📊 Creating visualization...")
            self.plot_collected_data(df, end_time_sec)
            
        except Exception as e:
            logger.error(f"❌ Failed to save CSV: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Arduino SOC Data Collector - Sammelt Daten in CSV')
    parser.add_argument('--port', '-p', default=ARDUINO_PORT, help=f'Serial port (default: {ARDUINO_PORT})')
    parser.add_argument('--baudrate', '-b', type=int, default=BAUDRATE, help=f'Baud rate (default: {BAUDRATE})')
    parser.add_argument('--no-upload', action='store_true', help='Skip automatic upload')
    parser.add_argument('--sketch', '-s', default=ARDUINO_SKETCH_PATH, help='Arduino sketch path')
    parser.add_argument('--fqbn', '-f', default=ARDUINO_FQBN, help='Arduino FQBN')
    parser.add_argument('--end-time', '-t', type=int, default=DEFAULT_END_TIME, 
                       help=f'End time in seconds (default: {DEFAULT_END_TIME}s)')
    
    args = parser.parse_args()
    
    collector = ArduinoSOCDataCollector(
        args.port, args.baudrate, DATA_PATH, 
        args.sketch, args.fqbn
    )
    
    print(f"🎯 Will collect data from 0s to {args.end_time}s")
    print(f"🔌 Arduino Port: {args.port}")
    print(f"📤 Auto Upload: {'No' if args.no_upload else 'Yes'}")
    
    try:
        success = collector.collect_data(
            end_time_sec=args.end_time,
            auto_upload=not args.no_upload
        )
        
        if success:
            print("✅ Data collection completed successfully!")
            print("💾 CSV file ready for analysis and plotting")
        else:
            print("❌ Data collection failed!")
            
    except KeyboardInterrupt:
        print("⏹️ Collection stopped by user")
        if collector.collected_data:
            print("💾 Saving partial data...")
            collector.save_to_csv(args.end_time)
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
