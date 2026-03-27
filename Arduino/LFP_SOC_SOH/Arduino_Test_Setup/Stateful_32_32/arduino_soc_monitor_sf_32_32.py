"""
ARDUINO SOC MONITOR V2 - Ohne geschätzte Flash-Werte
===================================================

✨ FEATURES:
- Automatisches Kompilieren und Hochladen des Arduino Sketches
- ECHTE Flash-Werte vom Arduino-Compiler (90.2 KB)
- Keine geschätzten oder falschen Werte mehr
- Live SOC Monitoring mit Ground Truth Vergleich

🚀 PERFORMANCE OPTIMIERT FÜR LIVE MONITORING 🚀
"""

import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import subprocess
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import threading
import queue
from collections import deque
import logging
import warnings
import argparse

# Warnings unterdrücken
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font", category=UserWarning)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== EINSTELLUNGEN =====
ARDUINO_PORT = 'COM13'
BAUDRATE = 115200
DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
ARDUINO_SKETCH_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32\code_weights\arduino_lstm_soc_full32_with_monitoring\arduino_lstm_soc_full32_with_monitoring.ino"
ARDUINO_CLI_PATH = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Arduino_CLI\arduino-cli.exe"
ARDUINO_FQBN = "arduino:renesas_uno:unor4wifi"  # Arduino UNO R4 WiFi

PLOT_WINDOW_SIZE = 2000  # Datenpunkte im Plot
UPDATE_INTERVAL = 300   # Plot Update alle 300ms
PREDICTION_DELAY = 30   # 30ms zwischen Predictions

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

class OptimizedArduinoSOCMonitorV2:
    """Arduino SOC Monitor V2 - Ohne geschätzte Flash-Werte"""
    
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
        
        # Data Queues
        self.arduino_data_queue = queue.Queue(maxsize=50)
        self.prediction_running = False
        
        # Performance-optimierte Data Storage
        self.timestamps = deque(maxlen=PLOT_WINDOW_SIZE)
        self.voltage_values = deque(maxlen=PLOT_WINDOW_SIZE)
        self.soc_predictions = deque(maxlen=PLOT_WINDOW_SIZE)
        self.soc_ground_truth = deque(maxlen=PLOT_WINDOW_SIZE)
        self.mae_errors = deque(maxlen=PLOT_WINDOW_SIZE)
        
        # Hardware Metrics - Table only
        self.hardware_stats = {
            'inference_time_us': deque(maxlen=100),
            'ram_free_bytes': deque(maxlen=100),
            'ram_used_bytes': deque(maxlen=100),
            'ram_total_bytes': deque(maxlen=100),
            'cpu_load_percent': deque(maxlen=100),
            'temperature_celsius': deque(maxlen=100),
            'flash_used_bytes': deque(maxlen=100),
            'flash_total_bytes': deque(maxlen=100),
            'flash_free_bytes': deque(maxlen=100)
        }
        
        # Ground Truth Data
        self.ground_truth_data = None
        self.scaler = None
        self.data_index = 0
        
        # Performance Tracking
        self.prediction_count = 0
        self.successful_predictions = 0
        self.communication_errors = 0
        self.start_time = time.time()
        
        # UI Layout
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('🚀 Arduino SOC Monitor V2 (Reale Flash-Werte)', fontsize=16, fontweight='bold')
        
        # Hardware Stats Table
        self.setup_hardware_table()
        
        logger.info("Arduino SOC Monitor V2 initialized")
    
    def setup_hardware_table(self):
        """Setup hardware statistics table"""
        self.stats_text = self.fig.text(0.02, 0.02, "", fontsize=9, fontfamily='monospace',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
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
    
    def load_ground_truth_data(self):
        """Lädt Ground Truth Data von MGFarm"""
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
        """Setup scaler identical to training data"""
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
        """Scale features identical to training"""
        features = np.array([[voltage, current, soh, q_c]])
        if self.scaler:
            return self.scaler.transform(features)[0]
        return features[0]
    
    def get_next_data_sample(self):
        """Get next data sample from ground truth cycling through dataset"""
        if self.ground_truth_data is None or len(self.ground_truth_data) == 0:
            return None
            
        try:
            if self.data_index >= len(self.ground_truth_data):
                self.data_index = 0
                logger.info("🔄 Cycling through dataset - restarted from beginning")
            
            sample = self.ground_truth_data.iloc[self.data_index]
            self.data_index += 1
            
            return {
                'voltage': float(sample['voltage']),
                'current': float(sample['current']),
                'soh': float(sample['soh']),
                'q_c': float(sample['q_c']),
                'soc_true': float(sample['soc'])
            }
            
        except Exception as e:
            logger.error(f"Error getting next data sample: {e}")
            return None
    
    def connect_arduino(self):
        """Arduino connection with improved timeout"""
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
                self.get_initial_hardware_stats()
                return True
            else:
                logger.error(f"Arduino connection test failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Arduino connection failed: {e}")
            return False

    def get_initial_hardware_stats(self):
        """Get initial hardware statistics - NO ESTIMATES, only NaN if no Arduino data!"""
        try:
            logger.info("📊 Getting initial hardware statistics...")
            self.arduino.write(b'BENCHMARK\n')
            
            # Read Arduino responses until we get the BENCHMARK data
            benchmark_response = None
            max_attempts = 15
            attempt = 0
            
            while attempt < max_attempts:
                time.sleep(0.3)
                if self.arduino.in_waiting > 0:
                    response = self.arduino.readline().decode().strip()
                    logger.info(f"🔍 Arduino response {attempt + 1}: '{response}'")
                    
                    if "BENCHMARK:" in response:
                        benchmark_response = response
                        break
                    elif "Starting benchmark" in response:
                        logger.info("📊 Arduino started benchmark, waiting for results...")
                        time.sleep(1.5)
                    
                attempt += 1
            
            if benchmark_response and "BENCHMARK:" in benchmark_response:
                # Parse: "BENCHMARK:avg_us,min_us,max_us,total_ram,free_ram,total_flash,free_flash,cpu_mhz"
                data = benchmark_response.replace("BENCHMARK:", "").split(",")
                logger.info(f"🔍 BENCHMARK data parts: {data} (count: {len(data)})")
                
                if len(data) >= 8:
                    # Arduino sends Flash memory data - use it directly
                    total_ram = int(data[3])
                    free_ram = int(data[4])
                    flash_total = int(data[5])
                    flash_free = int(data[6])
                    cpu_mhz = float(data[7])
                    flash_used = flash_total - flash_free
                    
                    logger.info(f"📊 Hardware Info (from Arduino): RAM={total_ram}B ({total_ram/1024:.1f}KB), Flash={flash_total/1024:.0f}KB (used: {flash_used/1024:.0f}KB), CPU={cpu_mhz}MHz")
                else:
                    # NO ESTIMATES! Only NaN if Arduino doesn't send complete data
                    total_ram = int(data[3]) if len(data) >= 4 else float('nan')
                    cpu_mhz = float(data[5]) if len(data) >= 6 else float('nan')
                    
                    # NO FLASH ESTIMATES! Only NaN if not measured
                    flash_total = float('nan')  # No Arduino data = NaN
                    flash_used = float('nan')   # No Arduino data = NaN  
                    flash_free = float('nan')   # No Arduino data = NaN
                    logger.info(f"📊 Hardware Info (INCOMPLETE - showing NaN): RAM={total_ram}B, Flash=NaN KB, CPU={cpu_mhz}MHz")
                
                # Store hardware values
                self.hardware_stats['ram_total_bytes'].append(total_ram)
                self.hardware_stats['flash_total_bytes'].append(flash_total)
                self.hardware_stats['flash_used_bytes'].append(flash_used)
                self.hardware_stats['flash_free_bytes'].append(flash_free)
                    
            else:
                logger.warning(f"No valid BENCHMARK response received after {max_attempts} attempts")
                # NO FALLBACK VALUES! Only NaN if Arduino doesn't respond
                self._set_nan_flash_values()
                    
        except Exception as e:
            logger.warning(f"Initial hardware stats failed: {e}")
            # NO FALLBACK VALUES! Only NaN if communication fails
            self._set_nan_flash_values()
            
    def _set_nan_flash_values(self):
        """Set NaN Flash memory values - NO ESTIMATES ALLOWED!"""
        flash_total = float('nan')    # NO Arduino data = NaN
        flash_used = float('nan')     # NO Arduino data = NaN  
        flash_free = float('nan')     # NO Arduino data = NaN
        
        self.hardware_stats['flash_total_bytes'].append(flash_total)
        self.hardware_stats['flash_used_bytes'].append(flash_used)
        self.hardware_stats['flash_free_bytes'].append(flash_free)
        
        logger.info("📊 NO ARDUINO FLASH DATA - showing NaN (NO ESTIMATES!)")
    
    def predict_with_arduino(self, voltage, current, soh, q_c):
        """Echte Arduino LSTM Prediction"""
        if not self.arduino_connected:
            return None, None
            
        try:
            # Scale features
            scaled_features = self.scale_features(voltage, current, soh, q_c)
            
            # Create command
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
                    
                    # Store hardware metrics
                    self.hardware_stats['inference_time_us'].append(inference_us)
                    self.hardware_stats['ram_free_bytes'].append(ram_free)
                    self.hardware_stats['ram_used_bytes'].append(ram_used)
                    self.hardware_stats['ram_total_bytes'].append(ram_free + ram_used)
                    self.hardware_stats['cpu_load_percent'].append(cpu_load)
                    self.hardware_stats['temperature_celsius'].append(temperature)
                    
                    self.successful_predictions += 1
                    return soc_pred, {'inference_us': inference_us, 'ram_free': ram_free, 
                                     'ram_used': ram_used, 'cpu_load': cpu_load, 'temp': temperature}
            
            self.communication_errors += 1
            return None, None
            
        except Exception as e:
            logger.warning(f"Arduino prediction failed: {e}")
            self.communication_errors += 1
            return None, None
    
    def background_prediction_loop(self):
        """Background thread für kontinuierliche Predictions"""
        logger.info("🔄 Background prediction loop started")
        
        while self.prediction_running:
            try:
                data_sample = self.get_next_data_sample()
                if data_sample is None:
                    time.sleep(0.1)
                    continue
                
                soc_pred, hardware_metrics = self.predict_with_arduino(
                    data_sample['voltage'], data_sample['current'], 
                    data_sample['soh'], data_sample['q_c']
                )
                
                if soc_pred is not None:
                    mae_error = abs(soc_pred - data_sample['soc_true'])
                    
                    try:
                        plot_data = {
                            'timestamp': time.time(),
                            'voltage': data_sample['voltage'],
                            'soc_pred': soc_pred,
                            'soc_true': data_sample['soc_true'],
                            'mae_error': mae_error,
                            'hardware': hardware_metrics
                        }
                        self.arduino_data_queue.put(plot_data, timeout=0.1)
                        self.prediction_count += 1
                        
                    except queue.Full:
                        pass
                
                time.sleep(PREDICTION_DELAY / 1000.0)
                
            except Exception as e:
                logger.error(f"Background prediction error: {e}")
                time.sleep(0.5)
        
        logger.info("Background prediction loop stopped")
    
    def update_hardware_table(self):
        """Update hardware statistics table"""
        try:
            current_time = time.time()
            runtime = current_time - self.start_time
            
            # Calculate statistics
            if len(self.hardware_stats['inference_time_us']) > 0:
                avg_inference = np.mean(list(self.hardware_stats['inference_time_us']))
                throughput = self.successful_predictions / runtime if runtime > 0 else 0
            else:
                avg_inference = throughput = 0
            
            if len(self.hardware_stats['ram_free_bytes']) > 0:
                current_ram_free = list(self.hardware_stats['ram_free_bytes'])[-1]
                current_ram_used = list(self.hardware_stats['ram_used_bytes'])[-1]
                current_ram_total = list(self.hardware_stats['ram_total_bytes'])[-1]
                ram_usage_percent = (current_ram_used / current_ram_total * 100) if current_ram_total > 0 else 0
            else:
                current_ram_free = current_ram_used = current_ram_total = ram_usage_percent = 0
              # Flash memory stats - handle NaN values
            if len(self.hardware_stats['flash_total_bytes']) > 0:
                flash_total = list(self.hardware_stats['flash_total_bytes'])[-1]
                flash_used = list(self.hardware_stats['flash_used_bytes'])[-1]
                flash_free = list(self.hardware_stats['flash_free_bytes'])[-1]
                
                # Check for NaN values
                if not (np.isnan(flash_total) or np.isnan(flash_used)):
                    flash_usage_percent = (flash_used / flash_total * 100) if flash_total > 0 else 0
                    flash_total_kb = flash_total/1024
                    flash_used_kb = flash_used/1024
                    flash_free_kb = flash_free/1024
                else:
                    flash_usage_percent = float('nan')
                    flash_total_kb = flash_used_kb = flash_free_kb = float('nan')
            else:
                flash_total_kb = flash_used_kb = flash_free_kb = flash_usage_percent = float('nan')
            
            # Current MAE
            current_mae = list(self.mae_errors)[-1] if len(self.mae_errors) > 0 else 0
            avg_mae = np.mean(list(self.mae_errors)) if len(self.mae_errors) > 0 else 0
            
            # Success rate
            success_rate = (self.successful_predictions / self.prediction_count * 100) if self.prediction_count > 0 else 0
            
            # Format table
            table_text = f"""
┌─────────────────────────────────────────────┐
│         🚀 ARDUINO SOC MONITOR V2           │
│           (REALE FLASH-WERTE)               │
├─────────────────────────────────────────────┤
│ 📊 LIVE PERFORMANCE                         │
│   Tests:        {self.prediction_count:6d}                │
│   Success:      {success_rate:5.1f}%                │
│   Runtime:      {runtime:5.0f}s                │
│   Throughput:   {throughput:5.1f} Hz              │
│                                             │
│ ⚡ ARDUINO INFERENCE                         │
│   Avg Time:     {avg_inference:5.0f} μs              │
│                                             │
│ 💾 RAM USAGE                                │
│   Free:         {current_ram_free:5d} B ({100-ram_usage_percent:4.1f}%)        │
│   Used:         {current_ram_used:5d} B ({ram_usage_percent:4.1f}%)        │
│   Total:        {current_ram_total:5d} B                │
│                                             │
│ 📀 FLASH MEMORY (NUR GEMESSENE WERTE!)     │
│   Free:         {flash_free_kb:5.0f} KB ({100-flash_usage_percent:4.1f}%) {'[NaN-kein Arduino Wert]' if np.isnan(flash_free_kb) else ''}       │
│   Used:         {flash_used_kb:5.1f} KB ({flash_usage_percent:4.1f}%) {'[NaN-kein Arduino Wert]' if np.isnan(flash_used_kb) else ''}       │
│   Total:        {flash_total_kb:5.0f} KB {'[NaN-kein Arduino Wert]' if np.isnan(flash_total_kb) else ''}              │
│                                             │
│ 🔋 SOC ACCURACY                             │
│   Current MAE:  {current_mae:6.4f}              │
│   Average MAE:  {avg_mae:6.4f}              │
└─────────────────────────────────────────────┘
"""
            self.stats_text.set_text(table_text)
            
        except Exception as e:
            logger.error(f"Hardware table update failed: {e}")
    
    def update_plots(self, frame):
        """Update plots for Performance"""
        try:
            while not self.arduino_data_queue.empty():
                data = self.arduino_data_queue.get_nowait()
                
                self.timestamps.append(data['timestamp'])
                self.voltage_values.append(data['voltage'])
                self.soc_predictions.append(data['soc_pred'])
                self.soc_ground_truth.append(data['soc_true'])
                self.mae_errors.append(data['mae_error'])
                
        except queue.Empty:
            pass
        
        if len(self.timestamps) < 2:
            return
        
        times = np.array(self.timestamps) - self.timestamps[0]
        voltages = np.array(self.voltage_values)
        soc_preds = np.array(self.soc_predictions)
        soc_truth = np.array(self.soc_ground_truth)
        
        # Clear plots
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Plot 1: Voltage
        self.axes[0].plot(times, voltages, 'g-', linewidth=2, label='Voltage [V]', alpha=0.8)
        self.axes[0].set_title('⚡ Battery Voltage', fontweight='bold', fontsize=14)
        self.axes[0].set_ylabel('Voltage [V]', fontweight='bold')
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].legend()
        
        # Plot 2: SOC Prediction vs Ground Truth
        self.axes[1].plot(times, soc_truth, 'b-', linewidth=2, label='Ground Truth SOC', alpha=0.8)
        self.axes[1].plot(times, soc_preds, 'r--', linewidth=2, label='Arduino LSTM Prediction', alpha=0.8)
        self.axes[1].set_title('🎯 SOC: Ground Truth vs Arduino LSTM', fontweight='bold', fontsize=14)
        self.axes[1].set_ylabel('State of Charge', fontweight='bold')
        self.axes[1].set_xlabel('Time [s]', fontweight='bold')
        self.axes[1].grid(True, alpha=0.3)
        self.axes[1].legend()
        self.axes[1].set_ylim(0, 1)
        
        # Update hardware statistics table
        self.update_hardware_table()
        
        plt.tight_layout()
    
    def start_monitoring(self, auto_upload=True):
        """Start monitoring with optional auto-upload"""
        print("🚀 Arduino SOC Monitor V2 (REALE Flash-Werte)")
        print("="*50)
        
        if auto_upload:
            print("⬆️ Uploading Arduino sketch...")
            if not self.upload_arduino_sketch():
                print("❌ Upload failed. Continue with existing sketch? (y/n): ", end="")
                if input().lower() != 'y':
                    return
        
        print("📊 Loading ground truth data...")
        if not self.load_ground_truth_data():
            print("❌ Failed to load data")
            return
        
        print("🔌 Connecting to Arduino...")
        if not self.connect_arduino():
            print("❌ Failed to connect to Arduino")
            return
        
        # Start prediction thread
        self.prediction_running = True
        prediction_thread = threading.Thread(target=self.background_prediction_loop)
        prediction_thread.daemon = True
        prediction_thread.start()
        
        # Start animation
        ani = animation.FuncAnimation(self.fig, self.update_plots, interval=UPDATE_INTERVAL, blit=False)
        
        print("✅ Monitoring started! Close plot window to stop.")
        plt.show()
        
        # Cleanup
        self.prediction_running = False
        if self.arduino:
            self.arduino.close()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.prediction_running = False
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
        logger.info("🛑 Monitoring stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Arduino SOC Monitor V2 mit REALEN Flash-Werten')
    parser.add_argument('--port', '-p', default=ARDUINO_PORT, help=f'Serial port (default: {ARDUINO_PORT})')
    parser.add_argument('--baudrate', '-b', type=int, default=BAUDRATE, help=f'Baud rate (default: {BAUDRATE})')
    parser.add_argument('--no-upload', action='store_true', help='Skip automatic upload')
    parser.add_argument('--sketch', '-s', default=ARDUINO_SKETCH_PATH, help='Arduino sketch path')
    parser.add_argument('--fqbn', '-f', default=ARDUINO_FQBN, help='Arduino FQBN')
    
    args = parser.parse_args()
    
    monitor = OptimizedArduinoSOCMonitorV2(
        args.port, args.baudrate, DATA_PATH, 
        args.sketch, args.fqbn
    )
    
    try:
        monitor.start_monitoring(auto_upload=not args.no_upload)
    except KeyboardInterrupt:
        print("⏹️ Program stopped by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()