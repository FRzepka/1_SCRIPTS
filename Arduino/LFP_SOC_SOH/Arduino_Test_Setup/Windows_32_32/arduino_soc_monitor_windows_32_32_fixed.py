"""
ARDUINO SOC MONITOR - WINDOW-BASED LSTM V2 (Fixed Version)
===========================================================

✨ FEATURES:
- Aligned with the structure of 'arduino_soc_monitor_sf_32_32.py'.
- Attempts to automatically compile and upload the Arduino sketch.
- Live SOC Monitoring with Ground Truth Vergleich.
- Hardware parameter display (RAM, Flash, CPU, Temp).

🚀 PERFORMANCE OPTIMIERT FÜR LIVE MONITORING 🚀
"""
# Standard library imports
import argparse
import logging
import pickle
import queue
import re
import subprocess
import threading
import time
import warnings
import os
from collections import deque
from pathlib import Path

# Third-party imports
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from sklearn.preprocessing import StandardScaler

# Warnings unterdrücken
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font", category=UserWarning)
warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator", category=UserWarning)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== EINSTELLUNGEN =====
ARDUINO_PORT = 'COM13'
BAUDRATE = 115200
DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
ARDUINO_SKETCH_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Windows_32_32\arduino_lstm_soc_windows_32_32\arduino_lstm_soc_windows_32_32.ino"
ARDUINO_CLI_PATH = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Arduino_CLI\arduino-cli.exe"
ARDUINO_FQBN = "arduino:renesas_uno:unor4wifi"
WINDOW_SIZE = 50  # CORRECTED FROM 5000
INPUT_SIZE = 4
PLOT_WINDOW_SIZE = 200
UPDATE_INTERVAL = 500  # Plot update in ms
PREDICTION_DELAY_MS = 100  # Min delay between sending windows to Arduino

class ArduinoUploader:
    """Handles automatic Arduino sketch compilation and upload (Adapted from sf_32_32)"""
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
                    logger.info(f"✅ Arduino CLI found at specified path: {result.stdout.strip()}")
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
                flash_match = re.search(r'Sketch uses (\d+) bytes.*Maximum is (\d+) bytes', output)
                if flash_match:
                    flash_used = int(flash_match.group(1))
                    logger.info(f"Flash used: {flash_used} bytes")
                
                # Upload
                logger.info(f"⬆️ Uploading to {self.port}...")
                cmd = [self.cli_path, 'upload', '--fqbn', self.fqbn, '-p', self.port, self.sketch_path]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    logger.info("✅ Upload successful")
                    return True, "Success"
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

class WindowBasedArduinoSOCMonitor:
    """Arduino SOC Monitor for Window-Based LSTM (aligned with sf_32_32)"""
    
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
        
        # Hardware Metrics - Similar to stateful version
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
        self.fig.suptitle(f'🚀 Arduino Window-Based LSTM SOC Monitor (Win={WINDOW_SIZE})', fontsize=16, fontweight='bold')
        
        # Hardware Stats Table
        self.setup_hardware_table()
        
        logger.info(f"Window-Based SOC Monitor initialized. WINDOW_SIZE={WINDOW_SIZE}")

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
            
            # Check for window-based requirements - need Q_m instead of Q_c
            required_cols = ['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_m', 'SOC_ZHU']
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
                'Q_m': 'q_m',  # Use Q_m for window-based
                'SOC_ZHU': 'soc'
            })
                
            self.ground_truth_data = df_clean[['voltage', 'current', 'soh', 'q_m', 'soc']].copy()
            logger.info(f"✅ Ground Truth data loaded: {len(self.ground_truth_data)} samples")
            
            if len(self.ground_truth_data) < WINDOW_SIZE:
                logger.error(f"Not enough data ({len(self.ground_truth_data)} points) for window size {WINDOW_SIZE}")
                return False
            
            self.setup_scaler()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Ground Truth data: {e}")
            return False

    def setup_scaler(self):
        """Setup scaler identical to training data"""
        try:
            scaler_path = Path(self.data_path).parent / "scaler_windows.pkl"
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Loaded existing scaler")
            else:
                feature_cols = ['voltage', 'current', 'soh', 'q_m']
                features = self.ground_truth_data[feature_cols].values
                
                self.scaler = StandardScaler()
                self.scaler.fit(features)
                
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info("✅ Created and saved new scaler")
                
        except Exception as e:
            logger.error(f"Scaler setup failed: {e}")
            self.scaler = StandardScaler()

    def scale_features(self, voltage, current, soh, q_m):
        """Scale features identical to training"""
        features = np.array([[voltage, current, soh, q_m]])
        if self.scaler:
            return self.scaler.transform(features)[0]
        return features[0]

    def get_next_data_window(self):
        """Get next data window from ground truth"""
        if self.ground_truth_data is None or len(self.ground_truth_data) == 0:
            return None
            
        try:
            if self.data_index + WINDOW_SIZE >= len(self.ground_truth_data):
                self.data_index = 0
                logger.info("🔄 Cycling through dataset - restarted from beginning")
            
            # Get window of data
            window_data = self.ground_truth_data.iloc[self.data_index : self.data_index + WINDOW_SIZE]
            self.data_index += 1  # Sliding window
            
            # Scale the window features
            features_window = []
            for _, row in window_data.iterrows():
                scaled_features = self.scale_features(row['voltage'], row['current'], row['soh'], row['q_m'])
                features_window.extend(scaled_features)
            
            # Ground truth SOC at end of window
            soc_gt = window_data.iloc[-1]['soc']
            voltage_end = window_data.iloc[-1]['voltage']
            
            return {
                'window_features': features_window,
                'soc_true': float(soc_gt),
                'voltage_end': float(voltage_end)
            }
            
        except Exception as e:
            logger.error(f"Error getting next data window: {e}")
            return None

    def connect_arduino(self):
        """Arduino connection with improved timeout"""
        try:
            logger.info(f"Connecting to Arduino on {self.port}...")
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(3)  # Arduino connection stabilization
            
            # Test connection with PING command
            self.arduino.write(b'PING\n')
            time.sleep(0.5)
            response = self.arduino.readline().decode().strip()
            if response and ("PONG" in response or "READY" in response or len(response) > 5):
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
        """Get initial hardware statistics"""
        try:
            logger.info("📊 Getting initial hardware statistics...")
            self.arduino.write(b'STATS\n')
            
            # Read Arduino responses for hardware stats
            stats_response = None
            max_attempts = 10
            attempt = 0
            
            while attempt < max_attempts:
                time.sleep(0.3)
                if self.arduino.in_waiting > 0:
                    response = self.arduino.readline().decode().strip()
                    if "STATS:" in response:
                        stats_response = response
                        break
                attempt += 1
            
            if stats_response and "STATS:" in stats_response:
                # Parse: "STATS:ram_total,ram_free,flash_total,flash_used"
                data = stats_response.replace("STATS:", "").split(",")
                if len(data) >= 4:
                    ram_total = int(data[0])
                    ram_free = int(data[1])
                    flash_total = int(data[2])
                    flash_used = int(data[3])
                    
                    ram_used = ram_total - ram_free
                    flash_free = flash_total - flash_used
                    
                    # Store hardware values
                    self.hardware_stats['ram_total_bytes'].append(ram_total)
                    self.hardware_stats['ram_free_bytes'].append(ram_free)
                    self.hardware_stats['ram_used_bytes'].append(ram_used)
                    self.hardware_stats['flash_total_bytes'].append(flash_total)
                    self.hardware_stats['flash_used_bytes'].append(flash_used)
                    self.hardware_stats['flash_free_bytes'].append(flash_free)
                    
                    logger.info(f"📊 Hardware Stats - RAM: {ram_used}/{ram_total} bytes, Flash: {flash_used}/{flash_total} bytes")
                else:
                    logger.warning("Invalid STATS response format")
            else:
                logger.warning("No valid STATS response received")
                    
        except Exception as e:
            logger.warning(f"Initial hardware stats failed: {e}")

    def predict_with_arduino(self, window_features):
        """Send window to Arduino for LSTM prediction"""
        if not self.arduino_connected:
            return None, None
            
        try:
            # Create window data string
            window_str = ','.join([f"{f:.6f}" for f in window_features])
            command = f"WINDOW:{window_str}\n"
            
            # Send to Arduino
            self.arduino.write(command.encode())
            
            # Read response with timeout
            response = self.arduino.readline().decode().strip()
            
            if response and response.startswith("RESULT:"):
                # Parse: "RESULT:SOC,inference_time_us,ram_free,cpu_load,temp"
                data = response.replace("RESULT:", "").split(",")
                if len(data) >= 5:
                    soc_pred = float(data[0])
                    inference_time = int(data[1])
                    ram_free = int(data[2])
                    cpu_load = float(data[3])
                    temperature = float(data[4])
                    
                    hardware_metrics = {
                        'inference_time_us': inference_time,
                        'ram_free_bytes': ram_free,
                        'cpu_load_percent': cpu_load,
                        'temperature_celsius': temperature
                    }
                    
                    self.successful_predictions += 1
                    return soc_pred, hardware_metrics
            
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
                data_sample = self.get_next_data_window()
                if data_sample is None:
                    time.sleep(0.1)
                    continue
                
                soc_pred, hardware_metrics = self.predict_with_arduino(data_sample['window_features'])
                
                if soc_pred is not None:
                    current_time = time.time()
                    
                    # Store results
                    prediction_data = {
                        'timestamp': current_time,
                        'voltage': data_sample['voltage_end'],
                        'soc_pred': soc_pred,
                        'soc_true': data_sample['soc_true'],
                        'mae': abs(soc_pred - data_sample['soc_true']),
                        'hardware': hardware_metrics
                    }
                    
                    try:
                        self.arduino_data_queue.put_nowait(prediction_data)
                    except queue.Full:
                        # Remove oldest if queue is full
                        try:
                            self.arduino_data_queue.get_nowait()
                            self.arduino_data_queue.put_nowait(prediction_data)
                        except queue.Empty:
                            pass
                
                time.sleep(PREDICTION_DELAY_MS / 1000.0)
                
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
                current_ram_total = list(self.hardware_stats['ram_total_bytes'])[-1]
                ram_usage_percent = ((current_ram_total - current_ram_free) / current_ram_total * 100)
            else:
                current_ram_free = current_ram_total = ram_usage_percent = 0
            
            # Flash memory stats
            if len(self.hardware_stats['flash_total_bytes']) > 0:
                flash_total_kb = list(self.hardware_stats['flash_total_bytes'])[-1] / 1024
                flash_used_kb = list(self.hardware_stats['flash_used_bytes'])[-1] / 1024
                flash_usage_percent = (list(self.hardware_stats['flash_used_bytes'])[-1] / list(self.hardware_stats['flash_total_bytes'])[-1] * 100)
            else:
                flash_total_kb = flash_used_kb = flash_usage_percent = 0
            
            # Current MAE
            current_mae = list(self.mae_errors)[-1] if len(self.mae_errors) > 0 else 0
            avg_mae = np.mean(list(self.mae_errors)) if len(self.mae_errors) > 0 else 0
            
            # Success rate
            success_rate = (self.successful_predictions / self.prediction_count * 100) if self.prediction_count > 0 else 0
            
            # Format table
            table_text = f"""
┌─────────────────────────────────────────────┐
│         🚀 ARDUINO WINDOW LSTM MONITOR      │
│           (Window Size: {WINDOW_SIZE:3d})             │
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
│   Used:         {current_ram_total-current_ram_free:5d} B ({ram_usage_percent:4.1f}%)        │
│   Total:        {current_ram_total:5d} B                │
│                                             │
│ 📀 FLASH MEMORY                             │
│   Used:         {flash_used_kb:5.1f} KB ({flash_usage_percent:4.1f}%)       │
│   Total:        {flash_total_kb:5.0f} KB              │
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
                try:
                    data = self.arduino_data_queue.get_nowait()
                    
                    self.timestamps.append(data['timestamp'])
                    self.voltage_values.append(data['voltage'])
                    self.soc_predictions.append(data['soc_pred'])
                    self.soc_ground_truth.append(data['soc_true'])
                    self.mae_errors.append(data['mae'])
                    
                    # Update hardware stats
                    if data['hardware']:
                        for key, value in data['hardware'].items():
                            if key in self.hardware_stats:
                                self.hardware_stats[key].append(value)
                    
                except queue.Empty:
                    break
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
        
        # Plot 1: SOC Comparison
        self.axes[0].plot(times, soc_truth, 'b-', linewidth=2, label='Ground Truth SOC', alpha=0.8)
        self.axes[0].plot(times, soc_preds, 'r--', linewidth=2, label=f'Arduino Window LSTM (W={WINDOW_SIZE})', alpha=0.8)
        self.axes[0].plot(times, voltages/4, 'g:', linewidth=1, label='Voltage/4 [V]', alpha=0.6)
        self.axes[0].set_title('🎯 SOC: Ground Truth vs Arduino Window LSTM', fontweight='bold', fontsize=14)
        self.axes[0].set_ylabel('State of Charge', fontweight='bold')
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].legend()
        self.axes[0].set_ylim(0, 1)
        
        # Plot 2: MAE
        mae_values = np.array(self.mae_errors)
        self.axes[1].plot(times, mae_values, 'm-', linewidth=2, label='MAE', alpha=0.8)
        self.axes[1].set_title('📊 Mean Absolute Error', fontweight='bold', fontsize=12)
        self.axes[1].set_ylabel('MAE', fontweight='bold')
        self.axes[1].set_xlabel('Time [s]', fontweight='bold')
        self.axes[1].grid(True, alpha=0.3)
        self.axes[1].legend()
        
        # Update hardware statistics table
        self.update_hardware_table()
        
        plt.tight_layout()

    def start_monitoring(self, auto_upload=True):
        """Start monitoring with optional auto-upload"""
        print("🚀 Arduino Window-Based LSTM SOC Monitor")
        print("="*50)
        
        if auto_upload:
            print("⬆️ Uploading Arduino sketch...")
            if not self.upload_arduino_sketch():
                cont = input("Upload failed. Continue with current sketch? (y/n): ")
                if cont.lower() != 'y':
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
    parser = argparse.ArgumentParser(description='Arduino Window-Based LSTM SOC Monitor')
    parser.add_argument('--port', '-p', default=ARDUINO_PORT, help=f'Serial port (default: {ARDUINO_PORT})')
    parser.add_argument('--baudrate', '-b', type=int, default=BAUDRATE, help=f'Baud rate (default: {BAUDRATE})')
    parser.add_argument('--no-upload', action='store_true', help='Skip automatic upload')
    parser.add_argument('--sketch', '-s', default=ARDUINO_SKETCH_PATH, help='Arduino sketch path')
    parser.add_argument('--fqbn', '-f', default=ARDUINO_FQBN, help='Arduino FQBN')
    
    args = parser.parse_args()
    
    monitor = WindowBasedArduinoSOCMonitor(
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
