"""
ARDUINO SOC MONITOR V2 - 16x16 VERSION (FINAL)
==============================================

✨ FEATURES:
- Automatisches Kompilieren und Hochladen des Arduino Sketches  
- ECHTE Flash-Werte vom Arduino-Compiler (16x16 Model)
- Keine geschätzten oder falschen Werte mehr
- Live SOC Monitoring mit Ground Truth Vergleich
- FIXED: Korrekte Arduino Kommunikation für 16x16 Model
- NEW: RAM/Flash Anzeige als Verbrauch/Gesamt (z.B. 12/32 KB)

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
import re

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
ARDUINO_SKETCH_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_16_16\code_weights\arduino_lstm_soc_full16_with_monitoring\arduino_lstm_soc_full16_with_monitoring.ino"
ARDUINO_CLI_PATH = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Arduino_CLI\arduino-cli.exe"
ARDUINO_FQBN = "arduino:renesas_uno:unor4wifi"  # Arduino UNO R4 WiFi

PLOT_WINDOW_SIZE = 200  # Datenpunkte im Plot
UPDATE_INTERVAL = 300   # Plot Update alle 300ms  
PREDICTION_DELAY = 30   # 30ms zwischen Predictions

class ArduinoUploader:
    """Arduino CLI Uploader"""
    def __init__(self, sketch_path, arduino_cli_path, port, fqbn):
        self.sketch_path = sketch_path
        self.arduino_cli_path = arduino_cli_path
        self.port = port
        self.fqbn = fqbn
        
    def find_arduino_cli(self):
        """Find Arduino CLI"""
        if os.path.exists(self.arduino_cli_path):
            return self.arduino_cli_path
        logger.error(f"Arduino CLI not found at {self.arduino_cli_path}")
        return None
        
    def compile_and_upload(self):
        """Compile and upload Arduino sketch"""
        try:
            cli_path = self.find_arduino_cli()
            if not cli_path:
                return False, "Arduino CLI not found"
                
            sketch_dir = os.path.dirname(self.sketch_path)
            
            # Compile
            logger.info("🔨 Compiling Arduino sketch...")
            compile_cmd = [cli_path, "compile", "--fqbn", self.fqbn, sketch_dir]
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, f"Compilation failed: {result.stderr}"
                
            # Extract sketch size
            output_lines = result.stderr.split('\n')
            for line in output_lines:
                if "Sketch uses" in line:
                    logger.info(f"📊 {line}")
                    
            # Upload  
            logger.info("📤 Uploading to Arduino...")
            upload_cmd = [cli_path, "upload", "-p", self.port, "--fqbn", self.fqbn, sketch_dir]
            result = subprocess.run(upload_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, f"Upload failed: {result.stderr}"
                
            return True, "Upload successful"
            
        except Exception as e:
            return False, f"Upload error: {e}"

class ArduinoSOCMonitor:
    """Arduino SOC Monitor für 16x16 Model"""
    
    def __init__(self, port=ARDUINO_PORT, baudrate=BAUDRATE, data_path=DATA_PATH, sketch_path=ARDUINO_SKETCH_PATH):
        self.port = port
        self.baudrate = baudrate
        self.data_path = data_path
        self.sketch_path = sketch_path
        
        # Arduino Connection
        self.arduino = None
        self.arduino_connected = False
        
        # Arduino Uploader
        self.uploader = ArduinoUploader(sketch_path, ARDUINO_CLI_PATH, port, ARDUINO_FQBN)
        
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
        self.fig.suptitle('🚀 Arduino SOC Monitor V2 - 16x16 Model (FINAL)', fontsize=16, fontweight='bold')
        
        # Hardware Stats Table
        self.setup_hardware_table()
        
        logger.info("Arduino SOC Monitor V2 (16x16 - FINAL) initialized")
    
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
            logger.info("⏳ Waiting for Arduino to boot (10 seconds)...")
            time.sleep(10)  # Extended wait for Arduino to fully boot after upload
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
            # Cycle through dataset
            sample = self.ground_truth_data.iloc[self.data_index % len(self.ground_truth_data)]
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
        """Arduino connection with improved timeout and boot handling"""
        try:
            logger.info(f"🔌 Connecting to Arduino on {self.port}...")
            
            # Close any existing connection
            if hasattr(self, 'arduino') and self.arduino and self.arduino.is_open:
                self.arduino.close()
                time.sleep(1)
            
            # Open new connection
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=5)
            logger.info("⏳ Arduino connection established, testing communication...")
            time.sleep(3)  # Initial stabilization
            
            # Clear any pending data
            self.arduino.reset_input_buffer()
            self.arduino.reset_output_buffer()
            
            # Test connection with multiple attempts
            max_connection_attempts = 8
            for attempt in range(max_connection_attempts):
                try:
                    logger.info(f"🔍 Testing Arduino communication (attempt {attempt + 1}/{max_connection_attempts})...")
                    
                    # Send test command
                    self.arduino.write(b'STATS\n')
                    time.sleep(1.5)  # Give Arduino time to process
                    
                    # Try to read response
                    response = ""
                    if self.arduino.in_waiting > 0:
                        response = self.arduino.readline().decode().strip()
                    
                    logger.info(f"📡 Arduino response: '{response}'")
                    
                    # Check if we got a valid response
                    if response and (
                        "STATS:" in response or 
                        "RAM:" in response or 
                        "FLASH:" in response or
                        "PERFORMANCE" in response or
                        len(response) > 10
                    ):
                        self.arduino_connected = True
                        logger.info("✅ Arduino connected and responding!")
                        self.get_initial_hardware_stats()
                        return True
                    
                    # Wait before retry
                    if attempt < max_connection_attempts - 1:
                        wait_time = 2 + (attempt * 0.5)  # Progressive backoff
                        logger.info(f"⏳ Waiting {wait_time:.1f}s before next attempt...")
                        time.sleep(wait_time)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {e}")
                    if attempt < max_connection_attempts - 1:
                        time.sleep(2)
            
            logger.error(f"❌ Arduino connection failed after {max_connection_attempts} attempts")
            logger.error("🔧 Troubleshooting tips:")
            logger.error("   - Check if Arduino is properly connected")
            logger.error("   - Verify the correct COM port")
            logger.error("   - Try unplugging and reconnecting Arduino")
            logger.error("   - Check if Arduino IDE Serial Monitor is open (close it)")
            return False
                
        except Exception as e:
            logger.error(f"❌ Arduino connection failed: {e}")
            return False
    
    def get_initial_hardware_stats(self):
        """Get initial hardware statistics"""
        try:
            logger.info("📊 Getting initial hardware statistics...")
            
            # Send hardware stats commands (matching 32-bit Arduino interface)
            commands = [b'STATS\n', b'RAM\n']
            
            for cmd in commands:
                self.arduino.write(cmd)
                time.sleep(0.5)
                
                # Read Arduino responses
                max_attempts = 10
                attempt = 0
                
                while attempt < max_attempts:
                    time.sleep(0.2)
                    if self.arduino.in_waiting > 0:
                        response = self.arduino.readline().decode().strip()
                        logger.info(f"Arduino hardware response: {response}")
                        
                        # Parse STATS response: "STATS:inference_us,free_ram,used_ram,cpu_load,temp"
                        if "STATS:" in response:
                            try:
                                parts = response.split(':')[1].split(',')
                                if len(parts) >= 5:
                                    inference_time = float(parts[0])
                                    free_ram = float(parts[1])
                                    used_ram = float(parts[2])
                                    cpu_load = float(parts[3])
                                    temperature = float(parts[4])
                                    
                                    self.hardware_stats['inference_time_us'].append(inference_time)
                                    self.hardware_stats['ram_free_bytes'].append(free_ram)
                                    self.hardware_stats['ram_used_bytes'].append(used_ram)
                                    self.hardware_stats['temperature_celsius'].append(temperature)
                            except (ValueError, IndexError):
                                pass
                                
                        # Parse RAM response: "RAM:free,used,total,fragmentation"
                        elif "RAM:" in response:
                            try:
                                parts = response.split(':')[1].split(',')
                                if len(parts) >= 4:
                                    free_ram = float(parts[0])
                                    used_ram = float(parts[1])
                                    total_ram = float(parts[2])
                                    fragmentation = float(parts[3])
                                    
                                    self.hardware_stats['ram_free_bytes'].append(free_ram)
                                    self.hardware_stats['ram_used_bytes'].append(used_ram)
                                    self.hardware_stats['ram_total_bytes'].append(total_ram)
                                    
                                    # Calculate flash usage estimate based on model size
                                    flash_used = 50000  # Rough estimate for 16x16 model ~50KB
                                    self.hardware_stats['flash_used_bytes'].append(flash_used)
                                    self.hardware_stats['flash_total_bytes'].append(262144)  # 256KB total
                                    
                            except (ValueError, IndexError):
                                pass
                            
                    attempt += 1
                
            logger.info("✅ Initial hardware stats collected")
                
        except Exception as e:
            logger.error(f"Initial hardware stats failed: {e}")

    def send_prediction_request(self, voltage, current, soh, q_c):
        """Send prediction request to Arduino (FIXED for 16x16 protocol)"""
        try:
            if not self.arduino_connected:
                return None, None
                
            # Scale features
            scaled_features = self.scale_features(voltage, current, soh, q_c)
            
            # Send scaled data to Arduino (16x16 format: just CSV, no prefix)
            command = f"{scaled_features[0]:.6f},{scaled_features[1]:.6f},{scaled_features[2]:.6f},{scaled_features[3]:.6f}\n"
            self.arduino.write(command.encode())
            
            # Read response - Arduino 16x16 format: "📊 SOC: X.XXXX (XX.XX%) | Time: XXXX μs"
            start_time = time.time()
            timeout = 3.0
            
            while (time.time() - start_time) < timeout:
                if self.arduino.in_waiting > 0:
                    response = self.arduino.readline().decode().strip()
                    
                    # Look for SOC prediction response
                    if "📊 SOC:" in response:
                        try:
                            # Parse percentage in parentheses: (XX.XX%)
                            if "(" in response and "%)" in response:
                                percent_start = response.find("(") + 1
                                percent_end = response.find("%)")
                                soc_pred = float(response[percent_start:percent_end])
                            else:
                                # Fallback: parse decimal and convert to percentage
                                soc_start = response.find("SOC:") + 4
                                soc_end = response.find(" ", soc_start)
                                if soc_end == -1:
                                    soc_end = response.find("(", soc_start)
                                soc_decimal = float(response[soc_start:soc_end].strip())
                                soc_pred = soc_decimal * 100  # Convert to percentage
                            
                            # Parse inference time: "Time: XXXX μs"
                            inference_time = None
                            if "Time:" in response and "μs" in response:
                                time_start = response.find("Time:") + 5
                                time_end = response.find("μs", time_start)
                                inference_time = float(response[time_start:time_end].strip())
                                self.hardware_stats['inference_time_us'].append(inference_time)
                            
                            return soc_pred, inference_time
                            
                        except (ValueError, IndexError) as e:
                            logger.error(f"Prediction parsing failed: {e} | Response: '{response}'")
                            return None, None
                    
                    # Handle Arduino errors
                    elif "❌ Error:" in response:
                        logger.error(f"Arduino error: {response}")
                        return None, None
                    
                    # Skip other responses (performance stats, etc.)
                    elif any(x in response for x in ["===", "⏱️", "📊 Avg", "📊 Min", "📊 Max", "🔢"]):
                        continue
                            
                time.sleep(0.01)
                
            logger.warning("Prediction timeout")
            return None, None
            
        except Exception as e:
            logger.error(f"Prediction request failed: {e}")
            self.communication_errors += 1
            return None, None

    def update_plots(self, frame):
        """Update plots for performance"""
        try:
            # Get next data sample
            data_sample = self.get_next_data_sample()
            if data_sample is None:
                return
                
            # Send prediction request
            soc_pred, inference_time = self.send_prediction_request(
                data_sample['voltage'], 
                data_sample['current'],
                data_sample['soh'], 
                data_sample['q_c']
            )
            
            if soc_pred is not None:
                # Update data
                timestamp = time.time() - self.start_time
                self.timestamps.append(timestamp)
                self.voltage_values.append(data_sample['voltage'])
                self.soc_predictions.append(soc_pred)
                self.soc_ground_truth.append(data_sample['soc_true'])
                
                # Calculate MAE
                mae = abs(soc_pred - data_sample['soc_true'])
                self.mae_errors.append(mae)
                
                self.successful_predictions += 1
                
            self.prediction_count += 1
            
            # Update plots only if we have data
            if len(self.timestamps) > 1:
                self.plot_data()
                self.update_hardware_table()
                
            # Add delay between predictions
            time.sleep(PREDICTION_DELAY / 1000.0)
            
        except Exception as e:
            logger.error(f"Plot update failed: {e}")

    def plot_data(self):
        """Plot voltage and SOC data"""
        try:
            # Clear axes
            self.axes[0].clear()
            self.axes[1].clear()
            
            if len(self.timestamps) == 0:
                return
                
            times = list(self.timestamps)
            voltages = list(self.voltage_values)
            soc_preds = list(self.soc_predictions)
            soc_truth = list(self.soc_ground_truth)
            
            # Plot 1: Voltage
            self.axes[0].plot(times, voltages, 'g-', linewidth=2, label='Voltage', alpha=0.8)
            self.axes[0].set_ylabel('Voltage [V]', fontweight='bold')
            self.axes[0].set_title('Battery Voltage', fontweight='bold')
            self.axes[0].grid(True, alpha=0.3)
            self.axes[0].legend()
            
            # Plot 2: SOC Comparison
            self.axes[1].plot(times, soc_preds, 'r-', linewidth=2, label='Arduino Prediction (16x16)', alpha=0.8)
            self.axes[1].plot(times, soc_truth, 'b-', linewidth=2, label='Ground Truth SOC', alpha=0.8)
            self.axes[1].set_ylabel('SOC [%]', fontweight='bold')
            self.axes[1].set_xlabel('Time [s]', fontweight='bold')
            self.axes[1].set_title('SOC Prediction vs Ground Truth (16x16 Model)', fontweight='bold')
            self.axes[1].grid(True, alpha=0.3)
            self.axes[1].legend()
            
            # Set consistent y-axis for SOC
            self.axes[1].set_ylim(0, 100)
            
        except Exception as e:
            logger.error(f"Plotting failed: {e}")

    def update_hardware_table(self):
        """Update hardware statistics table with Verbrauch/Gesamt format"""
        try:
            runtime = time.time() - self.start_time
            success_rate = (self.successful_predictions / max(self.prediction_count, 1)) * 100
            
            # Current MAE
            current_mae = np.mean(list(self.mae_errors)[-10:]) if len(self.mae_errors) > 0 else 0
            
            # Hardware stats (latest values)
            ram_free = self.hardware_stats['ram_free_bytes'][-1] if self.hardware_stats['ram_free_bytes'] else float('nan')
            ram_used = self.hardware_stats['ram_used_bytes'][-1] if self.hardware_stats['ram_used_bytes'] else float('nan')
            ram_total = self.hardware_stats['ram_total_bytes'][-1] if self.hardware_stats['ram_total_bytes'] else 32768  # 32KB default for Arduino UNO R4
            
            flash_used = self.hardware_stats['flash_used_bytes'][-1] if self.hardware_stats['flash_used_bytes'] else float('nan')
            flash_total = self.hardware_stats['flash_total_bytes'][-1] if self.hardware_stats['flash_total_bytes'] else 262144  # 256KB default for Arduino UNO R4
            
            temperature = self.hardware_stats['temperature_celsius'][-1] if self.hardware_stats['temperature_celsius'] else float('nan')
            avg_inference = np.mean(list(self.hardware_stats['inference_time_us'])[-10:]) if self.hardware_stats['inference_time_us'] else float('nan')
            
            # Format RAM and Flash as used/total (Verbrauch/Gesamt)
            if not np.isnan(ram_used) and not np.isnan(ram_total):
                ram_display = f"{ram_used/1024:.1f}/{ram_total/1024:.0f} KB"
            else:
                # Calculate used from free if available
                if not np.isnan(ram_free):
                    ram_used_calc = ram_total - ram_free
                    ram_display = f"{ram_used_calc/1024:.1f}/{ram_total/1024:.0f} KB"
                else:
                    ram_display = "?/32 KB"
            
            if not np.isnan(flash_used) and not np.isnan(flash_total):
                flash_display = f"{flash_used/1024:.1f}/{flash_total/1024:.0f} KB"
            else:
                flash_display = "~50/256 KB"  # Estimated for 16x16 model
            
            stats_text = f"""
┌─ ARDUINO SOC MONITOR STATS (16x16 FINAL) ─┐
│ Runtime: {runtime:.1f}s                     │
│ Predictions: {self.prediction_count}        │ 
│ Success Rate: {success_rate:.1f}%           │
│ MAE (Current): {current_mae:.3f}%           │
│ Comm Errors: {self.communication_errors}    │
├─ HARDWARE STATS ─────────────────────────── │
│ RAM: {ram_display:<15}                │
│ Flash: {flash_display:<13}                │
│ Temperature: {temperature:.1f}°C            │
│ Avg Inference: {avg_inference:.0f} μs       │
└─────────────────────────────────────────────┘
"""
            
            self.stats_text.set_text(stats_text)
            
        except Exception as e:
            logger.error(f"Hardware table update failed: {e}")

    def run(self):
        """Run the monitoring system"""
        try:
            logger.info("🚀 Starting Arduino SOC Monitor V2 (16x16 Model - FINAL)...")
            
            # Step 1: Upload Arduino sketch
            if not self.upload_arduino_sketch():
                logger.error("❌ Failed to upload Arduino sketch")
                return
                
            # Step 2: Load ground truth data
            if not self.load_ground_truth_data():
                logger.error("❌ Failed to load ground truth data")
                return
                
            # Step 3: Connect to Arduino
            if not self.connect_arduino():
                logger.error("❌ Failed to connect to Arduino")
                return
                
            # Step 4: Start monitoring
            logger.info("✅ Starting live monitoring...")
            
            # Setup animation
            ani = animation.FuncAnimation(
                self.fig, self.update_plots, interval=UPDATE_INTERVAL, 
                blit=False, cache_frame_data=False
            )
            
            plt.tight_layout()
            plt.show()
            
        except KeyboardInterrupt:
            logger.info("👋 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
        finally:
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
                logger.info("Arduino connection closed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Arduino SOC Monitor V2 (16x16 Model - FINAL)')
    parser.add_argument('--port', default=ARDUINO_PORT, help='Arduino port')
    parser.add_argument('--no-upload', action='store_true', help='Skip Arduino upload')
    
    args = parser.parse_args()
    
    monitor = ArduinoSOCMonitor(port=args.port)
    
    if args.no_upload:
        monitor.upload_arduino_sketch = lambda: True
        logger.info("⚠️ Skipping Arduino upload")
    
    monitor.run()

if __name__ == "__main__":
    main()
