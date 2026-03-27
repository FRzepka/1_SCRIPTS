"""
OPTIMIZED ARDUINO SOC MONITOR - High Performance Live Plots
=========================================================

OPTIMIERTE VERSION mit:
- NUR 2 Plots: Voltage + SOC Prediction vs Ground Truth
- Comprehensive Flash/Memory Statistics Table
- Reduzierte Plot-Komplexität für bessere Performance
- Real-time hardware monitoring als Tabelle
- Echte Arduino LSTM Hardware Communication

🚀 PERFORMANCE OPTIMIERT FÜR LIVE MONITORING 🚀
"""

import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import threading
import queue
from collections import deque
import logging
import warnings

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
PLOT_WINDOW_SIZE = 200  # Datenpunkte im Plot (weniger für Performance)
UPDATE_INTERVAL = 300   # Plot Update alle 300ms (schneller)
PREDICTION_DELAY = 30   # 30ms zwischen Predictions (schneller)

class OptimizedArduinoSOCMonitor:
    """Optimierter Arduino LSTM SOC Monitor mit minimalen Plots"""
    
    def __init__(self, port, baudrate, data_path):
        self.port = port
        self.baudrate = baudrate
        self.data_path = data_path
        
        # Hardware Connection
        self.arduino = None
        self.arduino_connected = False
        
        # Data Queues mit begrenzter Größe
        self.arduino_data_queue = queue.Queue(maxsize=50)  # Memory-efficient
        self.prediction_running = False
        
        # Performance-optimierte Data Storage
        self.timestamps = deque(maxlen=PLOT_WINDOW_SIZE)
        self.voltage_values = deque(maxlen=PLOT_WINDOW_SIZE)
        self.soc_predictions = deque(maxlen=PLOT_WINDOW_SIZE)
        self.soc_ground_truth = deque(maxlen=PLOT_WINDOW_SIZE)
        self.mae_errors = deque(maxlen=PLOT_WINDOW_SIZE)
        
        # Hardware Metrics - Table only (no plots)
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
        self.fig.suptitle('🚀 OPTIMIZED Arduino LSTM SOC Monitor', fontsize=16, fontweight='bold')
        
        # Hardware Stats Table
        self.setup_hardware_table()
        
        logger.info("Optimized Arduino SOC Monitor initialized")
        
    def setup_hardware_table(self):
        """Setup hardware statistics table"""
        # Create text area for hardware stats table
        self.stats_text = self.fig.text(0.02, 0.02, "", fontsize=9, fontfamily='monospace',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
    def load_ground_truth_data(self):
        """Lädt Ground Truth Data von MGFarm mit korrekten Spaltennamen"""
        try:
            logger.info(f"Loading Ground Truth data from: {self.data_path}")
            df = pd.read_parquet(self.data_path)
            
            # Korrekte MGFarm Spaltennamen verwenden
            required_cols = ['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_c', 'SOC_ZHU']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in data: {missing_cols}")
                logger.info(f"Available columns: {list(df.columns)}")
                raise ValueError(f"Missing columns in data: {missing_cols}")
            
            # Data preprocessing
            df = df.dropna(subset=required_cols)
            df = df[df['Voltage[V]'] > 0]  # Filter invalid voltages
            
            # Rename columns for consistency
            df_clean = df.rename(columns={
                'Voltage[V]': 'voltage',
                'Current[A]': 'current', 
                'SOH_ZHU': 'soh',
                'Q_c': 'q_c',
                'SOC_ZHU': 'soc'
            })
                
            self.ground_truth_data = df_clean[['voltage', 'current', 'soh', 'q_c', 'soc']].copy()
            logger.info(f"✅ Ground Truth data loaded: {len(self.ground_truth_data)} samples")
            
            # Setup scaler identical to training
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
                # Create new scaler from ground truth data using original column names
                feature_cols = ['voltage', 'current', 'soh', 'q_c']
                features = self.ground_truth_data[feature_cols].values
                
                self.scaler = StandardScaler()
                self.scaler.fit(features)
                
                # Save scaler for future use
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
            logger.warning("No ground truth data available")
            return None
            
        try:
            # Cycle through dataset
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
        """Arduino connection with timeout"""
        try:
            logger.info(f"Connecting to Arduino on {self.port}...")
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(3)  # Arduino connection stabilization
            
            # Test connection with STATS command
            self.arduino.write(b'STATS\n')
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
        """Get initial hardware statistics for Flash memory info"""
        try:
            # BENCHMARK command for comprehensive stats
            self.arduino.write(b'BENCHMARK\n')
            time.sleep(0.5)  # Give Arduino time to respond
            response = self.arduino.readline().decode().strip()
            
            logger.info(f"🔍 BENCHMARK response: '{response}'")
            if response and "BENCHMARK:" in response:
                # Parse: "BENCHMARK:avg_us,min_us,max_us,total_ram,free_ram,cpu_mhz"
                data = response.replace("BENCHMARK:", "").split(",")
                logger.info(f"🔍 BENCHMARK data parts: {data}")
                if len(data) >= 6:
                    total_ram = int(data[3])
                    free_ram = int(data[4])
                    cpu_mhz = float(data[5])
                    
                    # Improved Flash memory estimation based on Arduino detection
                    flash_total, flash_used = self._estimate_flash_memory(total_ram, cpu_mhz)
                    flash_free = flash_total - flash_used
                    
                    # Store initial values
                    self.hardware_stats['ram_total_bytes'].append(total_ram)
                    self.hardware_stats['flash_total_bytes'].append(flash_total)
                    self.hardware_stats['flash_used_bytes'].append(flash_used)
                    self.hardware_stats['flash_free_bytes'].append(flash_free)
                    
                    logger.info(f"📊 Hardware Info: RAM={total_ram}B ({total_ram/1024:.1f}KB), Flash={flash_total/1024:.0f}KB (used: {flash_used/1024:.0f}KB), CPU={cpu_mhz}MHz")
                    # Analyze real vs estimated values
                    self._analyze_hardware_efficiency(total_ram, free_ram, cpu_mhz)
                else:
                    logger.warning(f"BENCHMARK response has insufficient data: {len(data)} parts")
                    # Set default values for Flash memory
                    self._set_default_flash_values()
            else:
                logger.warning(f"No valid BENCHMARK response, setting defaults")
                self._set_default_flash_values()
                
        except Exception as e:
            logger.warning(f"Initial hardware stats failed: {e}")
            self._set_default_flash_values()

    def _estimate_flash_memory(self, ram_size, cpu_mhz):
        """Estimate Flash memory size and usage based on Arduino characteristics"""
        # Arduino Uno R4 WiFi: 256KB Flash (RA4M1), 32KB RAM, 48MHz
        # Fine-tuned detection range for more precise identification (47-49MHz vs original 45-50MHz)
        if 30000 <= ram_size <= 35000 and 47 <= cpu_mhz <= 49:
            flash_total = 256 * 1024  # 256KB Flash (RA4M1)
            flash_used = 120 * 1024   # LSTM model estimate for RA4M1
            logger.info("🔍 Detected: Arduino Uno R4 WiFi (RA4M1 + ESP32-S3) [FINE-TUNED DETECTION]")
            logger.info(f"    📋 RA4M1 Core Specifications:")
            logger.info(f"      - Flash Memory: 256KB")
            logger.info(f"      - SRAM: 32KB") 
            logger.info(f"      - CPU Frequency: 48MHz (Cortex-M4)")
            logger.info(f"    📋 ESP32-S3 Co-processor:")
            logger.info(f"      - ROM: 384KB")
            logger.info(f"      - SRAM: 512KB")
            logger.info(f"      - CPU Frequency: up to 240MHz (dual-core)")
            logger.info(f"    💾 LSTM Memory Allocation: {flash_used/1024:.0f}KB of {flash_total/1024:.0f}KB Flash")
            
        # Arduino Uno Classic: 32KB Flash, 2KB RAM, 16MHz
        elif ram_size <= 2048 and cpu_mhz == 16:
            flash_total = 32 * 1024  # 32KB Flash
            flash_used = 28 * 1024   # LSTM model is quite large, estimate 28KB used
            logger.info("🔍 Detected: Arduino Uno Classic (ATmega328P)")
            
        # Arduino Nano: 32KB Flash, 2KB RAM, 16MHz (same as Uno Classic)
        elif ram_size <= 2048 and cpu_mhz == 16:
            flash_total = 32 * 1024
            flash_used = 28 * 1024
            logger.info("🔍 Detected: Arduino Nano (ATmega328P)")
            
        # Arduino Due: 512KB Flash, 96KB RAM, 84MHz
        elif cpu_mhz == 84:
            flash_total = 512 * 1024  # 512KB Flash
            flash_used = 256 * 1024   # Estimate 256KB used for LSTM
            logger.info("🔍 Detected: Arduino Due")
            
        # ESP32 standalone: Variable Flash, ~300KB RAM, 240MHz
        elif cpu_mhz >= 160 and ram_size > 200000:
            flash_total = 4 * 1024 * 1024  # 4MB Flash (typical)
            flash_used = 1 * 1024 * 1024   # Estimate 1MB used
            logger.info("🔍 Detected: ESP32 (or similar)")
            
        # Arduino Mega: 256KB Flash, 8KB RAM, 16MHz
        elif ram_size >= 8000 and cpu_mhz == 16:
            flash_total = 256 * 1024  # 256KB Flash
            flash_used = 200 * 1024   # Estimate 200KB used
            logger.info("🔍 Detected: Arduino Mega")
              # Default fallback
        else:
            flash_total = 256 * 1024  # 256KB Flash
            flash_used = 150 * 1024   # Conservative estimate
            logger.info(f"🔍 Unknown Arduino type (RAM: {ram_size}B, CPU: {cpu_mhz}MHz) - using defaults")
          return flash_total, flash_used

    def _set_default_flash_values(self):
        """Set default Flash memory values when detection fails"""
        # Default to Arduino Uno R4 WiFi specs (most common modern Arduino)
        flash_total = 256 * 1024   # 256KB Flash (RA4M1)
        
        # REAL Flash usage calculation based on actual file analysis:
        # - LSTM weights file (lstm_weights.h): 85,082 bytes (~83KB)
        # - Arduino firmware (.ino): 16,947 bytes (~17KB)  
        # - Arduino core libraries (estimated): ~55KB
        # - Boot loader and system overhead: ~5KB
        # Total realistic usage: ~160KB
        lstm_weights_size = 85082      # Actual file size of lstm_weights.h
        firmware_size = 16947          # Actual file size of .ino file
        arduino_core_libs = 55 * 1024  # Estimated Arduino core libraries 
        system_overhead = 5 * 1024     # Boot loader and system overhead
        
        flash_used = lstm_weights_size + firmware_size + arduino_core_libs + system_overhead
        flash_free = flash_total - flash_used
        
        self.hardware_stats['flash_total_bytes'].append(flash_total)
        self.hardware_stats['flash_used_bytes'].append(flash_used)
        self.hardware_stats['flash_free_bytes'].append(flash_free)
        
        logger.info(f"📊 Default Flash values (Arduino Uno R4 WiFi): Total={flash_total/1024:.0f}KB, Used={flash_used/1024:.0f}KB")
        logger.info(f"   ├─ LSTM weights: {lstm_weights_size/1024:.1f}KB (actual file size)")
        logger.info(f"   ├─ Arduino firmware: {firmware_size/1024:.1f}KB (actual file size)")
        logger.info(f"   ├─ Arduino core libs: {arduino_core_libs/1024:.0f}KB (estimated)")
        logger.info(f"   └─ System overhead: {system_overhead/1024:.0f}KB (estimated)")
        
        flash_used = lstm_weights_size + firmware_size + arduino_core_libs + system_overhead
        flash_free = flash_total - flash_used
        
        self.hardware_stats['flash_total_bytes'].append(flash_total)
        self.hardware_stats['flash_used_bytes'].append(flash_used)
        self.hardware_stats['flash_free_bytes'].append(flash_free)
        
        logger.info(f"📊 Default Flash values (Arduino Uno R4 WiFi): Total={flash_total/1024:.0f}KB, Used={flash_used/1024:.0f}KB")
        logger.info(f"   ├─ LSTM weights: {lstm_weights_size/1024:.1f}KB (actual file size)")
        logger.info(f"   ├─ Arduino firmware: {firmware_size/1024:.1f}KB (actual file size)")
        logger.info(f"   ├─ Arduino core libs: {arduino_core_libs/1024:.0f}KB (estimated)")
        logger.info(f"   └─ System overhead: {system_overhead/1024:.0f}KB (estimated)")
    
    def _analyze_hardware_efficiency(self, total_ram, free_ram, cpu_mhz):
        """Analyze real hardware efficiency and log detailed information"""
        try:
            used_ram = total_ram - free_ram
            ram_usage_percent = (used_ram / total_ram * 100) if total_ram > 0 else 0
            
            logger.info("=" * 50)
            logger.info("🔍 DETAILED HARDWARE ANALYSIS")
            logger.info("=" * 50)
            
            # Arduino identification
            if 30000 <= total_ram <= 35000 and 45 <= cpu_mhz <= 50:
                board_type = "Arduino Uno R4 WiFi (RA4M1 + ESP32-S3)"
                expected_ram = 32 * 1024
                expected_flash = 256 * 1024
            elif total_ram <= 2048 and cpu_mhz == 16:
                board_type = "Arduino Uno Classic (ATmega328P)"
                expected_ram = 2 * 1024
                expected_flash = 32 * 1024
            elif total_ram >= 8000 and cpu_mhz == 16:
                board_type = "Arduino Mega 2560"
                expected_ram = 8 * 1024
                expected_flash = 256 * 1024
            else:
                board_type = f"Unknown Arduino (RAM: {total_ram}B, CPU: {cpu_mhz}MHz)"
                expected_ram = total_ram
                expected_flash = 256 * 1024
            
            logger.info(f"📋 Board Type: {board_type}")
            logger.info(f"💾 RAM Analysis:")
            logger.info(f"  - Total Available: {total_ram:,} bytes ({total_ram/1024:.1f} KB)")
            logger.info(f"  - Currently Used:  {used_ram:,} bytes ({used_ram/1024:.1f} KB) = {ram_usage_percent:.1f}%")
            logger.info(f"  - Currently Free:  {free_ram:,} bytes ({free_ram/1024:.1f} KB) = {100-ram_usage_percent:.1f}%")
            
            if expected_ram:
                ram_deviation = ((total_ram - expected_ram) / expected_ram * 100) if expected_ram > 0 else 0
                logger.info(f"  - Expected RAM:    {expected_ram:,} bytes ({expected_ram/1024:.1f} KB)")
                logger.info(f"  - Deviation:       {ram_deviation:+.1f}% from expected")
            
            # LSTM Model RAM requirements analysis
            lstm_estimated_ram = 32 * 32 * 4 + 4 * 4 * 4 + 1024  # LSTM states + inputs + buffers
            logger.info(f"🧠 LSTM Model RAM Usage:")
            logger.info(f"  - Estimated Need:  {lstm_estimated_ram:,} bytes ({lstm_estimated_ram/1024:.1f} KB)")
            logger.info(f"  - Available RAM:   {total_ram:,} bytes ({total_ram/1024:.1f} KB)")
            logger.info(f"  - RAM Capacity:    {(lstm_estimated_ram/total_ram*100):.1f}% of total RAM")
            
            if lstm_estimated_ram > total_ram:
                logger.warning("⚠️  LSTM model may exceed available RAM!")
            elif lstm_estimated_ram > total_ram * 0.8:
                logger.warning("⚠️  LSTM model uses >80% of RAM - might cause issues")
            else:
                logger.info("✅ LSTM model fits comfortably in available RAM")
                
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Hardware efficiency analysis failed: {e}")
    
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
                # Get next data sample
                data_sample = self.get_next_data_sample()
                if data_sample is None:
                    time.sleep(0.1)
                    continue
                
                # Arduino prediction
                soc_pred, hardware_metrics = self.predict_with_arduino(
                    data_sample['voltage'], data_sample['current'], 
                    data_sample['soh'], data_sample['q_c']
                )
                
                if soc_pred is not None:
                    # Calculate error
                    mae_error = abs(soc_pred - data_sample['soc_true'])
                      # Queue data for plotting
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
                        pass  # Skip if queue is full (performance optimization)
                
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
                min_inference = np.min(list(self.hardware_stats['inference_time_us']))
                max_inference = np.max(list(self.hardware_stats['inference_time_us']))
                throughput = self.successful_predictions / runtime if runtime > 0 else 0
            else:
                avg_inference = min_inference = max_inference = throughput = 0
            
            if len(self.hardware_stats['ram_free_bytes']) > 0:
                current_ram_free = list(self.hardware_stats['ram_free_bytes'])[-1]
                current_ram_used = list(self.hardware_stats['ram_used_bytes'])[-1]
                current_ram_total = list(self.hardware_stats['ram_total_bytes'])[-1]
                ram_usage_percent = (current_ram_used / current_ram_total * 100) if current_ram_total > 0 else 0
            else:
                current_ram_free = current_ram_used = current_ram_total = ram_usage_percent = 0
            
            if len(self.hardware_stats['cpu_load_percent']) > 0:
                current_cpu_load = list(self.hardware_stats['cpu_load_percent'])[-1]
                avg_cpu_load = np.mean(list(self.hardware_stats['cpu_load_percent']))
            else:
                current_cpu_load = avg_cpu_load = 0
            
            if len(self.hardware_stats['temperature_celsius']) > 0:
                current_temp = list(self.hardware_stats['temperature_celsius'])[-1]
            else:
                current_temp = 0
                
            # Flash memory stats with real usage analysis
            if len(self.hardware_stats['flash_total_bytes']) > 0:
                flash_total = list(self.hardware_stats['flash_total_bytes'])[-1]
                flash_used = list(self.hardware_stats['flash_used_bytes'])[-1]
                flash_free = list(self.hardware_stats['flash_free_bytes'])[-1]
                flash_usage_percent = (flash_used / flash_total * 100) if flash_total > 0 else 0
                
                # Calculate actual Flash efficiency based on RAM usage
                actual_ram_efficiency = (current_ram_used / current_ram_total * 100) if current_ram_total > 0 else 0
                flash_efficiency_estimate = min(flash_usage_percent, actual_ram_efficiency * 1.5)  # Flash usually higher than RAM
                
                # Debug logging for Flash memory
                if flash_total == 0:
                    logger.warning(f"🔍 Flash memory showing 0: total={flash_total}, used={flash_used}, free={flash_free}")
                    logger.warning(f"🔍 Flash stats arrays lengths: total={len(self.hardware_stats['flash_total_bytes'])}, used={len(self.hardware_stats['flash_used_bytes'])}")
                else:
                    logger.debug(f"🔍 Flash Analysis: Total={flash_total/1024:.0f}KB, Used={flash_used/1024:.0f}KB ({flash_usage_percent:.1f}%), RAM efficiency={actual_ram_efficiency:.1f}%")
            else:
                flash_total = flash_used = flash_free = flash_usage_percent = 0
                flash_efficiency_estimate = 0
                logger.warning("🔍 No Flash memory stats available in arrays")
            
            # Current MAE
            current_mae = list(self.mae_errors)[-1] if len(self.mae_errors) > 0 else 0
            avg_mae = np.mean(list(self.mae_errors)) if len(self.mae_errors) > 0 else 0
            
            # Success rate
            success_rate = (self.successful_predictions / self.prediction_count * 100) if self.prediction_count > 0 else 0
            
            # Format table with Arduino Uno R4 WiFi specific info
            table_text = f"""
┌─────────────────────────────────────────────┐
│              🚀 ARDUINO HARDWARE STATS      │
├─────────────────────────────────────────────┤
│ 📊 LIVE PERFORMANCE                         │
│   Tests:        {self.prediction_count:6d}                │
│   Success:      {success_rate:5.1f}%                │
│   Runtime:      {runtime:5.0f}s                │
│   Throughput:   {throughput:5.1f} Hz              │
│                                             │
│ ⚡ ARDUINO INFERENCE                         │
│   Current:      {avg_inference:5.0f} μs              │
│   Min/Max:      {min_inference:5.0f}/{max_inference:5.0f} μs           │
│                                             │
│ 💾 RAM USAGE (RA4M1 Core)                   │
│   Free:         {current_ram_free:5d} B ({100-ram_usage_percent:4.1f}%)        │
│   Used:         {current_ram_used:5d} B ({ram_usage_percent:4.1f}%)        │
│   Total:        {current_ram_total/1024:5.1f} KB               │
│                                             │
│ 📀 FLASH MEMORY (RA4M1)                     │
│   Free:         {flash_free/1024:5.0f} KB ({100-flash_usage_percent:4.1f}%)       │
│   Used:         {flash_used/1024:5.0f} KB ({flash_usage_percent:4.1f}%)       │
│   Total:        {flash_total/1024:5.0f} KB               │
│   *Estimated based on board detection       │
│                                             │
│ 🔋 SOC ACCURACY                             │
│   Current MAE:  {current_mae:6.4f}              │
│   Average MAE:  {avg_mae:6.4f}              │
│                                             │
│ 🌡️ SYSTEM STATUS                             │
│   CPU Load:     {current_cpu_load:5.1f}%    │
│   Temperature:  {current_temp:5.1f}°C       │
└─────────────────────────────────────────────┘
"""
            self.stats_text.set_text(table_text)
            
        except Exception as e:
            logger.error(f"Hardware table update failed: {e}")
    
    def update_plots(self, frame):
        """Update nur 2 plots für Performance"""
        # Check for new data
        try:
            while not self.arduino_data_queue.empty():
                data = self.arduino_data_queue.get_nowait()
                
                # Store in performance-optimized deques
                self.timestamps.append(data['timestamp'])
                self.voltage_values.append(data['voltage'])
                self.soc_predictions.append(data['soc_pred'])
                self.soc_ground_truth.append(data['soc_true'])
                self.mae_errors.append(data['mae_error'])
                
        except queue.Empty:
            pass
        
        if len(self.timestamps) < 2:
            return
        
        # Convert to arrays for plotting
        times = np.array(self.timestamps) - self.timestamps[0]  # Relative time
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
    
    def start_monitoring(self):
        """Start optimized monitoring with only 2 plots"""
        print("🚀 OPTIMIZED Arduino LSTM SOC Monitor")
        print("="*50)
        print("✨ Features:")
        print("  - Only 2 plots: Voltage + SOC vs Ground Truth")
        print("  - Comprehensive hardware stats table")
        print("  - Real Arduino LSTM predictions")
        print("  - Optimized for performance")
        print()
        
        logger.info("🚀 Starting Optimized Arduino SOC Monitor...")
        
        # Load data
        if not self.load_ground_truth_data():
            logger.error("❌ Failed to load ground truth data")
            return False
        
        # Connect Arduino
        if not self.connect_arduino():
            logger.error("❌ Failed to connect to Arduino")
            return False
        
        # Start background prediction loop
        self.prediction_running = True
        prediction_thread = threading.Thread(target=self.background_prediction_loop, daemon=True)
        prediction_thread.start()
        
        logger.info("✅ Background prediction loop started")
        logger.info("🎯 Starting live monitoring...")
        logger.info("⏹️ Close plot window to stop")
        
        # Start animation
        try:
            ani = animation.FuncAnimation(
                self.fig, self.update_plots,
                interval=UPDATE_INTERVAL,
                blit=False,
                cache_frame_data=False
            )
            
            plt.show()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Monitoring stopped by user")
        finally:
            self.stop_monitoring()
        
        return True
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        logger.info("🛑 Stopping monitoring...")
        self.prediction_running = False
        
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            logger.info("🔌 Arduino connection closed")
        
        # Print final statistics
        if self.prediction_count > 0:
            runtime = time.time() - self.start_time
            success_rate = (self.successful_predictions / self.prediction_count * 100)
            avg_mae = np.mean(list(self.mae_errors)) if len(self.mae_errors) > 0 else 0
            
            logger.info("="*50)
            logger.info("📊 FINAL STATISTICS:")
            logger.info(f"🎯 Total Predictions: {self.prediction_count}")
            logger.info(f"✅ Success Rate: {success_rate:.1f}%")
            logger.info(f"⏱️ Runtime: {runtime:.0f} seconds")
            logger.info(f"🔋 Final MAE: {avg_mae:.4f}")
            logger.info("✅ Monitoring completed!")


def main():
    """Main function"""
    monitor = OptimizedArduinoSOCMonitor(ARDUINO_PORT, BAUDRATE, DATA_PATH)
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("⏹️ Program stopped by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()
