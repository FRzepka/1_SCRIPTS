\
"""
ARDUINO SOC MONITOR V2 - 64x64 VERSION (FINAL)
==============================================

✨ FEATURES:
- Automatisches Kompilieren und Hochladen des Arduino Sketches  
- ECHTE Flash-Werte vom Arduino-Compiler (via "RAM" command parsing)
- RAM/Flash Anzeige als Verbrauch/Gesamt (z.B. Used/Total KB)
- Live SOC Monitoring mit Ground Truth Vergleich
- Angepasst für 64x64 Model Arduino Sketch Kommunikation

🚀 PERFORMANCE OPTIMIERT FÜR LIVE MONITORING 🚀
"""
import queue
import threading
import serial
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os
import subprocess
import re
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings

# Warnings unterdrücken
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from font", category=UserWarning)

# Logging Setup
# Ensure thread name is part of the format for better debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== EINSTELLUNGEN =====
ARDUINO_PORT = 'COM13'
BAUDRATE = 115200
DATA_PATH = r"c:\\Users\\Florian\\SynologyDrive\\TUB\\3_Projekte\\MG_Farm\\5_Data\\01_LFP\\00_Data\\MGFarm_18650_Dataframes_ZHU\\MGFarm_18650_C19\\df.parquet"
# IMPORTANT: Update this path to your 64x64 sketch
ARDUINO_SKETCH_PATH = r"c:\\Users\\Florian\\SynologyDrive\\TUB\\1_Dissertation\\5_Codes\\LFP_SOC_SOH\\Arduino_Test_Setup\\Stateful_64_64\\code_weights\\arduino_lstm_soc_full64_with_monitoring\\arduino_lstm_soc_full64_with_monitoring.ino"
ARDUINO_CLI_PATH = r"C:\\Users\\Florian\\SynologyDrive\\TUB\\1_Dissertation\\5_Codes\\LFP_SOC_SOH\\Arduino_Test_Setup\\Arduino_CLI\\arduino-cli.exe"
ARDUINO_FQBN = "arduino:renesas_uno:unor4wifi"  # Arduino UNO R4 WiFi

PLOT_WINDOW_SIZE = 200  # Datenpunkte im Plot
UPDATE_INTERVAL = 300   # Plot Update alle 300ms (visualization)
PREDICTION_DELAY = 30   # 30ms zwischen Predictions (data sending)
STATS_COMMAND_INTERVAL = 5  # Send "STATS" and "RAM" commands every 5 seconds

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
        if not os.path.exists(self.cli_path):
            logger.error(f"Arduino CLI not found at {self.cli_path}. Please check the path.")
            return False
        try:
            subprocess.run([self.cli_path, "version"], timeout=5, capture_output=True, text=True, check=True)
            logger.info(f"Arduino CLI found and working at {self.cli_path}")
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.error(f"Arduino CLI at {self.cli_path} is not working correctly: {e}")
            return False
    
    def compile_and_upload(self):
        """Compile and upload the sketch"""
        if not self.arduino_cli_available:
            return False, "Arduino CLI not available or not working."
        
        sketch_dir = os.path.dirname(self.sketch_path)
        if not os.path.exists(self.sketch_path):
            logger.error(f"Arduino sketch not found at {self.sketch_path}")
            return False, f"Sketch not found: {self.sketch_path}"

        try:
            logger.info(f"🔨 Compiling Arduino sketch: {self.sketch_path} for {self.fqbn}...")
            compile_cmd = [self.cli_path, "compile", "--fqbn", self.fqbn, sketch_dir, "--warnings", "all"]
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"Compilation failed. Return code: {result.returncode}")
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
                return False, f"Compilation failed: {result.stderr}"
            
            logger.info("Compilation successful.")
            logger.info(f"Compiler output (stderr):\\n{result.stderr}") # Stderr often contains size info

            logger.info(f"📤 Uploading to Arduino on {self.port}...")
            upload_cmd = [self.cli_path, "upload", "-p", self.port, "--fqbn", self.fqbn, sketch_dir, "--verbose"]
            result = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                logger.error(f"Upload failed. Return code: {result.returncode}")
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
                return False, f"Upload failed: {result.stderr}"
            
            logger.info("🎉 Upload successful!")
            return True, "Upload successful"
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout during compile/upload.")
            return False, "Timeout during compile/upload."
        except Exception as e:
            logger.error(f"An unexpected error occurred during compile/upload: {e}")
            return False, f"Unexpected error: {e}"

class ArduinoSOCMonitor64:
    """Arduino SOC Monitor für 64x64 Model"""
    
    def __init__(self, port, baudrate, data_path, sketch_path, fqbn, cli_path):
        self.port = port
        self.baudrate = baudrate
        self.data_path = data_path
        self.sketch_path = sketch_path
        self.fqbn = fqbn
        self.cli_path = cli_path
        self.uploader = ArduinoUploader(sketch_path, fqbn, port, cli_path)

        self.arduino = None
        self.arduino_connected = False
        self.arduino_data_queue = queue.Queue(maxsize=100)
        self.prediction_running = False
        self.reader_thread = None  # Initialize reader_thread
        self.prediction_loop_thread = None  # Initialize prediction_loop_thread
        self.arduino_ready_event = threading.Event() # Event to signal Arduino readiness

        self.timestamps = deque(maxlen=PLOT_WINDOW_SIZE)
        self.voltage_values = deque(maxlen=PLOT_WINDOW_SIZE)
        # For 64x64, inputs are V, C, SOH, Q_c - plotting SOH and Q_c is optional
        self.soh_values = deque(maxlen=PLOT_WINDOW_SIZE) 
        self.qc_values = deque(maxlen=PLOT_WINDOW_SIZE)
        self.soc_predictions = deque(maxlen=PLOT_WINDOW_SIZE)
        self.soc_ground_truth = deque(maxlen=PLOT_WINDOW_SIZE)
        self.mae_errors = deque(maxlen=PLOT_WINDOW_SIZE)
        
        # Hardware Metrics - Table only
        self.hardware_stats = {
            'inference_time_us': deque(maxlen=100),
            'ram_used_kb': deque(maxlen=100),
            'ram_total_kb': deque(maxlen=100),
            'ram_usage_percent': deque(maxlen=100),
            'flash_used_kb': deque(maxlen=100),
            'flash_total_kb': deque(maxlen=100),
            'flash_usage_percent': deque(maxlen=100),
            'cpu_load_percent': deque(maxlen=100),
            'temperature_celsius': deque(maxlen=100),
        }
        
        # Ground Truth Data
        self.ground_truth_data = None
        self.scaler = None # Scaler will be setup for V, I, Temp, Power
        self.data_index = 0
        
        # Performance Tracking
        self.prediction_count = 0
        self.successful_predictions = 0
        self.communication_errors = 0
        self.start_time = time.time()
        
        # UI Layout
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8)) # Keep as 2,1 for SOC and Voltage
        self.fig.suptitle('🚀 Arduino SOC Monitor V2 - 64x64 Model', fontsize=16, fontweight='bold')
        
        # Hardware Stats Table
        self.setup_hardware_table()
        
        logger.info("Arduino SOC Monitor V2 (64x64) initialized")
        logger.warning("The 64x64 Arduino model has high RAM requirements (>100KB). " +
                       f"Current FQBN is '{self.fqbn}'. Ensure this board meets requirements (e.g., Mega2560, ESP32). " +
                       "Using an Arduino UNO R4 WiFi may lead to instability on the Arduino.")

    def setup_hardware_table(self):
        """Setup hardware statistics table"""
        self.stats_text = self.fig.text(0.02, 0.02, "Initializing...", fontsize=9, fontfamily='monospace',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    def upload_arduino_sketch(self):
        """Upload Arduino sketch automatically"""
        logger.info("🚀 Starting automatic Arduino sketch upload...")
        success, message = self.uploader.compile_and_upload()
        
        if success:
            logger.info("✅ Arduino sketch uploaded successfully!")
            logger.info("⏳ Waiting for Arduino to boot (10 seconds)...")
            time.sleep(10) # Wait for Arduino to boot
            return True
        else:
            logger.error(f"❌ Arduino upload failed: {message}")
            # Optionally, allow continuing without upload or exit
            # For now, we'll let it try to connect anyway.
            return False
            
    def load_ground_truth_data(self):
        """Lädt Ground Truth Data von MGFarm"""
        try:
            logger.info(f"Loading Ground Truth data from: {self.data_path}")
            df = pd.read_parquet(self.data_path)
            
            # For 64x64, inputs are Voltage, Current, SOH, Q_c
            # User needs to ensure these columns exist in the parquet file.
            self.required_cols = ['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_c', 'SOC_ZHU']
            
            if not all(col in df.columns for col in self.required_cols):
                missing_cols = [col for col in self.required_cols if col not in df.columns]
                logger.error(f"Missing required columns in data: {missing_cols}. "
                               "The 64x64 model requires Voltage, Current, SOH_ZHU, Q_c and SOC_ZHU for ground truth.")
                # Pad with 0 if columns are missing, or handle as error
                for col in missing_cols:
                    logger.warning(f"Column {col} is missing in the Parquet file. Will be filled with 0.")
                    df[col] = 0 # Fill missing columns with 0
                # return False # Indicate failure - decided to fill with 0 and continue
            
            self.ground_truth_data = df[self.required_cols].copy()
            # Ensure numeric types
            for col in self.required_cols:
                self.ground_truth_data[col] = pd.to_numeric(self.ground_truth_data[col], errors='coerce')
            
            self.ground_truth_data.dropna(inplace=True) # Drop rows with NaNs after conversion
            
            if len(self.ground_truth_data) == 0:
                logger.error("No valid ground truth data loaded after processing. Check file and columns.")
                return False

            logger.info(f"Ground Truth data loaded successfully. Shape: {self.ground_truth_data.shape}")
            self.setup_scaler()
            return True
            
        except Exception as e:
            logger.error(f"Error loading ground truth data: {e}")
            self.ground_truth_data = None
            return False
            
    def setup_scaler(self):
        """Setup scaler for V, I, SOH, Q_c"""
        if self.ground_truth_data is not None:
            try:
                # Scale Voltage, Current, SOH, Q_c
                features_to_scale = self.ground_truth_data[['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_c']]
                self.scaler = StandardScaler()
                self.scaler.fit(features_to_scale)
                logger.info("Scaler setup for Voltage, Current, SOH_ZHU, Q_c.")
            except Exception as e:
                logger.error(f"Error setting up scaler: {e}")
                self.scaler = None
        else:
            logger.warning("Ground truth data not loaded, scaler not set up.")

    def scale_features(self, voltage, current, soh, q_c):
        """Scale features: Voltage, Current, SOH, Q_c"""
        if self.scaler:
            features = np.array([[voltage, current, soh, q_c]])
            scaled_features = self.scaler.transform(features)
            return scaled_features[0]
        # Return unscaled if scaler not available, though this is not ideal for the model
        logger.warning("Scaler not available, returning unscaled features.")
        return np.array([voltage, current, soh, q_c])

    def get_next_data_sample(self):
        """Get next data sample (V, I, SOH, Q_c, SOC_true) from ground truth"""
        if self.ground_truth_data is None or len(self.ground_truth_data) == 0:
            logger.warning("No ground truth data available for sending.")
            return None, None, None, None, None # V, I, SOH, Q_c, SOC_true
            
        try:
            sample = self.ground_truth_data.iloc[self.data_index]
            self.data_index = (self.data_index + 1) % len(self.ground_truth_data)
            
            # Return V, I, SOH, Q_c, and true SOC
            return sample['Voltage[V]'], sample['Current[A]'], sample['SOH_ZHU'], sample['Q_c'], sample['SOC_ZHU']
        except Exception as e:
            logger.error(f"Error getting next data sample: {e}")
            return None, None, None, None, None

    def connect_arduino(self):
        """
        Connects to the Arduino, starts the reader thread, and waits for Arduino to signal readiness.
        Returns True if connection is successful and Arduino is ready, False otherwise.
        """
        if self.arduino_connected and self.arduino and self.arduino.is_open:
            logger.info("Already connected to Arduino.")
            return True
        
        try:
            logger.info(f"Attempting to connect to Arduino on {self.port} at {self.baudrate} baud...")
            if self.arduino and self.arduino.is_open:
                try:
                    self.arduino.close()
                    logger.info("Closed existing serial port before reopening.")
                except Exception as e_close:
                    logger.warning(f"Error closing existing serial port: {e_close}")
            
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=1) # timeout for read operations
            self.arduino.flushInput() # Clear any stale data from the input buffer
            self.arduino.reset_input_buffer() # Explicitly reset input buffer
            logger.info("Serial port opened. Starting reader thread and waiting for Arduino to initialize and signal READY...")

            self.arduino_ready_event.clear() # Ensure event is clear before waiting

            # Ensure reader_thread is started or restarted if necessary
            # It needs self.prediction_running to be True to loop, which is set in start_monitoring before calling this.
            if self.reader_thread is None or not self.reader_thread.is_alive():
                logger.info("Starting Arduino output reader thread...")
                self.reader_thread = threading.Thread(target=self._read_arduino_output_thread, name="ArduinoReaderThread", daemon=True)
                self.reader_thread.start()
            else:
                logger.info("Reader thread already running (connect_arduino called again?).")


            logger.info("Giving Arduino an extended moment to boot (8s) before waiting for READY signal...")
            time.sleep(8) # Increased initial wait for Arduino to boot (was 2s)

            logger.info(f"Waiting for Arduino to signal READY (timeout: 25s)...")
            ready_received = self.arduino_ready_event.wait(timeout=25.0) # Generous timeout for READY signal

            if ready_received:
                logger.info("✅ Arduino signaled READY. Connection successful.")
                self.arduino_connected = True
                # Optional: Short delay for any immediate post-READY messages to be queued by the reader thread
                time.sleep(0.5) 
                return True
            else:
                logger.warning("⚠️ Timed out waiting for Arduino READY signal during connection attempt.")
                if self.arduino and self.arduino.is_open:
                    self.arduino.close()
                self.arduino = None
                self.arduino_connected = False
                return False
                
        except serial.SerialException as e_serial:
            logger.error(f"Serial error connecting to Arduino: {e_serial}")
            if self.arduino and self.arduino.is_open: self.arduino.close()
            self.arduino = None
            self.arduino_connected = False
            return False
        except Exception as e_conn:
            logger.error(f"An unexpected error occurred during Arduino connection: {e_conn}", exc_info=True)
            if self.arduino and self.arduino.is_open: self.arduino.close()
            self.arduino = None
            self.arduino_connected = False
            return False
            
    def _read_arduino_output_thread(self):
        """Reads output from Arduino and puts it into a queue."""
        logger.info("Arduino output reader thread started.")
        
        parsing_ram_flash = False
        ram_flash_lines = []
        parsing_perf_stats = False
        perf_stats_lines = []
        loop_count = 0 # For periodic logging

        while self.prediction_running and self.arduino and self.arduino.is_open:
            loop_count += 1
            if loop_count % 1000 == 0: # Log every 1000 iterations (approx every 1s if sleep is 0.001)
                logger.debug(f"Reader thread loop iteration {loop_count}, arduino.in_waiting: {self.arduino.in_waiting if self.arduino else 'N/A'}")

            try:
                if self.arduino.in_waiting > 0:
                    line = self.arduino.readline().decode('utf-8', errors='replace').strip()
                    if line:
                        logger.debug(f"Arduino raw: {line}") # UNCOMMENTED

                        if parsing_ram_flash:
                            ram_flash_lines.append(line)
                            # RAM/Flash block is typically 3 lines: Header, RAM, Flash
                            if len(ram_flash_lines) >= 3:
                                try:
                                    ram_l = next((s for s in ram_flash_lines if "🐏 RAM:" in s), None)
                                    flash_l = next((s for s in ram_flash_lines if "💿 Flash:" in s), None)
                                    
                                    r_used, r_total, r_perc = None, None, None
                                    f_used, f_total, f_perc = None, None, None

                                    if ram_l:
                                        m = re.search(r"RAM: ([\d.]+)/([\d.]+) KB \(([\d.]+)%\)", ram_l)
                                        if m: r_used, r_total, r_perc = float(m.group(1)), float(m.group(2)), float(m.group(3))
                                    
                                    if flash_l:
                                        m = re.search(r"Flash: ([\d.]+)/([\d.]+) KB \(([\d.]+)%\)", flash_l)
                                        if m: f_used, f_total, f_perc = float(m.group(1)), float(m.group(2)), float(m.group(3))

                                    if r_used is not None and f_used is not None:
                                        self.arduino_data_queue.put({
                                            'type': 'ram_flash',
                                            'ram_used_kb': r_used, 'ram_total_kb': r_total, 'ram_usage_percent': r_perc,
                                            'flash_used_kb': f_used, 'flash_total_kb': f_total, 'flash_usage_percent': f_perc
                                        })
                                    else:
                                        logger.warning(f"Incomplete RAM/Flash parse: {ram_flash_lines}")
                                except Exception as e:
                                    logger.error(f"Err parsing RAM/Flash: {e} from {ram_flash_lines}")
                                finally:
                                    parsing_ram_flash = False
                                    ram_flash_lines = []
                            continue

                        if parsing_perf_stats:
                            perf_stats_lines.append(line)
                            # Check for a line that typically ends the performance stats block
                            if "MCU Temp:" in line or len(perf_stats_lines) >= 7: # Heuristic: block is usually ~7 lines
                                try:
                                    cpu_l, temp_c = None, None
                                    # Additional stats like inference times can be parsed here if needed
                                    for stat_line in perf_stats_lines:
                                        if "CPU Load:" in stat_line:
                                            m = re.search(r"CPU Load: ([\d.]+)%", stat_line)
                                            if m: cpu_l = float(m.group(1))
                                        if "MCU Temp:" in stat_line:
                                            m = re.search(r"MCU Temp: ([\d.]+)°C", stat_line)
                                            if m: temp_c = float(m.group(1))
                                    
                                    if cpu_l is not None and temp_c is not None:
                                        self.arduino_data_queue.put({
                                            'type': 'perf_stats',
                                            'cpu_load_percent': cpu_l,
                                            'temperature_celsius': temp_c
                                        })
                                    else:
                                        logger.warning(f"Incomplete Perf Stats parse: {perf_stats_lines}")
                                except Exception as e:
                                    logger.error(f"Err parsing Perf Stats: {e} from {perf_stats_lines}")
                                finally:
                                    parsing_perf_stats = False
                                    perf_stats_lines = []
                            continue
                        
                        # --- Start of block detection ---
                        if line.startswith("💾 MEMORY STATUS"): 
                            parsing_ram_flash = True
                            ram_flash_lines = [line] # Start new block
                            # Ensure other parsing states are false
                            parsing_perf_stats = False
                            perf_stats_lines = []
                            continue # Skip other checks for this line

                        elif line.startswith("📈 PERFORMANCE STATS:"): 
                            parsing_perf_stats = True
                            perf_stats_lines = [line] # Start new block
                            # Ensure other parsing states are false
                            parsing_ram_flash = False
                            ram_flash_lines = []
                            continue # Skip other checks for this line

                        # --- Single line message parsing ---
                        elif line.startswith("📊 SOC:"): 
                            match = re.search(r"SOC: ([\d\.-]+).*Time: (\d+) μs", line) # Allow negative SOC temporarily if model outputs it
                            if match:
                                soc_pred = float(match.group(1))
                                inf_time = int(match.group(2))
                                true_soc_for_pred = None
                                try:
                                    if self.ground_truth_data is not None and self.data_index > 0:
                                        # data_index was incremented *after* sample was fetched by background_prediction_loop
                                        # So, the true SOC corresponds to the (data_index - 1) sample
                                        gt_idx = (self.data_index - 1 + len(self.ground_truth_data)) % len(self.ground_truth_data)
                                        true_soc_for_pred = self.ground_truth_data['SOC_ZHU'].iloc[gt_idx]
                                except IndexError as ie:
                                    logger.warning(f"IndexError getting ground truth for prediction: {ie}, data_index: {self.data_index}")
                                except Exception as e_gt:
                                    logger.error(f"Error getting ground truth for prediction: {e_gt}")

                                self.arduino_data_queue.put({'type': 'prediction', 'soc': soc_pred, 'time': inf_time, 'soc_true': true_soc_for_pred})
                            else:
                                logger.warning(f"Could not parse prediction line: {line}")
                        elif "READY" in line or "Arduino LSTM SOC Predictor" in line or "Model:" in line or "Ready for predictions" in line: # Added "Ready for predictions"
                            logger.info(f"Arduino status: {line}")
                            self.arduino_ready_event.set() # Signal that Arduino is ready
                            # This is an informational message, not typically queued for processing
                        elif "Error:" in line or "CRITICAL:" in line or "WARNING:" in line : # Note: "WARNING:" might be too broad if Python logs to same stream
                             logger.warning(f"Arduino Msg: {line}") # Log it clearly as from Arduino
                        # else:
                        #    if line: logger.debug(f"Arduino unhandled: {line}") # For debugging unhandled lines
            except serial.SerialException as e:
                logger.error(f"Serial error in reader thread: {e}")
                self.arduino_connected = False
                break 
            except Exception as e:
                logger.error(f"Error in Arduino reader thread: {e}")
                # Potentially break or attempt to recover
            time.sleep(0.001) # Small sleep to prevent busy-waiting

        logger.info("Arduino output reader thread stopped.")

    def send_to_arduino(self, command_str):
        if self.arduino_connected and self.arduino and self.arduino.is_open:
            try:
                # logger.debug(f"Sending to Arduino: {command_str.strip()}")
                self.arduino.write(command_str.encode('utf-8'))
                self.arduino.flush() # Ensure data is sent
                return True
            except serial.SerialException as e:
                logger.error(f"Serial error sending command '{command_str.strip()}': {e}")
                self.arduino_connected = False
                return False
            except Exception as e:
                logger.error(f"Error sending command '{command_str.strip()}': {e}")
                return False
        else:
            # logger.warning(f"Cannot send command \'{command_str.strip()}\', Arduino not connected.")
            return False

    def predict_with_arduino(self, voltage, current, soh, q_c):
        """Sends data for prediction. Response handled by reader thread."""
        if not self.arduino_connected:
            self.communication_errors += 1
            return
        
        # Scaling here is for consistency if other parts of the Python script expect it.
        # For this project, Arduino expects raw values.
        # scaled_v, scaled_c, scaled_soh, scaled_qc = self.scale_features(voltage, current, soh, q_c)
        
        # Arduino sketch expects raw values for V, I, SOH, Q_c
        command = f"DATA:{voltage:.4f},{current:.4f},{soh:.4f},{q_c:.4f}\\n" # MODIFIED for SOH, Q_c and corrected newline
        if self.send_to_arduino(command):
            self.prediction_count += 1
        else:
            self.communication_errors += 1
            
    def request_periodic_stats(self):
        """Sends STATS and RAM commands to Arduino periodically."""
        current_time = time.time()
        if current_time - self.last_stats_command_time > STATS_COMMAND_INTERVAL:
            # logger.debug("Requesting RAM and STATS from Arduino")
            if not self.send_to_arduino("RAM\\n"): # MODIFIED for corrected newline
                 self.communication_errors += 1
            time.sleep(0.1) # Small delay between commands
            if not self.send_to_arduino("STATS\\n"): # MODIFIED for corrected newline
                 self.communication_errors += 1
            self.last_stats_command_time = current_time

    def background_prediction_loop(self):
        """Background thread für kontinuierliche Predictions und Stats-Anfragen"""
        logger.info("🔄 Background prediction and stats loop started")
        
        while self.prediction_running:
            v, i, soh_val, qc_val, soc_true = self.get_next_data_sample() # MODIFIED
            if v is not None: # Check if data is valid
                self.predict_with_arduino(v, i, soh_val, qc_val) # MODIFIED
                
                # Store for plotting (raw voltage, soh, qc)
                # SOC true is stored when processing queue data
                current_timestamp = time.time() - self.start_time
                self.timestamps.append(current_timestamp)
                self.voltage_values.append(v)
                self.soh_values.append(soh_val) # MODIFIED
                self.qc_values.append(qc_val)   # MODIFIED
            
            self.request_periodic_stats() # Send RAM/STATS commands periodically
            time.sleep(PREDICTION_DELAY / 1000.0) # Control prediction rate
        
        logger.info("Background prediction and stats loop stopped")

    def update_hardware_table(self):
        """Update hardware statistics table with Verbrauch/Gesamt format"""
        stats_lines = [
            "==== HARDWARE STATS (64x64) ===="
        ]
        
        # Inference Time
        if self.hardware_stats['inference_time_us']:
            avg_inf_time = np.mean(self.hardware_stats['inference_time_us'])
            stats_lines.append(f"🕒 Inf. Time: {avg_inf_time:.0f} µs (avg)")
        else:
            stats_lines.append("🕒 Inf. Time: N/A")

        # RAM Usage
        if self.hardware_stats['ram_used_kb'] and self.hardware_stats['ram_total_kb'] and self.hardware_stats['ram_usage_percent']:
            avg_ram_used = np.mean(self.hardware_stats['ram_used_kb'])
            avg_ram_total = np.mean(self.hardware_stats['ram_total_kb']) # Should be constant
            avg_ram_percent = np.mean(self.hardware_stats['ram_usage_percent'])
            stats_lines.append(f"🐏 RAM: {avg_ram_used:.1f}/{avg_ram_total:.0f} KB ({avg_ram_percent:.1f}%)")
        else:
            stats_lines.append("🐏 RAM: N/A")

        # Flash Usage
        if self.hardware_stats['flash_used_kb'] and self.hardware_stats['flash_total_kb'] and self.hardware_stats['flash_usage_percent']:
            avg_flash_used = np.mean(self.hardware_stats['flash_used_kb'])
            avg_flash_total = np.mean(self.hardware_stats['flash_total_kb']) # Should be constant
            avg_flash_percent = np.mean(self.hardware_stats['flash_usage_percent'])
            stats_lines.append(f"💿 Flash: {avg_flash_used:.0f}/{avg_flash_total:.0f} KB ({avg_flash_percent:.1f}%)")
        else:
            stats_lines.append("💿 Flash: N/A")
            
        # CPU Load
        if self.hardware_stats['cpu_load_percent']:
            avg_cpu_load = np.mean(self.hardware_stats['cpu_load_percent'])
            stats_lines.append(f"⚙️ CPU Load: {avg_cpu_load:.1f}%")
        else:
            stats_lines.append("⚙️ CPU Load: N/A")

        # Temperature
        if self.hardware_stats['temperature_celsius']:
            avg_temp = np.mean(self.hardware_stats['temperature_celsius'])
            stats_lines.append(f"🌡️ MCU Temp: {avg_temp:.1f}°C")
        else:
            stats_lines.append("🌡️ MCU Temp: N/A")
            
        # Communication Stats
        run_time = time.time() - self.start_time
        preds_per_sec = self.successful_predictions / run_time if run_time > 0 else 0
        stats_lines.append(f"📶 Preds: {self.successful_predictions}/{self.prediction_count} ({preds_per_sec:.1f}/s)")
        stats_lines.append(f"⚠️ Comm Errors: {self.communication_errors}")

        self.stats_text.set_text("\n".join(stats_lines))

    def process_queued_data(self):
        """Process data from the Arduino queue."""
        try:
            while not self.arduino_data_queue.empty():
                data_item = self.arduino_data_queue.get_nowait()
                
                if data_item['type'] == 'prediction':
                    # Clip SOC prediction to 0-1 range as it physically cannot be outside this
                    predicted_soc = np.clip(data_item['soc'], 0.0, 1.0)
                    self.soc_predictions.append(predicted_soc)
                    self.hardware_stats['inference_time_us'].append(data_item['time'])
                    self.successful_predictions += 1
                    
                    if data_item['soc_true'] is not None:
                        self.soc_ground_truth.append(float(data_item['soc_true']))
                    else:
                        # If true SOC is None, append NaN to keep lengths somewhat aligned for plotting, MAE will handle NaNs
                        self.soc_ground_truth.append(float('nan'))
                        logger.warning("soc_true was None for a prediction, MAE might be affected for this point.")

                    # MAE Calculation
                    # Only calculate MAE if we have valid corresponding ground truth
                    if not np.isnan(self.soc_ground_truth[-1]) and len(self.soc_predictions) > 0:
                        # Calculate MAE over the window where both are valid
                        min_len = min(len(self.soc_predictions), len(self.soc_ground_truth))
                        preds_arr = np.array(list(self.soc_predictions)[-min_len:])
                        gt_arr = np.array(list(self.soc_ground_truth)[-min_len:])
                        
                        valid_gt_indices = ~np.isnan(gt_arr)
                        if np.any(valid_gt_indices):
                            mae = np.mean(np.abs(preds_arr[valid_gt_indices] - gt_arr[valid_gt_indices])) * 100 # MAE as percentage
                            self.mae_errors.append(mae)
                        else:
                            self.mae_errors.append(float('nan')) # No valid GT points to compare
                    else:
                        self.mae_errors.append(float('nan')) # Append NaN if no valid GT for current prediction

                elif data_item['type'] == 'ram_flash':
                    for key in ['ram_used_kb', 'ram_total_kb', 'ram_usage_percent', 'flash_used_kb', 'flash_total_kb', 'flash_usage_percent']:
                        if data_item.get(key) is not None:
                            self.hardware_stats[key].append(data_item[key])
                        else:
                            # Append NaN or a default if a specific key is missing, to maintain deque length for averaging
                            self.hardware_stats[key].append(float('nan')) 
                            logger.debug(f"Missing {key} in ram_flash data: {data_item}")
                
                elif data_item['type'] == 'perf_stats':
                    for key in ['cpu_load_percent', 'temperature_celsius']:
                        if data_item.get(key) is not None:
                            self.hardware_stats[key].append(data_item[key])
                        else:
                            self.hardware_stats[key].append(float('nan'))
                            logger.debug(f"Missing {key} in perf_stats data: {data_item}")

        except queue.Empty:
            pass 
        except Exception as e:
            logger.error(f"Error processing queued data: {e}", exc_info=True)

    def update_plots(self, frame):
        """Update plots for performance"""
        self.process_queued_data() # Process any new data from Arduino

        # Ensure timestamps match data length for plotting
        common_len_soc = min(len(self.timestamps), len(self.soc_predictions), len(self.soc_ground_truth))
        common_len_voltage = min(len(self.timestamps), len(self.voltage_values))

        # Plot 1: SOC Prediction vs Ground Truth
        self.axes[0].clear()
        if common_len_soc > 0:
            ts_soc = list(self.timestamps)[-common_len_soc:]
            soc_p = list(self.soc_predictions)[-common_len_soc:]
            soc_gt = list(self.soc_ground_truth)[-common_len_soc:]
            self.axes[0].plot(ts_soc, soc_p, label='SOC Prediction (Arduino)', color='blue', alpha=0.8)
            self.axes[0].plot(ts_soc, soc_gt, label='SOC Ground Truth', color='green', linestyle='--', alpha=0.7)
            self.axes[0].legend(loc='upper right') # Moved inside conditional block
        
        self.axes[0].set_title('SOC Prediction vs. Ground Truth (64x64 Model)')
        self.axes[0].set_ylabel('SOC')
        self.axes[0].grid(True)
        self.axes[0].set_ylim(0, 1.1) # SOC range 0-1

        # Plot 2: Voltage (can add Temp or Power if needed)
        self.axes[1].clear()
        if common_len_voltage > 0:
            ts_voltage = list(self.timestamps)[-common_len_voltage:]
            voltages = list(self.voltage_values)[-common_len_voltage:]
            self.axes[1].plot(ts_voltage, voltages, label='Voltage', color='red', alpha=0.8)
            self.axes[1].legend(loc='upper right') # Moved inside conditional block

        self.axes[1].set_title('Input Voltage')
        self.axes[1].set_ylabel('Voltage [V]')
        self.axes[1].set_xlabel('Time [s]')
        self.axes[1].grid(True)

        # Add MAE to SOC plot title if available
        if self.mae_errors:
            current_mae = self.mae_errors[-1] if self.mae_errors else float('nan')
            self.axes[0].set_title(f'SOC Prediction vs. Ground Truth (64x64 Model) - MAE: {current_mae:.4f}')

        self.update_hardware_table()
        plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjust for suptitle and stats_text

        return self.axes[0], self.axes[1], self.stats_text
    
    def start_monitoring(self, auto_upload=True):
        logger.info("Starting Arduino SOC Monitor (64x64)...")
        if auto_upload:
            # The uploader might have its own boot wait after upload
            if not self.uploader.compile_and_upload(): # CORRECTED: Changed to compile_and_upload
                logger.warning("Continuing without successful sketch upload, or upload failed.")
        
        if not self.load_ground_truth_data():
            logger.error("Failed to load ground truth data. Monitoring cannot start effectively.")
            return

        self.prediction_running = True # Enable reader thread's main loop
        self.start_time = time.time()
        # self.arduino_ready_event.clear() # Moved to connect_arduino

        # Connect to Arduino. This method now starts the reader thread and waits for READY.
        if not self.connect_arduino():
            logger.error("Failed to connect to Arduino or Arduino did not signal READY. Monitoring cannot start.")
            self.prediction_running = False # Ensure reader thread (if started) will stop
            # Attempt to join reader_thread if it was started and connection failed
            if self.reader_thread and self.reader_thread.is_alive():
                logger.info("Attempting to stop reader thread after failed connection...")
                self.reader_thread.join(timeout=2.0)
                if self.reader_thread.is_alive():
                    logger.warning("Reader thread did not stop after failed connection.")
            return # Cannot run without Arduino

        # At this point, Arduino is connected, reader_thread is running, and Arduino has signaled READY.
        
        # The 0.5s sleep that was in connect_arduino after ready_event is effectively here now,
        # or can be re-added if specific timing for auto-sent messages is needed.
        # logger.info("Giving reader thread a moment (0.5s) to process any auto-sent messages post-READY...")
        # time.sleep(0.5) # This was moved into connect_arduino, can be removed here if redundant

        logger.info("Requesting initial RAM and STATS from Arduino (or refreshing)...")
        if not self.send_to_arduino("RAM\\\\n"): # Corrected to RAM\\n
            self.communication_errors += 1
            logger.warning("Failed to send initial RAM command.")
        time.sleep(0.1) # Small delay
        if not self.send_to_arduino("STATS\\\\n"): # Corrected to STATS\\n
            self.communication_errors += 1
            logger.warning("Failed to send initial STATS command.")
        self.last_stats_command_time = time.time() # Reset timer for periodic stats

        # Wait for initial hardware stats to be processed by the reader thread
        logger.info("Waiting for initial hardware stats to be processed (timeout: 7s)...")
        stats_received_timeout = time.time() + 7.0 
        initial_stats_processed = False
        while time.time() < stats_received_timeout:
            self.process_queued_data() 
            ram_info_received = bool(self.hardware_stats.get('ram_total_kb') and len(self.hardware_stats['ram_total_kb']) > 0)
            perf_info_received = bool(self.hardware_stats.get('cpu_load_percent') and len(self.hardware_stats['cpu_load_percent']) > 0)

            if ram_info_received and perf_info_received:
                logger.info("✅ Initial RAM and Performance stats processed.")
                initial_stats_processed = True
                break
            elif ram_info_received and not perf_info_received:
                logger.info("...RAM info received, waiting for Performance stats...")
            elif not ram_info_received and perf_info_received:
                logger.info("...Performance stats received, waiting for RAM info...")
            
            time.sleep(0.25) 

        if not initial_stats_processed:
            logger.warning("⚠️ Timed out waiting for initial hardware stats to be fully processed. Table might be incomplete initially.")

        # Start the background prediction loop 
        if self.prediction_loop_thread is None or not self.prediction_loop_thread.is_alive():
            self.prediction_loop_thread = threading.Thread(target=self.background_prediction_loop, daemon=True)
            self.prediction_loop_thread.start()
        else:
            logger.warning("Prediction loop thread already running. This might indicate an issue with start/stop logic if called multiple times.")


        logger.info("Threads started. Initializing plot...")
        try:
            # Make sure self.fig is valid
            if self.fig is None:
                logger.error("Figure object is None. Cannot start animation.")
                return

            ani = FuncAnimation(self.fig, self.update_plots, interval=UPDATE_INTERVAL, cache_frame_data=False, blit=False)
            plt.show()
        except Exception as e:
            logger.error(f"Error during plotting: {e}", exc_info=True)
        finally:
            self.stop_monitoring()
            logger.info("Monitoring stopped.")

    def stop_monitoring(self):
        logger.info("Stopping monitoring...")
        self.prediction_running = False

        # Wait for threads to finish
        if self.reader_thread and self.reader_thread.is_alive():
            logger.info("Waiting for reader thread to stop...")
            self.reader_thread.join(timeout=2)
            if self.reader_thread.is_alive():
                logger.warning("Reader thread did not stop in time.")
        
        if self.prediction_loop_thread and self.prediction_loop_thread.is_alive():
            logger.info("Waiting for prediction loop thread to stop...")
            self.prediction_loop_thread.join(timeout=2)
            if self.prediction_loop_thread.is_alive():
                logger.warning("Prediction loop thread did not stop in time.")

        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.close()
                logger.info("Arduino serial port closed.")
            except Exception as e:
                logger.error(f"Error closing Arduino port: {e}")
        self.arduino_connected = False
        
        logger.info("Monitoring has been shut down.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Arduino SOC Monitor V2 mit REALEN Flash-Werten (64x64 Model)')
    parser.add_argument('--port', '-P', default=ARDUINO_PORT, help=f'Serial port (default: {ARDUINO_PORT})') # Changed -p to -P to avoid conflict
    parser.add_argument('--baudrate', '-b', type=int, default=BAUDRATE, help=f'Baud rate (default: {BAUDRATE})')
    parser.add_argument('--no-upload', action='store_true', help='Skip automatic upload')
    parser.add_argument('--sketch', '-s', default=ARDUINO_SKETCH_PATH, help='Arduino sketch path')
    parser.add_argument('--fqbn', '-f', default=ARDUINO_FQBN, help='Arduino FQBN')
    parser.add_argument('--cli-path', '-c', default=ARDUINO_CLI_PATH, help='Path to arduino-cli executable')
    parser.add_argument('--data-path', '-d', default=DATA_PATH, help='Path to ground truth data parquet file')


    args = parser.parse_args()
    
    monitor = ArduinoSOCMonitor64(
        port=args.port, 
        baudrate=args.baudrate, 
        data_path=args.data_path,  # Use arg for data_path
        sketch_path=args.sketch, 
        fqbn=args.fqbn,
        cli_path=args.cli_path # Pass cli_path
    )
    
    try:
        monitor.start_monitoring(auto_upload=not args.no_upload)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Shutting down...")
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}", exc_info=True)
    finally:
        monitor.stop_monitoring()
        logger.info("Application terminated.")

if __name__ == "__main__":
    main()
