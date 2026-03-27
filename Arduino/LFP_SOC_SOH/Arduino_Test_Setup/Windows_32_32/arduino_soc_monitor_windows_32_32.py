"""
ARDUINO SOC MONITOR - WINDOW-BASED LSTM V2 (aligned with Stateful Monitor)
=========================================================================

✨ FEATURES:
- Aligned with the structure of 'arduino_soc_monitor_sf_32_32.py'.
- Attempts to automatically compile and upload the Arduino sketch.
- Live SOC Monitoring with Ground Truth Vergleich.
- Hardware parameter display (RAM, Flash, CPU, Temp).

🔴 IMPORTANT CONSIDERATIONS FOR WINDOW-BASED MODEL:
1.  ARDUINO SKETCH MODIFICATION:
    The corresponding Arduino sketch ('arduino_lstm_soc_windows_32_32.ino')
    MUST be modified to:
    a. Receive a full window of 'WINDOW_SIZE * INPUT_SIZE' float values serially.
       Example: "float1,float2,float3,float4,float5,..." (WINDOW_SIZE * INPUT_SIZE total floats)
    b. Store this window in an appropriate buffer.
    c. Perform prediction using this received window when commanded.
    d. Send back the SOC prediction and other stats.
       Example Arduino reply: "SOC:0.75,TIME:12345,RAM_FREE:1024,CPU:50,TEMP:25\n"
    e. Optionally, respond to a "GET_HW_STATS\n" command for static compiler stats if needed,
       though this Python script now extracts compiler stats itself.

2.  RAM LIMITATIONS ON ARDUINO:
    The 'WINDOW_SIZE' constant (default 50) determines how much data is sent.
    Storing a window (e.g., 50 * 4 floats = 200 floats * 4 bytes/float = 800 bytes)
    plus model weights and activations requires careful RAM management on Arduino.
    A `WINDOW_SIZE` of 5000 (as in the original) would be ~80KB for data alone, far too much.

3.  SERIAL PROTOCOL:
    - Python sends data string: "f1,f2,f3,f4,...,fN\n" (N = WINDOW_SIZE * INPUT_SIZE floats)
    - Arduino replies after prediction: "SOC:value,TIME:value,RAM_FREE:value,CPU:value,TEMP:value\n"

🚀 PERFORMANCE OPTIMIERT FÜR LIVE MONITORING (within Python limits) 🚀
"""
# Standard library imports
import argparse
import logging
import pickle
import re
import subprocess
import threading
import time
import warnings
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
WINDOW_SIZE = 50 # CORRECTED FROM 5000
INPUT_SIZE = 4
PLOT_WINDOW_SIZE = 200
UPDATE_INTERVAL = 500 # Plot update in ms
PREDICTION_DELAY_MS = 100 # Min delay between sending windows to Arduino

# CMD_GET_HW_STATS = "GET_HW_STATS" # If implementing separate hardware stat request to Arduino

class ArduinoUploader:
    """Handles automatic Arduino sketch compilation and upload. (Adapted from sf_32_32)"""
    def __init__(self, sketch_path, fqbn, port, cli_path=None):
        self.sketch_path = Path(sketch_path)
        self.fqbn = fqbn
        self.port = port
        self.cli_path = cli_path or ARDUINO_CLI_PATH
        self.arduino_cli_available = self.check_arduino_cli()

    def check_arduino_cli(self):
        """Check if arduino-cli is available at the specified path or system PATH"""
        try:
            # First try the specified path
            if Path(self.cli_path).exists() and self.cli_path != 'arduino-cli': # Avoid double check if already 'arduino-cli'
                result = subprocess.run([str(self.cli_path), 'version'], capture_output=True, text=True, timeout=10, check=True)
                logger.info(f"✅ Arduino CLI found at specified path: {self.cli_path} ({result.stdout.strip()})")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.warning(f"Arduino CLI at specified path '{self.cli_path}' failed or not found: {e}")
        
        # Fallback to system PATH if specified path failed or was 'arduino-cli'
        try:
            result = subprocess.run(['arduino-cli', 'version'], capture_output=True, text=True, timeout=10, check=True)
            logger.info(f"✅ Arduino CLI found in system PATH: {result.stdout.strip()}")
            self.cli_path = 'arduino-cli'  # Use system PATH version
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e_path:
            logger.error(f"❌ Arduino CLI also not found in system PATH or check failed: {e_path}")
            return False


    def compile_and_upload(self):
        """Compile and upload the sketch. Returns (success, flash_bytes, ram_bytes_compiler)"""
        if not self.arduino_cli_available:
            logger.error("Cannot compile/upload: Arduino CLI not available or not working.")
            return False, None, None

        if not self.sketch_path.exists():
            logger.error(f"Sketch file not found: {self.sketch_path}")
            return False, None, None
        
        sketch_dir = self.sketch_path.parent
        
        try:
            logger.info(f"🔨 Compiling sketch: {self.sketch_path} for board {self.fqbn}")
            compile_cmd = [str(self.cli_path), "compile", "--fqbn", self.fqbn, str(sketch_dir)]
            compile_process = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120, check=True)
            logger.info("✅ Sketch compiled successfully.")
            # stderr often contains the compiler size summary
            compile_output_for_parsing = compile_process.stdout + compile_process.stderr
            logger.debug(f"Compile output for parsing:\n{compile_output_for_parsing}")
            
            flash_used, ram_used_compiler = self.parse_compiler_output(compile_output_for_parsing)

            logger.info(f"⬆️ Uploading sketch to {self.port}...")
            upload_cmd = [str(self.cli_path), "upload", "-p", self.port, "--fqbn", self.fqbn, str(sketch_dir)]
            upload_process = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=60, check=True)
            logger.info("✅ Sketch uploaded successfully.")
            logger.debug(f"Upload output:\n{upload_process.stdout}{upload_process.stderr}")
            return True, flash_used, ram_used_compiler

        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout during Arduino CLI operation: {e}")
            stdout_data = e.stdout.decode(errors='ignore') if isinstance(e.stdout, bytes) else e.stdout
            stderr_data = e.stderr.decode(errors='ignore') if isinstance(e.stderr, bytes) else e.stderr
            logger.error(f"Output: {stdout_data}")
            logger.error(f"Error: {stderr_data}")
            return False, None, None
        except subprocess.CalledProcessError as e:
            logger.error(f"Arduino CLI command failed (return code {e.returncode}): {e.cmd}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            return False, None, None
        except Exception as e:
            logger.error(f"An unexpected error occurred during compile/upload: {e}", exc_info=True)
            return False, None, None

    def parse_compiler_output(self, output):
        flash_used = None
        ram_used = None
        try:
            # Using regex for more robust parsing, similar to stateful monitor
            flash_match = re.search(r"Sketch uses (\\d+) bytes .* of program storage space.", output)
            if flash_match:
                flash_used = int(flash_match.group(1))
                logger.info(f"Extracted Flash used from compiler: {flash_used} bytes")

            ram_match = re.search(r"Global variables use (\\d+) bytes .* of dynamic memory", output)
            if ram_match:
                ram_used = int(ram_match.group(1))
                logger.info(f"Extracted RAM used from compiler: {ram_used} bytes")
            
            if flash_used is None or ram_used is None: # Check if any failed
                logger.warning(f"Could not parse compiler output fully for flash/RAM. Output snippet:\\n{output[:500]}")

        except Exception as e:
            logger.warning(f"Could not parse compiler output for flash/RAM: {e}")
        return flash_used, ram_used


class ArduinoSOCMonitorWindow:
    """Arduino SOC Monitor for Window-Based LSTM (aligned with sf_32_32)"""
    def __init__(self, port, baudrate, data_path, sketch_path, fqbn):
        self.port = port
        self.baudrate = baudrate
        self.data_path = Path(data_path)
        self.sketch_path = Path(sketch_path)
        self.fqbn = fqbn
        self.uploader = ArduinoUploader(str(self.sketch_path), self.fqbn, self.port, ARDUINO_CLI_PATH)

        self.arduino = None
        self.arduino_connected = False
        self.prediction_thread = None
        self.stop_event = threading.Event()
        self.prediction_running = False
        
        self.timestamps = deque(maxlen=PLOT_WINDOW_SIZE)
        self.voltage_values = deque(maxlen=PLOT_WINDOW_SIZE) # Voltage at end of window for plotting
        self.soc_predictions = deque(maxlen=PLOT_WINDOW_SIZE)
        self.soc_ground_truth = deque(maxlen=PLOT_WINDOW_SIZE)
        self.mae_errors = deque(maxlen=PLOT_WINDOW_SIZE)
        
        # self.current_window_data_buffer = [] # Not needed if get_next_data_window directly provides the window

        self.hardware_stats = {
            'inference_time_us': deque(maxlen=100),
            'ram_free_bytes_rt': deque(maxlen=100), 
            'cpu_load_percent_rt': deque(maxlen=100), 
            'temperature_celsius_rt': deque(maxlen=100),
            'ram_used_bytes_compiler': None, 
            'ram_total_bytes_board': 32784, # Arduino Uno R4 WiFi
            'flash_used_bytes_compiler': None,
            'flash_total_bytes_board': 262144, # Arduino Uno R4 WiFi
        }
        self.stats_text_content = {} # For easy update of the table text

        self.ground_truth_df = None
        self.scaler = None 
        self.data_index = 0 # Current starting index for reading a window from ground_truth_df

        self.prediction_count = 0
        self.successful_predictions = 0
        self.communication_errors = 0
        self.start_time = time.time()

        self.fig, self.axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})
        self.fig.suptitle(f'🚀 Arduino Window-Based LSTM SOC Monitor (Input: {INPUT_SIZE}, Win: {WINDOW_SIZE})', fontsize=16, fontweight='bold')
        plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.25, hspace=0.35) # Adjusted bottom and hspace

        self.setup_plots()
        self.setup_hardware_table()
        
        logger.info(f"Window-Based SOC Monitor initialized. WINDOW_SIZE={WINDOW_SIZE}. Ensure Arduino sketch is compatible.")

    def setup_plots(self):
        # SOC Plot
        self.axes[0].set_title("SOC Prediction vs. Ground Truth", fontsize=12)
        self.line_soc_pred, = self.axes[0].plot([], [], 'r-', label=f'Arduino SOC Pred. (Win={WINDOW_SIZE})', alpha=0.8)
        self.line_soc_gt, = self.axes[0].plot([], [], 'b--', label='Ground Truth SOC', alpha=0.7)
        self.line_voltage, = self.axes[0].plot([], [], 'g:', label='Voltage (V) @ Win End', alpha=0.5) 
        self.axes[0].set_ylabel("SOC / Voltage")
        self.axes[0].legend(loc='upper left')
        self.axes[0].grid(True, linestyle=':', alpha=0.7)
        # self.axes[0].set_ylim(-0.1, 1.1) # For SOC, voltage might be higher

        # MAE Plot
        self.axes[1].set_title("Mean Absolute Error (MAE)", fontsize=12)
        self.line_mae, = self.axes[1].plot([], [], 'm-', label='MAE')
        self.axes[1].set_xlabel("Time (s since start of plot)")
        self.axes[1].set_ylabel("MAE")
        self.axes[1].legend(loc='upper left')
        self.axes[1].grid(True, linestyle=':', alpha=0.7)
        self.axes[1].set_ylim(0, 0.1)

    def setup_hardware_table(self):
        self.stats_text = self.fig.text(0.05, 0.02, "Initializing hardware stats...", fontsize=8, fontfamily='monospace',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    def get_stats_string(self):
        self.stats_text_content['inf_time'] = f"{np.mean(self.hardware_stats['inference_time_us']):.0f}" if self.hardware_stats['inference_time_us'] else "N/A"
        self.stats_text_content['cpu_load_rt'] = f"{np.mean(self.hardware_stats['cpu_load_percent_rt']):.1f}" if self.hardware_stats['cpu_load_percent_rt'] else "N/A"
        self.stats_text_content['mcu_temp_rt'] = f"{np.mean(self.hardware_stats['temperature_celsius_rt']):.1f}" if self.hardware_stats['temperature_celsius_rt'] else "N/A"
        
        flash_total_kb = self.hardware_stats['flash_total_bytes_board'] / 1024
        if self.hardware_stats['flash_used_bytes_compiler'] is not None:
            flash_used_kb = self.hardware_stats['flash_used_bytes_compiler'] / 1024
            flash_perc = (self.hardware_stats['flash_used_bytes_compiler'] / self.hardware_stats['flash_total_bytes_board'] * 100)
            self.stats_text_content['flash_used_comp'] = f"{flash_used_kb:.1f}"
            self.stats_text_content['flash_perc_comp'] = f"{flash_perc:.1f}"
        else:
            self.stats_text_content['flash_used_comp'] = "N/A"
            self.stats_text_content['flash_perc_comp'] = "N/A"

        ram_total_kb = self.hardware_stats['ram_total_bytes_board'] / 1024
        if self.hardware_stats['ram_used_bytes_compiler'] is not None:
            ram_used_kb = self.hardware_stats['ram_used_bytes_compiler'] / 1024
            ram_perc = (self.hardware_stats['ram_used_bytes_compiler'] / self.hardware_stats['ram_total_bytes_board'] * 100)
            self.stats_text_content['ram_used_comp'] = f"{ram_used_kb:.1f}"
            self.stats_text_content['ram_perc_comp'] = f"{ram_perc:.1f}"
        else:
            self.stats_text_content['ram_used_comp'] = "N/A"
            self.stats_text_content['ram_perc_comp'] = "N/A"
            
        ram_free_rt_kb = np.mean(self.hardware_stats['ram_free_bytes_rt']) / 1024 if self.hardware_stats['ram_free_bytes_rt'] else float('nan')
        self.stats_text_content['ram_free_rt'] = f"{ram_free_rt_kb:.1f}" if not np.isnan(ram_free_rt_kb) else "N/A"

        lines = [
            "--- Arduino Window-Based LSTM Hardware Stats ---",
            f" WinSize: {WINDOW_SIZE}, Inputs/step: {INPUT_SIZE}",
            f" Inference Time (avg): {self.stats_text_content.get('inf_time', 'N/A')} µs",
            f" CPU Load (runtime):   {self.stats_text_content.get('cpu_load_rt', 'N/A')} %",
            f" MCU Temp (runtime):   {self.stats_text_content.get('mcu_temp_rt', 'N/A')} °C",
            "--- Memory (Compiler Estimates) ---",
            f" Flash Used: {self.stats_text_content.get('flash_used_comp', 'N/A')} KB ({self.stats_text_content.get('flash_perc_comp', 'N/A')} % of {flash_total_kb:.0f}KB)",
            f" RAM Globals: {self.stats_text_content.get('ram_used_comp', 'N/A')} KB ({self.stats_text_content.get('ram_perc_comp', 'N/A')} % of {ram_total_kb:.0f}KB)",
            "--- Memory (Runtime - from Arduino) ---",
            f" RAM Free:   {self.stats_text_content.get('ram_free_rt', 'N/A')} KB",
            "--- Performance ---",
            f" Preds (OK/Total): {self.successful_predictions}/{self.prediction_count}",
            f" Comm Errors:    {self.communication_errors}",
            f" Uptime:         {time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))}"
        ]
        return "\n".join(lines)

    def upload_arduino_sketch(self):
        logger.info("🚀 Attempting automatic Arduino sketch upload...")
        success, flash_bytes, ram_bytes_compiler = self.uploader.compile_and_upload()
        if success:
            logger.info("✅ Sketch upload appears successful.")
            if flash_bytes is not None:
                self.hardware_stats['flash_used_bytes_compiler'] = flash_bytes
            if ram_bytes_compiler is not None:
                self.hardware_stats['ram_used_bytes_compiler'] = ram_bytes_compiler
            return True
        else:
            logger.error("❌ Sketch upload failed. Check Arduino CLI output and settings.")
            return False

    def load_ground_truth_data(self):
        logger.info(f"Loading ground truth data from: {self.data_path}")
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return False
        try:
            self.ground_truth_df = pd.read_parquet(self.data_path)
            required_cols = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m", "SOC_ZHU", "Absolute_Time[yyyy-mm-dd hh:mm:ss]"]
            if not all(col in self.ground_truth_df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in self.ground_truth_df.columns]
                logger.error(f"Missing one or more required columns in {self.data_path}. Required: {required_cols}, Missing: {missing}")
                return False
            
            self.ground_truth_df['timestamp'] = pd.to_datetime(self.ground_truth_df['Absolute_Time[yyyy-mm-dd hh:mm:ss]']).astype(np.int64) // 10**9 
            self.ground_truth_df.dropna(subset=["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m", "SOC_ZHU"], inplace=True)
            self.ground_truth_df = self.ground_truth_df.reset_index(drop=True) # Ensure clean index after dropna
            logger.info(f"Ground truth data loaded. Shape: {self.ground_truth_df.shape}")
            if len(self.ground_truth_df) < WINDOW_SIZE:
                logger.error(f"Not enough data ({len(self.ground_truth_df)} points) for even one window of size {WINDOW_SIZE}.")
                return False
            return True
        except Exception as e:
            logger.error(f"Error loading ground truth data: {e}", exc_info=True)
            return False

    def setup_scaler(self):
        logger.info("Setting up scaler using the loaded ground truth data.")
        if self.ground_truth_df is None:
            logger.error("Ground truth data not loaded. Cannot setup scaler.")
            return False
        
        scaler_filename = "scaler_windows_v_c_soh_qm.pkl" # Specific for these features
        scaler_path = self.data_path.parent / scaler_filename
        
        try:
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✅ Loaded existing scaler from {scaler_path}")
            else:
                features_to_scale = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]
                self.scaler = StandardScaler()
                self.scaler.fit(self.ground_truth_df[features_to_scale])
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info(f"✅ Scaler fitted and saved to {scaler_path}.")
            return True
        except Exception as e:
            logger.error(f"Error setting up scaler: {e}", exc_info=True)
            return False

    def scale_features(self, features_np_array): 
        if self.scaler is None:
            logger.warning("Scaler not set up. Returning unscaled features.")
            return features_np_array
        try:
            return self.scaler.transform(features_np_array)
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return features_np_array

    def get_next_data_window(self):
        if self.ground_truth_df is None or (self.data_index + WINDOW_SIZE) > len(self.ground_truth_df):
            logger.info("End of dataset reached or not enough data for a full window. Resetting data_index.")
            self.data_index = 0 
            if (self.data_index + WINDOW_SIZE) > len(self.ground_truth_df):
                 logger.error("Dataset too small for even one window after reset. Halting data provision.")
                 return None, None, None, None # Scaled features, SOC_gt, timestamp, voltage_gt

        window_df = self.ground_truth_df.iloc[self.data_index : self.data_index + WINDOW_SIZE]
        
        # For continuous monitoring, advance index by 1 (sliding window). 
        # For distinct windows, advance by WINDOW_SIZE.
        self.data_index += 1 

        features_for_scaling = window_df[["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]].values
        scaled_features_window = self.scale_features(features_for_scaling) 
        
        soc_gt_for_window = window_df["SOC_ZHU"].iloc[-1] # SOC at the end of the window
        timestamp_at_end_of_window = window_df["timestamp"].iloc[-1]
        voltage_at_end_of_window = window_df["Voltage[V]"].iloc[-1] # For plotting context

        return scaled_features_window, soc_gt_for_window, timestamp_at_end_of_window, voltage_at_end_of_window

    def connect_arduino(self):
        logger.info(f"Attempting to connect to Arduino on {self.port} at {self.baudrate} baud.")
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2.5) # Allow Arduino to initialize/reset fully
            self.arduino.flushInput()
            self.arduino.flushOutput()
            
            # Send a PING or a specific ready command expected by the Arduino sketch
            self.arduino.write(b"PING\n") 
            response_bytes = self.arduino.readline()
            startup_msg = response_bytes.decode('utf-8', errors='ignore').strip()
            
            # Example ready messages from Arduino: "PONG", "READY", "WINDOW_LSTM_READY"
            if "PONG" in startup_msg or "READY" in startup_msg or "WINDOW_LSTM_READY" in startup_msg or "Window LSTM Ready" in startup_msg:
                 logger.info(f"✅ Arduino connected successfully. Startup/Ping response: '{startup_msg}'")
                 self.arduino_connected = True
                 return True
            elif len(startup_msg) > 0 : # Received something, but not the expected ready message
                logger.warning(f"Arduino connected, but unexpected startup/ping response: '{startup_msg}'. Assuming connected and proceeding.")
                self.arduino_connected = True 
                return True
            else: # Timeout, no response
                logger.error(f"Arduino connection attempt timed out. No response to PING. Check sketch and connections.")
                self.arduino.close()
                self.arduino_connected = False
                return False

        except serial.SerialException as e:
            logger.error(f"❌ Failed to connect to Arduino (SerialException): {e}")
            self.arduino_connected = False
            return False
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during Arduino connection: {e}", exc_info=True)
            self.arduino_connected = False
            return False

    def send_window_to_arduino_and_get_pred(self, scaled_window_data):
        if not self.arduino_connected or not self.arduino:
            logger.warning("Arduino not connected. Cannot send window.")
            self.communication_errors +=1
            return None

        self.prediction_count += 1
        try:
            # scaled_window_data is already a 2D numpy array (WINDOW_SIZE, INPUT_SIZE)
            flat_list = scaled_window_data.flatten().tolist()
            data_string = ",".join([f"{x:.6f}" for x in flat_list]) + "\n"
            
            # logger.debug(f"Sending data string ({len(flat_list)} floats, {len(data_string)} chars): {data_string[:100]}...")
            self.arduino.write(data_string.encode('utf-8'))
            self.arduino.flush() 

            response_bytes = self.arduino.readline()
            response = response_bytes.decode('utf-8', errors='ignore').strip()
            logger.debug(f"Arduino response: '{response}'")

            if response and response.startswith("SOC:"):
                parts = response.split(',')
                soc_pred = float(parts[0].split(':')[1])
                
                inf_time = int(parts[1].split(':')[1]) if len(parts) > 1 and "TIME:" in parts[1] else -1
                ram_free_rt = int(parts[2].split(':')[1]) if len(parts) > 2 and "RAM_FREE:" in parts[2] else -1
                cpu_load_rt = int(parts[3].split(':')[1]) if len(parts) > 3 and "CPU:" in parts[3] else -1 # Assuming CPU is int
                temp_rt = float(parts[4].split(':')[1]) if len(parts) > 4 and "TEMP:" in parts[4] else -1.0

                self.hardware_stats['inference_time_us'].append(inf_time)
                if ram_free_rt != -1: self.hardware_stats['ram_free_bytes_rt'].append(ram_free_rt)
                if cpu_load_rt != -1: self.hardware_stats['cpu_load_percent_rt'].append(cpu_load_rt)
                if temp_rt != -1.0: self.hardware_stats['temperature_celsius_rt'].append(temp_rt)
                
                self.successful_predictions += 1
                return soc_pred
            else:
                logger.warning(f"Unexpected response from Arduino after sending window: '{response}'")
                self.communication_errors += 1
                return None

        except Exception as e:
            logger.error(f"Error during window prediction with Arduino: {e}", exc_info=True)
            self.communication_errors += 1
            if self.arduino: # Try to clear buffers if error
                try:
                    self.arduino.flushInput()
                    self.arduino.flushOutput()
                except: pass # Ignore errors during flush on already problematic port
            return None

    def background_prediction_loop(self):
        self.prediction_running = True
        logger.info("Background prediction loop started.")
        
        last_prediction_send_time = time.time()

        while not self.stop_event.is_set():
            scaled_window, soc_gt, timestamp, voltage_gt = self.get_next_data_window()
            
            if scaled_window is None: 
                logger.info("No more data or error in data window generation. Pausing loop for 1s.")
                time.sleep(1) # Pause if no data, then try again (e.g., if data_index was reset)
                continue

            current_time = time.time()
            time_since_last_send = (current_time - last_prediction_send_time) * 1000
            
            if time_since_last_send < PREDICTION_DELAY_MS:
                sleep_duration = (PREDICTION_DELAY_MS - time_since_last_send) / 1000.0
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
            
            last_prediction_send_time = time.time()
            soc_pred = self.send_window_to_arduino_and_get_pred(scaled_window)

            if soc_pred is not None:
                self.timestamps.append(timestamp) 
                self.voltage_values.append(voltage_gt) 
                self.soc_predictions.append(soc_pred)
                self.soc_ground_truth.append(soc_gt)
                mae = abs(soc_pred - soc_gt)
                self.mae_errors.append(mae)
            # No sleep here, PREDICTION_DELAY_MS handles rate limiting

        self.prediction_running = False
        logger.info("Background prediction loop stopped.")

    def update_plots(self, frame):
        if not self.prediction_running and not list(self.timestamps):
            return self.line_soc_pred, self.line_soc_gt, self.line_voltage, self.line_mae, self.stats_text
        
        plot_times = np.array(list(self.timestamps))
        if len(plot_times) > 0:
            # Create relative time axis for the plot, starting from 0 for the current view
            current_plot_start_time = plot_times[0]
            relative_plot_times = plot_times - current_plot_start_time
        else:
            relative_plot_times = []

        self.line_soc_pred.set_data(relative_plot_times, list(self.soc_predictions))
        self.line_soc_gt.set_data(relative_plot_times, list(self.soc_ground_truth))
        self.line_voltage.set_data(relative_plot_times, list(self.voltage_values))
        self.line_mae.set_data(relative_plot_times, list(self.mae_errors))

        # SOC and Voltage Plot Y-axis
        all_soc_volt_data = list(self.soc_predictions) + list(self.soc_ground_truth) + list(self.voltage_values)
        if all_soc_volt_data:
            min_val_sv = min(all_soc_volt_data)
            max_val_sv = max(all_soc_volt_data)
            self.axes[0].set_ylim(min(min_val_sv * 0.95, -0.1), max(max_val_sv * 1.05, 1.1, 3.8 if max_val_sv > 2 else 1.1) ) # Ensure SOC 0-1 is visible, and typical voltage
        else:
            self.axes[0].set_ylim(-0.1, 1.1)
        self.axes[0].relim()
        self.axes[0].autoscale_view(scalex=True, scaley=False) # Autoscale X, Y is manual

        # MAE Plot Y-axis
        if self.mae_errors:
            max_mae = max(self.mae_errors)
            self.axes[1].set_ylim(0, max(0.1, max_mae * 1.2))
        else:
            self.axes[1].set_ylim(0, 0.1)
        self.axes[1].relim()
        self.axes[1].autoscale_view(scalex=True, scaley=False) # Autoscale X, Y is manual
        
        self.stats_text.set_text(self.get_stats_string())
        
        return self.line_soc_pred, self.line_soc_gt, self.line_voltage, self.line_mae, self.stats_text

    def start_monitoring(self, auto_upload=True):
        logger.info("🚀 Starting Window-Based SOC monitoring...")
        self.start_time = time.time()

        if auto_upload:
            if not self.upload_arduino_sketch():
                logger.error("Sketch upload failed.")
                cont = input("Sketch upload failed. Continue with current sketch on Arduino? (y/n): ")
                if cont.lower() != 'y':
                    logger.info("Halting due to failed sketch upload and user choice.")
                    return
                logger.info("Continuing with potentially outdated sketch on Arduino.")
            else: 
                logger.info("Waiting 5 seconds after upload for Arduino to initialize...")
                time.sleep(5)
        else:
            logger.info("Skipping automatic sketch upload as per user request.")

        if not self.load_ground_truth_data(): return
        if not self.setup_scaler(): return
        if not self.connect_arduino(): 
            logger.error("Failed to connect to Arduino. Please check connection, port, and ensure sketch is running.")
            return

        self.stop_event.clear()
        self.prediction_thread = threading.Thread(target=self.background_prediction_loop, daemon=True)
        self.prediction_thread.start()
        
        ani = animation.FuncAnimation(self.fig, self.update_plots, interval=UPDATE_INTERVAL, blit=True, cache_seq_data=False)
        try:
            plt.show()
        except Exception as e:
            logger.error(f"Error during matplotlib show: {e}", exc_info=True)

        logger.info("Plot window closed or error in display.")
        self.stop_monitoring()

    def stop_monitoring(self):
        logger.info("Stopping SOC monitoring...")
        self.stop_event.set()
        if self.prediction_thread and self.prediction_thread.is_alive():
            logger.info("Waiting for prediction thread to finish...")
            self.prediction_thread.join(timeout=5)
            if self.prediction_thread.is_alive():
                logger.warning("Prediction thread did not finish in time.")
        
        if self.arduino and self.arduino.is_open:
            logger.info("Closing Arduino connection.")
            try:
                self.arduino.close()
            except Exception as e:
                logger.error(f"Error closing Arduino port: {e}")
        self.arduino_connected = False
        logger.info("Monitoring stopped.")

def main():
    global WINDOW_SIZE # Moved to the top of the function
    parser = argparse.ArgumentParser(description=f'Arduino Window-Based LSTM SOC Monitor (Default WinSize: {WINDOW_SIZE})')
    parser.add_argument('--port', '-P', default=ARDUINO_PORT, help=f'Serial port (default: {ARDUINO_PORT})')
    parser.add_argument('--baudrate', '-b', type=int, default=BAUDRATE, help=f'Baud rate (default: {BAUDRATE})')
    parser.add_argument('--no-upload', action='store_true', help='Skip automatic Arduino sketch upload')
    parser.add_argument('--sketch', '-s', default=ARDUINO_SKETCH_PATH, help=f'Path to Arduino sketch .ino file (default: {ARDUINO_SKETCH_PATH})')
    parser.add_argument('--fqbn', '-F', default=ARDUINO_FQBN, help=f'Arduino FQBN (default: {ARDUINO_FQBN})')
    parser.add_argument('--data', '-d', default=DATA_PATH, help=f'Path to ground truth data .parquet file (default: {DATA_PATH})')
    parser.add_argument('--window-size', '-w', type=int, default=WINDOW_SIZE, help=f'Window size for LSTM (default: {WINDOW_SIZE})')
    
    args = parser.parse_args()
    
    # Update global WINDOW_SIZE if provided via CLI, as it's used in class suptitle and other places
    WINDOW_SIZE = args.window_size

    logger.info(f"--- Starting Window-Based SOC Monitor (V2 Aligned) ---")
    logger.info(f"Port: {args.port}, Baudrate: {args.baudrate}")
    logger.info(f"Sketch: {args.sketch}, FQBN: {args.fqbn}")
    logger.info(f"Ground Truth Data: {args.data}")
    logger.info(f"Effective WINDOW_SIZE: {WINDOW_SIZE} (🔴 Ensure Arduino sketch matches and can handle this RAM!)")
    logger.info(f"INPUT_SIZE (fixed): {INPUT_SIZE}")

    monitor = None # Initialize monitor to None for finally block
    try:
        monitor = ArduinoSOCMonitorWindow(
            port=args.port,
            baudrate=args.baudrate,
            data_path=args.data,
            sketch_path=args.sketch,
            fqbn=args.fqbn
            # WINDOW_SIZE is a global, accessed by the class instance
        )
        monitor.start_monitoring(auto_upload=not args.no_upload)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"An unhandled exception occurred in main: {e}", exc_info=True)
    finally:
        if monitor is not None: 
            monitor.stop_monitoring()
        logger.info("--- Monitor Exited ---")

if __name__ == "__main__":
    main()
