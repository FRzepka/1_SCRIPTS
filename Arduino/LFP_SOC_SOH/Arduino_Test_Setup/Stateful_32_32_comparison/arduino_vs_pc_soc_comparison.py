"""
🔥 ARDUINO vs PC SOC PREDICTION COMPARISON 🔥
==============================================

SEQUENZIELLER VERGLEICH: Arduino Hardware LSTM vs PC PyTorch LSTM
- Phase 1: Arduino Test (5 Min) - Progress Prints
- Phase 2: PC Test (gleiche Daten) - Progress Prints  
- Phase 3: Vergleich und finaler Plot
- Keine Live-Plots - nur finale Auswertung
- Perfekt für saubere Hardware-Validierung!

🚀 SEQUENZIELL = SAUBERER! 🚀
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
import torch
import torch.nn as nn

# Warnings unterdrücken
warnings.filterwarnings("ignore")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== EINSTELLUNGEN =====
ARDUINO_PORT = 'COM13'
BAUDRATE = 115200
DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
PC_MODEL_PATH = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\SOC\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\training_run_1_2_4_31\best_model.pth"
ARDUINO_SKETCH_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32\code_weights\arduino_lstm_soc_full32_with_monitoring\arduino_lstm_soc_full32_with_monitoring.ino"
ARDUINO_CLI_PATH = r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Arduino_CLI\arduino-cli.exe"
ARDUINO_FQBN = "arduino:renesas_uno:unor4wifi"

# Vergleichseinstellungen
COMPARISON_DURATION_MINS = 5  # 5 Minuten Test pro Phase
TEST_SAMPLES = 600  # Anzahl Samples für Test (bei 50ms = 30s, bei 500ms = 5min)
PREDICTION_DELAY = 500   # 500ms zwischen Predictions für stabilere Verbindung

# Model parameters (identisch zu beiden Modellen)
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32

class SOCModel(nn.Module):
    """
    PC PyTorch LSTM SOC Model - identisch zum Training Script
    """
    def __init__(self, input_size=4, dropout=0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(MLP_HIDDEN, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.contiguous()
        h, c = hidden
        h, c = h.contiguous(), c.contiguous()
        hidden = (h, c)
        out, hidden = self.lstm(x, hidden)
        batch, seq_len, hid = out.size()
        out_flat = out.contiguous().view(batch * seq_len, hid)
        soc_flat = self.mlp(out_flat)
        soc = soc_flat.view(batch, seq_len)
        return soc, hidden

class ArduinoUploader:
    """Arduino Sketch Uploader"""
    
    def __init__(self, sketch_path, fqbn, port, cli_path=None):
        self.sketch_path = sketch_path
        self.fqbn = fqbn
        self.port = port
        self.cli_path = cli_path or ARDUINO_CLI_PATH
        self.arduino_cli_available = self.check_arduino_cli()
    
    def check_arduino_cli(self):
        """Check if arduino-cli is available"""
        try:
            if self.cli_path and os.path.exists(self.cli_path):
                result = subprocess.run([self.cli_path, 'version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"✅ Arduino CLI found: {result.stdout.strip()}")
                    return True
            
            result = subprocess.run(['arduino-cli', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ Arduino CLI found in PATH: {result.stdout.strip()}")
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
            logger.info(f"🔨 Compiling Arduino sketch...")
            cmd = [self.cli_path, 'compile', '--fqbn', self.fqbn, self.sketch_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("✅ Arduino compilation successful")
                
                logger.info(f"⬆️ Uploading to {self.port}...")
                cmd = [self.cli_path, 'upload', '--fqbn', self.fqbn, '--port', self.port, self.sketch_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    logger.info("✅ Arduino upload successful")
                    return True, "Upload successful"
                else:
                    return False, f"Upload failed: {result.stderr}"
            else:
                return False, f"Compilation failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Compilation/Upload timeout"
        except Exception as e:
            return False, f"Error: {str(e)}"

class ArduinoVsPCComparison:
    """
    🔥 MEGA COMPARISON CLASS 🔥
    Arduino Hardware LSTM vs PC PyTorch LSTM
    """
    
    def __init__(self, port=ARDUINO_PORT, baudrate=BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        self.ground_truth_data = None
        self.scaler = None
        
        # PC Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pc_model = None
        self.pc_hidden_state = None
        
        # Data storage für Vergleich
        self.arduino_predictions = deque(maxlen=2000)
        self.pc_predictions = deque(maxlen=2000)
        self.ground_truths = deque(maxlen=2000)
        self.voltages = deque(maxlen=2000)
        self.timestamps = deque(maxlen=2000)
        
        # Threading
        self.prediction_running = False
        self.current_index = 0
        self.start_time = None
        
        # Plotting
        self.fig, self.axes = plt.subplots(2, 2, figsize=(20, 12))
        self.fig.suptitle('🔥 ARDUINO vs PC SOC PREDICTION COMPARISON 🔥', fontsize=20, fontweight='bold')
        
        logger.info(f"🚀 ArduinoVsPCComparison initialisiert - Device: {self.device}")
    
    def initialize_scaler(self):
        """
        Erstelle den EXAKT gleichen StandardScaler wie im Training Script
        """
        logger.info("🔧 Initialisiere StandardScaler...")
        
        base_path = Path(r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU")
        
        train_cells = [
            "MGFarm_18650_C01", "MGFarm_18650_C03", "MGFarm_18650_C05",
            "MGFarm_18650_C11", "MGFarm_18650_C17", "MGFarm_18650_C23"
        ]
        val_cells = [
            "MGFarm_18650_C07", "MGFarm_18650_C19", "MGFarm_18650_C21"
        ]
        
        all_cells = train_cells + val_cells
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        
        # StandardScaler über alle Zellen fitten
        self.scaler = StandardScaler()
        
        for cell_name in all_cells:
            cell_path = base_path / cell_name / "df.parquet"
            if cell_path.exists():
                df = pd.read_parquet(cell_path)
                df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
                self.scaler.partial_fit(df[feats])
                logger.info(f"✅ Scaler partial fit für {cell_name}")
            else:
                logger.warning(f"⚠️ Datei nicht gefunden: {cell_path}")
        
        logger.info("✅ StandardScaler initialisiert")
        return True
    
    def load_pc_model(self):
        """
        Lade das PC PyTorch Modell
        """
        logger.info(f"🧠 Lade PC PyTorch Modell: {PC_MODEL_PATH}")
        
        if not Path(PC_MODEL_PATH).exists():
            logger.error(f"❌ PC Modell nicht gefunden: {PC_MODEL_PATH}")
            return False
        
        try:
            # Erstelle Modell-Instanz
            self.pc_model = SOCModel(input_size=4, dropout=0.05)
            
            # Lade gespeicherte Gewichte
            checkpoint = torch.load(PC_MODEL_PATH, map_location=self.device)
            self.pc_model.load_state_dict(checkpoint)
            
            # Modell auf Device und in Evaluations-Modus
            self.pc_model.to(self.device)
            self.pc_model.eval()
            
            # Initialisiere Hidden State
            self.init_pc_hidden_state()
            
            logger.info("✅ PC Modell erfolgreich geladen!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden des PC Modells: {e}")
            return False
    
    def init_pc_hidden_state(self):
        """
        Initialisiere PC LSTM Hidden State
        """
        batch_size = 1
        self.pc_hidden_state = (
            torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device),
            torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device)
        )
        logger.info("✅ PC LSTM Hidden State initialisiert")
    
    def load_ground_truth_data(self):
        """
        Lade Ground Truth Daten (C19)
        """
        logger.info(f"📊 Lade Ground Truth Daten: {DATA_PATH}")
        if not Path(DATA_PATH).exists():
            logger.error(f"❌ Ground Truth Datei nicht gefunden: {DATA_PATH}")
            return False
        
        # Lade C19 Daten
        df = pd.read_parquet(DATA_PATH)
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        
        # Prüfe Spalten
        required_cols = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c", "SOC_ZHU"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Fehlende Spalten: {missing_cols}")
            return False
        
        # Skaliere Features
        if self.scaler is None:
            logger.error("❌ Scaler nicht initialisiert!")
            return False
        
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        df[feats] = self.scaler.transform(df[feats])
        
        self.ground_truth_data = df
        logger.info(f"✅ Ground Truth Daten geladen: {len(df)} Datenpunkte")
        logger.info(f"SOC range: {df['SOC_ZHU'].min():.3f} - {df['SOC_ZHU'].max():.3f}")
        
        return True
    
    def connect_arduino(self):
        """
        Verbinde mit Arduino
        """
        logger.info(f"🔌 Verbinde mit Arduino auf {self.port}...")
        
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(3)  # Arduino startup time
            
            # Arduino reset
            logger.info("🔄 Resette Arduino...")
            self.arduino.write(b"RESET\n")
            self.arduino.flush()
            
            # Warte auf Ready Signal
            logger.info("⏳ Warte auf Arduino Ready Signal...")
            ready_timeout = time.time() + 10
            while time.time() < ready_timeout:
                if self.arduino.in_waiting:
                    line = self.arduino.readline().decode('utf-8').strip()
                    if "READY" in line:
                        logger.info("✅ Arduino ist bereit!")
                        return True
                time.sleep(0.1)
            
            logger.warning("⚠️ Kein Ready Signal - fahre trotzdem fort")
            return True
            
        except Exception as e:
            logger.error(f"❌ Arduino Verbindung fehlgeschlagen: {e}")
            return False
    
    def predict_arduino_soc(self, voltage, current, soh, q_c):
        """
        Hole SOC Vorhersage vom Arduino
        """
        if not self.arduino or not self.arduino.is_open:
            return None, None
        
        start_time = time.time()
        
        try:
            # Sende Daten an Arduino
            data_str = f"{voltage:.6f},{current:.6f},{soh:.6f},{q_c:.6f}\n"
            self.arduino.write(data_str.encode('utf-8'))
            self.arduino.flush()
            
            # Lese Antwort
            response_timeout = time.time() + 1.0
            while time.time() < response_timeout:
                if self.arduino.in_waiting:
                    line = self.arduino.readline().decode('utf-8').strip()
                    if line.startswith("SOC:"):
                        try:
                            soc_value = float(line.split(":")[1])
                            inference_time = (time.time() - start_time) * 1000
                            return np.clip(soc_value, 0, 1), inference_time
                        except:
                            continue
                time.sleep(0.001)
            
            logger.warning("⚠️ Arduino Timeout")
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Arduino Prediction Fehler: {e}")
            return None, None
    
    def predict_pc_soc(self, voltage, current, soh, q_c):
        """
        PC PyTorch SOC Vorhersage
        """
        if self.pc_model is None:
            return None, None
        
        start_time = time.time()
        
        try:
            # Input als Tensor vorbereiten
            input_features = torch.tensor([[voltage, current, soh, q_c]], 
                                        dtype=torch.float32, device=self.device)
            input_features = input_features.unsqueeze(0)  # Shape: (1, 1, 4)
            
            with torch.no_grad():
                # Modell inference
                soc_pred, self.pc_hidden_state = self.pc_model(input_features, self.pc_hidden_state)
                
                # SOC Vorhersage extrahieren
                soc_value = soc_pred.squeeze().cpu().item()
                soc_value = np.clip(soc_value, 0, 1)
            
            inference_time = (time.time() - start_time) * 1000
            return soc_value, inference_time
            
        except Exception as e:
            logger.error(f"❌ PC Prediction Fehler: {e}")
            return None, None
    
    def background_prediction_loop(self):
        """
        Background Thread für Predictions
        """
        logger.info("🔄 Prediction Loop gestartet")
        self.start_time = time.time()
        sample_count = 0
        
        # Dauer in Sekunden
        max_duration = COMPARISON_DURATION_MINS * 60
        
        while self.prediction_running:
            current_time = time.time() - self.start_time
            
            # Stoppe nach der gewünschten Zeit
            if current_time > max_duration:
                logger.info(f"⏰ {COMPARISON_DURATION_MINS} Minuten erreicht - stoppe Prediction Loop")
                self.prediction_running = False
                break
            
            # Hole nächsten Datenpunkt
            if self.current_index >= len(self.ground_truth_data):
                logger.info("🔄 Alle Daten durchlaufen - starte von vorne")
                self.current_index = 0
            
            row = self.ground_truth_data.iloc[self.current_index]
            
            # Extrahiere Features
            voltage = row["Voltage[V]"]
            current = row["Current[A]"]
            soh = row["SOH_ZHU"]
            q_c = row["Q_c"]
            ground_truth_soc = row["SOC_ZHU"]
            
            # Arduino Vorhersage
            arduino_soc, arduino_time = self.predict_arduino_soc(voltage, current, soh, q_c)
            
            # PC Vorhersage
            pc_soc, pc_time = self.predict_pc_soc(voltage, current, soh, q_c)
            
            # Speichere nur wenn beide Vorhersagen erfolgreich
            if arduino_soc is not None and pc_soc is not None:
                self.arduino_predictions.append(arduino_soc)
                self.pc_predictions.append(pc_soc)
                self.ground_truths.append(ground_truth_soc)
                self.voltages.append(voltage)
                self.timestamps.append(current_time)
                
                sample_count += 1
                
                # Console Output alle 50 Samples
                if sample_count % 50 == 0:
                    arduino_mae = np.mean([abs(a - g) for a, g in zip(self.arduino_predictions, self.ground_truths)]) if self.arduino_predictions else 0
                    pc_mae = np.mean([abs(p - g) for p, g in zip(self.pc_predictions, self.ground_truths)]) if self.pc_predictions else 0
                    
                    logger.info(f"📊 Sample {sample_count:4d} | Zeit: {current_time:.1f}s | "
                              f"Arduino SOC: {arduino_soc:.4f} | PC SOC: {pc_soc:.4f} | "
                              f"Ground Truth: {ground_truth_soc:.4f} | "
                              f"Arduino MAE: {arduino_mae:.6f} | PC MAE: {pc_mae:.6f}")
            
            self.current_index += 1
            time.sleep(PREDICTION_DELAY / 1000.0)  # Delay in seconds
        
        logger.info(f"🏁 Prediction Loop beendet - {sample_count} Samples verarbeitet")
    
    def update_plots(self, frame):
        """
        Update der Live Plots
        """
        if len(self.timestamps) < 2:
            return
        
        # Aktuelle Daten für Plot
        times = list(self.timestamps)[-PLOT_WINDOW_SIZE:]
        arduino_socs = list(self.arduino_predictions)[-PLOT_WINDOW_SIZE:]
        pc_socs = list(self.pc_predictions)[-PLOT_WINDOW_SIZE:]
        ground_truths = list(self.ground_truths)[-PLOT_WINDOW_SIZE:]
        voltages = list(self.voltages)[-PLOT_WINDOW_SIZE:]
        
        # Clear alle Plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: SOC Comparison - DAS IST DER WICHTIGSTE!
        self.axes[0,0].plot(times, ground_truths, 'g-', linewidth=3, label='🎯 Ground Truth', alpha=0.9)
        self.axes[0,0].plot(times, arduino_socs, 'r--', linewidth=2, label='🤖 Arduino LSTM', alpha=0.8)
        self.axes[0,0].plot(times, pc_socs, 'b:', linewidth=2, label='💻 PC PyTorch', alpha=0.8)
        self.axes[0,0].set_title('🔥 SOC COMPARISON: Ground Truth vs Arduino vs PC 🔥', fontweight='bold', fontsize=14)
        self.axes[0,0].set_ylabel('State of Charge', fontweight='bold')
        self.axes[0,0].set_xlabel('Time [s]', fontweight='bold')
        self.axes[0,0].legend(fontsize=12)
        self.axes[0,0].grid(True, alpha=0.3)
        self.axes[0,0].set_ylim(-0.05, 1.05)
        
        # Plot 2: Voltage
        self.axes[0,1].plot(times, voltages, 'purple', linewidth=2, label='⚡ Voltage', alpha=0.8)
        self.axes[0,1].set_title('Battery Voltage', fontweight='bold', fontsize=12)
        self.axes[0,1].set_ylabel('Voltage [V] (scaled)', fontweight='bold')
        self.axes[0,1].set_xlabel('Time [s]', fontweight='bold')
        self.axes[0,1].legend()
        self.axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Error Comparison
        if len(arduino_socs) > 0 and len(pc_socs) > 0:
            arduino_errors = [abs(a - g) for a, g in zip(arduino_socs, ground_truths)]
            pc_errors = [abs(p - g) for p, g in zip(pc_socs, ground_truths)]
            
            self.axes[1,0].plot(times, arduino_errors, 'r-', linewidth=2, label='🤖 Arduino Error', alpha=0.8)
            self.axes[1,0].plot(times, pc_errors, 'b-', linewidth=2, label='💻 PC Error', alpha=0.8)
            self.axes[1,0].set_title('Prediction Errors', fontweight='bold', fontsize=12)
            self.axes[1,0].set_ylabel('Absolute Error', fontweight='bold')
            self.axes[1,0].set_xlabel('Time [s]', fontweight='bold')
            self.axes[1,0].legend()
            self.axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Statistics
        if len(self.arduino_predictions) > 10:
            arduino_mae = np.mean([abs(a - g) for a, g in zip(self.arduino_predictions, self.ground_truths)])
            pc_mae = np.mean([abs(p - g) for p, g in zip(self.pc_predictions, self.ground_truths)])
            
            current_time = self.timestamps[-1] if self.timestamps else 0
            samples = len(self.arduino_predictions)
            
            # Text Statistics
            stats_text = f"""
🔥 LIVE COMPARISON STATS 🔥

⏰ Time: {current_time:.1f}s / {COMPARISON_DURATION_MINS*60}s
📊 Samples: {samples}

🤖 ARDUINO LSTM:
   MAE: {arduino_mae:.6f}

💻 PC PYTORCH:
   MAE: {pc_mae:.6f}

🏆 WINNER: {'Arduino' if arduino_mae < pc_mae else 'PC' if pc_mae < arduino_mae else 'TIE'}
🎯 Difference: {abs(arduino_mae - pc_mae):.6f}
            """
            
            self.axes[1,1].text(0.05, 0.95, stats_text, transform=self.axes[1,1].transAxes,
                              fontsize=12, verticalalignment='top', fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            self.axes[1,1].set_title('📈 Live Statistics', fontweight='bold', fontsize=12)
            self.axes[1,1].axis('off')
        
        plt.tight_layout()
    
    def upload_arduino_sketch(self):
        """
        Upload Arduino Sketch
        """
        uploader = ArduinoUploader(ARDUINO_SKETCH_PATH, ARDUINO_FQBN, self.port)
        success, message = uploader.compile_and_upload()
        
        if success:
            logger.info("✅ Arduino Upload erfolgreich")
            time.sleep(3)  # Warte nach Upload
            return True
        else:
            logger.error(f"❌ Arduino Upload fehlgeschlagen: {message}")
            return False
    
    def start_comparison(self, auto_upload=True):
        """
        🔥 STARTE DEN MEGA VERGLEICH! 🔥
        """
        print("🔥" + "="*80 + "🔥")
        print("🔥 ARDUINO vs PC SOC PREDICTION COMPARISON 🔥")
        print("🔥" + "="*80 + "🔥")
        print(f"⏰ Vergleichsdauer: {COMPARISON_DURATION_MINS} Minuten")
        print()
        
        # 1. Arduino Upload (optional)
        if auto_upload:
            print("⬆️ 1. Uploading Arduino sketch...")
            if not self.upload_arduino_sketch():
                print("❌ Upload failed. Continue with existing sketch? (y/n): ", end="")
                if input().lower() != 'y':
                    return
        
        # 2. Scaler initialisieren
        print("🔧 2. Initialisiere Scaler...")
        if not self.initialize_scaler():
            print("❌ Scaler Initialisierung fehlgeschlagen!")
            return
        
        # 3. PC Modell laden
        print("🧠 3. Lade PC PyTorch Modell...")
        if not self.load_pc_model():
            print("❌ PC Modell laden fehlgeschlagen!")
            return
        
        # 4. Ground Truth Daten laden
        print("📊 4. Lade Ground Truth Daten...")
        if not self.load_ground_truth_data():
            print("❌ Ground Truth Daten laden fehlgeschlagen!")
            return
        
        # 5. Arduino Verbindung
        print("🔌 5. Verbinde mit Arduino...")
        if not self.connect_arduino():
            print("❌ Arduino Verbindung fehlgeschlagen!")
            return
        
        # 6. Start Prediction Thread
        print("🚀 6. Starte Prediction Thread...")
        self.prediction_running = True
        prediction_thread = threading.Thread(target=self.background_prediction_loop)
        prediction_thread.daemon = True
        prediction_thread.start()
        
        # 7. Start Animation
        print("📈 7. Starte Live Plots...")
        ani = animation.FuncAnimation(self.fig, self.update_plots, interval=UPDATE_INTERVAL, blit=False)
        
        print("✅ VERGLEICH GESTARTET! Schließe Plot-Fenster zum Stoppen.")
        print()
        plt.show()
        
        # 8. Cleanup
        print("🧹 Cleanup...")
        self.prediction_running = False
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
        
        # 9. Finale Statistiken
        self.print_final_stats()
        
        # 10. Speichere Ergebnisse
        self.save_results()
    
    def print_final_stats(self):
        """
        Drucke finale Vergleichsstatistiken
        """
        if len(self.arduino_predictions) < 10:
            print("❌ Zu wenige Daten für finale Statistiken")
            return
        
        arduino_mae = np.mean([abs(a - g) for a, g in zip(self.arduino_predictions, self.ground_truths)])
        pc_mae = np.mean([abs(p - g) for p, g in zip(self.pc_predictions, self.ground_truths)])
        
        arduino_rmse = np.sqrt(np.mean([(a - g)**2 for a, g in zip(self.arduino_predictions, self.ground_truths)]))
        pc_rmse = np.sqrt(np.mean([(p - g)**2 for p, g in zip(self.pc_predictions, self.ground_truths)]))
        
        total_samples = len(self.arduino_predictions)
        total_time = self.timestamps[-1] if self.timestamps else 0
        
        print("\n" + "🏆" + "="*80 + "🏆")
        print("🏆 FINALE VERGLEICHSSTATISTIKEN 🏆")
        print("🏆" + "="*80 + "🏆")
        print(f"📊 Total Samples: {total_samples}")
        print(f"⏰ Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print()
        print("🤖 ARDUINO LSTM:")
        print(f"   MAE:  {arduino_mae:.6f}")
        print(f"   RMSE: {arduino_rmse:.6f}")
        print()
        print("💻 PC PYTORCH:")
        print(f"   MAE:  {pc_mae:.6f}")
        print(f"   RMSE: {pc_rmse:.6f}")
        print()
        
        # Winner bestimmen
        if arduino_mae < pc_mae:
            winner = "🤖 ARDUINO"
            improvement = ((pc_mae - arduino_mae) / pc_mae) * 100
        elif pc_mae < arduino_mae:
            winner = "💻 PC"
            improvement = ((arduino_mae - pc_mae) / arduino_mae) * 100
        else:
            winner = "🤝 TIE"
            improvement = 0
        
        print(f"🏆 WINNER: {winner}")
        if improvement > 0:
            print(f"🎯 Improvement: {improvement:.2f}%")
        print(f"📈 Absolute Difference: {abs(arduino_mae - pc_mae):.6f}")
        print("🏆" + "="*80 + "🏆")
    
    def save_results(self):
        """
        Speichere Vergleichsergebnisse
        """
        if len(self.arduino_predictions) < 10:
            return
        
        # Erstelle DataFrame
        results_df = pd.DataFrame({
            'timestamp': list(self.timestamps),
            'ground_truth_soc': list(self.ground_truths),
            'arduino_soc': list(self.arduino_predictions),
            'pc_soc': list(self.pc_predictions),
            'voltage': list(self.voltages)
        })
        
        # Berechne Errors
        results_df['arduino_error'] = abs(results_df['arduino_soc'] - results_df['ground_truth_soc'])
        results_df['pc_error'] = abs(results_df['pc_soc'] - results_df['ground_truth_soc'])
        
        # Speicherpfad
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = Path(f"c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/Arduino_Test_Setup/Stateful_32_32_comparison/arduino_vs_pc_comparison_{timestamp}.csv")
        
        # Speichere CSV
        results_df.to_csv(save_path, index=False)
        logger.info(f"💾 Vergleichsergebnisse gespeichert: {save_path}")
        
        # Erstelle finales Vergleichsplot
        self.create_final_comparison_plot(results_df, timestamp)
    
    def create_final_comparison_plot(self, df, timestamp):
        """
        Erstelle finales Vergleichsplot
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'🔥 ARDUINO vs PC SOC COMPARISON - FINAL RESULTS 🔥\n{COMPARISON_DURATION_MINS} Minutes Test', 
                     fontsize=18, fontweight='bold')
        
        # Plot 1: SOC Comparison
        axes[0,0].plot(df['timestamp'], df['ground_truth_soc'], 'g-', linewidth=3, 
                      label='🎯 Ground Truth', alpha=0.9)
        axes[0,0].plot(df['timestamp'], df['arduino_soc'], 'r--', linewidth=2, 
                      label='🤖 Arduino LSTM', alpha=0.8)
        axes[0,0].plot(df['timestamp'], df['pc_soc'], 'b:', linewidth=2, 
                      label='💻 PC PyTorch', alpha=0.8)
        axes[0,0].set_title('SOC Prediction Comparison', fontweight='bold', fontsize=14)
        axes[0,0].set_ylabel('State of Charge', fontweight='bold')
        axes[0,0].set_xlabel('Time [s]', fontweight='bold')
        axes[0,0].legend(fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim(-0.05, 1.05)
        
        # Plot 2: Error Comparison
        axes[0,1].plot(df['timestamp'], df['arduino_error'], 'r-', linewidth=2, 
                      label='🤖 Arduino Error', alpha=0.8)
        axes[0,1].plot(df['timestamp'], df['pc_error'], 'b-', linewidth=2, 
                      label='💻 PC Error', alpha=0.8)
        axes[0,1].set_title('Prediction Errors Over Time', fontweight='bold', fontsize=14)
        axes[0,1].set_ylabel('Absolute Error', fontweight='bold')
        axes[0,1].set_xlabel('Time [s]', fontweight='bold')
        axes[0,1].legend(fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter Plot Arduino vs PC
        axes[1,0].scatter(df['arduino_soc'], df['pc_soc'], alpha=0.6, s=10)
        min_soc = min(df['arduino_soc'].min(), df['pc_soc'].min())
        max_soc = max(df['arduino_soc'].max(), df['pc_soc'].max())
        axes[1,0].plot([min_soc, max_soc], [min_soc, max_soc], 'k--', label='Perfect Agreement')
        axes[1,0].set_title('Arduino vs PC SOC Predictions', fontweight='bold', fontsize=14)
        axes[1,0].set_xlabel('Arduino SOC', fontweight='bold')
        axes[1,0].set_ylabel('PC SOC', fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Error Distribution
        axes[1,1].hist(df['arduino_error'], bins=50, alpha=0.7, label='🤖 Arduino', color='red')
        axes[1,1].hist(df['pc_error'], bins=50, alpha=0.7, label='💻 PC', color='blue')
        axes[1,1].set_title('Error Distribution', fontweight='bold', fontsize=14)
        axes[1,1].set_xlabel('Absolute Error', fontweight='bold')
        axes[1,1].set_ylabel('Frequency', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Speichere Plot
        plot_path = Path(f"c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/Arduino_Test_Setup/Stateful_32_32_comparison/arduino_vs_pc_final_comparison_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Finales Vergleichsplot gespeichert: {plot_path}")
        
        plt.show()

def main():
    """
    🔥 MAIN FUNCTION - STARTE DEN MEGA VERGLEICH! 🔥
    """
    comparison = ArduinoVsPCComparison()
    
    try:
        comparison.start_comparison(auto_upload=True)
    except KeyboardInterrupt:
        print("⏹️ Vergleich von Benutzer gestoppt")
    except Exception as e:
        logger.error(f"❌ Fehler: {e}")
    finally:
        print("🏁 Vergleich beendet")

if __name__ == "__main__":
    main()
