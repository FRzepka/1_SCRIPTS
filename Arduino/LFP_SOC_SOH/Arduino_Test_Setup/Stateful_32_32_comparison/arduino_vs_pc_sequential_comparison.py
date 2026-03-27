"""
🔥 ARDUINO vs PC SOC PREDICTION SEQUENTIAL COMPARISON 🔥
========================================================

SEQUENZIELLER VERGLEICH: Arduino Hardware LSTM vs PC PyTorch LSTM
- Phase 1: Arduino Test (Progress Prints)
- Phase 2: PC Test (gleiche Daten, Progress Prints)  
- Phase 3: Vergleich und finaler Plot
- Keine Live-Plots - nur finale Auswertung
- Saubere sequenzielle Datensammlung!

🚀 SEQUENZIELL = SAUBERER! 🚀
"""

import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging
import warnings
import torch
import torch.nn as nn
import json

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

# Test Einstellungen
START_MINUTE = 0        # Starte ab 60 Minuten (stabilere Daten)
TEST_DURATION_MINS = 100  # 15 Minuten Test (nicht 200!)
TEST_SAMPLES = TEST_DURATION_MINS * 60  # 1 Sample pro Sekunde = 900 Samples für 15min
PREDICTION_DELAY = 30    # 30ms zwischen Predictions (EXAKT wie funktionierendes Script!)

# Model parameters
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32

class SOCModel(nn.Module):
    """PC PyTorch LSTM SOC Model"""
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
                    return True
            
            result = subprocess.run(['arduino-cli', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.cli_path = 'arduino-cli'
                return True
            else:
                return False
        except:
            return False
    
    def compile_and_upload(self):
        """Compile and upload the sketch"""
        if not self.arduino_cli_available:
            return False, "Arduino CLI not installed"
        
        try:
            print("🔨 Kompiliere Arduino Sketch...")
            cmd = [self.cli_path, 'compile', '--fqbn', self.fqbn, self.sketch_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ Kompilierung erfolgreich")
                
                print(f"⬆️ Uploade auf {self.port}...")
                cmd = [self.cli_path, 'upload', '--fqbn', self.fqbn, '--port', self.port, self.sketch_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print("✅ Upload erfolgreich")
                    return True, "Upload successful"
                else:
                    return False, f"Upload failed: {result.stderr}"
            else:
                return False, f"Compilation failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Compilation/Upload timeout"
        except Exception as e:
            return False, f"Error: {str(e)}"

class SequentialComparison:
    """
    🔥 SEQUENZIELLER VERGLEICH 🔥
    Arduino Hardware LSTM vs PC PyTorch LSTM
    """
    
    def __init__(self):
        self.ground_truth_data = None
        self.scaler = None
        
        # PC Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pc_model = None
        self.pc_hidden_state = None        
        # Results Storage
        self.test_data = []  # Gemeinsame Testdaten
        self.arduino_results = []
        self.pc_results = []
        
        print(f"🚀 SequentialComparison initialisiert - Device: {self.device}")
    
    def initialize_scaler(self):
        """Erstelle StandardScaler EXAKT wie funktionierendes Script"""
        print("🔧 Initialisiere StandardScaler...")
        
        # VERWENDE LOKALEN SCALER wie funktionierendes Script!
        try:
            scaler_path = Path(DATA_PATH).parent / "scaler.pkl"
            
            if scaler_path.exists():
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✅ Loaded existing scaler from funktionierendes Script")
                return True
            else:
                print("⚠️ Kein scaler.pkl gefunden - erstelle neuen")
        except Exception as e:
            print(f"⚠️ Scaler load failed: {e} - erstelle neuen")
        
        # Fallback: Erstelle Scaler wie funktionierendes Script
        df = pd.read_parquet(DATA_PATH)
        required_cols = ['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_c', 'SOC_ZHU']
        df = df.dropna(subset=required_cols)
        df = df[df['Voltage[V]'] > 0]
        
        feature_cols = ['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_c']
        features = df[feature_cols].values
        
        self.scaler = StandardScaler()
        self.scaler.fit(features)
        
        # Speichere für nächstes Mal
        try:
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print("✅ Scaler gespeichert")
        except:
            pass
        
        print("✅ StandardScaler erstellt (wie funktionierendes Script)")
        return True
    
    def scale_features_for_arduino(self, voltage, current, soh, q_c):
        """Scale features EXAKT wie funktionierendes Script"""
        features = np.array([[voltage, current, soh, q_c]])
        if self.scaler:
            return self.scaler.transform(features)[0]
        return features[0]
    
    def load_pc_model(self):
        """Lade PC PyTorch Modell"""
        print(f"🧠 Lade PC PyTorch Modell...")
        
        if not Path(PC_MODEL_PATH).exists():
            print(f"❌ PC Modell nicht gefunden: {PC_MODEL_PATH}")
            return False
        
        try:
            self.pc_model = SOCModel(input_size=4, dropout=0.05)
            checkpoint = torch.load(PC_MODEL_PATH, map_location=self.device)
            self.pc_model.load_state_dict(checkpoint)
            self.pc_model.to(self.device)
            self.pc_model.eval()
              # Init Hidden State
            batch_size = 1
            self.pc_hidden_state = (
                torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device),
                torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device)
            )
            
            print("✅ PC Modell geladen")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Laden des PC Modells: {e}")
            return False
    
    def prepare_test_data(self):
        """Bereite Testdaten vor - ab 60 Minuten für 15 Minuten"""
        print(f"📊 Lade und bereite Testdaten vor...")
        print(f"⏰ Zeitbereich: ab {START_MINUTE} Minuten für {TEST_DURATION_MINS} Minuten")
        
        if not Path(DATA_PATH).exists():
            print(f"❌ Daten nicht gefunden: {DATA_PATH}")
            return False
        
        # Lade C19 Daten
        df = pd.read_parquet(DATA_PATH)
        
        # Prüfe Spalten
        required_cols = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c", "SOC_ZHU"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Fehlende Spalten: {missing_cols}")
            return False
        
        # Berechne Start- und End-Index basierend auf Zeit
        # Annahme: 1 Sample pro Sekunde (typisch für MGFarm Daten)
        start_sample = START_MINUTE * 60  # 60 Minuten * 60 Sekunden
        end_sample = start_sample + (TEST_DURATION_MINS * 60)  # + 15 Minuten
        
        print(f"📍 Datenbereich: Sample {start_sample} bis {end_sample} (von {len(df)} total)")
        
        # Prüfe ob genügend Daten vorhanden
        if end_sample > len(df):
            print(f"⚠️ Nicht genügend Daten! Verfügbar bis Sample {len(df)}")
            print(f"⚠️ Verwende verfügbare Daten ab Sample {start_sample}")
            end_sample = len(df)
        
        # Extrahiere Datenbereich
        df_segment = df.iloc[start_sample:end_sample].copy()
        
        if len(df_segment) < 100:
            print(f"❌ Zu wenige Daten im gewählten Zeitbereich: {len(df_segment)} Samples")
            return False
        
        # Skaliere Features
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        df_segment[feats] = self.scaler.transform(df_segment[feats])
        
        # Bereite Testdaten vor (nehme alle verfügbaren Samples aus dem Zeitbereich)
        self.test_data = []
        for idx, row in df_segment.iterrows():
            self.test_data.append({
                'voltage': row["Voltage[V]"],
                'current': row["Current[A]"],
                'soh': row["SOH_ZHU"],
                'q_c': row["Q_c"],
                'ground_truth_soc': row["SOC_ZHU"]
            })
        
        print(f"✅ {len(self.test_data)} Testdatenpunkte vorbereitet")
        print(f"📊 SOC Range: {df_segment['SOC_ZHU'].min():.3f} - {df_segment['SOC_ZHU'].max():.3f}")
        print(f"⏱️ Erwartete Testdauer: {(len(self.test_data) * PREDICTION_DELAY) / 1000 / 60:.1f} Minuten")
        
        return True
    
    def test_arduino(self):
        """Phase 1: Arduino Test"""
        print("\n" + "🤖" + "="*60 + "🤖")
        print("🤖 PHASE 1: ARDUINO HARDWARE LSTM TEST 🤖")
        print("🤖" + "="*60 + "🤖")
        
        # Upload Arduino Sketch
        print("⬆️ Uploade Arduino Sketch...")
        uploader = ArduinoUploader(ARDUINO_SKETCH_PATH, ARDUINO_FQBN, ARDUINO_PORT)
        success, message = uploader.compile_and_upload()
        
        if not success:
            print(f"❌ Arduino Upload fehlgeschlagen: {message}")
            return False
        
        time.sleep(3)  # Warte nach Upload
          # Verbinde mit Arduino - EXAKT wie funktionierendes Script
        print(f"🔌 Verbinde mit Arduino auf {ARDUINO_PORT}...")
        try:
            arduino = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=3)
            time.sleep(3)  # Arduino connection stabilization
            
            # Test connection with STATS command - EXAKT wie funktionierendes Script
            arduino.write(b'STATS\n')
            time.sleep(0.5)
            response = arduino.readline().decode().strip()
            if response and ("STATS:" in response or len(response) > 5):
                print("✅ Arduino verbunden")
            else:
                print(f"⚠️ Arduino Verbindungstest: {response} - fahre trotzdem fort")
            
        except Exception as e:
            print(f"❌ Arduino Verbindung fehlgeschlagen: {e}")
            return False        
        # 🔥 KRITISCH: Arduino LSTM Warmup für stateful Predictions!
        print("🔥 WARNUNG: Arduino LSTM ist STATEFUL - braucht Warmup!")
        if not self.warmup_arduino_lstm(arduino, warmup_samples=100):
            print("⚠️ Arduino Warmup fehlgeschlagen - Resultate könnten ungenau sein!")
        
        # Arduino Test durchführen
        print(f"🧪 Starte Arduino Test mit {len(self.test_data)} Samples...")
        print(f"⏱️ Geschätzte Dauer: {(len(self.test_data) * PREDICTION_DELAY) / 1000 / 60:.1f} Minuten")
        
        self.arduino_results = []
        successful_predictions = 0
        
        for i, data_point in enumerate(self.test_data):
            try:
                # Skaliere Features EXAKT wie funktionierendes Script
                scaled_features = self.scale_features_for_arduino(
                    data_point['voltage'], data_point['current'], 
                    data_point['soh'], data_point['q_c']
                )
                
                # Arduino Protokoll: DATA: command EXAKT wie funktionierendes Script
                command = f"DATA:{scaled_features[0]:.6f},{scaled_features[1]:.6f},{scaled_features[2]:.6f},{scaled_features[3]:.6f}\n"
                arduino.write(command.encode())
                
                # Lese Antwort - EXAKT wie funktionierendes Script
                response = arduino.readline().decode().strip()
                
                if response and response.startswith("DATA:"):
                    # Parse: "DATA:SOC,inference_time_us,ram_free,ram_used,cpu_load,temp"
                    data = response.replace("DATA:", "").split(",")
                    if len(data) >= 1:  # Mindestens SOC
                        try:
                            soc_value = float(data[0])
                            soc_value = np.clip(soc_value, 0, 1)
                            
                            self.arduino_results.append({
                                'sample': i,
                                'voltage': data_point['voltage'],
                                'current': data_point['current'],
                                'soh': data_point['soh'],
                                'q_c': data_point['q_c'],
                                'ground_truth_soc': data_point['ground_truth_soc'],
                                'predicted_soc': soc_value,
                                'error': abs(soc_value - data_point['ground_truth_soc'])
                            })
                            
                            successful_predictions += 1
                            response_received = True
                        except:
                            print(f"⚠️ Sample {i+1} Parse Error")
                            continue
                else:
                    print(f"⚠️ Sample {i+1} Invalid Response: {response}")
                
                # Progress Print alle 50 Samples
                if (i + 1) % 50 == 0:
                    success_rate = (successful_predictions / (i + 1)) * 100
                    current_mae = np.mean([r['error'] for r in self.arduino_results]) if self.arduino_results else 0
                    print(f"🤖 Arduino Progress: {i+1}/{len(self.test_data)} | Success: {success_rate:.1f}% | MAE: {current_mae:.6f}")
                
                time.sleep(PREDICTION_DELAY / 1000.0)
                
            except Exception as e:
                print(f"❌ Arduino Sample {i+1} Fehler: {e}")
                continue
        
        arduino.close()
        
        # Arduino Ergebnisse
        if self.arduino_results:
            final_mae = np.mean([r['error'] for r in self.arduino_results])
            print(f"✅ Arduino Test abgeschlossen!")
            print(f"📊 Erfolgreiche Predictions: {len(self.arduino_results)}/{len(self.test_data)}")
            print(f"🎯 Arduino MAE: {final_mae:.6f}")
            return True
        else:
            print("❌ Keine erfolgreichen Arduino Predictions")
            return False
      def test_pc(self):
        """Phase 2: PC Test"""
        print("\n" + "💻" + "="*60 + "💻")
        print("💻 PHASE 2: PC PYTORCH LSTM TEST 💻")
        print("💻" + "="*60 + "💻")
        
        if self.pc_model is None:
            print("❌ PC Modell nicht geladen")
            return False
        
        print(f"🧪 Starte PC Test mit {len(self.test_data)} Samples...")
        
        # 🔥 KRITISCH: PC LSTM Warmup für stateful Predictions!
        print("🔥 WARNUNG: PC LSTM ist STATEFUL - braucht Warmup!")
        if not self.warmup_pc_lstm(warmup_samples=100):
            print("⚠️ PC Warmup fehlgeschlagen - Resultate könnten ungenau sein!")
        
        self.pc_results = []
        
        # Reset Hidden State für fairen Vergleich
        batch_size = 1
        self.pc_hidden_state = (
            torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device),
            torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device)
        )
        
        for i, data_point in enumerate(self.test_data):
            try:
                # PC Vorhersage
                input_features = torch.tensor([[data_point['voltage'], data_point['current'], 
                                              data_point['soh'], data_point['q_c']]], 
                                            dtype=torch.float32, device=self.device)
                input_features = input_features.unsqueeze(0)
                
                with torch.no_grad():
                    soc_pred, self.pc_hidden_state = self.pc_model(input_features, self.pc_hidden_state)
                    soc_value = soc_pred.squeeze().cpu().item()
                    soc_value = np.clip(soc_value, 0, 1)
                
                self.pc_results.append({
                    'sample': i,
                    'voltage': data_point['voltage'],
                    'current': data_point['current'],
                    'soh': data_point['soh'],
                    'q_c': data_point['q_c'],
                    'ground_truth_soc': data_point['ground_truth_soc'],
                    'predicted_soc': soc_value,
                    'error': abs(soc_value - data_point['ground_truth_soc'])
                })
                  # Progress Print alle 100 Samples (PC ist schneller)
                if (i + 1) % 100 == 0:
                    current_mae = np.mean([r['error'] for r in self.pc_results])
                    print(f"💻 PC Progress: {i+1}/{len(self.test_data)} | MAE: {current_mae:.6f}")
                
                # KEIN Delay für PC - so schnell wie möglich!
                # time.sleep(0.01)  # Entfernt für Speed!
                
            except Exception as e:
                print(f"❌ PC Sample {i+1} Fehler: {e}")
                continue
        
        # PC Ergebnisse
        if self.pc_results:
            final_mae = np.mean([r['error'] for r in self.pc_results])
            print(f"✅ PC Test abgeschlossen!")
            print(f"📊 Erfolgreiche Predictions: {len(self.pc_results)}/{len(self.test_data)}")
            print(f"🎯 PC MAE: {final_mae:.6f}")
            return True
        else:
            print("❌ Keine erfolgreichen PC Predictions")
            return False
    
    def compare_results(self):
        """Phase 3: Vergleiche Ergebnisse"""
        print("\n" + "🏆" + "="*60 + "🏆")
        print("🏆 PHASE 3: ERGEBNISVERGLEICH 🏆")
        print("🏆" + "="*60 + "🏆")
        
        if not self.arduino_results or not self.pc_results:
            print("❌ Nicht genügend Daten für Vergleich")
            return
        
        # Statistiken berechnen
        arduino_mae = np.mean([r['error'] for r in self.arduino_results])
        pc_mae = np.mean([r['error'] for r in self.pc_results])
        
        arduino_rmse = np.sqrt(np.mean([r['error']**2 for r in self.arduino_results]))
        pc_rmse = np.sqrt(np.mean([r['error']**2 for r in self.pc_results]))
        
        # Gemeinsame Samples finden (für fairen Vergleich)
        arduino_samples = {r['sample']: r for r in self.arduino_results}
        pc_samples = {r['sample']: r for r in self.pc_results}
        common_samples = set(arduino_samples.keys()) & set(pc_samples.keys())
        
        print(f"📊 Arduino Samples: {len(self.arduino_results)}")
        print(f"📊 PC Samples: {len(self.pc_results)}")
        print(f"📊 Gemeinsame Samples: {len(common_samples)}")
        print()
        
        # Vergleichsstatistiken
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
        
        # Speichere Ergebnisse
        self.save_and_plot_results(common_samples, arduino_samples, pc_samples)
    
    def save_and_plot_results(self, common_samples, arduino_samples, pc_samples):
        """Speichere und plotte finale Ergebnisse"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Erstelle DataFrames für gemeinsame Samples
        comparison_data = []
        for sample_idx in sorted(common_samples):
            arduino_data = arduino_samples[sample_idx]
            pc_data = pc_samples[sample_idx]
            
            comparison_data.append({
                'sample': sample_idx,
                'ground_truth_soc': arduino_data['ground_truth_soc'],
                'arduino_soc': arduino_data['predicted_soc'],
                'pc_soc': pc_data['predicted_soc'],
                'arduino_error': arduino_data['error'],
                'pc_error': pc_data['error'],
                'voltage': arduino_data['voltage']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Speichere CSV
        csv_path = Path(f"c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/Arduino_Test_Setup/Stateful_32_32_comparison/sequential_comparison_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"💾 Ergebnisse gespeichert: {csv_path}")
        
        # Erstelle finalen Plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'🔥 ARDUINO vs PC SOC SEQUENTIAL COMPARISON 🔥\n{len(common_samples)} Samples', 
                     fontsize=18, fontweight='bold')
        
        # Plot 1: SOC Comparison - DER WICHTIGSTE!
        axes[0,0].plot(df['sample'], df['ground_truth_soc'], 'g-', linewidth=3, 
                      label='🎯 Ground Truth', alpha=0.9)
        axes[0,0].plot(df['sample'], df['arduino_soc'], 'r--', linewidth=2, 
                      label='🤖 Arduino LSTM', alpha=0.8)
        axes[0,0].plot(df['sample'], df['pc_soc'], 'b:', linewidth=2, 
                      label='💻 PC PyTorch', alpha=0.8)
        axes[0,0].set_title('🔥 SOC PREDICTION COMPARISON 🔥', fontweight='bold', fontsize=14)
        axes[0,0].set_ylabel('State of Charge', fontweight='bold')
        axes[0,0].set_xlabel('Sample Number', fontweight='bold')
        axes[0,0].legend(fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim(-0.05, 1.05)
        
        # Plot 2: Error Comparison
        axes[0,1].plot(df['sample'], df['arduino_error'], 'r-', linewidth=2, 
                      label='🤖 Arduino Error', alpha=0.8)
        axes[0,1].plot(df['sample'], df['pc_error'], 'b-', linewidth=2, 
                      label='💻 PC Error', alpha=0.8)
        axes[0,1].set_title('Prediction Errors', fontweight='bold', fontsize=14)
        axes[0,1].set_ylabel('Absolute Error', fontweight='bold')
        axes[0,1].set_xlabel('Sample Number', fontweight='bold')
        axes[0,1].legend(fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter Plot Arduino vs PC
        axes[1,0].scatter(df['arduino_soc'], df['pc_soc'], alpha=0.6, s=15)
        min_soc = min(df['arduino_soc'].min(), df['pc_soc'].min())
        max_soc = max(df['arduino_soc'].max(), df['pc_soc'].max())
        axes[1,0].plot([min_soc, max_soc], [min_soc, max_soc], 'k--', label='Perfect Agreement')
        axes[1,0].set_title('Arduino vs PC SOC Predictions', fontweight='bold', fontsize=14)
        axes[1,0].set_xlabel('Arduino SOC', fontweight='bold')
        axes[1,0].set_ylabel('PC SOC', fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Statistics Summary
        arduino_mae = df['arduino_error'].mean()
        pc_mae = df['pc_error'].mean()
        
        stats_text = f"""
🔥 FINAL COMPARISON RESULTS 🔥

📊 Total Samples: {len(df)}

🤖 ARDUINO LSTM:
   MAE: {arduino_mae:.6f}
   RMSE: {np.sqrt((df['arduino_error']**2).mean()):.6f}

💻 PC PYTORCH:
   MAE: {pc_mae:.6f}
   RMSE: {np.sqrt((df['pc_error']**2).mean()):.6f}

🏆 WINNER: {'Arduino' if arduino_mae < pc_mae else 'PC' if pc_mae < arduino_mae else 'TIE'}
📈 Difference: {abs(arduino_mae - pc_mae):.6f}
        """
        
        axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes,
                      fontsize=12, verticalalignment='top', fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1,1].set_title('📈 Final Statistics', fontweight='bold', fontsize=14)
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Speichere Plot
        plot_path = Path(f"c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/Arduino_Test_Setup/Stateful_32_32_comparison/sequential_comparison_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Finaler Plot gespeichert: {plot_path}")
        
        plt.show()
    
    def run_full_comparison(self):
        """Führe kompletten sequenziellen Vergleich durch"""
        print("🔥" + "="*80 + "🔥")
        print("🔥 ARDUINO vs PC SEQUENTIAL SOC COMPARISON 🔥")
        print("🔥" + "="*80 + "🔥")
        
        # Setup
        if not self.initialize_scaler():
            return False
        
        if not self.load_pc_model():
            return False
        
        if not self.prepare_test_data():
            return False
        
        # Phase 1: Arduino Test
        if not self.test_arduino():
            print("❌ Arduino Test fehlgeschlagen")
            return False
        
        # Phase 2: PC Test
        if not self.test_pc():
            print("❌ PC Test fehlgeschlagen")
            return False
        
        # Phase 3: Vergleich
        self.compare_results()
        
        print("\n🏁 SEQUENZIELLER VERGLEICH ABGESCHLOSSEN! 🏁")
        return True

    def warmup_arduino_lstm(self, arduino, warmup_samples=100):
        """🔥 KRITISCH: Arduino LSTM Warmup für stateful Predictions! 🔥"""
        print(f"🔥 WARMING UP Arduino LSTM mit {warmup_samples} Samples...")
        print("📍 GRUND: Stateful LSTM braucht kontinuierlichen Zustandsaufbau!")
        
        # Warmup mit Daten VOR dem Testbereich (kontinuierlich!)
        start_index = START_MINUTE * 60  # Test-Start
        warmup_start = max(0, start_index - warmup_samples)  # Warmup davor
        
        print(f"📊 Warmup: Sample {warmup_start} bis {start_index}")
        
        successful_warmups = 0
        for i in range(warmup_start, start_index):
            if i >= len(self.data):
                break
                
            sample = self.data.iloc[i]
            
            # Skaliere Features
            scaled_features = self.scale_features_for_arduino(
                sample['voltage'], sample['current'], 
                sample['soh'], sample['q_c']
            )
            
            # Arduino Warmup Command
            command = f"DATA:{scaled_features[0]:.6f},{scaled_features[1]:.6f},{scaled_features[2]:.6f},{scaled_features[3]:.6f}\n"
            arduino.write(command.encode())
            
            # Lese Response (aber ignoriere für Warmup)
            response = arduino.readline().decode().strip()
            if response and response.startswith("DATA:"):
                successful_warmups += 1
            
            # Kurze Pause zwischen Warmup Samples
            time.sleep(PREDICTION_DELAY / 1000.0)
            
            if (i - warmup_start + 1) % 20 == 0:
                progress = (i - warmup_start + 1) / warmup_samples * 100
                print(f"🔄 Warmup Progress: {progress:.0f}% ({successful_warmups}/{i - warmup_start + 1})")
        
        print(f"✅ Arduino LSTM Warmup abgeschlossen: {successful_warmups}/{warmup_samples} erfolgreich")
        print("🎯 LSTM-Zustand ist jetzt bereit für genaue Predictions!")
        return successful_warmups > 0

    def warmup_pc_lstm(self, warmup_samples=100):
        """🔥 KRITISCH: PC LSTM Warmup für stateful Predictions! 🔥"""
        print(f"🔥 WARMING UP PC LSTM mit {warmup_samples} Samples...")
        print("📍 GRUND: Stateful LSTM braucht kontinuierlichen Zustandsaufbau!")
        
        # Warmup mit Daten VOR dem Testbereich (kontinuierlich!)
        start_index = START_MINUTE * 60  # Test-Start
        warmup_start = max(0, start_index - warmup_samples)  # Warmup davor
        
        print(f"📊 PC Warmup: Sample {warmup_start} bis {start_index}")
        
        # Reset Hidden State
        batch_size = 1
        self.pc_hidden_state = (
            torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device),
            torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device)
        )
        
        successful_warmups = 0
        for i in range(warmup_start, start_index):
            if i >= len(self.data):
                break
                
            sample = self.data.iloc[i]
            
            # PC Warmup Prediction
            input_features = torch.tensor([[sample['voltage'], sample['current'], 
                                          sample['soh'], sample['q_c']]], 
                                        dtype=torch.float32, device=self.device)
            input_features = input_features.unsqueeze(0)
            
            with torch.no_grad():
                soc_pred, self.pc_hidden_state = self.pc_model(input_features, self.pc_hidden_state)
                successful_warmups += 1
            
            if (i - warmup_start + 1) % 20 == 0:
                progress = (i - warmup_start + 1) / warmup_samples * 100
                print(f"🔄 PC Warmup Progress: {progress:.0f}% ({successful_warmups}/{i - warmup_start + 1})")
        
        print(f"✅ PC LSTM Warmup abgeschlossen: {successful_warmups}/{warmup_samples} erfolgreich")
        print("🎯 PC LSTM-Zustand ist jetzt bereit für genaue Predictions!")
        return successful_warmups > 0

def main():
    """Main Function"""
    comparison = SequentialComparison()
    
    try:
        comparison.run_full_comparison()
    except KeyboardInterrupt:
        print("⏹️ Vergleich von Benutzer gestoppt")
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
