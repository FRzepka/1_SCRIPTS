"""
🔥 ARDUINO vs PC SOC PREDICTION SEQUENTIAL COMPARISON - FIXED! 🔥
================================================================

KRITISCHE ERKENNTNIS: STATEFUL LSTM braucht kontinuierlichen Zustandsaufbau!
- Warmup-Phase für beide LSTMs (Arduino + PC)
- Kontinuierliche Datenverarbeitung wie im funktionierenden Script
- PC Prediction EXAKT wie arduino_live_soc_prediction.py
- Arduino und PC getrennt aber korrekt implementiert

🚀 PROBLEM GELÖST: WARMUP + KONTINUIERLICHER ZUSTAND + KORREKTES PC PREDICTION! 🚀
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
import pickle

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

# Test Einstellungen - KRITISCH: Kontinuierlicher Zustandsaufbau!
START_MINUTE = 0       # Test startet ab 60 Minuten
TEST_DURATION_MINS = 300  # 15 Minuten Test
WARMUP_SAMPLES = 100     # 100 Samples Warmup für LSTM Zustandsaufbau
TEST_SAMPLES = TEST_DURATION_MINS * 60  # 900 Samples für 15min
PREDICTION_DELAY = 30    # 30ms zwischen Predictions (EXAKT wie funktionierendes Script!)

# Model parameters
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32

class SOCModel(nn.Module):
    """PC PyTorch LSTM SOC Model - EXAKT wie arduino_live_soc_prediction.py"""
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

class ArduinoVsPCComparison:
    """🔥 KORRIGIERTE Vergleichsklasse mit Warmup für stateful LSTM! 🔥"""
    
    def __init__(self):
        self.data = None
        self.test_data = None
        self.scaler = None
        self.pc_model = None
        self.pc_hidden_state = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results storage
        self.arduino_results = []
        self.pc_results = []
        
        print("✅ Arduino vs PC Comparison initialisiert")
        print(f"🖥️ Device: {self.device}")
        print(f"🔥 KRITISCH: Warmup für stateful LSTM aktiviert!")
        print(f"💻 PC Prediction wie arduino_live_soc_prediction.py!")
    
    def load_data(self):
        """Lade Daten EXAKT wie funktionierendes Script"""
        print("📊 Lade Ground Truth Daten...")
        
        if not Path(DATA_PATH).exists():
            print(f"❌ Daten nicht gefunden: {DATA_PATH}")
            return False
            
        df = pd.read_parquet(DATA_PATH)
        
        # Spalten prüfen
        required_cols = ['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_c', 'SOC_ZHU']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Fehlende Spalten: {missing_cols}")
            return False
        
        # Bereinigung EXAKT wie funktionierendes Script
        df = df.dropna(subset=required_cols)
        df = df[df['Voltage[V]'] > 0]
        
        # Umbennenung EXAKT wie funktionierendes Script
        self.data = df.rename(columns={
            'Voltage[V]': 'voltage',
            'Current[A]': 'current',
            'SOH_ZHU': 'soh',
            'Q_c': 'q_c',
            'SOC_ZHU': 'soc'
        })[['voltage', 'current', 'soh', 'q_c', 'soc']].copy()
        
        print(f"✅ Daten geladen: {len(self.data)} Samples")
        return True

    def setup_scaler(self):
        """Setup scaler EXAKT wie arduino_live_soc_prediction.py"""
        print("⚖️ Setup Scaler...")
        
        try:
            # Versuche bestehenden Scaler zu laden
            scaler_path = Path(DATA_PATH).parent / "scaler.pkl"
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✅ Bestehenden Scaler geladen")
                return True
        except Exception as e:
            print(f"⚠️ Scaler laden fehlgeschlagen: {e}")
        
        # Erstelle neuen Scaler EXAKT wie arduino_live_soc_prediction.py
        print("🔧 Erstelle neuen Scaler über alle Trainingszellen...")
        base_path = Path(r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU")
        
        train_cells = [
            "MGFarm_18650_C01",
            "MGFarm_18650_C03", 
            "MGFarm_18650_C05",
            "MGFarm_18650_C11",
            "MGFarm_18650_C17",
            "MGFarm_18650_C23"
        ]
        val_cells = [
            "MGFarm_18650_C07",
            "MGFarm_18650_C19", 
            "MGFarm_18650_C21"
        ]
        
        all_cells = train_cells + val_cells
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        
        # StandardScaler über alle Zellen fitten (exakt wie Train Script)
        self.scaler = StandardScaler()
        
        for cell_name in all_cells:
            cell_path = base_path / cell_name / "df.parquet"
            if cell_path.exists():
                df = pd.read_parquet(cell_path)
                self.scaler.partial_fit(df[feats])
                print(f"Partial fit für {cell_name}: {len(df)} Zeilen")
            else:
                print(f"⚠️ Datei nicht gefunden: {cell_path}")
        
        # Speichere für nächstes Mal
        try:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print("✅ Scaler gespeichert")
        except Exception as e:
            print(f"⚠️ Scaler speichern fehlgeschlagen: {e}")
        
        print("✅ Scaler erstellt (wie arduino_live_soc_prediction.py)")
        return True

    def scale_features(self, voltage, current, soh, q_c):
        """Scale features EXAKT wie funktionierendes Script"""
        features = np.array([[voltage, current, soh, q_c]])
        if self.scaler:
            return self.scaler.transform(features)[0]
        return features[0]

    def load_pc_model(self):
        """Lade PC PyTorch Modell EXAKT wie arduino_live_soc_prediction.py"""
        print(f"🧠 Lade PC PyTorch Modell...")
        
        if not Path(PC_MODEL_PATH).exists():
            print(f"❌ PC Modell nicht gefunden: {PC_MODEL_PATH}")
            return False
        
        try:
            # Erstelle Modell-Instanz EXAKT wie arduino_live_soc_prediction.py
            self.pc_model = SOCModel(input_size=4, dropout=0.05)
            
            # Lade gespeicherte Gewichte
            checkpoint = torch.load(PC_MODEL_PATH, map_location=self.device)
            self.pc_model.load_state_dict(checkpoint)
            
            # Modell auf Device und in Evaluations-Modus
            self.pc_model.to(self.device)
            self.pc_model.eval()
            
            # Initialisiere Hidden State EXAKT wie arduino_live_soc_prediction.py
            self.init_pc_hidden_state()
            
            print("✅ PC PyTorch Modell geladen (wie arduino_live_soc_prediction.py)")
            return True
            
        except Exception as e:
            print(f"❌ PC Modell laden fehlgeschlagen: {e}")
            return False

    def init_pc_hidden_state(self):
        """Initialisiere LSTM Hidden State EXAKT wie arduino_live_soc_prediction.py"""
        batch_size = 1
        self.pc_hidden_state = (
            torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device),
            torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device)
        )
        print("✅ PC LSTM Hidden State initialisiert")

    def predict_soc_pytorch(self, voltage, current, soh, q_c):
        """Echte SOC Vorhersage mit PyTorch LSTM Modell EXAKT wie arduino_live_soc_prediction.py"""
        if self.pc_model is None:
            print("❌ PC Modell nicht geladen!")
            return None, None
        
        start_time = time.time()
        
        try:
            # Input als Tensor vorbereiten (batch_size=1, seq_len=1, features=4)
            input_features = torch.tensor([[voltage, current, soh, q_c]], 
                                        dtype=torch.float32, device=self.device)
            input_features = input_features.unsqueeze(0)  # Shape: (1, 1, 4)
            
            with torch.no_grad():
                # Modell inference
                soc_pred, self.pc_hidden_state = self.pc_model(input_features, self.pc_hidden_state)
                
                # SOC Vorhersage extrahieren
                soc_value = soc_pred.squeeze().cpu().item()
                
                # Sicherstellen dass SOC im Bereich [0, 1] liegt
                soc_value = np.clip(soc_value, 0, 1)
            
            # Inference Zeit berechnen
            inference_time = (time.time() - start_time) * 1000  # in ms
            
            return soc_value, inference_time
            
        except Exception as e:
            print(f"❌ Fehler bei PC PyTorch Inference: {e}")
            return None, None

    def prepare_test_data(self):
        """🔥 KRITISCH: Kontinuierliche Testdaten für stateful LSTM! 🔥"""
        print("📊 Bereite kontinuierliche Testdaten vor...")
        print(f"🔥 KONTINUIERLICHER ZUSTAND: Warmup + Test sequenziell!")
        
        # Berechne Bereiche
        warmup_start_index = max(0, START_MINUTE * 60 - WARMUP_SAMPLES)
        test_start_index = START_MINUTE * 60
        test_end_index = test_start_index + TEST_SAMPLES
        
        print(f"📊 Warmup: Sample {warmup_start_index} bis {test_start_index}")
        print(f"📊 Test: Sample {test_start_index} bis {test_end_index}")
        
        if test_end_index > len(self.data):
            print(f"⚠️ Nicht genügend Daten! Verfügbar bis Sample {len(self.data)}")
            test_end_index = len(self.data)
            actual_test_samples = test_end_index - test_start_index
            print(f"📊 Angepasste Test Samples: {actual_test_samples}")
        
        # WICHTIG: Skaliere Features NACH der Auswahl (mit originalen Spaltennamen!)
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        
        # Erst Daten wieder in original Spaltennamen umwandeln
        data_orig = self.data.rename(columns={
            'voltage': 'Voltage[V]',
            'current': 'Current[A]',
            'soh': 'SOH_ZHU',
            'q_c': 'Q_c',
            'soc': 'SOC_ZHU'
        })
        
        # Dann skalieren
        data_orig[feats] = self.scaler.transform(data_orig[feats])
        
        # Kontinuierliche Testdaten extrahieren
        self.test_data = []
        for i in range(test_start_index, test_end_index):
            sample = data_orig.iloc[i]
            self.test_data.append({
                'voltage': float(sample['Voltage[V]']),     # Skaliert!
                'current': float(sample['Current[A]']),    # Skaliert!
                'soh': float(sample['SOH_ZHU']),          # Skaliert!
                'q_c': float(sample['Q_c']),              # Skaliert!
                'soc_true': float(sample['SOC_ZHU'])      # Original!
            })
        
        print(f"✅ {len(self.test_data)} kontinuierliche Testdaten bereit (Features skaliert)")
        return True

    def warmup_arduino_lstm(self, arduino, warmup_samples=WARMUP_SAMPLES):
        """🔥 KRITISCH: Arduino LSTM Warmup für kontinuierlichen Zustand! 🔥"""
        print(f"🔥 ARDUINO WARMUP mit {warmup_samples} Samples...")
        print("📍 GRUND: Stateful LSTM braucht kontinuierlichen Zustandsaufbau!")
        
        # Warmup mit Daten VOR dem Testbereich
        warmup_start = max(0, START_MINUTE * 60 - warmup_samples)
        warmup_end = START_MINUTE * 60
        
        print(f"📊 Arduino Warmup: Sample {warmup_start} bis {warmup_end}")
        
        # WICHTIG: Skaliere Features für Arduino auch!
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        
        # Erst Daten wieder in original Spaltennamen umwandeln
        data_orig = self.data.rename(columns={
            'voltage': 'Voltage[V]',
            'current': 'Current[A]',
            'soh': 'SOH_ZHU',
            'q_c': 'Q_c',
            'soc': 'SOC_ZHU'
        })
        
        # Dann skalieren
        data_orig[feats] = self.scaler.transform(data_orig[feats])
        
        successful_warmups = 0
        for i in range(warmup_start, warmup_end):
            if i >= len(data_orig):
                break
                
            sample = data_orig.iloc[i]
            
            # Arduino Warmup Command EXAKT wie funktionierendes Script (mit skalierten Features!)
            command = f"DATA:{sample['Voltage[V]']:.6f},{sample['Current[A]']:.6f},{sample['SOH_ZHU']:.6f},{sample['Q_c']:.6f}\n"
            arduino.write(command.encode())
            
            response = arduino.readline().decode().strip()
            if response and response.startswith("DATA:"):
                successful_warmups += 1
            
            time.sleep(PREDICTION_DELAY / 1000.0)
            
            if (i - warmup_start + 1) % 20 == 0:
                progress = (i - warmup_start + 1) / warmup_samples * 100
                print(f"🔄 Arduino Warmup: {progress:.0f}% ({successful_warmups}/{i - warmup_start + 1})")
        
        print(f"✅ Arduino LSTM Warmup: {successful_warmups}/{warmup_samples} erfolgreich")
        return successful_warmups > 0

    def warmup_pc_lstm(self, warmup_samples=WARMUP_SAMPLES):
        """🔥 KRITISCH: PC LSTM Warmup für kontinuierlichen Zustand! 🔥"""
        print(f"🔥 PC WARMUP mit {warmup_samples} Samples...")
        print("📍 GRUND: Stateful LSTM braucht kontinuierlichen Zustandsaufbau!")
        
        # Warmup mit Daten VOR dem Testbereich
        warmup_start = max(0, START_MINUTE * 60 - warmup_samples)
        warmup_end = START_MINUTE * 60
        
        print(f"📊 PC Warmup: Sample {warmup_start} bis {warmup_end}")
        
        # Reset Hidden State
        self.init_pc_hidden_state()
        
        # WICHTIG: Skaliere Features für PC auch!
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        
        # Erst Daten wieder in original Spaltennamen umwandeln
        data_orig = self.data.rename(columns={
            'voltage': 'Voltage[V]',
            'current': 'Current[A]',
            'soh': 'SOH_ZHU',
            'q_c': 'Q_c',
            'soc': 'SOC_ZHU'
        })
        
        # Dann skalieren
        data_orig[feats] = self.scaler.transform(data_orig[feats])
        
        successful_warmups = 0
        for i in range(warmup_start, warmup_end):
            if i >= len(data_orig):
                break
                
            sample = data_orig.iloc[i]
            
            # PC Warmup Prediction (mit skalierten Features!)
            soc_pred, inference_time = self.predict_soc_pytorch(
                sample['Voltage[V]'], sample['Current[A]'], 
                sample['SOH_ZHU'], sample['Q_c']
            )
            
            if soc_pred is not None:
                successful_warmups += 1
            
            if (i - warmup_start + 1) % 20 == 0:
                progress = (i - warmup_start + 1) / warmup_samples * 100
                print(f"🔄 PC Warmup: {progress:.0f}% ({successful_warmups}/{i - warmup_start + 1})")
        
        print(f"✅ PC LSTM Warmup: {successful_warmups}/{warmup_samples} erfolgreich")
        return successful_warmups > 0

    def test_arduino(self):
        """Phase 1: Arduino Test mit Warmup"""
        print("\n" + "🤖" + "="*60 + "🤖")
        print("🤖 PHASE 1: ARDUINO HARDWARE LSTM TEST 🤖")
        print("🤖" + "="*60 + "🤖")
        
        try:
            print(f"🔌 Verbinde zu Arduino auf {ARDUINO_PORT}...")
            arduino = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=3)
            time.sleep(3)
            
            # Test Verbindung
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
        
        # 🔥 KRITISCH: Arduino LSTM Warmup!
        print("🔥 WARNUNG: Arduino LSTM ist STATEFUL - braucht Warmup!")
        if not self.warmup_arduino_lstm(arduino, WARMUP_SAMPLES):
            print("⚠️ Arduino Warmup fehlgeschlagen - Resultate könnten ungenau sein!")
        
        # Arduino Test durchführen
        print(f"🧪 Starte Arduino Test mit {len(self.test_data)} Samples...")
        print(f"⏱️ Geschätzte Dauer: {(len(self.test_data) * PREDICTION_DELAY) / 1000 / 60:.1f} Minuten")
        
        self.arduino_results = []
        successful_predictions = 0
        
        for i, data_point in enumerate(self.test_data):
            try:
                # Arduino Protokoll EXAKT wie funktionierendes Script (Features bereits skaliert!)
                command = f"DATA:{data_point['voltage']:.6f},{data_point['current']:.6f},{data_point['soh']:.6f},{data_point['q_c']:.6f}\n"
                arduino.write(command.encode())
                
                # Lese Antwort EXAKT wie funktionierendes Script
                response = arduino.readline().decode().strip()
                
                if response and response.startswith("DATA:"):
                    data = response.replace("DATA:", "").split(",")
                    if len(data) >= 1:
                        try:
                            soc_value = float(data[0])
                            soc_value = np.clip(soc_value, 0, 1)
                            
                            self.arduino_results.append({
                                'sample': i,
                                'voltage': data_point['voltage'],
                                'soc_pred': soc_value,
                                'soc_true': data_point['soc_true'],
                                'mae_error': abs(soc_value - data_point['soc_true'])
                            })
                            successful_predictions += 1
                            
                        except ValueError as e:
                            print(f"⚠️ Arduino SOC Parse Fehler: {e}")
                            
                # Progress Update
                if (i + 1) % 100 == 0:
                    progress = (i + 1) / len(self.test_data) * 100
                    current_mae = np.mean([r['mae_error'] for r in self.arduino_results]) if len(self.arduino_results) > 0 else 0
                    print(f"🔄 Arduino Progress: {progress:.1f}% - MAE: {current_mae:.6f} - Success: {successful_predictions}/{i+1}")
                
                # Delay EXAKT wie funktionierendes Script
                time.sleep(PREDICTION_DELAY / 1000.0)
                
            except Exception as e:
                print(f"⚠️ Arduino Prediction Fehler: {e}")
        
        arduino.close()
        
        if len(self.arduino_results) > 0:
            final_mae = np.mean([r['mae_error'] for r in self.arduino_results])
            print(f"✅ Arduino Test abgeschlossen!")
            print(f"📊 Erfolgreiche Predictions: {len(self.arduino_results)}/{len(self.test_data)}")
            print(f"🎯 Arduino MAE: {final_mae:.6f}")
            return True
        else:
            print("❌ Keine erfolgreichen Arduino Predictions")
            return False

    def test_pc(self):
        """Phase 2: PC Test mit Warmup EXAKT wie arduino_live_soc_prediction.py"""
        print("\n" + "💻" + "="*60 + "💻")
        print("💻 PHASE 2: PC PYTORCH LSTM TEST 💻")
        print("💻" + "="*60 + "💻")
        
        if self.pc_model is None:
            print("❌ PC Modell nicht geladen")
            return False
        
        print(f"🧪 Starte PC Test mit {len(self.test_data)} Samples...")
        
        # 🔥 KRITISCH: PC LSTM Warmup!
        print("🔥 WARNUNG: PC LSTM ist STATEFUL - braucht Warmup!")
        if not self.warmup_pc_lstm(WARMUP_SAMPLES):
            print("⚠️ PC Warmup fehlgeschlagen - Resultate könnten ungenau sein!")
        
        self.pc_results = []
        
        for i, data_point in enumerate(self.test_data):
            try:
                # PC Prediction EXAKT wie arduino_live_soc_prediction.py (Features bereits skaliert!)
                soc_pred, inference_time = self.predict_soc_pytorch(
                    data_point['voltage'], data_point['current'], 
                    data_point['soh'], data_point['q_c']
                )
                
                if soc_pred is not None:
                    self.pc_results.append({
                        'sample': i,
                        'voltage': data_point['voltage'],
                        'soc_pred': soc_pred,
                        'soc_true': data_point['soc_true'],
                        'mae_error': abs(soc_pred - data_point['soc_true']),
                        'inference_time': inference_time
                    })
                
                # Progress Update
                if (i + 1) % 100 == 0:
                    progress = (i + 1) / len(self.test_data) * 100
                    current_mae = np.mean([r['mae_error'] for r in self.pc_results]) if len(self.pc_results) > 0 else 0
                    avg_inference = np.mean([r['inference_time'] for r in self.pc_results]) if len(self.pc_results) > 0 else 0
                    print(f"🔄 PC Progress: {progress:.1f}% - MAE: {current_mae:.6f} - Avg Inference: {avg_inference:.2f}ms")
                
            except Exception as e:
                print(f"⚠️ PC Prediction Fehler: {e}")
        
        if len(self.pc_results) > 0:
            final_mae = np.mean([r['mae_error'] for r in self.pc_results])
            avg_inference = np.mean([r['inference_time'] for r in self.pc_results])
            print(f"✅ PC Test abgeschlossen!")
            print(f"📊 Erfolgreiche Predictions: {len(self.pc_results)}/{len(self.test_data)}")
            print(f"🎯 PC MAE: {final_mae:.6f}")
            print(f"⚡ PC Avg Inference Time: {avg_inference:.2f}ms")
            return True
        else:
            print("❌ Keine erfolgreichen PC Predictions")
            return False

    def compare_and_plot(self):
        """Phase 3: Vergleich und Plot MIT SPEICHERUNG"""
        print("\n" + "📊" + "="*60 + "📊")
        print("📊 PHASE 3: VERGLEICH UND ANALYSE 📊")
        print("📊" + "="*60 + "📊")
        
        if len(self.arduino_results) == 0 or len(self.pc_results) == 0:
            print("❌ Nicht genügend Daten für Vergleich")
            return False
        
        # Berechne Statistiken
        arduino_mae = np.mean([r['mae_error'] for r in self.arduino_results])
        pc_mae = np.mean([r['mae_error'] for r in self.pc_results])
        
        arduino_soc = [r['soc_pred'] for r in self.arduino_results]
        pc_soc = [r['soc_pred'] for r in self.pc_results]
        true_soc = [r['soc_true'] for r in self.arduino_results]
        samples = list(range(len(arduino_soc)))
        
        print(f"🎯 FINALE ERGEBNISSE:")
        print(f"   Arduino MAE: {arduino_mae:.6f}")
        print(f"   PC MAE:      {pc_mae:.6f}")
        
        if arduino_mae < pc_mae:
            winner = "🤖 ARDUINO LSTM"
            improvement = ((pc_mae - arduino_mae) / pc_mae * 100)
        elif pc_mae < arduino_mae:
            winner = "💻 PC PyTorch"
            improvement = ((arduino_mae - pc_mae) / arduino_mae * 100)
        else:
            winner = "🤝 TIE"
            improvement = 0
            
        print(f"   🏆 WINNER: {winner}")
        if improvement > 0:
            print(f"   📈 Verbesserung: {improvement:.2f}%")
        
        # Erstelle Timestamp für Dateien
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Speichere Daten als CSV
        comparison_data = []
        for i in range(len(arduino_soc)):
            comparison_data.append({
                'sample': i,
                'ground_truth_soc': true_soc[i],
                'arduino_soc': arduino_soc[i],
                'pc_soc': pc_soc[i],
                'arduino_error': abs(arduino_soc[i] - true_soc[i]),
                'pc_error': abs(pc_soc[i] - true_soc[i])
            })
        
        df = pd.DataFrame(comparison_data)
        
        # CSV speichern
        csv_path = Path(f"c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/Arduino_Test_Setup/Stateful_32_32_comparison/sequential_comparison_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"💾 Ergebnisse gespeichert: {csv_path}")
        
        # Plot erstellen
        plt.figure(figsize=(20, 12))
        
        # Plot 1: SOC Vergleich
        plt.subplot(2, 2, 1)
        plt.plot(samples, true_soc, 'g-', linewidth=3, label='🎯 Ground Truth', alpha=0.9)
        plt.plot(samples, arduino_soc, 'r--', linewidth=2, label='🤖 Arduino LSTM', alpha=0.8)
        plt.plot(samples, pc_soc, 'b:', linewidth=2, label='💻 PC PyTorch', alpha=0.8)
        plt.title('🔥 SOC PREDICTION COMPARISON 🔥', fontweight='bold', fontsize=16)
        plt.ylabel('State of Charge', fontweight='bold')
        plt.xlabel('Sample Number', fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 2: MAE Fehler
        plt.subplot(2, 2, 2)
        arduino_errors = [r['mae_error'] for r in self.arduino_results]
        pc_errors = [r['mae_error'] for r in self.pc_results]
        
        plt.plot(samples, arduino_errors, 'r-', linewidth=1, label=f'🤖 Arduino MAE: {arduino_mae:.6f}', alpha=0.7)
        plt.plot(samples, pc_errors, 'b-', linewidth=1, label=f'💻 PC MAE: {pc_mae:.6f}', alpha=0.7)
        plt.title('📊 MEAN ABSOLUTE ERROR COMPARISON', fontweight='bold')
        plt.ylabel('MAE', fontweight='bold')
        plt.xlabel('Sample Number', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Scatter Plot Arduino vs PC
        plt.subplot(2, 2, 3)
        plt.scatter(arduino_soc, pc_soc, alpha=0.6, s=15)
        min_soc = min(min(arduino_soc), min(pc_soc))
        max_soc = max(max(arduino_soc), max(pc_soc))
        plt.plot([min_soc, max_soc], [min_soc, max_soc], 'k--', label='Perfect Agreement')
        plt.title('Arduino vs PC SOC Predictions', fontweight='bold')
        plt.xlabel('Arduino SOC', fontweight='bold')
        plt.ylabel('PC SOC', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Statistics Summary
        stats_text = f"""
🔥 FINAL COMPARISON RESULTS 🔥

📊 Total Samples: {len(df)}
⏱️ Test Duration: {TEST_DURATION_MINS} minutes
🔥 Warmup Samples: {WARMUP_SAMPLES}

🤖 ARDUINO LSTM:
   MAE: {arduino_mae:.6f}
   RMSE: {np.sqrt((df['arduino_error']**2).mean()):.6f}

💻 PC PYTORCH:
   MAE: {pc_mae:.6f}  
   RMSE: {np.sqrt((df['pc_error']**2).mean()):.6f}

🏆 WINNER: {winner}
📈 Improvement: {improvement:.2f}%
"""
        
        plt.subplot(2, 2, 4)
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.title('📈 Final Statistics', fontweight='bold')
        plt.axis('off')
        
        plt.suptitle(f'🔥 ARDUINO vs PC SOC SEQUENTIAL COMPARISON - {timestamp} 🔥', 
                     fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Plot speichern
        plot_path = Path(f"c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/Arduino_Test_Setup/Stateful_32_32_comparison/sequential_comparison_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Finaler Plot gespeichert: {plot_path}")
        
        plt.show()
        
        return True

    def run_full_comparison(self):
        """Führe vollständigen Vergleich durch"""
        print("🚀" * 20)
        print("🔥 ARDUINO vs PC SOC PREDICTION COMPARISON - FIXED! 🔥")
        print("🚀" * 20)
        print(f"📍 KRITISCHE KORREKTUR: Warmup für stateful LSTM!")
        print(f"💻 PC Prediction EXAKT wie arduino_live_soc_prediction.py!")
        print(f"📊 Test Setup: {START_MINUTE}min + {WARMUP_SAMPLES} Warmup + {TEST_DURATION_MINS}min Test")
        
        # Phase 0: Daten laden
        if not self.load_data():
            return False
        
        if not self.setup_scaler():
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
        if not self.compare_and_plot():
            print("❌ Vergleich fehlgeschlagen")
            return False
        
        print("✅ Vollständiger Vergleich abgeschlossen!")
        return True

def main():
    """Main function"""
    comparison = ArduinoVsPCComparison()
    
    try:
        comparison.run_full_comparison()
    except KeyboardInterrupt:
        print("⏹️ Vergleich durch Benutzer gestoppt")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        raise

if __name__ == "__main__":
    main()
