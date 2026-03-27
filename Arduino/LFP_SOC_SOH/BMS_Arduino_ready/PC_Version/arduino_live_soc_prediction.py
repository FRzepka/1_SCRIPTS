"""
BMS Arduino Live SOC Prediction System
======================================

PC Version für Live SOC Vorhersage mit Arduino LSTM-Hardware
- Real-time SOC prediction mit Ground Truth Vergleich  
- MAE Berechnung und kontinuierliches Monitoring
- Daten-Skalierung exakt wie 1.2.4 Train Script
- Voltage, Current, Inference Time und Error Monitoring

Features: Voltage[V], Current[A], SOH_ZHU, Q_c -> SOC_ZHU
Validierung: MGFarm C19 Cell Data
"""

import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import threading
import queue
from collections import deque
import logging
import torch
import torch.nn as nn

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model parameters from training script
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32

class SOCModel(nn.Module):
    """
    LSTM SOC Model - identisch zum Training Script
    Architecture: LSTM(4→32) + MLP(32→32→32→1)
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

class ArduinoSOCPredictor:
    def __init__(self, 
                 port='COM13', 
                 baudrate=115200,
                 data_path=r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet",
                 model_path=r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\SOC\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\training_run_1_2_4_31\best_model.pth"):
        
        self.port = port
        self.baudrate = baudrate
        self.data_path = data_path
        self.model_path = model_path
        self.serial_conn = None
        self.scaler = None
        
        # PyTorch model and device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.hidden_state = None
        
        # Data storage
        self.ground_truth_data = None
        self.current_index = 0
        
        # Real-time metrics
        self.predictions = deque(maxlen=1000)
        self.ground_truths = deque(maxlen=1000)
        self.voltages = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        self.errors = deque(maxlen=1000)
        
        # Threading
        self.running = False
        self.data_queue = queue.Queue()
        
        logger.info(f"ArduinoSOCPredictor initialisiert - Device: {self.device}")
    
    def load_model(self):
        """
        Lade das trainierte PyTorch Modell
        """
        logger.info(f"Lade trainiertes Modell: {self.model_path}")
        
        if not Path(self.model_path).exists():
            logger.error(f"Modell-Datei nicht gefunden: {self.model_path}")
            return False
        
        try:
            # Erstelle Modell-Instanz
            self.model = SOCModel(input_size=4, dropout=0.05)
            
            # Lade gespeicherte Gewichte
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            
            # Modell auf Device und in Evaluations-Modus
            self.model.to(self.device)
            self.model.eval()
            
            # Initialisiere Hidden State
            self.init_hidden_state()
            
            logger.info("Modell erfolgreich geladen!")
            logger.info(f"Model parameters: HIDDEN_SIZE={HIDDEN_SIZE}, NUM_LAYERS={NUM_LAYERS}, MLP_HIDDEN={MLP_HIDDEN}")
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            return False
    
    def init_hidden_state(self):
        """
        Initialisiere LSTM Hidden State
        """
        batch_size = 1
        self.hidden_state = (
            torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device),
            torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=self.device)
        )
        logger.info("LSTM Hidden State initialisiert")
    
    def initialize_scaler(self):
        """
        Erstelle den EXAKT gleichen StandardScaler wie im 1.2.4 Train Script
        """
        logger.info("Initialisiere StandardScaler...")
        
        # Exakt die gleiche Zellaufteilung wie im Train Script
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
        
        logger.info("Berechne StandardScaler über alle Trainingszellen...")
        for cell_name in all_cells:
            cell_path = base_path / cell_name / "df.parquet"
            if cell_path.exists():
                df = pd.read_parquet(cell_path)
                df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
                self.scaler.partial_fit(df[feats])
                logger.info(f"Partial fit für {cell_name}: {len(df)} Zeilen")
            else:
                logger.warning(f"Datei nicht gefunden: {cell_path}")
        
        # Debug: Scaler Parameter ausgeben
        scaler_params = dict(zip(feats, self.scaler.scale_))
        logger.info(f"Scaler scale_ parameter: {scaler_params}")
        return True
    
    def load_ground_truth_data(self):
        """
        Lade C19 Ground Truth Daten für Validierung
        """
        logger.info(f"Lade Ground Truth Daten: {self.data_path}")
        if not Path(self.data_path).exists():
            logger.error(f"Ground Truth Datei nicht gefunden: {self.data_path}")
            return False
        
        # Lade C19 Daten
        df = pd.read_parquet(self.data_path)
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        
        # Prüfe ob alle benötigten Spalten vorhanden sind
        required_cols = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c", "SOC_ZHU"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Fehlende Spalten in Ground Truth Daten: {missing_cols}")
            logger.info(f"Verfügbare Spalten: {list(df.columns)}")
            return False
          # Skaliere Features (exakt wie Train Script) - nur wenn Scaler initialisiert
        if self.scaler is None:
            logger.error("Scaler nicht initialisiert! Rufe initialize_scaler() zuerst auf.")
            return False
        
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        df[feats] = self.scaler.transform(df[feats])
        
        self.ground_truth_data = df
        logger.info(f"Ground Truth Daten geladen: {len(df)} Datenpunkte")
        logger.info(f"Voltage range: {df['Voltage[V]'].min():.3f} - {df['Voltage[V]'].max():.3f}")
        logger.info(f"SOC range: {df['SOC_ZHU'].min():.3f} - {df['SOC_ZHU'].max():.3f}")
        
        return True
    
    def predict_soc_pytorch(self, voltage, current, soh, q_c):
        """
        Echte SOC Vorhersage mit PyTorch LSTM Modell
        """
        if self.model is None:
            logger.error("Modell nicht geladen!")
            return None, None
        
        start_time = time.time()
        
        try:
            # Input als Tensor vorbereiten (batch_size=1, seq_len=1, features=4)
            input_features = torch.tensor([[voltage, current, soh, q_c]], 
                                        dtype=torch.float32, device=self.device)
            input_features = input_features.unsqueeze(0)  # Shape: (1, 1, 4)
            
            with torch.no_grad():
                # Modell inference
                soc_pred, self.hidden_state = self.model(input_features, self.hidden_state)
                
                # SOC Vorhersage extrahieren
                soc_value = soc_pred.squeeze().cpu().item()
                
                # Sicherstellen dass SOC im Bereich [0, 1] liegt
                soc_value = np.clip(soc_value, 0, 1)
              # Inference Zeit berechnen
            inference_time = (time.time() - start_time) * 1000  # in ms
            
            return soc_value, inference_time
            
        except Exception as e:
            logger.error(f"Fehler bei PyTorch Inference: {e}")
            return None, None
    
    def run_live_prediction(self, max_samples=None, delay=0.01):
        """
        Führe Live SOC Vorhersage durch mit Live Plotting
        max_samples=None für kontinuierliches Monitoring
        """
        if self.ground_truth_data is None:
            logger.error("Ground Truth Daten nicht geladen!")
            return
        
        # Setup Live Plotting - größeres Fenster
        plt.ion()  # Interactive mode
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Live SOC Monitor - Real-time PyTorch LSTM Prediction', fontsize=18)        
        # Initialize empty lists for plotting - zeitbasiert
        time_data, pred_data, truth_data, voltage_data, error_data = [], [], [], [], []
        start_time = time.time()
        
        # Initialize plot lines
        line_pred, = axes[0,0].plot([], [], 'b-', label='SOC Prediction', linewidth=2)
        line_truth, = axes[0,0].plot([], [], 'r-', label='Ground Truth', linewidth=2)
        axes[0,0].set_title('SOC Prediction vs Ground Truth', fontsize=14)
        axes[0,0].set_xlabel('Zeit (s)')
        axes[0,0].set_ylabel('SOC')
        axes[0,0].legend(fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        
        line_voltage, = axes[0,1].plot([], [], 'g-', label='Voltage', linewidth=2)
        axes[0,1].set_title('Battery Voltage', fontsize=14)
        axes[0,1].set_xlabel('Zeit (s)')
        axes[0,1].set_ylabel('Voltage [V] (scaled)')
        axes[0,1].legend(fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        line_error, = axes[1,0].plot([], [], 'orange', label='Absolute Error', linewidth=2)
        axes[1,0].set_title('Prediction Error', fontsize=14)
        axes[1,0].set_xlabel('Zeit (s)')
        axes[1,0].set_ylabel('Absolute Error')
        axes[1,0].legend(fontsize=12)
        axes[1,0].grid(True, alpha=0.3)        
        # MAE text display - größere Schrift
        mae_text = axes[1,1].text(0.5, 0.8, '', transform=axes[1,1].transAxes, 
                                 fontsize=24, ha='center', va='center', weight='bold')
        inference_text = axes[1,1].text(0.5, 0.6, '', transform=axes[1,1].transAxes,
                                       fontsize=18, ha='center', va='center')
        sample_text = axes[1,1].text(0.5, 0.4, '', transform=axes[1,1].transAxes,
                                    fontsize=16, ha='center', va='center')
        speed_text = axes[1,1].text(0.5, 0.2, '', transform=axes[1,1].transAxes,
                                   fontsize=14, ha='center', va='center', color='blue')
        axes[1,1].set_title('Live Metrics', fontsize=14)
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')        
        if max_samples is None:
            logger.info(f"🚀 STARTE OPTIMIERTEN Live Monitor - {delay*1000:.0f}ms Delay (Schließe Plot-Fenster zum Stoppen)")
        else:
            logger.info(f"🚀 STARTE OPTIMIERTEN Live Monitor - Max {max_samples} Samples, {delay*1000:.0f}ms Delay")
        
        self.running = True
        sample_count = 0
        window_size = 300  # Zeige letzten 300 Datenpunkte (5 Minuten bei 50ms)
        plot_update_interval = 5  # Update Plot nur alle 5 Samples für Performance
        
        try:
            while self.running and (max_samples is None or sample_count < max_samples):
                # Check if plot window is still open
                if not plt.fignum_exists(fig.number):
                    logger.info("Plot-Fenster geschlossen - beende Monitor")
                    break
                    
                # Hole nächsten Datenpunkt
                if self.current_index >= len(self.ground_truth_data):
                    logger.info("Alle Daten durchlaufen - starte von vorne")
                    self.current_index = 0
                
                row = self.ground_truth_data.iloc[self.current_index]
                
                # Extrahiere Features (bereits skaliert)
                voltage = row["Voltage[V]"]
                current = row["Current[A]"]
                soh = row["SOH_ZHU"]
                q_c = row["Q_c"]
                ground_truth_soc = row["SOC_ZHU"]
                  # Hole PyTorch Vorhersage
                pred_soc, inference_time = self.predict_soc_pytorch(voltage, current, soh, q_c)
                
                if pred_soc is not None:
                    # Berechne Error
                    error = abs(pred_soc - ground_truth_soc)
                      # Aktueller Timestamp
                    current_time = time.time() - start_time
                    
                    # Speichere Metriken
                    self.predictions.append(pred_soc)
                    self.ground_truths.append(ground_truth_soc)
                    self.voltages.append(voltage)
                    self.inference_times.append(inference_time)
                    self.errors.append(error)
                    
                    # Update plot data mit Zeit
                    time_data.append(current_time)
                    pred_data.append(pred_soc)
                    truth_data.append(ground_truth_soc)
                    voltage_data.append(voltage)
                    error_data.append(error)
                    
                    # Keep only last window_size points for plotting
                    if len(time_data) > window_size:
                        time_data = time_data[-window_size:]
                        pred_data = pred_data[-window_size:]
                        truth_data = truth_data[-window_size:]
                        voltage_data = voltage_data[-window_size:]
                        error_data = error_data[-window_size:]
                    
                    # Update plots nur alle plot_update_interval Samples für Performance
                    if sample_count % plot_update_interval == 0:
                        line_pred.set_data(time_data, pred_data)
                        line_truth.set_data(time_data, truth_data)
                        line_voltage.set_data(time_data, voltage_data)
                        line_error.set_data(time_data, error_data)
                        
                        # Auto-scale axes
                        for ax in [axes[0,0], axes[0,1], axes[1,0]]:
                            ax.relim()
                            ax.autoscale_view()
                        
                        # Update metrics text
                        current_mae = np.mean(list(self.errors)) if self.errors else 0
                        avg_inference_time = np.mean(list(self.inference_times)) if self.inference_times else 0
                        samples_per_sec = sample_count / current_time if current_time > 0 else 0
                        
                        mae_text.set_text(f'MAE: {current_mae:.6f}')
                        inference_text.set_text(f'Avg Inference: {avg_inference_time:.1f} ms')
                        sample_text.set_text(f'Samples: {sample_count} | Zeit: {current_time:.1f}s')
                        speed_text.set_text(f'Speed: {samples_per_sec:.1f} samples/s')
                        
                        # Update display
                        plt.pause(0.001)  # Minimaler Pause für Plot-Update                    
                    # Console Output weniger häufig für Performance
                    if sample_count % 100 == 0:
                        samples_per_sec = sample_count / current_time if current_time > 0 else 0
                        logger.info(f"⚡ Sample {sample_count:4d}: SOC_pred={pred_soc:.4f}, SOC_true={ground_truth_soc:.4f}, "
                                  f"Error={error:.4f}, MAE={current_mae:.4f}, Speed={samples_per_sec:.1f}/s")
                    
                    sample_count += 1
                    self.current_index += 1
                    
                    # Delay nur wenn nicht Plot-Update
                    if sample_count % plot_update_interval != 0:
                        time.sleep(delay)
                else:
                    logger.warning(f"Sample {sample_count} übersprungen - Vorhersage Fehler")
                    self.current_index += 1
        
        except KeyboardInterrupt:
            logger.info("Live Monitor von Benutzer gestoppt")
        
        finally:
            self.running = False
            plt.ioff()  # Turn off interactive mode
            logger.info(f"Live Monitor beendet - {sample_count} Samples verarbeitet")
    
    def calculate_metrics(self):
        """
        Berechne finale Metriken
        """
        if len(self.predictions) == 0:
            logger.warning("Keine Vorhersagen verfügbar für Metriken")
            return {}
        
        predictions = np.array(list(self.predictions))
        ground_truths = np.array(list(self.ground_truths))
        errors = np.array(list(self.errors))
        inference_times = np.array(list(self.inference_times))
        
        metrics = {
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'avg_inference_time': np.mean(inference_times),
            'max_inference_time': np.max(inference_times),
            'min_inference_time': np.min(inference_times),
            'total_samples': len(predictions)
        }
        
        logger.info("=== FINALE METRIKEN ===")
        logger.info(f"MAE: {metrics['mae']:.6f}")
        logger.info(f"RMSE: {metrics['rmse']:.6f}")
        logger.info(f"Max Error: {metrics['max_error']:.6f}")
        logger.info(f"Min Error: {metrics['min_error']:.6f}")
        logger.info(f"Avg Inference Time: {metrics['avg_inference_time']:.2f} ms")
        logger.info(f"Max Inference Time: {metrics['max_inference_time']:.2f} ms")
        logger.info(f"Total Samples: {metrics['total_samples']}")
        
        return metrics
    
    def plot_results(self, save_path=None):
        """
        Plotte Ergebnisse
        """
        if len(self.predictions) == 0:
            logger.warning("Keine Daten zum Plotten verfügbar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Zeitreihen für SOC
        x = range(len(self.predictions))
        axes[0,0].plot(x, list(self.predictions), 'b-', label='Arduino Prediction', alpha=0.7)
        axes[0,0].plot(x, list(self.ground_truths), 'r-', label='Ground Truth', alpha=0.7)
        axes[0,0].set_title('SOC Prediction vs Ground Truth')
        axes[0,0].set_xlabel('Sample')
        axes[0,0].set_ylabel('SOC')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Error über Zeit
        axes[0,1].plot(x, list(self.errors), 'g-', alpha=0.7)
        axes[0,1].set_title(f'Absolute Error (MAE: {np.mean(list(self.errors)):.6f})')
        axes[0,1].set_xlabel('Sample')
        axes[0,1].set_ylabel('Absolute Error')
        axes[0,1].grid(True, alpha=0.3)
        
        # Scatter Plot: Prediction vs Truth
        axes[1,0].scatter(list(self.ground_truths), list(self.predictions), alpha=0.6, s=2)
        min_val = min(min(self.ground_truths), min(self.predictions))
        max_val = max(max(self.ground_truths), max(self.predictions))
        axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        axes[1,0].set_title('Prediction vs Ground Truth')
        axes[1,0].set_xlabel('Ground Truth SOC')
        axes[1,0].set_ylabel('Predicted SOC')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
          # Inference Time
        axes[1,1].plot(x, list(self.inference_times), 'purple', alpha=0.7)
        axes[1,1].set_title(f'Inference Time (Avg: {np.mean(list(self.inference_times)):.1f} ms)')
        axes[1,1].set_xlabel('Sample')
        axes[1,1].set_ylabel('Inference Time (ms)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot gespeichert: {save_path}")
        
        plt.show()


def main():
    """
    Hauptfunktion für Live SOC Monitor (ohne Arduino)
    """
    # Initialisierung
    predictor = ArduinoSOCPredictor()
      # 1. Scaler initialisieren (exakt wie Train Script)
    if not predictor.initialize_scaler():
        logger.error("Scaler Initialisierung fehlgeschlagen!")
        return
    
    # 2. PyTorch Modell laden
    if not predictor.load_model():
        logger.error("Modell laden fehlgeschlagen!")
        return
      # 3. Ground Truth Daten laden
    if not predictor.load_ground_truth_data():
        logger.error("Ground Truth Daten laden fehlgeschlagen!")
        return
    
    try:        # 4. Live Monitor starten (mit PyTorch Modell)
        logger.info("=== STARTE LIVE SOC MONITOR MIT PYTORCH ===")
        predictor.run_live_prediction(max_samples=None, delay=0.005)  # Sehr schnell: 5ms delay
          # 5. Metriken berechnen
        metrics = predictor.calculate_metrics()
          # 6. Finale Ergebnisse plotten
        plot_path = Path("c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/BMS_Arduino_ready/PC_Version/live_monitor_results.png")
        predictor.plot_results(save_path=plot_path)
          # 7. Metriken speichern
        import json
        metrics_path = Path("c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/BMS_Arduino_ready/PC_Version/live_monitor_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metriken gespeichert: {metrics_path}")
        
    except Exception as e:
        logger.error(f"Fehler während Live Monitor: {e}")
    
    finally:        # 8. Aufräumen (kein Arduino disconnect nötig)
        logger.info("Live Monitor beendet")


if __name__ == "__main__":
    main()