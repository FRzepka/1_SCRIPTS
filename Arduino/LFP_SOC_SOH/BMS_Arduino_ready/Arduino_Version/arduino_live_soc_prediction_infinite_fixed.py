"""
BMS Arduino Live SOC Prediction System - INFINITE MODE
=====================================================

ARDUINO Version für UNENDLICHE Live SOC Vorhersage mit echter Arduino LSTM-Hardware
- Real-time SOC prediction mit Arduino LSTM Model
- Live plotting, MAE Berechnung und kontinuierliches Monitoring  
- Daten-Skalierung exakt wie PC Version
- Arduino Hardware Communication über Serial Interface
- LÄUFT UNENDLICH bis Ctrl+C (Keyboard Interrupt)

Features: Voltage[V], Current[A], SOH_ZHU, Q_c -> SOC_ZHU (Arduino LSTM)
Hardware: Arduino mit arduino_lstm_soc_full32.ino
Validierung: MGFarm C19 Cell Data (zyklisch wiederholt)

ARDUINO HARDWARE LSTM - INFINITE MODE!
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
import json
from datetime import datetime
import warnings

# Suppress sklearn warnings about feature names and matplotlib font warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='Glyph.*missing from font')

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArduinoLSTMSOCPredictor:
    """
    Arduino LSTM SOC Predictor - INFINITE Hardware Implementation
    ===========================================================
    
    Verwendet echte Arduino LSTM Hardware für SOC Vorhersagen
    - Direkte serielle Kommunikation mit Arduino
    - Real LSTM inference auf Mikrocontroller
    - UNENDLICH MODUS - läuft bis Ctrl+C
    """
    
    def __init__(self, 
                 port='COM13', 
                 baudrate=115200,
                 data_path=r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"):
        
        self.port = port
        self.baudrate = baudrate
        self.data_path = data_path
        self.serial_conn = None
        self.scaler = None
        
        # Data storage
        self.ground_truth_data = None
        self.current_index = 0
        
        # Real-time metrics (gleiche maxlen wie PC Version)
        self.predictions = deque(maxlen=1000)
        self.ground_truths = deque(maxlen=1000)
        self.voltages = deque(maxlen=1000)
        self.currents = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        self.errors = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        
        # Arduino communication metrics
        self.communication_times = deque(maxlen=1000)
        self.arduino_responses = deque(maxlen=1000)
        
        # Plotting
        self.figure = None
        self.axes = None
        self.plot_update_counter = 0
        self.plot_update_interval = 5  # Update every 5 samples (like PC version)
        
        # Performance tracking
        self.total_samples = 0
        self.start_time = None
        self.sample_rate = 0.0
        
        # Arduino connection status
        self.arduino_connected = False
        self.arduino_info = ""
        
        logger.info("Arduino LSTM SOC Predictor (INFINITE MODE) initialisiert")
        logger.info(f"Port: {port}, Baudrate: {baudrate}")
        logger.info(f"Data: {Path(data_path).name}")

    def connect_arduino(self):
        """Verbindet mit Arduino LSTM Hardware"""
        try:
            logger.info(f"Verbinde mit Arduino auf {self.port}...")
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(3)  # Arduino Reset Zeit
            
            # Leere Buffer von startup messages
            while self.serial_conn.in_waiting > 0:
                startup_msg = self.serial_conn.readline().decode().strip()
                logger.debug(f"Arduino Startup: {startup_msg}")
            
            # Test Verbindung mit Beispieldaten
            logger.info("Teste Arduino mit Beispieldaten...")
            test_data = "0.5,0.0,1.0,0.5"  # Test voltage, current, soh, q_c
            self.send_arduino_command(test_data)
            test_response = self.read_arduino_response()
            
            # Prüfe ob numerische SOC Antwort kommt
            try:
                soc_value = float(test_response)
                if 0.0 <= soc_value <= 1.0:  # Valider SOC Bereich
                    self.arduino_connected = True
                    logger.info(f"Arduino LSTM verbunden!")
                    logger.info(f"Test SOC Prediction: {soc_value:.6f}")
                    
                    # Reset LSTM States für sauberen Start
                    self.reset_arduino_lstm()
                    return True
                else:
                    logger.error(f"Invalide SOC Antwort: {soc_value}")
                    return False
            except ValueError:
                logger.error(f"Nicht-numerische Antwort: {test_response}")
                return False
                
        except Exception as e:
            logger.error(f"Arduino Verbindung fehlgeschlagen: {e}")
            return False
    
    def send_arduino_command(self, command):
        """Sendet Command an Arduino"""
        if self.serial_conn:
            self.serial_conn.write(f"{command}\n".encode())
            self.serial_conn.flush()
    
    def read_arduino_response(self, timeout=2.0):
        """Liest Antwort von Arduino"""
        if not self.serial_conn:
            return None
        
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.serial_conn.in_waiting > 0:
                response = self.serial_conn.readline().decode().strip()
                return response
            time.sleep(0.001)  # 1ms polling
        
        logger.warning("Arduino Timeout - keine Antwort erhalten")
        return None
    
    def reset_arduino_lstm(self):
        """Reset LSTM States auf Arduino"""
        logger.info("Reset Arduino LSTM States...")
        self.send_arduino_command("RESET")
        response = self.read_arduino_response()
        if response and "RESET" in response:
            logger.info("Arduino LSTM States zurückgesetzt")
        else:
            logger.warning("Arduino Reset nicht bestätigt")
    
    def predict_with_arduino(self, voltage, current, soh, q_c):
        """SOC Vorhersage mit Arduino LSTM Hardware"""
        start_time = time.time()
        
        try:
            # Features skalieren
            scaled_features = self.scale_features(voltage, current, soh, q_c)
            
            # Command für Arduino erstellen
            command = f"{scaled_features[0]:.6f},{scaled_features[1]:.6f},{scaled_features[2]:.6f},{scaled_features[3]:.6f}"
            
            # An Arduino senden
            self.send_arduino_command(command)
            
            # Antwort lesen
            response = self.read_arduino_response()
            communication_time = time.time() - start_time
            
            if response is not None:
                try:
                    predicted_soc = float(response)
                    
                    # Timing speichern
                    self.communication_times.append(communication_time)
                    self.arduino_responses.append(predicted_soc)
                    
                    return predicted_soc, communication_time
                    
                except ValueError:
                    logger.error(f"Invalide Arduino Antwort: {response}")
                    return None, communication_time
            else:
                logger.error("Keine Antwort von Arduino")
                return None, communication_time
                
        except Exception as e:
            logger.error(f"Arduino Prediction Fehler: {e}")
            return None, time.time() - start_time
    
    def load_scaler(self):
        """
        Erstelle den EXAKT gleichen StandardScaler wie im 1.2.4 Train Script
        
        Das ist der KRITISCHE Teil für identische Skalierung zwischen Training und Inference!
        Hier werden die EXAKT gleichen Skalierungsparameter wie beim Training verwendet.
        """
        try:
            # Versuche gespeicherten Scaler zu laden
            scaler_path = Path(self.data_path).parent / "scaler.pkl"
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Scaler geladen von: {scaler_path}")
                return True
            else:
                # Fallback: Erstelle Scaler basierend auf Trainingsdaten
                logger.warning("Gespeicherter Scaler nicht gefunden, erstelle neuen basierend auf Daten...")
                
                # Lade Daten für Scaler-Erstellung
                df = pd.read_parquet(self.data_path)
                features = df[["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]].copy()
                
                # StandardScaler mit EXAKT gleichen Parametern wie Training
                self.scaler = StandardScaler()
                self.scaler.fit(features)
                
                # Speichere für zukünftige Verwendung
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                
                logger.info(f"Neuer Scaler erstellt und gespeichert: {scaler_path}")
                return True
                
        except Exception as e:
            logger.error(f"Scaler laden fehlgeschlagen: {e}")
            return False
    
    def load_ground_truth_data(self):
        """Lädt Ground Truth Daten für Vergleich"""
        try:
            logger.info(f"Lade Ground Truth Daten: {self.data_path}")
            self.ground_truth_data = pd.read_parquet(self.data_path)
            
            # Prüfe erforderliche Spalten
            required_cols = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c", "SOC_ZHU"]
            missing_cols = [col for col in required_cols if col not in self.ground_truth_data.columns]
            
            if missing_cols:
                logger.error(f"Fehlende Spalten in Daten: {missing_cols}")
                return False
                
            logger.info(f"Ground Truth Daten geladen: {len(self.ground_truth_data)} Samples")
            return True
            
        except Exception as e:
            logger.error(f"Daten laden fehlgeschlagen: {e}")
            return False
    
    def scale_features(self, voltage, current, soh, q_c):
        """Skaliert Features mit dem geladenen Scaler"""
        try:
            # Feature array erstellen mit explicit feature names für sklearn
            features = pd.DataFrame([[voltage, current, soh, q_c]], 
                                  columns=["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"])
            
            # Skalierung anwenden (kein Feature Names Warning)
            scaled_features = self.scaler.transform(features)
            
            return scaled_features[0]  # Rückgabe als 1D array
            
        except Exception as e:
            logger.warning(f"Skalierung fehlgeschlagen: {e}")
            return voltage, current, soh, q_c
    
    def setup_plots(self):
        """Setup für Live Plotting"""
        plt.ion()
        self.figure, self.axes = plt.subplots(2, 2, figsize=(20, 12))
        self.figure.suptitle('Arduino LSTM SOC Live Prediction Monitor (INFINITE MODE)', 
                           fontsize=20, fontweight='bold')
        
        # Subplot Titel ohne Emojis
        self.axes[0, 0].set_title('SOC Prediction vs Ground Truth', fontsize=18)
        self.axes[0, 1].set_title('Prediction Error (MAE)', fontsize=18)
        self.axes[1, 0].set_title('Voltage & Current', fontsize=18)
        self.axes[1, 1].set_title('Arduino Communication Time', fontsize=18)
        
        # Grid für alle Subplots
        for ax in self.axes.flat:
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=14)
        
        # Labels setzen
        self.axes[0, 0].set_ylabel('SOC', fontsize=16)
        self.axes[0, 1].set_ylabel('Absolute Error', fontsize=16)
        self.axes[1, 0].set_ylabel('Voltage [V] / Current [A]', fontsize=16)
        self.axes[1, 1].set_ylabel('Communication Time [ms]', fontsize=16)
        
        for ax in self.axes[1, :]:  # Bottom row
            ax.set_xlabel('Sample #', fontsize=16)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update_plots(self):
        """Live Plot Update (nur alle 5 Samples wie PC Version)"""
        if len(self.predictions) < 2:  # Mindestens 2 Punkte für Plot
            return
            
        try:
            # Clear alle Subplots
            for ax in self.axes.flat:
                ax.clear()
                ax.grid(True, alpha=0.3)
            
            # Daten für Plotting vorbereiten
            samples = list(range(len(self.predictions)))
            preds = list(self.predictions)
            truths = list(self.ground_truths)
            errors = list(self.errors)
            voltages = list(self.voltages)
            currents = list(self.currents)
            comm_times = [t * 1000 for t in self.communication_times]  # ms
            
            # 1. SOC Prediction vs Ground Truth
            self.axes[0, 0].plot(samples, truths, 'b-', label='Ground Truth', linewidth=2)
            self.axes[0, 0].plot(samples, preds, 'r--', label='Arduino LSTM Prediction', linewidth=2)
            self.axes[0, 0].set_title('SOC Prediction vs Ground Truth', fontsize=16)
            self.axes[0, 0].set_ylabel('SOC', fontsize=14)
            self.axes[0, 0].legend(fontsize=12)
            self.axes[0, 0].set_ylim(0, 1)
            
            # 2. Prediction Error
            self.axes[0, 1].plot(samples, errors, 'r-', linewidth=2)
            mean_error = np.mean(errors) if errors else 0
            self.axes[0, 1].axhline(y=mean_error, color='black', linestyle='--', 
                                   label=f'Mean MAE: {mean_error:.4f}')
            self.axes[0, 1].set_title('Prediction Error (MAE)', fontsize=16)
            self.axes[0, 1].set_ylabel('Absolute Error', fontsize=14)
            self.axes[0, 1].legend(fontsize=12)
            
            # 3. Voltage & Current
            ax3_twin = self.axes[1, 0].twinx()
            line1 = self.axes[1, 0].plot(samples, voltages, 'g-', label='Voltage [V]', linewidth=2)
            line2 = ax3_twin.plot(samples, currents, 'orange', label='Current [A]', linewidth=2)
            
            self.axes[1, 0].set_title('Voltage & Current', fontsize=16)
            self.axes[1, 0].set_ylabel('Voltage [V]', fontsize=14, color='g')
            ax3_twin.set_ylabel('Current [A]', fontsize=14, color='orange')
            self.axes[1, 0].set_xlabel('Sample #', fontsize=14)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            self.axes[1, 0].legend(lines, labels, loc='upper left', fontsize=12)
            
            # 4. Arduino Communication Time
            if comm_times:
                self.axes[1, 1].plot(samples[-len(comm_times):], comm_times, 'purple', linewidth=2)
                mean_comm = np.mean(comm_times)
                self.axes[1, 1].axhline(y=mean_comm, color='black', linestyle='--',
                                       label=f'Mean: {mean_comm:.1f} ms')
                self.axes[1, 1].set_title('Arduino Communication Time', fontsize=16)
                self.axes[1, 1].set_ylabel('Communication Time [ms]', fontsize=14)
                self.axes[1, 1].set_xlabel('Sample #', fontsize=14)
                self.axes[1, 1].legend(fontsize=12)
            
            # Stats in Suptitle ohne Emojis
            if len(self.predictions) > 0:
                current_mae = np.mean(errors) if errors else 0
                sample_rate = len(self.predictions) / (time.time() - self.start_time) if self.start_time else 0
                
                stats_title = (f'Arduino LSTM SOC Monitor (INFINITE) | '
                             f'Samples: {len(self.predictions)} | '
                             f'MAE: {current_mae:.4f} | '
                             f'Rate: {sample_rate:.1f} Hz | '
                             f'Arduino: {"Connected" if self.arduino_connected else "Disconnected"}')
                
                self.figure.suptitle(stats_title, fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)  # Sehr kurze Pause für Update
            
        except Exception as e:
            logger.warning(f"Plot Update Fehler: {e}")
    
    def run_infinite_prediction(self, delay_ms=20):
        """
        Hauptschleife für UNENDLICHE Live SOC Prediction mit Arduino Hardware
        ====================================================================
        
        Parameter für kontinuierlichen Betrieb:
        - UNENDLICH MODE - läuft bis Ctrl+C (Keyboard Interrupt)
        - 20ms delay (realistischer für Arduino Kommunikation)
        - Plot updates alle 5 samples
        - Daten werden zyklisch wiederholt wenn Ende erreicht wird
        - Performance monitoring mit echten Timing-Messungen
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTE ARDUINO LSTM SOC INFINITE PREDICTION")
            logger.info("=" * 80)
            logger.info("Drücke Ctrl+C zum Beenden")
            
            # Setup
            self.start_time = time.time()
            self.setup_plots()
            
            logger.info(f"Infinite Mode gestartet mit {delay_ms}ms delay")
            logger.info("Live Plotting aktiv - Plot Updates alle 5 Samples")
            
            sample_count = 0
            
            # UNENDLICHE HAUPTSCHLEIFE
            while True:
                try:
                    # Zyklisch durch Daten iterieren
                    if self.current_index >= len(self.ground_truth_data):
                        self.current_index = 0
                        logger.info(f"Daten-Zyklus abgeschlossen, starte von vorne (Sample #{sample_count})")
                    
                    # Current sample
                    current_sample = self.ground_truth_data.iloc[self.current_index]
                    
                    voltage = current_sample["Voltage[V]"]
                    current_val = current_sample["Current[A]"]
                    soh = current_sample["SOH_ZHU"]
                    q_c = current_sample["Q_c"]
                    ground_truth_soc = current_sample["SOC_ZHU"]
                    
                    # Arduino Prediction
                    prediction_start = time.time()
                    predicted_soc, comm_time = self.predict_with_arduino(voltage, current_val, soh, q_c)
                    total_inference_time = time.time() - prediction_start
                    
                    if predicted_soc is not None:
                        # Metrics berechnen
                        error = abs(predicted_soc - ground_truth_soc)
                        
                        # Daten speichern
                        self.predictions.append(predicted_soc)
                        self.ground_truths.append(ground_truth_soc)
                        self.voltages.append(voltage)
                        self.currents.append(current_val)
                        self.errors.append(error)
                        self.inference_times.append(total_inference_time)
                        self.timestamps.append(time.time())
                        
                        # Progress logging alle 100 samples
                        sample_count += 1
                        if sample_count % 100 == 0:
                            current_mae = np.mean(list(self.errors)[-100:])  # MAE der letzten 100
                            avg_comm_time = np.mean(list(self.communication_times)[-100:]) * 1000
                            sample_rate = sample_count / (time.time() - self.start_time)
                            
                            logger.info(f"Sample #{sample_count}: SOC_pred={predicted_soc:.4f}, "
                                      f"SOC_true={ground_truth_soc:.4f}, Error={error:.4f}, "
                                      f"MAE_100={current_mae:.4f}, Comm={avg_comm_time:.1f}ms, "
                                      f"Rate={sample_rate:.1f}Hz")
                        
                        # Plot Update alle 5 Samples (wie PC Version)
                        self.plot_update_counter += 1
                        if self.plot_update_counter >= self.plot_update_interval:
                            self.update_plots()
                            self.plot_update_counter = 0
                    
                    else:
                        logger.warning(f"Sample #{sample_count}: Arduino Prediction fehlgeschlagen")
                    
                    # Nächster Index
                    self.current_index += 1
                    
                    # Delay für realistische Timing
                    time.sleep(delay_ms / 1000.0)
                    
                except KeyboardInterrupt:
                    logger.info("\nKeyboard Interrupt empfangen - Beende INFINITE MODE...")
                    break
                    
                except Exception as e:
                    logger.error(f"Fehler in Hauptschleife (Sample #{sample_count}): {e}")
                    time.sleep(0.1)  # Kurze Pause bei Fehlern
                    continue
            
            # Final Statistics
            self.print_final_statistics()
            
        except KeyboardInterrupt:
            logger.info("\nINFINITE MODE durch Benutzer beendet")
            self.print_final_statistics()
        except Exception as e:
            logger.error(f"Kritischer Fehler in INFINITE MODE: {e}")
            self.print_final_statistics()
    
    def print_final_statistics(self):
        """Abschluss-Statistiken und Speicherung"""
        try:
            total_time = time.time() - self.start_time if self.start_time else 0
            total_samples = len(self.predictions)
            
            if total_samples > 0:
                # Berechne Statistiken
                final_mae = np.mean(list(self.errors))
                std_error = np.std(list(self.errors))
                avg_comm_time = np.mean(list(self.communication_times)) * 1000
                avg_sample_rate = total_samples / total_time if total_time > 0 else 0
                
                logger.info("=" * 80)
                logger.info("ARDUINO LSTM SOC INFINITE PREDICTION - FINAL STATISTICS")
                logger.info("=" * 80)
                logger.info(f"Total Runtime: {total_time:.2f} seconds")
                logger.info(f"Total Samples: {total_samples}")
                logger.info(f"Average Sample Rate: {avg_sample_rate:.2f} Hz")
                logger.info(f"Final MAE: {final_mae:.6f}")
                logger.info(f"Error Std Dev: {std_error:.6f}")
                logger.info(f"Average Arduino Communication Time: {avg_comm_time:.2f} ms")
                logger.info(f"Arduino Connection: {'Stable' if self.arduino_connected else 'Failed'}")
                
                # Speichere Ergebnisse
                results = {
                    'total_time': total_time,
                    'total_samples': total_samples,
                    'sample_rate': avg_sample_rate,
                    'final_mae': final_mae,
                    'error_std': std_error,
                    'avg_comm_time_ms': avg_comm_time,
                    'predictions': list(self.predictions),
                    'ground_truths': list(self.ground_truths),
                    'errors': list(self.errors),
                    'communication_times': list(self.communication_times)
                }
                
                # Speichere in JSON
                result_file = f"arduino_infinite_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
                logger.info(f"Ergebnisse gespeichert in: {result_file}")
                logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der finalen Statistiken: {e}")

def main():
    """Hauptfunktion"""
    try:
        logger.info("=" * 80)
        logger.info("BMS Arduino Live SOC Prediction System - INFINITE MODE")
        logger.info("=" * 80)
        
        # Arduino LSTM SOC Predictor erstellen
        predictor = ArduinoLSTMSOCPredictor(
            port='COM13',  # Anpassen an deinen Arduino Port
            baudrate=115200
        )
        
        # Setup
        logger.info("Lade Scaler...")
        if not predictor.load_scaler():
            logger.error("Scaler laden fehlgeschlagen!")
            return
        
        logger.info("Lade Ground Truth Daten...")
        if not predictor.load_ground_truth_data():
            logger.error("Ground Truth Daten laden fehlgeschlagen!")
            return
        
        logger.info("Verbinde mit Arduino...")
        if not predictor.connect_arduino():
            logger.error("Arduino Verbindung fehlgeschlagen!")
            return
        
        # INFINITE PREDICTION STARTEN
        logger.info("Starte INFINITE PREDICTION MODE...")
        predictor.run_infinite_prediction(delay_ms=20)
        
    except KeyboardInterrupt:
        logger.info("Programm durch Benutzer beendet")
    except Exception as e:
        logger.error(f"Hauptprogramm Fehler: {e}")
    finally:
        logger.info("Programm beendet")

if __name__ == "__main__":
    main()
