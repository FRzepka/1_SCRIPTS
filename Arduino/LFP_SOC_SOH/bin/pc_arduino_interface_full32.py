"""
PC-Arduino Interface für VOLLSTÄNDIGE 32-Hidden-Unit LSTM SOC Prediction
Kommuniziert mit Arduino über Serial Port für Echtzeit-SOC-Vorhersagen
IDENTISCH zu live_test_soc.py - vollständige Modell-Architektur!
"""

import serial
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import logging

# Modell-Konstanten (identisch zu live_test_soc.py)
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32
MODEL_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\training_run_1_2_4_31\best_model.pth"

# Arduino-Konfiguration
ARDUINO_PORT = "COM13"  # Arduino UNO R4 Port
BAUD_RATE = 115200
TIMEOUT = 2.0

# Test-Daten Setup
TEST_DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\data_sender_C19.py"

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SOCModel(nn.Module):
    """Identische Modell-Definition wie live_test_soc.py"""
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

class ArduinoLSTMTester:
    def __init__(self, port=ARDUINO_PORT, baud=BAUD_RATE):
        self.port = port
        self.baud = baud
        self.arduino = None
        self.pytorch_model = None
        self.device = None
        self.pytorch_hidden = None
        
        # Statistiken
        self.stats = {
            'total_predictions': 0,
            'total_error': 0.0,
            'max_error': 0.0,
            'min_error': float('inf'),
            'processing_times': []
        }
        
    def connect_arduino(self):
        """Verbindung zu Arduino herstellen"""
        try:
            self.arduino = serial.Serial(self.port, self.baud, timeout=TIMEOUT)
            time.sleep(2)  # Arduino Reset warten
            
            # Arduino Info abrufen
            self.arduino.write("INFO\n".encode())
            time.sleep(0.5)
            
            response = ""
            while self.arduino.in_waiting:
                response += self.arduino.read(self.arduino.in_waiting).decode()
                time.sleep(0.1)
            
            logger.info(f"✅ Arduino verbunden auf {self.port}")
            logger.info("Arduino Info:")
            for line in response.strip().split('\n'):
                if line.strip():
                    logger.info(f"    {line.strip()}")
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ Arduino-Verbindung fehlgeschlagen: {e}")
            return False
            
    def load_pytorch_model(self):
        """PyTorch-Modell für Vergleich laden"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pytorch_model = SOCModel(input_size=4, dropout=0.05).to(self.device)
            
            if not Path(MODEL_PATH).exists():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            
            self.pytorch_model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
            self.pytorch_model.eval()
            
            # Hidden States initialisieren
            self.init_pytorch_hidden()
            
            logger.info(f"✅ PyTorch Model geladen von {MODEL_PATH}")
            logger.info(f"🎯 Device: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ PyTorch Model laden fehlgeschlagen: {e}")
            return False
            
    def init_pytorch_hidden(self):
        """PyTorch Hidden States initialisieren"""
        h = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE, device=self.device)
        c = torch.zeros_like(h)
        self.pytorch_hidden = (h, c)
        
    def reset_states(self):
        """Beide Modelle zurücksetzen"""
        # Arduino LSTM States zurücksetzen
        if self.arduino:
            self.arduino.write("RESET\n".encode())
            time.sleep(0.1)
            
        # PyTorch Hidden States zurücksetzen
        if self.pytorch_model:
            self.init_pytorch_hidden()
            
        logger.info("🔄 Beide Modelle zurückgesetzt")
        
    def predict_arduino(self, voltage, current, soh, q_c):
        """SOC-Vorhersage mit Arduino"""
        try:
            # Daten an Arduino senden
            data_str = f"{voltage:.6f},{current:.6f},{soh:.6f},{q_c:.6f}\n"
            
            start_time = time.time()
            self.arduino.write(data_str.encode())
            
            # Antwort lesen
            response = self.arduino.readline().decode().strip()
            processing_time = time.time() - start_time
            
            if response:
                soc_pred = float(response)
                self.stats['processing_times'].append(processing_time)
                return soc_pred, processing_time
            else:
                logger.warning("⚠️ Keine Antwort von Arduino")
                return None, processing_time
                
        except Exception as e:
            logger.error(f"❌ Arduino-Vorhersage fehlgeschlagen: {e}")
            return None, 0.0
            
    def predict_pytorch(self, voltage, current, soh, q_c):
        """SOC-Vorhersage mit PyTorch (Referenz)"""
        try:
            with torch.no_grad():
                # Input vorbereiten
                input_data = torch.tensor([[voltage, current, soh, q_c]], 
                                        dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Vorhersage
                soc_pred, self.pytorch_hidden = self.pytorch_model(input_data, self.pytorch_hidden)
                return soc_pred.item()
                
        except Exception as e:
            logger.error(f"❌ PyTorch-Vorhersage fehlgeschlagen: {e}")
            return None
            
    def test_single_prediction(self, voltage, current, soh, q_c):
        """Einzelne Vorhersage testen"""
        # Arduino-Vorhersage
        arduino_soc, arduino_time = self.predict_arduino(voltage, current, soh, q_c)
        
        # PyTorch-Vorhersage
        pytorch_soc = self.predict_pytorch(voltage, current, soh, q_c)
        
        if arduino_soc is not None and pytorch_soc is not None:
            error = abs(arduino_soc - pytorch_soc)
            
            # Statistiken aktualisieren
            self.stats['total_predictions'] += 1
            self.stats['total_error'] += error
            self.stats['max_error'] = max(self.stats['max_error'], error)
            self.stats['min_error'] = min(self.stats['min_error'], error)
            
            logger.info(f"📊 Input: V={voltage:.3f}, I={current:.3f}, SOH={soh:.3f}, Q_c={q_c:.3f}")
            logger.info(f"🎯 Arduino: {arduino_soc:.6f} | PyTorch: {pytorch_soc:.6f} | Error: {error:.6f} | Time: {arduino_time*1000:.1f}ms")
            
            return {
                'arduino_soc': arduino_soc,
                'pytorch_soc': pytorch_soc,
                'error': error,
                'processing_time': arduino_time,
                'input': [voltage, current, soh, q_c]
            }
        
        return None
        
    def run_test_sequence(self, test_data, max_tests=10):
        """Test-Sequenz ausführen"""
        logger.info(f"\n🎯 STARTE TEST-SEQUENZ (max {max_tests} Tests)")
        logger.info("=" * 60)
        
        self.reset_states()
        
        results = []
        for i, data_point in enumerate(test_data):
            if i >= max_tests:
                break
                
            result = self.test_single_prediction(
                data_point['voltage'],
                data_point['current'], 
                data_point['soh'],
                data_point['q_c']
            )
            
            if result:
                results.append(result)
                
            time.sleep(0.1)  # Kurze Pause zwischen Tests
            
        return results
        
    def print_statistics(self):
        """Test-Statistiken ausgeben"""
        if self.stats['total_predictions'] == 0:
            logger.info("❌ Keine erfolgreichen Vorhersagen")
            return
            
        avg_error = self.stats['total_error'] / self.stats['total_predictions']
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        logger.info(f"\n📊 TEST-STATISTIKEN:")
        logger.info("=" * 40)
        logger.info(f"🎯 Vorhersagen: {self.stats['total_predictions']}")
        logger.info(f"📏 Durchschnittlicher Fehler: {avg_error:.6f}")
        logger.info(f"📈 Max Fehler: {self.stats['max_error']:.6f}")
        logger.info(f"📉 Min Fehler: {self.stats['min_error']:.6f}")
        logger.info(f"⏱️ Durchschnittliche Zeit: {avg_time*1000:.1f}ms")
        logger.info(f"🎯 Genauigkeit: {(1-avg_error)*100:.2f}%")
        
    def close(self):
        """Verbindung schließen"""
        if self.arduino:
            self.arduino.close()
            logger.info("🔌 Arduino-Verbindung geschlossen")

def create_test_data():
    """Test-Daten erstellen"""
    test_data = [
        {'voltage': 3.2, 'current': 0.5, 'soh': 0.95, 'q_c': 1200},
        {'voltage': 3.4, 'current': -0.2, 'soh': 0.92, 'q_c': 1150},
        {'voltage': 3.6, 'current': 0.8, 'soh': 0.89, 'q_c': 1100},
        {'voltage': 3.8, 'current': -0.5, 'soh': 0.94, 'q_c': 1250},
        {'voltage': 3.3, 'current': 0.3, 'soh': 0.91, 'q_c': 1180},
        {'voltage': 3.5, 'current': -0.1, 'soh': 0.88, 'q_c': 1120},
        {'voltage': 3.7, 'current': 0.6, 'soh': 0.93, 'q_c': 1220},
        {'voltage': 3.1, 'current': 0.4, 'soh': 0.96, 'q_c': 1280},
        {'voltage': 3.9, 'current': -0.3, 'soh': 0.87, 'q_c': 1080},
        {'voltage': 3.45, 'current': 0.7, 'soh': 0.90, 'q_c': 1160}
    ]
    return test_data

def main():
    """Hauptfunktion"""
    logger.info("🎯 PC-ARDUINO LSTM SOC TESTER (VOLLSTÄNDIGE 32 HIDDEN UNITS)")
    logger.info("=" * 70)
    
    # Tester initialisieren
    tester = ArduinoLSTMTester()
    
    try:
        # Arduino verbinden
        if not tester.connect_arduino():
            return
            
        # PyTorch Modell laden
        if not tester.load_pytorch_model():
            return
            
        # Test-Daten erstellen
        test_data = create_test_data()
        
        # Tests ausführen
        results = tester.run_test_sequence(test_data, max_tests=10)
        
        # Statistiken anzeigen
        tester.print_statistics()
        
        logger.info(f"\n🎉 VOLLSTÄNDIGER TEST ABGESCHLOSSEN!")
        logger.info(f"✅ Arduino verwendet VOLLSTÄNDIGE 32 Hidden Units")
        logger.info(f"📊 {len(results)} erfolgreiche Vergleiche")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test durch Benutzer abgebrochen")
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {e}")
    finally:
        tester.close()

if __name__ == "__main__":
    main()
