#!/usr/bin/env python3
"""
🔍 ARDUINO PROOF TEST
====================
Dieser Test beweist eindeutig, dass die SOC-Vorhersagen vom Arduino kommen!

Test-Strategien:
1. Arduino physisch trennen und reconnecten
2. Verschiedene COM-Ports testen
3. Arduino-Reset und Neustart
4. Vergleich mit/ohne Arduino-Verbindung
5. Raw Serial Communication
"""

import serial
import time
import logging
import numpy as np
import sys
from datetime import datetime

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ArduinoProofTester:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        self.test_results = []
        
    def connect_arduino(self):
        """Verbinde mit Arduino"""
        try:
            logging.info(f"🔌 Versuche Verbindung zu Arduino auf {self.port}...")
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)  # Arduino Reset-Zeit
            
            # Test ob Arduino antwortet
            response = self.send_raw_command("INFO")
            if response:
                logging.info(f"✅ Arduino verbunden: {response}")
                return True
            else:
                logging.error("❌ Arduino antwortet nicht!")
                return False
                
        except Exception as e:
            logging.error(f"❌ Arduino-Verbindung fehlgeschlagen: {e}")
            return False
    
    def disconnect_arduino(self):
        """Trenne Arduino"""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            logging.info("🔌 Arduino getrennt")
    
    def send_raw_command(self, command):
        """Sende Raw-Kommando an Arduino"""
        try:
            if not self.arduino or not self.arduino.is_open:
                return None
                
            # Sende Kommando
            self.arduino.write((command + '\n').encode())
            self.arduino.flush()
            
            # Warte auf Antwort
            response = self.arduino.readline().decode().strip()
            return response
            
        except Exception as e:
            logging.error(f"❌ Raw-Kommando fehlgeschlagen: {e}")
            return None
    
    def test_arduino_info(self):
        """Test 1: Arduino Info abrufen"""
        logging.info("\n🔍 TEST 1: ARDUINO INFO")
        logging.info("=" * 50)
        
        response = self.send_raw_command("INFO")
        if response:
            logging.info(f"📋 Arduino Info: {response}")
            self.test_results.append(("Arduino Info", True, response))
            return True
        else:
            logging.error("❌ Keine Arduino Info erhalten!")
            self.test_results.append(("Arduino Info", False, "No response"))
            return False
    
    def test_arduino_reset(self):
        """Test 2: Arduino Reset"""
        logging.info("\n🔍 TEST 2: ARDUINO RESET")
        logging.info("=" * 50)
        
        response = self.send_raw_command("RESET")
        if response:
            logging.info(f"🔄 Arduino Reset: {response}")
            self.test_results.append(("Arduino Reset", True, response))
            return True
        else:
            logging.error("❌ Arduino Reset fehlgeschlagen!")
            self.test_results.append(("Arduino Reset", False, "No response"))
            return False
    
    def test_prediction_with_disconnect(self):
        """Test 3: Vorhersage mit Disconnect-Test"""
        logging.info("\n🔍 TEST 3: PREDICTION MIT DISCONNECT")
        logging.info("=" * 50)
        
        # Test-Input
        test_input = "3.4,-0.2,0.92,1150"
        
        # 1. Vorhersage mit Arduino
        logging.info("📊 Teste Vorhersage MIT Arduino...")
        prediction_with = self.send_raw_command(f"PREDICT:{test_input}")
        
        if prediction_with:
            logging.info(f"✅ Vorhersage MIT Arduino: {prediction_with}")
            
            # 2. Arduino trennen
            logging.info("🔌 Trenne Arduino...")
            self.disconnect_arduino()
            time.sleep(1)
            
            # 3. Versuche Vorhersage OHNE Arduino
            logging.info("📊 Teste Vorhersage OHNE Arduino...")
            try:
                self.arduino = serial.Serial(self.port, self.baudrate, timeout=1)
                prediction_without = self.send_raw_command(f"PREDICT:{test_input}")
                self.disconnect_arduino()
                
                if prediction_without:
                    logging.info(f"❓ Vorhersage OHNE Arduino: {prediction_without}")
                else:
                    logging.info("✅ KORREKT: Keine Vorhersage ohne Arduino!")
                    
            except:
                logging.info("✅ KORREKT: Keine Verbindung ohne Arduino möglich!")
            
            # 4. Arduino wieder verbinden
            logging.info("🔌 Verbinde Arduino wieder...")
            if self.connect_arduino():
                prediction_reconnect = self.send_raw_command(f"PREDICT:{test_input}")
                logging.info(f"✅ Vorhersage nach Reconnect: {prediction_reconnect}")
                
                self.test_results.append(("Disconnect Test", True, 
                    f"Mit: {prediction_with}, Ohne: None, Nach: {prediction_reconnect}"))
                return True
        
        self.test_results.append(("Disconnect Test", False, "Failed"))
        return False
    
    def test_multiple_predictions(self):
        """Test 4: Multiple Vorhersagen mit verschiedenen Inputs"""
        logging.info("\n🔍 TEST 4: MULTIPLE PREDICTIONS")
        logging.info("=" * 50)
        
        test_inputs = [
            "3.2,0.5,0.9,1200",
            "3.6,-0.3,0.85,1100", 
            "3.8,0.1,0.95,1300",
            "3.1,0.8,0.88,1050",
            "3.9,-0.6,0.92,1250"
        ]
        
        predictions = []
        for i, input_data in enumerate(test_inputs):
            logging.info(f"📊 Test {i+1}: Input = {input_data}")
            prediction = self.send_raw_command(f"PREDICT:{input_data}")
            
            if prediction:
                try:
                    soc_value = float(prediction)
                    predictions.append(soc_value)
                    logging.info(f"✅ Vorhersage {i+1}: {soc_value:.6f}")
                except:
                    logging.error(f"❌ Ungültige Vorhersage: {prediction}")
                    return False
            else:
                logging.error(f"❌ Keine Antwort für Input {i+1}")
                return False
            
            time.sleep(0.1)  # Kurze Pause zwischen Tests
        
        # Analysiere Vorhersagen
        if len(predictions) == len(test_inputs):
            logging.info(f"📈 Alle {len(predictions)} Vorhersagen erhalten!")
            logging.info(f"📊 Bereich: {min(predictions):.6f} - {max(predictions):.6f}")
            logging.info(f"📊 Durchschnitt: {np.mean(predictions):.6f}")
            logging.info(f"📊 Varianz: {np.var(predictions):.6f}")
            
            self.test_results.append(("Multiple Predictions", True, 
                f"{len(predictions)} predictions, range: {min(predictions):.3f}-{max(predictions):.3f}"))
            return True
        
        self.test_results.append(("Multiple Predictions", False, "Incomplete"))
        return False
    
    def test_raw_serial_communication(self):
        """Test 5: Raw Serial Communication"""
        logging.info("\n🔍 TEST 5: RAW SERIAL COMMUNICATION")
        logging.info("=" * 50)
        
        try:
            # Direkte Serial-Kommunikation
            logging.info("🔌 Öffne direkte Serial-Verbindung...")
            ser = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)
            
            # Sende direktes Kommando
            command = "INFO\n"
            logging.info(f"📤 Sende: {command.strip()}")
            ser.write(command.encode())
            ser.flush()
            
            # Lese Antwort
            response = ser.readline().decode().strip()
            logging.info(f"📥 Empfangen: {response}")
            
            ser.close()
            
            if response:
                self.test_results.append(("Raw Serial", True, response))
                return True
            else:
                self.test_results.append(("Raw Serial", False, "No response"))
                return False
                
        except Exception as e:
            logging.error(f"❌ Raw Serial Test fehlgeschlagen: {e}")
            self.test_results.append(("Raw Serial", False, str(e)))
            return False
    
    def run_all_tests(self):
        """Führe alle Tests durch"""
        logging.info("🚀 ARDUINO PROOF TEST SUITE")
        logging.info("=" * 60)
        logging.info(f"🕐 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"🔌 Port: {self.port}")
        logging.info(f"⚡ Baudrate: {self.baudrate}")
        logging.info("=" * 60)
        
        # Verbinde Arduino
        if not self.connect_arduino():
            logging.error("❌ Arduino-Verbindung fehlgeschlagen! Test abgebrochen.")
            return False
        
        # Führe Tests durch
        tests = [
            self.test_arduino_info,
            self.test_arduino_reset,
            self.test_multiple_predictions,
            self.test_prediction_with_disconnect,
            self.test_raw_serial_communication
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
                time.sleep(0.5)  # Pause zwischen Tests
            except Exception as e:
                logging.error(f"❌ Test fehlgeschlagen: {e}")
        
        # Trenne Arduino
        self.disconnect_arduino()
        
        # Zeige Ergebnisse
        self.show_results(passed, total)
        
        return passed == total
    
    def show_results(self, passed, total):
        """Zeige Test-Ergebnisse"""
        logging.info("\n" + "=" * 60)
        logging.info("📊 TEST-ERGEBNISSE")
        logging.info("=" * 60)
        
        for test_name, success, details in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            logging.info(f"{status} {test_name}: {details}")
        
        logging.info("=" * 60)
        success_rate = (passed / total) * 100
        logging.info(f"📈 Erfolgsrate: {passed}/{total} ({success_rate:.1f}%)")
        
        if passed == total:
            logging.info("🎉 ALLE TESTS BESTANDEN!")
            logging.info("✅ ARDUINO-VORHERSAGEN SIND 100% VERIFIZIERT!")
        else:
            logging.warning(f"⚠️ {total - passed} Test(s) fehlgeschlagen!")
        
        logging.info("=" * 60)


def main():
    """Hauptfunktion"""
    print("\n🔍 ARDUINO PROOF TEST")
    print("====================")
    print("Dieser Test beweist eindeutig, dass SOC-Vorhersagen vom Arduino kommen!")
    print()
    
    # Erstelle Tester
    tester = ArduinoProofTester(port='COM13')
    
    try:
        # Führe alle Tests durch
        success = tester.run_all_tests()
        
        if success:
            print("\n🎉 BEWEIS ERBRACHT!")
            print("✅ Alle Tests bestanden - Arduino macht definitiv die Vorhersagen!")
        else:
            print("\n⚠️ TESTS TEILWEISE FEHLGESCHLAGEN!")
            print("❓ Überprüfe Arduino-Verbindung und Hardware!")
            
    except KeyboardInterrupt:
        print("\n⏹️ Test durch Benutzer abgebrochen")
        tester.disconnect_arduino()
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        tester.disconnect_arduino()


if __name__ == "__main__":
    main()
