#!/usr/bin/env python3
"""
🚀 ARDUINO TEST DATA SENDER
============================
Sendet kontinuierlich Test-Daten an das Arduino für Live-Plotting
"""

import serial
import time
import random
import sys
import threading
from datetime import datetime

class ArduinoTestDataSender:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        self.running = False
        
    def connect(self):
        """Verbinde mit Arduino"""
        try:
            print(f"🔌 Verbinde mit Arduino auf {self.port}...")
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)  # Arduino Reset
            
            # Test Verbindung
            self.arduino.write(b"INFO\n")
            self.arduino.flush()
            response = self.arduino.readline().decode().strip()
            
            if "ARDUINO" in response:
                print(f"✅ Arduino verbunden: {response}")
                return True
            else:
                print(f"❌ Unerwartete Antwort: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Verbindung fehlgeschlagen: {e}")
            return False
    
    def disconnect(self):
        """Trenne Arduino"""
        self.running = False
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("🔌 Arduino getrennt")
    
    def send_test_data(self):
        """Sende kontinuierlich Test-Daten"""
        if not self.arduino or not self.arduino.is_open:
            print("❌ Arduino nicht verbunden!")
            return
        
        print("\n🚀 STARTE KONTINUIERLICHEN DATEN-STREAM")
        print("=" * 50)
        print("📊 Sende Test-Daten an Arduino...")
        print("🔌 JETZT KANNST DU DAS KABEL TRENNEN UM ZU TESTEN!")
        print("⏹️ Drücke Ctrl+C zum Stoppen")
        print("=" * 50)
        
        # Test-Daten-Sets
        test_data_sets = [
            # Realistische LFP-Batterie-Daten
            (3.2, -0.5, 0.90, 1200),   # Niedrige Spannung, Entladung
            (3.4, 0.2, 0.85, 1100),    # Mittlere Spannung, Ladung
            (3.6, -0.3, 0.92, 1250),   # Höhere Spannung, Entladung
            (3.3, 0.8, 0.88, 1050),    # Mittlere Spannung, starke Ladung
            (3.5, -0.1, 0.95, 1300),   # Gute Spannung, leichte Entladung
            (3.7, 0.4, 0.87, 1150),    # Hohe Spannung, Ladung
            (3.1, -0.8, 0.83, 1000),   # Niedrige Spannung, starke Entladung
            (3.8, 0.1, 0.94, 1280),    # Sehr hohe Spannung, leichte Ladung
            (3.45, -0.4, 0.91, 1180),  # Mittlere Spannung, Entladung
            (3.25, 0.6, 0.89, 1120),   # Niedrigere Spannung, Ladung
        ]
        
        self.running = True
        data_count = 0
        
        try:
            while self.running:
                # Wähle zufällige Test-Daten
                voltage, current, soh, capacity = random.choice(test_data_sets)
                
                # Füge kleine Variationen hinzu
                voltage += random.uniform(-0.05, 0.05)
                current += random.uniform(-0.1, 0.1)
                soh += random.uniform(-0.02, 0.02)
                capacity += random.uniform(-20, 20)
                
                # Begrenze Werte
                voltage = max(3.0, min(4.0, voltage))
                soh = max(0.8, min(1.0, soh))
                capacity = max(900, min(1400, capacity))
                
                # Erstelle Input-String
                input_data = f"{voltage:.3f},{current:.3f},{soh:.3f},{capacity:.1f}"
                
                try:
                    # Sende an Arduino
                    self.arduino.write((input_data + '\n').encode())
                    self.arduino.flush()
                    
                    # Lese Antwort
                    response = self.arduino.readline().decode().strip()
                    
                    if response:
                        try:
                            soc_prediction = float(response)
                            data_count += 1
                            
                            # Zeige Status
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            print(f"[{timestamp}] #{data_count:3d} | V:{voltage:.3f} I:{current:.3f} SOH:{soh:.3f} Q:{capacity:.0f} → SOC:{soc_prediction:.6f}")
                            
                        except ValueError:
                            print(f"❌ Ungültige Arduino-Antwort: {response}")
                    else:
                        print("❌ Keine Arduino-Antwort!")
                        
                except Exception as e:
                    print(f"❌ Kommunikationsfehler: {e}")
                    print("🔌 Arduino möglicherweise getrennt!")
                    break
                
                # Pause zwischen Messungen
                time.sleep(0.5)  # 2 Messungen pro Sekunde
                
        except KeyboardInterrupt:
            print("\n⏹️ Daten-Stream durch Benutzer gestoppt")
        except Exception as e:
            print(f"\n❌ Unerwarteter Fehler: {e}")
        
        print(f"\n📊 Insgesamt {data_count} Datenpunkte gesendet")

def main():
    print("🚀 ARDUINO TEST DATA SENDER")
    print("=" * 40)
    print("Sendet kontinuierlich Test-Daten an Arduino für Live-Plotting")
    print()
    
    sender = ArduinoTestDataSender()
    
    try:
        if sender.connect():
            print("\n✅ Bereit für Daten-Stream!")
            input("📋 Drücke ENTER um zu starten (stelle sicher dass das Live-Monitoring läuft)...")
            
            sender.send_test_data()
        else:
            print("❌ Arduino-Verbindung fehlgeschlagen!")
            
    except KeyboardInterrupt:
        print("\n⏹️ Programm durch Benutzer beendet")
    finally:
        sender.disconnect()

if __name__ == "__main__":
    main()
