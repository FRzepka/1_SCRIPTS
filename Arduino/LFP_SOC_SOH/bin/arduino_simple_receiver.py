"""
Arduino Simple Receiver - Nur empfangen und anzeigen
Zeigt was der Arduino wirklich sendet ohne Ground Truth Verwirrung
"""

import serial
import time
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class SimpleArduinoReceiver:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        
        # Live data für Plot
        self.soc_values = deque(maxlen=50)  # Letzte 50 Werte
        self.timestamps = deque(maxlen=50)
        self.all_soc_values = []  # Alle Werte speichern
        self.start_time = None
        
    def connect(self):
        """Arduino Verbindung"""
        try:
            print(f"🔌 Verbinde mit Arduino auf {self.port}...")
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)
            print("✅ Verbunden!")
            return True
        except Exception as e:
            print(f"❌ Fehler: {e}")
            return False
    
    def receive_only(self):
        """Nur empfangen - nichts senden!"""
        self.start_time = time.time()
        sample_count = 0
        
        print("🎯 NUR EMPFANG MODUS - Arduino sendet selbstständig")
        print("=" * 50)
        print("📨 Warte auf Arduino Daten...")
        
        # Live plot setup
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            while True:
                if self.arduino.in_waiting > 0:
                    try:
                        line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                        
                        if line:
                            current_time = time.time() - self.start_time
                            
                            # Versuche SOC zu extrahieren
                            try:
                                # Verschiedene Formate probieren
                                if ',' in line:
                                    # CSV Format
                                    parts = line.split(',')
                                    soc = float(parts[-1])  # Letzter Wert als SOC
                                else:
                                    # Einzelner Wert
                                    soc = float(line)
                                
                                # Realistische SOC Werte (0-1)
                                if 0.0 <= soc <= 1.0:
                                    sample_count += 1
                                    
                                    # Daten speichern
                                    self.soc_values.append(soc)
                                    self.timestamps.append(current_time)
                                    self.all_soc_values.append(soc)
                                    
                                    print(f"#{sample_count:3d} | {current_time:6.1f}s | SOC: {soc:.6f} | Raw: '{line}'")
                                    
                                    # Live plot update alle 5 samples
                                    if sample_count % 5 == 0:
                                        self.update_plot(ax)
                                        
                                else:
                                    print(f"⚠️ Unrealistischer SOC: {soc} | Raw: '{line}'")
                                    
                            except ValueError:
                                print(f"📨 Nicht-numerisch: '{line}'")
                                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Gestoppt. {sample_count} SOC Werte empfangen.")
            self.show_statistics()
            
        finally:
            if self.arduino:
                self.arduino.close()
            plt.ioff()
            plt.show()
    
    def update_plot(self, ax):
        """Live Plot aktualisieren"""
        if len(self.soc_values) < 2:
            return
            
        ax.clear()
        ax.plot(list(self.timestamps), list(self.soc_values), 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Zeit [s]')
        ax.set_ylabel('SOC')
        ax.set_title(f'Arduino SOC Live Stream - {len(self.soc_values)} Samples')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.pause(0.01)
    
    def show_statistics(self):
        """Statistiken anzeigen"""
        if not self.all_soc_values:
            return
            
        soc_array = np.array(self.all_soc_values)
        
        print("\\n📊 ARDUINO SOC STATISTIKEN:")
        print(f"   Anzahl Samples: {len(soc_array)}")
        print(f"   SOC Bereich: {soc_array.min():.6f} - {soc_array.max():.6f}")
        print(f"   SOC Mittelwert: {soc_array.mean():.6f}")
        print(f"   SOC Std: {soc_array.std():.6f}")
        
        # Stabilität prüfen
        if len(soc_array) > 10:
            recent_std = np.std(soc_array[-10:])  # Letzte 10 Werte
            print(f"   Stabilität (letzte 10): {recent_std:.6f}")
            
            if recent_std < 0.01:
                print("   🟢 Arduino Ausgabe ist STABIL")
            elif recent_std < 0.05:
                print("   🟡 Arduino Ausgabe ist MÄSSIG stabil")
            else:
                print("   🔴 Arduino Ausgabe ist INSTABIL")

def main():
    """Hauptfunktion"""
    print("🎯 ARDUINO SIMPLE RECEIVER")
    print("=" * 40)
    print("📨 Empfängt nur - sendet nichts!")
    print("🔍 Zeigt was Arduino wirklich macht")
    print("=" * 40)
    
    receiver = SimpleArduinoReceiver()
    
    if receiver.connect():
        receiver.receive_only()
    else:
        print("❌ Verbindung fehlgeschlagen")

if __name__ == "__main__":
    main()
