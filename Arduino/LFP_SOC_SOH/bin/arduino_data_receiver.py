"""
Arduino Data Receiver - Nur Empfangen und Visualisieren
Einfaches Script das nur Arduino Daten empfängt und live plottet
KEIN Senden von Daten - nur reines Monitoring
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
from datetime import datetime

class ArduinoDataReceiver:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        self.running = False
        
        # Data buffers für Live-Plot
        self.max_samples = 100  # Letzte 100 Samples zeigen
        self.timestamps = deque(maxlen=self.max_samples)
        self.voltage_data = deque(maxlen=self.max_samples)
        self.current_data = deque(maxlen=self.max_samples)
        self.soc_predictions = deque(maxlen=self.max_samples)
        
        # Alle empfangenen Daten speichern
        self.all_data = []
        self.start_time = None
        
    def connect(self):
        """Verbindung zum Arduino herstellen"""
        try:
            print(f"🔌 Verbinde mit Arduino auf {self.port}...")
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino Reset Zeit
            print("✅ Arduino verbunden!")
            return True
        except Exception as e:
            print(f"❌ Verbindung fehlgeschlagen: {e}")
            return False
    
    def parse_arduino_data(self, line):
        """Arduino Daten parsen - verschiedene Formate erkennen"""
        line = line.strip()
        
        # Debug: Zeige rohe Daten
        print(f"📨 Empfangen: '{line}'")
        
        # Verschiedene mögliche Formate prüfen
        try:
            # Format 1: Nur SOC Wert
            if ',' not in line and line.replace('.', '').replace('-', '').isdigit():
                soc = float(line)
                return {'soc': soc, 'voltage': None, 'current': None}
            
            # Format 2: CSV Format (V,I,SOC)
            elif ',' in line:
                parts = line.split(',')
                if len(parts) >= 3:
                    voltage = float(parts[0])
                    current = float(parts[1]) 
                    soc = float(parts[2])
                    return {'voltage': voltage, 'current': current, 'soc': soc}
                elif len(parts) == 1:
                    soc = float(parts[0])
                    return {'soc': soc, 'voltage': None, 'current': None}
            
            # Format 3: Text mit Zahlen
            else:
                # Versuche Zahlen aus Text zu extrahieren
                import re
                numbers = re.findall(r'-?\d+\.?\d*', line)
                if numbers:
                    soc = float(numbers[-1])  # Letzte Zahl als SOC
                    return {'soc': soc, 'voltage': None, 'current': None}
                    
        except ValueError as e:
            print(f"⚠️ Parsing Error: {e} für '{line}'")
            return None
            
        return None
    
    def receive_data(self):
        """Kontinuierlich Daten vom Arduino empfangen"""
        if not self.arduino:
            return
            
        self.running = True
        self.start_time = time.time()
        
        print("🎯 Starte Datenempfang (Ctrl+C zum Stoppen)...")
        print("=" * 60)
        
        sample_count = 0
        
        while self.running:
            try:
                if self.arduino.in_waiting > 0:
                    # Daten lesen
                    raw_line = self.arduino.readline()
                    
                    try:
                        line = raw_line.decode('utf-8', errors='ignore').strip()
                    except:
                        continue
                        
                    if not line:
                        continue
                    
                    # Daten parsen
                    data = self.parse_arduino_data(line)
                    
                    if data and data['soc'] is not None:
                        sample_count += 1
                        current_time = time.time() - self.start_time
                        
                        # Daten zu Buffern hinzufügen
                        self.timestamps.append(current_time)
                        self.soc_predictions.append(data['soc'])
                        self.voltage_data.append(data.get('voltage', 0))
                        self.current_data.append(data.get('current', 0))
                        
                        # Alle Daten speichern
                        self.all_data.append({
                            'timestamp': current_time,
                            'datetime': datetime.now(),
                            'soc': data['soc'],
                            'voltage': data.get('voltage'),
                            'current': data.get('current'),
                            'raw_line': line
                        })
                        
                        # Status ausgeben
                        print(f"#{sample_count:3d} | Zeit: {current_time:6.1f}s | SOC: {data['soc']:.6f} | V: {data.get('voltage', 'N/A')} | I: {data.get('current', 'N/A')}")
                        
                        # Alle 10 Samples Live-Plot aktualisieren
                        if sample_count % 10 == 0:
                            self.update_live_plot()
                
                time.sleep(0.01)  # Kurze Pause
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ Empfangsfehler: {e}")
                continue
        
        print(f"\n📊 Datenempfang beendet. {sample_count} Samples empfangen.")
    
    def update_live_plot(self):
        """Live-Plot aktualisieren (non-blocking)"""
        if len(self.timestamps) < 2:
            return
            
        try:
            plt.clf()  # Clear figure
            
            if len(self.soc_predictions) > 0:
                # SOC Plot
                plt.subplot(2, 1, 1)
                plt.plot(list(self.timestamps), list(self.soc_predictions), 'b-', linewidth=2, label='Arduino SOC')
                plt.ylabel('SOC')
                plt.title(f'Arduino LSTM SOC Predictions (Live) - {len(self.soc_predictions)} Samples')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Voltage/Current Plot (wenn verfügbar)
                plt.subplot(2, 1, 2)
                if any(v is not None and v != 0 for v in self.voltage_data):
                    plt.plot(list(self.timestamps), list(self.voltage_data), 'r-', label='Voltage [V]')
                if any(i is not None and i != 0 for i in self.current_data):
                    plt.plot(list(self.timestamps), list(self.current_data), 'g-', label='Current [A]')
                
                plt.xlabel('Zeit [s]')
                plt.ylabel('V / A')
                plt.title('Input Daten (falls empfangen)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.pause(0.01)  # Non-blocking update
            
        except Exception as e:
            print(f"⚠️ Plot-Error: {e}")
    
    def save_data(self):
        """Empfangene Daten speichern"""
        if not self.all_data:
            print("❌ Keine Daten zum Speichern")
            return
            
        import pandas as pd
        
        # DataFrame erstellen
        df = pd.DataFrame(self.all_data)
        
        # Dateiname mit Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arduino_received_data_{timestamp}.csv"
        
        # Speichern
        df.to_csv(filename, index=False)
        print(f"💾 Daten gespeichert: {filename}")
        
        # Statistiken anzeigen
        print(f"\n📊 EMPFANGENE DATEN STATISTIKEN:")
        print(f"   Anzahl Samples: {len(df)}")
        print(f"   SOC Bereich: {df['soc'].min():.6f} - {df['soc'].max():.6f}")
        print(f"   SOC Mittelwert: {df['soc'].mean():.6f}")
        print(f"   SOC Standardabweichung: {df['soc'].std():.6f}")
        
        if df['voltage'].notna().any():
            print(f"   Voltage verfügbar: Ja")
            print(f"   Voltage Bereich: {df['voltage'].min():.3f} - {df['voltage'].max():.3f}")
        else:
            print(f"   Voltage verfügbar: Nein")
            
        if df['current'].notna().any():
            print(f"   Current verfügbar: Ja") 
            print(f"   Current Bereich: {df['current'].min():.3f} - {df['current'].max():.3f}")
        else:
            print(f"   Current verfügbar: Nein")
    
    def create_final_plot(self):
        """Finalen Plot mit allen Daten erstellen"""
        if not self.all_data:
            return
            
        import pandas as pd
        df = pd.DataFrame(self.all_data)
        
        plt.figure(figsize=(12, 8))
        
        # SOC über Zeit
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['soc'], 'b-', linewidth=1.5, alpha=0.8)
        plt.ylabel('SOC')
        plt.title(f'Arduino LSTM SOC Predictions - {len(df)} Samples')
        plt.grid(True, alpha=0.3)
        
        # Voltage und Current (wenn verfügbar)
        plt.subplot(2, 1, 2)
        if df['voltage'].notna().any():
            plt.plot(df['timestamp'], df['voltage'], 'r-', label='Voltage [V]', alpha=0.7)
        if df['current'].notna().any():
            plt.plot(df['timestamp'], df['current'], 'g-', label='Current [A]', alpha=0.7)
            
        plt.xlabel('Zeit [s]')
        plt.ylabel('V / A')
        plt.title('Input Parameter')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Plot speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"arduino_data_plot_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plot gespeichert: {plot_filename}")
        
        plt.show()
    
    def run(self):
        """Hauptfunktion - Empfang starten"""
        if not self.connect():
            return
            
        # Live-Plot Fenster vorbereiten
        plt.ion()  # Interactive mode
        plt.figure(figsize=(10, 6))
        
        try:
            # Daten empfangen
            self.receive_data()
            
        except KeyboardInterrupt:
            print("\n⏹️ Gestoppt durch Benutzer")
            
        finally:
            self.running = False
            if self.arduino:
                self.arduino.close()
                print("🔌 Arduino Verbindung geschlossen")
            
            # Daten auswerten und speichern
            self.save_data()
            self.create_final_plot()

def main():
    """Hauptfunktion"""
    print("🎯 ARDUINO DATA RECEIVER")
    print("=" * 40)
    print("📨 Reines Empfangs-Script für Arduino Daten")
    print("🔍 Zeigt was wirklich vom Arduino kommt")
    print("=" * 40)
    
    receiver = ArduinoDataReceiver(port='COM13', baudrate=115200)
    receiver.run()

if __name__ == "__main__":
    main()
