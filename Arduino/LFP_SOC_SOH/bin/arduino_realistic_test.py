"""
Arduino Realistic Test - Ordentlicher Test mit festen Testpunkten
Sendet definierte Testdaten und vergleicht mit erwarteten Ergebnissen
"""

import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ArduinoRealisticTest:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        
        # Test Ergebnisse
        self.test_results = []
        
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
    
    def create_test_points(self):
        """Erstelle realistische Testpunkte"""
        print("📊 Erstelle Testpunkte...")
        
        # Realistische Bereiche für LFP Zellen
        test_points = []
        
        # Test 1: Verschiedene SOC Level bei normaler Spannung
        for soc_target in [0.2, 0.4, 0.6, 0.8]:
            voltage = 3.0 + soc_target * 0.6  # 3.0V (0%) bis 3.6V (100%)
            current = 0.0  # Ruhezustand
            soh = 0.95  # 95% SOH
            q_c = 2800  # mAh Kapazität
            
            test_points.append({
                'voltage': voltage,
                'current': current, 
                'soh': soh,
                'q_c': q_c,
                'expected_soc_range': (soc_target - 0.1, soc_target + 0.1),
                'description': f'SOC ~{soc_target*100:.0f}% Test'
            })
        
        # Test 2: Laden (positive Ströme)
        for current in [0.5, 1.0, 1.5]:
            test_points.append({
                'voltage': 3.4,
                'current': current,
                'soh': 0.9,
                'q_c': 2800,
                'expected_soc_range': (0.4, 0.8),
                'description': f'Laden {current}A'
            })
        
        # Test 3: Entladen (negative Ströme)
        for current in [-0.5, -1.0, -1.5]:
            test_points.append({
                'voltage': 3.2,
                'current': current,
                'soh': 0.85,
                'q_c': 2800,
                'expected_soc_range': (0.3, 0.7),
                'description': f'Entladen {abs(current)}A'
            })
        
        print(f"✅ {len(test_points)} Testpunkte erstellt")
        return test_points
    
    def send_and_receive(self, voltage, current, soh, q_c, max_attempts=3):
        """Sende Daten und empfange Antwort"""
        for attempt in range(max_attempts):
            try:
                # Buffer leeren
                self.arduino.flushInput()
                
                # Daten senden
                data_str = f"{voltage:.6f},{current:.6f},{soh:.6f},{q_c:.6f}\\n"
                self.arduino.write(data_str.encode())
                
                # Antwort warten
                time.sleep(0.5)
                
                if self.arduino.in_waiting > 0:
                    response = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    
                    try:
                        soc_pred = float(response)
                        if 0.0 <= soc_pred <= 1.0:
                            return soc_pred
                    except ValueError:
                        pass
                        
            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f"⚠️ Kommunikationsfehler: {e}")
                    
        return None
    
    def run_test(self):
        """Haupttest durchführen"""
        test_points = self.create_test_points()
        
        print("\\n🚀 Starte Arduino Realistic Test")
        print("=" * 60)
        print(f"{'#':<3} {'Beschreibung':<15} {'V':<6} {'I':<6} {'SOH':<5} {'Q_c':<6} {'SOC':<8} {'Erwart.':<12} {'Status':<8}")
        print("-" * 60)
        
        for i, test_point in enumerate(test_points):
            # Test durchführen
            soc_pred = self.send_and_receive(
                test_point['voltage'],
                test_point['current'],
                test_point['soh'],
                test_point['q_c']
            )
            
            if soc_pred is not None:
                # Prüfen ob im erwarteten Bereich
                exp_min, exp_max = test_point['expected_soc_range']
                is_in_range = exp_min <= soc_pred <= exp_max
                status = "✅ OK" if is_in_range else "❌ FAIL"
                
                # Ergebnis speichern
                result = {
                    'test_id': i + 1,
                    'description': test_point['description'],
                    'voltage': test_point['voltage'],
                    'current': test_point['current'],
                    'soh': test_point['soh'],
                    'q_c': test_point['q_c'],
                    'soc_predicted': soc_pred,
                    'soc_expected_min': exp_min,
                    'soc_expected_max': exp_max,
                    'in_range': is_in_range
                }
                self.test_results.append(result)
                
                # Ausgabe
                print(f"{i+1:<3} {test_point['description']:<15} "
                      f"{test_point['voltage']:<6.2f} {test_point['current']:<6.2f} "
                      f"{test_point['soh']:<5.2f} {test_point['q_c']:<6.0f} "
                      f"{soc_pred:<8.4f} {exp_min:.2f}-{exp_max:.2f}{'':>4} {status:<8}")
                      
            else:
                print(f"{i+1:<3} {test_point['description']:<15} "
                      f"{test_point['voltage']:<6.2f} {test_point['current']:<6.2f} "
                      f"{test_point['soh']:<5.2f} {test_point['q_c']:<6.0f} "
                      f"{'TIMEOUT':<8} {'':>12} {'❌ COMM':<8}")
            
            time.sleep(1)  # Pause zwischen Tests
        
        self.analyze_results()
    
    def analyze_results(self):
        """Ergebnisse analysieren"""
        if not self.test_results:
            print("❌ Keine Testergebnisse zum Analysieren")
            return
            
        print("\\n📊 TEST ANALYSE")
        print("=" * 50)
        
        df = pd.DataFrame(self.test_results)
        
        # Erfolgsrate
        success_rate = (df['in_range'].sum() / len(df)) * 100
        print(f"🎯 Erfolgsrate: {success_rate:.1f}% ({df['in_range'].sum()}/{len(df)})")
        
        # SOC Statistiken
        print(f"📈 SOC Vorhersagen:")
        print(f"   Bereich: {df['soc_predicted'].min():.4f} - {df['soc_predicted'].max():.4f}")
        print(f"   Mittelwert: {df['soc_predicted'].mean():.4f}")
        print(f"   Standardabweichung: {df['soc_predicted'].std():.4f}")
        
        # Fehlgeschlagene Tests
        failed_tests = df[~df['in_range']]
        if len(failed_tests) > 0:
            print(f"\\n❌ Fehlgeschlagene Tests:")
            for _, test in failed_tests.iterrows():
                error_size = min(abs(test['soc_predicted'] - test['soc_expected_min']),
                               abs(test['soc_predicted'] - test['soc_expected_max']))
                print(f"   {test['description']}: {test['soc_predicted']:.4f} "
                      f"(erwartet: {test['soc_expected_min']:.2f}-{test['soc_expected_max']:.2f}, "
                      f"Fehler: {error_size:.4f})")
        
        # Plot erstellen
        self.create_plots(df)
    
    def create_plots(self, df):
        """Visualisierung erstellen"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: SOC Vorhersagen nach Test
        ax1.bar(range(len(df)), df['soc_predicted'], 
                color=['green' if x else 'red' for x in df['in_range']], alpha=0.7)
        ax1.set_xlabel('Test #')
        ax1.set_ylabel('SOC Vorhersage')
        ax1.set_title('SOC Vorhersagen pro Test')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spannung vs SOC
        scatter = ax2.scatter(df['voltage'], df['soc_predicted'], 
                            c=df['current'], cmap='RdYlBu', s=60, alpha=0.7)
        ax2.set_xlabel('Spannung [V]')
        ax2.set_ylabel('SOC Vorhersage')
        ax2.set_title('Spannung vs SOC (Farbe = Strom)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Strom [A]')
        
        # Plot 3: Strom vs SOC
        ax3.scatter(df['current'], df['soc_predicted'], 
                   c=['green' if x else 'red' for x in df['in_range']], s=60, alpha=0.7)
        ax3.set_xlabel('Strom [A]')
        ax3.set_ylabel('SOC Vorhersage')
        ax3.set_title('Strom vs SOC (Grün=OK, Rot=Fehler)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: SOH vs SOC
        ax4.scatter(df['soh'], df['soc_predicted'], s=60, alpha=0.7)
        ax4.set_xlabel('SOH')
        ax4.set_ylabel('SOC Vorhersage')
        ax4.set_title('SOH vs SOC')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Speichern
        timestamp = int(time.time())
        filename = f"arduino_realistic_test_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plot gespeichert: {filename}")
        
        plt.show()
    
    def run(self):
        """Hauptfunktion"""
        if not self.connect():
            return
            
        try:
            self.run_test()
        except KeyboardInterrupt:
            print("\\n⏹️ Test abgebrochen")
        finally:
            if self.arduino:
                self.arduino.close()
                print("🔌 Arduino Verbindung geschlossen")

def main():
    """Hauptfunktion"""
    print("🎯 ARDUINO REALISTIC TEST")
    print("=" * 40)
    print("🧪 Testet Arduino mit definierten Testpunkten")
    print("📊 Prüft ob Vorhersagen im erwarteten Bereich")
    print("=" * 40)
    
    tester = ArduinoRealisticTest()
    tester.run()

if __name__ == "__main__":
    main()
