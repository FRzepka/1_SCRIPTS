#!/usr/bin/env python3
"""
🔬 PC vs Arduino SOC Prediction Comparison
==========================================

✨ FEATURES:
- Lädt CSV-Dateien von PC und Arduino SOC-Vorhersagen
- Plottet Ground Truth, PC Prediction und Arduino Prediction
- Vergleicht MAE zwischen PC und Arduino
- Publication-ready wissenschaftlicher Plot

📊 USAGE:
- Stelle sicher, dass PC und Arduino CSV-Dateien im gleichen Ordner sind
- Script automatisch erkennt die neuesten CSV-Dateien
- Oder gib spezifische Dateipfade an

🎯 OUTPUT:
- Vergleichsplot mit allen drei Kurven
- MAE-Vergleich in der Legende
- Gespeicherter Plot als PNG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
import argparse
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# === FARBSCHEMA ===
COLOR_SCHEME = {
    'ground_truth': '#2091C9',      # Blau für Ground Truth
    'pc_prediction': '#4CAF50',     # Grün für PC
    'arduino_prediction': '#FF6B6B', # Rot für Arduino
    'background': '#F5F5F5',        # Heller Hintergrund
    'grid': '#E0E0E0'              # Subtiles Grid
}

print("🔬 PC vs Arduino SOC Prediction Comparison")
print("🎯 Wissenschaftlicher Vergleich zwischen PC und Arduino LSTM")
print("="*65)

class SOCComparisonAnalyzer:
    """Analysiert und vergleicht PC vs Arduino SOC Vorhersagen"""
    
    def __init__(self, work_dir="."):
        self.work_dir = Path(work_dir)
        self.pc_data = None
        self.arduino_data = None
        self.comparison_data = None
        
    def find_latest_csv_files(self):
        """Findet automatisch die neuesten PC und Arduino CSV-Dateien"""
        try:
            # Suche PC CSV-Dateien
            pc_pattern = self.work_dir / "pc_soc_data_*.csv"
            pc_files = glob.glob(str(pc_pattern))
            pc_files.sort(key=os.path.getmtime, reverse=True)  # Neueste zuerst
            
            # Suche Arduino CSV-Dateien  
            arduino_pattern = self.work_dir / "arduino_soc_data_*.csv"
            arduino_files = glob.glob(str(arduino_pattern))
            arduino_files.sort(key=os.path.getmtime, reverse=True)  # Neueste zuerst
            
            if not pc_files:
                print("❌ Keine PC CSV-Dateien gefunden (pc_soc_data_*.csv)")
                return None, None
                
            if not arduino_files:
                print("❌ Keine Arduino CSV-Dateien gefunden (arduino_soc_data_*.csv)")
                return None, None
            
            pc_file = pc_files[0]
            arduino_file = arduino_files[0]
            
            print(f"📄 PC CSV gefunden: {Path(pc_file).name}")
            print(f"📄 Arduino CSV gefunden: {Path(arduino_file).name}")
            
            return pc_file, arduino_file
            
        except Exception as e:
            print(f"❌ Fehler beim Suchen der CSV-Dateien: {e}")
            return None, None
    
    def load_data(self, pc_csv_path=None, arduino_csv_path=None):
        """Lädt PC und Arduino CSV-Daten"""
        try:
            # Automatische Erkennung falls keine Pfade angegeben
            if pc_csv_path is None or arduino_csv_path is None:
                auto_pc, auto_arduino = self.find_latest_csv_files()
                pc_csv_path = pc_csv_path or auto_pc
                arduino_csv_path = arduino_csv_path or auto_arduino
            
            if not pc_csv_path or not arduino_csv_path:
                raise FileNotFoundError("PC oder Arduino CSV-Datei nicht gefunden")
            
            print(f"\n📊 Lade Daten...")
            print(f"   PC:      {Path(pc_csv_path).name}")
            print(f"   Arduino: {Path(arduino_csv_path).name}")
            
            # Lade PC-Daten
            self.pc_data = pd.read_csv(pc_csv_path)
            print(f"✅ PC-Daten geladen: {len(self.pc_data)} Zeilen")
            
            # Lade Arduino-Daten
            self.arduino_data = pd.read_csv(arduino_csv_path)
            print(f"✅ Arduino-Daten geladen: {len(self.arduino_data)} Zeilen")
            
            # Validiere Datenstruktur
            self._validate_data()
            
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Laden der Daten: {e}")
            return False
    
    def _validate_data(self):
        """Validiert die geladenen Daten"""
        # PC-Daten validieren
        required_pc_cols = ['time_seconds', 'soc_ground_truth', 'soc_pc_prediction']
        missing_pc = [col for col in required_pc_cols if col not in self.pc_data.columns]
        if missing_pc:
            raise ValueError(f"PC CSV fehlt Spalten: {missing_pc}")
        
        # Arduino-Daten validieren
        required_arduino_cols = ['time_seconds', 'soc_ground_truth', 'soc_arduino_prediction']
        missing_arduino = [col for col in required_arduino_cols if col not in self.arduino_data.columns]
        if missing_arduino:
            raise ValueError(f"Arduino CSV fehlt Spalten: {missing_arduino}")
        
        print("✅ Datenvalidierung erfolgreich")
    
    def merge_data(self):
        """Merged PC und Arduino Daten basierend auf time_seconds"""
        try:
            print(f"\n🔗 Merge PC und Arduino Daten...")
            
            # Merge basierend auf time_seconds
            merged = pd.merge(
                self.pc_data[['time_seconds', 'soc_ground_truth', 'soc_pc_prediction']],
                self.arduino_data[['time_seconds', 'soc_arduino_prediction', 'prediction_successful']],
                on='time_seconds',
                how='inner'
            )
            
            # Filtere nur erfolgreiche Arduino-Vorhersagen
            merged = merged[merged['prediction_successful'] == True].copy()
            
            # Entferne NaN-Werte
            merged = merged.dropna(subset=['soc_pc_prediction', 'soc_arduino_prediction'])
            
            self.comparison_data = merged
            
            print(f"✅ Daten gemerged: {len(self.comparison_data)} gemeinsame Zeitpunkte")
            print(f"📊 Zeitbereich: {self.comparison_data['time_seconds'].min()}s - {self.comparison_data['time_seconds'].max()}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Mergen der Daten: {e}")
            return False
    
    def calculate_metrics(self):
        """Berechnet Vergleichsmetriken"""
        try:
            if self.comparison_data is None:
                raise ValueError("Keine Vergleichsdaten verfügbar")
            
            # MAE für PC und Arduino
            pc_mae = mean_absolute_error(
                self.comparison_data['soc_ground_truth'], 
                self.comparison_data['soc_pc_prediction']
            )
            
            arduino_mae = mean_absolute_error(
                self.comparison_data['soc_ground_truth'], 
                self.comparison_data['soc_arduino_prediction']
            )
            
            # Relative Verbesserung/Verschlechterung
            relative_change = ((arduino_mae - pc_mae) / pc_mae) * 100
            
            # Weitere Statistiken
            pc_max_error = np.max(np.abs(
                self.comparison_data['soc_ground_truth'] - self.comparison_data['soc_pc_prediction']
            ))
            arduino_max_error = np.max(np.abs(
                self.comparison_data['soc_ground_truth'] - self.comparison_data['soc_arduino_prediction']
            ))
            
            metrics = {
                'pc_mae': pc_mae,
                'arduino_mae': arduino_mae,
                'relative_change_percent': relative_change,
                'pc_max_error': pc_max_error,
                'arduino_max_error': arduino_max_error,
                'samples_count': len(self.comparison_data)
            }
            
            print(f"\n📈 Vergleichsmetriken:")
            print(f"   PC MAE:        {pc_mae:.6f}")
            print(f"   Arduino MAE:   {arduino_mae:.6f}")
            print(f"   Unterschied:   {relative_change:+.2f}% ({arduino_mae - pc_mae:+.6f})")
            print(f"   PC Max Error:  {pc_max_error:.6f}")
            print(f"   Arduino Max:   {arduino_max_error:.6f}")
            print(f"   Samples:       {metrics['samples_count']:,}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Fehler bei Metrik-Berechnung: {e}")
            return None
    
    def create_comparison_plot(self, metrics):
        """Erstellt wissenschaftlichen Vergleichsplot"""
        try:
            print(f"\n📊 Erstelle Vergleichsplot...")
            
            # Figure Setup
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            fig.suptitle('PC vs Arduino SOC Prediction Comparison', 
                        fontsize=28, y=0.98)  # Hauptüberschrift ohne bold
            
            # Daten für Plot
            time_axis = self.comparison_data['time_seconds']
            ground_truth = self.comparison_data['soc_ground_truth']
            pc_pred = self.comparison_data['soc_pc_prediction']
            arduino_pred = self.comparison_data['soc_arduino_prediction']
            
            # Plot 1: SOC Vergleich
            ax1.plot(time_axis, ground_truth, 
                    color=COLOR_SCHEME['ground_truth'], linewidth=2.5, 
                    label='Ground Truth SOC', alpha=0.9, zorder=3)
            
            ax1.plot(time_axis, pc_pred, 
                    color=COLOR_SCHEME['pc_prediction'], linewidth=2.0, 
                    label=f'PC LSTM (MAE: {metrics["pc_mae"]:.4f})', alpha=0.8, zorder=2)
            
            ax1.plot(time_axis, arduino_pred, 
                    color=COLOR_SCHEME['arduino_prediction'], linewidth=1.8, linestyle='--',
                    label=f'Arduino LSTM (MAE: {metrics["arduino_mae"]:.4f})', alpha=0.8, zorder=1)
            
            ax1.set_xlabel('Time [seconds]', fontsize=22)  # Achsenbeschriftung ohne bold
            ax1.set_ylabel('State of Charge (SOC)', fontsize=22)
            ax1.legend(fontsize=20, loc='best', framealpha=0.9)  # Legende ohne bold
            ax1.grid(True, alpha=0.3, color=COLOR_SCHEME['grid'])
            ax1.set_ylim(0, 1)
            
            # Plot 2: Fehler-Vergleich
            pc_errors = np.abs(ground_truth - pc_pred)
            arduino_errors = np.abs(ground_truth - arduino_pred)
            
            ax2.plot(time_axis, pc_errors, 
                    color=COLOR_SCHEME['pc_prediction'], linewidth=2.0, 
                    label=f'PC Absolute Error (Avg: {np.mean(pc_errors):.4f})', alpha=0.8)
            
            ax2.plot(time_axis, arduino_errors, 
                    color=COLOR_SCHEME['arduino_prediction'], linewidth=1.8, linestyle='--',
                    label=f'Arduino Absolute Error (Avg: {np.mean(arduino_errors):.4f})', alpha=0.8)
            
            # Durchschnittslinien
            ax2.axhline(y=np.mean(pc_errors), color=COLOR_SCHEME['pc_prediction'], 
                       linestyle=':', alpha=0.6, linewidth=1)
            ax2.axhline(y=np.mean(arduino_errors), color=COLOR_SCHEME['arduino_prediction'], 
                       linestyle=':', alpha=0.6, linewidth=1)
            
            ax2.set_title('Absolute Error Comparison', fontsize=24, pad=15)  # Überschrift ohne bold
            ax2.set_xlabel('Time [seconds]', fontsize=22)  # Achsenbeschriftung ohne bold
            ax2.set_ylabel('Absolute Error', fontsize=22)
            ax2.legend(fontsize=20, loc='best', framealpha=0.9)  # Legende ohne bold
            ax2.grid(True, alpha=0.3, color=COLOR_SCHEME['grid'])
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Mehr Platz oben für Überschrift
            
            # Plot speichern
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"pc_vs_arduino_comparison_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Vergleichsplot gespeichert: {filename}")
            
            # Plot anzeigen
            plt.show()
            
            return filename
            
        except Exception as e:
            print(f"❌ Fehler beim Erstellen des Plots: {e}")
            return None
    
    def run_comparison(self, pc_csv_path=None, arduino_csv_path=None):
        """Führt komplette Vergleichsanalyse durch"""
        try:
            # Daten laden
            if not self.load_data(pc_csv_path, arduino_csv_path):
                return False
            
            # Daten mergen
            if not self.merge_data():
                return False
            
            # Metriken berechnen
            metrics = self.calculate_metrics()
            if metrics is None:
                return False
            
            # Plot erstellen
            plot_file = self.create_comparison_plot(metrics)
            
            print(f"\n✅ Vergleichsanalyse abgeschlossen!")
            print(f"📊 Plot: {plot_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Fehler bei Vergleichsanalyse: {e}")
            return False

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description='PC vs Arduino SOC Prediction Comparison')
    parser.add_argument('--pc-csv', help='Path to PC CSV file')
    parser.add_argument('--arduino-csv', help='Path to Arduino CSV file')
    parser.add_argument('--work-dir', default='.', help='Working directory (default: current)')
    
    args = parser.parse_args()
    
    analyzer = SOCComparisonAnalyzer(work_dir=args.work_dir)
    
    try:
        success = analyzer.run_comparison(
            pc_csv_path=args.pc_csv,
            arduino_csv_path=args.arduino_csv
        )
        
        if success:
            print("🎉 Analyse erfolgreich abgeschlossen!")
        else:
            print("❌ Analyse fehlgeschlagen!")
            
    except KeyboardInterrupt:
        print("⏹️ Analyse abgebrochen")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")

if __name__ == "__main__":
    main()
