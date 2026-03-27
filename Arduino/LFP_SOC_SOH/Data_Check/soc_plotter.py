"""
SOC_ZHU Plotter für ausgewählte Zellen
Erstellt Subplots für verschiedene Zellen untereinander
Basiert auf dem BMS_SOC_LSTM Training Script
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Konfiguration - hier können Sie die Zellen einstellen
DATA_DIR = Path(r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU")

# Output-Verzeichnis (automatisch erkannt basierend auf Script-Pfad)
SCRIPT_DIR = Path(__file__).parent  # Das Test-Verzeichnis
OUTPUT_DIR = SCRIPT_DIR  # Plots im selben Verzeichnis wie das Script speichern

# Hier können Sie die gewünschten Zellen einstellen
SELECTED_CELLS = [
    "MGFarm_18650_C01",
    "MGFarm_18650_C03", 
    "MGFarm_18650_C05",
    "MGFarm_18650_C11",
    "MGFarm_18650_C07"
]

# Optional: Zeitbereich eingrenzen (None für alle Daten)
MAX_SAMPLES = None  # None für alle Daten, oder z.B. 10000 für erste 10k Samples
SAMPLE_STEP = 1     # Jeden x-ten Datenpunkt nehmen (für Performance bei großen Datensätzen)

def load_cell_soc_data(data_dir: Path, cell_names: list):
    """Lade SOC_ZHU Daten für ausgewählte Zellen"""
    cell_data = {}
    
    for cell_name in cell_names:
        cell_folder = data_dir / cell_name
        parquet_file = cell_folder / "df.parquet"
        
        if parquet_file.exists():
            print(f"Lade {cell_name}...")
            df = pd.read_parquet(parquet_file)
            
            # Zeitstempel konvertieren
            df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
            
            # Optional: Daten begrenzen
            if MAX_SAMPLES is not None:
                df = df.head(MAX_SAMPLES)
            
            # Optional: Sampling für Performance
            if SAMPLE_STEP > 1:
                df = df.iloc[::SAMPLE_STEP].reset_index(drop=True)
            
            cell_data[cell_name] = df
            print(f"  -> {len(df)} Datenpunkte geladen")
            print(f"  -> SOC_ZHU Bereich: {df['SOC_ZHU'].min():.3f} - {df['SOC_ZHU'].max():.3f}")
            
            # Zeitspanne anzeigen
            time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            print(f"  -> Zeitspanne: {time_span:.1f} Stunden")
            
        else:
            print(f"WARNUNG: {parquet_file} nicht gefunden!")
    
    return cell_data

def plot_soc_subplots(cell_data: dict, save_path: str = "soc_comparison_subplots.png"):
    """Erstelle Subplot-Vergleich der SOC_ZHU Werte untereinander"""
    
    n_cells = len(cell_data)
    if n_cells == 0:
        print("Keine Daten zum Plotten!")
        return
    
    # Größe basierend auf Anzahl der Zellen
    fig_height = max(8, 2.5 * n_cells)
    fig, axes = plt.subplots(n_cells, 1, figsize=(14, fig_height), sharex=True)
    
    # Falls nur eine Zelle, axes zu Liste machen
    if n_cells == 1:
        axes = [axes]
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_cells))
    
    for i, (cell_name, df) in enumerate(cell_data.items()):
        ax = axes[i]
        
        # Zeitachse in Stunden seit Start
        time_hours = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600
        
        # SOC Plot
        ax.plot(time_hours, df['SOC_ZHU'], color=colors[i], linewidth=0.8, alpha=0.9)
        
        # Achsenbeschriftung und Titel
        ax.set_ylabel('SOC_ZHU', fontsize=10)
        cell_short = cell_name.replace("MGFarm_18650_", "")
        ax.set_title(f'{cell_short} - SOC Verlauf', fontsize=11, pad=10)
        
        # Grid und Limits
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Statistiken in der Ecke
        mean_soc = df['SOC_ZHU'].mean()
        std_soc = df['SOC_ZHU'].std()
        min_soc = df['SOC_ZHU'].min()
        max_soc = df['SOC_ZHU'].max()
        
        stats_text = f'μ={mean_soc:.3f}, σ={std_soc:.3f}\nMin={min_soc:.3f}, Max={max_soc:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Y-Achse Ticks
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Gemeinsame x-Achsen-Beschriftung
    axes[-1].set_xlabel('Zeit [Stunden]', fontsize=11)
    
    # Titel für gesamte Figur
    fig.suptitle('SOC_ZHU Vergleich - Ausgewählte Zellen', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Platz für Haupttitel
    
    # Vollständiger Pfad für das Speichern
    full_save_path = OUTPUT_DIR / save_path
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    print(f"Subplot-Plot gespeichert als: {full_save_path}")
    plt.show()

def plot_soc_overlay(cell_data: dict, save_path: str = "soc_overlay.png"):
    """Erstelle Overlay-Plot aller SOC_ZHU Kurven"""
    
    plt.figure(figsize=(14, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(cell_data)))
    
    for i, (cell_name, df) in enumerate(cell_data.items()):
        time_hours = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600
        label = cell_name.replace("MGFarm_18650_", "")
        plt.plot(time_hours, df['SOC_ZHU'], color=colors[i], 
                linewidth=1.2, alpha=0.8, label=label)
    
    plt.xlabel('Zeit [Stunden]', fontsize=11)
    plt.ylabel('SOC_ZHU', fontsize=11)
    plt.title('SOC Verlauf - Vergleich aller ausgewählten Zellen', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Vollständiger Pfad für das Speichern
    full_save_path = OUTPUT_DIR / save_path
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    print(f"Overlay-Plot gespeichert als: {full_save_path}")
    plt.show()

def print_soc_statistics(cell_data: dict):
    """Drucke SOC-Statistiken für alle Zellen"""
    print("\n" + "="*70)
    print("SOC_ZHU STATISTIKEN")
    print("="*70)
    
    for cell_name, df in cell_data.items():
        soc = df['SOC_ZHU']
        cell_short = cell_name.replace("MGFarm_18650_", "")
        
        print(f"\n{cell_short}:")
        print(f"  Datenpunkte: {len(soc):,}")
        print(f"  Min SOC:     {soc.min():.4f}")
        print(f"  Max SOC:     {soc.max():.4f}")
        print(f"  Mean SOC:    {soc.mean():.4f}")
        print(f"  Median SOC:  {soc.median():.4f}")
        print(f"  Std SOC:     {soc.std():.4f}")
        
        # Zeitspanne
        time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        print(f"  Zeitspanne:  {time_span:.1f} Stunden")
        
        # SOC-Bereiche
        soc_ranges = {
            "Niedrig (0-0.2)": ((soc >= 0) & (soc <= 0.2)).sum(),
            "Mittel (0.2-0.8)": ((soc > 0.2) & (soc < 0.8)).sum(),
            "Hoch (0.8-1.0)": ((soc >= 0.8) & (soc <= 1.0)).sum()
        }
        print(f"  SOC-Verteilung:")
        for range_name, count in soc_ranges.items():
            percentage = (count / len(soc)) * 100
            print(f"    {range_name}: {count:,} ({percentage:.1f}%)")

def plot_soc_histograms(cell_data: dict, save_path: str = "soc_histograms.png"):
    """Erstelle Histogramme der SOC-Verteilungen"""
    
    n_cells = len(cell_data)
    fig, axes = plt.subplots(n_cells, 1, figsize=(10, 2.5*n_cells))
    
    if n_cells == 1:
        axes = [axes]
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_cells))
    
    for i, (cell_name, df) in enumerate(cell_data.items()):
        ax = axes[i]
        cell_short = cell_name.replace("MGFarm_18650_", "")
        
        ax.hist(df['SOC_ZHU'], bins=50, color=colors[i], alpha=0.7, edgecolor='black')
        ax.set_xlabel('SOC_ZHU')
        ax.set_ylabel('Häufigkeit')
        ax.set_title(f'{cell_short} - SOC Verteilung')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    plt.tight_layout()
    
    # Vollständiger Pfad für das Speichern
    full_save_path = OUTPUT_DIR / save_path
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    print(f"Histogramm-Plot gespeichert als: {full_save_path}")
    plt.show()

if __name__ == "__main__":
    print("🔋 SOC_ZHU Plotter für LFP-Zellen")
    print(f"📂 Data Directory: {DATA_DIR}")
    print(f"🎯 Ausgewählte Zellen: {[cell.replace('MGFarm_18650_', '') for cell in SELECTED_CELLS]}")
    if MAX_SAMPLES:
        print(f"📊 Sample Limit: {MAX_SAMPLES}")
    if SAMPLE_STEP > 1:
        print(f"📉 Sample Step: jeden {SAMPLE_STEP}. Datenpunkt")
    
    # Daten laden
    print("\n" + "="*50)
    cell_data = load_cell_soc_data(DATA_DIR, SELECTED_CELLS)
    
    if not cell_data:
        print("❌ Keine Zellen-Daten gefunden!")
        print("Überprüfen Sie:")
        print(f"  - Existiert das Verzeichnis: {DATA_DIR}")
        print(f"  - Existieren die Unterordner für: {SELECTED_CELLS}")
        print(f"  - Existieren die df.parquet Dateien in den Unterordnern")
        exit(1)
    
    print(f"\n✅ {len(cell_data)} Zellen erfolgreich geladen")
    
    # Statistiken ausgeben
    print_soc_statistics(cell_data)
    # Plots erstellen
    print("\n" + "="*50)
    print("📈 Erstelle Plots...")
    
    # 1. Subplot-Version (untereinander) - Hauptplot
    plot_soc_subplots(cell_data, "soc_comparison_subplots.png")
    
    # 2. Overlay-Version (übereinander)
    plot_soc_overlay(cell_data, "soc_overlay.png")
    
    # 3. Histogramme der SOC-Verteilungen
    plot_soc_histograms(cell_data, "soc_histograms.png")
    
    print(f"\n✅ Fertig! Alle Plots wurden im Ordner gespeichert: {OUTPUT_DIR}")
    print("  - soc_comparison_subplots.png (Hauptplot - untereinander)")
    print("  - soc_overlay.png (Overlay - übereinander)")
    print("  - soc_histograms.png (SOC-Verteilungen)")
