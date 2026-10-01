import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Setze interaktives Backend für Matplotlib
matplotlib.use('TkAgg')
plt.ion()

def plot_all_features_overview(cell_id="C01"):
    """
    Plottet alle verfügbaren Features einer Zelle über die Zeit (Testtime[s]).
    Erstellt ein umfassendes Übersichts-Diagramm mit mehreren Subplots.
    
    Args:
        cell_id (str): Zellen-ID (z.B. "C01", "C03", etc.)
    """
    print(f"📊 Starte vollständige Feature-Übersicht für Zelle {cell_id}")
    
    # Pfad zum Feature-Engineering DataFrame
    base_dir = Path(__file__).parent
    df_fe_path = base_dir / f"df_FE_{cell_id}.parquet"
    
    if not df_fe_path.exists():
        print(f"❌ Feature-DataFrame nicht gefunden: {df_fe_path}")
        print(f"💡 Tipp: Führe zuerst das feature_engineering.py Script aus!")
        return
    
    try:
        # Feature-DataFrame laden
        df_fe = pd.read_parquet(df_fe_path)
        print(f"✅ Feature-DataFrame geladen: {df_fe.shape}")
        
        # Zeige alle verfügbaren Spalten
        print(f"\n🔍 Verfügbare Spalten in {cell_id}:")
        for i, col in enumerate(df_fe.columns, 1):
            print(f"  {i:2d}. {col}")
        print(f"    Gesamt: {len(df_fe.columns)} Spalten")
        
        # Prüfe ob Testtime verfügbar ist
        if 'Testtime[s]' not in df_fe.columns:
            print(f"❌ Testtime[s] Spalte nicht gefunden!")
            return
        
        # Extrahiere Testtime als X-Achse
        testtime = df_fe['Testtime[s]'].values
        time_hours = testtime / 3600  # Konvertiere zu Stunden für bessere Lesbarkeit
        
        print(f"\n📊 Zeitbereich: {time_hours.min():.2f} - {time_hours.max():.2f} h ({len(testtime)} Datenpunkte)")
        
        # Definiere Features zum Plotten (ausgenommen Zeitstempel)
        features_to_plot = []
        for col in df_fe.columns:
            if col not in ['Testtime[s]', 'Absolute_Time[yyyy-mm-dd hh:mm:ss]']:
                features_to_plot.append(col)
        
        print(f"\n🎯 Features zum Plotten: {len(features_to_plot)}")
        for i, feature in enumerate(features_to_plot, 1):
            print(f"  {i:2d}. {feature}")
        
        # Berechne optimales Layout
        n_features = len(features_to_plot)
        
        # Bestimme Anzahl Spalten und Zeilen für optimale Darstellung
        if n_features <= 4:
            cols = 2
            rows = 2
        elif n_features <= 6:
            cols = 2
            rows = 3
        elif n_features <= 9:
            cols = 3
            rows = 3
        elif n_features <= 12:
            cols = 3
            rows = 4
        else:
            cols = 4
            rows = (n_features + cols - 1) // cols
        
        print(f"📐 Plot-Layout: {rows} Zeilen × {cols} Spalten")
        
        # Erstelle große Figure
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharex=True)
        fig.suptitle(f'Vollständige Feature-Übersicht - Zelle {cell_id}\nAlle {n_features} Features über Zeit', 
                     fontsize=16, fontweight='bold')
        
        # Stelle sicher, dass axes immer 2D ist
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plotte jedes Feature
        for i, feature in enumerate(features_to_plot):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Extrahiere Feature-Daten
            feature_data = df_fe[feature].values
            
            # Bestimme Farbe basierend auf Feature-Typ
            if 'SOH' in feature:
                color = 'red'
                linestyle = '-'
                alpha = 0.8
            elif 'SOC' in feature:
                color = 'blue'
                linestyle = '-'
                alpha = 0.8
            elif 'EFC' in feature:
                color = 'green'
                linestyle = '-'
                alpha = 0.8
            elif 'Voltage' in feature:
                color = 'orange'
                linestyle = '-'
                alpha = 0.8
            elif 'Current' in feature:
                color = 'purple'
                linestyle = '-'
                alpha = 0.8
            elif 'Capacity' in feature:
                color = 'brown'
                linestyle = '-'
                alpha = 0.8
            elif 'Power' in feature:
                color = 'navy'
                linestyle = '-'
                alpha = 0.8
            elif 'C_Rate' in feature:
                color = 'darkred'
                linestyle = '-'
                alpha = 0.8
            elif 'Temperature' in feature:
                color = 'darkgreen'
                linestyle = '-'
                alpha = 0.8
            elif 'dt' in feature:  # Ableitungen
                color = 'gray'
                linestyle = '-'
                alpha = 0.7
            elif 'Resistance' in feature:
                color = 'black'
                linestyle = '-'
                alpha = 0.7
            elif 'Variability' in feature:
                color = 'pink'
                linestyle = '-'
                alpha = 0.7
            elif 'Cumulative' in feature:
                color = 'cyan'
                linestyle = '-'
                alpha = 0.7
            elif 'Q_m' in feature:  # Kumulativer Strom (vorher Cumulative_Current)
                color = 'cyan'
                linestyle = '-'
                alpha = 0.8
            elif 'Q_c' in feature:  # Kumulativer Strom mit V_max Reset
                color = 'darkturquoise'
                linestyle = '-'
                alpha = 0.8
            else:
                color = 'blue'
                linestyle = '-'
                alpha = 0.8
            
            # Plotte Feature über Zeit
            ax.plot(time_hours, feature_data, color=color, linewidth=1.5, 
                   linestyle=linestyle, alpha=alpha)
            
            # Formatierung
            ax.set_title(feature, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Y-Achsen-Label basierend auf Feature
            if 'SOH' in feature:
                ax.set_ylabel('SOH [-]')
                ax.set_ylim(0, 1.05)
                # Referenzlinien für SOH
                ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.3, linewidth=1)
                ax.axhline(y=0.8, color='orange', linestyle=':', alpha=0.3, linewidth=1)
            elif 'SOC' in feature:
                if feature_data.max() <= 1.1:  # SOC als Dezimalzahl (0-1)
                    ax.set_ylabel('SOC [-]')
                    ax.set_ylim(0, 1.05)
                    ax.axhline(y=1.0, color='blue', linestyle=':', alpha=0.3, linewidth=1)
                    ax.axhline(y=0.8, color='blue', linestyle=':', alpha=0.3, linewidth=1)
                else:  # SOC als Prozent (0-100)
                    ax.set_ylabel('SOC [%]')
                    ax.set_ylim(0, 105)
                    ax.axhline(y=100, color='blue', linestyle=':', alpha=0.3, linewidth=1)
                    ax.axhline(y=80, color='blue', linestyle=':', alpha=0.3, linewidth=1)
            elif 'Voltage' in feature:
                ax.set_ylabel('Voltage [V]')
                # LFP Referenzspannungen
                ax.axhline(y=3.65, color='red', linestyle=':', alpha=0.3, linewidth=1)
                ax.axhline(y=2.5, color='blue', linestyle=':', alpha=0.3, linewidth=1)
            elif 'Current' in feature:
                ax.set_ylabel('Current [A]')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            elif 'Power' in feature:
                ax.set_ylabel('Power [W]')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            elif 'C_Rate' in feature:
                ax.set_ylabel('C-Rate [-]')
                if feature_data.max() > 1.0:
                    ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, linewidth=1)
                if feature_data.max() > 2.0:
                    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            elif 'EFC' in feature:
                ax.set_ylabel('EFC [-]')
            elif 'Capacity' in feature:
                ax.set_ylabel('Capacity [Ah]')
            elif 'Temperature' in feature:
                ax.set_ylabel('Temperature [°C]')
            elif 'dU_dt' in feature:
                ax.set_ylabel('dU/dt [V/s]')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            elif 'dI_dt' in feature:
                ax.set_ylabel('dI/dt [A/s]')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            elif 'Resistance' in feature:
                ax.set_ylabel('Resistance [Ω]')
            elif 'Variability' in feature:
                ax.set_ylabel('Std [V]')
            elif 'Cumulative' in feature:
                ax.set_ylabel('Cumul. [Ah]')
            elif 'Q_m' in feature:  # Kumulativer Strom
                ax.set_ylabel('Q_m [Ah]')
            elif 'Q_c' in feature:  # Kumulativer Strom mit V_max Reset
                ax.set_ylabel('Q_c [Ah]')
                # Null-Linie für Reset-Punkte
                ax.axhline(y=0, color='red', linestyle=':', alpha=0.5, linewidth=1)
            else:
                ax.set_ylabel(feature)
            
            # Statistiken als Text im Plot (oben rechts)
            stats_text = f'Min: {feature_data.min():.3f}\n'
            stats_text += f'Max: {feature_data.max():.3f}\n'
            stats_text += f'Mean: {feature_data.mean():.3f}'
            
            ax.text(0.98, 0.98, stats_text, 
                   transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # X-Achsen-Label nur für unterste Reihe
            if row == rows - 1:
                ax.set_xlabel('Zeit [h]', fontweight='bold')
        
        # Verstecke leere Subplots
        for i in range(n_features, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Speichere Plot
        output_path = base_dir / f"all_features_overview_{cell_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        print(f"\n✅ Vollständige Feature-Übersicht erstellt für Zelle {cell_id}!")
        print(f"📊 Plot gespeichert als: {output_path}")
        
        # Zeige detaillierte Statistiken
        print(f"\n📊 Detaillierte Feature-Statistiken:")
        for feature in features_to_plot:
            feature_data = df_fe[feature].values
            print(f"\n  {feature}:")
            print(f"    Min:    {feature_data.min():.6f}")
            print(f"    Max:    {feature_data.max():.6f}")
            print(f"    Mean:   {feature_data.mean():.6f}")
            print(f"    Std:    {feature_data.std():.6f}")
            print(f"    Median: {np.median(feature_data):.6f}")
        
        plt.show()
        
        input("\nPlot ist interaktiv! Drücke Enter um fortzufahren...")
        
        print(f"\n✅ Vollständige Feature-Analyse abgeschlossen für Zelle {cell_id}!")
        
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der Feature-Übersicht: {e}")

if __name__ == "__main__":
    # Vollständige Feature-Übersicht für Zelle C01
    cell_id = "C01"
    print(f"🚀 Starte vollständige Feature-Übersicht für Zelle {cell_id}")
    plot_all_features_overview(cell_id)
