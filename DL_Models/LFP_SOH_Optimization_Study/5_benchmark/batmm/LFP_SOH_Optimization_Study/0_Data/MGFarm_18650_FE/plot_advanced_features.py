import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Setze interaktives Backend für Matplotlib
matplotlib.use('TkAgg')
plt.ion()

def plot_advanced_features(cell_id="C01"):
    """
    Plottet die erweiterten Features aus dem Feature Engineering über die Zeit.
    
    Features:
    - C_Rate: Belastungsanalyse |Strom| / aktuelle_Kapazität
    - Power[W]: Instantane Leistung U × I
    - dU_dt[V/s]: Spannungsableitung (Dynamik)
    - dI_dt[A/s]: Stromableitung (Dynamik)
    
    Args:
        cell_id (str): Zellen-ID (z.B. "C01", "C03", etc.)
    """
    print(f"📊 Starte Plot der erweiterten Features für Zelle {cell_id}")
    
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
        
        # Verfügbare Spalten überprüfen
        required_features = ['Testtime[s]', 'C_Rate', 'Power[W]', 'dU_dt[V/s]', 'dI_dt[A/s]']
        missing_features = [feat for feat in required_features if feat not in df_fe.columns]
        
        if missing_features:
            print(f"❌ Fehlende Features: {missing_features}")
            print(f"🔍 Verfügbare Spalten: {list(df_fe.columns)}")
            return
        
        # Daten extrahieren
        testtime = df_fe['Testtime[s]'].values
        c_rate = df_fe['C_Rate'].values
        power = df_fe['Power[W]'].values
        du_dt = df_fe['dU_dt[V/s]'].values
        di_dt = df_fe['dI_dt[A/s]'].values
        soc_calculated = df_fe['SOC_calculated'].values if 'SOC_calculated' in df_fe.columns else None
        soh = df_fe['SOH'].values if 'SOH' in df_fe.columns else None
        
        # Entferne die ZHU-SOC Lade-Logik (89 Zeilen)
        
        # Zeit in Stunden konvertieren für bessere Lesbarkeit
        time_hours = testtime / 3600
        
        print(f"📊 Datenbereich:")
        print(f"  Zeit: {time_hours.min():.2f} - {time_hours.max():.2f} h")
        print(f"  C-Rate: {c_rate.min():.4f} - {c_rate.max():.4f}")
        print(f"  Leistung: {power.min():.2f} - {power.max():.2f} W")
        print(f"  dU/dt: {du_dt.min():.6f} - {du_dt.max():.6f} V/s")
        print(f"  dI/dt: {di_dt.min():.6f} - {di_dt.max():.6f} A/s")
        if soc_calculated is not None:
            print(f"  SOC (berechnet): {soc_calculated.min():.4f} - {soc_calculated.max():.4f}")
        if soh is not None:
            print(f"  SOH: {soh.min():.4f} - {soh.max():.4f}")
        
        # Erstelle erweiterte Subplot-Darstellung (5 statt 4 wenn SOC verfügbar)
        num_plots = 5 if soc_calculated is not None else 4
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3.5 * num_plots), sharex=True)
        fig.suptitle(f'Erweiterte Features für neuronale Netze - Zelle {cell_id}', fontsize=16, fontweight='bold')
        
        # 1. SOC & SOH Kombinations-Plot (wenn verfügbar)
        plot_idx = 0
        if soc_calculated is not None:
            # Hauptachse für SOC (links)
            ax1 = axes[plot_idx]
            line1 = ax1.plot(time_hours, soc_calculated * 100, color='blue', linewidth=2, 
                           label='SOC (berechnet)', alpha=0.8)
            ax1.set_ylabel('SOC [%]', color='blue', fontweight='bold')
            ax1.set_title('SOC & SOH: Ladezustand und Gesundheitszustand')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 105)
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Referenzlinien für SOC
            ax1.axhline(y=100, color='blue', linestyle=':', alpha=0.3, linewidth=1)
            ax1.axhline(y=80, color='blue', linestyle=':', alpha=0.3, linewidth=1)
            ax1.axhline(y=50, color='blue', linestyle=':', alpha=0.3, linewidth=1)
            
            # Zweite Y-Achse für SOH (rechts)
            if soh is not None:
                ax2 = ax1.twinx()
                line2 = ax2.plot(time_hours, soh, color='red', linewidth=2, 
                               label='SOH', alpha=0.8, linestyle='--')
                ax2.set_ylabel('SOH [-]', color='red', fontweight='bold')
                ax2.set_ylim(0, 1.05)  # SOH von 0 bis 1 (0% bis 100%)
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Referenzlinien für SOH
                ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.3, linewidth=1)
                ax2.axhline(y=0.8, color='red', linestyle=':', alpha=0.3, linewidth=1)
                ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, linewidth=1)
                
                # Kombinierte Legende
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right')
            else:
                ax1.legend(loc='upper right')
            
            plot_idx += 1
        
        # 2. C-Rate (Belastungsanalyse)
        axes[plot_idx].plot(time_hours, c_rate, color='red', linewidth=1.5, alpha=0.8)
        axes[plot_idx].set_ylabel('C-Rate [-]')
        axes[plot_idx].set_title('C-Rate: Belastungsanalyse (|Strom| / initiale Kapazität)')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_ylim(0, max(c_rate.max() * 1.1, 1.0))  # Mindestens bis 1C anzeigen
        
        # Markiere kritische C-Rates
        if c_rate.max() > 1.0:
            axes[plot_idx].axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='1C')
        if c_rate.max() > 2.0:
            axes[plot_idx].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='2C')
        if c_rate.max() > 1.0:
            axes[plot_idx].legend(loc='upper right')
        
        plot_idx += 1
        
        # 3. Instantane Leistung
        axes[plot_idx].plot(time_hours, power, color='blue', linewidth=1.5, alpha=0.8)
        axes[plot_idx].set_ylabel('Leistung [W]')
        axes[plot_idx].set_title('Instantane Leistung: P = U × I')
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Markiere Nulllinie
        axes[plot_idx].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Farbcode für Laden (positiv) und Entladen (negativ)
        positive_mask = power > 0
        negative_mask = power < 0
        if np.any(positive_mask):
            axes[plot_idx].fill_between(time_hours, 0, power, where=positive_mask, 
                                       color='green', alpha=0.3, label='Laden')
        if np.any(negative_mask):
            axes[plot_idx].fill_between(time_hours, 0, power, where=negative_mask, 
                                       color='red', alpha=0.3, label='Entladen')
        axes[plot_idx].legend(loc='upper right')
        
        plot_idx += 1
        
        # 4. Spannungsableitung (dU/dt)
        axes[plot_idx].plot(time_hours, du_dt * 1000, color='green', linewidth=1.5, alpha=0.8)  # Konvertiere zu mV/s
        axes[plot_idx].set_ylabel('dU/dt [mV/s]')
        axes[plot_idx].set_title('Spannungsableitung: Dynamisches Verhalten')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plot_idx += 1
        
        # 5. Stromableitung (dI/dt)
        axes[plot_idx].plot(time_hours, di_dt * 1000, color='purple', linewidth=1.5, alpha=0.8)  # Konvertiere zu mA/s
        axes[plot_idx].set_ylabel('dI/dt [mA/s]')
        axes[plot_idx].set_xlabel('Zeit [h]')
        axes[plot_idx].set_title('Stromableitung: Dynamisches Verhalten')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # Speichere Plot
        output_path = base_dir / f"advanced_features_{cell_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        print(f"\n✅ Erweiterte Feature-Plots erstellt für Zelle {cell_id}!")
        print(f"📊 Plot gespeichert als: {output_path}")
        plt.show()
        
        # Feature-Statistiken ausgeben
        print(f"\n📊 Feature-Statistiken:")
        features_stats = {
            'C-Rate': c_rate,
            'Leistung [W]': power,
            'dU/dt [mV/s]': du_dt * 1000,
            'dI/dt [mA/s]': di_dt * 1000
        }
        
        if soc_calculated is not None:
            features_stats['SOC (berechnet) [%]'] = soc_calculated * 100
        if soh is not None:
            features_stats['SOH [-]'] = soh
        
        for name, values in features_stats.items():
            print(f"  {name}:")
            print(f"    Min: {values.min():.4f}")
            print(f"    Max: {values.max():.4f}")
            print(f"    Mittel: {values.mean():.4f}")
            print(f"    Std: {values.std():.4f}")
        
        # SOC und SOH Korrelationsanalyse wenn beide verfügbar
        if soc_calculated is not None and soh is not None:
            correlation = np.corrcoef(soc_calculated, soh)[0, 1]
            print(f"\n🔍 SOC-SOH Korrelationsanalyse:")
            print(f"  Korrelationskoeffizient: {correlation:.4f}")
            if abs(correlation) > 0.7:
                strength = "stark"
            elif abs(correlation) > 0.4:
                strength = "mittel"
            else:
                strength = "schwach"
            print(f"  Korrelationsstärke: {strength}")
            
            # SOH-Degradation über Zeit
            soh_start = soh[0]
            soh_end = soh[-1]
            soh_degradation = (soh_start - soh_end) / soh_start * 100
            print(f"  SOH-Degradation über Testzeit: {soh_degradation:.2f}%")
        
        input("\nPlot ist interaktiv! Drücke Enter um fortzufahren...")
        
        print(f"\n✅ Erweiterte Feature-Analyse abgeschlossen für Zelle {cell_id}!")
        
    except Exception as e:
        print(f"❌ Fehler beim Plotten der Features: {e}")

def compare_multiple_cells(cell_ids=["C01", "C03", "C05"]):
    """
    Vergleicht die Features zwischen mehreren Zellen (falls verfügbar)
    
    Args:
        cell_ids (list): Liste der Zellen-IDs zum Vergleich
    """
    print(f"🔄 Vergleiche Features zwischen Zellen: {cell_ids}")
    
    base_dir = Path(__file__).parent
    available_cells = []
    
    # Prüfe welche Zellen verfügbar sind
    for cell_id in cell_ids:
        df_path = base_dir / f"df_FE_{cell_id}.parquet"
        if df_path.exists():
            available_cells.append(cell_id)
        else:
            print(f"⚠️ Zelle {cell_id} nicht verfügbar")
    
    if len(available_cells) < 2:
        print(f"❌ Mindestens 2 Zellen benötigt für Vergleich. Verfügbar: {available_cells}")
        return
    
    print(f"✅ Vergleiche Zellen: {available_cells}")
    
    # Erstelle Vergleichsplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Feature-Vergleich zwischen Zellen: {", ".join(available_cells)}', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, cell_id in enumerate(available_cells):
        df_fe = pd.read_parquet(base_dir / f"df_FE_{cell_id}.parquet")
        time_hours = df_fe['Testtime[s]'].values / 3600
        color = colors[i % len(colors)]
        
        # Plot verschiedene Features
        axes[0, 0].plot(time_hours, df_fe['C_Rate'].values, color=color, label=f'Zelle {cell_id}', alpha=0.8)
        axes[0, 1].plot(time_hours, df_fe['Power[W]'].values, color=color, label=f'Zelle {cell_id}', alpha=0.8)
        axes[1, 0].plot(time_hours, df_fe['dU_dt[V/s]'].values * 1000, color=color, label=f'Zelle {cell_id}', alpha=0.8)
        axes[1, 1].plot(time_hours, df_fe['dI_dt[A/s]'].values * 1000, color=color, label=f'Zelle {cell_id}', alpha=0.8)
    
    # Titel und Labels setzen
    axes[0, 0].set_title('C-Rate Vergleich')
    axes[0, 0].set_ylabel('C-Rate [-]')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Leistung Vergleich')
    axes[0, 1].set_ylabel('Leistung [W]')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('dU/dt Vergleich')
    axes[1, 0].set_ylabel('dU/dt [mV/s]')
    axes[1, 0].set_xlabel('Zeit [h]')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('dI/dt Vergleich')
    axes[1, 1].set_ylabel('dI/dt [mA/s]')
    axes[1, 1].set_xlabel('Zeit [h]')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = base_dir / f"features_comparison_{'_'.join(available_cells)}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Vergleichsplot gespeichert: {output_path}")
    plt.show()

def plot_all_soc_comparison():
    """
    Plottet SOC-Kurven von allen verfügbaren Zellen (C01 bis C29) in einem Plot
    """
    print(f"📊 Starte SOC-Vergleich aller verfügbaren Zellen")
    
    base_dir = Path(__file__).parent
    available_cells = []
    soc_data = {}
    
    # Suche alle verfügbaren FE-Dateien
    cell_numbers = [f"{i:02d}" for i in range(1, 30, 2)]  # C01, C03, C05, ..., C29
    
    for cell_num in cell_numbers:
        cell_id = f"C{cell_num}"
        df_path = base_dir / f"df_FE_{cell_id}.parquet"
        
        if df_path.exists():
            try:
                df_fe = pd.read_parquet(df_path)
                if 'SOC_calculated' in df_fe.columns and 'Testtime[s]' in df_fe.columns:
                    available_cells.append(cell_id)
                    
                    # Zeit in Stunden konvertieren
                    time_hours = df_fe['Testtime[s]'].values / 3600
                    soc_percent = df_fe['SOC_calculated'].values * 100
                    
                    # Lade SOH-Daten aus FE-DataFrame
                    soh = df_fe['SOH'].values if 'SOH' in df_fe.columns else None
                    if soh is not None:
                        print(f"  ✅ {cell_id}: FE={len(time_hours)} Punkte, SOH={soh.min():.3f}-{soh.max():.3f}")
                    else:
                        print(f"  ✅ {cell_id}: FE={len(time_hours)} Punkte, kein SOH")
                    
                    soc_data[cell_id] = {
                        'time': time_hours,
                        'soc': soc_percent,
                        'soh': soh
                    }
                else:
                    print(f"  ⚠️ {cell_id}: SOC_calculated oder Testtime nicht gefunden")
            except Exception as e:
                print(f"  ❌ {cell_id}: Fehler beim Laden - {e}")
        else:
            print(f"  ➖ {cell_id}: Datei nicht gefunden")
    
    if len(available_cells) == 0:
        print("❌ Keine gültigen SOC-Daten gefunden!")
        return
    
    print(f"\n📈 Erstelle SOC-Vergleichsplot für {len(available_cells)} Zellen: {', '.join(available_cells)}")
    
    # Berechne Anzahl Reihen (3 Spalten pro Reihe)
    cols = 3
    rows = (len(available_cells) + cols - 1) // cols  # Aufrunden
    
    # Erstelle Subplot-Grid
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    fig.suptitle(f'SOC & SOH Kombination (C01-C29) - {len(available_cells)} Zellen', 
                 fontsize=16, fontweight='bold')
    
    # Falls nur eine Reihe: axes zu Liste machen
    if rows == 1:
        axes = [axes] if cols == 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Plotte jede Zelle in separatem Subplot
    for i, cell_id in enumerate(available_cells):
        time_data = soc_data[cell_id]['time']
        soc_data_cell = soc_data[cell_id]['soc']
        soh_data = soc_data[cell_id]['soh']
        
        ax = axes[i]
        
        # Plotte SOC (blau, linke Achse)
        line1 = ax.plot(time_data, soc_data_cell, 
                       color='blue', 
                       linewidth=2, 
                       alpha=0.8,
                       label='SOC [%]')
        
        ax.set_xlabel('Zeit [h]', fontsize=10)
        ax.set_ylabel('SOC [%]', color='blue', fontsize=10, fontweight='bold')
        ax.set_title(f'Zelle {cell_id}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        ax.tick_params(axis='y', labelcolor='blue')
        
        # SOC Referenzlinien
        ax.axhline(y=100, color='blue', linestyle=':', alpha=0.3, linewidth=1)
        ax.axhline(y=80, color='blue', linestyle=':', alpha=0.3, linewidth=1)
        ax.axhline(y=50, color='blue', linestyle=':', alpha=0.3, linewidth=1)
        
        # Plotte SOH falls vorhanden (rot, rechte Achse)
        if soh_data is not None:
            ax2 = ax.twinx()
            line2 = ax2.plot(time_data, soh_data, 
                           color='red', 
                           linewidth=2, 
                           alpha=0.7,
                           linestyle='--',
                           label='SOH [-]')
            
            ax2.set_ylabel('SOH [-]', color='red', fontsize=10, fontweight='bold')
            ax2.set_ylim(0, 1.05)  # SOH von 0 bis 1 (0% bis 100%)
            ax2.tick_params(axis='y', labelcolor='red')
            
            # SOH Referenzlinien
            ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.3, linewidth=1)
            ax2.axhline(y=0.8, color='red', linestyle=':', alpha=0.3, linewidth=1)
            ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, linewidth=1)
            
            # Kombinierte Legende
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right', fontsize=8)
        else:
            ax.legend(line1, ['SOC [%]'], loc='upper right', fontsize=8)
        
        # Zeige wichtige Statistiken im Plot
        soc_min = soc_data_cell.min()
        soc_max = soc_data_cell.max()
        duration = time_data.max()
        
        stats_text = f'Dauer: {duration:.1f}h\nSOC: {soc_min:.1f}%-{soc_max:.1f}%'
        
        # SOH-Statistiken hinzufügen falls verfügbar
        if soh_data is not None:
            soh_min = soh_data.min()
            soh_max = soh_data.max()
            soh_degradation = (soh_data[0] - soh_data[-1]) / soh_data[0] * 100
            stats_text += f'\nSOH: {soh_min:.3f}-{soh_max:.3f}\nDegrad.: {soh_degradation:.1f}%'
        
        ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Leere Subplots ausblenden
    for i in range(len(available_cells), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Speichere Plot
    output_path = base_dir / "soc_soh_combination_all_cells.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✅ SOC & SOH Kombinations-Plot erstellt!")
    print(f"📊 Plot gespeichert als: {output_path}")
    print(f"📐 Layout: {rows} Reihen × {cols} Spalten für {len(available_cells)} Zellen")
    
    # Zeige Statistiken
    print(f"\n📊 SOC & SOH Statistiken:")
    for cell_id in available_cells:
        soc_values = soc_data[cell_id]['soc']
        time_values = soc_data[cell_id]['time']
        soh_values = soc_data[cell_id]['soh']
        
        print(f"  {cell_id}:")
        print(f"    Dauer: {time_values.max():.1f}h")
        print(f"    SOC-Bereich: {soc_values.min():.1f}% - {soc_values.max():.1f}%")
        if soh_values is not None:
            soh_degradation = (soh_values[0] - soh_values[-1]) / soh_values[0] * 100
            print(f"    SOH-Bereich: {soh_values.min():.3f} - {soh_values.max():.3f}")
            print(f"    SOH-Degradation: {soh_degradation:.2f}%")
        else:
            print(f"    SOH: nicht verfügbar")
    
    plt.show()
    
    input("\nPlot ist interaktiv! Drücke Enter um fortzufahren...")
    
    print(f"\n✅ SOC & SOH Kombinations-Plot für alle Zellen abgeschlossen!")

def plot_all_capacity_over_efc():
    """
    Plottet den SOH über EFC (Equivalent Full Cycles) für alle verfügbaren Zellen.
    Zeigt SOH-Degradation über die Zyklen hinweg.
    """
    print(f"📊 Starte SOH-Plot über EFC für alle Zellen")
    
    # Pfad zum FE-Ordner
    base_dir = Path(__file__).parent
    
    # Suche alle verfügbaren FE-DataFrames
    fe_files = list(base_dir.glob("df_FE_*.parquet"))
    if not fe_files:
        print(f"❌ Keine Feature-DataFrames gefunden!")
        print(f"💡 Tipp: Führe zuerst das feature_engineering.py Script aus!")
        return
    
    # Extrahiere Zellen-IDs aus Dateinamen
    cell_ids = [f.stem.replace("df_FE_", "") for f in fe_files]
    cell_ids.sort()  # Sortiere für konsistente Reihenfolge
    
    print(f"✅ Gefundene FE-DataFrames für {len(cell_ids)} Zellen: {cell_ids}")
    
    # Sammle Daten für alle Zellen
    soh_data = {}
    available_cells = []
    
    for cell_id in cell_ids:
        df_fe_path = base_dir / f"df_FE_{cell_id}.parquet"
        
        try:
            # DataFrame laden
            df_fe = pd.read_parquet(df_fe_path)
            
            # Drucke Spaltennamen für die erste Zelle
            if cell_id == cell_ids[0]:
                print(f"\n🔍 Verfügbare Spalten in {cell_id}:")
                for i, col in enumerate(df_fe.columns, 1):
                    print(f"  {i:2d}. {col}")
                print(f"    Gesamt: {len(df_fe.columns)} Spalten")
            
            # Prüfe ob notwendige Spalten vorhanden sind
            if 'SOH' not in df_fe.columns:
                print(f"❌ Zelle {cell_id}: Keine 'SOH' Spalte gefunden!")
                continue
                
            if 'EFC' not in df_fe.columns:
                print(f"❌ Zelle {cell_id}: Keine 'EFC' Spalte gefunden!")
                continue
            
            # Extrahiere Daten
            efc = df_fe['EFC'].values
            soh = df_fe['SOH'].values
            
            # Filtere gültige Werte (keine NaN)
            valid_mask = ~(np.isnan(efc) | np.isnan(soh))
            efc_clean = efc[valid_mask]
            soh_clean = soh[valid_mask]
            
            if len(efc_clean) > 0:
                soh_data[cell_id] = {
                    'efc': efc_clean,
                    'soh': soh_clean
                }
                available_cells.append(cell_id)
                print(f"  ✅ {cell_id}: {len(efc_clean)} Datenpunkte, EFC: {efc_clean.min():.1f} - {efc_clean.max():.1f}, SOH: {soh_clean.min():.4f} - {soh_clean.max():.4f}")
            else:
                print(f"  ❌ {cell_id}: Keine gültigen Daten!")
                
        except Exception as e:
            print(f"❌ Fehler beim Laden von {cell_id}: {e}")
    
    if not available_cells:
        print(f"❌ Keine Zellen mit gültigen SOH-/EFC-Daten gefunden!")
        return
    
    print(f"\n📊 Erstelle SOH-Plot für {len(available_cells)} Zellen...")
    
    # Erstelle Plot-Layout
    n_cells = len(available_cells)
    cols = 3  # 3 Spalten
    rows = (n_cells + cols - 1) // cols  # Aufrunden für genügend Reihen
    
    # Erstelle große Figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)  # Stelle sicher, dass axes 2D ist
    
    fig.suptitle('SOH-Degradation über EFC (Equivalent Full Cycles)\nAlle Zellen', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Farben für verschiedene Zellen
    colors = plt.cm.tab10(np.linspace(0, 1, len(available_cells)))
    
    # Plotte jede Zelle
    for i, cell_id in enumerate(available_cells):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        efc = soh_data[cell_id]['efc']
        soh = soh_data[cell_id]['soh']
        
        # Hauptplot: SOH über EFC
        ax.plot(efc, soh, color=colors[i], linewidth=2, alpha=0.8, label=f'{cell_id}')
        
        # Formatierung
        ax.set_xlabel('EFC (Equivalent Full Cycles)', fontsize=10)
        ax.set_ylabel('SOH [-]', fontsize=10)
        ax.set_title(f'Zelle {cell_id}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)  # SOH von 0 bis 1.05
        
        # Referenzlinien für SOH
        ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, linewidth=1, label='100% SOH')
        ax.axhline(y=0.8, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='80% SOH')
        ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=1, label='50% SOH')
        
        # Statistiken berechnen
        initial_soh = soh[0] if len(soh) > 0 else 0
        final_soh = soh[-1] if len(soh) > 0 else 0
        soh_loss = initial_soh - final_soh
        soh_loss_percent = (soh_loss / initial_soh * 100) if initial_soh > 0 else 0
        max_efc = efc.max() if len(efc) > 0 else 0
        
        # Zusätzliche Infos als Text
        info_text = f"Start: {initial_soh:.4f}\n"
        info_text += f"Ende: {final_soh:.4f}\n"
        info_text += f"Verlust: {soh_loss:.4f} ({soh_loss_percent:.1f}%)\n"
        info_text += f"Max EFC: {max_efc:.0f}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Leere Subplots verstecken
    for i in range(n_cells, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Speichere Plot
    output_path = base_dir / "soh_over_efc_all_cells.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✅ SOH-Plot über EFC erstellt!")
    print(f"📊 Plot gespeichert als: {output_path}")
    print(f"📐 Layout: {rows} Reihen × {cols} Spalten für {len(available_cells)} Zellen")
    
    # Zeige Degradations-Statistiken
    print(f"\n📊 SOH-Degradations-Statistiken:")
    for cell_id in available_cells:
        efc = soh_data[cell_id]['efc']
        soh = soh_data[cell_id]['soh']
        
        if len(soh) > 1:
            initial_soh = soh[0]
            final_soh = soh[-1]
            soh_loss = initial_soh - final_soh
            soh_loss_percent = (soh_loss / initial_soh * 100)
            max_efc = efc.max()
            
            print(f"  {cell_id}:")
            print(f"    EFC-Bereich: 0 - {max_efc:.0f}")
            print(f"    SOH: {initial_soh:.4f} → {final_soh:.4f}")
            print(f"    Verlust: {soh_loss:.4f} ({soh_loss_percent:.2f}%)")
            print(f"    Degradation pro 100 EFC: {(soh_loss_percent / max_efc * 100):.3f}%") if max_efc > 0 else None
    
    plt.show()
    
    input("\nPlot ist interaktiv! Drücke Enter um fortzufahren...")
    
    print(f"\n✅ SOH-Plot über EFC für alle Zellen abgeschlossen!")

def plot_all_soh_comparison():
    """
    Plottet alle SOH-Kurven über EFC in einem einzigen Plot für direkten Vergleich.
    """
    print(f"📊 Starte SOH-Vergleichsplot aller Zellen in einem Fenster")
    
    # Pfad zum FE-Ordner
    base_dir = Path(__file__).parent
    
    # Suche alle verfügbaren FE-DataFrames
    fe_files = list(base_dir.glob("df_FE_*.parquet"))
    if not fe_files:
        print(f"❌ Keine Feature-DataFrames gefunden!")
        print(f"💡 Tipp: Führe zuerst das feature_engineering.py Script aus!")
        return
    
    # Extrahiere Zellen-IDs aus Dateinamen
    cell_ids = [f.stem.replace("df_FE_", "") for f in fe_files]
    cell_ids.sort()  # Sortiere für konsistente Reihenfolge
    
    print(f"✅ Gefundene FE-DataFrames für {len(cell_ids)} Zellen: {cell_ids}")
    
    # Sammle Daten für alle Zellen
    soh_data = {}
    available_cells = []
    
    for cell_id in cell_ids:
        df_fe_path = base_dir / f"df_FE_{cell_id}.parquet"
        
        try:
            # DataFrame laden
            df_fe = pd.read_parquet(df_fe_path)
            
            # Drucke Spaltennamen für die erste Zelle
            if cell_id == cell_ids[0]:
                print(f"\n🔍 Verfügbare Spalten in {cell_id}:")
                for i, col in enumerate(df_fe.columns, 1):
                    print(f"  {i:2d}. {col}")
                print(f"    Gesamt: {len(df_fe.columns)} Spalten")
            
            # Prüfe ob notwendige Spalten vorhanden sind
            if 'SOH' not in df_fe.columns:
                print(f"❌ Zelle {cell_id}: Keine 'SOH' Spalte gefunden!")
                continue
                
            if 'EFC' not in df_fe.columns:
                print(f"❌ Zelle {cell_id}: Keine 'EFC' Spalte gefunden!")
                continue
            
            # Extrahiere Daten
            efc = df_fe['EFC'].values
            soh = df_fe['SOH'].values
            
            # Filtere gültige Werte (keine NaN)
            valid_mask = ~(np.isnan(efc) | np.isnan(soh))
            efc_clean = efc[valid_mask]
            soh_clean = soh[valid_mask]
            
            if len(efc_clean) > 0:
                soh_data[cell_id] = {
                    'efc': efc_clean,
                    'soh': soh_clean
                }
                available_cells.append(cell_id)
                print(f"  ✅ {cell_id}: {len(efc_clean)} Datenpunkte, EFC: {efc_clean.min():.1f} - {efc_clean.max():.1f}, SOH: {soh_clean.min():.4f} - {soh_clean.max():.4f}")
            else:
                print(f"  ❌ {cell_id}: Keine gültigen Daten!")
                
        except Exception as e:
            print(f"❌ Fehler beim Laden von {cell_id}: {e}")
    
    if not available_cells:
        print(f"❌ Keine Zellen mit gültigen SOH-/EFC-Daten gefunden!")
        return
    
    print(f"\n📊 Erstelle SOH-Vergleichsplot für {len(available_cells)} Zellen...")
    
    # Erstelle einen einzigen großen Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    fig.suptitle('SOH-Degradation Vergleich aller Zellen über EFC (Equivalent Full Cycles)', 
                 fontsize=16, fontweight='bold')
    
    # Farben für verschiedene Zellen (mehr Farben für bessere Unterscheidung)
    colors = plt.cm.tab20(np.linspace(0, 1, len(available_cells)))
    
    # Plotte alle Zellen in einem Plot
    legend_labels = []
    for i, cell_id in enumerate(available_cells):
        efc = soh_data[cell_id]['efc']
        soh = soh_data[cell_id]['soh']
        
        # Plotte SOH-Kurve
        ax.plot(efc, soh, color=colors[i], linewidth=2, alpha=0.8, label=cell_id)
        legend_labels.append(cell_id)
        
        # Zeige Start- und Endwerte für jede Zelle
        initial_soh = soh[0] if len(soh) > 0 else 0
        final_soh = soh[-1] if len(soh) > 0 else 0
        print(f"  {cell_id}: SOH {initial_soh:.4f} → {final_soh:.4f} ({((initial_soh-final_soh)/initial_soh*100):.1f}% Verlust)")
    
    # Formatierung
    ax.set_xlabel('EFC (Equivalent Full Cycles)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SOH [-]', fontsize=12, fontweight='bold')
    ax.set_title('Alle Zellen im Vergleich', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)  # SOH von 0 bis 1.05
    
    # Referenzlinien für SOH
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label='100% SOH')
    ax.axhline(y=0.8, color='orange', linestyle=':', alpha=0.6, linewidth=1.5, label='80% SOH')
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label='50% SOH')
    
    # Legende oben rechts (außerhalb des Plots für bessere Lesbarkeit)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
              ncol=1, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Platz für Legende schaffen
    
    # Speichere Plot
    output_path = base_dir / "soh_comparison_all_cells_single_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✅ SOH-Vergleichsplot erstellt!")
    print(f"📊 Plot gespeichert als: {output_path}")
    print(f"📐 Alle {len(available_cells)} Zellen in einem Plot")
    
    # Zeige zusammenfassende Statistiken
    print(f"\n📊 SOH-Degradations-Zusammenfassung:")
    
    all_initial_soh = []
    all_final_soh = []
    all_degradation = []
    
    for cell_id in available_cells:
        efc = soh_data[cell_id]['efc']
        soh = soh_data[cell_id]['soh']
        
        if len(soh) > 1:
            initial_soh = soh[0]
            final_soh = soh[-1]
            soh_loss_percent = (initial_soh - final_soh) / initial_soh * 100
            max_efc = efc.max()
            
            all_initial_soh.append(initial_soh)
            all_final_soh.append(final_soh)
            all_degradation.append(soh_loss_percent)
            
            print(f"  {cell_id}: SOH {initial_soh:.4f} → {final_soh:.4f} ({soh_loss_percent:.2f}% Verlust, {max_efc:.0f} EFC)")
    
    # Gesamtstatistiken
    if all_degradation:
        print(f"\n🔍 Gesamtstatistiken:")
        print(f"  Mittlere Start-SOH: {np.mean(all_initial_soh):.4f} ± {np.std(all_initial_soh):.4f}")
        print(f"  Mittlere End-SOH: {np.mean(all_final_soh):.4f} ± {np.std(all_final_soh):.4f}")
        print(f"  Mittlere Degradation: {np.mean(all_degradation):.2f}% ± {np.std(all_degradation):.2f}%")
        print(f"  Min/Max Degradation: {np.min(all_degradation):.2f}% / {np.max(all_degradation):.2f}%")
    
    plt.show()
    
    input("\nPlot ist interaktiv! Drücke Enter um fortzufahren...")
    
    print(f"\n✅ SOH-Vergleichsplot für alle Zellen abgeschlossen!")

if __name__ == "__main__":
    # Plot für eine einzelne Zelle mit SOC & SOH
    # cell_id = "C01"
    # print(f"🚀 Starte Feature-Plot mit SOC & SOH für Zelle {cell_id}")
    # plot_advanced_features(cell_id)
    
    # SOH-Vergleichsplot aller Zellen in einem Fenster
    print(f"🚀 Starte SOH-Vergleichsplot aller Zellen in einem Fenster")
    plot_all_soh_comparison()
    
    # SOH-Plot über EFC für alle Zellen (einzelne Subplots)
    # print(f"🚀 Starte SOH-Plot über EFC für alle Zellen")
    # plot_all_capacity_over_efc()  # Funktionsname bleibt gleich, aber plottet jetzt SOH
    
    # SOC-Vergleich aller Zellen mit ZHU Daten (auskommentiert)
    # print(f"🚀 Starte SOC-Vergleich aller Zellen mit ZHU Referenzdaten")
    # plot_all_soc_comparison()
    
    # Optional: Vergleich zwischen mehreren Zellen
    # compare_multiple_cells(["C01", "C03", "C05"])

    # SOH-Vergleichsplot für alle Zellen
    print(f"🚀 Starte SOH-Vergleichsplot für alle Zellen")
    plot_all_soh_comparison()
