"""
📊 SOC DATA PREVIEW - Quick Check 📊
====================================

Schnelles Preview der SOC Daten für gewählten Zeitbereich
- Zeigt SOC Verlauf über Zeit
- Voltage, Current, SOH für Kontext  
- Schnelle Einschätzung ob sich der Zeitbereich lohnt

🚀 QUICK & SIMPLE! 🚀
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ===== EINSTELLUNGEN =====
DATA_PATH = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"

# Test Einstellungen - DEINE GEWÜNSCHTEN WERTE
START_MINUTE = 0     # Starte ab 300 Minuten
TEST_DURATION_MINS = 200   # 30 Minuten Test
PREDICTION_DELAY = 500    # 500ms zwischen Predictions (nur für Info)

def preview_soc_data():
    """
    🔍 Preview der SOC Daten für gewählten Zeitbereich
    """
    print("📊" + "="*60 + "📊")
    print("📊 SOC DATA PREVIEW - QUICK CHECK 📊")
    print("📊" + "="*60 + "📊")
    print(f"⏰ Zeitbereich: ab {START_MINUTE} Minuten für {TEST_DURATION_MINS} Minuten")
    print(f"🔍 Checking Data: {DATA_PATH}")
    print()
    
    # 1. Lade Daten
    if not Path(DATA_PATH).exists():
        print(f"❌ Daten nicht gefunden: {DATA_PATH}")
        return
    
    print("📥 Lade C19 Daten...")
    df = pd.read_parquet(DATA_PATH)
    
    # Prüfe Spalten
    required_cols = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c", "SOC_ZHU"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Fehlende Spalten: {missing_cols}")
        return
    
    print(f"✅ Daten geladen: {len(df)} total Samples")
    
    # 2. Berechne Zeitbereich
    # Annahme: 1 Sample pro Sekunde (typisch für MGFarm Daten)
    start_sample = START_MINUTE * 60  # Minuten * 60 Sekunden
    end_sample = start_sample + (TEST_DURATION_MINS * 60)  # + Testdauer
    
    print(f"📍 Gewählter Bereich: Sample {start_sample} bis {end_sample}")
    
    # Prüfe Verfügbarkeit
    if start_sample >= len(df):
        print(f"❌ Start-Zeit zu spät! Nur {len(df)} Samples verfügbar")
        print(f"💡 Max Start-Zeit: {len(df) // 60:.1f} Minuten")
        return
    
    if end_sample > len(df):
        print(f"⚠️ End-Zeit zu spät! Verwende verfügbare Daten bis Ende")
        end_sample = len(df)
        actual_duration = (end_sample - start_sample) / 60
        print(f"📏 Tatsächliche Testdauer: {actual_duration:.1f} Minuten")
    
    # 3. Extrahiere Datenbereich
    df_segment = df.iloc[start_sample:end_sample].copy()
    
    if len(df_segment) < 60:  # Mindestens 1 Minute Daten
        print(f"❌ Zu wenige Daten im Zeitbereich: {len(df_segment)} Samples")
        return
    
    # 4. Zeitachse erstellen (in Minuten relativ zum Start)
    time_minutes = np.arange(len(df_segment)) / 60.0  # Sekunden zu Minuten
    
    # 5. Datenstatistiken
    soc_values = df_segment["SOC_ZHU"]
    voltage_values = df_segment["Voltage[V]"]
    current_values = df_segment["Current[A]"]
    
    print(f"\n📊 DATENSTATISTIKEN:")
    print(f"   Samples: {len(df_segment)}")
    print(f"   Dauer: {time_minutes[-1]:.1f} Minuten")
    print(f"   SOC Range: {soc_values.min():.3f} - {soc_values.max():.3f}")
    print(f"   SOC Variation: {soc_values.std():.4f}")
    print(f"   Voltage Range: {voltage_values.min():.3f} - {voltage_values.max():.3f} V")
    print(f"   Current Range: {current_values.min():.3f} - {current_values.max():.3f} A")
    
    # 6. Bewertung der Datenqualität
    soc_variation = soc_values.std()
    soc_range = soc_values.max() - soc_values.min()
    
    print(f"\n🎯 BEWERTUNG:")
    if soc_range > 0.1:
        print("✅ Gute SOC-Variation - interessant für Test!")
    elif soc_range > 0.05:
        print("👍 Moderate SOC-Variation - OK für Test")
    else:
        print("⚠️ Geringe SOC-Variation - möglicherweise langweilig")
    
    if soc_variation > 0.02:
        print("✅ Dynamische SOC-Änderungen")
    else:
        print("📈 Relativ stabile SOC-Werte")
    
    # 7. Geschätzte Testzeit bei 500ms
    samples_for_test = len(df_segment)
    estimated_test_time = (samples_for_test * PREDICTION_DELAY) / 1000 / 60  # Minuten
    print(f"⏱️ Geschätzte Arduino Testzeit: {estimated_test_time:.1f} Minuten")
    
    # 8. Plot erstellen
    print(f"\n📈 Erstelle Preview Plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'📊 SOC Data Preview: {START_MINUTE}-{START_MINUTE+TEST_DURATION_MINS} Min (C19)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: SOC über Zeit (HAUPTPLOT!)
    axes[0,0].plot(time_minutes, soc_values, 'b-', linewidth=2, alpha=0.8)
    axes[0,0].set_title('🔋 SOC Verlauf über Zeit', fontweight='bold', fontsize=14)
    axes[0,0].set_xlabel('Zeit [Min]')
    axes[0,0].set_ylabel('State of Charge')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 1)
    
    # Plot 2: Voltage über Zeit
    axes[0,1].plot(time_minutes, voltage_values, 'g-', linewidth=2, alpha=0.8)
    axes[0,1].set_title('⚡ Voltage Verlauf', fontweight='bold', fontsize=14)
    axes[0,1].set_xlabel('Zeit [Min]')
    axes[0,1].set_ylabel('Voltage [V]')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Current über Zeit
    axes[1,0].plot(time_minutes, current_values, 'r-', linewidth=2, alpha=0.8)
    axes[1,0].set_title('⚡ Current Verlauf', fontweight='bold', fontsize=14)
    axes[1,0].set_xlabel('Zeit [Min]')
    axes[1,0].set_ylabel('Current [A]')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: SOC Histogram
    axes[1,1].hist(soc_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1,1].set_title('📊 SOC Verteilung', fontweight='bold', fontsize=14)
    axes[1,1].set_xlabel('State of Charge')
    axes[1,1].set_ylabel('Häufigkeit')
    axes[1,1].grid(True, alpha=0.3)
    
    # Zusätzliche Infos auf Plot
    info_text = f"""
📊 DATEN INFO:
Start: {START_MINUTE} min
Dauer: {TEST_DURATION_MINS} min
Samples: {len(df_segment)}
SOC Range: {soc_range:.3f}
Testzeit: ~{estimated_test_time:.1f} min
    """
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # 9. Speichere Plot
    timestamp = f"{START_MINUTE}min_{TEST_DURATION_MINS}min"
    save_path = Path(f"c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/Arduino_Test_Setup/Stateful_32_32_comparison/soc_preview_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Preview Plot gespeichert: {save_path}")
    
    plt.show()
    
    # 10. Empfehlung
    print(f"\n🎯 EMPFEHLUNG:")
    if soc_range > 0.1 and soc_variation > 0.02:
        print("🔥 PERFEKT! Dieser Zeitbereich ist ideal für Arduino vs PC Vergleich!")
        print("✅ Gute SOC-Dynamik, interessante Verläufe")
    elif soc_range > 0.05:
        print("👍 GUT! Dieser Zeitbereich ist OK für den Vergleich")
        print("✅ Moderate SOC-Änderungen vorhanden")
    else:
        print("⚠️ LANGWEILIG! Wenig SOC-Variation - vielleicht anderen Zeitbereich wählen?")
        print("💡 Versuche einen anderen START_MINUTE Wert")
    
    print(f"\n🚀 Wenn zufrieden: Verwende diese Settings im Vergleichsscript!")

def main():
    """Main function"""
    try:
        preview_soc_data()
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
