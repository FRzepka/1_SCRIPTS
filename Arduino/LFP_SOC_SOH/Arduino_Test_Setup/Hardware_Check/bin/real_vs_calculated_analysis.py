"""
REAL vs CALCULATED Arduino Memory Analysis
==========================================

Analysiert die REALEN Arduino-Werte aus dem Live-Monitoring
vs. die berechneten theoretischen Werte

Ziel: Verstehen, warum die Werte abweichen und realistische Vorhersagen treffen
"""

import os
from pathlib import Path

def analyze_real_arduino_data():
    """Analysiere reale Arduino-Daten aus dem Live-Monitoring"""
    
    print("🔍 REAL vs CALCULATED Arduino Memory Analysis")
    print("=" * 60)
    
    # REALE WERTE vom Arduino (32x32)
    real_32x32 = {
        'ram_total_bytes': 98312,      # 96 KB
        'ram_used_bytes': 74520,       # 72.8 KB  
        'ram_free_bytes': 23792,       # 23.2 KB
        'ram_usage_percent': 75.8,
        
        'flash_total_bytes': 32 * 1024,  # 32 KB
        'flash_used_bytes': 24 * 1024,   # 24 KB
        'flash_free_bytes': 8 * 1024,    # 8 KB
        'flash_usage_percent': 75.0
    }
    
    # DATEIGRÖSSEN (tatsächlich)
    weights_32x32_bytes = 85082      # 85 KB
    weights_64x64_bytes = 308064     # 308 KB
    
    print("\n📊 REALE ARDUINO WERTE (32x32 LSTM):")
    print(f"RAM Total:    {real_32x32['ram_total_bytes']:,} bytes ({real_32x32['ram_total_bytes']/1024:.1f} KB)")
    print(f"RAM Used:     {real_32x32['ram_used_bytes']:,} bytes ({real_32x32['ram_used_bytes']/1024:.1f} KB)")
    print(f"RAM Free:     {real_32x32['ram_free_bytes']:,} bytes ({real_32x32['ram_free_bytes']/1024:.1f} KB)")
    print(f"RAM Usage:    {real_32x32['ram_usage_percent']:.1f}%")
    print()
    print(f"Flash Total:  {real_32x32['flash_total_bytes']:,} bytes ({real_32x32['flash_total_bytes']/1024:.1f} KB)")
    print(f"Flash Used:   {real_32x32['flash_used_bytes']:,} bytes ({real_32x32['flash_used_bytes']/1024:.1f} KB)")
    print(f"Flash Free:   {real_32x32['flash_free_bytes']:,} bytes ({real_32x32['flash_free_bytes']/1024:.1f} KB)")
    print(f"Flash Usage:  {real_32x32['flash_usage_percent']:.1f}%")
    
    print("\n📁 REALE DATEIGRÖSSEN:")
    print(f"32x32 Weights: {weights_32x32_bytes:,} bytes ({weights_32x32_bytes/1024:.1f} KB)")
    print(f"64x64 Weights: {weights_64x64_bytes:,} bytes ({weights_64x64_bytes/1024:.1f} KB)")
    print(f"Verhältnis:    {weights_64x64_bytes/weights_32x32_bytes:.1f}x größer")
    
    # ANALYSE: Wo sind die Gewichte?
    print("\n🔍 ANALYSE - WO SIND DIE GEWICHTE?")
    print("=" * 40)
    
    # Theorie 1: Gewichte im Flash, nur teilweise in RAM
    flash_weights_size = weights_32x32_bytes  # 85 KB
    print(f"Theory 1 - Gewichte im Flash:")
    print(f"  Weights File: {flash_weights_size/1024:.1f} KB")
    print(f"  Aber Flash zeigt nur 24 KB used?")
    print(f"  🤔 MISMATCH: {flash_weights_size/1024:.1f} KB vs 24 KB")
    
    # Theorie 2: Gewichte werden komprimiert oder teilweise geladen
    actual_ram_usage = real_32x32['ram_used_bytes']
    print(f"\nTheory 2 - Gewichte teilweise im RAM:")
    print(f"  RAM Used: {actual_ram_usage/1024:.1f} KB")
    print(f"  Weights: {weights_32x32_bytes/1024:.1f} KB")
    print(f"  Wenn alle Weights im RAM: {(actual_ram_usage + weights_32x32_bytes)/1024:.1f} KB")
    print(f"  Das wäre: {(actual_ram_usage + weights_32x32_bytes)/real_32x32['ram_total_bytes']*100:.1f}% RAM usage")
    
    # ARDUINO-TYP ERKENNUNG
    print(f"\n🔧 ARDUINO-TYP ERKENNUNG:")
    print("=" * 30)
    
    total_ram = real_32x32['ram_total_bytes']
    if total_ram > 90000:  # > 90 KB
        print(f"✅ Arduino Mega/Due/ESP32 erkannt ({total_ram/1024:.1f} KB RAM)")
        print("  🎯 Das erklärt, warum 32x32 funktioniert!")
    elif total_ram <= 2048:  # 2 KB
        print(f"❌ Arduino Uno/Nano erkannt ({total_ram/1024:.1f} KB RAM)")
        print("  🚨 Das sollte NICHT funktionieren!")
    else:
        print(f"🤔 Unbekannter Arduino-Typ ({total_ram/1024:.1f} KB RAM)")
    
    # VORHERSAGE FÜR 64x64
    print(f"\n🔮 VORHERSAGE FÜR 64x64 LSTM:")
    print("=" * 35)
    
    # Gewichte-Verhältnis
    weight_ratio = weights_64x64_bytes / weights_32x32_bytes
    print(f"Gewichte-Verhältnis: {weight_ratio:.1f}x")
    
    # Hochrechnung RAM
    estimated_64x64_ram = actual_ram_usage * weight_ratio
    print(f"Geschätzte RAM-Nutzung 64x64: {estimated_64x64_ram/1024:.1f} KB")
    print(f"Verfügbarer RAM: {real_32x32['ram_total_bytes']/1024:.1f} KB")
    print(f"Vorhergesagte RAM-Auslastung: {estimated_64x64_ram/real_32x32['ram_total_bytes']*100:.1f}%")
    
    if estimated_64x64_ram > real_32x32['ram_total_bytes']:
        print("❌ PREDICTION: 64x64 wird NICHT funktionieren!")
        print(f"   Benötigt: {estimated_64x64_ram/1024:.1f} KB, Verfügbar: {real_32x32['ram_total_bytes']/1024:.1f} KB")
        print(f"   Über-Limit: {(estimated_64x64_ram - real_32x32['ram_total_bytes'])/1024:.1f} KB")
    else:
        print("✅ PREDICTION: 64x64 könnte funktionieren!")
        remaining = real_32x32['ram_total_bytes'] - estimated_64x64_ram
        print(f"   Verbleibt: {remaining/1024:.1f} KB RAM")
    
    # FLASH ANALYSE
    print(f"\n💾 FLASH MEMORY ANALYSE:")
    print("=" * 25)
    
    # Flash-Diskrepanz verstehen
    reported_flash_used = real_32x32['flash_used_bytes']
    weights_file_size = weights_32x32_bytes
    
    print(f"Arduino berichtet: {reported_flash_used/1024:.1f} KB Flash used")
    print(f"Weights-Datei:     {weights_file_size/1024:.1f} KB")
    print(f"Diskrepanz:        {(weights_file_size - reported_flash_used)/1024:.1f} KB")
    
    if weights_file_size > reported_flash_used:
        print("🤔 THEORIE: Arduino zeigt nur Code-Größe, nicht Daten-Größe")
        print("   Die Gewichte sind zusätzlich im Flash gespeichert")
        total_flash_with_weights = reported_flash_used + weights_file_size
        print(f"   Echter Flash-Verbrauch: {total_flash_with_weights/1024:.1f} KB")
        
        # 64x64 Flash-Vorhersage
        predicted_64x64_flash = reported_flash_used + weights_64x64_bytes
        print(f"   64x64 Flash-Bedarf: {predicted_64x64_flash/1024:.1f} KB")
        
        if predicted_64x64_flash > real_32x32['flash_total_bytes']:
            print("❌ 64x64 FLASH PROBLEM!")
            print(f"   Benötigt: {predicted_64x64_flash/1024:.1f} KB")
            print(f"   Verfügbar: {real_32x32['flash_total_bytes']/1024:.1f} KB")
        else:
            print("✅ 64x64 Flash sollte passen")
    
    # ZUSAMMENFASSUNG
    print(f"\n📋 ZUSAMMENFASSUNG:")
    print("=" * 20)
    print("✅ 32x32 LSTM funktioniert auf diesem Arduino")
    print(f"✅ Arduino hat {total_ram/1024:.1f} KB RAM (genug für größere Modelle)")
    
    if estimated_64x64_ram > real_32x32['ram_total_bytes']:
        print("❌ 64x64 LSTM wird wahrscheinlich RAM-Probleme haben")
        print("💡 Lösungen: ESP32, Gewichts-Komprimierung, oder Streaming")
    else:
        print("✅ 64x64 LSTM könnte auf diesem Arduino funktionieren")
        print("🔬 Test empfohlen!")
    
    return {
        'arduino_type': 'mega_or_larger' if total_ram > 90000 else 'uno_nano',
        'ram_sufficient_64x64': estimated_64x64_ram <= real_32x32['ram_total_bytes'],
        'estimated_64x64_ram_kb': estimated_64x64_ram / 1024,
        'weight_ratio': weight_ratio
    }

def main():
    """Hauptanalyse-Funktion"""
    results = analyze_real_arduino_data()
    
    print(f"\n🎯 SCHNELLE ANTWORT:")
    print("=" * 20)
    print(f"Arduino-Typ: {results['arduino_type']}")
    print(f"64x64 RAM OK: {results['ram_sufficient_64x64']}")
    print(f"64x64 RAM-Bedarf: {results['estimated_64x64_ram_kb']:.1f} KB")
    print(f"Gewichte-Faktor: {results['weight_ratio']:.1f}x")

if __name__ == "__main__":
    main()
