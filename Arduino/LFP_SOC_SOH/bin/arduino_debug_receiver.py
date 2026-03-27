"""
Arduino Debug Receiver - Mit detailliertem Debug Output
"""

import serial
import time

def test_arduino_connection():
    """Teste Arduino Verbindung mit Debug Output"""
    print("🎯 ARDUINO DEBUG RECEIVER")
    print("=" * 40)
    
    port = 'COM13'
    baudrate = 115200
    
    print(f"🔌 Versuche Verbindung zu {port} mit {baudrate} baud...")
      try:
        arduino = serial.Serial(port, baudrate, timeout=2)
        print("✅ Serial Port geöffnet!")
        
        print("⏳ Warte 3 Sekunden auf Arduino Reset...")
        time.sleep(3)
        
        # Clear any startup messages
        arduino.flushInput()
        time.sleep(0.5)
        
        print("🔧 Teste Arduino Kommunikation...")
        
        # Test 1: INFO command
        print("\n1️⃣ Sende INFO Kommando...")
        arduino.write(b'INFO\n')
        time.sleep(1)
        
        info_count = 0
        while arduino.in_waiting > 0:
            try:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    info_count += 1
                    print(f"   📋 INFO {info_count}: '{line}'")
            except Exception as e:
                print(f"   ❌ Fehler: {e}")
                break
        
        # Test 2: RESET command
        print("\n2️⃣ Sende RESET Kommando...")
        arduino.flushInput()
        arduino.write(b'RESET\n')
        time.sleep(0.5)
        
        response = arduino.readline().decode('utf-8', errors='ignore').strip()
        print(f"   🔄 Reset Response: '{response}'")
        
        # Test 3: Send test data and get predictions
        print("\n3️⃣ Teste Daten-Vorhersagen...")
        test_cases = [
            (3.2, -1.0, 0.95, 5000),  # Low voltage, discharging
            (3.6, 0.0, 0.95, 5000),   # Mid voltage, no current  
            (4.1, 1.0, 0.95, 5000),   # High voltage, charging
        ]
        
        prediction_count = 0
        for i, (voltage, current, soh, q_c) in enumerate(test_cases, 1):
            try:
                arduino.flushInput()
                data_str = f"{voltage},{current},{soh},{q_c}\n"
                print(f"   📤 Sende {i}: {data_str.strip()}")
                
                arduino.write(data_str.encode())
                time.sleep(0.5)
                
                if arduino.in_waiting > 0:
                    response = arduino.readline().decode('utf-8', errors='ignore').strip()
                    try:
                        soc_pred = float(response)
                        prediction_count += 1
                        print(f"   📥 Antwort {i}: {soc_pred:.6f}")
                    except ValueError:
                        print(f"   ❓ Unerwartete Antwort {i}: '{response}'")
                else:
                    print(f"   ❌ Keine Antwort für {i}")
                    
            except Exception as e:
                print(f"   ❌ Fehler bei Test {i}: {e}")
        
        # Test 4: Check if Arduino sends data autonomously
        print("\n4️⃣ Teste autonome Daten (5 Sekunden Lauschen)...")
        start_time = time.time()
        autonomous_count = 0
        
        while time.time() - start_time < 5:  # 5 Sekunden
            if arduino.in_waiting > 0:
                try:
                    line = arduino.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        autonomous_count += 1
                        print(f"   📨 Autonom {autonomous_count}: '{line}'")
                except Exception as e:
                    print(f"   ❌ Fehler: {e}")            else:
                time.sleep(0.1)
        
        print(f"\n📊 ZUSAMMENFASSUNG:")
        print(f"   📋 INFO Zeilen: {info_count}")
        print(f"   🔄 RESET funktioniert: {'✅ Ja' if response else '❌ Nein'}")
        print(f"   🎯 Erfolgreiche Vorhersagen: {prediction_count}/3")
        print(f"   📨 Autonome Nachrichten: {autonomous_count}")
        print(f"   🤖 Arduino Modus: {'Reaktiv (sendet auf Anfrage)' if prediction_count > 0 and autonomous_count == 0 else 'Autonom' if autonomous_count > 0 else 'Keine Kommunikation'}")
        
        arduino.close()
        print("\n🔌 Verbindung geschlossen")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    test_arduino_connection()
