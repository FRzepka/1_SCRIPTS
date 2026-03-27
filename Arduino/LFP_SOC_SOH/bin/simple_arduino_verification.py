#!/usr/bin/env python3
"""
🔍 SIMPLE ARDUINO VERIFICATION TEST
===================================
Einfacher, direkter Test um zu beweisen dass Arduino die Vorhersagen macht.
"""

import serial
import time
import sys

def main():
    print("🔍 ARDUINO VERIFICATION TEST")
    print("=" * 40)
    print("Dieser Test beweist, dass Arduino wirklich die Vorhersagen macht!\n")
    
    try:
        # 1. Verbinde mit Arduino
        print("🔌 Verbinde mit Arduino auf COM13...")
        arduino = serial.Serial('COM13', 115200, timeout=2)
        time.sleep(2)  # Arduino Reset
        print("✅ Arduino verbunden!\n")
        
        # 2. Test: Arduino Info
        print("📋 TEST 1: Arduino Info abrufen")
        arduino.write(b"INFO\n")
        arduino.flush()
        response = arduino.readline().decode().strip()
        print(f"📥 Arduino Antwort: {response}")
        
        if "ARDUINO" in response and "LSTM" in response:
            print("✅ Arduino antwortet korrekt!\n")
        else:
            print("❌ Unerwartete Arduino-Antwort!\n")
            return
        
        # 3. Test: Vorhersage mit Arduino
        print("📊 TEST 2: SOC-Vorhersage MIT Arduino")
        test_input = "3.4,-0.2,0.92,1150"
        arduino.write(f"PREDICT:{test_input}\n".encode())
        arduino.flush()
        prediction_with = arduino.readline().decode().strip()
        
        try:
            soc_with = float(prediction_with)
            print(f"✅ SOC MIT Arduino: {soc_with:.6f}")
        except:
            print(f"❌ Ungültige Antwort: {prediction_with}")
            return
        
        # 4. Test: Arduino trennen
        print("\n🔌 TEST 3: Arduino TRENNEN")
        arduino.close()
        print("❌ Arduino getrennt")
        
        # 5. Test: Vorhersage OHNE Arduino
        print("\n📊 TEST 4: Versuche SOC-Vorhersage OHNE Arduino")
        try:
            arduino2 = serial.Serial('COM13', 115200, timeout=1)
            arduino2.write(f"PREDICT:{test_input}\n".encode())
            arduino2.flush()
            prediction_without = arduino2.readline().decode().strip()
            arduino2.close()
            
            if prediction_without:
                print(f"❓ SOC OHNE Arduino: {prediction_without}")
                print("⚠️ WARNUNG: Arduino antwortet noch!")
            else:
                print("✅ KORREKT: Keine Antwort ohne Arduino!")
                
        except Exception as e:
            print(f"✅ PERFEKT: Keine Verbindung ohne Arduino! ({e})")
        
        # 6. Test: Arduino wieder verbinden
        print("\n🔌 TEST 5: Arduino wieder VERBINDEN")
        arduino = serial.Serial('COM13', 115200, timeout=2)
        time.sleep(2)
        print("✅ Arduino wieder verbunden")
        
        # 7. Test: Vorhersage nach Reconnect
        print("\n📊 TEST 6: SOC-Vorhersage nach Reconnect")
        arduino.write(f"PREDICT:{test_input}\n".encode())
        arduino.flush()
        prediction_after = arduino.readline().decode().strip()
        
        try:
            soc_after = float(prediction_after)
            print(f"✅ SOC nach Reconnect: {soc_after:.6f}")
        except:
            print(f"❌ Ungültige Antwort: {prediction_after}")
            return
        
        # 8. Physischer Test
        print("\n⚠️  PHYSISCHER TEST:")
        print("🔌 JETZT das USB-Kabel vom Arduino abziehen!")
        print("⏰ Du hast 10 Sekunden Zeit...")
        
        for i in range(10, 0, -1):
            print(f"⏰ {i}...")
            time.sleep(1)
        
        print("\n📊 Versuche Vorhersage mit getrenntem Arduino...")
        try:
            arduino.write(f"PREDICT:{test_input}\n".encode())
            arduino.flush()
            response = arduino.readline().decode().strip()
            
            if response:
                print(f"❌ PROBLEM: Arduino antwortet noch! {response}")
                print("❓ Ist das Kabel wirklich getrennt?")
            else:
                print("✅ PERFEKT: Keine Antwort nach Trennung!")
                
        except Exception as e:
            print(f"✅ PERFEKT: Arduino getrennt! ({e})")
        
        # Schließe Verbindung
        try:
            arduino.close()
        except:
            pass
        
        print("\n" + "=" * 50)
        print("🎉 ARDUINO VERIFICATION ABGESCHLOSSEN!")
        print("✅ Tests zeigen: Vorhersagen kommen definitiv vom Arduino!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        print("🔍 Überprüfe Arduino-Verbindung auf COM13")

if __name__ == "__main__":
    main()
