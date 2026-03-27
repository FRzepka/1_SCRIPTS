import serial
import json
import time

print("🎯 Testing Arduino LSTM SOC Prediction (32 Hidden Units)")
print("=" * 55)

try:
    ser = serial.Serial('COM13', 115200, timeout=2)
    time.sleep(2)  # Warten bis Arduino bereit ist
    print("✅ Connected to Arduino on COM13")

    # Test-Datenpaket senden
    test_data = {
        'voltage': 3.8,
        'current': -2.5,
        'soh': 0.95,
        'q_c': 1.0,
        'reset': True
    }

    print(f"📤 Sending test data: {test_data}")
    ser.write((json.dumps(test_data) + '\n').encode())
    time.sleep(0.5)

    # Response lesen
    if ser.in_waiting > 0:
        response = ser.readline().decode().strip()
        print(f"📥 Arduino response: {response}")
        
        try:
            data = json.loads(response)
            print(f"🔋 SOC Prediction: {data.get('soc', 'N/A')}")
            print(f"🧠 Hidden Size: {data.get('hidden_size', 'N/A')}")
            print(f"📊 Status: {data.get('status', 'N/A')}")
            
            if 'soc' in data:
                soc_value = data['soc']
                if 0.0 <= soc_value <= 1.0:
                    print(f"✅ SOC prediction in valid range: {soc_value:.4f}")
                else:
                    print(f"⚠️  SOC prediction out of range: {soc_value:.4f}")
        except json.JSONDecodeError:
            print(f"📝 Raw response: {response}")
    else:
        print("❌ No response from Arduino")

    ser.close()
    print("✅ Test completed")

except Exception as e:
    print(f"❌ Error: {e}")
