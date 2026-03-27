"""
Arduino MAE Test - Ultra Simple Version
No matplotlib, just basic MAE calculation
"""

import serial
import time

def main():
    print("🧪 Arduino MAE Test - Ultra Simple")
    
    try:
        print("🔌 Connecting...")
        arduino = serial.Serial('COM13', 115200, timeout=3)
        time.sleep(2)
        print("✅ Connected")
        
        # Simple test
        arduino.flushInput()
        arduino.write(b'3.2,-1.0,0.95,5000\\n')
        time.sleep(0.2)
        
        if arduino.in_waiting:
            response = arduino.readline().decode().strip()
            print(f"📊 Prediction: {response}")
        else:
            print("❌ No response")
        
        arduino.close()
        print("✅ Done")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
