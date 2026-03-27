"""
Simple Arduino Test - Check what model is actually loaded
"""

import serial
import time

def test_arduino():
    print("🔬 Testing Arduino LSTM Model...")
    
    try:
        # Connect to Arduino
        arduino = serial.Serial('COM13', 115200, timeout=3)
        time.sleep(2)
        
        print("✅ Connected to Arduino")
          # Clear any startup messages
        time.sleep(1)
        arduino.flushInput()
        
        # Send INFO command
        arduino.write(b'INFO\n')
        time.sleep(1)
        
        # Read all available lines from INFO command
        print("\n📊 Arduino INFO:")
        info_lines = []
        while arduino.in_waiting:
            try:
                line = arduino.readline().decode().strip()
                if line:
                    info_lines.append(line)
                    print(f"  {line}")
            except:
                break
        
        # Send RESET command
        print("\n🔄 Sending RESET...")
        arduino.flushInput()
        arduino.write(b'RESET\n')
        time.sleep(0.5)
        
        response = arduino.readline().decode().strip()
        print(f"Reset response: {response}")
        
        # Clear buffer again
        arduino.flushInput()
        
        # Test a few predictions with known values
        test_cases = [
            (3.2, -1.0, 0.95, 5000),  # Low voltage, discharging
            (3.6, 0.0, 0.95, 5000),   # Mid voltage, no current  
            (4.1, 1.0, 0.95, 5000),   # High voltage, charging
        ]
          print("\n🧪 Test Predictions:")
        print("Input (V, I, SOH, Q_c) → Arduino SOC")
        print("-" * 40)
        
        for v, i, soh, q_c in test_cases:
            try:
                # Clear input buffer before sending
                arduino.flushInput()
                
                # Send data
                data_str = f"{v},{i},{soh},{q_c}\n"
                arduino.write(data_str.encode())
                time.sleep(0.2)  # Give Arduino time to process
                
                # Read response - should be a single float
                response = arduino.readline().decode().strip()
                
                # Try to parse as float
                try:
                    soc = float(response)
                    print(f"({v:4.1f}, {i:4.1f}, {soh:4.2f}, {q_c:4.0f}) → {soc:.6f}")
                except ValueError:
                    print(f"({v:4.1f}, {i:4.1f}, {soh:4.2f}, {q_c:4.0f}) → UNEXPECTED: '{response}'")
                
            except Exception as e:
                print(f"({v:4.1f}, {i:4.1f}, {soh:4.2f}, {q_c:4.0f}) → ERROR: {e}")
        
        arduino.close()
        print("\n✅ Test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_arduino()
