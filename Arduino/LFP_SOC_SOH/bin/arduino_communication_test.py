"""
Arduino Communication Test - Detect and test both protocols
"""

import serial
import time
import re

def test_arduino_communication():
    print("🔍 Arduino Communication Protocol Detection...")
    
    try:
        # Connect to Arduino
        arduino = serial.Serial('COM13', 115200, timeout=2)
        time.sleep(3)  # Give Arduino time to start
        
        print("✅ Connected to Arduino")
        
        # Clear any buffered data
        arduino.flushInput()
        time.sleep(0.5)
        
        # Capture startup messages to identify the program
        print("\n📊 Startup Messages:")
        startup_msgs = []
        for i in range(10):
            if arduino.in_waiting:
                try:
                    line = arduino.readline().decode().strip()
                    if line:
                        startup_msgs.append(line)
                        print(f"  {line}")
                except:
                    pass
            else:
                time.sleep(0.1)
        
        # Detect protocol based on startup messages
        protocol = "unknown"
        if any("VOLLSTÄNDIG" in msg for msg in startup_msgs):
            protocol = "full32"
            print("\n🎯 Detected: arduino_lstm_soc_full32.ino (INFO/RESET protocol)")
        elif any("LIVE EXACT" in msg for msg in startup_msgs):
            protocol = "live_exact"
            print("\n🎯 Detected: arduino_lstm_soc_live_exact.ino (CSV protocol)")
        else:
            print("\n❓ Unknown Arduino program")
        
        # Test predictions based on detected protocol
        print("\n🧪 Testing Predictions:")
        
        if protocol == "full32":
            test_full32_protocol(arduino)
        elif protocol == "live_exact":
            test_live_exact_protocol(arduino)
        else:
            # Try both protocols
            print("Trying full32 protocol...")
            test_full32_protocol(arduino)
            
        arduino.close()
        print("\n✅ Communication test completed")
        return protocol
        
    except Exception as e:
        print(f"❌ Communication test failed: {e}")
        return None

def test_full32_protocol(arduino):
    """Test the full32 protocol (INFO/RESET commands)"""
    try:
        # Send RESET first
        arduino.flushInput()
        arduino.write(b'RESET\n')
        time.sleep(0.5)
        
        # Read reset response
        response = arduino.readline().decode().strip()
        print(f"RESET response: {response}")
        
        # Test predictions
        test_cases = [
            (3.2, -1.0, 0.95, 5000),
            (3.6, 0.0, 0.95, 5000),
            (4.1, 1.0, 0.95, 5000),
        ]
        
        for v, i, soh, q_c in test_cases:
            arduino.flushInput()
            data_str = f"{v},{i},{soh},{q_c}\n"
            arduino.write(data_str.encode())
            time.sleep(0.3)
            
            # Read all responses and find the SOC value
            responses = []
            for _ in range(10):
                if arduino.in_waiting:
                    try:
                        line = arduino.readline().decode().strip()
                        if line:
                            responses.append(line)
                    except:
                        break
                else:
                    break
            
            # Try to find SOC value (should be a float)
            soc_found = None
            for resp in responses:
                try:
                    soc_val = float(resp)
                    if 0.0 <= soc_val <= 1.0:  # Reasonable SOC range
                        soc_found = soc_val
                        break
                except:
                    continue
            
            if soc_found is not None:
                print(f"({v:4.1f}, {i:4.1f}, {soh:4.2f}, {q_c:4.0f}) → {soc_found:.6f} ✅")
            else:
                print(f"({v:4.1f}, {i:4.1f}, {soh:4.2f}, {q_c:4.0f}) → No valid SOC found")
                for resp in responses[:3]:  # Show first 3 responses
                    print(f"    Response: '{resp}'")
                    
    except Exception as e:
        print(f"Full32 protocol test failed: {e}")

def test_live_exact_protocol(arduino):
    """Test the live_exact protocol (CSV in/out)"""
    try:
        test_cases = [
            (3.2, -1.0, 0.95, 5000),
            (3.6, 0.0, 0.95, 5000),
            (4.1, 1.0, 0.95, 5000),
        ]
        
        for v, i, soh, q_c in test_cases:
            arduino.flushInput()
            data_str = f"{v},{i},{soh},{q_c}\n"
            arduino.write(data_str.encode())
            time.sleep(0.2)
            
            # Read response
            response = arduino.readline().decode().strip()
            
            # Parse SOC from response like "SOC: 0.996614 | Time: 1234 µs | Count: 1"
            soc_match = re.search(r'SOC:\s*([\d.]+)', response)
            if soc_match:
                soc_val = float(soc_match.group(1))
                print(f"({v:4.1f}, {i:4.1f}, {soh:4.2f}, {q_c:4.0f}) → {soc_val:.6f} ✅")
            else:
                print(f"({v:4.1f}, {i:4.1f}, {soh:4.2f}, {q_c:4.0f}) → Response: '{response}'")
                
    except Exception as e:
        print(f"Live_exact protocol test failed: {e}")

if __name__ == "__main__":
    detected_protocol = test_arduino_communication()
