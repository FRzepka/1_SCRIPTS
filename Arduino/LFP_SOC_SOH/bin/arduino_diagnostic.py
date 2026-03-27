"""
Arduino Diagnostic - Capture all communication
"""

import serial
import time

def diagnostic_test():
    print("🔍 Arduino Communication Diagnostic...")
    
    try:
        # Connect to Arduino
        arduino = serial.Serial('COM13', 115200, timeout=1)
        time.sleep(3)  # Give Arduino time to start
        
        print("✅ Connected to Arduino")
        
        # Capture startup messages
        print("\n📊 Startup Messages:")
        startup_messages = []
        for i in range(20):  # Read up to 20 lines
            if arduino.in_waiting:
                try:
                    line = arduino.readline().decode().strip()
                    if line:
                        startup_messages.append(line)
                        print(f"  {line}")
                except:
                    pass
            else:
                time.sleep(0.1)
        
        # Test simple commands
        print("\n🔧 Testing Commands:")
        
        # Test INFO command
        print("Sending: INFO")
        arduino.write(b'INFO\n')
        time.sleep(1)
        
        print("Response to INFO:")
        for i in range(15):
            if arduino.in_waiting:
                try:
                    line = arduino.readline().decode().strip()
                    if line:
                        print(f"  {line}")
                except:
                    pass
            else:
                break
        
        # Test RESET command
        print("\nSending: RESET")
        arduino.write(b'RESET\n')
        time.sleep(0.5)
        
        print("Response to RESET:")
        for i in range(5):
            if arduino.in_waiting:
                try:
                    line = arduino.readline().decode().strip()
                    if line:
                        print(f"  {line}")
                except:
                    pass
            else:
                break
        
        # Test prediction with detailed output capture
        print("\n🧪 Testing Prediction:")
        test_input = "3.2,-1.0,0.95,5000"
        print(f"Sending: {test_input}")
        
        arduino.write(f"{test_input}\n".encode())
        time.sleep(0.5)
        
        print("All responses:")
        responses = []
        for i in range(10):
            if arduino.in_waiting:
                try:
                    line = arduino.readline().decode().strip()
                    if line:
                        responses.append(line)
                        print(f"  [{i}] '{line}'")
                except:
                    pass
            else:
                time.sleep(0.1)
        
        # Try to identify which is the actual SOC prediction
        print("\n🔍 Analysis:")
        for i, resp in enumerate(responses):
            try:
                float_val = float(resp)
                print(f"  Response {i} is a valid float: {float_val}")
            except:
                print(f"  Response {i} is text: '{resp}'")
        
        arduino.close()
        print("\n✅ Diagnostic completed")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")

if __name__ == "__main__":
    diagnostic_test()
