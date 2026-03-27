"""
Debug script to identify why MAE test scripts hang
"""

print("🔍 Starting MAE test debug...")

try:
    print("📦 Testing imports...")
    import serial
    print("  ✅ serial")
    
    import time
    print("  ✅ time")
    
    import numpy as np
    print("  ✅ numpy")
    
    import matplotlib.pyplot as plt
    print("  ✅ matplotlib")
    
    from datetime import datetime
    print("  ✅ datetime")
    
    print("📦 All imports successful!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

try:
    print("🔧 Testing basic Arduino connection...")
    
    arduino = serial.Serial('COM13', 115200, timeout=3)
    time.sleep(2)
    print("  ✅ Arduino connected")
    
    # Clear and test
    arduino.flushInput()
    arduino.write(b'INFO\n')
    time.sleep(1)
    
    if arduino.in_waiting:
        response = arduino.readline().decode().strip()
        print(f"  📋 Arduino response: {response}")
    else:
        print("  ⚠️ No Arduino response")
    
    arduino.close()
    print("  🔌 Connection closed")
    
except Exception as e:
    print(f"❌ Arduino test error: {e}")

print("🔍 Now testing the ArduinoMAETestSimple class creation...")

try:
    from arduino_mae_test_simple import ArduinoMAETestSimple
    print("  ✅ Class import successful")
    
    print("🏗️ Creating test instance...")
    tester = ArduinoMAETestSimple(port='COM13')
    print("  ✅ Instance created")
    
    print("📊 Testing data creation...")
    success = tester.create_test_data()
    print(f"  📊 Data creation: {success}")
    print(f"  📊 Test data count: {len(tester.test_data)}")
    
except Exception as e:
    print(f"❌ Class test error: {e}")
    import traceback
    traceback.print_exc()

print("🔍 Debug completed!")
