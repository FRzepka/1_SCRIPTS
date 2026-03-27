"""
Quick tool to release COM port
"""
import serial
import time

PORT = "COM7"
BAUD = 115200

print(f"Attempting to open and close {PORT} to release it...")

try:
    # Open port
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"  Opened {PORT}")
    time.sleep(0.5)
    
    # Close immediately
    ser.close()
    print(f"  Closed {PORT}")
    print("\n✓ Port should be released now!")
    
except serial.SerialException as e:
    print(f"  ERROR: {e}")
    print("\nTry these solutions:")
    print("1. Unplug STM32 USB cable")
    print("2. Wait 3 seconds")
    print("3. Plug it back in")
    print("4. Check Device Manager if COM7 still exists")
