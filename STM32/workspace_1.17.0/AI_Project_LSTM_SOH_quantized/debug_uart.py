"""
Debug UART Communication - See what STM32 actually sends
"""
import serial
import time

SERIAL_PORT = "COM7"
BAUD_RATE = 115200
TIMEOUT = 1.0

print("="*80)
print("STM32 UART Debug - Raw Communication")
print("="*80)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print(f"✓ Connected to {SERIAL_PORT}")
    print("\nWaiting 2 seconds for STM32 reset...")
    time.sleep(2)
    
    # Read startup messages
    print("\n--- STARTUP MESSAGES ---")
    for i in range(20):
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"  [{i}] {repr(line)}")
        time.sleep(0.1)
    
    # Clear buffer
    while ser.in_waiting > 0:
        ser.readline()
    
    print("\n--- SENDING TEST SAMPLES ---")
    # Send 5 test samples
    test_samples = [
        "3.300000 0.500000 25.000000 1000.000000 0.000100 0.000050",
        "3.350000 0.600000 26.000000 1100.000000 0.000110 0.000060",
        "3.400000 0.700000 27.000000 1200.000000 0.000120 0.000070",
        "3.450000 0.800000 28.000000 1300.000000 0.000130 0.000080",
        "3.500000 0.900000 29.000000 1400.000000 0.000140 0.000090",
    ]
    
    for i, sample in enumerate(test_samples):
        print(f"\n[Sample {i+1}]")
        print(f"  Sending: {sample}")
        
        ser.write((sample + "\n").encode('utf-8'))
        ser.flush()
        
        # Wait and collect ALL responses
        time.sleep(0.2)
        
        responses = []
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            responses.append(line)
        
        if responses:
            print(f"  Received ({len(responses)} lines):")
            for j, resp in enumerate(responses):
                print(f"    [{j}] {repr(resp)}")
        else:
            print(f"  Received: (nothing)")
    
    print("\n--- WAITING FOR LATE RESPONSES (2 seconds) ---")
    time.sleep(2)
    while ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"  Late: {repr(line)}")
    
    ser.close()
    print("\n✓ Port closed")
    print("="*80)

except serial.SerialException as e:
    print(f"\n❌ ERROR: {e}")
    print("\nCheck:")
    print("  1. STM32 connected and powered")
    print("  2. Correct COM port")
    print("  3. Firmware flashed correctly")
    exit(1)
