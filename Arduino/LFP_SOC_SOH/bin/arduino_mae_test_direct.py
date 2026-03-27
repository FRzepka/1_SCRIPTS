"""
Direct copy of working simple test with just MAE extension
"""

import serial
import time

def test_arduino():
    print("🔬 Testing Arduino LSTM Model with MAE...")
    
    try:
        # Connect to Arduino
        arduino = serial.Serial('COM13', 115200, timeout=3)
        time.sleep(2)
        
        print("✅ Connected to Arduino")
        
        # Clear any startup messages
        time.sleep(1)
        arduino.flushInput()
        
        # Send INFO command
        arduino.write(b'INFO\\n')
        time.sleep(1)
        
        # Read all available lines from INFO command
        print("\\n📊 Arduino INFO:")
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
        print("\\n🔄 Sending RESET...")
        arduino.flushInput()
        arduino.write(b'RESET\\n')
        time.sleep(0.5)
        
        response = arduino.readline().decode().strip()
        print(f"Reset response: {response}")
        
        # Clear buffer again
        arduino.flushInput()
        
        # Extended test cases for MAE
        test_cases = [
            (3.0, -2.0, 0.95, 5000),  # Very low SOC
            (3.2, -1.0, 0.95, 5000),  # Low voltage, discharging
            (3.4, 0.0, 0.95, 5000),   # Mid voltage, no current  
            (3.6, 1.0, 0.95, 5000),   # High voltage, charging
            (3.8, 2.0, 0.95, 5000),   # Very high SOC
        ]
        
        # Estimated ground truth (rough estimates based on LiFePO4 characteristics)
        expected_soc = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        print("\\n🧪 MAE Test Predictions:")
        print("Input (V, I, SOH, Q_c) → Arduino SOC | Expected | Error")
        print("-" * 70)
        
        predictions = []
        errors = []
        
        for i, ((v, curr, soh, q_c), exp_soc) in enumerate(zip(test_cases, expected_soc)):
            try:
                # Clear input buffer before sending
                arduino.flushInput()
                
                # Send data
                data_str = f"{v},{curr},{soh},{q_c}\\n"
                arduino.write(data_str.encode())
                time.sleep(0.2)  # Give Arduino time to process
                
                # Read response - should be a single float
                response = arduino.readline().decode().strip()
                
                # Try to parse as float
                try:
                    soc = float(response)
                    error = abs(soc - exp_soc)
                    predictions.append(soc)
                    errors.append(error)
                    print(f"({v:4.1f}, {curr:4.1f}, {soh:4.2f}, {q_c:4.0f}) → {soc:.6f} | {exp_soc:.3f} | {error:.6f}")
                except ValueError:
                    print(f"({v:4.1f}, {curr:4.1f}, {soh:4.2f}, {q_c:4.0f}) → UNEXPECTED: '{response}'")
                
            except Exception as e:
                print(f"({v:4.1f}, {curr:4.1f}, {soh:4.2f}, {q_c:4.0f}) → ERROR: {e}")
        
        # Calculate MAE (simple calculation without numpy)
        if errors:
            mae = sum(errors) / len(errors)
            avg_pred = sum(predictions) / len(predictions)
            print(f"\\n📊 MAE Results:")
            print(f"  Tests: {len(predictions)}/{len(test_cases)}")
            print(f"  MAE: {mae:.6f}")
            print(f"  Avg Prediction: {avg_pred:.6f}")
            print(f"  Prediction Range: {min(predictions):.3f} - {max(predictions):.3f}")
        
        arduino.close()
        print("\\n✅ Test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_arduino()
