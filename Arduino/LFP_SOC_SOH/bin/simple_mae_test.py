"""
Simple MAE Test - Test basic functionality first
"""

import serial
import time
import numpy as np

def simple_mae_test():
    print("🧪 Simple Arduino MAE Test")
    print("=" * 40)
    
    try:
        # Connect to Arduino
        print("🔌 Connecting to Arduino...")
        arduino = serial.Serial('COM13', 115200, timeout=2)
        time.sleep(3)
        print("✅ Connected!")
        
        # Reset Arduino
        print("🔄 Resetting Arduino...")
        arduino.flushInput()
        arduino.write(b'RESET\n')
        time.sleep(0.5)
        
        response = arduino.readline().decode().strip()
        print(f"Reset response: {response}")
        
        # Test cases with known SOC values (estimated)
        test_cases = [
            # (voltage, current, soh, q_c, expected_soc_range)
            (3.0, -1.0, 0.95, 5000, (0.0, 0.3)),   # Low voltage, discharging -> Low SOC
            (3.3, 0.0, 0.95, 5000, (0.2, 0.6)),    # Mid voltage, no current -> Mid SOC  
            (3.6, 1.0, 0.95, 5000, (0.5, 0.9)),    # High voltage, charging -> High SOC
            (4.0, 0.5, 0.95, 5000, (0.8, 1.0)),    # Very high voltage -> Very high SOC
        ]
        
        predictions = []
        ground_truths = []
        errors = []
        
        print("\\n🎯 Running predictions...")
        print("Input (V, I, SOH, Q_c) → Arduino SOC | Expected Range | Error")
        print("-" * 70)
        
        for i, (v, i_curr, soh, q_c, expected_range) in enumerate(test_cases):
            # Get Arduino prediction
            arduino.flushInput()
            data_str = f"{v},{i_curr},{soh},{q_c}\\n"
            arduino.write(data_str.encode())
            time.sleep(0.3)
            
            # Collect all responses
            responses = []
            for _ in range(15):
                if arduino.in_waiting:
                    try:
                        line = arduino.readline().decode().strip()
                        if line:
                            responses.append(line)
                    except:
                        continue
                else:
                    break
            
            # Find valid SOC prediction
            pred_soc = None
            for response in responses:
                try:
                    soc_val = float(response)
                    if 0.0 <= soc_val <= 1.0:
                        pred_soc = soc_val
                        break
                except ValueError:
                    continue
            
            if pred_soc is not None:
                # Use middle of expected range as "ground truth" for error calculation
                expected_soc = (expected_range[0] + expected_range[1]) / 2
                error = abs(pred_soc - expected_soc)
                
                predictions.append(pred_soc)
                ground_truths.append(expected_soc)
                errors.append(error)
                
                in_range = expected_range[0] <= pred_soc <= expected_range[1]
                status = "✅" if in_range else "⚠️"
                
                print(f"({v:4.1f}, {i_curr:4.1f}, {soh:4.2f}, {q_c:4.0f}) → {pred_soc:.6f} | {expected_range[0]:.1f}-{expected_range[1]:.1f} | {error:.4f} {status}")
                
            else:
                print(f"({v:4.1f}, {i_curr:4.1f}, {soh:4.2f}, {q_c:4.0f}) → ❌ No valid prediction")
                print(f"  Responses: {responses[:3]}")
        
        # Calculate metrics
        if len(predictions) > 0:
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean(np.array(errors)**2))
            max_error = np.max(errors)
            
            print("\\n" + "=" * 70)
            print("📊 RESULTS:")
            print(f"✅ Successful predictions: {len(predictions)}/{len(test_cases)}")
            print(f"📈 MAE (Mean Absolute Error): {mae:.6f}")
            print(f"📈 RMSE (Root Mean Square Error): {rmse:.6f}")
            print(f"📈 Maximum Error: {max_error:.6f}")
            print(f"📈 SOC Range: {min(predictions):.3f} - {max(predictions):.3f}")
            
            # Test cable disconnection
            print("\\n🔌 Testing cable disconnection...")
            print("   Please unplug the Arduino cable now and press Enter...")
            input()
            
            try:
                arduino.write(b"3.5,0,0.95,5000\\n")
                time.sleep(1)
                response = arduino.readline().decode().strip()
                print(f"❌ Unexpected response after disconnect: {response}")
            except Exception as e:
                print(f"✅ Cable disconnection detected: {e}")
            
        else:
            print("\\n❌ No successful predictions!")
        
        arduino.close()
        print("\\n✅ Test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    simple_mae_test()
