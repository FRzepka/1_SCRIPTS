"""
Arduino MAE Test - Extended from working simple test
"""

import serial
import time

def arduino_mae_test():
    print("🧪 Arduino MAE Test - Extended Version")
    print("=" * 60)
    
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
        
        # Read Arduino info
        print("\\n📊 Arduino INFO:")
        while arduino.in_waiting:
            try:
                line = arduino.readline().decode().strip()
                if line:
                    print(f"  {line}")
            except:
                break
        
        # Send RESET command
        print("\\n🔄 Resetting LSTM states...")
        arduino.flushInput()
        arduino.write(b'RESET\\n')
        time.sleep(0.5)
        
        try:
            response = arduino.readline().decode().strip()
            print(f"  Reset response: {response}")
        except:
            print("  Reset completed")
        
        # Clear buffer
        arduino.flushInput()
        
        # Extended test cases for MAE calculation
        test_cases = [
            # (voltage, current, soh, q_c, expected_soc_estimate, description)
            (3.0, -2.0, 0.95, 5000, 0.15, "Very low SOC (discharged)"),
            (3.1, -1.5, 0.95, 5000, 0.25, "Low SOC (discharging)"),
            (3.2, -1.0, 0.95, 5000, 0.35, "Low-mid SOC (mild discharge)"),
            (3.3, -0.5, 0.95, 5000, 0.45, "Mid SOC (light discharge)"),
            (3.4, 0.0, 0.95, 5000, 0.55, "Mid SOC (no current)"),
            (3.5, 0.5, 0.95, 5000, 0.65, "Mid-high SOC (light charge)"),
            (3.6, 1.0, 0.95, 5000, 0.75, "High SOC (charging)"),
            (3.7, 1.5, 0.95, 5000, 0.85, "Very high SOC (fast charge)"),
            (3.8, 2.0, 0.95, 5000, 0.90, "Near full SOC"),
            (4.0, 1.0, 0.95, 5000, 0.95, "Full SOC"),
            
            # Test with different SOH and Q_c
            (3.3, -1.0, 0.85, 4500, 0.30, "Aged battery (low SOH)"),
            (3.5, 0.0, 0.90, 4800, 0.60, "Partially aged battery"),
        ]
        
        print(f"\\n🧪 Running MAE Test with {len(test_cases)} scenarios...")
        print("=" * 100)
        print("Test | Input (V, I, SOH, Q_c) → Arduino SOC | Expected | Error | Description")
        print("=" * 100)
        
        predictions = []
        ground_truth = []
        errors = []
        success_count = 0
        
        for i, (v, curr, soh, q_c, expected_soc, description) in enumerate(test_cases):
            try:
                # Clear input buffer before sending
                arduino.flushInput()
                
                # Send data
                data_str = f"{v},{curr},{soh},{q_c}\\n"
                arduino.write(data_str.encode())
                time.sleep(0.2)  # Give Arduino time to process
                
                # Read response
                if arduino.in_waiting:
                    response = arduino.readline().decode().strip()
                    
                    try:
                        soc_pred = float(response)
                        error = abs(soc_pred - expected_soc)
                        
                        predictions.append(soc_pred)
                        ground_truth.append(expected_soc)
                        errors.append(error)
                        success_count += 1
                        
                        print(f"{i+1:4d} | ({v:4.1f}, {curr:5.1f}, {soh:4.2f}, {q_c:4.0f}) → {soc_pred:.6f} | {expected_soc:.3f} | {error:.6f} | {description}")
                        
                    except ValueError:
                        print(f"{i+1:4d} | ({v:4.1f}, {curr:5.1f}, {soh:4.2f}, {q_c:4.0f}) → PARSE_ERROR: '{response}' | {description}")
                else:
                    print(f"{i+1:4d} | ({v:4.1f}, {curr:5.1f}, {soh:4.2f}, {q_c:4.0f}) → NO_RESPONSE | {description}")
                
            except Exception as e:
                print(f"{i+1:4d} | ({v:4.1f}, {curr:5.1f}, {soh:4.2f}, {q_c:4.0f}) → ERROR: {e} | {description}")
        
        # Calculate MAE and other metrics
        print("\\n" + "=" * 100)
        print("📊 RESULTS SUMMARY:")
        print("=" * 100)
        
        if predictions and ground_truth:
            # Calculate metrics manually (no numpy to avoid hanging)
            mae = sum(errors) / len(errors)
            rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
            
            avg_prediction = sum(predictions) / len(predictions)
            min_pred = min(predictions)
            max_pred = max(predictions)
            
            avg_gt = sum(ground_truth) / len(ground_truth)
            
            print(f"  📈 Total Tests: {len(test_cases)}")
            print(f"  ✅ Successful: {success_count}")
            print(f"  📊 Success Rate: {success_count}/{len(test_cases)} ({100*success_count/len(test_cases):.1f}%)")
            print(f"  🎯 Mean Absolute Error (MAE): {mae:.6f}")
            print(f"  📐 Root Mean Square Error (RMSE): {rmse:.6f}")
            print(f"  📊 Average Prediction: {avg_prediction:.6f}")
            print(f"  📊 Average Ground Truth: {avg_gt:.6f}")
            print(f"  📊 Prediction Range: {min_pred:.6f} - {max_pred:.6f}")
            
            # Performance evaluation
            print(f"\\n🎯 PERFORMANCE EVALUATION:")
            if mae < 0.05:
                print("  ✅ EXCELLENT: MAE < 0.05 (very high accuracy)")
            elif mae < 0.10:
                print("  ✅ VERY GOOD: MAE < 0.10 (high accuracy)")
            elif mae < 0.15:
                print("  ✅ GOOD: MAE < 0.15 (good accuracy)")
            elif mae < 0.20:
                print("  ⚠️ ACCEPTABLE: MAE < 0.20 (acceptable accuracy)")
            elif mae < 0.30:
                print("  ⚠️ POOR: MAE < 0.30 (needs improvement)")
            else:
                print("  ❌ VERY POOR: MAE >= 0.30 (significant error)")
            
            # Analyze prediction distribution
            high_soc_preds = [p for p in predictions if p > 0.8]
            low_soc_preds = [p for p in predictions if p < 0.3]
            
            print(f"\\n📊 PREDICTION DISTRIBUTION:")
            print(f"  High SOC predictions (>0.8): {len(high_soc_preds)}")
            print(f"  Low SOC predictions (<0.3): {len(low_soc_preds)}")
            print(f"  Mid SOC predictions (0.3-0.8): {len(predictions) - len(high_soc_preds) - len(low_soc_preds)}")
            
        else:
            print("  ❌ No successful predictions to analyze")
        
        # Close connection
        arduino.close()
        print("\\n🔌 Arduino connection closed")
        print("✅ MAE Test completed!")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = arduino_mae_test()
    if success:
        print("\\n🎉 Arduino MAE testing successful!")
    else:
        print("\\n💥 Arduino MAE testing failed!")
