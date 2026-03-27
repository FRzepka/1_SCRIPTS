"""
Arduino MAE Test - Minimal Working Version
Based on the successful simple_arduino_test_fixed.py
"""

import serial
import time
import numpy as np

def main():
    print("🧪 Arduino MAE Test - Minimal Version")
    print("=" * 50)
    
    try:
        # Connect to Arduino
        print("🔌 Connecting to Arduino...")
        arduino = serial.Serial('COM13', 115200, timeout=3)
        time.sleep(2)
        print("✅ Connected")
        
        # Clear and get INFO
        arduino.flushInput()
        arduino.write(b'INFO\n')
        time.sleep(1)
        
        info_response = ""
        while arduino.in_waiting:
            try:
                line = arduino.readline().decode().strip()
                if line:
                    info_response = line
                    break
            except:
                break
        
        print(f"📋 Arduino Model: {info_response}")
        
        # Reset LSTM
        print("🔄 Resetting LSTM...")
        arduino.flushInput()
        arduino.write(b'RESET\n')
        time.sleep(0.5)
        
        try:
            reset_response = arduino.readline().decode().strip()
            print(f"   Reset: {reset_response}")
        except:
            print("   Reset completed")
        
        # Test data (simple scenarios)
        test_cases = [
            # (voltage, current, soh, q_c, expected_description)
            (3.0, -2.0, 0.95, 5000, "Low SOC (discharged)"),
            (3.3, -0.5, 0.95, 5000, "Mid SOC (slight discharge)"),
            (3.6, 0.0, 0.95, 5000, "Mid-High SOC (no current)"),
            (3.8, 1.0, 0.95, 5000, "High SOC (charging)"),
            (4.0, 2.0, 0.95, 5000, "Very High SOC (fast charge)"),
        ]
        
        print(f"\\n🧪 Running MAE Test with {len(test_cases)} scenarios...")
        print("=" * 80)
        print("Test | Input (V, I, SOH, Q_c) → Arduino SOC | Description")
        print("=" * 80)
        
        predictions = []
        ground_truth = []  # We'll estimate reasonable ground truth values
        
        for i, (v, curr, soh, q_c, description) in enumerate(test_cases):
            # Clear input buffer
            arduino.flushInput()
            
            # Send data
            data_str = f"{v:.6f},{curr:.6f},{soh:.6f},{q_c:.6f}\\n"
            arduino.write(data_str.encode())
            
            # Wait and read response
            time.sleep(0.15)
            
            if arduino.in_waiting:
                try:
                    response = arduino.readline().decode().strip()
                    soc_pred = float(response)
                    
                    # Estimate ground truth based on voltage (rough approximation)
                    # LiFePO4: ~3.0V = 0% SOC, ~3.6V = 100% SOC (simplified)
                    if v <= 3.0:
                        soc_gt = 0.1
                    elif v >= 3.6:
                        soc_gt = 0.9
                    else:
                        soc_gt = 0.1 + (v - 3.0) / (3.6 - 3.0) * 0.8
                    
                    # Adjust for current (charging increases SOC estimate)
                    if curr > 0:  # Charging
                        soc_gt = min(1.0, soc_gt + 0.1)
                    elif curr < -1:  # Heavy discharge
                        soc_gt = max(0.0, soc_gt - 0.1)
                    
                    error = abs(soc_pred - soc_gt)
                    
                    predictions.append(soc_pred)
                    ground_truth.append(soc_gt)
                    
                    print(f"{i+1:4d} | ({v:4.1f}, {curr:5.1f}, {soh:4.2f}, {q_c:4.0f}) → {soc_pred:.6f} | {description}")
                    print(f"     | Ground Truth: {soc_gt:.3f}, Error: {error:.6f}")
                    
                except Exception as e:
                    print(f"{i+1:4d} | ({v:4.1f}, {curr:5.1f}, {soh:4.2f}, {q_c:4.0f}) → ERROR: {e}")
            else:
                print(f"{i+1:4d} | ({v:4.1f}, {curr:5.1f}, {soh:4.2f}, {q_c:4.0f}) → NO RESPONSE")
        
        # Calculate MAE
        if predictions and ground_truth:
            predictions = np.array(predictions)
            ground_truth = np.array(ground_truth)
            
            mae = np.mean(np.abs(predictions - ground_truth))
            rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
            
            print("\\n" + "=" * 80)
            print("📊 RESULTS:")
            print(f"  Total Tests: {len(predictions)}")
            print(f"  Success Rate: {len(predictions)}/{len(test_cases)} ({100*len(predictions)/len(test_cases):.1f}%)")
            print(f"  Mean Absolute Error (MAE): {mae:.6f}")
            print(f"  Root Mean Square Error (RMSE): {rmse:.6f}")
            print(f"  Average Prediction: {np.mean(predictions):.6f}")
            print(f"  Prediction Range: {np.min(predictions):.6f} - {np.max(predictions):.6f}")
            print("=" * 80)
            
            # Simple analysis
            if mae < 0.1:
                print("✅ EXCELLENT: MAE < 0.1 (very good accuracy)")
            elif mae < 0.2:
                print("✅ GOOD: MAE < 0.2 (acceptable accuracy)")
            elif mae < 0.3:
                print("⚠️ FAIR: MAE < 0.3 (needs improvement)")
            else:
                print("❌ POOR: MAE >= 0.3 (significant error)")
        
        # Close connection
        arduino.close()
        print("\\n🔌 Arduino connection closed")
        print("✅ MAE Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
