"""
Arduino MAE Test - Simple and Robust Version
Hardware verification test for Arduino LSTM SOC predictions
Uses synthetic test data with known expected values
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class ArduinoMAETestSimple:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        
        # Test results
        self.predictions = []
        self.ground_truth = []
        self.absolute_errors = []
        self.input_data = []
        self.timestamps = []
        self.failures = 0
        
    def create_test_data(self):
        """Create synthetic test data with realistic battery values"""
        print("📊 Creating synthetic test data...")
        
        # Create realistic battery test scenarios
        # Format: (voltage, current, soh, q_c, expected_soc_range)
        test_scenarios = [
            # Low SOC scenarios (discharged battery)
            (3.0, -2.0, 0.95, 5000, (0.1, 0.3)),  # Low voltage, discharging
            (3.1, -1.5, 0.95, 5000, (0.2, 0.4)),  # Low voltage, mild discharge
            (3.2, -1.0, 0.95, 5000, (0.3, 0.5)),  # Low-mid voltage, light discharge
            
            # Mid SOC scenarios
            (3.3, -0.5, 0.95, 5000, (0.4, 0.6)),  # Mid voltage, light discharge
            (3.4, 0.0, 0.95, 5000, (0.5, 0.7)),   # Mid voltage, no current
            (3.5, 0.5, 0.95, 5000, (0.6, 0.8)),   # Mid voltage, light charge
            
            # High SOC scenarios (charged battery)
            (3.6, 1.0, 0.95, 5000, (0.7, 0.9)),   # High voltage, charging
            (3.8, 1.5, 0.95, 5000, (0.8, 0.95)),  # High voltage, moderate charge
            (4.0, 2.0, 0.95, 5000, (0.9, 1.0)),   # Very high voltage, fast charge
            
            # Edge cases with varying SOH and Q_c
            (3.3, -1.0, 0.85, 4500, (0.2, 0.5)),  # Aged battery
            (3.5, 0.0, 0.90, 4800, (0.4, 0.7)),   # Partially aged
            (3.7, 1.0, 0.98, 5200, (0.7, 0.9)),   # Good battery, high capacity
        ]
        
        # Convert to test data format
        self.test_data = []
        
        for v, i, soh, q_c, soc_range in test_scenarios:
            # Use middle of expected range as ground truth
            soc_gt = (soc_range[0] + soc_range[1]) / 2
            
            self.test_data.append({
                'voltage': v,
                'current': i,
                'soh': soh,
                'q_c': q_c,
                'soc_ground_truth': soc_gt,
                'soc_range': soc_range
            })
        
        print(f"✅ Created {len(self.test_data)} test scenarios")
        return True
    
    def connect_arduino(self):
        """Connect to Arduino and verify communication"""
        print(f"🔌 Connecting to Arduino on {self.port}...")
        
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(2)  # Arduino initialization time
            
            print("✅ Arduino connected")
            
            # Clear startup messages
            self.arduino.flushInput()
            time.sleep(0.5)
            
            # Test with INFO command
            print("🔧 Testing Arduino communication...")
            self.arduino.write(b'INFO\n')
            time.sleep(1)
            
            # Read INFO response
            info_received = False
            while self.arduino.in_waiting:
                try:
                    line = self.arduino.readline().decode().strip()
                    if line and ('LSTM' in line or 'Model' in line):
                        print(f"   📋 {line}")
                        info_received = True
                    elif line and line.replace('.', '').isdigit():
                        # Sometimes only prediction value is returned
                        print(f"   📋 Model ready (got prediction: {line})")
                        info_received = True
                except:
                    break
            
            if not info_received:
                print("⚠️ No clear Arduino model info, but connection established")
            
            # Reset LSTM states
            print("🔄 Resetting LSTM states...")
            self.arduino.flushInput()
            self.arduino.write(b'RESET\n')
            time.sleep(0.5)
            
            try:
                response = self.arduino.readline().decode().strip()
                if response:
                    print(f"   Reset response: {response}")
            except:
                print("   Reset completed (no response)")
            
            self.arduino.flushInput()
            return True
            
        except Exception as e:
            print(f"❌ Arduino connection failed: {e}")
            return False
    
    def get_arduino_prediction(self, voltage, current, soh, q_c, max_attempts=2):
        """Get SOC prediction from Arduino with robust error handling"""
        
        for attempt in range(max_attempts):
            try:
                # Clear input buffer
                self.arduino.flushInput()
                
                # Send data (Arduino expects raw values, not scaled)
                data_str = f"{voltage:.6f},{current:.6f},{soh:.6f},{q_c:.6f}\n"
                self.arduino.write(data_str.encode())
                
                # Wait for response
                time.sleep(0.15)  # Give Arduino time to compute
                
                # Read response
                if self.arduino.in_waiting:
                    response = self.arduino.readline().decode().strip()
                    
                    # Parse SOC value
                    try:
                        soc_pred = float(response)
                        if 0.0 <= soc_pred <= 1.0:  # Valid SOC range
                            return soc_pred
                        else:
                            if attempt == 0:  # Only show warning on first attempt
                                print(f"   ⚠️ SOC out of range: {soc_pred}")
                    except ValueError:
                        if attempt == 0:
                            print(f"   ⚠️ Invalid response: '{response}'")
                
            except Exception as e:
                if attempt == 0:
                    print(f"   ⚠️ Communication error: {e}")
        
        return None
    
    def run_mae_test(self):
        """Run MAE test with synthetic test data"""
        if not self.test_data:
            print("❌ No test data available")
            return False
        
        print(f"\n🧪 Running MAE Test with {len(self.test_data)} scenarios...")
        print("=" * 80)
        print("Sample | Input (V, I, SOH, Q_c) → Arduino | Ground Truth | Error | Status")
        print("=" * 80)
        
        start_time = time.time()
        
        for i, test_case in enumerate(self.test_data):
            v = test_case['voltage']
            i_curr = test_case['current']
            soh = test_case['soh']
            q_c = test_case['q_c']
            soc_gt = test_case['soc_ground_truth']
            soc_range = test_case['soc_range']
            
            # Get Arduino prediction
            soc_pred = self.get_arduino_prediction(v, i_curr, soh, q_c)
            
            if soc_pred is not None:
                # Calculate error
                error = abs(soc_pred - soc_gt)
                
                # Check if prediction is within expected range
                in_range = soc_range[0] <= soc_pred <= soc_range[1]
                status = "✅ GOOD" if in_range else "⚠️ OUT_OF_RANGE"
                
                # Store results
                self.predictions.append(soc_pred)
                self.ground_truth.append(soc_gt)
                self.absolute_errors.append(error)
                self.input_data.append([v, i_curr, soh, q_c])
                self.timestamps.append(time.time())
                
                # Display result
                print(f"{i+1:6d} | ({v:4.1f}, {i_curr:5.1f}, {soh:4.2f}, {q_c:4.0f}) → {soc_pred:.4f} | {soc_gt:.4f} | {error:.4f} | {status}")
                
            else:
                self.failures += 1
                print(f"{i+1:6d} | ({v:4.1f}, {i_curr:5.1f}, {soh:4.2f}, {q_c:4.0f}) → FAILED  | {soc_gt:.4f} | ----- | ❌ FAIL")
            
            # Small delay between samples to avoid overwhelming Arduino
            time.sleep(0.1)
        
        test_duration = time.time() - start_time
        
        print("=" * 80)
        print(f"✅ MAE test completed in {test_duration:.1f} seconds")
        
        return len(self.predictions) > 0
    
    def calculate_metrics(self):
        """Calculate and display test metrics"""
        if len(self.predictions) == 0:
            print("❌ No successful predictions to analyze")
            return None
        
        # Convert to numpy arrays
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        errors = np.array(self.absolute_errors)
        
        # Calculate metrics
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean((predictions - ground_truth)**2))
        max_error = np.max(errors)
        min_error = np.min(errors)
        std_error = np.std(errors)
        
        # Success rate
        total_samples = len(self.test_data)
        success_rate = len(self.predictions) / total_samples * 100
        
        print(f"\n📊 MAE TEST RESULTS")
        print("=" * 50)
        print(f"🎯 SUCCESS RATE: {success_rate:.1f}% ({len(self.predictions)}/{total_samples})")
        print(f"❌ COMMUNICATION FAILURES: {self.failures}")
        print(f"\n📈 ACCURACY METRICS:")
        print(f"   🔥 Mean Absolute Error (MAE): {mae:.6f}")
        print(f"   📊 Root Mean Square Error (RMSE): {rmse:.6f}")
        print(f"   📈 Maximum Error: {max_error:.6f}")
        print(f"   📉 Minimum Error: {min_error:.6f}")
        print(f"   📊 Error Standard Deviation: {std_error:.6f}")
        
        print(f"\n📈 SOC PREDICTION RANGE:")
        print(f"   🎯 Arduino Predictions: {predictions.min():.3f} - {predictions.max():.3f}")
        print(f"   📊 Ground Truth: {ground_truth.min():.3f} - {ground_truth.max():.3f}")
        
        # Input data range
        input_array = np.array(self.input_data)
        print(f"\n🔋 INPUT DATA RANGE:")
        print(f"   ⚡ Voltage: {input_array[:, 0].min():.3f} - {input_array[:, 0].max():.3f} V")
        print(f"   🔋 Current: {input_array[:, 1].min():.3f} - {input_array[:, 1].max():.3f} A")
        print(f"   📊 SOH: {input_array[:, 2].min():.3f} - {input_array[:, 2].max():.3f}")
        print(f"   ⚡ Q_c: {input_array[:, 3].min():.1f} - {input_array[:, 3].max():.1f}")
        
        return {'mae': mae, 'rmse': rmse, 'success_rate': success_rate}
    
    def create_visualization(self):
        """Create visualization of test results"""
        if len(self.predictions) == 0:
            print("❌ No data to visualize")
            return
        
        print(f"\n📊 Creating visualization...")
        
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        errors = np.array(self.absolute_errors)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Arduino LSTM SOC Prediction - MAE Test Results', fontsize=16, fontweight='bold')
        
        # 1. Predictions vs Ground Truth
        ax1.scatter(ground_truth, predictions, alpha=0.7, s=50, c='blue')
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        ax1.set_xlabel('Ground Truth SOC')
        ax1.set_ylabel('Arduino Prediction SOC')
        ax1.set_title('Predictions vs Ground Truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Absolute Errors
        sample_numbers = range(1, len(errors) + 1)
        ax2.plot(sample_numbers, errors, 'b-', marker='o', markersize=4, alpha=0.7)
        ax2.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'MAE = {np.mean(errors):.4f}')
        ax2.set_xlabel('Sample Number')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Absolute Error per Sample')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Error Distribution
        ax3.hist(errors, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Mean = {np.mean(errors):.4f}')
        ax3.set_xlabel('Absolute Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Input vs Prediction
        input_voltages = np.array(self.input_data)[:, 0]
        ax4.scatter(input_voltages, predictions, alpha=0.7, s=50, c='orange')
        ax4.set_xlabel('Input Voltage (V)')
        ax4.set_ylabel('Predicted SOC')
        ax4.set_title('Voltage vs Predicted SOC')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arduino_mae_test_simple_{timestamp}.png"
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved as: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save visualization: {e}")
        
        plt.show()
    
    def test_cable_disconnection(self):
        """Test Arduino response to cable disconnection"""
        print(f"\n🔌 Testing cable disconnection robustness...")
        print("⚠️ Please disconnect the USB cable when prompted...")
        
        # First verify connection is working
        test_v, test_i, test_soh, test_qc = 3.5, 0.0, 0.95, 5000
        
        print(f"1. Testing normal operation...")
        pred1 = self.get_arduino_prediction(test_v, test_i, test_soh, test_qc)
        if pred1 is not None:
            print(f"   ✅ Normal prediction: {pred1:.4f}")
        else:
            print(f"   ❌ Normal prediction failed")
            return
        
        print(f"\n🔌 PLEASE DISCONNECT THE USB CABLE NOW!")
        print("   (You have 5 seconds to disconnect...)")
        
        for i in range(5, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print(f"\n2. Testing disconnected state...")
        try:
            pred2 = self.get_arduino_prediction(test_v, test_i, test_soh, test_qc)
            if pred2 is not None:
                print(f"   ⚠️ Unexpected prediction during disconnection: {pred2:.4f}")
            else:
                print(f"   ✅ Correctly detected disconnection (no prediction)")
        except Exception as e:
            print(f"   ✅ Correctly detected disconnection: {e}")
        
        print(f"\n🔌 PLEASE RECONNECT THE USB CABLE NOW!")
        print("   Waiting for reconnection...")
        
        # Wait for reconnection
        reconnected = False
        for attempt in range(20):  # Wait up to 20 seconds
            try:
                if self.arduino and not self.arduino.is_open:
                    self.arduino.open()
                
                # Test if we can communicate
                time.sleep(0.5)
                pred3 = self.get_arduino_prediction(test_v, test_i, test_soh, test_qc)
                if pred3 is not None:
                    print(f"   ✅ Reconnected! Test prediction: {pred3:.4f}")
                    reconnected = True
                    break
            except:
                pass
            
            print(f"   Waiting... ({attempt+1}/20)")
            time.sleep(1)
        
        if not reconnected:
            print(f"   ❌ Could not automatically reconnect. Manual restart may be needed.")
        
        return reconnected
    
    def run_full_test(self):
        """Run complete MAE test"""
        print("🚀 Arduino LSTM SOC - Simple MAE Hardware Test")
        print("=" * 50)
        
        # Step 1: Create test data
        if not self.create_test_data():
            return False
        
        # Step 2: Connect to Arduino
        if not self.connect_arduino():
            return False
        
        # Step 3: Run MAE test
        if not self.run_mae_test():
            return False
        
        # Step 4: Calculate metrics
        metrics = self.calculate_metrics()
        
        if metrics is None:
            return False
        
        # Step 5: Create visualization
        self.create_visualization()
        
        # Step 6: Test cable disconnection (optional)
        print(f"\n🔌 Would you like to test cable disconnection robustness? (Auto-skipping for now)")
        # Uncomment the line below to enable cable disconnection test
        # self.test_cable_disconnection()
        
        # Step 7: Final summary
        print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"📊 MAE: {metrics['mae']:.6f}")
        print(f"🎯 Success Rate: {metrics['success_rate']:.1f}%")
        
        # Cleanup
        if self.arduino:
            self.arduino.close()
            print("🔌 Arduino connection closed")
        
        return True

def main():
    """Main test execution"""
    tester = ArduinoMAETestSimple(port='COM13')
    success = tester.run_full_test()
    
    if success:
        print("\n✅ Arduino MAE test completed successfully!")
    else:
        print("\n❌ Arduino MAE test failed!")

if __name__ == "__main__":
    main()
