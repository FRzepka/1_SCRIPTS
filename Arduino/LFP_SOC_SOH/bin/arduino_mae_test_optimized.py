"""
Arduino MAE Test - Optimized Version
Hardware verification test for Arduino LSTM SOC predictions against ground truth data
Based on successful simple test pattern with proper MAE calculation
"""

import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

class ArduinoMAETestOptimized:
    def __init__(self, port='COM13', baudrate=115200, max_samples=50):
        self.port = port
        self.baudrate = baudrate
        self.max_samples = max_samples
        self.arduino = None
        
        # Test results
        self.predictions = []
        self.ground_truth = []
        self.absolute_errors = []
        self.input_data = []
        self.timestamps = []
        self.failures = 0
        
        # Data processing
        self.scaler = None
        self.test_data = None
        
    def load_ground_truth_data(self):
        """Load ground truth data from C19 cell (small sample for testing)"""
        print("📊 Loading ground truth data from MGFarm C19...")
        
        try:
            # Load C19 data
            c19_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
            
            if not os.path.exists(c19_path):
                print(f"❌ Data file not found: {c19_path}")
                return False
                
            df_c19 = pd.read_parquet(c19_path)
            print(f"✅ Loaded {len(df_c19)} samples from C19")
            
            # Check required columns
            required_cols = ['Voltage[V]', 'Current[A]', 'SOH', 'Q_c', 'SOC_ZHU']
            missing_cols = [col for col in required_cols if col not in df_c19.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                print(f"📋 Available columns: {df_c19.columns.tolist()}")
                return False
            
            print("✅ All required columns found")
            
            # Create simple scaler using C19 data (fast approach)
            print("🔧 Creating StandardScaler...")
            features_data = df_c19[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].dropna()
            
            if len(features_data) == 0:
                print("❌ No valid feature data found")
                return False
                
            print(f"   📊 Using {len(features_data)} samples for scaling")
            
            self.scaler = StandardScaler()
            self.scaler.fit(features_data)
            
            print(f"📈 Feature Statistics:")
            print(f"   Voltage:  {features_data['Voltage[V]'].mean():.3f} ± {features_data['Voltage[V]'].std():.3f} V")
            print(f"   Current:  {features_data['Current[A]'].mean():.3f} ± {features_data['Current[A]'].std():.3f} A")
            print(f"   SOH:      {features_data['SOH'].mean():.3f} ± {features_data['SOH'].std():.3f}")
            print(f"   Q_c:      {features_data['Q_c'].mean():.1f} ± {features_data['Q_c'].std():.1f}")
            
            # Sample test data evenly (avoid memory issues)
            step_size = max(1, len(df_c19) // self.max_samples)
            test_indices = range(0, len(df_c19), step_size)[:self.max_samples]
            
            test_df = df_c19.iloc[test_indices].copy()
            test_df = test_df.dropna(subset=required_cols)
            
            if len(test_df) == 0:
                print("❌ No valid test samples found")
                return False
            
            # Extract features and ground truth SOC
            test_features = test_df[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].values
            test_soc_gt = test_df['SOC_ZHU'].values
            
            self.test_data = {
                'features_raw': test_features,
                'soc_ground_truth': test_soc_gt,
                'sample_indices': test_indices[:len(test_features)]
            }
            
            print(f"✅ Prepared {len(test_features)} test samples")
            print(f"   📊 SOC Ground Truth Range: {test_soc_gt.min():.3f} - {test_soc_gt.max():.3f}")
            print(f"   ⚡ Voltage Range: {test_features[:, 0].min():.3f} - {test_features[:, 0].max():.3f} V")
            print(f"   🔋 Current Range: {test_features[:, 1].min():.3f} - {test_features[:, 1].max():.3f} A")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ground truth data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
                    if line and 'LSTM' in line:
                        print(f"   📋 {line}")
                        info_received = True
                except:
                    break
            
            if not info_received:
                print("⚠️ No Arduino model info received")
            
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
        """Run MAE test with ground truth data"""
        if self.test_data is None:
            print("❌ No test data available")
            return False
        
        print(f"\n🧪 Running MAE Test with {len(self.test_data['features_raw'])} samples...")
        print("=" * 60)
        print("Sample | Input (V, I, SOH, Q_c) → Arduino | Ground Truth | Error")
        print("=" * 60)
        
        start_time = time.time()
        
        for i, (features, soc_gt) in enumerate(zip(self.test_data['features_raw'], 
                                                   self.test_data['soc_ground_truth'])):
            v, i_curr, soh, q_c = features
            
            # Get Arduino prediction
            soc_pred = self.get_arduino_prediction(v, i_curr, soh, q_c)
            
            if soc_pred is not None:
                # Calculate error
                error = abs(soc_pred - soc_gt)
                
                # Store results
                self.predictions.append(soc_pred)
                self.ground_truth.append(soc_gt)
                self.absolute_errors.append(error)
                self.input_data.append([v, i_curr, soh, q_c])
                self.timestamps.append(time.time())
                
                # Display result
                print(f"{i+1:6d} | ({v:4.2f}, {i_curr:5.2f}, {soh:4.2f}, {q_c:4.0f}) → {soc_pred:.4f} | {soc_gt:.4f} | {error:.4f}")
                
            else:
                self.failures += 1
                print(f"{i+1:6d} | ({v:4.2f}, {i_curr:5.2f}, {soh:4.2f}, {q_c:4.0f}) → FAILED  | {soc_gt:.4f} | -----")
            
            # Small delay between samples to avoid overwhelming Arduino
            time.sleep(0.1)
        
        test_duration = time.time() - start_time
        
        print("=" * 60)
        print(f"✅ MAE test completed in {test_duration:.1f} seconds")
        
        return len(self.predictions) > 0
    
    def calculate_metrics(self):
        """Calculate and display test metrics"""
        if len(self.predictions) == 0:
            print("❌ No successful predictions to analyze")
            return
        
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
        total_samples = len(self.test_data['features_raw'])
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
        ax1.scatter(ground_truth, predictions, alpha=0.7, s=50)
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
        ax3.hist(errors, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Mean = {np.mean(errors):.4f}')
        ax3.set_xlabel('Absolute Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Time Series
        timestamps_rel = np.array(self.timestamps) - self.timestamps[0]
        ax4.plot(timestamps_rel, predictions, 'b-', label='Arduino Predictions', marker='o', markersize=3)
        ax4.plot(timestamps_rel, ground_truth, 'r-', label='Ground Truth', marker='s', markersize=3)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('SOC')
        ax4.set_title('SOC Predictions Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arduino_mae_test_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved as: {filename}")
        plt.show()
    
    def run_full_test(self):
        """Run complete MAE test"""
        print("🚀 Arduino LSTM SOC - MAE Hardware Test")
        print("=" * 50)
        
        # Step 1: Load ground truth data
        if not self.load_ground_truth_data():
            return False
        
        # Step 2: Connect to Arduino
        if not self.connect_arduino():
            return False
        
        # Step 3: Run MAE test
        if not self.run_mae_test():
            return False
        
        # Step 4: Calculate metrics
        metrics = self.calculate_metrics()
        
        # Step 5: Create visualization
        self.create_visualization()
        
        # Step 6: Final summary
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
    # You can adjust max_samples to control test size (50 is good for quick test)
    tester = ArduinoMAETestOptimized(port='COM13', max_samples=30)
    success = tester.run_full_test()
    
    if success:
        print("\n✅ Arduino MAE test completed successfully!")
    else:
        print("\n❌ Arduino MAE test failed!")

if __name__ == "__main__":
    main()
