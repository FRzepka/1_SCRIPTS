"""
Final Arduino MAE Test - Comprehensive live hardware verification test
Works with current Arduino setup and provides full MAE analysis
"""

import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

class FinalArduinoMAETest:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        
        # Test results
        self.successful_predictions = []
        self.ground_truth_values = []
        self.absolute_errors = []
        self.timestamps = []
        self.input_data = []
        self.communication_failures = 0
        
        # Ground truth data
        self.test_data = None
        self.scaler = None
        
    def setup_ground_truth_data(self):
        """Setup ground truth data from C19 cell"""
        print("📊 Setting up ground truth data from MGFarm C19...")
        
        try:
            # Load C19 data  
            c19_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
            
            if not os.path.exists(c19_path):
                print(f"❌ C19 data file not found: {c19_path}")
                return False
                
            df_c19 = pd.read_parquet(c19_path)
            print(f"✅ Loaded {len(df_c19)} samples from C19")
            
            # Create scaler using training procedure
            print("🔧 Creating StandardScaler with training data...")
            
            # Load all training cells
            cell_names = ['C01', 'C03', 'C05', 'C11', 'C17', 'C23', 'C07', 'C19', 'C21']
            all_features = []
            
            base_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
            features = ['voltage', 'current', 'soh', 'q_c']
            
            for cell in cell_names:
                try:
                    cell_path = os.path.join(base_path, f"MGFarm_18650_{cell}", "df.parquet")
                    if os.path.exists(cell_path):
                        df_cell = pd.read_parquet(cell_path)
                        all_features.append(df_cell[features])
                        print(f"   ✅ Loaded {cell}: {len(df_cell)} samples")
                    else:
                        print(f"   ⚠️ Skipped {cell}: file not found")
                except Exception as e:
                    print(f"   ⚠️ Error loading {cell}: {e}")
            
            if len(all_features) == 0:
                print("❌ No training data could be loaded for scaler")
                return False
                
            # Combine all training data
            combined_features = pd.concat(all_features, ignore_index=True)
            print(f"✅ Combined {len(combined_features)} training samples from {len(all_features)} cells")
            
            # Fit scaler
            self.scaler = StandardScaler()
            self.scaler.fit(combined_features)
            
            print(f"📈 Scaler Statistics:")
            print(f"   Mean: {self.scaler.mean_}")
            print(f"   Std:  {self.scaler.scale_}")
            
            # Prepare test data (sample from C19 for variety)
            test_indices = np.linspace(0, len(df_c19)-1, 200, dtype=int)  # Sample 200 points
            test_features = df_c19.iloc[test_indices][features].values
            test_soc = df_c19.iloc[test_indices]['soc'].values
            
            self.test_data = {
                'features_raw': test_features,
                'soc_true': test_soc,
                'indices': test_indices
            }
            
            print(f"✅ Prepared {len(test_features)} test samples")
            print(f"   SOC range: {test_soc.min():.3f} - {test_soc.max():.3f}")
            print(f"   Voltage range: {test_features[:, 0].min():.3f} - {test_features[:, 0].max():.3f} V")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup ground truth data: {e}")
            return False
    
    def connect_arduino(self):
        """Connect to Arduino with robust error handling"""
        print(f"🔌 Connecting to Arduino on {self.port}...")
        
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(3)  # Give Arduino time to initialize
            
            print("✅ Arduino connected")
            
            # Clear startup messages
            self.arduino.flushInput()
            time.sleep(0.5)
              # Test connection with INFO command
            print("🔧 Testing Arduino communication...")
            self.arduino.write(b'INFO\n')
            time.sleep(1)
            
            info_lines = []
            while self.arduino.in_waiting:
                try:
                    line = self.arduino.readline().decode().strip()
                    if line:
                        info_lines.append(line)
                except:
                    break
            
            if info_lines:
                print("📊 Arduino Info:")
                for line in info_lines[:3]:  # Show first 3 lines
                    print(f"   {line}")
                if len(info_lines) > 3:
                    print(f"   ... and {len(info_lines)-3} more lines")
              # Send RESET command
            print("🔄 Resetting Arduino LSTM states...")
            self.arduino.flushInput()
            self.arduino.write(b'RESET\n')
            time.sleep(0.5)
            
            reset_response = self.arduino.readline().decode().strip()
            print(f"   Reset response: {reset_response}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Arduino: {e}")
            return False
    
    def get_arduino_prediction(self, voltage, current, soh, q_c):
        """Get SOC prediction from Arduino with robust parsing"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Clear input buffer
                self.arduino.flushInput()
                
                # Send prediction request
                data_str = f"{voltage:.6f},{current:.6f},{soh:.6f},{q_c:.6f}\\n"
                self.arduino.write(data_str.encode())
                time.sleep(0.4)  # Give Arduino time to process
                
                # Collect all responses
                responses = []
                start_time = time.time()
                
                while time.time() - start_time < 2.0:  # 2 second timeout
                    if self.arduino.in_waiting:
                        try:
                            line = self.arduino.readline().decode().strip()
                            if line:
                                responses.append(line)
                                
                                # Try to parse each line as SOC
                                try:
                                    soc_val = float(line)
                                    if 0.0 <= soc_val <= 1.0:  # Valid SOC range
                                        return soc_val
                                except ValueError:
                                    continue
                        except:
                            continue
                    else:
                        time.sleep(0.05)
                
                # If no immediate SOC found, search through all responses
                for response in responses:
                    try:
                        soc_val = float(response)
                        if 0.0 <= soc_val <= 1.0:
                            return soc_val
                    except ValueError:
                        continue
                
                if attempt == 0 and responses:
                    print(f"   ⚠️ No valid SOC in responses: {responses[:2]}")
                
            except Exception as e:
                if attempt == 0:
                    print(f"   ⚠️ Communication error: {e}")
        
        return None
    
    def run_mae_test(self, num_samples=50):
        """Run the main MAE test"""
        print(f"\\n🚀 STARTING ARDUINO MAE VERIFICATION TEST")
        print("=" * 60)
        print(f"📊 Testing {num_samples} predictions with ground truth comparison")
        print("🔌 You can unplug the Arduino cable during the test to verify disconnection detection")
        print("⏹️  Press Ctrl+C to stop the test early")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            for i in range(num_samples):
                # Get test sample
                sample_idx = i % len(self.test_data['features_raw'])
                
                voltage = self.test_data['features_raw'][sample_idx, 0]
                current = self.test_data['features_raw'][sample_idx, 1]
                soh = self.test_data['features_raw'][sample_idx, 2]
                q_c = self.test_data['features_raw'][sample_idx, 3]
                true_soc = self.test_data['soc_true'][sample_idx]
                
                # Get Arduino prediction
                try:
                    pred_soc = self.get_arduino_prediction(voltage, current, soh, q_c)
                    
                    if pred_soc is not None:
                        # Calculate error
                        error = abs(pred_soc - true_soc)
                        
                        # Store results
                        self.successful_predictions.append(pred_soc)
                        self.ground_truth_values.append(true_soc)
                        self.absolute_errors.append(error)
                        self.timestamps.append(time.time() - start_time)
                        self.input_data.append([voltage, current, soh, q_c])
                        
                        # Progress update
                        if (i + 1) % 10 == 0 or i < 5:
                            current_mae = np.mean(self.absolute_errors[-10:]) if len(self.absolute_errors) >= 10 else np.mean(self.absolute_errors)
                            print(f"[{i+1:3d}/{num_samples}] V:{voltage:.3f} I:{current:.2f} → SOC: {pred_soc:.4f} | True: {true_soc:.4f} | Error: {error:.4f} | MAE-10: {current_mae:.4f}")
                    
                    else:
                        self.communication_failures += 1
                        print(f"[{i+1:3d}/{num_samples}] ❌ Communication failure (total: {self.communication_failures})")
                        
                        # Check for cable disconnection
                        if self.communication_failures >= 3:
                            print("🔌 Multiple communication failures detected - cable may be disconnected")
                            
                            # Try to reconnect
                            try:
                                self.arduino.close()
                                time.sleep(2)
                                if self.connect_arduino():
                                    print("✅ Reconnected successfully")
                                    self.communication_failures = 0
                                else:
                                    print("❌ Reconnection failed - stopping test")
                                    break
                            except:
                                print("❌ Unable to reconnect - stopping test")
                                break
                
                except Exception as e:
                    print(f"[{i+1:3d}/{num_samples}] ❌ Unexpected error: {e}")
                    self.communication_failures += 1
                
                # Small delay between predictions
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            print("\\n⏹️ Test stopped by user")
        
        # Show results
        self.show_results()
    
    def show_results(self):
        """Display comprehensive test results"""
        print("\\n" + "=" * 80)
        print("🎯 ARDUINO LSTM SOC PREDICTION - MAE TEST RESULTS")
        print("=" * 80)
        
        if len(self.successful_predictions) == 0:
            print("❌ No successful predictions recorded!")
            print(f"💥 Communication failures: {self.communication_failures}")
            print("\\n🔧 TROUBLESHOOTING:")
            print("   1. Check Arduino connection on COM13")
            print("   2. Verify correct Arduino program is uploaded")
            print("   3. Check if Arduino is responding to INFO commands")
            return
        
        # Calculate metrics
        mae = np.mean(self.absolute_errors)
        rmse = np.sqrt(np.mean(np.array(self.absolute_errors)**2))
        max_error = np.max(self.absolute_errors)
        min_error = np.min(self.absolute_errors)
        
        total_attempts = len(self.successful_predictions) + self.communication_failures
        success_rate = len(self.successful_predictions) / total_attempts * 100
        
        # Statistics
        print(f"📊 TEST STATISTICS:")
        print(f"   ✅ Successful predictions: {len(self.successful_predictions)}")
        print(f"   ❌ Communication failures: {self.communication_failures}")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        print(f"   ⏱️  Total test duration: {self.timestamps[-1]:.1f} seconds")
        
        print(f"\\n🎯 ACCURACY METRICS:")
        print(f"   📏 MAE (Mean Absolute Error): {mae:.6f}")
        print(f"   📐 RMSE (Root Mean Square Error): {rmse:.6f}")
        print(f"   📊 Maximum Error: {max_error:.6f}")
        print(f"   📊 Minimum Error: {min_error:.6f}")
        print(f"   📊 Error Standard Deviation: {np.std(self.absolute_errors):.6f}")
        
        print(f"\\n📈 SOC PREDICTION RANGE:")
        print(f"   🎯 Predicted SOC: {np.min(self.successful_predictions):.3f} - {np.max(self.successful_predictions):.3f}")
        print(f"   📊 True SOC: {np.min(self.ground_truth_values):.3f} - {np.max(self.ground_truth_values):.3f}")
        
        # Input data statistics
        input_array = np.array(self.input_data)
        print(f"\\n🔋 INPUT DATA RANGE:")
        print(f"   ⚡ Voltage: {input_array[:, 0].min():.3f} - {input_array[:, 0].max():.3f} V")
        print(f"   🔄 Current: {input_array[:, 1].min():.3f} - {input_array[:, 1].max():.3f} A")
        print(f"   🔋 SOH: {input_array[:, 2].min():.3f} - {input_array[:, 2].max():.3f}")
        print(f"   📊 Q_c: {input_array[:, 3].min():.0f} - {input_array[:, 3].max():.0f}")
        
        # Create visualization
        self.create_visualization()
        
        # Performance assessment
        print(f"\\n🏆 PERFORMANCE ASSESSMENT:")
        if mae < 0.05:
            print("   🌟 EXCELLENT: MAE < 0.05 - Very high accuracy")
        elif mae < 0.1:
            print("   ✅ GOOD: MAE < 0.1 - Good accuracy for SOC prediction")
        elif mae < 0.2:
            print("   ⚠️ ACCEPTABLE: MAE < 0.2 - Reasonable accuracy")
        else:
            print("   ❌ POOR: MAE >= 0.2 - Accuracy needs improvement")
        
        if success_rate >= 95:
            print("   🔗 EXCELLENT: Communication reliability > 95%")
        elif success_rate >= 80:
            print("   ✅ GOOD: Communication reliability > 80%")
        else:
            print("   ⚠️ POOR: Communication reliability < 80%")
        
        print("=" * 80)
    
    def create_visualization(self):
        """Create comprehensive visualization of results"""
        if len(self.successful_predictions) < 5:
            print("⚠️ Not enough data for visualization")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Arduino LSTM SOC Prediction - Live MAE Test Results', fontsize=16)
            
            # Plot 1: Predictions vs Ground Truth
            axes[0,0].scatter(self.ground_truth_values, self.successful_predictions, alpha=0.6, s=30)
            min_soc = min(min(self.ground_truth_values), min(self.successful_predictions))
            max_soc = max(max(self.ground_truth_values), max(self.successful_predictions))
            axes[0,0].plot([min_soc, max_soc], [min_soc, max_soc], 'r--', label='Perfect Prediction')
            axes[0,0].set_xlabel('True SOC')
            axes[0,0].set_ylabel('Predicted SOC')
            axes[0,0].set_title('Predictions vs Ground Truth')
            axes[0,0].legend()
            axes[0,0].grid(True)
            
            # Plot 2: Error over time
            axes[0,1].plot(self.timestamps, self.absolute_errors, 'orange', alpha=0.7)
            axes[0,1].axhline(y=np.mean(self.absolute_errors), color='red', linestyle='--', label=f'Mean Error: {np.mean(self.absolute_errors):.4f}')
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('Absolute Error')
            axes[0,1].set_title('Prediction Error Over Time')
            axes[0,1].legend()
            axes[0,1].grid(True)
            
            # Plot 3: Error histogram
            axes[1,0].hist(self.absolute_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1,0].axvline(x=np.mean(self.absolute_errors), color='red', linestyle='--', label=f'MAE: {np.mean(self.absolute_errors):.4f}')
            axes[1,0].set_xlabel('Absolute Error')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Error Distribution')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # Plot 4: Rolling MAE
            window_size = min(10, len(self.absolute_errors))
            rolling_mae = []
            for i in range(window_size-1, len(self.absolute_errors)):
                mae_window = np.mean(self.absolute_errors[i-window_size+1:i+1])
                rolling_mae.append(mae_window)
            
            rolling_times = self.timestamps[window_size-1:]
            axes[1,1].plot(rolling_times, rolling_mae, 'green', linewidth=2)
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('Rolling MAE')
            axes[1,1].set_title(f'Rolling MAE (Window: {window_size})')
            axes[1,1].grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print("📊 Visualization created and displayed")
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
    
    def close(self):
        """Clean up resources"""
        if self.arduino:
            try:
                self.arduino.close()
                print("🔌 Arduino connection closed")
            except:
                pass

def main():
    """Main test execution"""
    print("🎯 ARDUINO LSTM SOC PREDICTION - LIVE MAE VERIFICATION TEST")
    print("=" * 70)
    print("📊 This test verifies Arduino LSTM predictions against ground truth data")
    print("🔌 Hardware disconnection detection included")
    print("📈 Comprehensive MAE analysis with visualization")
    print("=" * 70)
    
    # Create test instance
    mae_test = FinalArduinoMAETest()
    
    try:
        # Setup ground truth data
        if not mae_test.setup_ground_truth_data():
            print("❌ Failed to setup ground truth data")
            return
        
        # Connect to Arduino
        if not mae_test.connect_arduino():
            print("❌ Failed to connect to Arduino")
            return
        
        # Run the MAE test
        mae_test.run_mae_test(num_samples=50)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
    finally:
        mae_test.close()

if __name__ == "__main__":
    main()
