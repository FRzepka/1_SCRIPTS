"""
Arduino MAE Test - Corrected Version
Live hardware verification test for Arduino LSTM SOC predictions with proper labeling
Uses SOC_ZHU from training data as ground truth (SOC_GT)
"""

import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import gc

class ArduinoMAETestCorrected:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
          # Test results with proper naming
        self.soc_pred_values = []  # Arduino predictions
        self.soc_gt_values = []    # Ground truth SOC_ZHU values
        self.absolute_errors = []
        self.timestamps = []
        self.input_data = []
        self.communication_failures = 0
        
        # Ground truth data
        self.test_data = None
        self.scaler = None
        self.feature_names = None  # Store feature names for sklearn
        
    def load_all_training_data(self):
        """Load all training data for proper scaling (like in original training)"""
        try:
            base_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
            cell_names = ['MGFarm_18650_C01', 'MGFarm_18650_C03', 'MGFarm_18650_C05', 'MGFarm_18650_C07', 
                         'MGFarm_18650_C11', 'MGFarm_18650_C17', 'MGFarm_18650_C19', 'MGFarm_18650_C21', 'MGFarm_18650_C23']
            
            print(f"   🔍 Loading data from {len(cell_names)} cells for scaling...")
            all_features = []
            loaded_cells = 0
            
            for cell_name in cell_names:
                cell_path = os.path.join(base_path, cell_name, "df.parquet")
                if os.path.exists(cell_path):
                    try:
                        df = pd.read_parquet(cell_path)
                        if all(col in df.columns for col in ['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']):
                            features = df[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].dropna()
                            all_features.append(features)
                            loaded_cells += 1
                            print(f"   ✅ {cell_name}: {len(features)} samples")
                        else:
                            print(f"   ⚠️ {cell_name}: Missing required columns")
                    except Exception as e:
                        print(f"   ❌ {cell_name}: {e}")
                else:
                    print(f"   ❌ {cell_name}: File not found")
            
            if loaded_cells > 0:
                combined_features = pd.concat(all_features, ignore_index=True)
                print(f"   ✅ Combined {loaded_cells} cells with {len(combined_features)} total samples")
                return combined_features
            else:
                print(f"   ❌ No cells could be loaded")
                return None
                
        except Exception as e:
            print(f"   ❌ Error loading training data: {e}")
            return None
        
    def setup_ground_truth_data(self):
        """Setup ground truth data from C19 cell with correct column names"""
        print("📊 Setting up ground truth data from MGFarm C19...")
        
        try:
            # Load C19 data  
            c19_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
            
            if not os.path.exists(c19_path):
                print(f"❌ C19 data file not found: {c19_path}")
                return False
                
            df_c19 = pd.read_parquet(c19_path)
            print(f"✅ Loaded {len(df_c19)} samples from C19")
            
            # Define correct column mapping
            column_mapping = {
                'Voltage[V]': 'voltage',
                'Current[A]': 'current',
                'SOH': 'soh',
                'Q_c': 'q_c',
                'SOC_ZHU': 'soc_target'
            }
            
            # Check if required columns exist
            required_cols = list(column_mapping.keys())
            missing_cols = [col for col in required_cols if col not in df_c19.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                print(f"📋 Available columns: {df_c19.columns.tolist()}")
                return False
            
            print("✅ All required columns found")
              # Create scaler with ALL training data (like in original training)
            print("🔧 Creating StandardScaler with ALL training data...")
            scaler_data = self.load_all_training_data()
            if scaler_data is None:
                # Fallback to C19 only if all data loading fails
                print("⚠️ Fallback: Using only C19 data for scaling")
                features_data = df_c19[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].dropna()
            else:
                features_data = scaler_data
                
            print(f"   📊 Using {len(features_data)} samples for scaling")
            
            # Fit scaler
            self.scaler = StandardScaler()
            self.scaler.fit(features_data)
            
            print(f"📈 Scaler Statistics:")
            print(f"   Mean: {self.scaler.mean_}")
            print(f"   Std:  {self.scaler.scale_}")
            
            # Prepare test data (sample from C19 for variety)
            # Use every 10000th sample to get good variety
            step_size = max(1, len(df_c19) // 100)  # Get about 100 samples
            test_indices = range(0, len(df_c19), step_size)[:50]  # Limit to 50 samples
            
            test_df = df_c19.iloc[test_indices].copy()
            test_df = test_df.dropna(subset=required_cols)  # Remove any NaN values
            
            # Extract features and target
            test_features = test_df[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].values
            test_soc = test_df['SOC_ZHU'].values
            
            self.test_data = {
                'features_raw': test_features,
                'soc_true': test_soc,
                'indices': test_indices[:len(test_features)]
            }
            
            print(f"✅ Prepared {len(test_features)} test samples")
            print(f"   SOC range: {test_soc.min():.3f} - {test_soc.max():.3f}")
            print(f"   Voltage range: {test_features[:, 0].min():.3f} - {test_features[:, 0].max():.3f} V")
            print(f"   Current range: {test_features[:, 1].min():.3f} - {test_features[:, 1].max():.3f} A")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup ground truth data: {e}")
            import traceback
            traceback.print_exc()
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
      def get_arduino_prediction(self, voltage, current, soh, q_c, max_attempts=3):
        """Get SOC prediction from Arduino"""
        if self.arduino is None:
            return None
            
        for attempt in range(max_attempts):
            try:
                # Scale features using DataFrame with proper column names to avoid sklearn warnings
                features_df = pd.DataFrame([[voltage, current, soh, q_c]], columns=self.feature_names)
                features_scaled = self.scaler.transform(features_df)
                
                # Send scaled data to Arduino
                data_str = f"{features_scaled[0,0]:.6f},{features_scaled[0,1]:.6f},{features_scaled[0,2]:.6f},{features_scaled[0,3]:.6f}\n"
                self.arduino.write(data_str.encode())
                
                # Wait for response
                time.sleep(0.1)
                
                if self.arduino.in_waiting > 0:
                    response = self.arduino.readline().decode().strip()
                    try:
                        soc_pred = float(response)
                        if 0.0 <= soc_pred <= 1.0:  # Valid SOC range
                            return soc_pred
                    except ValueError:
                        pass
                        
            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f"⚠️ Communication error: {e}")
                    
        self.communication_failures += 1
        return None
                
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
    
    def run_mae_test(self, num_samples=30):
        """Run the main MAE test - CONTINUOUS VERSION"""
        print(f"\n🚀 STARTING CONTINUOUS ARDUINO MAE TEST")
        print("=" * 60)
        print(f"📊 Continuous predictions with ground truth comparison")
        print("🔌 You can unplug the Arduino cable during the test to verify disconnection detection")
        print("⏹️ Press Ctrl+C to stop the test when you want")
        print("-" * 60)
        
        start_time = time.time()
        data_index = 0
        
        try:
            while True:  # CONTINUOUS LOOP
                # Cycle through data
                if data_index >= len(self.test_data['features_raw']):
                    data_index = 0  # Loop back to start
                
                i = data_index
                data_index += 1
                # Get test sample
                voltage = self.test_data['features_raw'][i, 0]
                current = self.test_data['features_raw'][i, 1]
                soh = self.test_data['features_raw'][i, 2]
                q_c = self.test_data['features_raw'][i, 3]
                true_soc = self.test_data['soc_true'][i]
                
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
                        self.input_data.append([voltage, current, soh, q_c])                        # Progress update - show running count with clear labels
                        total_predictions = len(self.successful_predictions)
                        current_mae = np.mean(self.absolute_errors[-5:]) if len(self.absolute_errors) >= 5 else np.mean(self.absolute_errors)
                        running_mae = np.mean(self.absolute_errors)
                        print(f"[{total_predictions:4d}] V:{voltage:.3f} I:{current:.2f} → Arduino: {pred_soc:.4f} | Ground Truth: {true_soc:.4f} | Error: {error:.4f} | MAE: {running_mae:.4f} | MAE-5: {current_mae:.4f}")
                
                    else:
                        self.communication_failures += 1
                        total_attempts = len(self.successful_predictions) + self.communication_failures
                        print(f"[{total_attempts:4d}] ❌ Communication failure (total: {self.communication_failures})")
                        
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
                    total_attempts = len(self.successful_predictions) + self.communication_failures + 1
                    print(f"[{total_attempts:4d}] ❌ Unexpected error: {e}")
                    self.communication_failures += 1
                
                # Small delay between predictions
                time.sleep(0.3)
                
        except KeyboardInterrupt:
            print("\n⏹️ Test stopped by user")
        
        # Show results
        self.show_results()
    
    def show_results(self):
        """Display comprehensive test results"""
        print("\n" + "=" * 80)
        print("🎯 ARDUINO LSTM SOC PREDICTION - MAE TEST RESULTS")
        print("=" * 80)
        
        if len(self.successful_predictions) == 0:
            print("❌ No successful predictions recorded!")
            print(f"💥 Communication failures: {self.communication_failures}")
            print("\n🔧 TROUBLESHOOTING:")
            print("   1. Check Arduino connection on COM13")
            print("   2. Verify correct Arduino program is uploaded")
            print("   3. Check if Arduino is responding to INFO commands")
            print("   4. Try uploading the fixed Arduino program")
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
        print(f"   ⏱️ Total test duration: {self.timestamps[-1]:.1f} seconds")
        
        print(f"\n🎯 ACCURACY METRICS:")
        print(f"   📏 MAE (Mean Absolute Error): {mae:.6f}")
        print(f"   📐 RMSE (Root Mean Square Error): {rmse:.6f}")
        print(f"   📊 Maximum Error: {max_error:.6f}")
        print(f"   📊 Minimum Error: {min_error:.6f}")
        print(f"   📊 Error Standard Deviation: {np.std(self.absolute_errors):.6f}")
        
        print(f"\n📈 SOC PREDICTION RANGE:")
        print(f"   🎯 Predicted SOC: {np.min(self.successful_predictions):.3f} - {np.max(self.successful_predictions):.3f}")
        print(f"   📊 True SOC: {np.min(self.ground_truth_values):.3f} - {np.max(self.ground_truth_values):.3f}")
        
        # Input data statistics
        input_array = np.array(self.input_data)
        print(f"\n🔋 INPUT DATA RANGE:")
        print(f"   ⚡ Voltage: {input_array[:, 0].min():.3f} - {input_array[:, 0].max():.3f} V")
        print(f"   🔄 Current: {input_array[:, 1].min():.3f} - {input_array[:, 1].max():.3f} A")
        print(f"   🔋 SOH: {input_array[:, 2].min():.3f} - {input_array[:, 2].max():.3f}")
        print(f"   📊 Q_c: {input_array[:, 3].min():.0f} - {input_array[:, 3].max():.0f}")
        
        # Create visualization
        self.create_visualization()
        
        # Performance assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
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
            
            # Plot 1: Predictions vs Ground Truth            axes[0,0].scatter(self.ground_truth_values, self.successful_predictions, alpha=0.6, s=30)
            min_soc = min(min(self.ground_truth_values), min(self.successful_predictions))
            max_soc = max(max(self.ground_truth_values), max(self.successful_predictions))
            axes[0,0].plot([min_soc, max_soc], [min_soc, max_soc], 'r--', label='Perfect Prediction')
            axes[0,0].set_xlabel('Ground Truth SOC')
            axes[0,0].set_ylabel('Arduino Predicted SOC')
            axes[0,0].set_title('Arduino Predictions vs Ground Truth')
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
            axes[1,0].hist(self.absolute_errors, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1,0].axvline(x=np.mean(self.absolute_errors), color='red', linestyle='--', label=f'MAE: {np.mean(self.absolute_errors):.4f}')
            axes[1,0].set_xlabel('Absolute Error')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Error Distribution')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # Plot 4: Rolling MAE
            window_size = min(5, len(self.absolute_errors))
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
    """Main function to run the corrected Arduino MAE test"""
    print("🚀 Arduino LSTM SOC Prediction - Corrected MAE Test")
    print("🔧 Using proper SOC_ZHU labeling from training data")
    
    tester = ArduinoMAETestCorrected(port='COM13', baudrate=115200)
    tester.run_continuous_test()
    
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
        mae_test.run_mae_test(num_samples=30)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mae_test.close()

if __name__ == "__main__":
    main()
