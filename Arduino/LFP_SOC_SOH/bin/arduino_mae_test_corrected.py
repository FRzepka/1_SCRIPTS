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
        
    def load_all_training_data(self):
        """Load all training data for proper scaling (matching training procedure)"""
        try:
            base_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
            
            # Use same cells as in training (from BMS_SOC_LSTM_stateful_1.2.4_Train.py)
            cell_names = [
                'MGFarm_18650_C01', 'MGFarm_18650_C03', 'MGFarm_18650_C05',
                'MGFarm_18650_C11', 'MGFarm_18650_C17', 'MGFarm_18650_C23',
                'MGFarm_18650_C07', 'MGFarm_18650_C19', 'MGFarm_18650_C21'
            ]
            
            print(f"   🔍 Loading {len(cell_names)} training cells for scaling...")
            all_features = []
            loaded_cells = 0
            
            # Features matching training: ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
            feature_cols = ['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_c']
            
            for cell_name in cell_names:
                cell_path = os.path.join(base_path, cell_name, "df.parquet")
                if os.path.exists(cell_path):
                    try:
                        df = pd.read_parquet(cell_path)
                        
                        # Check for training features
                        if all(col in df.columns for col in feature_cols):
                            features = df[feature_cols].dropna()
                            all_features.append(features)
                            loaded_cells += 1
                            print(f"   ✅ {cell_name}: {len(features)} samples")
                        else:
                            print(f"   ⚠️ {cell_name}: Missing required columns")
                            
                    except Exception as e:
                        print(f"   ❌ {cell_name}: Load error - {e}")
                        
            if loaded_cells == 0:
                print("   ⚠️ No training data loaded!")
                return None
                
            # Combine all features
            combined_features = pd.concat(all_features, ignore_index=True)
            print(f"   ✅ Combined: {len(combined_features)} total samples from {loaded_cells} cells")
            
            return combined_features
            
        except Exception as e:
            print(f"   ❌ Error loading training data: {e}")
            return None
    
    def setup_ground_truth_data(self):
        """Setup ground truth data from C19 cell with correct column names"""
        print("📊 Setting up ground truth data from MGFarm C19...")
        print("   Using SOC_ZHU as ground truth (SOC_GT)")
        
        try:
            # Load C19 data  
            c19_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
            
            if not os.path.exists(c19_path):
                print(f"❌ C19 data file not found: {c19_path}")
                return False
                
            df_c19 = pd.read_parquet(c19_path)
            print(f"✅ Loaded {len(df_c19)} samples from C19")
            
            # Check columns - using training feature names
            feature_cols = ['Voltage[V]', 'Current[A]', 'SOH_ZHU', 'Q_c']
            target_col = 'SOC_ZHU'  # This is the ground truth from training
            
            required_cols = feature_cols + [target_col]
            missing_cols = [col for col in required_cols if col not in df_c19.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                print(f"📋 Available columns: {df_c19.columns.tolist()}")
                return False
            
            print("✅ All required columns found")
            
            # Check SOC_ZHU values
            soc_values = df_c19[target_col].dropna()
            print(f"📈 SOC_ZHU statistics:")
            print(f"   Range: {soc_values.min():.6f} - {soc_values.max():.6f}")
            print(f"   Mean: {soc_values.mean():.6f}")
            print(f"   Std: {soc_values.std():.6f}")
            print(f"   Non-null values: {len(soc_values)}/{len(df_c19)}")
              # Create scaler with ALL training data (matching training procedure)
            print("🔧 Creating StandardScaler with ALL training data...")
            scaler_data = self.load_all_training_data()
            
            if scaler_data is None:
                print("⚠️ Fallback: Using only C19 data for scaling")
                features_data = df_c19[feature_cols].dropna()
            else:
                features_data = scaler_data
                
            print(f"   📊 Using {len(features_data)} samples for scaling")
            
            # Fit scaler with proper feature names to eliminate sklearn warnings
            self.scaler = StandardScaler()
            self.scaler.fit(features_data)  # features_data is already a DataFrame with column names
            
            # Store feature names for consistent usage
            self.feature_names = feature_cols
            
            print(f"📈 Scaler Statistics:")
            print(f"   Mean: {self.scaler.mean_}")
            print(f"   Std:  {self.scaler.scale_}")
            
            # Prepare test data (sample from C19 for variety)
            step_size = max(1, len(df_c19) // 100)  # Get about 100 samples
            test_indices = range(0, len(df_c19), step_size)[:50]  # Limit to 50 samples
            
            test_df = df_c19.iloc[test_indices].copy()
            test_df = test_df.dropna(subset=required_cols)  # Remove any NaN values
            
            # Extract features and target
            test_features = test_df[feature_cols].values
            test_soc_gt = test_df[target_col].values  # SOC_ZHU as ground truth
            
            self.test_data = {
                'features_raw': test_features,
                'soc_gt': test_soc_gt,  # Ground truth SOC values
                'indices': test_indices[:len(test_features)]
            }
            
            print(f"✅ Prepared {len(test_features)} test samples")
            print(f"   SOC_GT range: {test_soc_gt.min():.6f} - {test_soc_gt.max():.6f}")
            print(f"   Voltage range: {test_features[:, 0].min():.3f} - {test_features[:, 0].max():.3f} V")
            print(f"   Current range: {test_features[:, 1].min():.3f} - {test_features[:, 1].max():.3f} A")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup ground truth data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def connect_arduino(self):
        """Connect to Arduino"""
        print(f"🔌 Connecting to Arduino on {self.port}...")        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)  # Arduino reset time
            print("✅ Connected to Arduino successfully")
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
    
    def run_continuous_test(self):
        """Run continuous MAE test with proper labeling"""
        print("\n🚀 Starting Continuous Arduino MAE Test")
        print("=" * 60)
        print("📊 SOC_Pred = Arduino LSTM predictions")
        print("📊 SOC_GT = Ground truth SOC_ZHU values from training data")
        print("=" * 60)
        
        if not self.setup_ground_truth_data():
            print("❌ Failed to setup ground truth data")
            return False
            
        if not self.connect_arduino():
            print("❌ Failed to connect to Arduino")
            return False
        
        try:
            data_index = 0
            start_time = time.time()
            mae_window = []  # For rolling MAE calculation
            
            print("\n🔄 Starting continuous predictions (Ctrl+C to stop)...")
            print(f"{'#':<4} {'V':<6} {'I':<6} {'SOH':<6} {'Q_c':<6} {'SOC_Pred':<10} {'SOC_GT':<10} {'Error':<8} {'MAE':<8}")
            print("-" * 70)
            
            while True:  # CONTINUOUS LOOP
                # Loop through data continuously
                if data_index >= len(self.test_data['features_raw']):
                    data_index = 0  # Loop back to start
                    print(f"\n🔄 Looped back to start of dataset")
                
                i = data_index
                
                # Get test data
                voltage = self.test_data['features_raw'][i, 0]
                current = self.test_data['features_raw'][i, 1]
                soh = self.test_data['features_raw'][i, 2]  # SOH_ZHU from training
                q_c = self.test_data['features_raw'][i, 3]
                soc_gt = self.test_data['soc_gt'][i]  # SOC_ZHU ground truth
                
                # Get Arduino prediction
                soc_pred = self.get_arduino_prediction(voltage, current, soh, q_c)
                
                if soc_pred is not None:
                    # Calculate error
                    error = abs(soc_pred - soc_gt)
                    
                    # Store results with correct naming
                    self.soc_pred_values.append(soc_pred)
                    self.soc_gt_values.append(soc_gt)
                    self.absolute_errors.append(error)
                    self.timestamps.append(time.time() - start_time)
                    self.input_data.append([voltage, current, soh, q_c])
                    
                    # Calculate running MAE
                    mae_window.append(error)
                    if len(mae_window) > 5:  # Keep last 5 predictions for rolling MAE
                        mae_window.pop(0)
                    running_mae = np.mean(mae_window)
                    
                    # Progress update with clear labels
                    total_predictions = len(self.soc_pred_values)
                    print(f"{total_predictions:<4} {voltage:<6.2f} {current:<6.2f} {soh:<6.3f} {q_c:<6.0f} {soc_pred:<10.6f} {soc_gt:<10.6f} {error:<8.6f} {running_mae:<8.6f}")
                    
                    # Show summary every 25 predictions
                    if total_predictions % 25 == 0:
                        overall_mae = np.mean(self.absolute_errors)
                        elapsed = time.time() - start_time
                        rate = total_predictions / elapsed
                        print(f"\n📊 Summary after {total_predictions} predictions:")
                        print(f"   Overall MAE: {overall_mae:.6f}")
                        print(f"   Prediction rate: {rate:.1f} pred/sec")
                        print(f"   Communication failures: {self.communication_failures}")
                        print("-" * 70)
                
                data_index += 1
                time.sleep(0.5)  # Control prediction rate
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️ Test stopped by user")
            self.generate_results()
        except Exception as e:
            print(f"\n❌ Test error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.arduino:
                self.arduino.close()
                print("🔌 Arduino connection closed")
    
    def generate_results(self):
        """Generate comprehensive test results with proper labeling"""
        if len(self.soc_pred_values) == 0:
            print("❌ No successful predictions to analyze")
            return
            
        print(f"\n📊 ARDUINO MAE TEST RESULTS")
        print("=" * 60)
        
        # Calculate metrics
        mae = np.mean(self.absolute_errors)
        rmse = np.sqrt(np.mean(np.array(self.absolute_errors) ** 2))
        max_error = np.max(self.absolute_errors)
        min_error = np.min(self.absolute_errors)
        
        total_attempts = len(self.soc_pred_values) + self.communication_failures
        success_rate = len(self.soc_pred_values) / total_attempts * 100
        
        # Data ranges
        pred_range = f"{np.min(self.soc_pred_values):.6f} - {np.max(self.soc_pred_values):.6f}"
        gt_range = f"{np.min(self.soc_gt_values):.6f} - {np.max(self.soc_gt_values):.6f}"
        
        # Input data statistics
        input_array = np.array(self.input_data)
        voltage_range = f"{input_array[:, 0].min():.3f} - {input_array[:, 0].max():.3f} V"
        current_range = f"{input_array[:, 1].min():.3f} - {input_array[:, 1].max():.3f} A"
        
        print(f"🎯 PREDICTION METRICS:")
        print(f"   Total successful predictions: {len(self.soc_pred_values)}")
        print(f"   Communication success rate: {success_rate:.1f}%")
        print(f"   Mean Absolute Error (MAE): {mae:.6f}")
        print(f"   Root Mean Square Error (RMSE): {rmse:.6f}")
        print(f"   Max error: {max_error:.6f}")
        print(f"   Min error: {min_error:.6f}")
        
        print(f"\n📈 DATA RANGES:")
        print(f"   SOC_Pred (Arduino): {pred_range}")
        print(f"   SOC_GT (Ground Truth): {gt_range}")
        print(f"   Input voltage: {voltage_range}")
        print(f"   Input current: {current_range}")
        
        # Performance classification
        if mae < 0.05:
            performance = "🟢 EXCELLENT"
        elif mae < 0.1:
            performance = "🟡 GOOD"
        elif mae < 0.2:
            performance = "🟠 ACCEPTABLE"
        else:
            performance = "🔴 POOR"
            
        print(f"\n🏆 PERFORMANCE CLASSIFICATION: {performance}")
        
        # Generate visualization
        self.create_visualization()
        
        print("✅ Test results saved and visualization generated")
    
    def create_visualization(self):
        """Create comprehensive visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. SOC Predictions vs Ground Truth
        ax1.plot(self.timestamps, self.soc_pred_values, 'b-', alpha=0.7, label='SOC_Pred (Arduino)', linewidth=1.5)
        ax1.plot(self.timestamps, self.soc_gt_values, 'r--', alpha=0.8, label='SOC_GT (Ground Truth)', linewidth=1)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('SOC')
        ax1.set_title('Arduino LSTM Predictions vs Ground Truth SOC_ZHU')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Absolute Error over time
        ax2.plot(self.timestamps, self.absolute_errors, 'g-', alpha=0.7, linewidth=1)
        ax2.axhline(y=np.mean(self.absolute_errors), color='orange', linestyle='--', 
                   label=f'Mean MAE: {np.mean(self.absolute_errors):.6f}')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Prediction Error over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Error histogram
        ax3.hist(self.absolute_errors, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(x=np.mean(self.absolute_errors), color='red', linestyle='--', 
                   label=f'MAE: {np.mean(self.absolute_errors):.6f}')
        ax3.set_xlabel('Absolute Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Scatter plot: Predicted vs Ground Truth
        ax4.scatter(self.soc_gt_values, self.soc_pred_values, alpha=0.6, s=30)
        min_val = min(min(self.soc_gt_values), min(self.soc_pred_values))
        max_val = max(max(self.soc_gt_values), max(self.soc_pred_values))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')
        ax4.set_xlabel('SOC_GT (Ground Truth)')
        ax4.set_ylabel('SOC_Pred (Arduino)')
        ax4.set_title('Prediction Accuracy Scatter Plot')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = int(time.time())
        filename = f"arduino_mae_test_corrected_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved as: {filename}")
        plt.show()

def main():
    """Main function to run the corrected Arduino MAE test"""
    print("🚀 Arduino LSTM SOC Prediction - Corrected MAE Test")
    print("🔧 Using proper SOC_ZHU labeling from training data")
    
    tester = ArduinoMAETestCorrected(port='COM13', baudrate=115200)
    tester.run_continuous_test()

if __name__ == "__main__":
    main()
