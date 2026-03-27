"""
Simple Continuous Arduino MAE Test
Based on working_arduino_mae_test_fixed.py but runs continuously
"""

import serial
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

class SimpleContinuousMAETest:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        
        # Results storage
        self.predictions = []
        self.ground_truth = []
        self.errors = []
        self.communication_failures = 0
        self.test_count = 0
        
        # Data
        self.test_data = None
        self.scaler = None
        self.data_index = 0
        
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
            
            # Check required columns
            required_cols = ['Voltage[V]', 'Current[A]', 'SOH', 'Q_c', 'SOC_ZHU']
            missing_cols = [col for col in required_cols if col not in df_c19.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return False
            
            print("✅ All required columns found")
            
            # Create scaler with C19 data
            print("🔧 Creating StandardScaler with C19 data...")
            features_data = df_c19[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].copy()
            features_data = features_data.dropna()
            print(f"   📊 Using {len(features_data)} clean samples for scaling")
            
            self.scaler = StandardScaler()
            self.scaler.fit(features_data)
            
            # Prepare test data (larger sample for continuous testing)
            step_size = max(1, len(df_c19) // 200)  # Get about 200 samples
            test_indices = range(0, len(df_c19), step_size)
            
            test_df = df_c19.iloc[test_indices].copy()
            test_df = test_df.dropna(subset=required_cols)
            
            # Extract features and target
            test_features = test_df[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].values
            test_soc = test_df['SOC_ZHU'].values
            
            self.test_data = {
                'features_raw': test_features,
                'soc_true': test_soc
            }
            
            print(f"✅ Prepared {len(test_features)} test samples")
            print(f"   SOC range: {test_soc.min():.3f} - {test_soc.max():.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup ground truth data: {e}")
            return False
    
    def connect_arduino(self):
        """Connect to Arduino"""
        print(f"🔌 Connecting to Arduino on {self.port}...")
        
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(3)
            print("✅ Arduino connected")
            
            # Clear and reset
            self.arduino.flushInput()
            self.arduino.write(b'RESET\n')
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Arduino: {e}")
            return False
    
    def get_arduino_prediction(self, voltage, current, soh, q_c):
        """Get SOC prediction from Arduino"""
        try:
            # Clear input buffer
            self.arduino.flushInput()
            
            # Send prediction request
            data_str = f"{voltage:.6f},{current:.6f},{soh:.6f},{q_c:.6f}\n"
            self.arduino.write(data_str.encode())
            time.sleep(0.3)
            
            # Read response
            response = self.arduino.readline().decode().strip()
            
            # Parse SOC prediction
            try:
                soc_val = float(response)
                if 0.0 <= soc_val <= 1.0:
                    return soc_val
            except ValueError:
                pass
            
            return None
            
        except Exception as e:
            return None
    
    def get_next_sample(self):
        """Get next test sample (cycles through data)"""
        if self.data_index >= len(self.test_data['features_raw']):
            self.data_index = 0  # Loop back to beginning
            
        features = self.test_data['features_raw'][self.data_index]
        soc_true = self.test_data['soc_true'][self.data_index]
        self.data_index += 1
        
        return features, soc_true
    
    def run_continuous_test(self):
        """Run continuous MAE test"""
        print(f"\n🎯 CONTINUOUS ARDUINO MAE TEST")
        print("=" * 50)
        print("📊 Arduino predicts SOC, MAE calculated vs ground truth")
        print("⏹️  Press Ctrl+C to stop when you want")
        print("=" * 50)
        
        try:
            while True:
                self.test_count += 1
                
                # Get next test sample
                features, soc_true = self.get_next_sample()
                voltage, current, soh, q_c = features
                
                # Get Arduino prediction
                soc_pred = self.get_arduino_prediction(voltage, current, soh, q_c)
                
                if soc_pred is not None:
                    # Calculate error
                    error = abs(soc_pred - soc_true)
                    
                    # Store results
                    self.predictions.append(soc_pred)
                    self.ground_truth.append(soc_true)
                    self.errors.append(error)
                    
                    # Calculate running MAE
                    running_mae = np.mean(self.errors)
                    recent_mae = np.mean(self.errors[-10:]) if len(self.errors) >= 10 else running_mae
                    
                    # Display results
                    print(f"[{self.test_count:4d}] V:{voltage:.3f} I:{current:.2f} → SOC:{soc_pred:.4f} | True:{soc_true:.4f} | Error:{error:.4f} | MAE:{running_mae:.4f} | MAE-10:{recent_mae:.4f}")
                    
                else:
                    self.communication_failures += 1
                    print(f"[{self.test_count:4d}] ❌ Communication failed (Total failures: {self.communication_failures})")
                
                # Small delay
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Test stopped by user (Ctrl+C)")
            self.show_final_results()
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
        finally:
            if self.arduino:
                self.arduino.close()
                print("🔌 Arduino connection closed")
    
    def show_final_results(self):
        """Show final results"""
        if not self.predictions:
            print("❌ No successful predictions recorded")
            return
        
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        errors = np.array(self.errors)
        
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        total_attempts = len(self.predictions) + self.communication_failures
        success_rate = len(self.predictions) / total_attempts * 100 if total_attempts > 0 else 0
        
        print("\n" + "=" * 60)
        print("🎯 FINAL CONTINUOUS TEST RESULTS")
        print("=" * 60)
        
        print("📊 TEST STATISTICS:")
        print(f"   ✅ Successful predictions: {len(self.predictions)}")
        print(f"   ❌ Communication failures: {self.communication_failures}")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        print(f"   ⏱️ Total test samples: {self.test_count}")
        
        print("\n🎯 ACCURACY METRICS:")
        print(f"   📏 Final MAE: {mae:.6f}")
        print(f"   📐 Final RMSE: {rmse:.6f}")
        print(f"   📊 Max Error: {max_error:.6f}")
        print(f"   📊 Min Error: {min_error:.6f}")
        
        print("\n📈 PREDICTION RANGES:")
        print(f"   🎯 Predicted SOC: {predictions.min():.3f} - {predictions.max():.3f}")
        print(f"   📊 True SOC: {ground_truth.min():.3f} - {ground_truth.max():.3f}")
        
        print("=" * 60)
    
    def start(self):
        """Start the test"""
        if not self.setup_ground_truth_data():
            return False
        
        if not self.connect_arduino():
            return False
        
        self.run_continuous_test()
        return True

if __name__ == "__main__":
    print("🚀 Starting Simple Continuous Arduino MAE Test...")
    test = SimpleContinuousMAETest()
    test.start()
