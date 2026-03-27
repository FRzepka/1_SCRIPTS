"""
Continuous Arduino MAE Test - Runs indefinitely until manually stopped
Simple continuous test: Arduino predicts SOC, calculate MAE vs ground truth
Press Ctrl+C to stop when you want
"""

import serial
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

class ContinuousArduinoMAETest:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        
        # Running results
        self.predictions = []
        self.ground_truth = []
        self.errors = []
        self.test_count = 0
        self.communication_failures = 0
        
        # Ground truth data
        self.test_data = None
        self.scaler = None
        self.data_index = 0  # Current position in test data
        
    def setup_data(self):
        """Setup ground truth data and scaler"""
        print("📊 Setting up ground truth data...")
        
        try:
            # Load C19 data
            c19_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
            
            if not os.path.exists(c19_path):
                print(f"❌ Data file not found")
                return False
                
            df_c19 = pd.read_parquet(c19_path)
            print(f"✅ Loaded {len(df_c19)} samples")
            
            # Check columns
            required_cols = ['Voltage[V]', 'Current[A]', 'SOH', 'Q_c', 'SOC_ZHU']
            if not all(col in df_c19.columns for col in required_cols):
                print(f"❌ Missing required columns")
                return False
            
            # Clean data
            clean_data = df_c19[required_cols].dropna()
            print(f"✅ {len(clean_data)} clean samples available")
            
            # Create scaler
            features_data = clean_data[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']]
            self.scaler = StandardScaler()
            self.scaler.fit(features_data)
            print("✅ Scaler created")
            
            # Store test data
            self.test_data = {
                'features': clean_data[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].values,
                'soc_true': clean_data['SOC_ZHU'].values
            }
            
            print(f"✅ Test data prepared: {len(self.test_data['features'])} samples")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def connect_arduino(self):
        """Connect to Arduino"""
        print(f"🔌 Connecting to Arduino on {self.port}...")
        
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(3)
            print("✅ Arduino connected")
            
            # Clear buffer and reset
            self.arduino.flushInput()
            self.arduino.write(b'RESET\n')
            time.sleep(1)
            
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def get_next_test_sample(self):
        """Get next test sample (cycles through data)"""
        if self.data_index >= len(self.test_data['features']):
            self.data_index = 0  # Loop back to beginning
        
        features = self.test_data['features'][self.data_index]
        soc_true = self.test_data['soc_true'][self.data_index]
        self.data_index += 1
        
        return features, soc_true
    
    def send_data_get_prediction(self, voltage, current, soh, q_c):
        """Send data to Arduino and get SOC prediction"""
        try:
            # Normalize input
            features = np.array([[voltage, current, soh, q_c]])
            features_scaled = self.scaler.transform(features)
            
            # Send to Arduino
            data_str = f"{features_scaled[0,0]:.6f},{features_scaled[0,1]:.6f},{features_scaled[0,2]:.6f},{features_scaled[0,3]:.6f}\n"
            self.arduino.write(data_str.encode())
            
            # Read response
            time.sleep(0.1)
            response = self.arduino.readline().decode().strip()
            
            # Parse SOC prediction
            try:
                soc_pred = float(response)
                if 0.0 <= soc_pred <= 1.0:
                    return soc_pred
            except:
                pass
            
            return None
            
        except Exception as e:
            return None
    
    def calculate_running_mae(self):
        """Calculate running MAE"""
        if len(self.errors) == 0:
            return 0.0
        return np.mean(self.errors)
    
    def calculate_recent_mae(self, window=10):
        """Calculate MAE for recent predictions"""
        if len(self.errors) == 0:
            return 0.0
        recent_errors = self.errors[-window:]
        return np.mean(recent_errors)
    
    def run_continuous_test(self):
        """Run continuous MAE test until manually stopped"""
        print("\n🎯 CONTINUOUS ARDUINO MAE TEST")
        print("=" * 50)
        print("📊 Arduino predicts SOC, MAE calculated vs ground truth")
        print("⏹️  Press Ctrl+C to stop when you want")
        print("=" * 50)
        
        try:
            while True:
                # Get next test sample
                features, soc_true = self.get_next_test_sample()
                voltage, current, soh, q_c = features
                
                # Get Arduino prediction
                soc_pred = self.send_data_get_prediction(voltage, current, soh, q_c)
                
                self.test_count += 1
                
                if soc_pred is not None:
                    # Calculate error
                    error = abs(soc_pred - soc_true)
                    
                    # Store results
                    self.predictions.append(soc_pred)
                    self.ground_truth.append(soc_true)
                    self.errors.append(error)
                    
                    # Calculate running metrics
                    running_mae = self.calculate_running_mae()
                    recent_mae = self.calculate_recent_mae(10)
                    
                    # Display results
                    print(f"[{self.test_count:4d}] V:{voltage:.3f} I:{current:.2f} → SOC:{soc_pred:.4f} | True:{soc_true:.4f} | Error:{error:.4f} | MAE:{running_mae:.4f} | MAE-10:{recent_mae:.4f}")
                    
                else:
                    self.communication_failures += 1
                    print(f"[{self.test_count:4d}] ❌ Communication failed (Total failures: {self.communication_failures})")
                
                # Small delay between tests
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Test stopped by user (Ctrl+C)")
            self.display_final_results()
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
        finally:
            if self.arduino:
                self.arduino.close()
                print("🔌 Arduino connection closed")
    
    def display_final_results(self):
        """Display final test results"""
        if not self.predictions:
            print("❌ No successful predictions recorded")
            return
        
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        errors = np.array(self.errors)
        
        final_mae = np.mean(errors)
        final_rmse = np.sqrt(np.mean(errors**2))
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
        print(f"   📏 Final MAE: {final_mae:.6f}")
        print(f"   📐 Final RMSE: {final_rmse:.6f}")
        print(f"   📊 Max Error: {max_error:.6f}")
        print(f"   📊 Min Error: {min_error:.6f}")
        
        print("\n📈 PREDICTION RANGES:")
        print(f"   🎯 Predicted SOC: {predictions.min():.3f} - {predictions.max():.3f}")
        print(f"   📊 True SOC: {ground_truth.min():.3f} - {ground_truth.max():.3f}")
        
        print("=" * 60)
    
    def start(self):
        """Start the continuous test"""
        if not self.setup_data():
            return False
        
        if not self.connect_arduino():
            return False
        
        self.run_continuous_test()
        return True

if __name__ == "__main__":
    print("🚀 Starting Continuous Arduino MAE Test...")
    test = ContinuousArduinoMAETest()
    test.start()
