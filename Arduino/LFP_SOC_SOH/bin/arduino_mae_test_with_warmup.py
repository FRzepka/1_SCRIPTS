"""
Arduino MAE Test with LSTM Warm-up Period
Addresses the LSTM settling time issue by including a warm-up phase
"""

import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

class ArduinoMAETestWithWarmup:
    def __init__(self, port='COM13', baudrate=115200, warmup_steps=10):
        self.port = port
        self.baudrate = baudrate
        self.warmup_steps = warmup_steps
        self.arduino = None
        
        # Test results
        self.warmup_predictions = []
        self.test_predictions = []
        self.test_ground_truth = []
        self.test_errors = []
        self.timestamps = []
        self.communication_failures = 0
        
        # Ground truth data
        self.test_data = None
        self.scaler = None
        
    def setup_ground_truth_data(self):
        """Setup ground truth data from C19 cell"""
        print("📊 Setting up ground truth data from MGFarm C19...")
        
        try:
            c19_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
            
            if not os.path.exists(c19_path):
                print(f"❌ C19 data file not found")
                return False
                
            df_c19 = pd.read_parquet(c19_path)
            print(f"✅ Loaded {len(df_c19)} samples from C19")
            
            # Check required columns
            required_cols = ['Voltage[V]', 'Current[A]', 'SOH', 'Q_c', 'SOC_ZHU']
            missing_cols = [col for col in required_cols if col not in df_c19.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return False
            
            # Create scaler with C19 data
            print("🔧 Creating StandardScaler with C19 data...")
            features_data = df_c19[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].dropna()
            
            self.scaler = StandardScaler()
            self.scaler.fit(features_data)
            
            # Prepare sequential test data (consecutive samples for LSTM continuity)
            print("📈 Preparing sequential test data for LSTM...")
            
            # Find a good continuous sequence
            start_idx = len(df_c19) // 2  # Start from middle of dataset
            total_samples = self.warmup_steps + 30  # Warmup + test samples
            
            if start_idx + total_samples > len(df_c19):
                start_idx = len(df_c19) - total_samples
            
            sequence_df = df_c19.iloc[start_idx:start_idx + total_samples].copy()
            sequence_df = sequence_df.dropna(subset=required_cols)
            
            if len(sequence_df) < total_samples:
                print(f"⚠️ Only {len(sequence_df)} clean sequential samples available")
            
            # Extract features and targets
            features = sequence_df[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].values
            soc_true = sequence_df['SOC_ZHU'].values
            
            self.test_data = {
                'features': features,
                'soc_true': soc_true,
                'warmup_end': min(self.warmup_steps, len(features)),
                'test_start': min(self.warmup_steps, len(features))
            }
            
            print(f"✅ Prepared {len(features)} sequential samples")
            print(f"   🔥 Warmup samples: {self.test_data['warmup_end']}")
            print(f"   🎯 Test samples: {len(features) - self.test_data['warmup_end']}")
            print(f"   📊 SOC range: {soc_true.min():.3f} - {soc_true.max():.3f}")
            
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
            
            # Clear and test communication
            self.arduino.flushInput()
            self.arduino.write(b'RESET\n')
            time.sleep(1)
            
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def send_data_get_prediction(self, voltage, current, soh, q_c):
        """Send data to Arduino and get SOC prediction"""
        try:
            # Normalize input data
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
            print(f"⚠️ Communication error: {e}")
            return None
    
    def run_warmup_phase(self):
        """Run warmup phase to let LSTM settle"""
        print(f"\n🔥 WARMUP PHASE - Letting LSTM settle ({self.warmup_steps} steps)")
        print("=" * 60)
        
        warmup_end = self.test_data['warmup_end']
        
        for i in range(warmup_end):
            features = self.test_data['features'][i]
            voltage, current, soh, q_c = features
            
            soc_pred = self.send_data_get_prediction(voltage, current, soh, q_c)
            
            if soc_pred is not None:
                self.warmup_predictions.append(soc_pred)
                print(f"[W{i+1:2d}/{warmup_end}] V:{voltage:.3f} I:{current:.2f} → SOC: {soc_pred:.4f} (warming up...)")
            else:
                print(f"[W{i+1:2d}/{warmup_end}] Communication failed")
                self.communication_failures += 1
            
            time.sleep(0.5)  # Give LSTM time to process
        
        print(f"🔥 Warmup completed. LSTM should now be settled.")
    
    def run_test_phase(self):
        """Run actual test phase with settled LSTM"""
        print(f"\n🎯 TEST PHASE - Measuring accuracy with settled LSTM")
        print("=" * 60)
        
        test_start = self.test_data['test_start']
        total_samples = len(self.test_data['features'])
        
        for i in range(test_start, total_samples):
            features = self.test_data['features'][i]
            voltage, current, soh, q_c = features
            soc_true = self.test_data['soc_true'][i]
            
            start_time = time.time()
            soc_pred = self.send_data_get_prediction(voltage, current, soh, q_c)
            response_time = time.time() - start_time
            
            if soc_pred is not None:
                error = abs(soc_pred - soc_true)
                self.test_predictions.append(soc_pred)
                self.test_ground_truth.append(soc_true)
                self.test_errors.append(error)
                self.timestamps.append(response_time)
                
                # Rolling MAE for last 5 predictions
                recent_errors = self.test_errors[-5:]
                mae_recent = np.mean(recent_errors)
                
                test_idx = i - test_start + 1
                total_test = total_samples - test_start
                
                print(f"[{test_idx:2d}/{total_test}] V:{voltage:.3f} I:{current:.2f} → SOC: {soc_pred:.4f} | True: {soc_true:.4f} | Error: {error:.4f} | MAE-5: {mae_recent:.4f}")
            else:
                print(f"[{i-test_start+1:2d}/{total_samples-test_start}] Communication failed")
                self.communication_failures += 1
            
            time.sleep(0.5)
    
    def calculate_metrics(self):
        """Calculate and display final metrics"""
        if not self.test_predictions:
            print("❌ No successful predictions to analyze")
            return
        
        predictions = np.array(self.test_predictions)
        ground_truth = np.array(self.test_ground_truth)
        errors = np.array(self.test_errors)
        
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(errors)
        min_error = np.min(errors)
        error_std = np.std(errors)
        
        total_attempts = len(self.test_predictions) + self.communication_failures
        success_rate = len(self.test_predictions) / total_attempts * 100 if total_attempts > 0 else 0
        
        print("\n" + "=" * 80)
        print("🎯 ARDUINO LSTM SOC PREDICTION - WARMUP MAE TEST RESULTS")
        print("=" * 80)
        
        print("📊 TEST STATISTICS:")
        print(f"   🔥 Warmup predictions: {len(self.warmup_predictions)}")
        print(f"   ✅ Test predictions: {len(self.test_predictions)}")
        print(f"   ❌ Communication failures: {self.communication_failures}")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        
        print("\n🎯 ACCURACY METRICS (Post-Warmup):")
        print(f"   📏 MAE (Mean Absolute Error): {mae:.6f}")
        print(f"   📐 RMSE (Root Mean Square Error): {rmse:.6f}")
        print(f"   📊 Maximum Error: {max_error:.6f}")
        print(f"   📊 Minimum Error: {min_error:.6f}")
        print(f"   📊 Error Standard Deviation: {error_std:.6f}")
        
        print("\n📈 SOC PREDICTION COMPARISON:")
        print(f"   🎯 Predicted SOC: {predictions.min():.3f} - {predictions.max():.3f}")
        print(f"   📊 True SOC: {ground_truth.min():.3f} - {ground_truth.max():.3f}")
        
        # Performance assessment
        print("\n🏆 PERFORMANCE ASSESSMENT:")
        if mae < 0.05:
            print("   🟢 EXCELLENT: MAE < 0.05 - Very high accuracy")
        elif mae < 0.1:
            print("   🟡 GOOD: MAE < 0.10 - Good accuracy")
        elif mae < 0.2:
            print("   🟠 ACCEPTABLE: MAE < 0.20 - Moderate accuracy")
        else:
            print("   🔴 POOR: MAE >= 0.20 - Accuracy needs improvement")
        
        if success_rate > 95:
            print("   🟢 EXCELLENT: Communication reliability > 95%")
        elif success_rate > 90:
            print("   🟡 GOOD: Communication reliability > 90%")
        else:
            print("   🔴 POOR: Communication reliability < 90%")
        
        print("=" * 80)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'success_rate': success_rate,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'errors': errors
        }
    
    def create_visualization(self, metrics):
        """Create comprehensive visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Arduino LSTM SOC Prediction Analysis (With Warmup)', fontsize=16, fontweight='bold')
            
            predictions = metrics['predictions']
            ground_truth = metrics['ground_truth']
            errors = metrics['errors']
            
            # 1. SOC Comparison
            ax1 = axes[0, 0]
            ax1.scatter(ground_truth, predictions, alpha=0.7, s=50)
            ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
            ax1.set_xlabel('True SOC')
            ax1.set_ylabel('Predicted SOC')
            ax1.set_title(f'SOC Prediction Accuracy\nMAE: {metrics["mae"]:.4f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Error over time
            ax2 = axes[0, 1]
            ax2.plot(errors, 'b-', linewidth=2, label='Absolute Error')
            ax2.axhline(y=metrics['mae'], color='r', linestyle='--', label=f'Mean Error: {metrics["mae"]:.4f}')
            ax2.set_xlabel('Test Sample')
            ax2.set_ylabel('Absolute Error')
            ax2.set_title('Error Over Time (Post-Warmup)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Error histogram
            ax3 = axes[1, 0]
            ax3.hist(errors, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=metrics['mae'], color='r', linestyle='--', linewidth=2, label=f'MAE: {metrics["mae"]:.4f}')
            ax3.set_xlabel('Absolute Error')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Error Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Warmup vs Test comparison
            ax4 = axes[1, 1]
            if self.warmup_predictions:
                ax4.plot(range(len(self.warmup_predictions)), self.warmup_predictions, 'r-', 
                        linewidth=2, label='Warmup Predictions', alpha=0.7)
            ax4.plot(range(len(self.warmup_predictions), len(self.warmup_predictions) + len(predictions)), 
                    predictions, 'b-', linewidth=2, label='Test Predictions')
            ax4.axvline(x=len(self.warmup_predictions)-0.5, color='g', linestyle='--', 
                       linewidth=2, label='Warmup End')
            ax4.set_xlabel('Sample Index')
            ax4.set_ylabel('Predicted SOC')
            ax4.set_title('Warmup vs Test Predictions')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            print("📊 Visualization created and displayed")
            
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
    
    def run_full_test(self):
        """Run complete test with warmup and analysis"""
        print("🎯 ARDUINO LSTM SOC PREDICTION - WARMUP MAE TEST")
        print("=" * 70)
        print("📊 This test includes LSTM warmup for accurate measurements")
        print("🔌 Hardware disconnection detection included")
        print("📈 Comprehensive MAE analysis with visualization")
        print("=" * 70)
        
        # Setup
        if not self.setup_ground_truth_data():
            return False
        
        if not self.connect_arduino():
            return False
        
        try:
            start_time = time.time()
            
            # Run warmup phase
            self.run_warmup_phase()
            
            # Run test phase
            self.run_test_phase()
            
            # Calculate metrics and create visualization
            metrics = self.calculate_metrics()
            if metrics:
                self.create_visualization(metrics)
            
            total_time = time.time() - start_time
            print(f"⏱️ Total test duration: {total_time:.1f} seconds")
            
        finally:
            if self.arduino:
                self.arduino.close()
                print("🔌 Arduino connection closed")
        
        return True

if __name__ == "__main__":
    # Create and run test with 10-step warmup
    test = ArduinoMAETestWithWarmup(warmup_steps=10)
    test.run_full_test()
