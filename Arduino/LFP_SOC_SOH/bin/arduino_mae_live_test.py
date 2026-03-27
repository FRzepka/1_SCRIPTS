"""
Arduino LSTM Live MAE Verification Test
Combines Arduino hardware predictions with ground truth SOC data for real-time MAE calculation
Tests physical disconnection while monitoring accuracy metrics
"""

import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import threading
import queue
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import gc
import json

class ArduinoMAELiveTest:
    def __init__(self, port='COM13', max_points=200):
        self.port = port
        self.max_points = max_points
        self.arduino = None
        
        # Data structures for live plotting
        self.data_queue = queue.Queue()
        self.timestamps = deque(maxlen=max_points)
        self.arduino_predictions = deque(maxlen=max_points)
        self.ground_truth_soc = deque(maxlen=max_points)
        self.absolute_errors = deque(maxlen=max_points)
        self.mae_values = deque(maxlen=max_points)
        self.input_voltages = deque(maxlen=max_points)
        self.input_currents = deque(maxlen=max_points)
        
        # Performance metrics
        self.total_predictions = 0
        self.total_errors = 0
        self.communication_failures = 0
        
        # Running flag
        self.running = True
        
        # Load ground truth data
        self.ground_truth_data = None
        self.ground_truth_index = 0
        
        # Setup plots
        plt.style.use('dark_background')
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('🔥 ARDUINO LSTM MAE LIVE VERIFICATION TEST 🔥', fontsize=16, color='cyan')
        
    def load_ground_truth_data(self, data_path=None, cell_name="MGFarm_18650_C19"):
        """Load ground truth SOC data from C19 cell"""
        if data_path is None:
            data_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
        
        base = Path(data_path)
        folder = base / cell_name
        dfp = folder / "df.parquet"
        
        if not dfp.exists():
            raise FileNotFoundError(f"Ground truth data not found: {dfp}")
        
        df = pd.read_parquet(dfp)
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        
        # Create scaler for data preparation
        self.scaler = self._create_scaler(base)
        
        # Prepare the data
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        df_scaled = df.copy()
        df_scaled[feats] = self.scaler.transform(df[feats])
        
        print(f"✅ Loaded ground truth data: {len(df)} rows from {cell_name}")
        print(f"📊 SOC range: {df['SOC_ZHU'].min():.3f} - {df['SOC_ZHU'].max():.3f}")
        
        self.ground_truth_data = df_scaled
        return True
    
    def _create_scaler(self, base_path):
        """Create scaler matching the training procedure"""
        # Use same cells as training
        all_cells = [
            "MGFarm_18650_C01", "MGFarm_18650_C03", "MGFarm_18650_C05",
            "MGFarm_18650_C11", "MGFarm_18650_C17", "MGFarm_18650_C23",
            "MGFarm_18650_C07", "MGFarm_18650_C19", "MGFarm_18650_C21"
        ]
        
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        scaler = StandardScaler()
        
        print("🔧 Creating scaler from training data...")
        for cell_name in all_cells:
            folder = base_path / cell_name
            if folder.exists():
                dfp = folder / "df.parquet"
                if dfp.exists():
                    df = pd.read_parquet(dfp)
                    scaler.partial_fit(df[feats])
        
        print("✅ Scaler created successfully")
        return scaler
    
    def connect_arduino(self):
        """Connect to Arduino"""
        try:
            self.arduino = serial.Serial(self.port, 115200, timeout=2)
            time.sleep(2)  # Wait for reset
            print(f"✅ Arduino connected on {self.port}")
            
            # Send RESET command
            self.arduino.write(b'RESET\n')
            response = self.arduino.readline().decode().strip()
            print(f"🔄 Reset response: {response}")
            
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def get_next_ground_truth_point(self):
        """Get next data point from ground truth"""
        if self.ground_truth_data is None or self.ground_truth_index >= len(self.ground_truth_data):
            return None
        
        row = self.ground_truth_data.iloc[self.ground_truth_index]
        self.ground_truth_index += 1
        
        return {
            'voltage': row['Voltage[V]'],
            'current': row['Current[A]'], 
            'soh': row['SOH_ZHU'],
            'q_c': row['Q_c'],
            'true_soc': row['SOC_ZHU']
        }
    
    def data_reader_thread(self):
        """Thread to continuously read data and get Arduino predictions"""
        prediction_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                # Get next ground truth point
                gt_point = self.get_next_ground_truth_point()
                if gt_point is None:
                    print("📊 Reached end of ground truth data")
                    break
                
                # Send to Arduino
                data_str = f"{gt_point['voltage']:.6f},{gt_point['current']:.6f},{gt_point['soh']:.6f},{gt_point['q_c']:.6f}\n"
                
                try:
                    self.arduino.write(data_str.encode())
                    response = self.arduino.readline().decode().strip()
                    
                    if response:
                        try:
                            arduino_soc = float(response)
                            
                            # Calculate error
                            error = abs(arduino_soc - gt_point['true_soc'])
                            
                            # Store data for plotting
                            self.data_queue.put({
                                'timestamp': time.time() - start_time,
                                'arduino_soc': arduino_soc,
                                'true_soc': gt_point['true_soc'],
                                'error': error,
                                'voltage': gt_point['voltage'],
                                'current': gt_point['current']
                            })
                            
                            prediction_count += 1
                            self.total_predictions += 1
                            
                            if prediction_count % 50 == 0:
                                print(f"📊 Processed {prediction_count} predictions (Arduino: {arduino_soc:.4f}, True: {gt_point['true_soc']:.4f}, Error: {error:.4f})")
                        
                        except ValueError:
                            print(f"❌ Invalid Arduino response: {response}")
                            self.communication_failures += 1
                    else:
                        print("❌ No response from Arduino")
                        self.communication_failures += 1
                        
                except Exception as e:
                    print(f"❌ Communication error: {e}")
                    self.communication_failures += 1
                    # Try to reconnect
                    if "PermissionError" in str(e) or "device does not recognize" in str(e):
                        print("🔌 Physical disconnection detected!")
                        break
                    
                # Control rate (about 10 Hz)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Data reader error: {e}")
                break
        
        print(f"📊 Data reader finished. Total predictions: {self.total_predictions}, Failures: {self.communication_failures}")
    
    def update_plots(self):
        """Update live plots with current data"""
        # Process queue data
        new_data = []
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                new_data.append(data)
            except:
                break
        
        # Add new data to deques
        for data in new_data:
            self.timestamps.append(data['timestamp'])
            self.arduino_predictions.append(data['arduino_soc'])
            self.ground_truth_soc.append(data['true_soc'])
            self.absolute_errors.append(data['error'])
            self.input_voltages.append(data['voltage'])
            self.input_currents.append(data['current'])
            
            # Calculate rolling MAE
            if len(self.absolute_errors) >= 10:
                mae = np.mean(list(self.absolute_errors)[-50:])  # Last 50 points
                self.mae_values.append(mae)
            else:
                self.mae_values.append(data['error'])
        
        if len(self.timestamps) == 0:
            return
        
        # Clear plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        times = list(self.timestamps)
        
        # Plot 1: SOC Predictions vs Ground Truth
        self.ax1.plot(times, list(self.ground_truth_soc), 'b-', label='Ground Truth SOC', linewidth=2)
        self.ax1.plot(times, list(self.arduino_predictions), 'r-', label='Arduino LSTM', linewidth=2, alpha=0.8)
        self.ax1.set_ylabel('SOC', color='white')
        self.ax1.set_title('🎯 SOC Predictions vs Ground Truth', color='cyan')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Absolute Error and Rolling MAE
        self.ax2.plot(times, list(self.absolute_errors), 'orange', label='Absolute Error', alpha=0.7)
        if len(self.mae_values) > 0:
            mae_times = times[-len(self.mae_values):]
            self.ax2.plot(mae_times, list(self.mae_values), 'g-', label='Rolling MAE (50pts)', linewidth=3)
        self.ax2.set_ylabel('Error', color='white')
        self.ax2.set_title('📊 Accuracy Metrics', color='yellow')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Input Voltage
        self.ax3.plot(times, list(self.input_voltages), 'cyan', label='Voltage [V]', linewidth=2)
        self.ax3.set_ylabel('Voltage [V]', color='white')
        self.ax3.set_title('⚡ Input Voltage', color='cyan')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        # Plot 4: Input Current
        self.ax4.plot(times, list(self.input_currents), 'magenta', label='Current [A]', linewidth=2)
        self.ax4.set_ylabel('Current [A]', color='white')
        self.ax4.set_xlabel('Time [s]', color='white')
        self.ax4.set_title('🔋 Input Current', color='magenta')
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)
        
        # Add statistics
        if len(self.absolute_errors) > 0:
            current_mae = np.mean(list(self.absolute_errors))
            current_rmse = np.sqrt(np.mean([e**2 for e in self.absolute_errors]))
            max_error = np.max(list(self.absolute_errors))
            
            stats_text = f'📊 Live Stats:\nMAE: {current_mae:.4f}\nRMSE: {current_rmse:.4f}\nMax Error: {max_error:.4f}\nPredictions: {self.total_predictions}\nFailed: {self.communication_failures}'
            self.fig.text(0.02, 0.02, stats_text, fontsize=10, color='yellow', 
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8))
    
    def run_live_test(self, duration_minutes=5):
        """Run live MAE test with Arduino"""
        if not self.connect_arduino():
            return False
        
        if not self.load_ground_truth_data():
            return False
        
        print(f"\n🚀 Starting Live MAE Test for {duration_minutes} minutes...")
        print("📋 Instructions:")
        print("  - Watch live MAE calculation")
        print("  - Physically disconnect USB cable to test")
        print("  - Press Ctrl+C to stop")
        print("  - Close plot window to exit")
        
        # Start data reading thread
        data_thread = threading.Thread(target=self.data_reader_thread, daemon=True)
        data_thread.start()
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        try:
            while self.running and time.time() < end_time:
                self.update_plots()
                plt.pause(0.1)
                
                # Check if window is closed
                if not plt.fignum_exists(self.fig.number):
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Test error: {e}")
        finally:
            self.running = False
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
        
        # Final statistics
        self.print_final_stats()
        
        return True
    
    def print_final_stats(self):
        """Print final test statistics"""
        print("\n" + "="*60)
        print("🏁 ARDUINO LSTM MAE LIVE TEST - FINAL RESULTS")
        print("="*60)
        
        if len(self.absolute_errors) > 0:
            final_mae = np.mean(list(self.absolute_errors))
            final_rmse = np.sqrt(np.mean([e**2 for e in self.absolute_errors]))
            max_error = np.max(list(self.absolute_errors))
            min_error = np.min(list(self.absolute_errors))
            
            print(f"📊 ACCURACY METRICS:")
            print(f"  Mean Absolute Error (MAE): {final_mae:.4f}")
            print(f"  Root Mean Square Error (RMSE): {final_rmse:.4f}")
            print(f"  Maximum Error: {max_error:.4f}")
            print(f"  Minimum Error: {min_error:.4f}")
            
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"  Total Predictions: {self.total_predictions}")
            print(f"  Communication Failures: {self.communication_failures}")
            error_rate = (self.communication_failures / max(self.total_predictions, 1)) * 100
            print(f"  Error Rate: {error_rate:.2f}%")
            
            print(f"\n🎯 QUALITY ASSESSMENT:")
            if final_mae < 0.01:
                print("  ✅ EXCELLENT: MAE < 0.01 (Very high accuracy)")
            elif final_mae < 0.02:
                print("  ✅ GOOD: MAE < 0.02 (High accuracy)")
            elif final_mae < 0.05:
                print("  ⚠️  ACCEPTABLE: MAE < 0.05 (Moderate accuracy)")
            else:
                print("  ❌ POOR: MAE > 0.05 (Low accuracy)")
                
            if error_rate < 1:
                print("  ✅ COMMUNICATION: Excellent reliability")
            elif error_rate < 5:
                print("  ⚠️  COMMUNICATION: Acceptable reliability")
            else:
                print("  ❌ COMMUNICATION: Poor reliability")
        else:
            print("❌ No valid predictions received!")

def main():
    print("🚀 Arduino LSTM MAE Live Verification Test")
    print("="*50)
    print("DEBUG: Starting main function...")
      # Configuration
    arduino_port = 'COM13'  # Adjust as needed
    test_duration = 2  # minutes - Shortened for demo
    
    tester = ArduinoMAELiveTest(port=arduino_port)
    
    print(f"🔧 Configuration:")
    print(f"  Arduino Port: {arduino_port}")
    print(f"  Test Duration: {test_duration} minutes")
    print(f"  Ground Truth: MGFarm_18650_C19")
    print("\n💡 This test will:")
    print("  1. Load ground truth SOC data from C19 cell")
    print("  2. Send data to Arduino and get predictions")
    print("  3. Calculate live MAE between Arduino and ground truth")
    print("  4. Show real-time accuracy visualization")
    print("  5. Test physical disconnection robustness")
    
    input("\nPress Enter to start the test...")
    
    try:
        success = tester.run_live_test(duration_minutes=test_duration)
        if success:
            print("✅ Test completed successfully!")
        else:
            print("❌ Test failed!")
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    print("\n🎯 Test finished. Check the plots and statistics above.")

if __name__ == "__main__":
    main()
