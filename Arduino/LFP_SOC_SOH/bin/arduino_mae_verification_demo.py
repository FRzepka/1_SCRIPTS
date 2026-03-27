"""
Arduino LSTM MAE Verification Demo
Demonstrates:
1. Live MAE calculation with ground truth data
2. Physical cable disconnection test
3. Real-time accuracy monitoring
"""

import serial
import pandas as pd
import numpy as np
import time
import threading
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class ArduinoMAEDemo:
    def __init__(self, port='COM13'):
        self.port = port
        self.arduino = None
        self.running = True
        
    def connect_arduino(self):
        """Connect to Arduino"""
        try:
            self.arduino = serial.Serial(self.port, 115200, timeout=2)
            time.sleep(2)
            print(f"✅ Arduino connected on {self.port}")
            
            # Reset Arduino
            self.arduino.write(b'RESET\n')
            response = self.arduino.readline().decode().strip()
            print(f"🔄 Reset: {response}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def load_sample_ground_truth(self, num_samples=100):
        """Load a small sample of ground truth data for demo"""
        data_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
        cell_name = "MGFarm_18650_C19"
        
        base = Path(data_path)
        folder = base / cell_name
        dfp = folder / "df.parquet"
        
        if not dfp.exists():
            print(f"❌ Ground truth data not found: {dfp}")
            return None
        
        # Load full data
        df = pd.read_parquet(dfp)
        
        # Take a sample from middle of dataset (varied SOC range)
        start_idx = len(df) // 3
        sample_df = df.iloc[start_idx:start_idx + num_samples].copy()
        
        # Create scaler
        print("🔧 Creating scaler...")
        scaler = self._create_scaler(base)
        
        # Scale features
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        sample_df[feats] = scaler.transform(sample_df[feats])
        
        print(f"✅ Loaded {len(sample_df)} ground truth samples")
        print(f"📊 SOC range: {sample_df['SOC_ZHU'].min():.3f} - {sample_df['SOC_ZHU'].max():.3f}")
        
        return sample_df
    
    def _create_scaler(self, base_path):
        """Create scaler matching training procedure"""
        all_cells = [
            "MGFarm_18650_C01", "MGFarm_18650_C03", "MGFarm_18650_C05",
            "MGFarm_18650_C11", "MGFarm_18650_C17", "MGFarm_18650_C23",
            "MGFarm_18650_C07", "MGFarm_18650_C19", "MGFarm_18650_C21"
        ]
        
        feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
        scaler = StandardScaler()
        
        for cell_name in all_cells:
            folder = base_path / cell_name
            if folder.exists():
                dfp = folder / "df.parquet"
                if dfp.exists():
                    df = pd.read_parquet(dfp)
                    scaler.partial_fit(df[feats])
        
        return scaler
    
    def run_mae_verification_demo(self):
        """Run complete MAE verification demo"""
        print("\n🚀 ARDUINO LSTM MAE VERIFICATION DEMO")
        print("="*50)
        
        # Step 1: Connect Arduino
        if not self.connect_arduino():
            return False
        
        # Step 2: Load ground truth data
        ground_truth = self.load_sample_ground_truth(num_samples=50)
        if ground_truth is None:
            return False
        
        # Step 3: MAE calculation demo
        print(f"\n📊 STEP 1: MAE CALCULATION DEMO")
        print("-" * 30)
        
        arduino_predictions = []
        true_socs = []
        errors = []
        
        for i, (_, row) in enumerate(ground_truth.iterrows()):
            # Send data to Arduino
            data_str = f"{row['Voltage[V]']:.6f},{row['Current[A]']:.6f},{row['SOH_ZHU']:.6f},{row['Q_c']:.6f}\n"
            
            try:
                self.arduino.write(data_str.encode())
                response = self.arduino.readline().decode().strip()
                
                if response:
                    arduino_soc = float(response)
                    true_soc = row['SOC_ZHU']
                    error = abs(arduino_soc - true_soc)
                    
                    arduino_predictions.append(arduino_soc)
                    true_socs.append(true_soc)
                    errors.append(error)
                    
                    if (i + 1) % 10 == 0:
                        current_mae = np.mean(errors)
                        print(f"  Point {i+1:2d}: Arduino={arduino_soc:.4f}, True={true_soc:.4f}, Error={error:.4f}, Running MAE={current_mae:.4f}")
                
            except Exception as e:
                print(f"❌ Error at point {i+1}: {e}")
        
        # Calculate final MAE
        if arduino_predictions:
            final_mae = np.mean(errors)
            final_rmse = np.sqrt(np.mean([e**2 for e in errors]))
            
            print(f"\n📊 MAE VERIFICATION RESULTS:")
            print(f"  Mean Absolute Error (MAE): {final_mae:.4f}")
            print(f"  Root Mean Square Error (RMSE): {final_rmse:.4f}")
            print(f"  Total predictions: {len(arduino_predictions)}")
            
            # Quality assessment
            if final_mae < 0.02:
                print("  ✅ EXCELLENT accuracy (MAE < 0.02)")
            elif final_mae < 0.05:
                print("  ✅ GOOD accuracy (MAE < 0.05)")
            elif final_mae < 0.10:
                print("  ⚠️ ACCEPTABLE accuracy (MAE < 0.10)")
            else:
                print("  ❌ POOR accuracy (MAE > 0.10)")
        
        # Step 4: Physical disconnection test
        print(f"\n🔌 STEP 2: PHYSICAL DISCONNECTION TEST")
        print("-" * 40)
        print("📋 Instructions:")
        print("  1. Arduino will send test predictions")
        print("  2. When countdown starts, physically disconnect USB cable")
        print("  3. System should detect disconnection immediately")
        
        input("\nPress Enter to start disconnection test...")
        
        # Send a few test predictions first
        print("\n🧪 Sending test predictions...")
        test_data = [
            (3.4, -0.8, 0.91, 1180),
            (3.5, -1.2, 0.92, 1200),
            (3.6, -0.6, 0.93, 1220)
        ]
        
        for i, (v, c, s, q) in enumerate(test_data):
            try:
                data_str = f"{v},{c},{s},{q}\n"
                self.arduino.write(data_str.encode())
                response = self.arduino.readline().decode().strip()
                soc = float(response) if response else "No response"
                print(f"  Test {i+1}: V={v}, I={c}, SOH={s}, Q_c={q} → SOC={soc}")
                time.sleep(0.5)
            except Exception as e:
                print(f"  ❌ Test {i+1} failed: {e}")
        
        # Countdown for cable disconnection
        print(f"\n🔌 DISCONNECT USB CABLE NOW! Countdown:")
        for countdown in range(8, 0, -1):
            print(f"   {countdown}...", end="", flush=True)
            time.sleep(1)
        print("\n")
        
        # Test after disconnection
        print("🔍 Testing after disconnection...")
        try:
            test_str = "3.7,-1.0,0.94,1250\n"
            self.arduino.write(test_str.encode())
            response = self.arduino.readline().decode().strip()
            print(f"❌ UNEXPECTED: Still receiving data: {response}")
            print("⚠️ Cable might not be disconnected properly")
        except Exception as e:
            error_msg = str(e)
            if "PermissionError" in error_msg or "device does not recognize" in error_msg:
                print("✅ CORRECT: Physical disconnection detected!")
                print(f"   Error type: {type(e).__name__}")
            else:
                print(f"❌ Unexpected error: {e}")
        
        print(f"\n🏁 MAE VERIFICATION DEMO COMPLETE!")
        return True

def main():
    """Main demo function"""
    print("🎯 Arduino LSTM MAE Verification Demo")
    print("This demo shows:")
    print("  1. Real-time MAE calculation with ground truth data")
    print("  2. Hardware accuracy verification")
    print("  3. Physical disconnection robustness testing")
    
    demo = ArduinoMAEDemo(port='COM13')
    
    try:
        success = demo.run_mae_verification_demo()
        if success:
            print("\n✅ Demo completed successfully!")
        else:
            print("\n❌ Demo failed!")
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        if demo.arduino and demo.arduino.is_open:
            demo.arduino.close()
            print("🔌 Arduino disconnected")

if __name__ == "__main__":
    main()
