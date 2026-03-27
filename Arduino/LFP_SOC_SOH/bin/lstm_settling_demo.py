"""
Quick Arduino Test - Show LSTM Settling Behavior
Demonstrates how LSTM predictions improve after warmup period
"""

import serial
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def quick_settling_demo():
    """Quick demo showing LSTM settling behavior"""
    
    print("🔥 ARDUINO LSTM SETTLING BEHAVIOR DEMO")
    print("=" * 50)
    print("This shows how predictions improve after warmup")
    print("=" * 50)
    
    # Connect to Arduino
    try:
        arduino = serial.Serial('COM13', 115200, timeout=3)
        time.sleep(3)
        print("✅ Arduino connected")
        
        # Reset LSTM
        arduino.flushInput()
        arduino.write(b'RESET\n')
        time.sleep(1)
        
        # Load some test data
        c19_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
        df = pd.read_parquet(c19_path)
        
        # Get a small sample
        sample_data = df.iloc[10000:10020][['Voltage[V]', 'Current[A]', 'SOH', 'Q_c', 'SOC_ZHU']].dropna()
        
        # Create scaler
        scaler = StandardScaler()
        scaler.fit(df[['Voltage[V]', 'Current[A]', 'SOH', 'Q_c']].dropna())
        
        predictions = []
        true_values = []
        
        print("\n📊 Sending sequential data and monitoring predictions:")
        print("=" * 60)
        
        for i, (idx, row) in enumerate(sample_data.iterrows()):
            voltage = row['Voltage[V]']
            current = row['Current[A]']
            soh = row['SOH']
            q_c = row['Q_c']
            soc_true = row['SOC_ZHU']
            
            # Normalize
            features = np.array([[voltage, current, soh, q_c]])
            features_scaled = scaler.transform(features)
            
            # Send to Arduino
            data_str = f"{features_scaled[0,0]:.6f},{features_scaled[0,1]:.6f},{features_scaled[0,2]:.6f},{features_scaled[0,3]:.6f}\n"
            arduino.write(data_str.encode())
            
            time.sleep(0.1)
            response = arduino.readline().decode().strip()
            
            try:
                soc_pred = float(response)
                predictions.append(soc_pred)
                true_values.append(soc_true)
                
                error = abs(soc_pred - soc_true)
                
                # Show warmup vs settled
                if i < 5:
                    status = "🔥 WARMUP"
                elif i < 10:
                    status = "⚡ SETTLING"
                else:
                    status = "✅ SETTLED"
                
                print(f"[{i+1:2d}] {status} | V:{voltage:.3f} I:{current:.2f} → Pred:{soc_pred:.4f} True:{soc_true:.4f} Error:{error:.4f}")
                
            except:
                print(f"[{i+1:2d}] ❌ Invalid response: {response}")
            
            time.sleep(0.3)
        
        arduino.close()
        
        # Quick analysis
        if len(predictions) >= 15:
            warmup_errors = [abs(p - t) for p, t in zip(predictions[:5], true_values[:5])]
            settling_errors = [abs(p - t) for p, t in zip(predictions[5:10], true_values[5:10])]
            settled_errors = [abs(p - t) for p, t in zip(predictions[10:], true_values[10:])]
            
            print("\n📈 SETTLING ANALYSIS:")
            print("=" * 40)
            print(f"🔥 Warmup MAE (first 5):  {np.mean(warmup_errors):.4f}")
            print(f"⚡ Settling MAE (5-10):   {np.mean(settling_errors):.4f}")
            print(f"✅ Settled MAE (10+):     {np.mean(settled_errors):.4f}")
            
            improvement = (np.mean(warmup_errors) - np.mean(settled_errors)) / np.mean(warmup_errors) * 100
            print(f"📊 Improvement: {improvement:.1f}% better after settling")
        
        print("\n💡 CONCLUSION:")
        print("The LSTM needs several time steps to build proper internal states.")
        print("For accurate measurements, always include a warmup period!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if 'arduino' in locals():
            arduino.close()

if __name__ == "__main__":
    quick_settling_demo()
