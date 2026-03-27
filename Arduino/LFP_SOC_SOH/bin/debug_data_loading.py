"""
Debug test to check data loading
"""

import os
import pandas as pd

def test_data_loading():
    print("🔍 Testing data loading...")
    
    # Check C19 path
    c19_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
    
    print(f"📂 Checking path: {c19_path}")
    print(f"   Exists: {os.path.exists(c19_path)}")
    
    if os.path.exists(c19_path):
        try:
            df = pd.read_parquet(c19_path)
            print(f"   ✅ Loaded: {len(df)} rows")
            print(f"   📋 Columns: {df.columns.tolist()}")
            
            # Check required columns
            required = ['Voltage[V]', 'Current[A]', 'SOH', 'Q_c', 'SOC_ZHU']
            for col in required:
                if col in df.columns:
                    print(f"   ✅ {col}: Available")
                else:
                    print(f"   ❌ {col}: Missing")
        except Exception as e:
            print(f"   ❌ Error loading: {e}")
    else:
        print("   ❌ File not found")

if __name__ == "__main__":
    test_data_loading()
