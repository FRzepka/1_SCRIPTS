"""
Extrahiere EXAKTE Scaler-Parameter aus dem BMS_SOC_LSTM_stateful_1.2.4_Train.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import gc

def extract_exact_scaler_params():
    """Extrahiere die EXAKTEN Scaler-Parameter wie im Training"""
    
    base_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
    base = Path(base_path)
    
    # EXAKTE Zellaufteilung aus dem Training
    train_cells = [
        "MGFarm_18650_C01",
        "MGFarm_18650_C03", 
        "MGFarm_18650_C05",
        "MGFarm_18650_C11",
        "MGFarm_18650_C17",
        "MGFarm_18650_C23"
    ]
    val_cells = [
        "MGFarm_18650_C07",
        "MGFarm_18650_C19", 
        "MGFarm_18650_C21"
    ]
    
    all_cells = train_cells + val_cells
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]  # EXAKT wie im Training
    
    print("🔍 Extrahiere EXAKTE StandardScaler Parameter...")
    
    # StandardScaler erstellen (EXAKT wie im Training)
    scaler = StandardScaler()
    
    # Iterativ über alle Zellen fitten (EXAKT wie im Training)
    for cell_name in all_cells:
        folder_path = base / cell_name
        df_path = folder_path / "df.parquet"
        
        if df_path.exists():
            print(f"Loading {cell_name}...")
            df = pd.read_parquet(df_path)
            
            # Partial fit auf die EXAKTEN Features
            scaler.partial_fit(df[feats])
            del df
            gc.collect()
        else:
            print(f"⚠️  Skipping {cell_name} - file not found")
    
    # Extrahiere die finalen Parameter
    means = scaler.mean_
    scales = scaler.scale_  # Das ist die Standardabweichung
    
    print("\n✅ EXAKTE StandardScaler Parameter:")
    print("=" * 50)
    
    scaler_params = {}
    for i, feat in enumerate(feats):
        mean_val = means[i]
        std_val = scales[i]
        scaler_params[feat] = {'mean': mean_val, 'std': std_val}
        print(f"{feat}:")
        print(f"  mean = {mean_val:.6f}")
        print(f"  std  = {std_val:.6f}")
    
    # Arduino-Format ausgeben
    print("\n🔧 ARDUINO CODE (copy-paste):")
    print("=" * 40)
    print(f"float voltage_mean = {scaler_params['Voltage[V]']['mean']:.6f}f;")
    print(f"float voltage_std = {scaler_params['Voltage[V]']['std']:.6f}f;")
    print(f"float current_mean = {scaler_params['Current[A]']['mean']:.6f}f;")
    print(f"float current_std = {scaler_params['Current[A]']['std']:.6f}f;")
    print(f"float soh_mean = {scaler_params['SOH_ZHU']['mean']:.6f}f;")
    print(f"float soh_std = {scaler_params['SOH_ZHU']['std']:.6f}f;")
    print(f"float qc_mean = {scaler_params['Q_c']['mean']:.6f}f;")
    print(f"float qc_std = {scaler_params['Q_c']['std']:.6f}f;")
    
    return scaler_params

if __name__ == "__main__":
    extract_exact_scaler_params()
