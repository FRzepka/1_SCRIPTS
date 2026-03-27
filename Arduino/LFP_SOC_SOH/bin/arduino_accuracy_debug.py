"""
Arduino Accuracy Debug - Check if Arduino predictions are reasonable
Compare Arduino vs PyTorch predictions on same input data
"""

import serial
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import time

# Load the exact PyTorch model
class SOCModel(nn.Module):
    def __init__(self, input_size=4, dropout=0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 32, 1, batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.contiguous()
        h, c = hidden
        h, c = h.contiguous(), c.contiguous()
        hidden = (h, c)
        out, hidden = self.lstm(x, hidden)
        batch, seq_len, hid = out.size()
        out_flat = out.contiguous().view(batch * seq_len, hid)
        soc_flat = self.mlp(out_flat)
        soc = soc_flat.view(batch, seq_len)
        return soc, hidden

def load_pytorch_model():
    """Load the exact trained PyTorch model"""
    model_path = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\training_run_1_2_4_31\best_model.pth"
    
    model = SOCModel()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✅ PyTorch model loaded from {model_path}")
    return model

def load_test_data():
    """Load a small sample of C19 test data"""
    data_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
    base = Path(data_path)
    folder = base / "MGFarm_18650_C19"
    dfp = folder / "df.parquet"
    
    if not dfp.exists():
        raise FileNotFoundError(f"Test data not found: {dfp}")
    
    df = pd.read_parquet(dfp)
    
    # Create scaler
    all_cells = [
        "MGFarm_18650_C01", "MGFarm_18650_C03", "MGFarm_18650_C05",
        "MGFarm_18650_C11", "MGFarm_18650_C17", "MGFarm_18650_C23",
        "MGFarm_18650_C07", "MGFarm_18650_C19", "MGFarm_18650_C21"
    ]
    
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
    scaler = StandardScaler()
    
    for cell_name in all_cells:
        cell_folder = base / cell_name
        if cell_folder.exists():
            cell_dfp = cell_folder / "df.parquet"
            if cell_dfp.exists():
                cell_df = pd.read_parquet(cell_dfp)
                scaler.partial_fit(cell_df[feats])
    
    # Scale the data
    df_scaled = df.copy()
    df_scaled[feats] = scaler.transform(df[feats])
    
    print(f"✅ Loaded test data: {len(df)} rows")
    print(f"📊 SOC range: {df['SOC_ZHU'].min():.3f} - {df['SOC_ZHU'].max():.3f}")
    
    return df_scaled

def connect_arduino(port='COM13'):
    """Connect to Arduino"""
    try:
        arduino = serial.Serial(port, 115200, timeout=2)
        time.sleep(2)
        
        # Reset
        arduino.write(b'RESET\n')
        response = arduino.readline().decode().strip()
        print(f"✅ Arduino connected: {response}")
        
        return arduino
    except Exception as e:
        print(f"❌ Arduino connection failed: {e}")
        return None

def test_predictions():
    """Test Arduino vs PyTorch predictions"""
    print("🔬 ARDUINO vs PYTORCH ACCURACY DEBUG")
    print("="*50)
    
    # Load models and data
    pytorch_model = load_pytorch_model()
    test_data = load_test_data()
    arduino = connect_arduino()
    
    if arduino is None:
        return
    
    # Take a few test samples
    test_samples = test_data.iloc[1000:1010]  # Middle range
    
    print(f"\n🧪 Testing {len(test_samples)} samples:")
    print("Idx | True SOC | PyTorch | Arduino | Py-Err | Ar-Err | Voltage | Current")
    print("-" * 80)
    
    # Initialize PyTorch hidden state
    hidden = (torch.zeros(1, 1, 32), torch.zeros(1, 1, 32))
    
    arduino_fails = 0
    
    for i, (idx, row) in enumerate(test_samples.iterrows()):
        # Prepare input
        voltage = row['Voltage[V]']
        current = row['Current[A]']
        soh = row['SOH_ZHU']
        q_c = row['Q_c']
        true_soc = row['SOC_ZHU']
        
        # PyTorch prediction
        with torch.no_grad():
            input_tensor = torch.tensor([[voltage, current, soh, q_c]], dtype=torch.float32).unsqueeze(1)
            pytorch_soc, hidden = pytorch_model(input_tensor, hidden)
            pytorch_soc = pytorch_soc.item()
        
        # Arduino prediction
        try:
            data_str = f"{voltage:.6f},{current:.6f},{soh:.6f},{q_c:.6f}\n"
            arduino.write(data_str.encode())
            response = arduino.readline().decode().strip()
            arduino_soc = float(response)
        except:
            arduino_soc = -999
            arduino_fails += 1
        
        # Calculate errors
        py_error = abs(pytorch_soc - true_soc)
        ar_error = abs(arduino_soc - true_soc) if arduino_soc != -999 else 999
        
        print(f"{i:3d} | {true_soc:8.4f} | {pytorch_soc:7.4f} | {arduino_soc:7.4f} | {py_error:6.4f} | {ar_error:6.4f} | {voltage:7.3f} | {current:7.3f}")
    
    arduino.close()
    
    print(f"\n📊 Summary:")
    print(f"  Arduino communication failures: {arduino_fails}/{len(test_samples)}")
    print(f"  If Arduino predictions are very different from PyTorch, there might be an issue with:")
    print(f"    1. Weight extraction/conversion")
    print(f"    2. Data scaling in Arduino")
    print(f"    3. LSTM implementation differences")
    print(f"    4. Hidden state management")

if __name__ == "__main__":
    test_predictions()
