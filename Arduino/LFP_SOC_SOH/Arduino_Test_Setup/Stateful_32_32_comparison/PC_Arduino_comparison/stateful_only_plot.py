#!/usr/bin/env python3
"""
🔬 Stateful LSTM SOC Prediction Plot
Microcontroller Deployment Analysis for STM32H757

Features:
- Nur Stateful LSTM Vorhersage
- Konfigurierbarer Zeitbereich (Start + Duration in Sekunden)
- Ground Truth vs. Prediction Plot
- MAE Berechnung und Anzeige
- Publication-ready Scientific Plot

Target Hardware: STM32H757 (Cortex-M7, 1MB RAM, 2MB Flash)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# === ZEITBEREICH KONFIGURATION ===
START_TIME_SEC = 0      # Start in Sekunden
DURATION_SEC = 20000     # Dauer in Sekunden

# === DEFINIERTES FARBSCHEMA ===
COLOR_SCHEME = {
    # 🔴 Main Stateful LSTM colors - orange/red tones
    'stateful_lstm': '#FF6B6B',      # Red/Orange for Stateful LSTM
    'stateful_secondary': '#FFB3B3', # Soft Red/Pink for secondary elements
    
    # 🔷 Blue accent - professional accent color
    'blue_accent': '#2091C9',         # Lebendiges Blau für Linien, Punkte, Verbindungen
    'ground_truth': '#2091C9',        # Blau für Ground Truth
    
    # ⚡ Kräftige Akzent-Farben für Details
    'accent_blue': '#2091C9',        # Lebendiges Blau für Linien, Punkte, Verbindungen
    'accent_violet': '#BB76F7',      # Elegantes Violett für spezielle Highlights
    'error_color': '#D9140E'         # Signalfarbe Rot für Fehler
}

# === PARAMETER FÜR STATEFUL LSTM ===
STATEFUL_SEQ_CHUNK_SIZE = 4096
STATEFUL_HIDDEN_SIZE = 32
STATEFUL_NUM_LAYERS = 1
STATEFUL_MLP_HIDDEN = 32

# === STM32H757 MICROCONTROLLER SPECIFICATIONS ===
STM32_RAM_BYTES = 1024 * 1024  # 1MB RAM
STM32_FLASH_BYTES = 2 * 1024 * 1024  # 2MB Flash 
STM32_CORE_FREQ_HZ = 480_000_000  # 480MHz Cortex-M7

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")
print(f"🎯 Target MCU: STM32H757 @ {STM32_CORE_FREQ_HZ/1e6:.0f}MHz")
print(f"💾 Available RAM: {STM32_RAM_BYTES/1024/1024:.1f}MB")
print(f"💽 Available Flash: {STM32_FLASH_BYTES/1024/1024:.1f}MB")
print(f"⏱️  Plot Range: {START_TIME_SEC}s - {START_TIME_SEC + DURATION_SEC}s ({DURATION_SEC}s duration)")

# === STATEFUL LSTM MODEL (1.2.4.36) ===
class StatefulSOCModel(nn.Module):
    def __init__(self, input_size=4, dropout=0.03):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = STATEFUL_HIDDEN_SIZE
        self.num_layers = STATEFUL_NUM_LAYERS
        
        self.lstm = nn.LSTM(
            input_size, STATEFUL_HIDDEN_SIZE, STATEFUL_NUM_LAYERS,
            batch_first=True, dropout=0.0
        )
        
        # Konservative MLP-Architektur mit LayerNorm (wie im Original)
        self.mlp = nn.Sequential(
            nn.Linear(STATEFUL_HIDDEN_SIZE, STATEFUL_MLP_HIDDEN),
            nn.LayerNorm(STATEFUL_MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(STATEFUL_MLP_HIDDEN, STATEFUL_MLP_HIDDEN),
            nn.LayerNorm(STATEFUL_MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(STATEFUL_MLP_HIDDEN, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.contiguous()
        h, c = hidden
        h, c = h.contiguous(), c.contiguous()
        
        out, new_hidden = self.lstm(x, (h, c))
        out = self.mlp(out)
        return out, new_hidden

def init_hidden_stateful(model, batch_size=1, device=None):
    """Initialize hidden states for Stateful LSTM"""
    if device is None:
        device = next(model.parameters()).device
    h = torch.zeros(STATEFUL_NUM_LAYERS, batch_size, STATEFUL_HIDDEN_SIZE, device=device)
    c = torch.zeros_like(h)
    return h, c

def load_test_data(data_path=r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"):
    """Lädt die C19 Testdaten"""
    base = Path(data_path)
    test_cell = "MGFarm_18650_C19"
    
    # Lade C19 Daten
    cell_path = base / test_cell / "df.parquet"
    if not cell_path.exists():
        raise FileNotFoundError(f"Test data not found: {cell_path}")
    
    df = pd.read_parquet(cell_path)
    print(f"📊 Loaded test data: {test_cell} - {len(df):,} rows")
    
    # Prepare data like in training
    df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
    
    return df

def create_stateful_scaler():
    """Erstellt den RobustScaler für Stateful LSTM"""
    base_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
    base = Path(base_path)
    
    # Alle Zellen wie im Training (train + val)
    train_cells = [
        "MGFarm_18650_C01", "MGFarm_18650_C03", "MGFarm_18650_C05",
        "MGFarm_18650_C11", "MGFarm_18650_C17", "MGFarm_18650_C23"
    ]
    val_cells = [
        "MGFarm_18650_C07", "MGFarm_18650_C19", "MGFarm_18650_C21"
    ]
    all_cells = train_cells + val_cells
    
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
    
    print("🔧 Creating RobustScaler for Stateful LSTM...")
    
    # Sammle alle Daten für Scaler-Fitting
    combined_data_list = []
    for cell_name in all_cells:
        cell_path = base / cell_name / "df.parquet"
        if cell_path.exists():
            df_cell = pd.read_parquet(cell_path)
            nan_count = df_cell[feats].isna().sum().sum()
            if nan_count > 0:
                print(f"   - WARNING: {cell_name} has {nan_count} NaNs - filling with median")
                df_cell[feats] = df_cell[feats].fillna(df_cell[feats].median())
            combined_data_list.append(df_cell[feats])
        else:
            print(f"   ❌ WARNING: {cell_path} not found")
    
    if not combined_data_list:
        raise ValueError("No training data found for scaler!")
    else:
        combined_data = pd.concat(combined_data_list, ignore_index=True)
    
    # Fit RobustScaler
    scaler = RobustScaler()
    scaler.fit(combined_data)
    
    print(f"✅ RobustScaler fitted on {len(combined_data):,} samples")
    
    return scaler, feats

def predict_stateful_time_range(model, df, scaler, feats, device, start_sec, duration_sec):
    """Führt Stateful Prediction für bestimmten Zeitbereich durch"""
    model.eval()
    
    # Scale features
    df_scaled = df.copy()
    df_scaled[feats] = scaler.transform(df[feats])
    
    # Calculate time range indices (assuming 1Hz sampling rate)
    end_sec = start_sec + duration_sec
    start_idx = max(0, start_sec)
    end_idx = min(len(df), end_sec)
    
    print(f"📊 Time range: {start_sec}s - {end_sec}s")
    print(f"📊 Data indices: {start_idx} - {end_idx} ({end_idx - start_idx} samples)")
    
    # Extract time range data
    seq = df_scaled[feats].values[start_idx:end_idx]
    labels = df["SOC_ZHU"].values[start_idx:end_idx]
    timestamps = df['timestamp'].values[start_idx:end_idx]
    
    total = len(seq)
    n_chunks = math.ceil(total / STATEFUL_SEQ_CHUNK_SIZE)
    print(f"🧠 Stateful Prediction: {total:,} timesteps in {n_chunks} chunks...")
    
    # Initialize hidden states
    h, c = init_hidden_stateful(model, batch_size=1, device=device)
    h, c = h.contiguous(), c.contiguous()
    preds = []

    with torch.no_grad():
        progress_bar = tqdm(range(n_chunks), desc="🧠 Stateful Inference", 
                           unit="chunk", ncols=100, 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for i in progress_bar:
            s = i * STATEFUL_SEQ_CHUNK_SIZE
            e = min(s + STATEFUL_SEQ_CHUNK_SIZE, total)
            
            chunk_data = torch.tensor(seq[s:e], dtype=torch.float32, device=device).unsqueeze(0)
            chunk_data = chunk_data.contiguous()
            
            model.lstm.flatten_parameters()
            out, (h, c) = model(chunk_data, (h, c))
            h, c = h.detach().contiguous(), c.detach().contiguous()
            
            preds.extend(out.squeeze(0).cpu().numpy())
            
            progress_bar.set_postfix({
                'Samples': f'{e:,}',
                'Progress': f'{(e/total)*100:.1f}%'
            })
    
    preds = np.array(preds)
    gts = labels[:len(preds)]
    
    print(f"✅ Stateful prediction completed! Generated {len(preds):,} predictions.")
    return preds, gts, timestamps[:len(preds)]

def create_stateful_plot(preds, gts, timestamps, mae_value, start_sec):
    """Erstellt den Stateful LSTM Plot"""
    
    # Erstelle Zeit-Achse in Sekunden relativ zum Start
    time_seconds = np.arange(len(preds)) + start_sec
    
    # === PLOT ERSTELLEN ===
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Plot Ground Truth und Prediction
    ax.plot(time_seconds, gts, 
           color=COLOR_SCHEME['ground_truth'], 
           linewidth=2.0, 
           label='Ground Truth SOC',
           alpha=0.8)
    
    ax.plot(time_seconds, preds, 
           color=COLOR_SCHEME['stateful_lstm'], 
           linewidth=1.8, 
           label='Stateful LSTM Prediction',
           alpha=0.9)
    
    # Styling
    ax.set_xlabel('Time [seconds]', fontsize=14, fontweight='bold')
    ax.set_ylabel('State of Charge (SOC)', fontsize=14, fontweight='bold')
    ax.set_title(f'🧠 Stateful LSTM SOC Prediction\n'
                f'Time Range: {START_TIME_SEC}s - {START_TIME_SEC + DURATION_SEC}s | MAE: {mae_value:.4f}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid und Layout
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    # Y-Achse auf SOC-Bereich beschränken
    ax.set_ylim(0, 1)
    
    # MAE Text Box
    textstr = f'MAE: {mae_value:.4f}\nSamples: {len(preds):,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Speichern
    output_path = Path("stateful_soc_prediction.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved: {output_path.absolute()}")
    
    plt.show()

def save_pc_predictions_to_csv(preds, gts, timestamps, start_sec, duration_sec):
    """Speichert PC-Vorhersagen in CSV-Datei für Vergleich mit Arduino"""
    try:
        # Erstelle DataFrame mit PC-Daten
        data = []
        for i, (pred, gt, ts) in enumerate(zip(preds, gts, timestamps)):
            data_point = {
                'time_seconds': start_sec + i,  # Zeit in Sekunden wie beim Arduino
                'soc_ground_truth': float(gt),
                'soc_pc_prediction': float(pred),
                'mae_error': abs(float(pred) - float(gt)),
                'timestamp_original': ts if hasattr(ts, 'strftime') else str(ts)
            }
            data.append(data_point)
        
        df = pd.DataFrame(data)
        
        # Filename mit Zeitstempel
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pc_soc_data_{start_sec}s_to_{start_sec + duration_sec}s_{timestamp}.csv"
        
        # Speichere CSV
        df.to_csv(filename, index=False)
        
        print(f"💾 PC predictions saved to: {filename}")
        print(f"📊 CSV contains {len(df)} rows and {len(df.columns)} columns")
        print(f"📋 Columns: {', '.join(df.columns.tolist())}")
        
        # Statistiken
        avg_mae = df['mae_error'].mean()
        max_mae = df['mae_error'].max()
        min_mae = df['mae_error'].min()
        
        print(f"\n📈 PC Model Performance:")
        print(f"   Average MAE: {avg_mae:.6f}")
        print(f"   Min MAE:     {min_mae:.6f}")
        print(f"   Max MAE:     {max_mae:.6f}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Failed to save PC predictions CSV: {e}")
        return None

def main():
    """Hauptfunktion für Stateful LSTM Plot"""
    
    print("🔬 Starting Stateful LSTM SOC Prediction Plot")
    print("=" * 60)
    
    # === STATEFUL LSTM LADEN ===
    print("\n📥 Loading Stateful LSTM Model...")
    stateful_model_path = Path(r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32_comparison\PC_only\models_for_PC_only\best_model_sf_32")
    if not stateful_model_path.exists():
        raise FileNotFoundError(f"Stateful model not found: {stateful_model_path}")
    
    stateful_model = StatefulSOCModel(input_size=4, dropout=0.03)
    stateful_model.load_state_dict(torch.load(stateful_model_path, map_location=device, weights_only=False))
    stateful_model.to(device)
    stateful_model.eval()
    print("✅ Stateful LSTM loaded!")
    
    # === SCALER ERSTELLEN ===
    print("\n🔧 Creating Scaler...")
    stateful_scaler, feats = create_stateful_scaler()
    
    # === TESTDATEN LADEN ===
    print("\n📊 Loading Test Data...")
    test_df = load_test_data()
    
    # === VORHERSAGE FÜR ZEITBEREICH ===
    print(f"\n🧠 Running Stateful Prediction for time range {START_TIME_SEC}s - {START_TIME_SEC + DURATION_SEC}s...")
    preds, gts, timestamps = predict_stateful_time_range(
        stateful_model, test_df, stateful_scaler, feats, device, 
        START_TIME_SEC, DURATION_SEC
    )
      # === MAE BERECHNUNG ===
    mae_value = mean_absolute_error(gts, preds)
    rmse_value = np.sqrt(mean_squared_error(gts, preds))
    r2_value = r2_score(gts, preds)
    
    print(f"\n📊 Performance Metrics:")
    print(f"   MAE:  {mae_value:.6f}")
    print(f"   RMSE: {rmse_value:.6f}")
    print(f"   R²:   {r2_value:.6f}")
    
    # === CSV SPEICHERN ===
    print(f"\n💾 Saving PC predictions to CSV...")
    csv_filename = save_pc_predictions_to_csv(preds, gts, timestamps, START_TIME_SEC, DURATION_SEC)
    
    # === PLOT ERSTELLEN ===
    print(f"\n📊 Creating Plot...")
    create_stateful_plot(preds, gts, timestamps, mae_value, START_TIME_SEC)
    
    # === CSV EXPORT FÜR PC-VORHERSAGEN ===
    print(f"\n💾 Exporting PC predictions to CSV...")
    csv_filename = save_pc_predictions_to_csv(preds, gts, timestamps, START_TIME_SEC, DURATION_SEC)
    
    print("\n✅ Analysis completed successfully!")

if __name__ == "__main__":
    main()
