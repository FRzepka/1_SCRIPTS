#!/usr/bin/env python3
"""
🔬 Comprehensive Scientific Comparison: Stateful LSTM vs. Windows-based LSTM
Microcontroller Deployment Analysis for STM32H757

Erstellt wissenschaftliche Vergleichsdiagramme zwischen:
- Stateful LSTM (BMS_SOC_LSTM_stateful_1.2.4.36) 
- Windows-based LSTM (BMS_SOC_LSTM_windows_2.0.1)

Features:
- Side-by-side Memory Usage Comparison
- Stacked Bar Charts für detaillierte Komponenten-Analyse
- Performance Metrics Comparison
- Energy Consumption Analysis
- Reduced Plot Data (erste & letzte 100k Samples)
- Publication-ready Scientific Plots

Target Hardware: STM32H757 (Cortex-M    # === COMPARISON PLOT 2: Prediction Accuracy Comparison ===
    fig = plt.figure(figsize=(20, 15))
    # Erstelle ein 3x2 Grid: 3 Reihen, 2 Spalten 
    # width_ratios=[2, 1] bedeutet: linke Spalte ist doppelt so breit wie rechte
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1])
    
    # Create 5 subplots: 
    ax1 = fig.add_subplot(gs[0, 0])  # Stateful LSTM time series (row 0, col 0)
    ax2 = fig.add_subplot(gs[1, 0])  # Windows LSTM time series (row 1, col 0) - UNTEREINANDER
    ax3 = fig.add_subplot(gs[:2, 1])  # Scatter plot comparison (row 0-1, col 1) - spans 2 rows
    ax4 = fig.add_subplot(gs[2, 0])  # Error distribution (row 2, col 0) 
    ax5 = fig.add_subplot(gs[2, 1])  # Metrics comparison (row 2, col 1), 1MB RAM, 2MB Flash)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from collections import defaultdict

# === DEFINIERTES FARBSCHEMA ===
COLOR_SCHEME = {
    # 🔴 Main Stateful LSTM colors - orange/red tones
    'stateful_lstm': '#FF6B6B',      # Red/Orange for Stateful LSTM
    'stateful_secondary': '#FFB3B3', # Soft Red/Pink for secondary elements
    
    # 🌿 Main Windows LSTM colors - green/mint tones
    'windows_lstm': '#A8E6CF',       # Soft mint green for Windows LSTM
    'windows_secondary': '#C8F0DB',  # Very light mint for secondary elements
    
    # Architecture-specific colors - red/orange variants for Stateful
    'stateful_16': '#FF6B6B',        # Base red/orange
    'stateful_32': '#FF4444',        # Darker red for variation
    'stateful_64': '#FF6B6B',        # Base red/orange
    'stateful_16_secondary': '#FFB3B3',  # Light red/pink
    'stateful_32_secondary': '#FFB3B3',  # Light red/pink
    'stateful_64_secondary': '#FFB3B3',  # Light red/pink
    
    # Architecture-specific colors - green variants for Windows
    'windows_16': '#A8E6CF',         # Base soft mint green
    'windows_32': '#7DD3C0',         # Medium mint green for variation
    'windows_64': '#A8E6CF',         # Base soft mint green
    'windows_16_secondary': '#C8F0DB',   # Light mint
    'windows_32_secondary': '#C8F0DB',   # Light mint
    'windows_64_secondary': '#C8F0DB',   # Light mint
    
    # 🔷 Blue accent - professional accent color
    'blue_accent': '#2091C9',         # Lebendiges Blau für Linien, Punkte, Verbindungen
    
    # ⚡ Kräftige Akzent-Farben für Details
    'accent_blue': '#2091C9',        # Lebendiges Blau für Linien, Punkte, Verbindungen
    'accent_violet': '#BB76F7',      # Elegantes Violett für spezielle Highlights
    'error_color': '#D9140E'         # Signalfarbe Rot für Fehler
}

# === PARAMETER FÜR BEIDE MODELLE ===
# Stateful LSTM Parameters
STATEFUL_SEQ_CHUNK_SIZE = 4096
STATEFUL_HIDDEN_SIZE = 32
STATEFUL_NUM_LAYERS = 1
STATEFUL_MLP_HIDDEN = 32

# Windows-based LSTM Parameters  
WINDOWS_WINDOW_SIZE = 5000
WINDOWS_HIDDEN_SIZE = 32
WINDOWS_MLP_HIDDEN = 32

# === STM32H757 MICROCONTROLLER SPECIFICATIONS ===
STM32_RAM_BYTES = 1024 * 1024  # 1MB RAM
STM32_FLASH_BYTES = 2 * 1024 * 1024  # 2MB Flash 
STM32_CORE_FREQ_HZ = 480_000_000  # 480MHz Cortex-M7
STM32_POWER_ACTIVE_W = 0.5  # Geschätzte aktive Leistung in Watt
STM32_CYCLES_PER_MAC = 1  # MAC operations per cycle (optimistic)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")
print(f"🎯 Target MCU: STM32H757 @ {STM32_CORE_FREQ_HZ/1e6:.0f}MHz")
print(f"💾 Available RAM: {STM32_RAM_BYTES/1024/1024:.1f}MB")
print(f"💽 Available Flash: {STM32_FLASH_BYTES/1024/1024:.1f}MB")

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

# === WINDOWS-BASED LSTM MODEL (2.0.1) ===
class WindowsSOCModel(nn.Module):
    def __init__(self, input_size=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = WINDOWS_HIDDEN_SIZE
        self.num_layers = num_layers
        
        # LSTM ohne Dropout (voller Informationsfluss)
        self.lstm = nn.LSTM(
            input_size, WINDOWS_HIDDEN_SIZE, num_layers,
            batch_first=True, dropout=0.0
        )
        
        # MLP-Head (wie im Original)
        self.mlp = nn.Sequential(
            nn.Linear(WINDOWS_HIDDEN_SIZE, WINDOWS_MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(WINDOWS_MLP_HIDDEN, WINDOWS_MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(WINDOWS_MLP_HIDDEN, 1),
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

def init_hidden_windows(model, batch_size=1, device=None):
    """Initialize hidden states for Windows LSTM"""
    if device is None:
        device = next(model.parameters()).device
    h = torch.zeros(1, batch_size, WINDOWS_HIDDEN_SIZE, device=device)  # num_layers=1
    c = torch.zeros_like(h)
    return h, c

def count_model_parameters(model, model_type="unknown"):
    """Zählt die Parameter und berechnet Speicherbedarf"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Speicherbedarf in Bytes (32-bit float = 4 bytes)
    memory_bytes = total_params * 4
    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024
    
    return {
        'model_type': model_type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_bytes': memory_bytes,
        'memory_kb': memory_kb,
        'memory_mb': memory_mb
    }

def estimate_inference_ops(model, model_type="unknown"):
    """Schätzt die Anzahl der MAC-Operationen pro Inferenz basierend auf echten Model-Parametern"""
    input_size = 4
    
    # Extrahiere echte Parameter aus dem Modell
    lstm_layer = model.lstm
    mlp_layers = model.mlp
    
    # LSTM Parameter
    hidden_size = lstm_layer.hidden_size
    num_layers = lstm_layer.num_layers
    
    # MLP Parameter - extrahiere aus den Linear Layers
    mlp_sizes = []
    for layer in mlp_layers:
        if isinstance(layer, torch.nn.Linear):
            mlp_sizes.append((layer.in_features, layer.out_features))
    
    print(f"🔍 {model_type.upper()} Model Analysis:")
    print(f"   LSTM: input={input_size}, hidden={hidden_size}, layers={num_layers}")
    print(f"   MLP layers: {mlp_sizes}")
    
    # LSTM Operations: 4 gates × (input×hidden + hidden×hidden) × layers
    lstm_ops = num_layers * 4 * (input_size * hidden_size + hidden_size * hidden_size)
    
    # MLP Operations
    mlp_ops = 0
    for in_features, out_features in mlp_sizes:
        mlp_ops += in_features * out_features
    
    total_ops = lstm_ops + mlp_ops
    
    print(f"   Operations: LSTM={lstm_ops:,}, MLP={mlp_ops:,}, Total={total_ops:,}")
    
    return {
        'model_type': model_type,
        'total_ops': total_ops,
        'lstm_ops': lstm_ops,
        'mlp_ops': mlp_ops,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'mlp_sizes': mlp_sizes
    }

def calculate_mcu_performance_metrics(ops_count, sampling_rate_hz=1.0):
    """Berechnet Performance-Metriken für STM32H757"""
    
    # Zeit pro Inferenz in Zyklen (pessimistisch: 2 Zyklen pro MAC)
    cycles_per_inference = ops_count * 2
    
    # Zeit pro Inferenz in Sekunden
    time_per_inference_s = cycles_per_inference / STM32_CORE_FREQ_HZ
    time_per_inference_ms = time_per_inference_s * 1000
    time_per_inference_us = time_per_inference_s * 1000000
    
    # CPU Load bei gegebener Sampling-Rate
    cpu_load_percent = (time_per_inference_s * sampling_rate_hz) * 100
    
    # Energie pro Inferenz (E = P * t)
    energy_per_inference_j = STM32_POWER_ACTIVE_W * time_per_inference_s
    energy_per_inference_mj = energy_per_inference_j * 1000
    energy_per_inference_uj = energy_per_inference_j * 1000000
    
    # Energie pro Stunde bei kontinuierlicher Ausführung
    inferences_per_hour = 3600 * sampling_rate_hz
    energy_per_hour_j = energy_per_inference_j * inferences_per_hour
    energy_per_hour_wh = energy_per_hour_j / 3600  # Wh = J/3600
    
    return {
        'cycles_per_inference': cycles_per_inference,
        'time_per_inference_s': time_per_inference_s,
        'time_per_inference_ms': time_per_inference_ms,
        'time_per_inference_us': time_per_inference_us,
        'cpu_load_percent': cpu_load_percent,
        'energy_per_inference_j': energy_per_inference_j,
        'energy_per_inference_mj': energy_per_inference_mj,
        'energy_per_inference_uj': energy_per_inference_uj,
        'energy_per_hour_wh': energy_per_hour_wh,
        'max_sampling_rate_hz': min(STM32_CORE_FREQ_HZ / cycles_per_inference, 100)
    }

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

def create_windows_scaler():
    """Erstellt den StandardScaler für Windows LSTM"""
    base_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
    base = Path(base_path)
    
    # Alle Zellen wie im Training 
    train_cells = [f"MGFarm_18650_C{str(i).zfill(2)}" for i in [1,3,5,9,11,13,19,21,23,25,27]]
    val_cells = ["MGFarm_18650_C07","MGFarm_18650_C15","MGFarm_18650_C17"]
    all_cells = train_cells + val_cells
    
    feats = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]
    
    print("🔧 Creating StandardScaler for Windows LSTM...")
    
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
    
    # Fit StandardScaler
    scaler = StandardScaler()
    scaler.fit(combined_data)
    
    print(f"✅ StandardScaler fitted on {len(combined_data):,} samples")
    
    return scaler, feats

def predict_stateful_sequence(model, df, scaler, feats, device):
    """Führt Stateful Prediction durch (reduziert auf erste & letzte 100k)"""
    model.eval()
    
    # Scale features
    df_scaled = df.copy()
    df_scaled[feats] = scaler.transform(df[feats])
    
    # Get sequence data
    seq = df_scaled[feats].values
    labels = df["SOC_ZHU"].values
    total = len(seq)
    
    # Reduce data: erste 100k + letzte 100k
    if total > 200000:
        first_100k = slice(0, 100000)
        last_100k = slice(-100000, None)
        
        seq_reduced = np.concatenate([seq[first_100k], seq[last_100k]], axis=0)
        labels_reduced = np.concatenate([labels[first_100k], labels[last_100k]], axis=0)
        
        print(f"🔍 Reduced Stateful data: {total:,} → {len(seq_reduced):,} (first & last 100k)")
        seq, labels = seq_reduced, labels_reduced
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
    return preds, gts

def predict_windows_sequence(model, df, scaler, feats, device):
    """Führt Windows Prediction durch (reduziert auf erste & letzte 100k)"""
    model.eval()
    
    # Scale features
    df_scaled = df.copy()
    df_scaled[feats] = scaler.transform(df[feats])
    
    # Get sequence data
    seq = df_scaled[feats].values
    labels = df["SOC_ZHU"].values
    total = len(seq)
    
    # Reduce data: erste 100k + letzte 100k
    if total > 200000:
        first_100k = slice(0, 100000)
        last_100k = slice(-100000, None)
        
        seq_reduced = np.concatenate([seq[first_100k], seq[last_100k]], axis=0)
        labels_reduced = np.concatenate([labels[first_100k], labels[last_100k]], axis=0)
        
        print(f"🔍 Reduced Windows data: {total:,} → {len(seq_reduced):,} (first & last 100k)")
        seq, labels = seq_reduced, labels_reduced
        total = len(seq)
    
    n_windows = math.ceil(total / WINDOWS_WINDOW_SIZE)
    print(f"🪟 Windows Prediction: {total:,} timesteps in {n_windows} windows...")
    
    preds = []

    with torch.no_grad():
        progress_bar = tqdm(range(n_windows), desc="🪟 Windows Inference", 
                           unit="window", ncols=100, 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for i in progress_bar:
            s = i * WINDOWS_WINDOW_SIZE
            e = min(s + WINDOWS_WINDOW_SIZE, total)
            
            # Für jeden Window: neue hidden states (kein State zwischen Windows!)
            h, c = init_hidden_windows(model, batch_size=1, device=device)
            h, c = h.contiguous(), c.contiguous()
            
            window_data = torch.tensor(seq[s:e], dtype=torch.float32, device=device).unsqueeze(0)
            window_data = window_data.contiguous()
            
            model.lstm.flatten_parameters()
            with torch.backends.cudnn.flags(enabled=False):
                out, _ = model(window_data, (h, c))
            
            preds.extend(out.squeeze(0).cpu().numpy())
            
            progress_bar.set_postfix({
                'Samples': f'{e:,}',
                'Progress': f'{(e/total)*100:.1f}%'
            })
    
    preds = np.array(preds)
    # Fix: Ensure preds and gts have the same length
    min_length = min(len(preds), len(labels))
    preds = preds[:min_length]
    gts = labels[:min_length]
    
    print(f"✅ Windows prediction completed! Generated {len(preds):,} predictions.")
    return preds, gts

def create_comprehensive_comparison_plots(stateful_info, windows_info, stateful_results, windows_results, save_dir):
    """Erstellt wissenschaftliche Vergleichsplots zwischen beiden Modellen"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Set scientific plot style with LARGER fonts for PowerPoint
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 18,              # Base font size (erhöht von 16)
        'axes.labelsize': 20,         # Achsenbeschriftung (erhöht von 18)
        'axes.titlesize': 22,         # Plot titles (erhöht von 20)
        'xtick.labelsize': 16,        # X-Achsen Ticks (erhöht von 14)
        'ytick.labelsize': 16,        # Y-Achsen Ticks (erhöht von 14)
        'legend.fontsize': 16,        # Legend (erhöht von 14)
        'figure.titlesize': 26,       # Figure title (erhöht von 24)
        'axes.linewidth': 1.5,        # Dickere Achsenlinien
        'grid.linewidth': 1.0,        # Dickere Grid-Linien
        'lines.linewidth': 2.5,       # Dickere Plot-Linien
        'axes.labelpad': 8,           # Mehr Abstand zu Achsenbeschriftungen
        'xtick.major.size': 6,        # Größere Tick-Marks
        'ytick.major.size': 6,        # Größere Tick-Marks
        'xtick.major.pad': 6,         # Mehr Abstand für Tick-Labels
        'ytick.major.pad': 6          # Mehr Abstand für Tick-Labels
    })
    
    # === COMPARISON PLOT 1: Memory Usage Side-by-Side ===
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Memory Components Comparison (Stacked Bar Chart)
    models = ['Stateful LSTM\n(1.2.4.36)', 'Windows LSTM\n(2.0.1)']
    
    # Memory components (KB) - dynamisch basierend auf echten Modell-Parametern
    model_weights = [stateful_info['model']['memory_kb'], windows_info['model']['memory_kb']]
    
    # Hidden States - berechne basierend auf echten Modell-Parametern  
    stateful_hidden_size = stateful_info['ops']['hidden_size']
    stateful_num_layers = stateful_info['ops']['num_layers']
    windows_hidden_size = windows_info['ops']['hidden_size'] 
    windows_num_layers = windows_info['ops']['num_layers']
    
    hidden_states = [
        (stateful_hidden_size * stateful_num_layers * 2 * 4) / 1024,  # Stateful h+c
        (windows_hidden_size * windows_num_layers * 2 * 4) / 1024     # Windows h+c
    ]
    
    input_tensors = [
        (STATEFUL_SEQ_CHUNK_SIZE * 4 * 4) / 1024,  # Stateful chunk
        (WINDOWS_WINDOW_SIZE * 4 * 4) / 1024       # Windows window
    ]
    framework_overhead = [5, 5]  # Estimated framework overhead
    
    width = 0.6
    x = np.arange(len(models))
    
    p1 = ax1.bar(x, model_weights, width, label='Model Weights', color=COLOR_SCHEME['stateful_lstm'], alpha=0.8)
    p2 = ax1.bar(x, hidden_states, width, bottom=model_weights, label='Hidden States', color=COLOR_SCHEME['windows_lstm'], alpha=0.8)
    p3 = ax1.bar(x, input_tensors, width, bottom=np.array(model_weights)+np.array(hidden_states), 
                label='Input Tensors', color=COLOR_SCHEME['blue_accent'], alpha=0.8)
    p4 = ax1.bar(x, framework_overhead, width, 
                bottom=np.array(model_weights)+np.array(hidden_states)+np.array(input_tensors),
                label='Framework Overhead', color=COLOR_SCHEME['stateful_secondary'], alpha=0.8)
    
    ax1.axhline(y=STM32_RAM_BYTES/1024, color=COLOR_SCHEME['error_color'], linestyle='--', alpha=0.7, label='RAM Limit (1024 KB)')
    ax1.set_ylabel('Memory Usage (KB)')
    ax1.set_title('Memory Usage Comparison: Stateful vs Windows LSTM')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars with LARGER font
    for i, (w, h, t, f) in enumerate(zip(model_weights, hidden_states, input_tensors, framework_overhead)):
        total = w + h + t + f
        ax1.text(i, total + 5, f'{total:.1f} KB', ha='center', va='bottom', 
                fontsize=18)
    
    # Computational Load Comparison - PIE CHARTS
    # Create pie charts for computational operations distribution
    stateful_ops = [stateful_info['ops']['lstm_ops'], stateful_info['ops']['mlp_ops']]
    windows_ops = [windows_info['ops']['lstm_ops'], windows_info['ops']['mlp_ops']]
    
    ops_labels = ['LSTM Ops', 'MLP Ops']
    colors = [COLOR_SCHEME['stateful_32'], COLOR_SCHEME['windows_32']]  # stateful_32 bleibt grün, windows_32 wird rot
    
    # Create pie chart for combined comparison
    total_stateful = sum(stateful_ops)
    total_windows = sum(windows_ops)
    
    # Create a combined comparison showing distribution
    combined_ops = [total_stateful, total_windows]
    model_labels = ['Stateful LSTM', 'Windows LSTM']
    model_colors = [COLOR_SCHEME['stateful_lstm'], COLOR_SCHEME['windows_lstm']]
    
    wedges, texts, autotexts = ax2.pie(combined_ops, labels=model_labels, colors=model_colors, 
                                      autopct='%1.1f%%', startangle=90)
    
    # Add operation counts to the labels with LARGER font
    for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
        autotext.set_text(f'{combined_ops[i]:,}\nops\n({autotext.get_text()})')
        autotext.set_fontsize(16)  # Größer: war 10
    
    ax2.set_title('Computational Load Comparison\n(Total MAC Operations per Inference)')
    
    # Add legend with detailed breakdown
    legend_labels = [f'{model_labels[i]}: {combined_ops[i]:,} ops' for i in range(len(model_labels))]
    ax2.legend(wedges, legend_labels, loc="upper left", bbox_to_anchor=(1, 1))
    
    # Performance Metrics Comparison
    sampling_rates = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    stateful_cpu_loads = []
    windows_cpu_loads = []
    stateful_energy = []
    windows_energy = []
    
    for sr in sampling_rates:
        s_metrics = calculate_mcu_performance_metrics(stateful_info['ops']['total_ops'], sr)
        w_metrics = calculate_mcu_performance_metrics(windows_info['ops']['total_ops'], sr)
        
        stateful_cpu_loads.append(s_metrics['cpu_load_percent'])
        windows_cpu_loads.append(w_metrics['cpu_load_percent'])
        stateful_energy.append(s_metrics['energy_per_hour_wh'] * 1000)  # mWh
        windows_energy.append(w_metrics['energy_per_hour_wh'] * 1000)   # mWh
    
    # Debug: Check if values are identical
    print(f"🔍 DEBUG CPU Loads:")
    print(f"   Stateful: {stateful_cpu_loads}")
    print(f"   Windows:  {windows_cpu_loads}")
    print(f"   Identical? {stateful_cpu_loads == windows_cpu_loads}")
    
    # CPU Load with thicker lines for PowerPoint visibility
    ax3.plot(sampling_rates, stateful_cpu_loads, 'o-', linewidth=4, markersize=10,
            label='Stateful LSTM', color=COLOR_SCHEME['stateful_lstm'], alpha=0.8)
    ax3.plot(sampling_rates, windows_cpu_loads, 's--', linewidth=3, markersize=8,
            label='Windows LSTM', color=COLOR_SCHEME['windows_lstm'], alpha=0.9)
    
    ax3.set_xlabel('Sampling Rate (Hz)')
    ax3.set_ylabel('CPU Load (%)')
    ax3.set_title('CPU Load vs Sampling Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Energy Consumption with thicker lines for PowerPoint visibility
    ax4.plot(sampling_rates, stateful_energy, 'o-', linewidth=4, markersize=10,
            label='Stateful LSTM', color=COLOR_SCHEME['stateful_lstm'], alpha=0.8)
    ax4.plot(sampling_rates, windows_energy, 's--', linewidth=3, markersize=8,
            label='Windows LSTM', color=COLOR_SCHEME['windows_lstm'], alpha=0.9)
    
    ax4.set_xlabel('Sampling Rate (Hz)')
    ax4.set_ylabel('Energy Consumption (mWh/hour)')
    ax4.set_title('Energy Efficiency Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comprehensive_comparison_microcontroller.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # === COMPARISON PLOT 2: Prediction Accuracy Comparison ===
    fig = plt.figure(figsize=(20, 18))
    # Neues Layout: 3 Zeilen
    # Zeile 1: Stateful Time Series (volle Breite)
    # Zeile 2: Windows Time Series (volle Breite) 
    # Zeile 3: 3 Plots nebeneinander (Scatter, Error, Metrics)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], width_ratios=[1, 1, 1])
    
    # Create 5 subplots: 
    ax1 = fig.add_subplot(gs[0, :])   # Stateful LSTM time series (row 0, spans all columns)
    ax2 = fig.add_subplot(gs[1, :])   # Windows LSTM time series (row 1, spans all columns)
    ax3 = fig.add_subplot(gs[2, 0])   # Scatter plot comparison (row 2, col 0)
    ax4 = fig.add_subplot(gs[2, 1])   # Error distribution (row 2, col 1)
    ax5 = fig.add_subplot(gs[2, 2])   # Metrics comparison (row 2, col 2)
    
    # Extract data from results
    stateful_preds, stateful_gts = stateful_results['preds'], stateful_results['gts']
    windows_preds, windows_gts = windows_results['preds'], windows_results['gts']
    
    # Flatten arrays to ensure they are 1D
    stateful_preds_flat = stateful_preds.flatten()
    stateful_gts_flat = stateful_gts.flatten()
    windows_preds_flat = windows_preds.flatten()
    windows_gts_flat = windows_gts.flatten()
    
    # Use same length for comparison
    min_len = min(len(stateful_preds_flat), len(windows_preds_flat))
    sample_size = min(20000, min_len)
    sample_indices = np.linspace(0, min_len-1, sample_size, dtype=int)
    
    # Plot 1: Stateful LSTM time series with thicker lines
    ax1.plot(range(sample_size), stateful_gts_flat[sample_indices], label='Ground Truth', 
             color='black', alpha=0.8, linewidth=2.5)
    ax1.plot(range(sample_size), stateful_preds_flat[sample_indices], label='Stateful LSTM', 
             color=COLOR_SCHEME['stateful_lstm'], alpha=0.9, linewidth=2)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('SOC')
    ax1.set_title(f'Stateful LSTM SOC Prediction\n(Sample of {sample_size:,} points)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Windows LSTM time series with thicker lines
    ax2.plot(range(sample_size), windows_gts_flat[sample_indices], label='Ground Truth', 
             color='black', alpha=0.8, linewidth=2.5)
    ax2.plot(range(sample_size), windows_preds_flat[sample_indices], label='Windows LSTM', 
             color=COLOR_SCHEME['windows_lstm'], alpha=0.9, linewidth=2)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('SOC')
    ax2.set_title(f'Windows LSTM SOC Prediction\n(Sample of {sample_size:,} points)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plots comparison
    scatter_sample = min(5000, min_len)
    scatter_indices = np.random.choice(min_len, scatter_sample, replace=False)
    
    ax3.scatter(stateful_gts_flat[scatter_indices], stateful_preds_flat[scatter_indices], 
               alpha=0.7, s=4, label='Stateful LSTM', color=COLOR_SCHEME['stateful_lstm'])
    ax3.scatter(windows_gts_flat[scatter_indices], windows_preds_flat[scatter_indices], 
               alpha=0.7, s=4, label='Windows LSTM', color=COLOR_SCHEME['windows_lstm'])
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.8, linewidth=2)
    ax3.set_xlabel('Ground Truth SOC')
    ax3.set_ylabel('Predicted SOC')
    ax3.set_title('Prediction Accuracy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution comparison
    # Use same length for fair comparison
    stateful_preds_flat = stateful_preds_flat[:min_len]
    stateful_gts_flat = stateful_gts_flat[:min_len]
    windows_preds_flat = windows_preds_flat[:min_len]
    windows_gts_flat = windows_gts_flat[:min_len]
    
    stateful_errors = stateful_preds_flat - stateful_gts_flat
    windows_errors = windows_preds_flat - windows_gts_flat
    
    ax4.hist(stateful_errors, bins=100, alpha=0.6, color=COLOR_SCHEME['stateful_lstm'], 
            label=f'Stateful (σ={np.std(stateful_errors):.4f})', density=True)
    ax4.hist(windows_errors, bins=100, alpha=0.6, color=COLOR_SCHEME['windows_lstm'], 
            label=f'Windows (σ={np.std(windows_errors):.4f})', density=True)
    ax4.axvline(np.mean(stateful_errors), color=COLOR_SCHEME['stateful_32'], linestyle='--', alpha=0.8)
    ax4.axvline(np.mean(windows_errors), color=COLOR_SCHEME['windows_32'], linestyle='--', alpha=0.8)
    ax4.set_xlabel('Prediction Error (Predicted - True)')
    ax4.set_ylabel('Density')
    ax4.set_title('Error Distribution Comparison')
    ax4.set_xlim(-0.2, 0.2)  # Zoom in on error range for better visibility
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Metrics comparison bar chart
    stateful_mae = mean_absolute_error(stateful_gts_flat, stateful_preds_flat)
    stateful_rmse = np.sqrt(mean_squared_error(stateful_gts_flat, stateful_preds_flat))
    stateful_r2 = r2_score(stateful_gts_flat, stateful_preds_flat)
    
    windows_mae = mean_absolute_error(windows_gts_flat, windows_preds_flat)
    windows_rmse = np.sqrt(mean_squared_error(windows_gts_flat, windows_preds_flat))
    windows_r2 = r2_score(windows_gts_flat, windows_preds_flat)
    
    metrics = ['MAE', 'RMSE', 'R²']
    stateful_values = [stateful_mae, stateful_rmse, stateful_r2]
    windows_values = [windows_mae, windows_rmse, windows_r2]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, stateful_values, width, label='Stateful LSTM', 
                   color=COLOR_SCHEME['stateful_lstm'], alpha=0.8)
    bars2 = ax5.bar(x + width/2, windows_values, width, label='Windows LSTM', 
                   color=COLOR_SCHEME['windows_lstm'], alpha=0.8)
    
    ax5.set_ylabel('Metric Value')
    ax5.set_title('Accuracy Metrics Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels with LARGER font
    for bars, values in [(bars1, stateful_values), (bars2, windows_values)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comprehensive_comparison_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Comprehensive comparison plots created!")
    return {
        'stateful_metrics': {'mae': stateful_mae, 'rmse': stateful_rmse, 'r2': stateful_r2},
        'windows_metrics': {'mae': windows_mae, 'rmse': windows_rmse, 'r2': windows_r2}
    }

def create_comparison_summary_table(stateful_info, windows_info, accuracy_comparison, save_dir):
    """Erstellt eine wissenschaftliche Vergleichstabelle"""
    save_dir = Path(save_dir)
    
    # Performance metrics at 1Hz
    stateful_perf = calculate_mcu_performance_metrics(stateful_info['ops']['total_ops'], 1.0)
    windows_perf = calculate_mcu_performance_metrics(windows_info['ops']['total_ops'], 1.0)
    
    # Create comparison table
    comparison_data = {
        'Metric': [
            'Model Parameters',
            'Memory Usage (KB)',
            'Flash Storage (KB)', 
            'MAC Operations/Inference',
            'Inference Time (μs)',
            'CPU Load @ 1Hz (%)',
            'Energy @ 1Hz (mWh/h)',
            'Max Sampling Rate (Hz)',
            'MAE',
            'RMSE', 
            'R²',
            'RAM Feasibility',
            'Flash Feasibility'
        ],
        'Stateful LSTM (1.2.4.36)': [
            f"{stateful_info['model']['total_params']:,}",
            f"{stateful_info['model']['memory_kb']:.1f}",
            f"{stateful_info['model']['memory_kb']:.1f}",
            f"{stateful_info['ops']['total_ops']:,}",
            f"{stateful_perf['time_per_inference_us']:.1f}",
            f"{stateful_perf['cpu_load_percent']:.3f}",
            f"{stateful_perf['energy_per_hour_wh']*1000:.2f}",
            f"{stateful_perf['max_sampling_rate_hz']:.1f}",
            f"{accuracy_comparison['stateful_metrics']['mae']:.6f}",
            f"{accuracy_comparison['stateful_metrics']['rmse']:.6f}",
            f"{accuracy_comparison['stateful_metrics']['r2']:.6f}",
            "✅ FEASIBLE" if stateful_info['model']['memory_kb'] < STM32_RAM_BYTES/1024 else "❌ TOO LARGE",
            "✅ FEASIBLE" if stateful_info['model']['memory_kb'] < STM32_FLASH_BYTES/1024 else "❌ TOO LARGE"
        ],
        'Windows LSTM (2.0.1)': [
            f"{windows_info['model']['total_params']:,}",
            f"{windows_info['model']['memory_kb']:.1f}",
            f"{windows_info['model']['memory_kb']:.1f}",
            f"{windows_info['ops']['total_ops']:,}",
            f"{windows_perf['time_per_inference_us']:.1f}",
            f"{windows_perf['cpu_load_percent']:.3f}",
            f"{windows_perf['energy_per_hour_wh']*1000:.2f}",
            f"{windows_perf['max_sampling_rate_hz']:.1f}",
            f"{accuracy_comparison['windows_metrics']['mae']:.6f}",
            f"{accuracy_comparison['windows_metrics']['rmse']:.6f}",
            f"{accuracy_comparison['windows_metrics']['r2']:.6f}",
            "✅ FEASIBLE" if windows_info['model']['memory_kb'] < STM32_RAM_BYTES/1024 else "❌ TOO LARGE",
            "✅ FEASIBLE" if windows_info['model']['memory_kb'] < STM32_FLASH_BYTES/1024 else "❌ TOO LARGE"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save to CSV
    df_comparison.to_csv(save_dir / 'comprehensive_comparison_table.csv', index=False)
    
    print("📊 Comprehensive Comparison Table:")
    print("=" * 120)
    print(df_comparison.to_string(index=False))
    print("=" * 120)
    
    return df_comparison

def main():
    """Hauptfunktion für die umfassende Vergleichsanalyse"""
    
    print("🔬 Starting Comprehensive Comparison: Stateful vs Windows LSTM")
    print("=" * 80)    # === STATEFUL LSTM LADEN ===
    print("\n📥 Loading Stateful LSTM Model...")
    stateful_model_path = Path(r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32_comparison\PC_only\models_for_PC_only\best_model_sf_32")
    if not stateful_model_path.exists():
        raise FileNotFoundError(f"Stateful model not found: {stateful_model_path}")
    
    stateful_model = StatefulSOCModel(input_size=4, dropout=0.03)
    stateful_model.load_state_dict(torch.load(stateful_model_path, map_location=device, weights_only=False))
    stateful_model.to(device)
    stateful_model.eval()
    print("✅ Stateful LSTM loaded!")    # === WINDOWS LSTM LADEN ===
    print("\n📥 Loading Windows LSTM Model...")
    windows_model_path = Path(r"C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32_comparison\PC_only\models_for_PC_only\best_model_windows_32")
    if not windows_model_path.exists():
        raise FileNotFoundError(f"Windows model not found: {windows_model_path}")
    
    windows_model = WindowsSOCModel(input_size=4, num_layers=1, dropout=0.1)
    state_dict = torch.load(windows_model_path, map_location=device, weights_only=False)
    windows_model.load_state_dict(state_dict)
    windows_model.to(device)
    windows_model.eval()
    print("✅ Windows LSTM loaded!")
    
    # === MODELL-ANALYSE ===
    print("\n🔍 Analyzing Models...")
    stateful_model_info = count_model_parameters(stateful_model, "stateful")
    stateful_ops_info = estimate_inference_ops(stateful_model, "stateful")
    
    windows_model_info = count_model_parameters(windows_model, "windows")
    windows_ops_info = estimate_inference_ops(windows_model, "windows")
    
    stateful_info = {'model': stateful_model_info, 'ops': stateful_ops_info}
    windows_info = {'model': windows_model_info, 'ops': windows_ops_info}
    
    print(f"   Stateful: {stateful_model_info['total_params']:,} params, {stateful_ops_info['total_ops']:,} ops")
    print(f"   Windows:  {windows_model_info['total_params']:,} params, {windows_ops_info['total_ops']:,} ops")
    
    # === TESTDATEN LADEN ===
    print("\n📊 Loading Test Data...")
    df = load_test_data()
    
    # === SCALER ERSTELLEN ===
    stateful_scaler, stateful_feats = create_stateful_scaler()
    windows_scaler, windows_feats = create_windows_scaler()
    
    # === PREDICTIONS DURCHFÜHREN ===
    print("\n🔮 Running Predictions...")
    stateful_preds, stateful_gts = predict_stateful_sequence(stateful_model, df, stateful_scaler, stateful_feats, device)
    windows_preds, windows_gts = predict_windows_sequence(windows_model, df, windows_scaler, windows_feats, device)
    
    stateful_results = {'preds': stateful_preds, 'gts': stateful_gts}
    windows_results = {'preds': windows_preds, 'gts': windows_gts}
    
    # === VERGLEICHSPLOTS ERSTELLEN ===
    print("\n📊 Creating Comprehensive Comparison Plots...")
    save_dir = Path("comprehensive_comparison_results")
    save_dir.mkdir(exist_ok=True)
    
    accuracy_comparison = create_comprehensive_comparison_plots(
        stateful_info, windows_info, stateful_results, windows_results, save_dir
    )
    
    # === VERGLEICHSTABELLE ERSTELLEN ===
    print("\n📋 Creating Comparison Summary Table...")
    comparison_table = create_comparison_summary_table(
        stateful_info, windows_info, accuracy_comparison, save_dir
    )
    
    # === ZUSAMMENFASSUNG ===
    print(f"\n🎯 COMPREHENSIVE COMPARISON SUMMARY:")
    print(f"=" * 80)
    print(f"📊 STATEFUL LSTM (1.2.4.36):")
    print(f"   Memory: {stateful_info['model']['memory_kb']:.1f} KB | Ops: {stateful_info['ops']['total_ops']:,}")
    print(f"   Accuracy: MAE={accuracy_comparison['stateful_metrics']['mae']:.6f}, R²={accuracy_comparison['stateful_metrics']['r2']:.6f}")
    
    print(f"\n📊 WINDOWS LSTM (2.0.1):")
    print(f"   Memory: {windows_info['model']['memory_kb']:.1f} KB | Ops: {windows_info['ops']['total_ops']:,}")
    print(f"   Accuracy: MAE={accuracy_comparison['windows_metrics']['mae']:.6f}, R²={accuracy_comparison['windows_metrics']['r2']:.6f}")
    
    # Deployment feasibility
    stateful_feasible = (stateful_info['model']['memory_kb'] < STM32_RAM_BYTES/1024)
    windows_feasible = (windows_info['model']['memory_kb'] < STM32_RAM_BYTES/1024)
    
    print(f"\n🎯 STM32H757 DEPLOYMENT FEASIBILITY:")
    print(f"   Stateful LSTM: {'🚀 FEASIBLE' if stateful_feasible else '⚠️  CHALLENGING'}")
    print(f"   Windows LSTM:  {'🚀 FEASIBLE' if windows_feasible else '⚠️  CHALLENGING'}")
    
    print(f"\n📁 All comparison results saved to: {save_dir.absolute()}")
    print(f"=" * 80)

if __name__ == "__main__":
    main()
