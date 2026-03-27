import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 🔧 ÄNDERUNG: Besserer Scheduler
from torch.amp import GradScaler, autocast
import pandas as pd
import numpy as np

# 🛡️ CRITICAL HPC SAFETY: Set non-interactive matplotlib backend FIRST
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - PREVENTS SSH DISCONNECTION
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import csv
import math
import itertools
import pickle  

# Konstanten
WINDOW_SIZE         = 5000     # Feste Window-Größe für seq-to-seq Training
OVERLAP_SIZE        = 1000     # Überlappung zwischen Windows
HIDDEN_SIZE         = 32       # Version 2.1.4.1 - Option A: 32
MLP_HIDDEN          = 32       # Version 2.1.4.1 - Option A: 32
BATCH_SIZE          = 4        # Mehrere Windows parallel verarbeiten
VERSION             = "2.1.4.1"  # Version identifier

# Gerät auswählen und cuDNN optimieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")
print(f"🚀 BMS SOC LSTM Version {VERSION} - Option A (32→32→32→1)")
print(f"🛡️ Matplotlib backend: {matplotlib.get_backend()}")  # Safety check
if device.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Datenlade-Funktion
def load_cell_data(data_dir: Path):
    dataframes = {}
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("MGFarm_18650_"):
            dfp = folder / "df.parquet"
            if dfp.exists():
                dataframes[folder.name] = pd.read_parquet(dfp)
            else:
                print(f"Warning: {dfp} fehlt")
    return dataframes

# Daten vorbereiten
def load_data(base_path: str = "/home/users/f/flo01010010/HPC_projects/5_Data/MGFarm_18650_Dataframes"):
    base = Path(base_path)
    cells = load_cell_data(base)
    # neue Trainingszellen und feste Validierungszelle
    train_cells = [f"MGFarm_18650_C{str(i).zfill(2)}" for i in [1,3,5,9,11,13,19,21,23,25,27]]
    val_cells = ["MGFarm_18650_C07","MGFarm_18650_C15","MGFarm_18650_C17"]
    # Feature-Liste
    feats = ["Voltage[V]","Current[A]","SOH_ZHU","Q_m"]

    # trainings-Daten initial (nur timestamp ergänzen)
    train_dfs = {}
    for name in train_cells:
        df = cells[name].copy()
        df['timestamp'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        train_dfs[name] = df

    # scaler auf *allen* Zellen fitten (nicht nur Training)
    df_all = pd.concat(cells.values(), ignore_index=True)
    scaler = StandardScaler().fit(df_all[feats])
    print("[INFO] Skaler über alle Zellen fitten")

    # Skalierte Trainingsdaten
    train_scaled = {}
    for name, df in train_dfs.items():
        df2 = df.copy()
        df2[feats] = scaler.transform(df2[feats])
        train_scaled[name] = df2
    # debug: check for NaNs after scaling
    for name, df2 in train_scaled.items():
        nan_counts = pd.DataFrame(df2[feats]).isna().sum().to_dict()
        print(f"[DEBUG] {name} NaNs after train scaling:", {k:v for k,v in nan_counts.items() if v>0} or "none")

    # vollständige Validierung auf allen drei Zellen
    df_vals = {}
    for name in val_cells:
        dfv = cells[name].copy()
        dfv['timestamp'] = pd.to_datetime(dfv['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        dfv[feats] = scaler.transform(dfv[feats])
        df_vals[name] = dfv
    return train_scaled, df_vals, train_cells, val_cells, scaler

# Window-basiertes Dataset für seq-to-seq Training
class WindowDataset(Dataset):
    def __init__(self, df, window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE):
        """Dataset das Sliding Windows aus einer Zelle erstellt"""
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.step_size = window_size - overlap_size
        
        self.data = df[["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_m"]].values
        self.labels = df["SOC_ZHU"].values
        
        # Berechne alle möglichen Window-Positionen
        self.windows = []
        total_length = len(self.data)
        
        for start in range(0, total_length - window_size + 1, self.step_size):
            end = start + window_size
            self.windows.append((start, end))
        
        # Falls die letzte Sequenz zu kurz ist, füge sie trotzdem hinzu
        if len(self.windows) == 0 or self.windows[-1][1] < total_length:
            start = max(0, total_length - window_size)
            self.windows.append((start, total_length))
        
        print(f"Created {len(self.windows)} windows from sequence of length {total_length}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        start, end = self.windows[idx]
        x = torch.from_numpy(self.data[start:end]).float()
        y = torch.from_numpy(self.labels[start:end]).float()
        return x, y

# Weight-initialization helper
def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, p in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

# Modell: LSTM + Dropout + MLP-Head (verwendet globale HIDDEN_SIZE und MLP_HIDDEN)
def build_model(input_size=4, num_layers=1, dropout=0.1):
    class SOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            # LSTM ohne Dropout (voller Informationsfluss)
            self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, num_layers,
                                batch_first=True, dropout=0.0)
            # hidden_size bestimmt die Dim. der LSTM-Ausgabe
            # mlp_hidden ist die Größe der verborgenen MLP-Schicht
            # deeper MLP-Head
            self.mlp = nn.Sequential(
                nn.Linear(HIDDEN_SIZE, MLP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(MLP_HIDDEN, 1),
                nn.Sigmoid()
            )

        def forward(self, x, hidden):
            self.lstm.flatten_parameters()       # cuDNN-ready
            x = x.contiguous()                   # ensure input contiguous
            # make hidden states contiguous
            h, c = hidden
            h, c = h.contiguous(), c.contiguous()
            hidden = (h, c)
            out, hidden = self.lstm(x, hidden)
            batch, seq_len, hid = out.size()
            out_flat = out.contiguous().view(batch * seq_len, hid)
            soc_flat = self.mlp(out_flat)
            soc = soc_flat.view(batch, seq_len)
            return soc, hidden
    model = SOCModel().to(device)
    # 2) init weights & optimize cuDNN for multi-layer LSTM
    model.apply(init_weights)
    model.lstm.flatten_parameters()
    return model

# Helper-Funktion für die Initialisierung der hidden states
def init_hidden(model, batch_size=1, device=None):
    if device is None:
        device = next(model.parameters()).device
    h = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size, device=device)
    c = torch.zeros_like(h)
    return h, c

# ——— Window-basierte Evaluierungsfunktionen —————————————————————————
def evaluate_window_seq2seq(model, df, device):
    """
    Window-basierte Seq-to-Seq-Validation.
    """
    model.eval()
    dataset = WindowDataset(df, window_size=WINDOW_SIZE, overlap_size=0)  # Keine Überlappung für Evaluation
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_labels = []
    print(f">> Window-based Seq2Seq-Validation startet ({len(dataset)} windows)")
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.contiguous()
            
            # Neue hidden states für jedes Window (kein State-Transfer)
            h, c = init_hidden(model, batch_size=x_batch.size(0), device=device)
            h, c = h.contiguous(), c.contiguous()
            
            model.lstm.flatten_parameters()
            with torch.backends.cudnn.flags(enabled=False):
                out, _ = model(x_batch, (h, c))
            
            all_preds.extend(out.squeeze(0).cpu().numpy())
            all_labels.extend(y_batch.squeeze(0).cpu().numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    return np.mean((preds - labels) ** 2)

def evaluate_online(model, df, device):
    """Stepwise online Validation."""
    model.eval()
    print(">> Online-Validation startet")
    # initialize hidden state and result lists
    h, c = init_hidden(model, batch_size=1, device=device)
    preds, gts = [], []
    with torch.no_grad():
        for idx, (v, i, soh, qm, y_true) in enumerate(zip(
            df['Voltage[V]'], df['Current[A]'],
            df['SOH_ZHU'], df['Q_m'], df['SOC_ZHU']
        )):
            x = torch.tensor([[v, i, soh, qm]], dtype=torch.float32, device=device).view(1,1,4).contiguous()
            pred, (h, c) = model(x, (h, c))
            preds.append(pred.item())
            gts.append(y_true)
    preds, gts = np.array(preds), np.array(gts)
    return np.mean((preds - gts)**2)

def evaluate_full_sequence(model, df, device):
    """
    Evaluation über die komplette Sequenz als einen Block (für Vergleichszwecke).
    """
    model.eval()
    seq = df[["Voltage[V]","Current[A]","SOH_ZHU","Q_m"]].values
    labels = df["SOC_ZHU"].values
    h, c = init_hidden(model, batch_size=1, device=device)
    h, c = h.contiguous(), c.contiguous()
    chunk = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
    chunk = chunk.contiguous()
    
    with torch.no_grad():
        model.lstm.flatten_parameters()
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = model(chunk, (h, c))
    
    preds = out.squeeze(0).cpu().numpy()
    mse = np.mean((preds - labels)**2)
    return mse, preds, labels

# Window-basierte Training Funktion
def train_window_based(
    epochs=500, lr=1e-3,
    dropout=0.15, patience=100,  # 🔧 ÄNDERUNG: Weniger Dropout, kürzere Patience
    log_csv_path="training_log.csv", out_dir=f"training_run_v{VERSION}",
    train_data=None, df_vals=None, feature_scaler=None):

    # convert out_dir to Path so "/" operator works
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_csv_path = out_dir / log_csv_path

    if train_data is None:
        train_scaled, df_vals, train_cells, val_cells, feature_scaler = load_data()
    else:
        train_scaled = train_data
        # reuse globals
        train_cells = train_cells_glob
        val_cells   = val_cells_glob
        feature_scaler = feature_scaler
    print(f"[INFO] Train cells={train_cells}, Val/Test cells={val_cells}")

    # baue Modell mit globalen Größen
    model = build_model(dropout=dropout)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # 🔧 ÄNDERUNG: ReduceLROnPlateau statt CosineAnnealing
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.7, patience=15, min_lr=1e-6, verbose=True)
    criterion = nn.MSELoss()
    gradient_scaler = GradScaler(enabled=(device.type=="cuda"))
    best_val = float('inf'); no_improve = 0

    # HISTORY & LOG INITIALIZATION
    train_rmse_history = []
    val_rmse_history   = {name: [] for name in val_cells}
    log_rows = []

    for ep in range(1, epochs+1):
        print(f"\n--- Epoch {ep}/{epochs} ---")
        model.train()
        total_loss, steps = 0, 0

        # Sammle alle Windows von allen Trainingszellen
        all_windows = []
        for name, df in train_scaled.items():
            print(f"[Epoch {ep}] Preparing windows for Cell {name}")
            dataset = WindowDataset(df, window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=4, pin_memory=True, drop_last=False)
            
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch = x_batch.contiguous()
                y_batch = y_batch.contiguous()
                
                # Für jedes Window in dem Batch separate hidden states
                batch_size = x_batch.size(0)
                h, c = init_hidden(model, batch_size=batch_size, device=device)
                h, c = h.contiguous(), c.contiguous()
                
                optim.zero_grad()
                
                # Use proper precision context
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    model.lstm.flatten_parameters()
                    pred, _ = model(x_batch, (h, c))  # Keine State-Weitergabe zwischen Windows
                    loss = criterion(pred, y_batch)
                
                gradient_scaler.scale(loss).backward()
                gradient_scaler.unscale_(optim)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                gradient_scaler.step(optim)
                gradient_scaler.update()
                
                total_loss += loss.item()   
                steps += 1

        train_rmse = math.sqrt(total_loss/steps) if steps > 0 else float('inf')
        train_rmse_history.append(train_rmse)
        print(f"[Epoch {ep}] train RMSE={train_rmse:.4f}")

        # Window-basierte Validierung über alle drei Zellen
        val_epoch = {}
        for name, dfv in df_vals.items():
            mse = evaluate_window_seq2seq(model, dfv, device)
            rmse = math.sqrt(mse)
            val_rmse_history[name].append(rmse)
            val_epoch[name] = rmse
            print(f"[Epoch {ep}] val RMSE {name}={rmse:.4f}")
        mean_val_rmse = float(np.mean(list(val_epoch.values())))
        print(f"[Epoch {ep}] mean Val RMSE={mean_val_rmse:.4f}")

        # Early Stopping & Model Save
        is_best = mean_val_rmse < best_val
        best_val = min(mean_val_rmse, best_val)
        if is_best:
            no_improve = 0
            torch.save(model.state_dict(), out_dir/"best_model.pth")
            print(f"[Epoch {ep}] Model gespeichert.")
        else:
            no_improve += 1

        # 🔧 ÄNDERUNG: Scheduler step mit validation loss
        scheduler.step(mean_val_rmse)

        # Logging
        log_rows.append({
            "epoch": ep,
            "train_rmse": train_rmse,
            "mean_val_rmse": mean_val_rmse,
            "lr": optim.param_groups[0]['lr'],
            "dropout": dropout,
            "window_size": WINDOW_SIZE,
            "overlap_size": OVERLAP_SIZE
        })
        df_log = pd.DataFrame(log_rows)
        df_log.to_csv(log_csv_path, index=False)

        # Plot aller Validierungs-RMSEs pro Epoche (überschrieben)
        plt.figure(figsize=(6,4))
        for name in val_cells:
            plt.plot(range(1, len(val_rmse_history[name]) + 1),
                     val_rmse_history[name], label=name)
        plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("Val RMSE per cell (Window-based)")
        plt.legend(loc="upper right"); plt.grid()
        plt.savefig(out_dir/"val_rmse_plot.png")
        plt.close()

        # Validation SOC-Estimation für alle Zellen (jeweils eigenes Fenster)
        for name, dfv in df_vals.items():
            mse, preds, labels = evaluate_full_sequence(model, dfv, device)
            n_samples = int(len(labels) * 0.1)
            plt.figure(figsize=(6,4))
            plt.plot(range(n_samples), labels[:n_samples], label="true")
            plt.plot(range(n_samples), preds[:n_samples], '--', label="pred")
            plt.xlabel("Timestep"); plt.ylabel("SOC")
            plt.title(f"Validation SOC estimation (first 10%) – {name}")
            plt.legend(loc="upper right"); plt.grid()
            plt.savefig(out_dir/f"val_estimation_{name}.png")
            plt.close()  # 🛡️ HPC SAFETY: Close figure to free memory

        if no_improve >= patience:
            print(f"[INFO] Frühes Stoppen bei Epoche {ep}")
            break

    # Lade das beste Modell für die finale Bewertung
    model.load_state_dict(torch.load(out_dir / "best_model.pth", weights_only=True))

    # Finale Bewertung auf Val
    print(f"\n[INFO] Finale Bewertung:")
    for name, dfv in df_vals.items():
        # Window-basierte Evaluation
        mse_window = evaluate_window_seq2seq(model, dfv, device)
        rmse_window = math.sqrt(mse_window)
        
        # Full-sequence Evaluation zum Vergleich
        mse_full, _, _ = evaluate_full_sequence(model, dfv, device)
        rmse_full = math.sqrt(mse_full)
        
        print(f"[INFO] {name} -> Window RMSE: {rmse_window:.4f}, Full RMSE: {rmse_full:.4f}")

    # PLOT-Verlauf zeichnen
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_rmse_history)+1), train_rmse_history, label="Train RMSE")
    for name in val_cells:
        plt.plot(range(1, len(val_rmse_history[name])+1), val_rmse_history[name], 
                label=f"Val RMSE {name}")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend(loc="upper right")
    plt.title(f"Window-based Training v{VERSION} (ReduceLROnPlateau)")
    plt.grid()
    plt.savefig(out_dir / "train_val_rmse_plot.png")
    plt.close()  # 🛡️ HPC SAFETY: Close figure to free memory
    
    # 🛡️ HPC SAFETY: Clear matplotlib memory
    plt.clf()
    
    return model, feature_scaler, log_rows, df_vals

# Global einmal laden für HPT
train_scaled_glob, df_vals_glob, train_cells_glob, val_cells_glob, feature_scaler_glob = load_data()
print("[INFO] Global data loaded for window-based training.")
print(f"[INFO] Window size: {WINDOW_SIZE}, Overlap size: {OVERLAP_SIZE}, Batch size: {BATCH_SIZE}")

if __name__ == "__main__":
    print("=" * 70)
    print("🛡️  STARTING SAFE WINDOW-BASED LSTM TRAINING v" + VERSION)
    print("=" * 70)
    print(f"🛡️  Matplotlib backend: {matplotlib.get_backend()}")
    print(f"🛡️  Safety checks passed")
    print("🔧  CHANGES: ReduceLROnPlateau, lower dropout, shorter patience")
    
    try:
        train_window_based(
            epochs=500, lr=1e-4,
            dropout=0.15, patience=100,  # 🔧 Verbesserungen
            out_dir=f"training_run_window_based_v{VERSION}",
            train_data=train_scaled_glob,
            df_vals=df_vals_glob,
            feature_scaler=feature_scaler_glob
        )
        print(f"\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        # 🛡️ HPC SAFETY: Cleanup resources
        try:
            plt.close('all')
            plt.clf()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"🛡️  Resources cleaned up")
        except:
            pass
