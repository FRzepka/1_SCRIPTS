# SOC Model Overview — soc_epoch0001_rmse0.02897.pt

## Snapshot

| Item | Details |
| --- | --- |
| Checkpoint | `/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/models/torch/1.5.0.0/outputs/checkpoints/soc_epoch0001_rmse0.02897.pt` |
| File size | 276.95 KB |
| Total parameters | 22,657 |
| Window length | 2024 Zeitschritte (seq-to-one) |
| Features | `Voltage[V]`, `Current[A]`, `Temperature[°C]`, `Q_c`, `dU_dt[V/s]`, `dI_dt[A/s]` |
| Targets | SOC des letzten Samples je Fenster (normiert auf [0,1]) |
| Best val. RMSE | 0.02897 (Epoche 1) |

## Architektur

- **Encoder**: Einlagiges LSTM (`hidden_size=64`, batch-first).
- **Head**: MLP `64 → 64 → 1` mit ReLU, Dropout 0.05 und finalem Sigmoid.
- **Initialisierung**: Xavier für alle Linear-Gewichte und LSTM-Matrizen; Bias = 0.
- **Determinismus**: Seeds für Python, NumPy und PyTorch; cudnn deterministic + benchmark deaktiviert.

## Daten & Vorverarbeitung

1. **Quellen**: Parquet-Dateien unter `/0_Data/MGFarm_18650_FE`, geladen pro Zelle (Cxx).
2. **Reinigung**: NaNs, ±Inf entfernen (`SeqDataset`).
3. **Skalierung**: `RobustScaler` fit nur auf Trainingszellen, anschließend auf alle Sequenzen angewendet.
4. **Sequenzierung**: Sliding Window mit Stride 1 ⇒ jedes Sample erzeugt ein frisches Fenster.

## Trainingsregime

- **Task**: seq-to-one Regression, komplett stateless je Batch.
- **Batching & Shuffle**: `DataLoader` mischt alle Fenster pro Epoche (shuffle=True); 512 Fenster werden parallel auf der GPU verarbeitet.
- **Optimierer**: AdamW (lr 1.5e-4, weight decay 1e-4), Batchgröße 512, kein Grad-Accumulation.
- **Stabilität**: Gradient Clipping (1.0), optional Mixed Precision (`torch.amp`).
- **Scheduler**: Cosine Warm Restarts (`T₀=150`, `η_min = lr × 0.05`).
- **Laufzeit**: 120 geplante Epochen; Trainingsverlust (MSE) wird kontinuierlich geloggt (`training_progress.png/json`).

## Validierungsstrategie

- **Splits**:
  - Train: C01, C03, C05, C11, C17, C23.
  - Val: C07, C19, C21.
- **Modus**: Identisch zum Training → seq-to-one, non-stateful. Jeder Validierungsbatch startet mit leerem Hidden-State, `SeqDataset` liefert nur das Ziel des letzten Zeitschritts.
- **Kadenzen**: Evaluierung alle 5 Epochen (inkl. Epoche 1) mit RMSE, MAE, R².
- **Frühes Stoppen**: Mindestwartezeit 100 Validierungszyklen ⇒ faktisch deaktiviert für 120-Epochen-Lauf.
- **Artefakte**: Bester Checkpoint + RobustScaler (`scaler_robust.joblib`), CSV-Log `training_log.csv`.

## Exporte & Deployment

- **Standard-ONNX**: `soc_best_epochXXXX.onnx` (seq-to-one, batches beliebiger Länge).
- **Stateful ONNX**: Wrapper mit `x_step`, `h0`, `c0` erlaubt Streaming und Zustandsweitergabe (siehe `StatefulWrapper`).
- **Manifest**: `export_manifest.json` mit Epoche, RMSE, Feature-Liste, Skalerpfad.
- **Hilfsplots**: `training_progress.png` (Loss + Val-Metriken).

## Test Evidence (Version 1.4.1.2)

Diese Tests wurden bereits mit dem stateful Export durchgeführt, d.h. die Hidden States wurden über aufeinanderfolgende Zeitpunkte weitergereicht.

![SOC Prediction vs True — C07](archive/Tests/soc_test_1.4.1.2/MGFarm_18650_C07_first-1.png)

*Zustandsbehaftete Vorhersage für Zelle C07: Modell folgt Messverlauf eng.*

![SOC Pred vs True — Seq2Many](archive/Tests/soc_test_1.4.1.2/soct_test_1.4.1.2_first_50000_seq2many/soc_pred_vs_true_seq2many.png)

*Seq-to-many Streaming-Test über 50k Samples mit weitergereichtem LSTM-State.*

## Parameter-Anhang

- **Gesamtparameter**: 22,657.
- **Speicherabschätzung**: float32 ≈ 88.5 KB, float16 ≈ 44.3 KB, int8 ≈ 22.1 KB.
- **Top-Shapes**: (256,), (256, 6), (256, 64), (64, 64), (64,), (1, 64), (1,).

---

Erstellt aus `describe_checkpoint.py` + manueller Analyse (`train_soc.py`, `train_soc.yaml`).
