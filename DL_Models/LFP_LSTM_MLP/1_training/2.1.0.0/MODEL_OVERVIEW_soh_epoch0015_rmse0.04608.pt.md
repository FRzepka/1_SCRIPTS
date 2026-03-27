# SOH Model Overview — soh_epoch0015_rmse0.04608.pt

## Snapshot

| Item | Details |
| --- | --- |
| Checkpoint | `/home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/1_training/2.1.0.0/outputs/checkpoints/soh_epoch0015_rmse0.04608.pt` |
| File size | 1 047 313 bytes (≈1.00 MB) |
| Total parameters | 86 273 |
| Window length | 2048 Zeitschritte (seq-to-one) |
| Features | `Testtime[s]`, `Voltage[V]`, `Current[A]`, `Temperature[°C]`, `EFC`, `Q_c` |
| Target | SOH des letzten Samples je Fenster (reell, kein Sigmoid-Clip) |
| Best val. RMSE | 0.04608 (Epoche 15, MAE 0.03768, R² −0.0823) |

## Architektur

- **Encoder**: Einlagiges LSTM (`hidden_size=128`, batch-first, unidirektional).
- **Head**: MLP `128 → 128 → 1` mit ReLU, Dropout 0.05; lineare Ausgabe belässt SOH im physikalischen Wertebereich.
- **Initialisierung**: Xavier für alle Linear- und LSTM-Gewichte, Bias = 0.
- **Determinismus**: Seeds für Python/NumPy/PyTorch; `torch.backends.cudnn` deterministisch konfiguriert.

## Daten & Vorverarbeitung

1. **Quellen**: Parquet-Dateien unter `/0_Data/MGFarm_18650_FE`, Wahl über Zellen-IDs aus der Config.
2. **Bereinigung**: Entfernt NaNs/±Inf vor Sequenzbildung (`SeqDataset`).
3. **Skalierung**: `RobustScaler`, ausschließlich auf Trainingszellen gefittet und auf alle Fenster angewandt.
4. **Sequenzierung**: Sliding Window mit Stride 1 → jedes Sample erzeugt ein eigenständiges Fenster (`seq_chunk_size=2048`).

## Trainingsregime

- **Task**: seq-to-one Regression, komplett stateless je Batch.
- **Batching & Shuffle**: Trainings-`DataLoader` mischt alle Fenster pro Epoche (`shuffle=True`); 128 Fenster laufen parallel auf der GPU.
- **Optimierer**: AdamW (lr 2e-4, weight decay 1e-4), keine Grad-Akkumulation.
- **Stabilität**: Gradient Clipping (1.0), Dropout 0.05, optional Mixed Precision via `torch.amp`.
- **Scheduler**: Cosine Warm Restarts (`T₀=150`, `η_min = lr × 0.05`).
- **Monitoring**: Verlustfunktion MSE, obwohl Modellselektion über RMSE erfolgt; `training_progress.{json,png}` und CSV-Log protokollieren jede Epoche.
- **Early Stopping**: Konfiguriertes Limit 25 wird im Code auf ≥100 Validierungszyklen angehoben → praktisch deaktiviert für 150 geplante Epochen.

## Validierungsstrategie

- **Splits**: Train = {C01, C03, C05, C11, C17, C23}, Val = {C07, C19, C21}.
- **Modus**: Identisch zum Training → seq-to-one, non-stateful. Jeder Validierungsbatch startet ohne Hidden-State, `SeqDataset` liefert ausschließlich das Ziel des letzten Zeitschritts.
- **Kadenz & Metriken**: Evaluierung in Epoche 1 sowie alle 5 Epochen; berechnet MAE, RMSE, R². Primäre Auswahlmetrik ist RMSE.
- **Artefaktpflege**: Bester Checkpoint plus `scaler_robust.joblib` im Output-Run; CSV-Log aktualisiert `training_log.csv`.

## Exporte & Deployment

- **Standard-ONNX**: `soh_best_epochXXXX.onnx`, unterstützt dynamische Batch-Größen bei seq-to-one Inferenz.
- **Stateful ONNX**: Optionaler `StatefulWrapper` exportiert (`_stateful.onnx`) für Streaming mit weitergereichten Hidden-States (`x_step`, `h0`, `c0`).
- **Manifest**: `export_manifest.json` hält Best-Epoche, RMSE, Featureliste, Fensterlänge und Skalerpfad fest.
- **Visualisierung**: `training_progress.png` kombiniert Trainingsverlust, Val-MAE und Val-RMSE.

## Test Notes

- In diesem Repository-Stand liegen keine gesonderten SOH-Testplots. Für stateful Inferenz entspricht der Export dem im Training verwendeten LSTM; Hidden-States können optional weitergereicht werden.

## Parameter-Anhang

- **Gesamtparameter**: 86 273.
- **Speicherabschätzung**: float32 ≈ 337.00 KB (345 092 Bytes), float16 ≈ 168.50 KB (172 546 Bytes), int8 ≈ 84.25 KB (86 273 Bytes).
- **Top-Shapes**: (512, 6), (512, 128), (512,)×2, (128, 128), (128,), (1, 128), (1,).

---

Erstellt aus `train_soh.py`, `train_soh.yaml` und Trainingslogs (Stand Epoche 15).
