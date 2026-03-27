Pruned STM32 Tests (AI_Project_LSTM_pruned)
=========================================

Scripts
- run_pruned_stream_and_plot.py
  - Streamt N Zeilen aus Parquet an den STM32, sammelt SOC, speichert Plots + metrics.
  - Robust gegen zusätzliche DBG/ERR‑Zeilen; liest bis SOC oder Timeout.
  - Optional: `--reset-state` sendet vor dem Streamen einen RESET an das Gerät.

Beispiele (vom Repo‑Root)

Windows (PowerShell, Einzeiler)
- Batch (5000 Samples)
  python "DL_Models\LFP_LSTM_MLP\6_test\STM32\pruned\run_pruned_stream_and_plot.py" --port COM7 --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" --yaml "DL_Models\LFP_LSTM_MLP\1_training\1.5.0.0\config\train_soc.yaml" --n 5000 --delay 0.02 --reset-state

Linux/macOS (bash/zsh, mit \ für Zeilenumbruch)
- Batch (5000 Samples)
  python "DL_Models\LFP_LSTM_MLP\6_test\STM32\pruned\run_pruned_stream_and_plot.py" \
    --port COM7 \
    --parquet "C:\\Users\\Florian\\SynologyDrive\\TUB\\3_Projekte\\MG_Farm\\5_Data\\01_LFP\\00_Data\\Versuch_18650_standart\\MGFarm_18650_FE\\df_FE_C07.parquet" \
    --yaml "DL_Models\\LFP_LSTM_MLP\\1_training\\1.5.0.0\\config\\train_soc.yaml" \
    --n 5000 --delay 0.02 --reset-state

Hinweise
- Projekt: `AI_Project_LSTM_pruned` (separat vom Base‑Projekt, Base bleibt unverändert).
- Das Gerät sendet Zeilen im Format "SOC: <float>"; der Parser liest die erste Zahl nach SOC.
- Wenn Spaltennamen im Parquet abweichen, `--cols` setzen.

