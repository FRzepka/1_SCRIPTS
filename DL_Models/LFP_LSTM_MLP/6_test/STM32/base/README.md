Base STM32 Tests (AI_Project_LSTM)
==================================

Scripts
- run_base_stream_and_plot.py
  - Streamt N Zeilen aus Parquet an den STM32, sammelt SOC, speichert Plots + metrics.
  - Optional: `--reset-state` sendet vor dem Streamen einen RESET an das Gerät.

- run_base_live_plot.py
  - Live Rolling Plot (Default window=1000) für GT vs. Pred + Error.

Beispiele (vom Repo‑Root)

Windows (PowerShell, Einzeiler)
- Batch (5000 Samples)
  python "DL_Models\LFP_LSTM_MLP\6_test\STM32\base\run_base_stream_and_plot.py" --port COM7 --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" --yaml "DL_Models\LFP_LSTM_MLP\1_training\1.5.0.0\config\train_soc.yaml" --n 5000 --delay 0.01 --reset-state

- Live (Rolling Window 1000)
  python "DL_Models\LFP_LSTM_MLP\6_test\STM32\base\run_base_live_plot.py" --port COM7 --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" --yaml "DL_Models\LFP_LSTM_MLP\1_training\1.5.0.0\config\train_soc.yaml" --window 1000 --delay 0.01 --n 0

Linux/macOS (bash/zsh, mit \ für Zeilenumbruch)
- Batch (5000 Samples)
  python "DL_Models\LFP_LSTM_MLP\6_test\STM32\base\run_base_stream_and_plot.py" \
    --port COM7 \
    --parquet "C:\\Users\\Florian\\SynologyDrive\\TUB\\3_Projekte\\MG_Farm\\5_Data\\01_LFP\\00_Data\\Versuch_18650_standart\\MGFarm_18650_FE\\df_FE_C07.parquet" \
    --yaml "DL_Models\\LFP_LSTM_MLP\\1_training\\1.5.0.0\\config\\train_soc.yaml" \
    --n 5000 --delay 0.01 --reset-state

- Live (Rolling Window 1000)
  python "DL_Models\LFP_LSTM_MLP\6_test\STM32\base\run_base_live_plot.py" \
    --port COM7 \
    --parquet "C:\\Users\\Florian\\SynologyDrive\\TUB\\3_Projekte\\MG_Farm\\5_Data\\01_LFP\\00_Data\\Versuch_18650_standart\\MGFarm_18650_FE\\df_FE_C07.parquet" \
    --yaml "DL_Models\\LFP_LSTM_MLP\\1_training\\1.5.0.0\\config\\train_soc.yaml" \
    --window 1000 --delay 0.01 --n 0

Hinweise
- Das Board sendet Zeilen im Format "SOC: <float>"; der Parser liest die erste Zahl nach SOC.
- Bei Bedarf `--cols` setzen, um die 6 Feature‑Spalten explizit zu wählen.
- `--reset-state` setzt den LSTM‑Zustand vor dem Lauf zurück (nur stream_and_plot).

