Quantized STM32 Tests (AI_Project_LSTM_quantized)
================================================

Scripts
- run_quantized_stream_and_plot.py
  - Streamt N Zeilen aus Parquet an den STM32, sammelt SOC, speichert Plots + metrics.
  - Robust gegen zusätzliche DBG/ERR‑Zeilen; liest bis SOC oder Timeout.

- run_quantized_live_plot.py
  - Live Rolling Plot (Default window=1000) für GT vs. Pred + Error.

Beispiele (vom Repo‑Root)

Windows (PowerShell, Einzeiler)
- Batch (5000 Samples)
  python "DL_Models\LFP_LSTM_MLP\6_test\STM32\quantized\run_quantized_stream_and_plot.py" --port COM7 --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" --yaml "DL_Models\LFP_LSTM_MLP\1_training\1.5.0.0\config\train_soc.yaml" --n 5000 --delay 0.02

- Live (Rolling 1000)
  python "DL_Models\LFP_LSTM_MLP\6_test\STM32\quantized\run_quantized_live_plot.py" --port COM7 --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" --yaml "DL_Models\LFP_LSTM_MLP\1_training\1.5.0.0\config\train_soc.yaml" --window 1000 --delay 0.02 --n 0

Linux/macOS (bash/zsh, mit \ für Zeilenumbruch)
- Batch (5000 Samples)
  python "DL_Models\LFP_LSTM_MLP\6_test\STM32\quantized\run_quantized_stream_and_plot.py" \
    --port COM7 \
    --parquet "C:\\Users\\Florian\\SynologyDrive\\TUB\\3_Projekte\\MG_Farm\\5_Data\\01_LFP\\00_Data\\Versuch_18650_standart\\MGFarm_18650_FE\\df_FE_C07.parquet" \
    --yaml "DL_Models\\LFP_LSTM_MLP\\1_training\\1.5.0.0\\config\\train_soc.yaml" \
    --n 5000 --delay 0.02

- Live (Rolling 1000)
  python "DL_Models\LFP_LSTM_MLP\6_test\STM32\quantized\run_quantized_live_plot.py" \
    --port COM7 \
    --parquet "C:\\Users\\Florian\\SynologyDrive\\TUB\\3_Projekte\\MG_Farm\\5_Data\\01_LFP\\00_Data\\Versuch_18650_standart\\MGFarm_18650_FE\\df_FE_C07.parquet" \
    --yaml "DL_Models\\LFP_LSTM_MLP\\1_training\\1.5.0.0\\config\\train_soc.yaml" \
    --window 1000 --delay 0.02 --n 0

Hinweise
- Das Projekt `AI_Project_LSTM_quantized` gibt Zeilen im Format "SOC: <float>" aus; die Parser lesen die erste Zahl nach SOC.
- Wenn Spaltennamen im Parquet abweichen, `--cols` setzen.

