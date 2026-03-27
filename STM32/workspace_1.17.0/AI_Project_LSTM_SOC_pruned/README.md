# AI_Project_LSTM_pruned

This is a copy of the base STM32 project configured to use a pruned FP32 LSTM+MLP model.

- Header: `Core/Inc/model_weights.h` (already set to the pruned export)
- Scaler: `Core/Inc/scaler_params.h` (same as base)
- UART protocol: identical to base (`RESET` supported; output lines with `SOC:`)

Rebuild/Flash in CubeIDE, then run the Python stream harness:

python "DL_Models\LFP_LSTM_MLP\6_test\STM32\pruned\run_pruned_stream_and_plot.py" \ 
  --port COM7 \ 
  --parquet "C:\\Users\\Florian\\SynologyDrive\\TUB\\3_Projekte\\MG_Farm\\5_Data\\01_LFP\\00_Data\\Versuch_18650_standart\\MGFarm_18650_FE\\df_FE_C07.parquet" \ 
  --yaml "DL_Models\LFP_LSTM_MLP\1_training\1.5.0.0\config\train_soc.yaml" \ 
  --n 5000 --delay 0.01 --reset-state

