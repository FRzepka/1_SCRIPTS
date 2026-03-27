@echo off
REM Quick runner for streaming Parquet features to STM32
REM Usage: rt.bat COM9 [additional streamer args]

set "PARQUET=C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C01.parquet"
set "YAML=C:\Users\Florian\STM32CubeIDE\workspace_1.17.0\ML\LFP_LSTM_MLP\1_training\1.5.0.0\config\train_soc.yaml"
set "SCRIPT=%~dp0stream_parquet_features.py"

if "%~1"=="" (
  echo Usage: rt.bat COM_PORT [--step] [--n 100] [--delay 0.05] [--start 0]
  echo Example: rt.bat COM9 --step --n 20
  exit /b 1
)

set "PORT=%~1"
shift

python "%SCRIPT%" "%PARQUET%" --port %PORT% --yaml "%YAML%" %*
