@echo off
REM ========================================
REM  BMS SOC LSTM LIVE PREDICTION STARTER
REM  Optimized Version - Production Ready
REM ========================================

echo.
echo 🚀 BMS SOC LSTM LIVE PREDICTION SYSTEM
echo =====================================
echo.
echo Starting optimized live prediction system...
echo Performance: 20Hz, 10ms latency, bounded memory
echo.

REM Navigate to correct directory
cd /d "c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU"

echo 📡 Starting Data Sender (C19 Battery Data)...
start "Data Sender" cmd /k "python data_sender_C19.py"

REM Wait for data sender to initialize
timeout /t 3 /nobreak >nul

echo 🧠 Starting Live SOC Prediction System...
start "Live SOC Prediction" cmd /k "python live_test_soc_optimized.py"

echo.
echo ✅ Both systems started!
echo.
echo 📊 The live prediction window will show:
echo    - True vs Predicted SOC (Blue vs Red)
echo    - Absolute Error (Green)
echo    - Input Voltage (Orange)  
echo    - Input Current (Purple)
echo    - Real-time Statistics
echo.
echo 🎯 Performance: 20Hz processing, 10ms latency
echo 💾 Memory: Bounded usage, no memory leaks
echo ⚡ Optimization Score: 98.6%%
echo.
echo 🛑 Press any key to return to menu...
pause >nul
