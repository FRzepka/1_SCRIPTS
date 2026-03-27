# Run command

```
python C:/Users/Florian/SynologyDrive/TUB/1_Dissertation/1_Scripts/DL_Models/LFP_LSTM_MLP/6_test/Python/quantized_SOH/compare_c_vs_quant_step_direct.py --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt --parquet C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet --num-samples 2000000 --prime --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 --calib-start-one --calib-kind scale --calib-anchor by_pred --calib-mode pin --calib-apply before_filter --plot-max 0 --quant-scale p99_9 --quant-hh-precision int8 --export-quantized-to DL_Models\LFP_LSTM_MLP\2_models\quantized\manual_int8_lstm_soh_export_p99_9 --out-dir DL_Models\LFP_LSTM_MLP\6_test\Python\quantized_SOH\COMPARE_Q_P99_9_I8_PIN_BEFORE_FILTER_2M
```
