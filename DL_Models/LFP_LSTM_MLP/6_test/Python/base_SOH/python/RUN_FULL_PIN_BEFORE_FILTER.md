Goal: Start exactly at 1, then drop smoothly to the model’s natural trajectory (PIN applied before strict filter so EMA/clamp bleed the first sample).

Working dir: `C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts`

Command:

```
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\run_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --block-len 8192 ^
  --scaler-mode refit ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-start-one --calib-kind scale --calib-anchor by_pred --calib-mode pin ^
  --calib-apply before_filter ^
  --calib-apply before_filter ^
  --arrays none --plot-max 100000 ^
  --out-dir DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\FULL_REFIT_STRICT_PIN_BEFORE_FILTER
```

Notes
- This matches the visual behavior “start at 1 then drop”, because PIN is applied before the EMA/clamp filter.
- Do NOT enable `--norm-start-one`; that is plot-only and will confuse comparisons.
