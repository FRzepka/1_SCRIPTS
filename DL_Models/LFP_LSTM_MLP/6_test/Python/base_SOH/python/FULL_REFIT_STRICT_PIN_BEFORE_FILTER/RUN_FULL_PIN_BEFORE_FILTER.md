Full run — REFIT + strict filter + PIN applied BEFORE filter (Windows PowerShell, tqdm)

Working dir: `C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts`

Command (exact):

```
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\run_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --block-len 8192 ^
  --scaler-mode refit ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-start-one --calib-kind scale --calib-anchor by_pred --calib-mode pin --calib-apply before_filter ^
  --arrays none --plot-max 100000 ^
  --out-dir DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\FULL_REFIT_STRICT_PIN_BEFORE_FILTER
```

Notes
- PIN is applied before EMA/clamp, so the first value is exactly 1 and then the series naturally “drops” due to the filter.
- Keep plot normalization disabled (`--norm-start-one off`, default) to avoid visual confusion.
- Outputs land in this folder (overlay_full.png, overlay_first2000.png, metrics.json, scaler_robust_refit.joblib, arrays.npz if enabled).

