C vs Python (full, exact REFIT scaler + strict + PIN) — Windows PowerShell

Working dir: `C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts`

Option A — Use the REFIT scaler from the known “good” run

```
$SCALER = "DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\FULL_REFIT_STRICT_PIN_20251112_151345\scaler_robust_refit.joblib"
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\c_code\compare_c_vs_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --scaler "$SCALER" ^
  --num-samples 500000 ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-kind scale --calib-anchor by_pred --calib-mode pin --calib-apply before_filter
```

Option B — Use the REFIT scaler saved by your fresh full Python run

After running the Python full run above, set the scaler to that output folder’s `scaler_robust_refit.joblib` and re-run the command:

```
$SCALER = "DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\SCALER_REFIT_FULL_STRICT_PIN_YYYYMMDD_HHMMSS\scaler_robust_refit.joblib"
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\c_code\compare_c_vs_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --scaler "$SCALER" ^
  --num-samples 500000 ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-kind scale --calib-anchor by_pred --calib-mode pin --calib-apply before_filter
```

Notes
- Shows tqdm progress; outputs `overlay_full.png`, `overlay_firstN.png`, `metrics.json` under an auto-named folder in `.../c_code/`.
- Using the same REFIT scaler is critical for exact overlays.
