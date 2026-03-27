# Run: Full C step-by-step vs Python seq2many (EXACT, REFIT + strict + PIN)

This documents the exact commands we use to reproduce the good runs and to compare C step-by-step vs Python seq2many with identical post-processing (strict filter + pin calibration) and the same refit scaler.

## 1) Full Python (REFIT + strict filter + PIN)

PowerShell:

```
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\run_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --block-len 8192 ^
  --scaler-mode refit ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-start-one --calib-kind scale --calib-anchor by_pred ^
  --arrays none --plot-max 100000 ^
  --out-dir DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\FULL_REFIT_STRICT_PIN_REPEAT
```

Notes:
- Uses REFIT scaler from the selected parquet (same behavior as the original good run).
- Progress bars (tqdm) are enabled by default.

## 2) Full C (step) vs Python (seq2many) — EXACT pipeline match

PowerShell:

```
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\c_code\compare_c_vs_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --num-samples 50000000 --prime ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-mode pin --calib-kind scale --calib-anchor by_pred ^
  --scaler DL_Models\LFP_LSTM_MLP\1_training\2.1.0.0\outputs\scaler_robust_refit_soh.joblib ^
  --out-dir DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\c_code\C_VS_SEQ2MANY_EXACT_FULL
```

Notes:
- If you want to force the exact refit scaler from a specific Python run folder instead, use:

```
  --scaler DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\FULL_REFIT_STRICT_PIN_20251112_151345\scaler_robust_refit.joblib
```

- The script will attempt to compile a native C test binary; if that fails, it falls back to a Python C-sim path (still step-by-step). Add `--no-compile` to force the C-sim path.

Outputs:
- Python run folder contains overlay_full.png, overlay_first2000.png, metrics.json, arrays.npz (optional).
- C vs Python folder contains overlay_full.png, overlay_firstN.png, diff_hist.png, metrics.json (MAE/RMSE/Max diff).

