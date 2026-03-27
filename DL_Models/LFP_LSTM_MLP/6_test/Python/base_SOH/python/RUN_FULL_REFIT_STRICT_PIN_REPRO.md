Full run (REFIT + strict filter + PIN) — Windows PowerShell, tqdm enabled

Working dir: `C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts`

Command (one line):

```
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\run_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --block-len 8192 ^
  --scaler-mode refit ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-start-one --calib-kind scale --calib-anchor by_pred --calib-mode pin ^
  --arrays none --plot-max 100000
```

Notes
- Prints tqdm progress during processing.
- Output folder is auto-named (e.g. `SCALER_REFIT_FULL_STRICT_PIN_YYYYMMDD_HHMMSS`) under
  `DL_Models/LFP_LSTM_MLP/6_test/Python/base_SOH/python` and contains `overlay_full.png`, `overlay_first2000.png`, `metrics.json`, `stdout.txt`, `stderr.txt`, and `scaler_robust_refit.joblib`.
- Ensure the parquet file path matches your dataset.

