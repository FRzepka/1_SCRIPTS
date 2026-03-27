# Full SOH seq2many – PIN calibration (REFIT scaler) – command reference

This file captures the exact commands for reproducing the “good” PIN runs and a few close variants. All runs plot overlay_full.png and overlay_first2000.png and write metrics.json in their respective output folders.

Common paths
- Checkpoint: `DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt`
- Parquet: `C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet`

Filter/Calib settings (identisch zu den „guten“ Läufen)
- `--strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5`
- `--calib-start-one --calib-kind scale --calib-anchor by_pred --calib-mode pin`
- `--arrays none --plot-max 100000` (Plot gedrosselt, Inferenz full)

## A) REFIT + strict + PIN (Hauptlauf)

```
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\run_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --block-len 8192 ^
  --scaler-mode refit ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-start-one --calib-kind scale --calib-anchor by_pred --calib-mode pin ^
  --arrays none --plot-max 100000 ^
  --out-dir DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\FULL_REFIT_STRICT_PIN_REPEAT
```

## B) HYBRID_CENTER + strict + PIN (Variante)

```
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\run_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --block-len 8192 ^
  --scaler-mode hybrid_center ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-start-one --calib-kind scale --calib-anchor by_pred --calib-mode pin ^
  --arrays none --plot-max 100000 ^
  --out-dir DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\FULL_HYBRIDCENTER_STRICT_PIN
```

## C) REFIT + strict + PIN (kleineres Fenster – Diagnose)

```
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\run_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --block-len 8192 ^
  --scaler-mode refit ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-start-one --calib-kind scale --calib-anchor by_pred --calib-mode pin ^
  --arrays none --plot-max 100000 ^
  --limit 50000 ^
  --out-dir DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\REFIT_STRICT_PIN_first50k
```

