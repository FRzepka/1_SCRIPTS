Goal: Start at 1, blend correction out over tau steps (decay), keeping filter first.

Working dir: `C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts`

Command:

```
python DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\run_seq2many.py ^
  --ckpt DL_Models\LFP_LSTM_MLP\2_models\base\2.1.0.0_soh_epoch0120_rmse0.03359.pt ^
  --parquet "C:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_FE\df_FE_C07.parquet" ^
  --block-len 8192 ^
  --scaler-mode refit ^
  --strict-filter --post-max-rel 1e-4 --post-max-abs 2e-6 --post-ema-alpha 2e-4 --filter-passes 5 ^
  --calib-start-one --calib-kind scale --calib-anchor by_pred --calib-mode decay --calib-decay-tau 5000 ^
  --calib-apply after_filter ^
  --calib-apply after_filter ^
  --arrays none --plot-max 100000 ^
  --out-dir DL_Models\LFP_LSTM_MLP\6_test\Python\base_SOH\python\FULL_REFIT_STRICT_DECAY_AFTER_FILTER
```

Notes
- Decay gives a controlled “drop” profile via an exponential fade (tau=5000 by default). Try 2000/10000 for faster/slower blending.
- Keep plot normalization off.
