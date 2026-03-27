# Selected Models (Final)

Diese Auswahl ist aktiv in `2_models`:

- `LSTM` (gross):
  - `Base/0.1.2.4`
  - `Pruned/0.1.2.4_s20_struct`
  - `PrunedFT/0.1.2.4_s20_struct_ft`
  - `Quantized/0.1.2.4_s20_struct_ft_int8`
- `GRU` (gross):
  - `Base/0.3.1.2`
  - `Pruned/0.3.1.2_s20_struct`
  - `PrunedFT/0.3.1.2_s20_struct_ft`
  - `Quantized/0.3.1.2_s20_struct_ft_int8`
- `TCN` (gross):
  - `Base/0.2.2.2`
  - `Pruned/0.2.2.2_s20_struct`
  - `PrunedFT/0.2.2.2_s20_struct_ft`
  - `Quantized/0.2.2.2_s20_struct_ft_int8`
- `CNN` (klein):
  - `Base/0.4.2.1_hp`
  - `Pruned/0.4.2.1_hp_s20_struct`
  - `PrunedFT/0.4.2.1_hp_s20_struct_ft`
  - `Quantized/0.4.2.1_hp_s20_struct_ft_int8`

Verschoben ins Archiv:

- `archive/20260216_154618_cleanup_keep_smallCNN_bigLSTM_GRU_TCN`

Jeder behaltene Modellordner enthält die nötigen Nutzdateien:

- `config/train_soh.yaml`
- `scripts/train_soh.py`
- `scaler_robust.joblib`
- Checkpoint/Export-Datei:
  - Base: `checkpoints/best_model.pt`
  - Pruned: `checkpoints/best_model_pruned.pt`
  - PrunedFT: `checkpoints/best_model_finetuned.pt`
  - Quantized: `quantized_state_dict.pt` + `quantize_meta.json`
