Pruned FP32 → STM32 (Base Project)
==================================

Goal
- Start from the SOC FP32 checkpoint (1.5.0.0), prune LSTM hidden units (optionally MLP hidden),
  fine‑tune briefly, then export the pruned model weights to a C header compatible with the
  base STM32 project `AI_Project_LSTM`.

Steps
1) Prune + (optional) fine‑tune + export ONNX

   Example (unit‑wise pruning 30%, stateful ONNX optional):

   ```bash
   python DL_Models/LFP_LSTM_MLP/3_pruning/prune_lstm_soc.py \
     --ckpt DL_Models/LFP_LSTM_MLP/2_models/base/1.5.0.0_soc_epoch0001_rmse0.02897.pt \
     --yaml DL_Models/LFP_LSTM_MLP/1_training/1.5.0.0/config/train_soc.yaml \
     --prune-ratio 0.30 \
     --finetune-epochs 3 \
     --export-stateful \
     --out-dir DL_Models/LFP_LSTM_MLP/2_models/pruned/soc_1.5.0.0
   ```

   Output folder contains e.g. `soc_pruned_hidden45.pt`, `manifest.json`, and ONNX files.

2) Export pruned checkpoint to STM32 C header

   ```bash
   python DL_Models/LFP_LSTM_MLP/3_pruning/export_pruned_to_c_base.py \
     --ckpt DL_Models/LFP_LSTM_MLP/2_models/pruned/soc_1.5.0.0/prune_30pct_*/soc_pruned_hidden45.pt \
     --out_dir DL_Models/LFP_LSTM_MLP/2_models/pruned/soc_1.5.0.0/prune_30pct_*/c_implementation
   ```

   This creates `model_weights.h` with macros INPUT_SIZE/HIDDEN_SIZE/MLP_HIDDEN matching the
   pruned architecture.

3) Update STM32 base project header and build

   ```powershell
   # PowerShell
   STM32\workspace_1.17.0\AI_Project_LSTM\copy_pruned_headers.ps1 `
     -HeaderPath "C:\\...\\2_models\\pruned\\soc_1.5.0.0\\prune_30pct_*\\c_implementation\\model_weights.h"
   ```

   Open `AI_Project_LSTM` in CubeIDE, build and flash. The base code uses the macros from the
   header so no source changes are required.

Notes
- The pruned sizes propagate to MLP_FC1 automatically (input = pruned hidden size). Output layer
  (1 neuron) remains unchanged.
- If you also prune MLP hidden neurons (via `prune_lstm_soc_advanced.py --mlp-prune-ratio`), the
  exported header reflects the new MLP_HIDDEN accordingly.

