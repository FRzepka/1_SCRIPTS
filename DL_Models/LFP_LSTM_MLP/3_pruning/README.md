# Pruning Toolkit (SOC LSTM+MLP)

This folder contains scripts to prune the SOC LSTM+MLP model and export ONNX artifacts.

## Scripts

- prune_lstm_soc.py
  - One-shot, magnitude-based unit pruning for the LSTM hidden units (structured).
  - Copies the selected units into a smaller LSTM; adapts the first MLP Linear input.
  - Optional: `--export-stateful` to also export a stateful streaming ONNX (x_step, h0, c0).

- prune_lstm_soc_advanced.py (new)
  - All features of `prune_lstm_soc.py`, plus:
    - `--saliency {unit,gate-aware}`: choose unit-wise vs gate-aware saliency.
    - `--combine {sum,mean,max,min}`: reduce gate scores (only for gate-aware).
    - `--mlp-prune-ratio FLOAT`: additionally prune the MLP hidden layer neurons structurally.
    - `--export-stateful`: also export stateful ONNX (recommended for streaming eval).

## What you currently used

- Your existing pruned model corresponds to unit-wise LSTM pruning (structured), magnitude-based, one-shot, with short fine-tuning. The MLP head was not pruned (only input sliced to match reduced hidden size).

## Recommended usage

Create a pruned model with stateful ONNX export (unit-wise saliency):

```bash
python /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/3_pruning/prune_lstm_soc.py \
  --ckpt /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/1_training/1.5.0.0/outputs/checkpoints/soc_epoch0005_rmse0.02568.pt \
  --yaml /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/1_training/1.5.0.0/config/train_soc.yaml \
  --prune-ratio 0.3 \
  --finetune-epochs 5 \
  --export-stateful \
  --out-dir /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/2_models/pruned/soc_1.5.0.0
```

Use advanced options (gate-aware + MLP-head pruning):

```bash
python /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/3_pruning/prune_lstm_soc_advanced.py \
  --ckpt /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/1_training/1.5.0.0/outputs/checkpoints/soc_epoch0005_rmse0.02568.pt \
  --yaml /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/1_training/1.5.0.0/config/train_soc.yaml \
  --prune-ratio 0.3 \
  --saliency gate-aware \
  --combine sum \
  --mlp-prune-ratio 0.25 \
  --finetune-epochs 5 \
  --export-stateful \
  --out-dir /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/2_models/pruned/soc_1.5.0.0
```

## Notes

- Stateful ONNX (`*_stateful.onnx`) enables true streaming (step-by-step) with hidden state forwarding in benchmarks.
- If you only export the dense ONNX (fixed time length), the benchmark will automatically use window mode for that variant.
- For stability you may want to ensure forget-gate bias stays positive during fine-tuning (handled by the pretrained weights; adjust if you re-initialize).