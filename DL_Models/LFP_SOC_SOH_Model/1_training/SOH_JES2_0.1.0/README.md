# SOH JES2 0.1.0

This run retrains the existing deployed SOH LSTM architecture with the same
cell exposure as the frozen SOC 1.7.x models.

- Train: C01, C03, C05, C11, C17, C23
- Validation: C07, C19, C21
- Final hold-out: C09, C13, C15, C25, C27, C29
- Architecture: LSTM hybrid seq2seq, embed 128, hidden 112, two layers
- Scaler: fitted only on the six training cells

Training entrypoint:

```bash
/home/florianr/anaconda3/envs/ml1/bin/python \
  DL_Models/LFP_SOH_Optimization_Study/1_training/0.1.2.5/scripts/train_soh.py \
  --config DL_Models/LFP_SOC_SOH_Model/1_training/SOH_JES2_0.1.0/config/train_soh.yaml \
  --run-id <run-id>
```

The final hold-out cells must not be used for checkpoint selection, scaler
fitting, pruning, quantization, or benchmark tuning.
