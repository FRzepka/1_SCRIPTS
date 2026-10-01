# Stateful Base Model Comparison

This benchmark evaluates the current CNN, GRU, LSTM, and TCN base checkpoints
on the test cells C11, C23, and C29. Recurrent state and causal convolution
context are retained for the complete trajectory.

```bash
pip install -r requirements.txt
python run_stateful_benchmark.py --data-root /path/to/MGFarm_18650_FE
```

The data directory can alternatively be set through `MGFARM_FE_DATA_ROOT`.
The command writes cell-level metrics, aggregate metrics, a compact MAE table,
the C11 SOH trajectories, and model hashes to `results`.
