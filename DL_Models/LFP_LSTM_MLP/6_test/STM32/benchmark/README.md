STM32 SOH Benchmark (Base vs Quantized vs Pruned)

This folder provides small, non-invasive tools to:
- Orchestrate a single streaming run per model and time it.
- Aggregate prior runs (metrics.json) into a single CSV.
- Generate quick comparison plots (Flash/RAM optional, if map paths provided).

We intentionally do not alter existing scripts; we call them as-is and attach timing/metadata.

Quick start (examples)

1) Run a timed stream for one model
   - Base (FP32):
     python DL_Models/LFP_LSTM_MLP/6_test/STM32/benchmark/run_bench_stream_soh.py \
       --model base --port COM7 \
       --parquet "C:\\...\\MGFarm_18650_FE\\df_FE_C07.parquet" \
       --yaml DL_Models/LFP_LSTM_MLP/1_training/2.1.0.0/config/train_soh.yaml \
       --ckpt DL_Models/LFP_LSTM_MLP/2_models/base/2.1.0.0_soh_epoch0120_rmse0.03359.pt \
       --n 10000 --prime 2047 --delay 0.01 --timeout 2.5 --strict

   - Quantized (INT8): same but --model quantized

   - Pruned: same as base (firmware is base with pruned weights); use --model pruned to label the run.

2) Aggregate past runs into a CSV (optionally parse .map files)
   python DL_Models/LFP_LSTM_MLP/6_test/STM32/benchmark/aggregate_benchmarks.py \
     --base_map "STM32/workspace_1.17.0/AI_Project_LSTM_SOH_base/Debug/AI_Project_LSTM.map" \
     --quant_map "STM32/workspace_1.17.0/AI_Project_LSTM_SOH_quantized/Debug/AI_Project_LSTM.map"

   Output: summary.csv in this folder.

3) Plot a compact report from summary.csv
   python DL_Models/LFP_LSTM_MLP/6_test/STM32/benchmark/plot_benchmark_report.py

Notes
- Throughput is end-to-end (PC↔UART↔STM32) from wall clock; good for relative comparison.
- If map files are provided, Flash/RAM (KB) are parsed; otherwise those fields stay blank.
- Keep firmware unchanged; use existing streaming scripts and STM32 projects.

