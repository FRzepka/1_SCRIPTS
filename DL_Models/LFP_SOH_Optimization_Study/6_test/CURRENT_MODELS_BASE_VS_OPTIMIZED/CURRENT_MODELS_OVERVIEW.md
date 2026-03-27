# Current Models Overview (Base vs Optimized)

Optimized = **structured pruning + finetune + int8 quantization**.

Plots: `6_test/CURRENT_MODELS_BASE_VS_OPTIMIZED/base_vs_optimized`

| Model | Version | Base MAE | Opt MAE | MAE Delta | Base RMSE | Opt RMSE | Base kB | Pruned kB | Optimized kB (quant) | Pruning Reduktion | Quantization Reduktion | Gesamt Reduktion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4.2.1_hp | 0.014011 | 0.014672 | +0.000661 (+4.7%) | 0.017885 | 0.019303 | 1966.0 | 1305.0 | 363.7 | 33.6% | 72.1% | 81.5% |
| GRU | 0.3.1.2 | 0.012377 | 0.012427 | +0.000051 (+0.4%) | 0.015248 | 0.015303 | 3298.1 | 2454.1 | 658.2 | 25.6% | 73.2% | 80.0% |
| LSTM | 0.1.2.4 | 0.014226 | 0.013438 | -0.000788 (-5.5%) | 0.017129 | 0.016632 | 3499.8 | 2540.5 | 673.7 | 27.4% | 73.5% | 80.7% |
| TCN | 0.2.2.2 | 0.012102 | 0.011306 | -0.000796 (-6.6%) | 0.015626 | 0.014405 | 1718.1 | 1206.9 | 1216.8 | 29.8% | -0.8% | 29.2% |
