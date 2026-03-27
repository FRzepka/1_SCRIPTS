# SOC PC Benchmark

Vergleicht SOC-Modelle (Base, Pruned, Quant) in Python (FP32) und C (FP32/INT8) auf den ersten N Schritten eines Parquet-Datensatzes.

## Modelle & Daten
- Base: `2_models/base/soc_1.5.0.0_base/1.5.0.0_soc_epoch0001_rmse0.02897.pt`
- Pruned: `2_models/pruned/soc_1.5.0.0_pruned/prune_30pct_20250916_140404/soc_pruned_hidden45.pt`
- Quant (INT8 LSTM + FP32 MLP): `2_models/quantized/soc_1.5.0.0_quantized`
- C-Libs (werden von `run_benchmark_soc.py` genutzt):
  - Base: `.../base/soc_1.5.0.0_base/c_implementation/liblstm_soc_base.so`
  - Pruned: `.../pruned/soc_1.5.0.0_pruned/prune_30pct_20250916_140404/c_implementation/liblstm_soc_pruned.so`
  - Quant: `.../quantized/soc_1.5.0.0_quantized/liblstm_soc_quant.so`
- Features: `["Voltage[V]", "Current[A]", "Temperature[°C]", "Q_c", "dU_dt[V/s]", "dI_dt[A/s]"]`
- Scaler: `1_training/1.5.0.0/outputs/scaler_robust.joblib`
- Default Cell: `MGFarm_18650_C07`
- Data root: `/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE`

## Hauptskript
`run_benchmark_soc.py` – führt step-by-step Inferenz durch:
- Python FP32: Base, Pruned
- C FP32: Base, Pruned
- C Quant (INT8 LSTM, FP32 MLP): Quant
- Speichert: `BENCH_SOC_<timestamp>/arrays.npz`, `metrics.json`, `overlay_firstN.png`, `errors_firstN.png`

### Beispiel (20k Schritte, mit Plots)
```bash
cd /home/florianr/MG_Farm/1_Scripts
python -u DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOC/run_benchmark_soc.py \
  --limit 20000 \
  --cell MGFarm_18650_C07
```
Optional: `--device cpu` erzwingt CPU (bei GPU-Knappheit), `--no-plots` spart Plotzeit.

### Ausgabe
Im neu angelegten `BENCH_SOC_<timestamp>/`:
- `metrics.json`: MAE/RMSE pro Modell + C-vs-Python-Deltas
- `arrays.npz`: `y`, `base_py`, `base_c`, `pruned_py`, `pruned_c`, `quant_c`
- Plots: Overlay und Error (falls nicht mit `--no-plots` deaktiviert)

## Kurzresultate (letzter Lauf 20251124_132114, limit 20k)
```
Model           MAE       RMSE
Base Py         0.05859   0.21161
Base C          0.05859   0.21161
Pruned Py       0.06053   0.21182
Pruned C        0.06053   0.21182
Quant C         0.05910   0.21157
Base C vs Py    ~0        ~0
Pruned C vs Py  ~0        ~0
```
