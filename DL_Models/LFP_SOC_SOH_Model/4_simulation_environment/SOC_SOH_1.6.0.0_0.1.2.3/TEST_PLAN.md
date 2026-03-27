# SOC+SOH Stress Test Plan (SOC_1.6.0.0 + SOH_0.1.2.3)

This folder contains the SOC+SOH runner used inside the broader robustness benchmark. The benchmark manipulates only measured signals and timing, not hidden model parameters.

## Scope (current)
- **Method under test:** SOC_1.6.0.0 + SOH_0.1.2.3 (hourly SOH updates, held constant between updates)
- **Primary metrics:** RMSE, MAE, bias, p95 error, max error, jump count, drift, recovery

## Robustness categories
**A. Input disturbances**
- Current offset / bias
- Voltage offset / bias
- Temperature offset / bias
- Current / voltage / temperature noise
- ADC quantization
- Sparse spikes / outliers

**B. Initialization errors**
- Initial SOC error for stateful estimators

**C. Signal integrity**
- Missing samples
- Irregular sampling
- Burst dropout / missing gap

## Scenarios available in the runner
- `baseline`
- `current_offset`
- `voltage_offset`
- `temp_offset`
- `current_noise`
- `voltage_noise`
- `temp_noise`
- `adc_quantization`
- `spikes`
- `initial_soc_error`
- `missing_samples`
- `irregular_sampling`
- `missing_gap`
- `temp_mask`
- `downsample`
- `missing_segments`

## Output artifacts per run
- `soc_pred_fullcell_<CELL>.csv`
- `soh_hourly_<CELL>.csv`
- `summary.json`
- `soc_pred_fullcell_<CELL>.png`
- `soh_hourly_<CELL>.png`

## Example runs
```bash
source /home/florianr/anaconda3/etc/profile.d/conda.sh
conda activate ml1
python run_soc_soh_scenario.py --cell MGFarm_18650_C07 --scenario current_offset --current_offset_a 0.05 --out_dir ./runs/current_offset/$(date +%F_%H%M)_current_offset_50mA
```

```bash
python run_soc_soh_scenario.py --cell MGFarm_18650_C07 --scenario adc_quantization --quantize_current_a 0.01 --quantize_voltage_v 0.005 --out_dir ./runs/adc_quantization/$(date +%F_%H%M)_adc_quant
```

```bash
python run_soc_soh_scenario.py --cell MGFarm_18650_C07 --scenario irregular_sampling --irregular_dt_jitter 0.2 --out_dir ./runs/irregular_sampling/$(date +%F_%H%M)_jitter200ms
```
