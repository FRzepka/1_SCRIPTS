# CC Model (Coulomb Counting) – Online SOC Block

## Purpose
This is a lightweight **online** Coulomb‑Counting SOC block. It processes **one sample at a time** and returns SOC. The algorithm uses **only current, voltage, temperature** (temperature is optional and currently not used in the formula). It does **not** use Q_c/Q_m from the dataset.

## Inputs (per sample)
- `Current[A]`
- `Voltage[V]`
- `Temperature[°C]` (optional)
- `dt_s` (seconds) or `timestamp_s` (seconds)
- Optional: `capacity_ah` per sample (e.g., from an SOH estimator)

## Output
- `SOC` in 

## Algorithm
1) **Integrate current** to get `Q_m_new` (Ah):
   - `Q_m_new += current * dt_s / 3600`
2) **Detect CV phase**:
   - If `Voltage[V] >= (V_max - V_tol)` **continuously** for `cv_seconds`, then
     - reset `Q_m_new = 0` (thus SOC = 1)
3) **SOC**:
   - `SOC = 1 + Q_m_new / capacity_ah`
4) **Clip** SOC to [0, 1]

## Default Parameters
- `capacity_ah = 1.8`
- `soc_init = 1.0`
- `current_sign = 1.0`
- `v_max = 3.65`
- `v_tol = 0.02`
- `cv_seconds = 300` (5 minutes)

## Usage Example
```python
from cc_model import CCModel, CCModelConfig

cfg = CCModelConfig(capacity_ah=1.8, v_max=3.65, v_tol=0.02, cv_seconds=300)
model = CCModel(cfg)

soc = model.step(current_a=1.2, voltage_v=3.55, temperature_c=25.0, dt_s=1.0)
```

## Notes
- This is designed for **online** execution in the simulation environment.
- The model is intentionally simple and hardware‑friendly.
- Capacity can be **constant** (parameter) or **per‑sample** (e.g., fed by an SOH model).
