# CC_SOH_1.0.0

CC + SOH coupling model (online-style).

## What it does
- Uses SOH model **0.1.2.3** to predict SOH at **hourly** cadence.
- Holds that SOH constant within each hour (online constraint).
- Converts SOH to effective capacity: `C_eff = C_nom * SOH`.
- Runs Coulomb Counting (CC) with `C_eff` to estimate SOC each second.

## Inputs (per second)
- `Current[A]`
- `Voltage[V]`
- `Temperature[°C]`
- `EFC`
- `Q_c`
- `Testtime[s]`
- `SOC` (for evaluation)

## Outputs
- SOC estimate `soc_cc`
- SOH prediction per row `soh_pred`

## Dependencies
- SOH model files in `2_models/SOH_0.1.2.3`
- CC model in `2_models/CC_1.0.0`

## Usage
Use the scenario runner in `4_simulation_environment/CC_SOH_1.0.0`.
