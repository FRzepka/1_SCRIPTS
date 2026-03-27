# ECM_Model Python Runner

This folder contains the Python entrypoint used to call the new native ECM EKF in:
- `../battery_ekf.c`
- `../interpolation.c`
- `../ECM_parameter.c`

Files:
- `run_ecm_test.py`: FE dataset runner (same interface as previous runner).
- `ecm_native.py`: ctypes wrapper for native EKF library.
- `run_ecm_simple.py`: minimal FE runner (few parameters, direct call path).

Default native library lookup:
1. `../libecm_ekf_new.so`
2. `../libecm_ekf.so`

Example:
```bash
python DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/ECM_Model/python_model/run_ecm_test.py \
  --native \
  --cell MGFarm_18650_C07 \
  --scenario baseline \
  --dt_mode fixed --dt_s 60 \
  --max_rows 1193040 --downsample 60 \
  --out_dir /tmp/ecm_model_test
```

Minimal example:
```bash
python DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/ECM_Model/python_model/run_ecm_simple.py \
  --cell MGFarm_18650_C07 \
  --out_dir /tmp/ecm_model_simple
```

Build native library from `ECM_Model`:
```bash
bash DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/ECM_Model/build_native.sh
```
