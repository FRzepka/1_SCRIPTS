# ECM new test notes

Source model:
- `DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/ECM_Model`

Native library built from new C files:
- `DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/ECM_Model/libecm_ekf_new.so`

Build command:
```bash
gcc -O2 -fPIC -shared -include math.h \
  -I DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/python_model/native/include \
  -I DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/ECM_Model \
  DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/ECM_Model/battery_ekf.c \
  DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/ECM_Model/interpolation.c \
  DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/ECM_Model/ECM_parameter.c \
  -lm \
  -o DL_Models/LFP_SOC_SOH_Model/2_models/ECM_qinnan/ECM_Model/libecm_ekf_new.so
```

Important runtime note:
- `battery_ekf.c` uses fixed `deltaT = 60s`.
- FE data must be sampled accordingly (`--downsample 60`) for realistic behavior.

Runs created:
- `2026-02-17_1024_smoke_C07`
  - quick smoke, raw step rate (large error expected due 1s data vs 60s EKF model)
- `2026-02-17_1024_smoke_C07_ds60`
  - smoke with `--downsample 60`
- `2026-02-17_1025_baseline_C07_ds60_full`
  - full C07 baseline with `--downsample 60`

Baseline full metrics (`2026-02-17_1025_baseline_C07_ds60_full/summary.json`):
- `mae = 0.03752354013188058`
- `rmse = 0.04882996654885249`
- `mae_after_warmup = 0.03748370928943795`
- `rmse_after_warmup = 0.04840537781556555`
