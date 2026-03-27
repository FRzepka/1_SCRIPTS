/*
 * RobustScaler parameters for SOH preprocessing
 * Auto-generated from scaler_robust.joblib
 */

#ifndef SCALER_PARAMS_H
#define SCALER_PARAMS_H

#define SCALER_NUM_FEATURES 6

const float SCALER_SOH_CENTER[SCALER_NUM_FEATURES] = {
    6894909.0000000000f, 3.3605999947f, 0.6542999744f, 27.3999996185f, 1225.1582031250f, -0.5109897852f
};

const float SCALER_SOH_SCALE[SCALER_NUM_FEATURES] = {
    7007862.0625000000f, 0.2009000778f, 2.6982000470f, 1.1000003815f, 1226.3476715088f, 0.5354322791f
};

static inline void scaler_transform(const float in[SCALER_NUM_FEATURES], float out[SCALER_NUM_FEATURES]){
    for (int i=0;i<SCALER_NUM_FEATURES;i++){ out[i] = (in[i] - SCALER_SOH_CENTER[i]) / SCALER_SOH_SCALE[i]; }
}

#endif /* SCALER_PARAMS_H */
