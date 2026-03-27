/*
 * RobustScaler parameters for SOC model preprocessing
 * Features: Voltage[V], Current[A], Temperature[°C], Q_c, dU_dt[V/s], dI_dt[A/s]
 * DO NOT EDIT MANUALLY
 */

#ifndef SCALER_PARAMS_H
#define SCALER_PARAMS_H

#define SCALER_NUM_FEATURES 6

/* RobustScaler center values (subtract from raw input) */
const float SCALER_CENTER[SCALER_NUM_FEATURES] = {
    3.3605999947f,
    0.6542999744f,
    27.3999996185f,
    -0.5109897852f,
    0.0000000000f,
    0.0000000000f
};

/* RobustScaler scale values (divide by these after centering) */
const float SCALER_SCALE[SCALER_NUM_FEATURES] = {
    0.2009000778f,
    2.6982000470f,
    1.1000003815f,
    0.5354322791f,
    1.0000000000f,
    1.0000000000f
};

/* Apply RobustScaler transformation: scaled = (raw - center) / scale */
static inline void scaler_transform(const float raw[SCALER_NUM_FEATURES], 
                                    float scaled[SCALER_NUM_FEATURES])
{
    for (int i = 0; i < SCALER_NUM_FEATURES; i++) {
        scaled[i] = (raw[i] - SCALER_CENTER[i]) / SCALER_SCALE[i];
    }
}

#endif /* SCALER_PARAMS_H */
