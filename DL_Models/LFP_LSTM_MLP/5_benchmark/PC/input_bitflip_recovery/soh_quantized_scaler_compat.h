#ifndef SOH_QUANTIZED_SCALER_COMPAT_H
#define SOH_QUANTIZED_SCALER_COMPAT_H

#include <string.h>
#include "scaler_params_soh.h"

static inline void runner_scaler_soh_transform(
    const float raw[SCALER_NUM_FEATURES],
    float scaled[SCALER_NUM_FEATURES])
{
    memcpy(scaled, raw, SCALER_NUM_FEATURES * sizeof(float));
    scaler_soh_transform(scaled);
}

/* The active source expects a two-buffer API, while its scaler is in-place. */
#define scaler_soh_transform(raw, scaled) runner_scaler_soh_transform(raw, scaled)

#endif
