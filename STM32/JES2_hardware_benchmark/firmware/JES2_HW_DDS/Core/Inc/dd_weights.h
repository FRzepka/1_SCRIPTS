#ifndef DD_WEIGHTS_H
#define DD_WEIGHTS_H

#include "dd_model.h"

extern const float dd_scaler_center[DD_INPUT_SIZE];
extern const float dd_scaler_scale[DD_INPUT_SIZE];
extern const float dd_gru_w[3U * DD_HIDDEN_SIZE * DD_INPUT_SIZE];
extern const float dd_gru_r[3U * DD_HIDDEN_SIZE * DD_HIDDEN_SIZE];
extern const float dd_gru_b[6U * DD_HIDDEN_SIZE];
extern const float dd_mlp_w1[DD_MLP_SIZE * DD_HIDDEN_SIZE];
extern const float dd_mlp_b1[DD_MLP_SIZE];
extern const float dd_mlp_w2[DD_MLP_SIZE];
extern const float dd_mlp_b2;

#endif
