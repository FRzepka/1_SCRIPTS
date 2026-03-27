/* Auto-generated: SOH model 0.1.2.3 (hourly stats) */
#ifndef SOH_0_1_2_3_WEIGHTS_H
#define SOH_0_1_2_3_WEIGHTS_H

#include <stdint.h>

#define SOH_0123_IN_FEATURES 20
#define SOH_0123_EMBED 128
#define SOH_0123_HIDDEN 192
#define SOH_0123_LAYERS 3
#define SOH_0123_MLP_HIDDEN 160

extern const int8_t SOH0123_FP0_W_Q[SOH_0123_EMBED * SOH_0123_IN_FEATURES];
extern const float  SOH0123_FP0_W_S[SOH_0123_EMBED];
extern const float  SOH0123_FP0_B[SOH_0123_EMBED];
extern const float  SOH0123_LN0_W[SOH_0123_EMBED];
extern const float  SOH0123_LN0_B[SOH_0123_EMBED];

extern const int8_t SOH0123_FP1_W_Q[SOH_0123_EMBED * SOH_0123_EMBED];
extern const float  SOH0123_FP1_W_S[SOH_0123_EMBED];
extern const float  SOH0123_FP1_B[SOH_0123_EMBED];

/* LSTM: layer 0 input size = embed, layers >=1 input size = hidden */
extern const int8_t SOH0123_L0_WIH_Q[4 * SOH_0123_HIDDEN * SOH_0123_EMBED];
extern const float  SOH0123_L0_WIH_S[4 * SOH_0123_HIDDEN];
extern const int8_t SOH0123_L0_WHH_Q[4 * SOH_0123_HIDDEN * SOH_0123_HIDDEN];
extern const float  SOH0123_L0_WHH_S[4 * SOH_0123_HIDDEN];
extern const float  SOH0123_L0_B[4 * SOH_0123_HIDDEN];

extern const int8_t SOH0123_L1_WIH_Q[4 * SOH_0123_HIDDEN * SOH_0123_HIDDEN];
extern const float  SOH0123_L1_WIH_S[4 * SOH_0123_HIDDEN];
extern const int8_t SOH0123_L1_WHH_Q[4 * SOH_0123_HIDDEN * SOH_0123_HIDDEN];
extern const float  SOH0123_L1_WHH_S[4 * SOH_0123_HIDDEN];
extern const float  SOH0123_L1_B[4 * SOH_0123_HIDDEN];

extern const int8_t SOH0123_L2_WIH_Q[4 * SOH_0123_HIDDEN * SOH_0123_HIDDEN];
extern const float  SOH0123_L2_WIH_S[4 * SOH_0123_HIDDEN];
extern const int8_t SOH0123_L2_WHH_Q[4 * SOH_0123_HIDDEN * SOH_0123_HIDDEN];
extern const float  SOH0123_L2_WHH_S[4 * SOH_0123_HIDDEN];
extern const float  SOH0123_L2_B[4 * SOH_0123_HIDDEN];

extern const float  SOH0123_POSTLN_W[SOH_0123_HIDDEN];
extern const float  SOH0123_POSTLN_B[SOH_0123_HIDDEN];

/* Residual blocks */
extern const int8_t SOH0123_RB0_FC1_W_Q[SOH_0123_MLP_HIDDEN * SOH_0123_HIDDEN];
extern const float  SOH0123_RB0_FC1_W_S[SOH_0123_MLP_HIDDEN];
extern const float  SOH0123_RB0_FC1_B[SOH_0123_MLP_HIDDEN];
extern const int8_t SOH0123_RB0_FC2_W_Q[SOH_0123_HIDDEN * SOH_0123_MLP_HIDDEN];
extern const float  SOH0123_RB0_FC2_W_S[SOH_0123_HIDDEN];
extern const float  SOH0123_RB0_FC2_B[SOH_0123_HIDDEN];
extern const float  SOH0123_RB0_LN_W[SOH_0123_HIDDEN];
extern const float  SOH0123_RB0_LN_B[SOH_0123_HIDDEN];

extern const int8_t SOH0123_RB1_FC1_W_Q[SOH_0123_MLP_HIDDEN * SOH_0123_HIDDEN];
extern const float  SOH0123_RB1_FC1_W_S[SOH_0123_MLP_HIDDEN];
extern const float  SOH0123_RB1_FC1_B[SOH_0123_MLP_HIDDEN];
extern const int8_t SOH0123_RB1_FC2_W_Q[SOH_0123_HIDDEN * SOH_0123_MLP_HIDDEN];
extern const float  SOH0123_RB1_FC2_W_S[SOH_0123_HIDDEN];
extern const float  SOH0123_RB1_FC2_B[SOH_0123_HIDDEN];
extern const float  SOH0123_RB1_LN_W[SOH_0123_HIDDEN];
extern const float  SOH0123_RB1_LN_B[SOH_0123_HIDDEN];

/* Head */
extern const int8_t SOH0123_HEAD0_W_Q[SOH_0123_MLP_HIDDEN * SOH_0123_HIDDEN];
extern const float  SOH0123_HEAD0_W_S[SOH_0123_MLP_HIDDEN];
extern const float  SOH0123_HEAD0_B[SOH_0123_MLP_HIDDEN];
extern const int8_t SOH0123_HEAD1_W_Q[SOH_0123_MLP_HIDDEN * SOH_0123_MLP_HIDDEN];
extern const float  SOH0123_HEAD1_W_S[SOH_0123_MLP_HIDDEN];
extern const float  SOH0123_HEAD1_B[SOH_0123_MLP_HIDDEN];
extern const int8_t SOH0123_HEAD2_W_Q[1 * SOH_0123_MLP_HIDDEN];
extern const float  SOH0123_HEAD2_W_S[1];
extern const float  SOH0123_HEAD2_B[1];

#endif /* SOH_0_1_2_3_WEIGHTS_H */
