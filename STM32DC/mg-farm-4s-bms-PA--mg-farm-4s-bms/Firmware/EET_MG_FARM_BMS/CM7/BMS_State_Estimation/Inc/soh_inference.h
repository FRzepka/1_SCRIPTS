/*
 * soh_inference.h
 *
 * Minimal on-device SOH inference wrapper using the existing C implementation
 * from `DL_Models/LFP_LSTM_MLP/.../soh_2.1.0.0_quantized/c_implementation`.
 */

#ifndef SOH_INFERENCE_H_
#define SOH_INFERENCE_H_

#include "Shared_DataTypes.h"

void soh_inference_init(void);
float soh_inference_step(const Cell_Module_t *module);
uint32_t soh_inference_last_timestamp_ms(void);

#endif /* SOH_INFERENCE_H_ */
