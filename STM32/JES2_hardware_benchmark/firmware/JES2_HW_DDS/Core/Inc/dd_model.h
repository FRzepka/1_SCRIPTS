#ifndef DD_MODEL_H
#define DD_MODEL_H

#include <stdint.h>

#define DD_INPUT_SIZE 8U
#define DD_HIDDEN_SIZE 67U
#define DD_MLP_SIZE 96U
#define DD_SEQUENCE_LENGTH 2024U

void DD_ModelReset(void);
uint8_t DD_ModelPush(const float input[DD_INPUT_SIZE], float *soc);

#endif
