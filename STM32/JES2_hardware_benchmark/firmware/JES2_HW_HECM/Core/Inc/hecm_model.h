#ifndef HECM_MODEL_H
#define HECM_MODEL_H
void HECM_Reset(void);
float HECM_Step(float current, float voltage, float soh, float dt);
#endif
