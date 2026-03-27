#ifndef ECM_PARAMETER_H
#define ECM_PARAMETER_H

#include <stdint.h>

/* ---------------- META INFORMATION ----------------
   comment: second-oder ECM parameters identified from HPPC test.
   author: [unsupported type]
   source: [unsupported type]
---------------------------------------------------- */

typedef struct {
    float Ri[101];
    float R1[101];
    float R2[101];
    float tau1[101];
    float tau2[101];
    float ocv[101];
    float dOCV[101];
} para_discharge_t;

typedef struct {
    float Ri[101];
    float R1[101];
    float R2[101];
    float tau1[101];
    float tau2[101];
    float ocv[101];
    float dOCV[101];
} para_charge_t;

typedef struct {
    para_discharge_t para_discharge;
    para_charge_t para_charge;
    float soc[101];
} ECM_t;

extern const ECM_t ECM;

#endif
