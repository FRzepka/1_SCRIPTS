/*
 * interpolation.h
 *
 *  Created on: Oct 17, 2025
 *      Author: CQN
 */

#ifndef INC_INTERPOLATION_H_
#define INC_INTERPOLATION_H_



#include "main.h"
#include <stdint.h>
#include "ECM_parameter.h"

typedef struct {
    const float *x;
    const float *y;
    uint16_t size;
} LookupTable1D;

float interpolate1D(const LookupTable1D *table, float x);

extern LookupTable1D charge_Ri;
extern LookupTable1D charge_R1;
extern LookupTable1D charge_R2;
extern LookupTable1D charge_tau1;
extern LookupTable1D charge_tau2;
extern LookupTable1D charge_OCV;
extern LookupTable1D charge_dOCV;

extern LookupTable1D discharge_Ri;
extern LookupTable1D discharge_R1;
extern LookupTable1D discharge_R2;
extern LookupTable1D discharge_tau1;
extern LookupTable1D discharge_tau2;
extern LookupTable1D discharge_OCV;
extern LookupTable1D discharge_dOCV;

#endif /* INC_INTERPOLATION_H_ */
