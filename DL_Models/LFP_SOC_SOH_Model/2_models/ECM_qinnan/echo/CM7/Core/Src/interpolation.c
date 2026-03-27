/*
 * interpolation.c
 *
 *  Created on: Oct 17, 2025
 *      Author: CQN
 */

#include "interpolation.h"

LookupTable1D charge_Ri={ECM.soc, ECM.para_charge.Ri,sizeof(ECM.soc)/sizeof(float)};
LookupTable1D charge_R1={ECM.soc, ECM.para_charge.R1,sizeof(ECM.soc)/sizeof(float)};
LookupTable1D charge_R2={ECM.soc, ECM.para_charge.R2,sizeof(ECM.soc)/sizeof(float)};
LookupTable1D charge_tau1={ECM.soc, ECM.para_charge.tau1,sizeof(ECM.soc)/sizeof(float)};
LookupTable1D charge_tau2={ECM.soc, ECM.para_charge.tau2,sizeof(ECM.soc)/sizeof(float)};
LookupTable1D charge_OCV={ECM.soc, ECM.para_charge.ocv,sizeof(ECM.soc)/sizeof(float)};
LookupTable1D charge_dOCV={ECM.soc, ECM.para_charge.dOCV,sizeof(ECM.soc)/sizeof(float)};

LookupTable1D discharge_Ri={ECM.soc, ECM.para_discharge.Ri, sizeof(ECM.soc)/sizeof(float)};
LookupTable1D discharge_R1={ECM.soc, ECM.para_discharge.R1, sizeof(ECM.soc)/sizeof(float)};
LookupTable1D discharge_R2={ECM.soc, ECM.para_discharge.R2, sizeof(ECM.soc)/sizeof(float)};
LookupTable1D discharge_tau1={ECM.soc, ECM.para_discharge.tau1, sizeof(ECM.soc)/sizeof(float)};
LookupTable1D discharge_tau2={ECM.soc, ECM.para_discharge.tau2, sizeof(ECM.soc)/sizeof(float)};
LookupTable1D discharge_OCV={ECM.soc, ECM.para_discharge.ocv, sizeof(ECM.soc)/sizeof(float)};
LookupTable1D discharge_dOCV={ECM.soc, ECM.para_discharge.dOCV, sizeof(ECM.soc)/sizeof(float)};


float interpolate1D(const LookupTable1D *table, float x)
{
	if (x<= table->x[0])
		return table->y[0];
	if (x >=table->x[table->size -1])
		return table->y[table->size-1];


	for(uint16_t i=0; i< table->size-1;i++)
	{
		if(x<= table->x[i+1] && x>=table->x[i])
		{
			float x0=table->x[i], x1=table->x[i+1];
			float y0=table->y[i], y1=table->y[i+1];
			float t=(x-x0)/(x1-x0);
			return y0+t*(y1-y0);
		}

	}
	return table->y[table->size -1];
}

