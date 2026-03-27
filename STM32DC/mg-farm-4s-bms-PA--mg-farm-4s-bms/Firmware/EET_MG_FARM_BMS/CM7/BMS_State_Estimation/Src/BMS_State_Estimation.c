/*
 * BMS_State_Estimation.c
 *
 *  Created on: Jan 15, 2024
 *      Author: max
 */


#include "BMS_State_Estimation.h"
#include "Shared_DataTypes.h"
#include "soh_inference.h"


float StateEstimationGetSoCbyCoulombCounting(Cell_Module_t* module);

//Do all State Estimation in this callback

uint32_t last_timestamp = 0;
float coulomb_counter_value = BATTERYCAPACITY;




State_Estimation_t bmsStateEstimationCallback(Cell_Module_t* module)
{


	State_Estimation_t new_state_estimation = {};
	//Set new State Estimations
	new_state_estimation.module_SoC = StateEstimationGetSoCbyCoulombCounting(module);
	new_state_estimation.module_SoH = soh_inference_step(module);
	new_state_estimation.timestamp = soh_inference_last_timestamp_ms();


	return new_state_estimation;
}

float StateEstimationGetSoCbyCoulombCounting(Cell_Module_t* module)
{
	if(last_timestamp == 0)
	{
		//Only called in first iteration after bootup
		last_timestamp = module->timestamp;
	}
	else
	{
		/* PC replay can restart timestamps at 0 without a board reset. */
		if (module->timestamp < last_timestamp)
		{
			last_timestamp = module->timestamp;
			coulomb_counter_value = BATTERYCAPACITY;
			return coulomb_counter_value / BATTERYCAPACITY;
		}

		uint32_t delta_t = module->timestamp - last_timestamp;
		last_timestamp = module->timestamp;

		/* Current convention: +charge, -discharge. */
		coulomb_counter_value = coulomb_counter_value + (module->current*delta_t)/1000;
	}

	//Coulomb Counter Upper Limit
	if(coulomb_counter_value > BATTERYCAPACITY)
		coulomb_counter_value = BATTERYCAPACITY;

	//Coulomb Counter Lower Limit
	if(coulomb_counter_value < 0)
		coulomb_counter_value = 0;

	return coulomb_counter_value/BATTERYCAPACITY;
}
