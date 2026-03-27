/*
 * battery_ekf.h
 *
 *  Created on: Oct 19, 2025
 *      Author: CQN
 */

#ifndef INC_BATTERY_EKF_H_
#define INC_BATTERY_EKF_H_

#include "arm_math.h"

void battery_EKF(
		float *x_k1, float *y_k1,
		float I, float Ut, float I_prev, float SoH,
		float P_k1[3][3],
		float x[3], float P[3][3]);

#endif /* INC_BATTERY_EKF_H_ */
