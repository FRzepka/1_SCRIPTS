/*
 * battery_ekf.c
 *
 *  Created on: Oct 19, 2025
 *      Author: CQN
 */

#include "arm_math.h"
#include "ECM_parameter.h"
#include "interpolation.h"

static void battery_EKF_core(
	float *x_k1, float *y_k1,
	float I, float Ut, float I_prev, float SoH,
	float deltaT,
	float P_k1[3][3],
	float x[3], float P[3][3])
{
	float C0= 1.7838f*3600.0f;
	float Cb=C0*SoH;
	float eff=1.0f;
	float Q[3][3]={{1e-10,0,0},{0,2e-5,0},{0,0,2e-5}};
	float R=9e-4f;

	float SoC=x[0];
	if(SoC<0) SoC=0;
	if(SoC>1) SoC=1;


	// Parameter of charge or discharge
	float R1,R2,tau1,tau2,Ri,OCV,dOCV;
	if(I_prev>=0){
		R1=lookup2D_nearestSOH(&charge_R1, SoC,SoH);
		R2=lookup2D_nearestSOH(&charge_R2, SoC,SoH);
		tau1=lookup2D_nearestSOH(&charge_tau1, SoC,SoH);
		tau2=lookup2D_nearestSOH(&charge_tau2, SoC,SoH);
		eff=0.999f;
	}
	else{
		R1=lookup2D_nearestSOH(&discharge_R1, SoC,SoH);
		R2=lookup2D_nearestSOH(&discharge_R2, SoC,SoH);
		tau1=lookup2D_nearestSOH(&discharge_tau1, SoC,SoH);
		tau2=lookup2D_nearestSOH(&discharge_tau2, SoC,SoH);
	}


	// prediction
	float Ad[3][3]={
			{1,0,0},
			{0,expf(-deltaT/tau1),0},
			{0,0,expf(-deltaT/tau2)}
	};
	float Bd[3]={
			eff*deltaT/Cb,
			R1*(1-expf(-deltaT/tau1)),
			R2*(1-expf(-deltaT/tau2))
	};
	float x_p[3];
	for(int i=0;i<3;i++){
		x_p[i] = Ad[i][0]*x[0] + Ad[i][1]*x[1] + Ad[i][2]*x[2] + Bd[i]*I_prev;
	}


	arm_matrix_instance_f32 Ad_m, P_m, AdT_m, Pp_m, Q_m;
	float AdT[9], Pp[9];

	arm_mat_init_f32(&Ad_m, 3,3, (float*)Ad);
	arm_mat_init_f32(&P_m, 3,3, (float*)P);
	arm_mat_init_f32(&Q_m, 3,3, (float*)Q);
	arm_mat_init_f32(&AdT_m, 3, 3, (float*)AdT);
	arm_mat_init_f32(&Pp_m, 3, 3, (float*)Pp);

	arm_mat_trans_f32(&Ad_m, &AdT_m);
	arm_mat_mult_f32(&Ad_m, &P_m, &Pp_m);
	arm_mat_mult_f32(&Pp_m, &AdT_m, &Pp_m);
	arm_mat_add_f32(&Pp_m, &Q_m, &Pp_m);


	if(I>=0){
		Ri=lookup2D_nearestSOH(&charge_Ri, SoC,SoH);
		OCV=lookup2D_nearestSOH(&charge_OCV, SoC,SoH);
		dOCV=lookup2D_nearestSOH(&charge_dOCV, SoC,SoH);
	}
	else{
		Ri=lookup2D_nearestSOH(&discharge_Ri, SoC,SoH);
		OCV=lookup2D_nearestSOH(&discharge_OCV, SoC,SoH);
		dOCV=lookup2D_nearestSOH(&discharge_dOCV, SoC,SoH);
	}

	float Cd[3]={dOCV,1.0f,1.0f};
	float y_p=OCV+dOCV*(x_p[0]-SoC)+x_p[1]+x_p[2]+Ri*I;

	float delta_y=Ut-y_p;
	//S=Cd*P_p*Cd'+R
	float S= Cd[0]*(Pp[0]*Cd[0] + Pp[1]*Cd[1] + Pp[2]*Cd[2]) +
            Cd[1]*(Pp[3]*Cd[0] + Pp[4]*Cd[1] + Pp[5]*Cd[2]) +
            Cd[2]*(Pp[6]*Cd[0] + Pp[7]*Cd[1] + Pp[8]*Cd[2]) + R;//需要多建几个中间矩阵，所以就干脆展开写了

	float K[3];
	//K=P_p*Cd'/S;
	for(int i=0;i<3;i++){
		float temp = Pp[i*3+0]*Cd[0] + Pp[i*3+1]*Cd[1] + Pp[i*3+2]*Cd[2];
		K[i]=temp/S;
	}
	//x_k1=x_p+K*delta_y;
	for(int i=0;i<3;i++) x_k1[i] = x_p[i] + K[i]*delta_y;
	if(x_k1[0]<0) x_k1[0]=0;
	if(x_k1[0]>1) x_k1[0]=1;


	*y_k1 = OCV + dOCV*(x_k1[0]-SoC) + x_k1[1] + x_k1[2] + Ri*I;


}

// Legacy API: keeps previous fixed-step behavior at 60 s.
void battery_EKF(
	float *x_k1, float *y_k1,
	float I, float Ut, float I_prev, float SoH,
	float P_k1[3][3],
	float x[3], float P[3][3])
{
	battery_EKF_core(x_k1, y_k1, I, Ut, I_prev, SoH, 60.0f, P_k1, x, P);
}

// New API: allows runtime sampling time (seconds).
void battery_EKF_dt(
	float *x_k1, float *y_k1,
	float I, float Ut, float I_prev, float SoH, float deltaT,
	float P_k1[3][3],
	float x[3], float P[3][3])
{
	if(deltaT <= 0.0f) deltaT = 1.0f;
	battery_EKF_core(x_k1, y_k1, I, Ut, I_prev, SoH, deltaT, P_k1, x, P);
}
