/*
 * battery_ekf.c
 *
 * Implementierung eines erweiterten Kalman-Filters (EKF)
 * zur Zustandsabschätzung einer Batterie (SoC + RC-Glieder)
 *
 * Zustände:
 *  x[0] = State of Charge (SoC)
 *  x[1] = Spannung am RC-Glied 1
 *  x[2] = Spannung am RC-Glied 2
 *
 *  Created on: Oct 19, 2025
 *      Author: CQN
 */

#include "arm_math.h"        // CMSIS-DSP Matrixfunktionen
#include "ECM_parameter.h"  // Batterieparameter (Lookup-Tabellen)
#include "interpolation.h"  // 1D-Interpolation der Parameter

void battery_EKF(
	float *x_k1, float *y_k1,   // Ausgang: geschätzter Zustand & Spannung
	float I, float Ut, float I_prev, float SoH, // Strom, Klemmenspannung, vorheriger Strom, State of Health
	float P_k1[3][3],           // Ausgang: aktualisierte Kovarianz
	float x[3], float P[3][3])  // Eingang: aktueller Zustand & Kovarianz
{
	/* ==========================
	 * Batterie- und Filterparameter
	 * ========================== */

	float C0 = 1.7838f * 3600.0f; // Nennkapazität [As]
	float Cb = C0 * SoH;          // effektive Kapazität unter Berücksichtigung des SoH
	float deltaT = 60.0f;         // Abtastzeit [s]
	float eff = 1.0f;             // Coulomb-Wirkungsgrad
	float Q[3][3] = {             // Prozessrauschkovarianz
		{1e-10, 0, 0},
		{0, 2e-5, 0},
		{0, 0, 2e-5}
	};
	float R = 9e-4f;              // Messrauschvarianz (Spannung)

	/* ==========================
	 * SoC-Sättigung
	 * ========================== */

	float SoC = x[0];
	if (SoC < 0) SoC = 0;
	if (SoC > 1) SoC = 1;

	/* ==========================
	 * Parameterwahl je nach Lade-/Entladerichtung
	 * ========================== */

	float R1, R2, tau1, tau2, Ri, OCV, dOCV;

	if (I_prev >= 0) {
		// Ladebetrieb
		R1   = interpolate1D(&charge_R1, SoC);
		R2   = interpolate1D(&charge_R2, SoC);
		tau1 = interpolate1D(&charge_tau1, SoC);
		tau2 = interpolate1D(&charge_tau2, SoC);
		eff  = 0.999f; // leicht reduzierter Wirkungsgrad beim Laden
	} else {
		// Entladebetrieb
		R1   = interpolate1D(&discharge_R1, SoC);
		R2   = interpolate1D(&discharge_R2, SoC);
		tau1 = interpolate1D(&discharge_tau1, SoC);
		tau2 = interpolate1D(&discharge_tau2, SoC);
	}

	/* ==========================
	 * Prädiktionsschritt (Zustandsvorhersage)
	 * ========================== */

	// Diskrete Zustandsmatrix
	float Ad[3][3] = {
		{1, 0, 0},
		{0, expf(-deltaT / tau1), 0},
		{0, 0, expf(-deltaT / tau2)}
	};

	// Diskreter Eingangseinfluss
	float Bd[3] = {
		eff * deltaT / Cb,                    // SoC-Änderung durch Strom
		R1 * (1 - expf(-deltaT / tau1)),      // RC-Glied 1
		R2 * (1 - expf(-deltaT / tau2))       // RC-Glied 2
	};

	// Zustandsvorhersage x(k|k-1)
	float x_p[3];
	for (int i = 0; i < 3; i++) {
		x_p[i] = Ad[i][0] * x[0]
		       + Ad[i][1] * x[1]
		       + Ad[i][2] * x[2]
		       + Bd[i] * I_prev;
	}

	/* ==========================
	 * Prädiktion der Kovarianz Kalman Filter
	 * P(k|k-1) = A*P*A' + Q
	 * ========================== */

	arm_matrix_instance_f32 Ad_m, P_m, AdT_m, Pp_m, Q_m;
	float AdT[9], Pp[9];

	arm_mat_init_f32(&Ad_m,  3, 3, (float*)Ad);
	arm_mat_init_f32(&P_m,   3, 3, (float*)P);
	arm_mat_init_f32(&Q_m,   3, 3, (float*)Q);
	arm_mat_init_f32(&AdT_m, 3, 3, (float*)AdT);
	arm_mat_init_f32(&Pp_m,  3, 3, (float*)Pp);

	arm_mat_trans_f32(&Ad_m, &AdT_m);          // A'
	arm_mat_mult_f32(&Ad_m, &P_m, &Pp_m);      // A*P
	arm_mat_mult_f32(&Pp_m, &AdT_m, &Pp_m);    // A*P*A'
	arm_mat_add_f32(&Pp_m, &Q_m, &Pp_m);       // + Q

	/* ==========================
	 * Messmodell (Spannung)
	 * ========================== */

	if (I >= 0) {
		Ri   = interpolate1D(&charge_Ri, SoC);
		OCV  = interpolate1D(&charge_OCV, SoC);
		dOCV = interpolate1D(&charge_dOCV, SoC);
	} else {
		Ri   = interpolate1D(&discharge_Ri, SoC);
		OCV  = interpolate1D(&discharge_OCV, SoC);
		dOCV = interpolate1D(&discharge_dOCV, SoC);
	}

	// Messmatrix (linearisiert)
	float Cd[3] = { dOCV, 1.0f, 1.0f };

	// Vorhergesagte Klemmenspannung
	float y_p = OCV
	          + dOCV * (x_p[0] - SoC)
	          + x_p[1]
	          + x_p[2]
	          + Ri * I;

	float delta_y = Ut - y_p;  // Innovationssignal

	/* ==========================
	 * Innovationskovarianz
	 * S = C*P*C' + R
	 * ========================== */

	float S =
		Cd[0] * (Pp[0] * Cd[0] + Pp[1] * Cd[1] + Pp[2] * Cd[2]) +
		Cd[1] * (Pp[3] * Cd[0] + Pp[4] * Cd[1] + Pp[5] * Cd[2]) +
		Cd[2] * (Pp[6] * Cd[0] + Pp[7] * Cd[1] + Pp[8] * Cd[2]) +
		R; // explizit ausmultipliziert (Rechenaufwand sparen)

	/* ==========================
	 * Kalman-Gewinn
	 * K = P*C' / S
	 * ========================== */

	float K[3];
	for (int i = 0; i < 3; i++) {
		float temp =
			Pp[i * 3 + 0] * Cd[0] +
			Pp[i * 3 + 1] * Cd[1] +
			Pp[i * 3 + 2] * Cd[2];
		K[i] = temp / S;
	}

	/* ==========================
	 * Zustandskorrektur
	 * x(k|k) = x_p + K*(y - y_p)
	 * ========================== */

	for (int i = 0; i < 3; i++)
		x_k1[i] = x_p[i] + K[i] * delta_y;

	// SoC-Sättigung nach Update
	if (x_k1[0] < 0) x_k1[0] = 0;
	if (x_k1[0] > 1) x_k1[0] = 1;

	/* ==========================
	 * Ausgangsspannung nach Update
	 * ========================== */

	*y_k1 = OCV
	      + dOCV * (x_k1[0] - SoC)
	      + x_k1[1]
	      + x_k1[2]
	      + Ri * I;
}
