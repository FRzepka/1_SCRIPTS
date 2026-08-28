#ifndef ECM_TABLES_H
#define ECM_TABLES_H
#define ECM_SOC_COUNT 100U
#define ECM_SOH_COUNT 40U
extern const double ecm_soc[ECM_SOC_COUNT];
extern const double ecm_soh[ECM_SOH_COUNT];
#define ECM_DECLARE(mode, name) extern const double ecm_##mode##_##name[ECM_SOC_COUNT * ECM_SOH_COUNT]
ECM_DECLARE(discharge, Ri); ECM_DECLARE(discharge, R1); ECM_DECLARE(discharge, R2);
ECM_DECLARE(discharge, tau1); ECM_DECLARE(discharge, tau2); ECM_DECLARE(discharge, ocv); ECM_DECLARE(discharge, dOCV);
ECM_DECLARE(charge, Ri); ECM_DECLARE(charge, R1); ECM_DECLARE(charge, R2);
ECM_DECLARE(charge, tau1); ECM_DECLARE(charge, tau2); ECM_DECLARE(charge, ocv); ECM_DECLARE(charge, dOCV);
#endif
