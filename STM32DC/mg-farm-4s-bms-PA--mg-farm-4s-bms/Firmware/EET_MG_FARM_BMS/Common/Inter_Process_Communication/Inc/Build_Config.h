/*
 * Build_Config.h
 *
 * Central compile-time switches for devboard bring-up vs. real BMS hardware.
 *
 * Note: Use `#if ...` in code (not `#ifdef`) so these can be overridden with
 * `-DNAME=0` from the IDE.
 */

#ifndef BUILD_CONFIG_H_
#define BUILD_CONFIG_H_

/* NUCLEO-H755ZI-Q devboard support (LEDs, no external peripherals). */
#ifndef DEVBOARD_NUCLEO
#define DEVBOARD_NUCLEO 1
#endif

/*
 * When 1: skip TLE9012 access and accept measurements injected via USART3 (RS485).
 * When 0: use the real TLE9012-based measurement path.
 */
#ifndef BMS_SIM_INPUT
#define BMS_SIM_INPUT 1
#endif

/*
 * Bring-up mode: keep boot/clock sequences minimal to make the target always debuggable.
 * Set to 0 once the basic CM4<->CM7 + UART pipeline is stable.
 */
#ifndef BRINGUP_MINIMAL
#define BRINGUP_MINIMAL 1
#endif

/*
 * Reconstruct cycle features on-device (recommended for STM32 bring-up):
 * - Q_c: coulomb count since full (reset at U_max), clamped to <=0
 * - EFC: equivalent full cycles from absolute throughput
 *
 * If enabled, incoming EFC/Q_c fields from the host are ignored.
 */
#ifndef BMS_RECONSTRUCT_CYCLE_FEATURES
#define BMS_RECONSTRUCT_CYCLE_FEATURES 1
#endif

/* Cell-level voltage threshold (V) for Q_c reset ("full reached"). */
#ifndef BMS_QC_U_MAX_CELL_V
#define BMS_QC_U_MAX_CELL_V 3.6002f
#endif

/*
 * Reference capacities [Ah] used by the dataset pre-processing.
 * These values were inferred from df_FE_C11.parquet and are used so STM32-side reconstruction
 * matches the PC features closely during validation runs.
 */
#ifndef BMS_QC_CAP_REF_AH
#define BMS_QC_CAP_REF_AH 1.8170391f
#endif

#ifndef BMS_EFC_CAP_REF_AH
#define BMS_EFC_CAP_REF_AH 1.8162782f
#endif

/*
 * Step-by-step bring-up switch:
 * 0 = dummy SOH (quick compile, proves dual-core + UART + mailbox path)
 * 1 = use the embedded quantized SOH model (slower build, more flash/RAM)
 */
#ifndef SOH_USE_MODEL
#define SOH_USE_MODEL 1
#endif

#endif /* BUILD_CONFIG_H_ */
