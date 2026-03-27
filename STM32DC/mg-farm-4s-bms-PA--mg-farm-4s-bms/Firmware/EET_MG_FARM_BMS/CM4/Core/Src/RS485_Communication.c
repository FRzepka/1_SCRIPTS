#include "RS485_Communication.h"
#include "Shared_DataTypes.h"
#include "BMS_Functions.h"
#include "Build_Config.h"
#include "IPPC_Functions.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

volatile uint8_t rs485_tx_cplt_flag = 1;
volatile uint8_t rs485_rx_cplt_flag = 0;

uint8_t receive_in_progress = 0;
uint8_t rs485_rxbuffer[RS485_RX_BUFFERSIZE];
static uint8_t tx_buffer[255];

UART_HandleTypeDef *rs485uart;
static Cell_Module_t *rs485_module;
static State_Estimation_t *rs485_estimations;

typedef enum
{
	RS485_RX_STATE_HEADER = 0,
	RS485_RX_STATE_PAYLOAD = 1,
} rs485_rx_state_t;

static volatile rs485_rx_state_t rs485_rx_state = RS485_RX_STATE_HEADER;
static volatile uint8_t rs485_expected_len = 0;
static volatile uint8_t rs485_frame_ready = 0;
static volatile uint8_t rs485_drop_count = 0;

static struct
{
	uint8_t cmd;
	uint8_t len;
	uint8_t payload[RS485_RX_BUFFERSIZE - 3U];
} rs485_frame;

static void rs485_start_header_rx(void)
{
	rs485_rx_state = RS485_RX_STATE_HEADER;
	rs485_expected_len = 0;
	rs485_rx_cplt_flag = 0;
	receive_in_progress = 1;
	(void)HAL_UART_Receive_IT(rs485uart, rs485_rxbuffer, 3);
}

/** @brief init routine for the rs485 bus
 * 	@note this function has to be called before any other function calls from this library
 * 	@param huart UART handle
 * 	@param hmodule pointer to the global module variable
 */

void rs485_communication_init(UART_HandleTypeDef *huart, Cell_Module_t *hmodule, State_Estimation_t *hestimations)
{
	rs485uart = huart;
	rs485_module = hmodule;
	rs485_estimations = hestimations;

	/* Arm RX immediately so we can catch back-to-back frames (header+payload). */
	rs485_start_header_rx();
}

/** @brief init routine for the rs485 bus
 * 	@note this function has to be called before any other function calls from this library
 * 	@param huart UART handle
 * 	@param hmodule pointer to the global module variable
 */

void rs485_communication_loop()
{
	if (rs485_frame_ready == 0)
	{
		return;
	}

	/*
	 * Copy message out of the ISR-owned buffer (the ISR re-arms reception immediately).
	 * Keep this critical section small; payload is <= 125 bytes.
	 */
	uint8_t cmd = 0;
	uint8_t len = 0;
	uint8_t payload[RS485_RX_BUFFERSIZE - 3U];

	__disable_irq();
	cmd = rs485_frame.cmd;
	len = rs485_frame.len;
	if (len > 0U)
	{
		memcpy(payload, rs485_frame.payload, len);
	}
	rs485_frame_ready = 0;
	__enable_irq();

	rs485_process_package(cmd, len, (len > 0U) ? payload : NULL);
}

void rs485_uart_rx_cplt_isr(UART_HandleTypeDef *huart)
{
	if (huart != rs485uart)
	{
		return;
	}

	if (rs485_rx_state == RS485_RX_STATE_HEADER)
	{
		const uint8_t id = rs485_rxbuffer[0];
		const uint8_t cmd = rs485_rxbuffer[1];
		const uint8_t len = rs485_rxbuffer[2];

		if (id != RS485_ID || len > (RS485_RX_BUFFERSIZE - 3U))
		{
			rs485_start_header_rx();
			return;
		}

		if (len == 0U)
		{
			if (rs485_frame_ready == 0)
			{
				rs485_frame.cmd = cmd;
				rs485_frame.len = 0;
				rs485_frame_ready = 1;
			}
			else
			{
				rs485_drop_count++;
			}
			rs485_start_header_rx();
			return;
		}

		rs485_rx_state = RS485_RX_STATE_PAYLOAD;
		rs485_expected_len = len;
		rs485_rx_cplt_flag = 0;
		(void)HAL_UART_Receive_IT(rs485uart, &rs485_rxbuffer[3], rs485_expected_len);
		return;
	}

	/* Payload complete */
	if (rs485_frame_ready == 0)
	{
		rs485_frame.cmd = rs485_rxbuffer[1];
		rs485_frame.len = rs485_expected_len;
		memcpy(rs485_frame.payload, &rs485_rxbuffer[3], rs485_expected_len);
		rs485_frame_ready = 1;
	}
	else
	{
		rs485_drop_count++;
	}

	rs485_start_header_rx();
}

void rs485_uart_error_isr(UART_HandleTypeDef *huart)
{
	if (huart != rs485uart)
	{
		return;
	}

	(void)HAL_UART_AbortReceive_IT(rs485uart);
	rs485_start_header_rx();
}

/** @brief init routine for the rs485 bus
 * 	@note this function has to be called before any other function calls from this library
 * 	@param huart UART handle
 * 	@param hmodule pointer to the global module variable
 */

void rs485_process_package(uint8_t command, uint8_t payloadlen, uint8_t* payload)
{

	/* Avoid deadlocks: if a previous TX completion was missed, do not spin forever. */
	if (rs485_tx_cplt_flag != 1)
	{
		const uint32_t start = HAL_GetTick();
		while ((rs485_tx_cplt_flag != 1) && ((HAL_GetTick() - start) < 10U))
		{
			/* wait up to 10ms */
		}
		if (rs485_tx_cplt_flag != 1)
		{
			(void)HAL_UART_AbortTransmit_IT(rs485uart);
			rs485_tx_cplt_flag = 1;
		}
	}

	switch(command)	{
		case GET_MODULE_INFO_COMMAND:
			rs485_command_send_cell_data_as_struct(payloadlen, payload);
			break;

		case GET_MODULE_INFO_AS_STRING:
			rs485_command_send_cell_data_as_string(payloadlen, payload);
			break;

		case GET_CURRENT_BMS_LIMITS_AS_STRING:
			rs485_command_send_bms_limits_as_string(payloadlen, payload);
			break;

		case SET_BMS_LIMITS:
			rs485_command_set_bms_limits(payloadlen, payload);
			break;

		case GET_STATE_ESTIMATIONS_AS_STRING:
			rs485_command_send_state_estimations_as_string(payloadlen, payload);
			break;

		case SET_SIM_MEASUREMENTS:
			rs485_command_set_sim_measurements(payloadlen, payload);
			break;
	}
}

/** @brief init routine for the rs485 bus
 * 	@note this function has to be called before any other function calls from this library
 * 	@param huart UART handle
 * 	@param hmodule pointer to the global module variable
 */

void rs485_command_send_cell_data_as_struct(uint8_t payloadlen, uint8_t* payload)
{
	uint8_t buffersize = MAXIMUM_CELL_VOLTAGES*2 + MAXIMUM_AUX_VOLTAGES*2 + 4 + 4;


				tx_buffer[0] = 0x80;
				tx_buffer[1] = GET_MODULE_INFO_COMMAND;
				tx_buffer[2] = buffersize - 4;
				tx_buffer[buffersize-1] = '\n';

				for(uint8_t n = 0; n < MAXIMUM_CELL_VOLTAGES; n++)
				{
					tx_buffer[3+2*n] = (uint8_t)(rs485_module->cellVoltages[n]);
					tx_buffer[4+2*n] = (uint8_t)(rs485_module->cellVoltages[n]>>8);
				}

				uint8_t baseadr = MAXIMUM_CELL_VOLTAGES*2+3;

				for(uint8_t n = 0; n < MAXIMUM_AUX_VOLTAGES; n++)
				{
					tx_buffer[baseadr+2*n] =  (uint8_t)(rs485_module->auxVoltages[n]);
					tx_buffer[baseadr+1+2*n] = (uint8_t)(rs485_module->auxVoltages[n]>>8);
				}

				baseadr += 2*MAXIMUM_AUX_VOLTAGES;

				uint32_t current_int = *((uint32_t*)&rs485_module->current);
				tx_buffer[baseadr] = (uint8_t) current_int;
				tx_buffer[baseadr+1] = (uint8_t) (current_int>>8);
				tx_buffer[baseadr+2] = (uint8_t) (current_int>>16);
				tx_buffer[baseadr+3] = (uint8_t) (current_int>>24);

				rs485_tx_cplt_flag = 0;
				HAL_UART_Transmit_IT(rs485uart, tx_buffer, buffersize);
}

/** @brief init routine for the rs485 bus
 * 	@note this function has to be called before any other function calls from this library
 * 	@param huart UART handle
 * 	@param hmodule pointer to the global module variable
 */

void rs485_command_send_cell_data_as_string(uint8_t payloadlen, uint8_t* payload)
{
	static uint8_t rollingcounter = 0;
	uint8_t txlen;

	tx_buffer[0] = 0x80;
	tx_buffer[1] = GET_MODULE_INFO_AS_STRING;

	txlen = sprintf((char*) &tx_buffer[3], "%d,%d,%lu,%lu,%lu,%lu,%ld,%lu,%f\n",
		RS485_ID,
		rollingcounter,
		(uint32_t)((float) rs485_module->cellVoltages[0]*50000)/65536,
		(uint32_t)((float) rs485_module->cellVoltages[1]*50000)/65536,
		(uint32_t)((float) rs485_module->cellVoltages[2]*50000)/65536,
		(uint32_t)((float) rs485_module->cellVoltages[3]*50000)/65536,
		(int32_t)(1000*rs485_module->current),
		HAL_GetTick(),
		rs485_estimations->module_SoC);

	tx_buffer[2] = txlen+1;
	tx_buffer[txlen+3] = '\n';
	rollingcounter ++;
	rs485_tx_cplt_flag = 0;
	HAL_UART_Transmit_IT(rs485uart, tx_buffer, txlen+3);

}

/** @brief init routine for the rs485 bus
 * 	@note this function has to be called before any other function calls from this library
 * 	@param huart UART handle
 * 	@param hmodule pointer to the global module variable
 */

void rs485_command_send_bms_limits_as_string(uint8_t payloadlen, uint8_t* payload)
{
	uint8_t txlen = 0;

	tx_buffer[0] = 0x80;
	tx_buffer[1] = GET_CURRENT_BMS_LIMITS_AS_STRING;

	bms_config_limits_t limits = BMS_GetLimits();

	txlen = sprintf((char*) &tx_buffer[3], "%f,%f,%f,%f,%f,%f,%f,%f\n",
			limits.overvoltage,
			limits.undervoltage,
			limits.max_current_charge,
			limits.max_current_discharge,
			limits.overtemp,
			limits.undertemp,
			limits.shuntvalue,
			limits.senseampgain);

	tx_buffer[2] = txlen+1;
	tx_buffer[txlen+3] = '\n';

	rs485_tx_cplt_flag = 0;
	HAL_UART_Transmit_IT(rs485uart, tx_buffer, txlen+3);
}

/** @brief init routine for the rs485 bus
 * 	@note this function has to be called before any other function calls from this library
 * 	@param huart UART handle
 * 	@param hmodule pointer to the global module variable
 */

void rs485_command_set_bms_limits(uint8_t payloadlen, uint8_t* payload)
{
	if(sizeof(bms_config_limits_t) == (payloadlen*4))
	{
		bms_config_limits_t newlimits;
		memcpy(&newlimits, (bms_config_limits_t*)payload, sizeof(bms_config_limits_t));
		BMS_SetNewLimits(newlimits);

		tx_buffer[0] = 0x80;
		tx_buffer[1] = SET_BMS_LIMITS;
		tx_buffer[2] = 0;
		tx_buffer[3] = 0;
		rs485_tx_cplt_flag = 0;
		HAL_UART_Transmit_IT(rs485uart, tx_buffer, 4);
	}
}

void rs485_command_set_sim_measurements(uint8_t payloadlen, uint8_t* payload)
{
	/*
	 * Payload format (little-endian):
	 * - float pack_voltage_v
	 * - float current_a
	 * - float temperature_c
	 * - float EFC (optional; if not provided it will be estimated on-device)
	 * - float Q_c (optional; if not provided it will be estimated on-device)
	 * - uint32 timestamp_ms
	 *
	 * This is intended for devboard simulation when the TLE9012 is not present.
	 * The payload updates the shared module struct; the regular BMS loop can
	 * then use these values (if BMS_SIM_INPUT is enabled in BMS_Functions.c).
	 */
	if (payloadlen < (3U * 4U + 4U))
	{
		return;
	}

	float pack_voltage_v = 0.0f;
	float current_a = 0.0f;
	float temperature_c = 0.0f;
	float efc_in = rs485_module->EFC;
	float q_c_in = rs485_module->Q_c;
	uint32_t timestamp_ms = 0;

	memcpy(&pack_voltage_v, &payload[0], 4);
	memcpy(&current_a, &payload[4], 4);
	memcpy(&temperature_c, &payload[8], 4);

	/* Backwards compatible: old payload was 3 floats + uint32. */
	if (payloadlen >= (5U * 4U + 4U))
	{
		memcpy(&efc_in, &payload[12], 4);
		memcpy(&q_c_in, &payload[16], 4);
		memcpy(&timestamp_ms, &payload[20], 4);
	}
	else
	{
		memcpy(&timestamp_ms, &payload[12], 4);
	}

	rs485_module->timestamp = timestamp_ms;
	rs485_module->current = current_a;
	rs485_module->Temperatures[0] = temperature_c;

	/* Per-cell voltage used for reset threshold + cell voltage reporting. */
	const float cell_v = pack_voltage_v / (float)NUMBEROFCELLS;

#if BMS_RECONSTRUCT_CYCLE_FEATURES
	/*
	 * Reconstruct EFC and Q_c on-device (STM32-intended pipeline).
	 *
	 * Q_c: coulomb count since full (0 at full), negative while discharging.
	 * - Integrate current over dt derived from timestamps (ms)
	 * - Reset to 0 when cell voltage reaches U_max (during charge)
	 * - Clamp to <= 0
	 *
	 * EFC: equivalent full cycles as absolute throughput / capacity_ref.
	 */
	static uint8_t init = 0;
	static uint32_t last_ts_ms = 0;
	static double q_c_ah = 0.0;
	static double efc = 0.0;

	if (init == 0U)
	{
		init = 1U;
		last_ts_ms = timestamp_ms;
		q_c_ah = 0.0;
		efc = 0.0;
	}

	int32_t dms = (int32_t)timestamp_ms - (int32_t)last_ts_ms;
	if (dms < 0)
	{
		dms = 0;
	}
	const double dt_s = ((double)dms) / 1000.0;
	last_ts_ms = timestamp_ms;

	q_c_ah += ((double)current_a) * dt_s / 3600.0;

	/* Reset at U_max (full reached) – only during charge to avoid accidental resets. */
	if ((cell_v >= BMS_QC_U_MAX_CELL_V) && (current_a > 0.1f))
	{
		q_c_ah = 0.0;
	}

	if (q_c_ah > 0.0)
	{
		q_c_ah = 0.0;
	}
	if (q_c_ah < (-(double)BMS_QC_CAP_REF_AH))
	{
		q_c_ah = (-(double)BMS_QC_CAP_REF_AH);
	}

	efc += (fabs((double)current_a) * dt_s) / 3600.0 / (double)BMS_EFC_CAP_REF_AH;

	rs485_module->EFC = (float)efc;
	rs485_module->Q_c = (float)q_c_ah;
#else
	/* Host-provided features (legacy). */
	rs485_module->EFC = efc_in;
	rs485_module->Q_c = q_c_in;
#endif

	/* Populate cell voltages with a simple equal-split of pack voltage. */
	float cell_raw_f = (cell_v * 65536.0f) / 5.0f; /* inverse of CONVERT_16_BIT_VALUE_TO_VOLTAGE */
	if (cell_raw_f < 0.0f) cell_raw_f = 0.0f;
	if (cell_raw_f > 65535.0f) cell_raw_f = 65535.0f;
	const uint16_t cell_raw = (uint16_t)cell_raw_f;
	for (uint8_t n = 0; n < NUMBEROFCELLS; n++)
	{
		rs485_module->cellVoltages[n] = cell_raw;
	}

	/* Event-driven: push to CM7 only when new measurements were injected. */
	(void)write_mailbox(rs485_module, sizeof(*rs485_module), BMS_MEASUREMENTS_MAILBOX_NR);

	/*
	 * No ACK for simulation frames.
	 *
	 * Reason: at high replay rates the ACKs accumulate on the UART TX side and
	 * can starve/obscure cmd6 responses on the host, making long HW tests flaky.
	 *
	 * The host side does not require an ACK here; cmd6 polling is the integrity check.
	 */
}


/** @brief init routine for the rs485 bus
 * 	@note this function has to be called before any other function calls from this library
 * 	@param huart UART handle
 * 	@param hmodule pointer to the global module variable
 */

void rs485_command_bms_enter_sleep_mode(uint8_t payloadlen, uint8_t* payload)
{

	BMS_Sleep();

	tx_buffer[0] = 0x80;
	tx_buffer[1] = BMS_ENTER_SLEEP_COMMAND;
	tx_buffer[2] = 0;
	tx_buffer[3] = 0;
	rs485_tx_cplt_flag = 0;
	HAL_UART_Transmit_IT(rs485uart, tx_buffer, 4);
}


void rs485_command_send_state_estimations_as_string(uint8_t payloadlen,uint8_t* payload)
{
	uint8_t txlen = 0;

	tx_buffer[0] = 0x80;
	tx_buffer[1] = GET_STATE_ESTIMATIONS_AS_STRING;

	(void)payloadlen;
	(void)payload;
#if DEVBOARD_NUCLEO || BMS_SIM_INPUT
	/* SOH-only for the current bring-up/testing. */
	txlen = sprintf((char*) &tx_buffer[3], "ts:%lu,SoH:%f\n",
			(unsigned long)rs485_estimations->timestamp,
			rs485_estimations->module_SoH);
#else
	txlen = sprintf((char*) &tx_buffer[3], "SoC:%f,SoH:%f\n",
			rs485_estimations->module_SoC,
			rs485_estimations->module_SoH);
#endif

	tx_buffer[2] = txlen+1;
	tx_buffer[txlen+3] = '\n';

	rs485_tx_cplt_flag = 0;
	HAL_UART_Transmit_IT(rs485uart, tx_buffer, txlen+3);
}
