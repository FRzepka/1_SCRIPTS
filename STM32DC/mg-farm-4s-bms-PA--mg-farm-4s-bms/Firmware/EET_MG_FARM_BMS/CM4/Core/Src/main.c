/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "TLE_Abstraction.h"
#include "IPPC_Functions.h"
#include "Shared_DataTypes.h"
#include "RS485_Communication.h"
#include "BMS_Functions.h"
#include "Build_Config.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#ifndef HSEM_ID_0
#define HSEM_ID_0 (0U) /* HW semaphore 0*/
#endif

/* Dual-core test status: CM4 writes this magic value once it is running. */
#define CM4_ALIVE_MAGIC (0x434D3441UL) /* 'CM4A' */
#define CM4_ALIVE_ADDR  (SRAM4_BASE_ADDRESS + 0x3F00UL)
static volatile uint32_t *const cm4_alive_magic = (volatile uint32_t *)CM4_ALIVE_ADDR;
static volatile uint32_t *const cm4_alive_heartbeat = (volatile uint32_t *)(CM4_ALIVE_ADDR + 4U);
/* Simple stage marker for bring-up diagnostics (read via SWD). */
static volatile uint32_t *const cm4_stage = (volatile uint32_t *)(CM4_ALIVE_ADDR + 0x20U);

#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

IWDG_HandleTypeDef hiwdg2;

UART_HandleTypeDef huart4;
UART_HandleTypeDef huart2;
UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */
Cell_Module_t module;
State_Estimation_t estimations;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_UART4_Init(void);
static void MX_IWDG2_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  // FRÜHE GPIO Init GANZ FRÜH (vor allem anderen!)
  *cm4_stage = 1;
  *cm4_alive_magic = CM4_ALIVE_MAGIC;
  *cm4_alive_heartbeat = 0;

  /* Avoid devboard-only direct GPIO register writes here; keep CM4 generic. */
  /* USER CODE END 1 */

/* USER CODE BEGIN Boot_Mode_Sequence_1 */
  // DEAKTIVIERT - Test ob CM4 überhaupt startet
  #if !BRINGUP_MINIMAL
  /*HW semaphore Clock enable*/
  __HAL_RCC_HSEM_CLK_ENABLE();
  /* Activate HSEM notification for Cortex-M4*/
  HAL_HSEM_ActivateNotification(__HAL_HSEM_SEMID_TO_MASK(HSEM_ID_0));
  /*
  Domain D2 goes to STOP mode (Cortex-M4 in deep-sleep) waiting for Cortex-M7 to
  perform system initialization (system clock config, external memory configuration.. )
  */
  HAL_PWREx_ClearPendingEvent();
  HAL_DBGMCU_EnableDBGStopMode(); // Debug in STOP mode
  HAL_PWREx_EnterSTOPMode(PWR_MAINREGULATOR_ON, PWR_STOPENTRY_WFE, PWR_D2_DOMAIN);
  /* Clear HSEM flag */
  __HAL_HSEM_CLEAR_FLAG(__HAL_HSEM_SEMID_TO_MASK(HSEM_ID_0));
  #endif /* !BRINGUP_MINIMAL */

/* USER CODE END Boot_Mode_Sequence_1 */
  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  //HAL_Init();  // TEST: komplett ohne HAL, um sicher zu sehen, ob CM4 läuft

  /* USER CODE BEGIN Init */
  HAL_Init();
  *cm4_stage = 2;

  /* USER CODE END Init */

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  //MX_GPIO_Init();  // TEST: GPIO wird oben direkt per Register gesetzt
  //MX_USART2_UART_Init();  // Deaktiviert für Test
  //MX_USART3_UART_Init();  // Deaktiviert für Test
  //MX_UART4_Init();  // Deaktiviert für Test
  //MX_IWDG2_Init();  // Watchdog temporär deaktiviert für Test
  /* USER CODE BEGIN 2 */
  MX_GPIO_Init();
  *cm4_stage = 3;
  MX_USART3_UART_Init();
  *cm4_stage = 4;
#if !BMS_SIM_INPUT
  MX_UART4_Init();
#endif

  memset(&module, 0, sizeof(module));
  memset(&estimations, 0, sizeof(estimations));

  /* BMS init (real TLE path or simulation, depending on BMS_SIM_INPUT). */
  BMS_Init(&huart4, &module, NUMBEROFCELLS, NUMBEROFTEMPS);
  *cm4_stage = 5;

  /* PC/RS485 comms on USART3. */
  rs485_communication_init(&huart3, &module, &estimations);
  *cm4_stage = 6;
  
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHALE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  // CM4: LED1 (PB0) blinkt (ohne HAL_Delay, unabhängig von SysTick/Clock-Setup)
	  *cm4_stage = 10;
	  *cm4_alive_magic = CM4_ALIVE_MAGIC;
	  (*cm4_alive_heartbeat)++;

	  *cm4_stage = 11;
	  rs485_communication_loop();
	  *cm4_stage = 12;
	  BMS_loop();

	  *cm4_stage = 13;
	  /*
	   * Simulation mode: only forward measurements to CM7 when the PC injected new data.
	   * Otherwise CM7 would constantly recompute and the activity LED would look "always on".
	   */
#if !BMS_SIM_INPUT
	  (void)write_mailbox(&module, sizeof(module), BMS_MEASUREMENTS_MAILBOX_NR);
#endif

	  *cm4_stage = 14;
	  uint16_t rxlen = 0;
	  if (check_mailbox(&rxlen, BMS_STATE_ESTIMATION_MAILBOX_NR) == MAILBOX_NEWMESSAGE)
	  {
		  if (rxlen >= sizeof(estimations))
		  {
			  (void)read_mailbox(&estimations, sizeof(estimations), BMS_STATE_ESTIMATION_MAILBOX_NR);
		  }
	  }

	  *cm4_stage = 15;
#if BMS_SIM_INPUT
	  /* In simulation mode the PC can stream at high rate; keep the loop responsive. */
	  HAL_Delay(1);
#else
	  HAL_Delay(20);
#endif
  }
  /* USER CODE END 3 */
}

/**
  * @brief IWDG2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_IWDG2_Init(void)
{

  /* USER CODE BEGIN IWDG2_Init 0 */

  /* USER CODE END IWDG2_Init 0 */

  /* USER CODE BEGIN IWDG2_Init 1 */

  /* USER CODE END IWDG2_Init 1 */
  hiwdg2.Instance = IWDG2;
  hiwdg2.Init.Prescaler = IWDG_PRESCALER_256;
  hiwdg2.Init.Window = 4095;
  hiwdg2.Init.Reload = 4095;
  if (HAL_IWDG_Init(&hiwdg2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN IWDG2_Init 2 */

  /* USER CODE END IWDG2_Init 2 */

}

/**
  * @brief UART4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_UART4_Init(void)
{

  /* USER CODE BEGIN UART4_Init 0 */

  /* USER CODE END UART4_Init 0 */

  /* USER CODE BEGIN UART4_Init 1 */

  /* USER CODE END UART4_Init 1 */
  huart4.Instance = UART4;
  huart4.Init.BaudRate = 1000000;
  huart4.Init.WordLength = UART_WORDLENGTH_8B;
  huart4.Init.StopBits = UART_STOPBITS_1;
  huart4.Init.Parity = UART_PARITY_NONE;
  huart4.Init.Mode = UART_MODE_RX;
  huart4.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart4.Init.OverSampling = UART_OVERSAMPLING_16;
  huart4.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart4.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart4.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_MSBFIRST_INIT;
  huart4.AdvancedInit.MSBFirst = UART_ADVFEATURE_MSBFIRST_ENABLE;
  if (HAL_UART_Init(&huart4) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart4, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart4, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN UART4_Init 2 */

  /* USER CODE END UART4_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 1000000;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_MSBFIRST_INIT;
  huart2.AdvancedInit.MSBFirst = UART_ADVFEATURE_MSBFIRST_ENABLE;
  if (HAL_HalfDuplex_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart2, UART_TXFIFO_THRESHOLD_8_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart2, UART_RXFIFO_THRESHOLD_8_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_EnableFifoMode(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
#if DEVBOARD_NUCLEO
  /* Devboard: use plain UART over ST-LINK VCP (no external RS485 transceiver). */
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
#else
  if (HAL_RS485Ex_Init(&huart3, UART_DE_POLARITY_HIGH, 0, 0) != HAL_OK)
  {
    Error_Handler();
  }
#endif
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */
  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(BAT_Discharge_GPIO_Port, BAT_Discharge_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(BAT_Charge_GPIO_Port, BAT_Charge_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : BAT_Discharge_Pin */
  GPIO_InitStruct.Pin = BAT_Discharge_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(BAT_Discharge_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : BAT_Charge_Pin */
  GPIO_InitStruct.Pin = BAT_Charge_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(BAT_Charge_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : TLE_Error_Pin */
  GPIO_InitStruct.Pin = TLE_Error_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(TLE_Error_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */
  // Configure PB0 (LED1) as output for CM4 test
  __HAL_RCC_GPIOB_CLK_ENABLE();
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);
  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

PUTCHAR_PROTOTYPE
{
	HAL_UART_Transmit(&huart3, (uint8_t *) &ch, 1, 1000);
	return ch;
}

int _write(int fd, char *ptr, int len) {

return HAL_UART_Transmit(&huart3, (uint8_t*)ptr, len, 1000);

}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
