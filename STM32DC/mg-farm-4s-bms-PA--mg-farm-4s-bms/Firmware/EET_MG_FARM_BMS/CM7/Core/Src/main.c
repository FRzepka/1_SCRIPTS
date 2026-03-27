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
#include "ssd1306.h"
#include "ssd1306_tests.h"
#include <stdio.h>
#include "IPPC_Functions.h"
#include "Shared_DataTypes.h"
#include "Display_Functions.h"
#include "BMS_State_Estimation.h"
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
#define CM4_FAULT_MAGIC (0x434D3446UL) /* 'CM4F' */
#define CM4_ALIVE_ADDR  (SRAM4_BASE_ADDRESS + 0x3F00UL)
static volatile uint32_t *const cm4_alive_magic = (volatile uint32_t *)(CM4_ALIVE_ADDR + 0U);
static volatile uint32_t *const cm4_alive_heartbeat = (volatile uint32_t *)(CM4_ALIVE_ADDR + 4U);

#if DEVBOARD_NUCLEO
/* NUCLEO-H755ZI-Q user LEDs:
 * LD1 (green)  = PB0
 * LD2 (orange) = PE1 (as configured in this project / MX_GPIO_Init)
 * LD3 (red)    = PB14
 */
#define LED_GREEN_GPIO GPIOB
#define LED_GREEN_PIN  GPIO_PIN_0
#define LED_ORANGE_GPIO GPIOE
#define LED_ORANGE_PIN  GPIO_PIN_1
#define LED_RED_GPIO GPIOB
#define LED_RED_PIN  GPIO_PIN_14
#endif

/* Devboard-only activity pulse: toggles LED2 quickly during estimation. */
static uint32_t led_activity_until_ms = 0;
static uint32_t led_activity_last_toggle_ms = 0;


#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

I2C_HandleTypeDef hi2c3;

IWDG_HandleTypeDef hiwdg1;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C3_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_IWDG1_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
Cell_Module_t module;
State_Estimation_t estimations;
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */
/* USER CODE BEGIN Boot_Mode_Sequence_0 */
  int32_t timeout;
/* USER CODE END Boot_Mode_Sequence_0 */

/* USER CODE BEGIN Boot_Mode_Sequence_1 */
  /* TEST: Deaktiviert - CM7 wartet nicht auf CM4 */
  //timeout = 0xFFFFFF;
  //while((__HAL_RCC_GET_FLAG(RCC_FLAG_D2CKRDY) != RESET) && (timeout-- > 0));
  //if ( timeout < 0 )
  //{
  //  Error_Handler();
  //}
/* USER CODE END Boot_Mode_Sequence_1 */
  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();  // TEST: HAL_Init() ohne SystemClock_Config()

  /* USER CODE BEGIN Init */
  #if !BRINGUP_MINIMAL
  SystemClock_Config();
  #endif

  /*
   * Force-start CM4 from Flash Bank2 (0x08100000).
   * Otherwise CM4 may never run (depends on device option-bytes / boot config).
   */
  __HAL_RCC_SYSCFG_CLK_ENABLE();
  HAL_SYSCFG_CM4BootAddConfig(SYSCFG_BOOT_ADDR0, 0x08100000U);
  HAL_SYSCFG_EnableCM4BOOT();
  HAL_RCCEx_EnableBootCore(RCC_BOOT_C2);

  /* Clear stale status (SRAM4 can survive some reset scenarios). */
  *cm4_alive_magic = 0;
  *cm4_alive_heartbeat = 0;

  /* Initialize shared SRAM4 mailboxes before waking CM4. */
  init_mailbox_system();

  #if !BRINGUP_MINIMAL
  /* Wait until CPU2 enters STOP mode (D2 clock off). */
  timeout = 0xFFFFFF;
  while((__HAL_RCC_GET_FLAG(RCC_FLAG_D2CKRDY) != RESET) && (timeout-- > 0));
  if ( timeout < 0 )
  {
    Error_Handler();
  }
  #endif

  /* Wake CM4 (either from STOP mode or just to be safe in bring-up). */
  __HAL_RCC_HSEM_CLK_ENABLE();
  HAL_HSEM_FastTake(HSEM_ID_0);
  HAL_HSEM_Release(HSEM_ID_0, 0);

  #if !BRINGUP_MINIMAL
  /* Wait until CPU2 wakes up from STOP mode (D2 clock on). */
  timeout = 0xFFFFFF;
  while((__HAL_RCC_GET_FLAG(RCC_FLAG_D2CKRDY) == RESET) && (timeout-- > 0));
  if ( timeout < 0 )
  {
    Error_Handler();
  }
  #endif
  /* USER CODE END Init */

  /* Configure the system clock */
  //SystemClock_Config();  // DEAKTIVIERT - PWR/Voltage Scaling crasht (braucht D2 Domain?)
  // Läuft mit Default HSI 64MHz nach Reset
/* USER CODE BEGIN Boot_Mode_Sequence_2 */
/* TEST: Deaktiviert - kein HSEM Release */
/*HW semaphore Clock enable*/
//__HAL_RCC_HSEM_CLK_ENABLE();
/*Take HSEM */
//HAL_HSEM_FastTake(HSEM_ID_0);
/*Release HSEM in order to notify the CPU2(CM4)*/
//HAL_HSEM_Release(HSEM_ID_0,0);
/* wait until CPU2 wakes up from stop mode */
//timeout = 0xFFFFFF;
//while((__HAL_RCC_GET_FLAG(RCC_FLAG_D2CKRDY) == RESET) && (timeout-- > 0));
//if ( timeout < 0 )
//{
//  Error_Handler();
//}
/* USER CODE END Boot_Mode_Sequence_2 */

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
#if !DEVBOARD_NUCLEO
  MX_I2C3_Init();
  MX_USART1_UART_Init();
#endif
  //MX_IWDG1_Init();
  /* USER CODE BEGIN 2 */
#if DEVBOARD_NUCLEO
  /* NUCLEO-H755ZI-Q user LEDs (devboard-only; not part of the original BMS pinout). */
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

  GPIO_InitStruct.Pin = LED_GREEN_PIN | LED_RED_PIN;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
  GPIO_InitStruct.Pin = LED_ORANGE_PIN;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /* CM7-only pattern: green on, orange blinking, red off. */
  HAL_GPIO_WritePin(LED_GREEN_GPIO, LED_GREEN_PIN, GPIO_PIN_SET);
  HAL_GPIO_WritePin(LED_ORANGE_GPIO, LED_ORANGE_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(LED_RED_GPIO, LED_RED_PIN, GPIO_PIN_RESET);
#endif
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHALE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
 #if DEVBOARD_NUCLEO
 	  /*
 	   * LED behavior (devboard):
 	   * - Always: green ON, red heartbeat blink (shows firmware is alive)
 	   * - CM4 not running: orange OFF
 	   * - CM4 running, idle: orange ON
 	   * - CM4 running, computing (during activity window): orange fast blink
 	   *
 	   * Goal: fast blink only when new data was processed (i.e. when PC sends measurements).
 	   */
 	  const uint32_t now = HAL_GetTick();
 	  const uint8_t cm4_running = (*cm4_alive_magic == CM4_ALIVE_MAGIC) ? 1U : 0U;

 	  static uint32_t last_red_toggle_ms = 0;

 	  HAL_GPIO_WritePin(LED_GREEN_GPIO, LED_GREEN_PIN, GPIO_PIN_SET);

 	  if ((now - last_red_toggle_ms) >= 1000U)
 	  {
 		  last_red_toggle_ms = now;
 		  HAL_GPIO_TogglePin(LED_RED_GPIO, LED_RED_PIN);
 	  }

 	  if (cm4_running == 0U)
 	  {
 		  HAL_GPIO_WritePin(LED_ORANGE_GPIO, LED_ORANGE_PIN, GPIO_PIN_RESET);
 	  }
 	  else
 	  {
 		  if (now < led_activity_until_ms)
 		  {
 			  if ((now - led_activity_last_toggle_ms) >= 50U)
 			  {
 				  led_activity_last_toggle_ms = now;
 				  HAL_GPIO_TogglePin(LED_ORANGE_GPIO, LED_ORANGE_PIN);
 			  }
 		  }
 		  else
 		  {
 			  HAL_GPIO_WritePin(LED_ORANGE_GPIO, LED_ORANGE_PIN, GPIO_PIN_SET);
 		  }
 	  }
 #endif

 	  uint16_t rxlen = 0;
	  if (check_mailbox(&rxlen, BMS_MEASUREMENTS_MAILBOX_NR) == MAILBOX_NEWMESSAGE)
	  {
		  if (rxlen >= sizeof(module))
		  {
			  (void)read_mailbox(&module, sizeof(module), BMS_MEASUREMENTS_MAILBOX_NR);
			  estimations = bmsStateEstimationCallback(&module);
			  (void)write_mailbox(&estimations, sizeof(estimations), BMS_STATE_ESTIMATION_MAILBOX_NR);
		  }

#if DEVBOARD_NUCLEO
		  /*
		   * Trigger a short activity blink window only when the SOH model actually produced a new
		   * hourly estimation (otherwise we'd blink at 1 Hz for every received sample).
		   */
		  static uint32_t last_est_ts = 0;
		  if (estimations.timestamp != 0U && estimations.timestamp != last_est_ts)
		  {
			  last_est_ts = estimations.timestamp;
			  led_activity_until_ms = HAL_GetTick() + 250U;
		  }
#endif
	  }

	  (void)cm4_alive_magic;
	  (void)cm4_alive_heartbeat;
	  HAL_Delay(10);
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  // ULTRA-MINIMALE Clock Config - nur HSI, keine PLL
  // HSI = 64MHz interner Oszillator - immer verfügbar
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration - minimal
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);
  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Nur HSI, keine PLL, kein LSI - absolut minimal
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;  // KEINE PLL!
  
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    // Falls das fehlschlägt: Endlosschleife mit sichtbarem Muster
    while(1) {
      GPIOB->BSRR = GPIO_PIN_0;  // LED an
      for(volatile int i=0; i<100000; i++);
      GPIOB->BSRR = (uint32_t)GPIO_PIN_0 << 16;  // LED aus
      for(volatile int i=0; i<100000; i++);
    }
  }

  /** System Clock direkt von HSI (64MHz)
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
                              // D1/D3PCLK1 entfernt - könnte D2 Domain brauchen!
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;  // Direkt HSI, nicht PLL!
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;  // 32MHz
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    // Falls das fehlschlägt: Endlosschleife mit anderem Muster
    while(1) {
      GPIOB->BSRR = GPIO_PIN_0;
      for(volatile int i=0; i<50000; i++);
      GPIOB->BSRR = (uint32_t)GPIO_PIN_0 << 16;
      for(volatile int i=0; i<500000; i++);
    }
  }
}

/**
  * @brief I2C3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C3_Init(void)
{

  /* USER CODE BEGIN I2C3_Init 0 */

  /* USER CODE END I2C3_Init 0 */

  /* USER CODE BEGIN I2C3_Init 1 */

  /* USER CODE END I2C3_Init 1 */
  hi2c3.Instance = I2C3;
  hi2c3.Init.Timing = 0x00707CBB;
  hi2c3.Init.OwnAddress1 = 0;
  hi2c3.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c3.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c3.Init.OwnAddress2 = 0;
  hi2c3.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c3.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c3.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c3, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c3, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C3_Init 2 */

  /* USER CODE END I2C3_Init 2 */

}

/**
  * @brief IWDG1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_IWDG1_Init(void)
{

  /* USER CODE BEGIN IWDG1_Init 0 */

  /* USER CODE END IWDG1_Init 0 */

  /* USER CODE BEGIN IWDG1_Init 1 */

  /* USER CODE END IWDG1_Init 1 */
  hiwdg1.Instance = IWDG1;
  hiwdg1.Init.Prescaler = IWDG_PRESCALER_256;
  hiwdg1.Init.Window = 4095;
  hiwdg1.Init.Reload = 4095;
  if (HAL_IWDG_Init(&hiwdg1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN IWDG1_Init 2 */

  /* USER CODE END IWDG1_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

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
  __HAL_RCC_GPIOB_CLK_ENABLE();
  //__HAL_RCC_GPIOC_CLK_ENABLE();
  //__HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */
  // Configure PB0 (LD1 green) + PB14 (LD3 red) and PE1 (LD2 yellow/orange) as outputs for CM7 test
  GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_14;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_14, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = GPIO_PIN_1;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1, GPIO_PIN_RESET);
  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

//Redefine Putchar for debugging Output

PUTCHAR_PROTOTYPE
{
	HAL_UART_Transmit(&huart1, (uint8_t *) &ch, 1, 1000);
	return ch;
}

int _write(int fd, char *ptr, int len) {

return HAL_UART_Transmit(&huart1, (uint8_t*)ptr, len, 1000);

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
