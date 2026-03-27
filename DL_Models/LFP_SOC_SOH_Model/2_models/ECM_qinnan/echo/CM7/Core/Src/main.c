/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
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
#include "memorymap.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "string.h"
#include <stdlib.h>
#include "ECM_parameter.h"
#include "interpolation.h"
#include "arm_math.h"
#include "battery_ekf.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#ifndef HSEM_ID_0
#define HSEM_ID_0 (0U) /* HW semaphore 0*/
#endif

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
extern UART_HandleTypeDef huart3;

#define BUFFER_SIZE 64

uint8_t rx_buffer[BUFFER_SIZE];
uint8_t tx_buffer[BUFFER_SIZE];
uint8_t rx_index = 0;
uint8_t rx_byte;

float I,Ut;
volatile uint8_t rx_ready = 0;/*中断完成标志*/

float x[3]={1.0f,0.0f,0.0f}; // initial Wert für SoC
float P[3][3]={{1e-4,0,0},{0,1e-3,0},{0,0,1e-3}};
float x_k1[3];
float y_k1;
float P_k1[3][3];
float SoH=1.0f;
float I_prev=0.0f;
float SoC_init = 0.0f;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int _write(int file, char *ptr, int len){
HAL_UART_Transmit(&huart3, (uint8_t *)ptr, len, HAL_MAX_DELAY);
return len;
}

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

/* USER CODE END Boot_Mode_Sequence_0 */

/* USER CODE BEGIN Boot_Mode_Sequence_1 */

/* USER CODE END Boot_Mode_Sequence_1 */
  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();
/* USER CODE BEGIN Boot_Mode_Sequence_2 */
/* When system initialization is finished, Cortex-M7 will release Cortex-M4 by means of
HSEM notification */
/*HW semaphore Clock enable*/
__HAL_RCC_HSEM_CLK_ENABLE();
/*Take HSEM */
HAL_HSEM_FastTake(HSEM_ID_0);
/*Release HSEM in order to notify the CPU2(CM4)*/
HAL_HSEM_Release(HSEM_ID_0,0);

/* USER CODE END Boot_Mode_Sequence_2 */

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART3_UART_Init();
  /* USER CODE BEGIN 2 */
  HAL_UART_Receive_IT(&huart3, &rx_byte, 1);

  /* USER CODE END 2 */

  /* Initialize leds */
  BSP_LED_Init(LED_GREEN);
  BSP_LED_Init(LED_YELLOW);
  BSP_LED_Init(LED_RED);

  /* Initialize USER push-button, will be used to trigger an interrupt each time it's pressed.*/
  BSP_PB_Init(BUTTON_USER, BUTTON_MODE_EXTI);

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

  BSP_LED_Toggle(LED_RED);
  HAL_Delay(2000);
  BSP_LED_Toggle(LED_GREEN);
  HAL_Delay(2000);
  BSP_LED_Toggle(LED_RED);
  HAL_Delay(2000);
  BSP_LED_Toggle(LED_GREEN);


  while (1)
  {
	  if(rx_ready){

		  	  //if (SoC_init_flag){
		  	  	  SoC_init = interpolate1D(&charge_OCV, Ut);
		  	  //}



	  		  rx_ready=0;

	  		  battery_EKF(x_k1,&y_k1,I,Ut,I_prev,SoH,P_k1,x,P);

	  		  //update
	  		  I_prev=I;
	  		  memcpy(x, x_k1, sizeof(x));
	  		  memcpy(P, P_k1, sizeof(P));

	  		  snprintf((char *)tx_buffer, BUFFER_SIZE, "%.3f,%.3f\r\n", x_k1[0],y_k1);
	  		  HAL_UART_Transmit(&huart3, tx_buffer, strlen((char *)tx_buffer), HAL_MAX_DELAY);
	  		  BSP_LED_Toggle(LED_YELLOW);

	  	  }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_DIRECT_SMPS_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
/*void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if (huart->Instance == USART3){

		if(rx_index<RX_BUFFER_SIZE){

			rx_buffer[rx_index++]=rx_byte;

			if(rx_byte=='\n'){

				rx_buffer[rx_index-1]='\0';
				int value =atoi((char*)rx_buffer);
				int tx_data = value*value;
				snprintf((char *)tx_buffer, RX_BUFFER_SIZE, "%d\n", tx_data);
				HAL_UART_Transmit(&huart3, tx_buffer, strlen((char *)tx_buffer), HAL_MAX_DELAY);
				BSP_LED_Toggle(LED_YELLOW);
				rx_index = 0;
			}
		}
		HAL_UART_Receive_IT(&huart3, &rx_byte, 1);
	}
}*/
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if (huart->Instance == USART3){

			if(rx_index < BUFFER_SIZE){

				rx_buffer[rx_index++]=rx_byte;

				if(rx_byte=='\n'){//end of the receive

					rx_buffer[rx_index-1]='\0';//delete the line break

					//printf("RX raw: %s\r\n", rx_buffer);
					int parsed = sscanf((char*)rx_buffer, "%f,%f,%f", &I, &Ut, &SoH);
					//printf("Parsed=%d, I=%.3f, U=%.3f, SoH=%.3f\r\n", parsed, I, Ut, SoH);

					// Analyzing two floating-point numbers: current and voltage;
					if (parsed==3)
					{
						rx_ready=1;//succeed analyze
						 BSP_LED_Toggle(LED_GREEN);
					}
					else
					{
						printf("Parse error!\r\n");
					}

					rx_index = 0;
				}
		}
		HAL_UART_Receive_IT(&huart3, &rx_byte, 1);
	}
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
