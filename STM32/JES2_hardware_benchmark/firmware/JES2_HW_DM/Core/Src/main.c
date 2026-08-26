#include "main.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FIRMWARE_GIT_SHA
#define FIRMWARE_GIT_SHA "unknown"
#endif

#define CPU_CLOCK_HZ 480000000UL
#define RX_BUFFER_SIZE 256U

#define LED_IDLE_PORT GPIOB
#define LED_IDLE_PIN GPIO_PIN_0
#define LED_ERROR_PORT GPIOB
#define LED_ERROR_PIN GPIO_PIN_14
#define LED_AUX_PORT GPIOE
#define LED_AUX_PIN GPIO_PIN_1

typedef struct {
  float q_m_ah;
  float cv_time_s;
  float soc;
} dm_state_t;

static dm_state_t dm_state;
static uint32_t activity_until_ms;
static uint32_t error_until_ms;
static uint32_t led_last_ms;
static uint8_t led_state;

static void SystemClock_Config(void);
static void GPIO_Init(void);
static void UART3_Init(void);
static void UART3_Send(const char *text);
static int UART3_ReadChar(char *value);
static void DWT_Init(void);
static void LED_Startup(void);
static void LED_Service(void);
static void DM_Reset(void);
static float DM_Step(float current_a, float voltage_v, float dt_s);
static void Process_Command(char *line);

static const char ready_line[] =
    "READY,JES2_HW_V1,DM," FIRMWARE_GIT_SHA ",480000000\r\n";

int main(void)
{
  SCB_EnableICache();
  SCB_EnableDCache();
  HAL_Init();
  SystemClock_Config();
  GPIO_Init();
  UART3_Init();
  DWT_Init();
  DM_Reset();
  LED_Startup();
  UART3_Send(ready_line);

  char line[RX_BUFFER_SIZE];
  size_t length = 0U;

  while (1) {
    char value;
    while (UART3_ReadChar(&value)) {
      if ((value == '\r') || (value == '\n')) {
        if (length > 0U) {
          line[length] = '\0';
          Process_Command(line);
          length = 0U;
        }
      } else if (length < (RX_BUFFER_SIZE - 1U)) {
        line[length++] = value;
      } else {
        length = 0U;
        error_until_ms = HAL_GetTick() + 1000U;
        UART3_Send("ERROR,0,LINE_TOO_LONG\r\n");
      }
    }
    LED_Service();
  }
}

static void Process_Command(char *line)
{
  if (strcmp(line, "HELLO") == 0) {
    UART3_Send(ready_line);
    return;
  }

  if (strcmp(line, "RESET") == 0) {
    DM_Reset();
    UART3_Send("ACK,RESET\r\n");
    return;
  }

  char *fields[10];
  size_t field_count = 0U;
  char *token = strtok(line, ",");
  while ((token != NULL) && (field_count < 10U)) {
    fields[field_count++] = token;
    token = strtok(NULL, ",");
  }

  if ((field_count != 10U) || (token != NULL) || (strcmp(fields[0], "STEP") != 0)) {
    error_until_ms = HAL_GetTick() + 1000U;
    UART3_Send("ERROR,0,MALFORMED_COMMAND\r\n");
    return;
  }

  errno = 0;
  char *end = NULL;
  unsigned long sample_id = strtoul(fields[1], &end, 10);
  if ((errno != 0) || (end == fields[1]) || (*end != '\0')) {
    error_until_ms = HAL_GetTick() + 1000U;
    UART3_Send("ERROR,0,BAD_SAMPLE_ID\r\n");
    return;
  }

  float values[8];
  for (size_t index = 0U; index < 8U; ++index) {
    errno = 0;
    end = NULL;
    values[index] = strtof(fields[index + 2U], &end);
    if ((errno != 0) || (end == fields[index + 2U]) || (*end != '\0')) {
      char response[64];
      snprintf(response, sizeof(response), "ERROR,%lu,BAD_NUMBER\r\n", sample_id);
      error_until_ms = HAL_GetTick() + 1000U;
      UART3_Send(response);
      return;
    }
  }

  activity_until_ms = HAL_GetTick() + 750U;
  uint32_t start_cycles = DWT->CYCCNT;
  float soc = DM_Step(values[1], values[0], values[7]);
  uint32_t cycles = DWT->CYCCNT - start_cycles;

  char response[96];
  snprintf(response, sizeof(response), "RESULT,%lu,DM,%.9g,%lu,OK\r\n",
           sample_id, (double)soc, (unsigned long)cycles);
  UART3_Send(response);
}

static void DM_Reset(void)
{
  dm_state.q_m_ah = 0.0f;
  dm_state.cv_time_s = 0.0f;
  dm_state.soc = 1.0f;
}

static float DM_Step(float current_a, float voltage_v, float dt_s)
{
  const float capacity_ah = 1.8f;
  const float cv_threshold_v = 3.63f;
  const float cv_reset_time_s = 300.0f;

  if (dt_s < 0.0f) {
    dt_s = 0.0f;
  }

  if (voltage_v >= cv_threshold_v) {
    dm_state.cv_time_s += dt_s;
  } else {
    dm_state.cv_time_s = 0.0f;
  }

  if (dm_state.cv_time_s >= cv_reset_time_s) {
    dm_state.q_m_ah = 0.0f;
  } else {
    dm_state.q_m_ah += current_a * dt_s / 3600.0f;
  }

  dm_state.soc = 1.0f + dm_state.q_m_ah / capacity_ah;
  if (dm_state.soc < 0.0f) {
    dm_state.soc = 0.0f;
  } else if (dm_state.soc > 1.0f) {
    dm_state.soc = 1.0f;
  }
  return dm_state.soc;
}

static void LED_Service(void)
{
  uint32_t now = HAL_GetTick();
  uint32_t interval_ms = ((int32_t)(activity_until_ms - now) > 0) ? 50U : 250U;

  if ((now - led_last_ms) >= interval_ms) {
    led_last_ms = now;
    led_state ^= 1U;
    HAL_GPIO_WritePin(LED_IDLE_PORT, LED_IDLE_PIN,
                      led_state ? GPIO_PIN_SET : GPIO_PIN_RESET);
  }

  HAL_GPIO_WritePin(LED_ERROR_PORT, LED_ERROR_PIN,
                    ((int32_t)(error_until_ms - now) > 0) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static void LED_Startup(void)
{
  for (uint32_t index = 0U; index < 3U; ++index) {
    HAL_GPIO_WritePin(LED_IDLE_PORT, LED_IDLE_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(LED_AUX_PORT, LED_AUX_PIN, GPIO_PIN_SET);
    HAL_Delay(100U);
    HAL_GPIO_WritePin(LED_IDLE_PORT, LED_IDLE_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LED_AUX_PORT, LED_AUX_PIN, GPIO_PIN_RESET);
    HAL_Delay(100U);
  }
  led_last_ms = HAL_GetTick();
}

static void GPIO_Init(void)
{
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  GPIO_InitTypeDef gpio = {0};
  gpio.Pin = LED_IDLE_PIN | LED_ERROR_PIN;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &gpio);

  gpio.Pin = LED_AUX_PIN;
  HAL_GPIO_Init(GPIOE, &gpio);

  gpio.Pin = GPIO_PIN_8 | GPIO_PIN_9;
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Alternate = GPIO_AF7_USART3;
  HAL_GPIO_Init(GPIOD, &gpio);

  HAL_GPIO_WritePin(GPIOB, LED_IDLE_PIN | LED_ERROR_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOE, LED_AUX_PIN, GPIO_PIN_RESET);
}

static void UART3_Init(void)
{
  __HAL_RCC_USART3_CLK_ENABLE();
  CLEAR_BIT(USART3->CR1, USART_CR1_UE);
  USART3->CR1 = 0U;
  USART3->CR2 = 0U;
  USART3->CR3 = 0U;
  uint32_t pclk = HAL_RCC_GetPCLK1Freq();
  USART3->BRR = (pclk + 57600U) / 115200U;
  SET_BIT(USART3->CR1, USART_CR1_TE | USART_CR1_RE | USART_CR1_UE);
  USART3->ICR = 0xFFFFFFFFU;
}

static void UART3_Send(const char *text)
{
  while (*text != '\0') {
    while ((USART3->ISR & USART_ISR_TXE_TXFNF) == 0U) {
    }
    USART3->TDR = (uint8_t)*text++;
  }
  while ((USART3->ISR & USART_ISR_TC) == 0U) {
  }
}

static int UART3_ReadChar(char *value)
{
  uint32_t status = USART3->ISR;
  if ((status & (USART_ISR_ORE | USART_ISR_NE | USART_ISR_FE | USART_ISR_PE)) != 0U) {
    USART3->ICR = USART_ICR_ORECF | USART_ICR_NECF | USART_ICR_FECF | USART_ICR_PECF;
  }
  if ((status & USART_ISR_RXNE_RXFNE) == 0U) {
    return 0;
  }
  *value = (char)(USART3->RDR & 0xFFU);
  return 1;
}

static void DWT_Init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0U;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void SystemClock_Config(void)
{
  RCC_OscInitTypeDef oscillator = {0};
  RCC_ClkInitTypeDef clock = {0};

  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);
  while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {
  }

  oscillator.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  oscillator.HSEState = RCC_HSE_BYPASS;
  oscillator.PLL.PLLState = RCC_PLL_ON;
  oscillator.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  oscillator.PLL.PLLM = 4U;
  oscillator.PLL.PLLN = 480U;
  oscillator.PLL.PLLP = 2U;
  oscillator.PLL.PLLQ = 20U;
  oscillator.PLL.PLLR = 2U;
  oscillator.PLL.PLLRGE = RCC_PLL1VCIRANGE_1;
  oscillator.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  oscillator.PLL.PLLFRACN = 0U;
  if (HAL_RCC_OscConfig(&oscillator) != HAL_OK) {
    Error_Handler();
  }

  clock.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                    RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2 |
                    RCC_CLOCKTYPE_D3PCLK1 | RCC_CLOCKTYPE_D1PCLK1;
  clock.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  clock.SYSCLKDivider = RCC_SYSCLK_DIV1;
  clock.AHBCLKDivider = RCC_HCLK_DIV2;
  clock.APB3CLKDivider = RCC_APB3_DIV2;
  clock.APB1CLKDivider = RCC_APB1_DIV2;
  clock.APB2CLKDivider = RCC_APB2_DIV2;
  clock.APB4CLKDivider = RCC_APB4_DIV2;
  if (HAL_RCC_ClockConfig(&clock, FLASH_LATENCY_4) != HAL_OK) {
    Error_Handler();
  }
}

void Error_Handler(void)
{
  __disable_irq();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  GPIOB->MODER = (GPIOB->MODER & ~(3UL << 28U)) | (1UL << 28U);
  while (1) {
    GPIOB->ODR ^= GPIO_PIN_14;
    for (volatile uint32_t delay = 0U; delay < 5000000U; ++delay) {
      __NOP();
    }
  }
}

