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
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "ai_wrapper.h"
#include "lstm_model.h"
#include <stdio.h>  // For sprintf

// Linker symbols for RAM measurement
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;
extern uint32_t _estack;

static void print_memory_info(void);
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// === Minimal USART3 (PD8 TX, PD9 RX) register helpers for quick bring-up ===
// Use CMSIS device header definitions to avoid redefinition warnings
#include "stm32h753xx.h"

#define RCC_APB1LENR        (RCC->APB1LENR)
#define USART3_CR1          (USART3->CR1)
#define USART3_CR2          (USART3->CR2)
#define USART3_CR3          (USART3->CR3)
#define USART3_BRR          (USART3->BRR)
#define USART3_ISR          (USART3->ISR)
#define USART3_ICR          (USART3->ICR)
#define USART3_RDR          (USART3->RDR)
#define USART3_TDR          (USART3->TDR)

static void UART3_Init_Quick(void)
{
  // Enable peripheral clock
  RCC_APB1LENR |= RCC_APB1LENR_USART3EN;

  // Disable USART while configuring
  USART3_CR1 &= ~(1UL << 0); // UE = 0

  // Baud rate from current APB1 clock
  uint32_t pclk = HAL_RCC_GetPCLK1Freq();
  uint32_t baud = 115200U;
  uint32_t brr = (pclk + (baud/2U)) / baud; // round to nearest
  if (brr == 0U) brr = 1U;
  USART3_BRR = brr;

  // 8N1, oversampling by 16 (reset defaults), enable TX and RX, then UE
  USART3_CR1 = (1UL << 3) | (1UL << 2); // TE | RE
  USART3_CR2 = 0;
  USART3_CR3 = 0;
  USART3_CR1 |= (1UL << 0); // UE

  // Clear any lingering errors
  USART3_ICR = 0xFFFFFFFFU;
}

static void UART3_SendString(const char *s)
{
  while (*s) {
    while ((USART3_ISR & (1UL << 7)) == 0) { /* wait TXE_TXFNF */ }
    USART3_TDR = (uint8_t)(*s++);
  }
  // Wait for TC
  while ((USART3_ISR & (1UL << 6)) == 0) { /* wait TC */ }
}

static inline void UART3_SendChar(char c)
{
  while ((USART3_ISR & (1UL << 7)) == 0) { /* wait TXFNF */ }
  USART3_TDR = (uint8_t)c;
}

// RX helpers
static inline int UART3_DataAvailable(void)
{
  return (USART3_ISR & (1UL << 5)) != 0; // RXFNE
}

static inline char UART3_ReadChar(void)
{
  // Clear error flags if any
  if (USART3_ISR & ((1UL<<0)|(1UL<<1)|(1UL<<2)|(1UL<<3))) {
    USART3_ICR = (1UL<<0)|(1UL<<1)|(1UL<<2)|(1UL<<3);
  }
  if (UART3_DataAvailable()) {
    return (char)(USART3_RDR & 0xFF);
  }
  return 0;
}

// Simple float parser for first token
static int parse_float_simple(const char **ps, float *out)
{
  const char *s = *ps; int sign = 1; int seen = 0;
  while (*s==' '||*s=='\t') s++;
  if (*s=='+') s++; else if (*s=='-') { sign=-1; s++; }
  uint32_t ip=0, fp=0, scale=1;
  while (*s>='0'&&*s<='9') { ip = ip*10 + (uint32_t)(*s-'0'); s++; seen=1; }
  if (*s=='.') { s++; while (*s>='0'&&*s<='9') { if (scale<1000000U){ fp = fp*10 + (uint32_t)(*s-'0'); scale*=10; } s++; seen=1; } }
  if (!seen) return 0;
  *out = (float)sign * ((float)ip + (scale>1 ? ((float)fp/(float)scale) : 0.0f));
  *ps = s;
  return 1;
}

static void float_to_str_3(float v, char *buf, int n)
{
  if (n<8) { if (n>0) buf[0]='\0'; return; }
  int pos=0; if (v<0){ buf[pos++]='-'; v=-v; }
  int i = (int)v; int frac = (int)((v - (float)i)*1000.0f + 0.5f); if (frac>=1000){ i+=1; frac=0; }
  // integer
  char tmp[12]; int t=0; if (i==0){ tmp[t++]='0'; } else { while (i>0 && t<11){ tmp[t++] = '0'+(i%10); i/=10; } }
  for (int k=t-1;k>=0;k--) buf[pos++]=tmp[k];
  buf[pos++]='.'; buf[pos++]='0'+(frac/100); buf[pos++]='0'+((frac/10)%10); buf[pos++]='0'+(frac%10); buf[pos]='\0';
}

static void float_to_str_6(float v, char *buf, int n)
{
  if (n<12) { if (n>0) buf[0]='\0'; return; }
  int pos=0; if (v<0){ buf[pos++]='-'; v=-v; }
  int i = (int)v; int frac = (int)((v - (float)i)*1000000.0f + 0.5f); if (frac>=1000000){ i+=1; frac=0; }
  // integer
  char tmp[12]; int t=0; if (i==0){ tmp[t++]='0'; } else { while (i>0 && t<11){ tmp[t++] = '0'+(i%10); i/=10; } }
  for (int k=t-1;k>=0;k--) buf[pos++]=tmp[k];
  buf[pos++]='.';
  // 6 decimal places
  buf[pos++]='0'+(frac/100000); 
  buf[pos++]='0'+((frac/10000)%10); 
  buf[pos++]='0'+((frac/1000)%10);
  buf[pos++]='0'+((frac/100)%10);
  buf[pos++]='0'+((frac/10)%10);
  buf[pos++]='0'+(frac%10);
  buf[pos]='\0';
}

static void print_memory_info(void)
{
  uint32_t data_size = (uint32_t)&_edata - (uint32_t)&_sdata;
  uint32_t bss_size = (uint32_t)&_ebss - (uint32_t)&_sbss;
  uint32_t static_ram = data_size + bss_size;
  uint32_t stack_top = (uint32_t)&_estack;
  uint32_t current_sp;
  __asm volatile ("mov %0, sp" : "=r" (current_sp));
  
  char buf[128];
  sprintf(buf, "RAM_MEASURE: Static=%lu (Data=%lu, BSS=%lu), StackTop=%08lX, SP=%08lX\r\n", 
          static_ram, data_size, bss_size, stack_top, current_sp);
  UART3_SendString(buf);
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  // PRE-HAL DEBUG: Minimal LED blink using registers to prove we reached main()
  // Enable GPIOB (LD1 PB0, LD3 PB14) and GPIOE (LD2 PE1) clocks
  *((volatile uint32_t *)0x580244E0U) |= (1UL << 1) | (1UL << 4); // RCC AHB4ENR GPIOBEN | GPIOEEN
  // Configure PB0 and PB14 as outputs
  volatile uint32_t *GPIOB_MODER = (uint32_t *)0x58020400U;
  *GPIOB_MODER &= ~((3UL << (0*2)) | (3UL << (14*2)));
  *GPIOB_MODER |=  ((1UL << (0*2)) | (1UL << (14*2)));
  // Configure PE1 as output
  volatile uint32_t *GPIOE_MODER = (uint32_t *)0x58021000U;
  *GPIOE_MODER &= ~(3UL << (1*2));
  *GPIOE_MODER |=  (1UL << (1*2));
  // Toggle all three LEDs once (quick)
  volatile uint32_t *GPIOB_BSRR = (uint32_t *)0x58020418U;
  volatile uint32_t *GPIOE_BSRR = (uint32_t *)0x58021018U;
  *GPIOB_BSRR = (1UL << 0) | (1UL << 14);   // PB0+PB14 ON
  *GPIOE_BSRR = (1UL << 1);                 // PE1 ON
  for (volatile uint32_t i = 0; i < 2000000U; ++i) __asm volatile("nop");
  *GPIOB_BSRR = (1UL << (0+16)) | (1UL << (14+16)); // PB0+PB14 OFF
  *GPIOE_BSRR = (1UL << (1+16));                    // PE1 OFF
  for (volatile uint32_t i = 0; i < 2000000U; ++i) __asm volatile("nop");

  /* USER CODE END 1 */

  /* Enable the CPU Cache */

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  /* USER CODE BEGIN 2 */
  // Startup: Blink all LEDs 3x fast
  for (int i = 0; i < 3; i++) {
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_14, GPIO_PIN_SET);   // Green + Red ON
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1, GPIO_PIN_SET);                 // Blue ON
    HAL_Delay(200);
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_14, GPIO_PIN_RESET); // All OFF
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1, GPIO_PIN_RESET);
    HAL_Delay(200);
  }

  // Enable DWT for cycle counting
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  // Init USART3 quickly and announce
  UART3_Init_Quick();
  UART3_SendString("READY\r\n");
  
  // Print Memory Info (RAM Measurement)
  print_memory_info();

  // Initialize AI network
  if (ai_wrapper_init() == 0) {
    UART3_SendString("AI:OK\r\n");
  } else {
    UART3_SendString("AI:ERR\r\n");
  }
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    static uint32_t last_inference_tick = 0;
    // UART RX polling + simple line buffer
    static char rxbuf[128]; static uint16_t rxi=0; 
    while (UART3_DataAvailable()) {
      char c = UART3_ReadChar(); if (!c) break;
      // NO ECHO for automated testing (Python script needs clean responses)
      // if (c >= 32 && c <= 126) {
      //   UART3_SendChar(c);
      // }
      if (c=='\r' || c=='\n') {
        if (rxi > 0) {
          rxbuf[rxi] = '\0';
          
          // Command: RESET -> clear LSTM internal state
          if (rxbuf[0]=='R' && rxbuf[1]=='E' && rxbuf[2]=='S' && rxbuf[3]=='E' && rxbuf[4]=='T' && (rxbuf[5]=='\0')) {
            ai_wrapper_reset_state();
            UART3_SendString("RST:OK\r\n");
            rxi = 0;
            continue;
          }

          // Command: MODEL:x -> Switch model (0=Base, 1=Pruned, 2=Quantized)
          if (rxbuf[0]=='M' && rxbuf[1]=='O' && rxbuf[2]=='D' && rxbuf[3]=='E' && rxbuf[4]=='L' && rxbuf[5]==':') {
            int mid = rxbuf[6] - '0';
            if (mid >= 0 && mid <= 2) {
                lstm_model_set_type((ModelType)mid);
                ai_wrapper_reset_state(); // Reset state when switching
                UART3_SendString("MOD:OK\r\n");
            } else {
                UART3_SendString("MOD:ERR\r\n");
            }
            rxi = 0;
            continue;
          }
          
          const char *p = rxbuf;
          float feat[6]; int ok = 1;
          for (int i = 0; i < 6; ++i) {
            if (!parse_float_simple(&p, &feat[i])) { ok = 0; break; }
          }
          if (ok) {
            // Debug prints disabled for stable throughput
            
            lstm_reset_stack_measure(); // Reset stack tracker before run
            
            uint32_t t0 = DWT->CYCCNT;
            // PRUNED -> Use LD3 (Red, PB14) for activity
            // HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_SET); // LED ON
            float y = 0.0f; int rc = ai_wrapper_run(feat, &y);
            // HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET); // LED OFF
            uint32_t t1 = DWT->CYCCNT;
            last_inference_tick = HAL_GetTick();
            
            if (rc == 0) {
              // Calculate Metrics
              uint32_t cycles = t1 - t0;
              float us = (float)cycles / 480.0f; // 480 MHz
              float e_uj = us * 0.5f; // Approx 0.5W active power
              
              // Calculate Stack Usage
              uint32_t stack_top = (uint32_t)&_estack;
              uint32_t stack_used = (g_min_stack_ptr != 0xFFFFFFFF) ? (stack_top - g_min_stack_ptr) : 0;

              char s_us[16], s_e[16];
              float_to_str_3(us, s_us, 16);
              float_to_str_6(e_uj, s_e, 16);
              
              char mbuf[128];
              sprintf(mbuf, "METRICS: cycles=%lu us=%s E_uJ=%s Stack=%lu\r\n", cycles, s_us, s_e, stack_used);
              UART3_SendString(mbuf);

              // Output: SOH only (no V*3 debug output for speed)
              char out_soc[24]; float_to_str_3(y, out_soc, sizeof(out_soc));
              char msg1[64]; int m1=0;
              msg1[m1++]='S'; msg1[m1++]='O'; msg1[m1++]='H'; msg1[m1++]=':'; msg1[m1++]=' ';
              for (int i=0; out_soc[i] && m1<(int)sizeof(msg1)-3; ++i) msg1[m1++]=out_soc[i];
              msg1[m1++]='\r'; msg1[m1++]='\n'; msg1[m1]='\0';
              UART3_SendString(msg1);
            } else {
              // Print detailed AI error (type/code)
              int et=0, ec=0; ai_wrapper_get_error(&et, &ec);
              char msg[48]; int m=0;
              const char *prefix = "ERR:AI t="; while (*prefix && m<(int)sizeof(msg)-1) msg[m++]=*prefix++;
              // simple int to ascii for type
              int v = et; if (v<0){ msg[m++]='-'; v=-v; }
              char tmp[12]; int t=0; if (v==0){ tmp[t++]='0'; } else { while(v>0&&t<11){ tmp[t++] = '0'+(v%10); v/=10; } }
              for (int k=t-1;k>=0 && m<(int)sizeof(msg)-1;--k) msg[m++]=tmp[k];
              if (m<(int)sizeof(msg)-1) msg[m++]=' ';
              if (m<(int)sizeof(msg)-1) msg[m++]='c';
              if (m<(int)sizeof(msg)-1) msg[m++]='=';
              v = ec; if (v<0){ if (m<(int)sizeof(msg)-1) msg[m++]='-'; v=-v; }
              t=0; if (v==0){ tmp[t++]='0'; } else { while(v>0&&t<11){ tmp[t++] = '0'+(v%10); v/=10; } }
              for (int k=t-1;k>=0 && m<(int)sizeof(msg)-1;--k) msg[m++]=tmp[k];
              if (m<(int)sizeof(msg)-1) msg[m++]='\r';
              if (m<(int)sizeof(msg)-1) msg[m++]='\n';
              msg[m]='\0';
              UART3_SendString(msg);
            }
          } else {
            UART3_SendString("ERR:PARSE\r\n");
          }
          rxi = 0;
        }
        // NO newline echo for automated testing
        // UART3_SendString("\r\n");
      } else if (rxi < sizeof(rxbuf)-1) {
        rxbuf[rxi++] = c;
      } else {
        // Overflow: reset buffer
        rxi = 0;
      }
    }

    // Adaptive Heartbeat: Fast (50ms) if active, Slow (500ms) if idle
    // Blink ALL LEDs (PB0, PB14, PE1)
    uint32_t interval = (HAL_GetTick() - last_inference_tick < 1000) ? 50 : 500;
    static uint32_t last=0; 
    if (HAL_GetTick() - last > interval) { 
        last = HAL_GetTick(); 
        HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_14); 
        HAL_GPIO_TogglePin(GPIOE, GPIO_PIN_1);
    }
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
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 480;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 20;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_1;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

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
#ifdef USE_FULL_ASSERT
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
