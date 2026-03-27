/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body (INT8 LSTM weights + FP32 MLP)
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
#include "lstm_model_soh.h"
#include "scaler_params_soh.h"
  #define BASE_INPUT_SIZE INPUT_SIZE
#include <stdio.h>
#include <string.h>

// Linker symbols for RAM measurement
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;
extern uint32_t _estack;

static void print_memory_info(void);

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);

/* Minimal UART helpers (USART3) */
#include "stm32h753xx.h"

static void UART3_Init_Quick(void)
{
  RCC->APB1LENR |= RCC_APB1LENR_USART3EN;
  USART3->CR1 &= ~(1UL << 0);
  uint32_t pclk = HAL_RCC_GetPCLK1Freq();
  uint32_t baud = 115200U;
  uint32_t brr = (pclk + (baud/2U)) / baud;
  if (brr == 0U) brr = 1U;
  USART3->BRR = brr;
  USART3->CR1 = (1UL << 3) | (1UL << 2);
  USART3->CR2 = 0; USART3->CR3 = 0; USART3->CR1 |= (1UL << 0);
  USART3->ICR = 0xFFFFFFFFU;
}
static inline void UART3_SendChar(char c){ while ((USART3->ISR & (1UL << 7)) == 0){} USART3->TDR = (uint8_t)c; }
static void UART3_SendString(const char *s){ while (*s){ UART3_SendChar(*s++);} while ((USART3->ISR & (1UL << 6)) == 0){} }
static inline int UART3_DataAvailable(void){ return (USART3->ISR & (1UL << 5)) != 0; }
static inline char UART3_ReadChar(void){ if (USART3->ISR & ((1UL<<0)|(1UL<<1)|(1UL<<2)|(1UL<<3))) USART3->ICR = (1UL<<0)|(1UL<<1)|(1UL<<2)|(1UL<<3); return (char)(USART3->RDR & 0xFF); }

static void float_to_str_3(float v, char *buf, int n)
{
  if (n<8) { if (n>0) buf[0]='\0'; return; }
  int pos=0; if (v<0){ buf[pos++]='-'; v=-v; }
  int i = (int)v; int frac = (int)((v - (float)i)*1000.0f + 0.5f); if (frac>=1000){ i+=1; frac=0; }
  char tmp[12]; int t=0; if (i==0){ tmp[t++]='0'; } else { while (i>0 && t<11){ tmp[t++] = '0'+(i%10); i/=10; } }
  for (int k=t-1;k>=0;k--) buf[pos++]=tmp[k];
  buf[pos++]='.'; buf[pos++]='0'+(frac/100); buf[pos++]='0'+((frac/10)%10); buf[pos++]='0'+(frac%10); buf[pos]='\0';
}

static void float_to_str_6(float v, char *buf, int n)
{
  if (n<12) { if (n>0) buf[0]='\0'; return; }
  int pos=0; if (v<0){ buf[pos++]='-'; v=-v; }
  int i = (int)v; int frac = (int)((v - (float)i)*1000000.0f + 0.5f); if (frac>=1000000){ i+=1; frac=0; }
  char tmp[12]; int t=0; if (i==0){ tmp[t++]='0'; } else { while (i>0 && t<11){ tmp[t++] = '0'+(i%10); i/=10; } }
  for (int k=t-1;k>=0;k--) buf[pos++]=tmp[k];
  buf[pos++]='.';
  buf[pos++]='0'+(frac/100000); buf[pos++]='0'+((frac/10000)%10); buf[pos++]='0'+((frac/1000)%10);
  buf[pos++]='0'+((frac/100)%10); buf[pos++]='0'+((frac/10)%10); buf[pos++]='0'+(frac%10);
  buf[pos]='\0';
}

static int parse_float_simple(const char **pp, float *out)
{
  const char *p = *pp;
  while (*p==' '||*p=='\t') p++;
  int s = 1, seen=0; if (*p=='-'){ s=-1; p++; }
  long ip=0; while (*p>='0'&&*p<='9'){ ip = ip*10 + (*p-'0'); p++; seen=1; }
  long fp=0, fz=1; if (*p=='.'){ p++; while (*p>='0'&&*p<='9'){ fp = fp*10 + (*p-'0'); fz*=10; p++; seen=1; } }
  if (!seen) return 0; { *out = s * ( (float)ip + (float)fp/(float)fz ); *pp = p; return 1; }
}

static uint32_t g_static_ram_bytes = 0;

static void print_memory_info(void)
{
  uint32_t data_size = (uint32_t)&_edata - (uint32_t)&_sdata;
  uint32_t bss_size = (uint32_t)&_ebss - (uint32_t)&_sbss;
  uint32_t static_ram = data_size + bss_size;
  g_static_ram_bytes = static_ram;
  uint32_t stack_top = (uint32_t)&_estack;
  uint32_t current_sp;
  __asm volatile ("mov %0, sp" : "=r" (current_sp));
  
  char buf[128];
  sprintf(buf, "RAM_MEASURE: Static=%lu (Data=%lu, BSS=%lu), StackTop=%08lX, SP=%08lX\r\n", 
          static_ram, data_size, bss_size, stack_top, current_sp);
  UART3_SendString(buf);
}

int main(void)
{
  // PRE-HAL: configure LEDs (LD1 PB0, LD2 PE1, LD3 PB14) as outputs and quick blink
  RCC->AHB4ENR |= (RCC_AHB4ENR_GPIOBEN | RCC_AHB4ENR_GPIOEEN);
  // PB0, PB14 output
  GPIOB->MODER &= ~((3UL << (0*2)) | (3UL << (14*2)));
  GPIOB->MODER |=  ((1UL << (0*2)) | (1UL << (14*2)));
  // PE1 output
  GPIOE->MODER &= ~(3UL << (1*2));
  GPIOE->MODER |=  (1UL << (1*2));
  // All ON then OFF
  GPIOB->BSRR = (1UL<<0) | (1UL<<14);
  GPIOE->BSRR = (1UL<<1);
  for (volatile uint32_t i=0;i<1000000U;++i) __ASM volatile("nop");
  GPIOB->BSRR = (1UL<<(0+16)) | (1UL<<(14+16));
  GPIOE->BSRR = (1UL<<(1+16));

  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  UART3_Init_Quick();
  UART3_SendString("BOOT FP32 LSTM SOH PRUNED\r\n");
  
  // Print Memory Info (RAM Measurement)
  print_memory_info();

  {
    char cfgbuf[64];
    int n = sprintf(cfgbuf, "CFG: INPUT_SIZE=%d PAD_MISSING_TO_CENTER=ON\r\n", (int)INPUT_SIZE);
    (void)n; UART3_SendString(cfgbuf);
  }

  // Enable DWT for cycle counting
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  // Ensure LED pins configured via HAL after clock setup
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  GPIO_InitTypeDef gi;
  memset(&gi, 0, sizeof(gi));
  gi.Mode = GPIO_MODE_OUTPUT_PP; gi.Pull = GPIO_NOPULL; gi.Speed = GPIO_SPEED_FREQ_LOW;
  gi.Pin = GPIO_PIN_0 | GPIO_PIN_14; HAL_GPIO_Init(GPIOB, &gi); // LD1 green (PB0), LD3 red (PB14)
  gi.Pin = GPIO_PIN_1; HAL_GPIO_Init(GPIOE, &gi);              // LD2 blue (PE1)
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_14, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1, GPIO_PIN_RESET);

  // Startup: Blink all LEDs 3x fast (like quantized)
  for (int i = 0; i < 3; i++) {
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_14, GPIO_PIN_SET);   // Green + Red ON
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1, GPIO_PIN_SET);                 // Blue ON
    HAL_Delay(200);
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_14, GPIO_PIN_RESET); // All OFF
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1, GPIO_PIN_RESET);
    HAL_Delay(200);
  }

  // Ensure LD2 (PE1 blue) and LD3 (PB14 red) OFF for BASE model; only LD1 (PB0 green) will blink
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);

  // States
  LSTMModelSOH model;
  lstm_model_soh_init(&model);

  char rxbuf[256]; int rxi = 0;
  while (1){
    if (UART3_DataAvailable()){
      char c = UART3_ReadChar();
      if (c=='\r' || c=='\n'){
        if (rxi>0){
          rxbuf[rxi]=0;
          const char *p=rxbuf;
          // Handle simple commands
          if (strncmp(rxbuf, "RESET", 5) == 0){
            lstm_model_soh_reset(&model);
            UART3_SendString("OK RESET\r\n");
            rxi=0;
            continue;
          }
          // Handle MODEL:x command (ignore for single model firmware, but ack it)
          if (strncmp(rxbuf, "MODEL:", 6) == 0){
            lstm_model_soh_reset(&model);
            UART3_SendString("MOD:OK\r\n");
            rxi=0;
            continue;
          }

          float feat[BASE_INPUT_SIZE];
          int parsed=0;
          // Parse up to BASE_INPUT_SIZE floats from the line
          for (int i=0;i<BASE_INPUT_SIZE;i++){
            float v;
            const char *p_before = p;
            if(parse_float_simple(&p,&v)){
              feat[i]=v; parsed++;
            } else {
              // no more numbers
              p = p_before; break;
            }
          }
          if (parsed > 0){
            // If fewer than BASE_INPUT_SIZE provided, fill missing raw features with scaler centers
            for (int i=parsed;i<BASE_INPUT_SIZE;i++){
              // Use the MCU scaler center so scaled value becomes 0
              feat[i] = SCALER_SOH_CENTER[i];
            }
            // Optional: echo parsed input for debugging
            // UART3_SendString("DBG_IN:"); UART3_SendString(rxbuf); UART3_SendString("\r\n");

            // Apply device-side scaling
            float x_scaled[BASE_INPUT_SIZE];
            scaler_soh_transform(feat, x_scaled);

            lstm_reset_stack_measure(); // Reset stack tracker before run
            uint32_t t0 = DWT->CYCCNT;

            float y=0.0f; lstm_model_soh_inference(&model, x_scaled, &y);
            
            uint32_t t1 = DWT->CYCCNT;
            
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
            
            char mbuf[160];
            sprintf(mbuf, "METRICS: cycles=%lu us=%s E_uJ=%s Stack=%lu Static=%lu\r\n",
                    cycles, s_us, s_e, stack_used, g_static_ram_bytes);
            UART3_SendString(mbuf);

            // Print fixed-point with 6 decimals to avoid visible quantization steps
            int micro = (int)(y * 1000000.0f + (y>=0?0.5f:-0.5f));
            int ones  = micro / 1000000; if (micro < 0 && ones==0) ones = -1;
            int frac  = micro % 1000000; if (frac < 0) frac = -frac;
            char msg[64];
            int m = sprintf(msg, "SOH: %d.%06d\r\n", ones, frac); (void)m; UART3_SendString(msg);
            // PRUNED: Blink LD3 (PB14 Red) on each processed sample
            HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_14);
          } else {
            UART3_SendString("ERR:PARSE\r\n");
          }
          rxi=0;
        }
      } else if (rxi < (int)sizeof(rxbuf)-1){ rxbuf[rxi++]=c; } else { rxi=0; }
    }
    // Heartbeat on LD3 (PB14) for PRUNED model
    static uint32_t last=0; if (HAL_GetTick()-last>333){ last=HAL_GetTick(); HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_14);}    
  }
}


void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);
  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}
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
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) { Error_Handler(); }
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2|RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK) { Error_Handler(); }
}

void Error_Handler(void)
{
  __disable_irq();
  while(1){}
}


