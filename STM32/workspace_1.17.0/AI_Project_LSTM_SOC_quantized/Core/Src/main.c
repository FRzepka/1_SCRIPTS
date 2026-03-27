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
#include "lstm_model_int8.h"
#include "scaler_params.h"
#include <stdio.h>
#include <string.h>

// Linker symbols for RAM measurement
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;
extern uint32_t _estack;

// Stack Measurement Global
uint32_t g_min_stack_ptr = 0xFFFFFFFF;

void lstm_reset_stack_measure(void) {
    g_min_stack_ptr = 0xFFFFFFFF;
}

static void print_memory_info(void);

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);

/* Minimal UART helpers (USART3) */
#include "stm32h753xx.h"

/* Bench config: typical power delta (busy - idle) in mW for coarse energy estimate */
#ifndef TYP_POWER_BUSY_MW
#define TYP_POWER_BUSY_MW   350.0f
#endif
#ifndef TYP_POWER_IDLE_MW
#define TYP_POWER_IDLE_MW   250.0f
#endif

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
  UART3_SendString("BOOT INT8 LSTM FP32 MLP (SOC)\r\n");
  
  // Print Memory Info (RAM Measurement)
  print_memory_info();

  // Startup: Blink all LEDs 3x fast
  for (int i = 0; i < 3; i++) {
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_14, GPIO_PIN_SET);   // Green + Red ON
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1, GPIO_PIN_SET);                 // Blue ON
    HAL_Delay(200);
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_14, GPIO_PIN_RESET); // All OFF
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1, GPIO_PIN_RESET);
    HAL_Delay(200);
  }

  // States
  float h_state[LSTM_HIDDEN_SIZE];
  float c_state[LSTM_HIDDEN_SIZE];
  lstm_model_reset_states(h_state, c_state);

  // Enable DWT cycle counter for precise MCU timing
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  char rxbuf[256]; int rxi = 0;
  static uint32_t last_inference_tick = 0;

  while (1){
    if (UART3_DataAvailable()){
      char c = UART3_ReadChar();
      if (c=='\r' || c=='\n'){
        if (rxi>0){ rxbuf[rxi]=0; const char *p=rxbuf; float feat[6]; int ok=1; for (int i=0;i<6;i++){ if(!parse_float_simple(&p,&feat[i])){ ok=0; break; }}
          if (ok){
            // Measure cycles around inference only (no UART)
            lstm_reset_stack_measure(); // Reset stack tracker before run
            uint32_t start_cyc = DWT->CYCCNT;
            // Inference
            float y=0.0f; lstm_model_predict_int8(feat, h_state, c_state, &y);
            
            uint32_t cyc = DWT->CYCCNT - start_cyc;
            last_inference_tick = HAL_GetTick();

            float us = (float)cyc / 480.0f; // 480 MHz
            float e_uJ = us * 0.5f; // Approx 0.5W active power
            
            // Calculate Stack Usage
            uint32_t stack_top = (uint32_t)&_estack;
            uint32_t stack_used = (g_min_stack_ptr != 0xFFFFFFFF) ? (stack_top - g_min_stack_ptr) : 0;

            char s_us[16], s_e[16];
            float_to_str_3(us, s_us, 16);
            float_to_str_6(e_uJ, s_e, 16);
            
            char mbuf[128];
            sprintf(mbuf, "METRICS: cycles=%lu us=%s E_uJ=%s Stack=%lu Static=%lu\r\n",
                    (unsigned long)cyc, s_us, s_e, stack_used, g_static_ram_bytes);
            UART3_SendString(mbuf);

            // Output: SOC only
            char out_soc[24]; float_to_str_3(y, out_soc, sizeof(out_soc));
            char msg1[64]; int m1=0;
            msg1[m1++]='S'; msg1[m1++]='O'; msg1[m1++]='C'; msg1[m1++]=':'; msg1[m1++]=' ';
            for (int i=0; out_soc[i] && m1<(int)sizeof(msg1)-3; ++i) msg1[m1++]=out_soc[i];
            msg1[m1++]='\r'; msg1[m1++]='\n'; msg1[m1]='\0';
            UART3_SendString(msg1);

          } else {
            UART3_SendString("ERR:PARSE\r\n");
          }
          rxi=0;
        }
      } else if (rxi < (int)sizeof(rxbuf)-1){ rxbuf[rxi++]=c; } else { rxi=0; }
    }
    // Adaptive Heartbeat: Fast (50ms) if active, Slow (500ms) if idle
    // QUANTIZED Model -> Use LD2 (Yellow/Blue, PE1)
    uint32_t interval = (HAL_GetTick() - last_inference_tick < 1000) ? 50 : 500;
    static uint32_t last=0; if (HAL_GetTick() - last > interval) { last = HAL_GetTick(); HAL_GPIO_TogglePin(GPIOE, GPIO_PIN_1); }
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


