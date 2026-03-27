/* Minimal STM32 Main - UART komplett inaktiv */
#include "main.h"

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  
  /* Kein UART Init! */
  
  while (1)
  {
    HAL_Delay(1000);  // Einfach warten
  }
}

void SystemClock_Config(void) { /* Deine normale Clock Config */ }
void Error_Handler(void) { while(1); }
