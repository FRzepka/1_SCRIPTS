#include "ram_profiler.h"

#include <stddef.h>
#include <stdint.h>

#define RAM_STACK_PATTERN 0xA5A5A5A5UL

extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;
extern uint32_t _estack;

void RAM_ProfilerRead(ram_profile_t *profile)
{
  const uint32_t *cursor = &_ebss;
  const uint32_t *stack_top = &_estack;

  while ((cursor < stack_top) && (*cursor == RAM_STACK_PATTERN)) {
    ++cursor;
  }

  profile->data_bytes = (uint32_t)((uintptr_t)&_edata - (uintptr_t)&_sdata);
  profile->bss_bytes = (uint32_t)((uintptr_t)&_ebss - (uintptr_t)&_sbss);
  profile->static_bytes = profile->data_bytes + profile->bss_bytes;
  profile->heap_peak_bytes = RAM_ProfilerHeapPeakBytes();
  profile->stack_peak_bytes = (uint32_t)((uintptr_t)stack_top - (uintptr_t)cursor);
  profile->total_peak_bytes = profile->static_bytes + profile->heap_peak_bytes
                              + profile->stack_peak_bytes;
}
