#ifndef RAM_PROFILER_H
#define RAM_PROFILER_H

#include <stdint.h>

typedef struct {
  uint32_t data_bytes;
  uint32_t bss_bytes;
  uint32_t static_bytes;
  uint32_t heap_peak_bytes;
  uint32_t stack_peak_bytes;
  uint32_t total_peak_bytes;
} ram_profile_t;

void RAM_ProfilerRead(ram_profile_t *profile);
uint32_t RAM_ProfilerHeapPeakBytes(void);

#endif
