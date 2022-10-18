//===-- PathProfiling.h - Path profiler runtime -------------------*- C -*-===//
///
/// \file
/// Path profiler runtime.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

typedef struct FTEntry {
  uint32_t size;
  void *array;
} FTEntry;
