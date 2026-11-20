#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct SplashScaledElements {
  uint32_t count;
  float scale;
};
