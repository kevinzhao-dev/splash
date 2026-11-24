#include "metal/abi/RowMath.h"

using namespace metal;

kernel void scale_bf16(device bfloat *rows [[buffer(0)]],
                       constant SplashScaledElements &params [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
  if (index < params.count) rows[index] = bfloat(float(rows[index]) * params.scale);
}

kernel void add_scaled_bf16(device bfloat *rows [[buffer(0)]],
                            device const bfloat *update [[buffer(1)]],
                            constant SplashScaledElements &params [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  if (index < params.count)
    rows[index] = bfloat((float(rows[index]) + float(update[index])) * params.scale);
}

kernel void gelu_gate_bf16(device bfloat *gate [[buffer(0)]],
                           device const bfloat *up [[buffer(1)]],
                           constant uint &count [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  const float x = float(gate[index]);
  // tanh(10) rounds to 1 in float32. Keep its argument in that range so
  // the device approximation's exponentials do not overflow on large gates.
  const float argument = clamp(0.7978845608028654f * x * (1.0f + 0.044715f * x * x), -10.0f, 10.0f);
  const float gelu = 0.5f * x * (1.0f + tanh(argument));
  gate[index] = bfloat(gelu * float(up[index]));
}
