#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::ops {

// Elementwise operations on the first `count` values of packed BF16 rows.
void scaleRows(metal::CommandGraph &graph, metal::MetalBuffer rows, uint32_t count, float scale);
void addScaledResidual(metal::CommandGraph &graph, metal::MetalBuffer rows, metal::MetalBuffer update,
                       uint32_t count, float scale);
void addGeluGate(metal::CommandGraph &graph, metal::MetalBuffer gate, metal::MetalBuffer up, uint32_t count);

} // namespace splash::ops
