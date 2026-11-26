#include "ops/RowMath.hpp"

#include "metal/abi/RowMath.h"
#include "ops/BufferExtent.hpp"

#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kThreads = 256;
metal::DispatchSize elementGroups(uint32_t count) { return {(uint64_t{count} + kThreads - 1) / kThreads, 1, 1}; }

} // namespace

void scaleRows(metal::CommandGraph &graph, metal::MetalBuffer rows, uint32_t count, float scale) {
  requireBytes(rows, uint64_t{count} * sizeof(uint16_t), "scaled rows");
  graph.add("scale_bf16", {std::move(rows)}, SplashScaledElements{count, scale},
            elementGroups(count), {kThreads, 1, 1});
}

void addScaledResidual(metal::CommandGraph &graph, metal::MetalBuffer rows, metal::MetalBuffer update,
                       uint32_t count, float scale) {
  requireBytes(rows, uint64_t{count} * sizeof(uint16_t), "residual rows");
  requireBytes(update, uint64_t{count} * sizeof(uint16_t), "residual update");
  graph.add("add_scaled_bf16", {std::move(rows), std::move(update)}, SplashScaledElements{count, scale},
            elementGroups(count), {kThreads, 1, 1});
}

void addGeluGate(metal::CommandGraph &graph, metal::MetalBuffer gate, metal::MetalBuffer up, uint32_t count) {
  requireBytes(gate, uint64_t{count} * sizeof(uint16_t), "GELU gate");
  requireBytes(up, uint64_t{count} * sizeof(uint16_t), "GELU up projection");
  graph.add("gelu_gate_bf16", {std::move(gate), std::move(up)}, count, elementGroups(count), {kThreads, 1, 1});
}

} // namespace splash::ops
