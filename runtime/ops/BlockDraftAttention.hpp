#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/BlockDraft.h"
#include "ops/Normalization.hpp"

#include <cstdint>

namespace splash::ops {

// The attention of a DFlash block draft (a v1 checkpoint's Qwen3 layers):
// its KV cache holds a key and a value row of every committed context
// position, keys from the target's context rows, and the block's own at the
// positions after them, which the next context rows overwrite. Every head has
// SPLASH_BLOCK_DRAFT_HEAD_DIMENSION dimensions; the norms are BF16.
class BlockDraftAttention final {
public:
  static constexpr uint32_t kHeadDimension = SPLASH_BLOCK_DRAFT_HEAD_DIMENSION;

  // The q/k norms and RoPE of packed projection rows (SplashBlockDraftRows):
  // their queries into `queries`, [row][query head][dimension], and their
  // keys and values into the cache, [KV head][position][dimension]. The
  // tables hold kHeadDimension / 2 angles a row. Rows of no query heads
  // need no query rows.
  static void addPrepare(metal::CommandGraph &graph, metal::MetalBuffer packed, const NormWeights &queryNorm,
                         const NormWeights &keyNorm, metal::MetalBuffer cosines, metal::MetalBuffer sines,
                         metal::MetalBuffer queries, metal::MetalBuffer keys, metal::MetalBuffer values,
                         const SplashBlockDraftRows &rows);
  // The query rows' attention over the cache (SplashBlockDraftAttention),
  // into `output` in the queries' layout, the out-projection's input rows.
  static void addAttention(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer keys,
                           metal::MetalBuffer values, metal::MetalBuffer output,
                           const SplashBlockDraftAttention &attention);
};

} // namespace splash::ops
