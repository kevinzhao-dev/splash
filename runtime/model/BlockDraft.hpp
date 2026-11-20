#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/QuantFormat.h"
#include "model/WeightImages.hpp"
#include "ops/Embedding.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace splash::model {

// A DFlash block draft as a v1 checkpoint gives it (config.json and BF16
// safetensors, DFlashDraftModel): Qwen3 layers over a block of an anchor
// token and mask tokens, whose keys and values extend those of the target's
// context. A context row is the concatenation of the target's output rows of
// targetLayers, projected (fc) and normalized; every layer projects that same
// row into its keys and values. The block's rows take the target's token
// table rows, times inputScale, and its last blockSize - 1 rows the target's
// vocabulary projection.
struct BlockDraftLayout final {
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t queryHeads = 0;
  uint32_t kvHeads = 0;
  uint32_t intermediateSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t blockSize = 0;
  uint32_t maskToken = 0;
  std::vector<uint32_t> targetLayers;
  // A sliding layer's rows attend causally to the keys less than `window`
  // positions before them; another layer's to every context key and the
  // whole block.
  uint64_t slidingLayers = 0;
  uint32_t window = 0;
  float ropeTheta = 0.0F;
  float epsilon = 0.0F;
  float inputScale = 1.0F;
  // The projections' format as they load (metal/abi/QuantFormat.h):
  // GGUF_FMT_AF4G64 or GGUF_FMT_Q80.
  uint32_t format = GGUF_FMT_AF4G64;

  [[nodiscard]] bool sliding(uint32_t layer) const noexcept { return (slidingLayers >> layer) & 1U; }
  [[nodiscard]] uint32_t contextWidth() const noexcept {
    return hiddenSize * static_cast<uint32_t>(targetLayers.size());
  }
};

// The layout config.json in `checkpoint` states, of a draft whose target has
// `targetLayers` layers of `hiddenSize` and a `vocabularySize` vocabulary;
// throws for a draft the kernels do not run (ModelDescriptor.mm).
[[nodiscard]] BlockDraftLayout readBlockDraftLayout(const std::filesystem::path &checkpoint, uint32_t targetLayers,
                                                    uint32_t hiddenSize, uint32_t vocabularySize);

// The draft's weights, quantized into af4g64 planes as they load, and a KV
// cache of `capacity` positions. A command adds the K/V of context rows the
// target captured (context(), contextRows rows at most) and drafts a block;
// the block's own K/V sit at the positions after the context's until the
// next context rows overwrite them.
class BlockDraft final {
public:
  BlockDraft(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &checkpoint,
             BlockDraftLayout layout, uint32_t capacity, uint32_t contextRows);
  ~BlockDraft();
  BlockDraft(const BlockDraft &) = delete;
  BlockDraft &operator=(const BlockDraft &) = delete;

  [[nodiscard]] const BlockDraftLayout &layout() const noexcept;
  // Rows of contextWidth() bf16 values: target layer i's output row at
  // column i * hiddenSize.
  [[nodiscard]] const metal::MetalBuffer &context() const noexcept;
  // The keys and values of the first `rows` context rows, at positions
  // `position` on.
  void addContext(metal::CommandGraph &graph, uint32_t rows, uint32_t position);
  // The block of `anchor` at `position` and blockSize - 1 mask tokens after
  // it, over the context before `position`: logits() then holds the raw
  // logits of the tokens after the anchor.
  void addBlock(metal::CommandGraph &graph, uint32_t anchor, uint32_t position,
                const ops::EmbeddingWeights &embedding, const ops::Projection &head);
  [[nodiscard]] std::span<const float> logits() const noexcept;
  [[nodiscard]] uint64_t bytes() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
