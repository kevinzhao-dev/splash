#include "model/BlockDraft.hpp"

#include "metal/abi/QuantFormat.h"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightStore.hpp"
#include "ops/BlockDraftAttention.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Normalization.hpp"
#include "ops/RowMath.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

constexpr uint32_t kHeadDimension = ops::BlockDraftAttention::kHeadDimension;
constexpr uint64_t kBf16 = sizeof(uint16_t);

// Plans one image from the checkpoint's BF16 tensors, which must have the
// shapes the draft reads: norms copied as stored, projections quantized into
// af4g64 planes as the DFlash2 drafts' are (model/DraftCheckpoint.cpp), or
// into Q8_0 as a GGUF's BF16 tensors are (quantize_row_q8_0_ref).
class Builder {
public:
  Builder(const SafetensorsCheckpoint &checkpoint, std::string name, uint32_t layer, uint32_t type,
          uint32_t format)
      : checkpoint_(checkpoint), image_(std::move(name), layer, type), format_(format) {}

  void stored(const std::string &name, uint64_t width) {
    const SourceTensor &tensor = require(name, {width});
    image_.copy({name, ggml::kBF16, tensor.offset, 1, tensor.bytes, {}, tensor.file});
  }

  // A projection of `columns` inputs: its parts' rows in order.
  void projection(std::initializer_list<std::pair<std::string, uint64_t>> parts, uint64_t columns) {
    const QuantFormat &format = kQuantFormats[format_];
    std::vector<gguf::TensorRows> sources;
    uint64_t rows = 0;
    for (const auto &[name, count] : parts) {
      const SourceTensor &tensor = require(name + ".weight", {count, columns});
      if (format_ == GGUF_FMT_AF4G64) {
        sources.push_back({name, format.ggml_type, 0, count, ggufRowBytes(format, columns), {}, tensor.file,
                           {.bfloat16 = &tensor}});
      } else {
        gguf::TensorRows source{name, ggml::kQ8_0, tensor.offset, count, ggufRowBytes(format, columns), {},
                                tensor.file};
        source.ggufFloatType = ggml::kBF16;
        source.ggufFloatRowBytes = columns * sizeof(uint16_t);
        sources.push_back(std::move(source));
      }
      rows += count;
    }
    gguf::Repack repack = image_.planes(format_, rows, columns, parts.begin()->first);
    repack.sources = std::move(sources);
    image_.repack(std::move(repack));
  }

  [[nodiscard]] gguf::Image finish() { return image_.finish(); }

private:
  const SourceTensor &require(const std::string &name, const std::vector<uint64_t> &shape) const {
    const SourceTensor &tensor = checkpoint_.require(name);
    if (tensor.dtype != "BF16" || tensor.shape != shape)
      throw WeightStoreError("block draft tensor type or shape does not match: " + name);
    return tensor;
  }

  const SafetensorsCheckpoint &checkpoint_;
  gguf::ImageBuilder image_;
  uint32_t format_;
};

// Each layer's image: its input norm, its block rows' QKV projection, its
// context rows' KV projection (the same keys' and values' rows), its q/k
// norms, output projection, post-attention norm and SwiGLU projections. Then
// model.bin: the context projection (fc) and the hidden and final norms.
std::vector<gguf::Image> planImages(const SafetensorsCheckpoint &checkpoint, const BlockDraftLayout &layout) {
  std::vector<gguf::Image> images;
  const uint64_t hidden = layout.hiddenSize, queries = uint64_t{layout.queryHeads} * kHeadDimension;
  const uint64_t kv = uint64_t{layout.kvHeads} * kHeadDimension, intermediate = layout.intermediateSize;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    Builder b(checkpoint, "layer-" + std::to_string(layer) + ".bin", layer, 0, layout.format);
    const std::string prefix = "layers." + std::to_string(layer) + ".", attention = prefix + "self_attn.";
    b.stored(prefix + "input_layernorm.weight", hidden);
    b.projection({{attention + "q_proj", queries}, {attention + "k_proj", kv}, {attention + "v_proj", kv}}, hidden);
    b.projection({{attention + "k_proj", kv}, {attention + "v_proj", kv}}, hidden);
    b.stored(attention + "q_norm.weight", kHeadDimension);
    b.stored(attention + "k_norm.weight", kHeadDimension);
    b.projection({{attention + "o_proj", hidden}}, queries);
    b.stored(prefix + "post_attention_layernorm.weight", hidden);
    b.projection({{prefix + "mlp.gate_proj", intermediate}}, hidden);
    b.projection({{prefix + "mlp.up_proj", intermediate}}, hidden);
    b.projection({{prefix + "mlp.down_proj", hidden}}, intermediate);
    images.push_back(b.finish());
  }
  Builder b(checkpoint, "model.bin", layout.layers, 1, layout.format);
  b.projection({{"fc", hidden}}, layout.contextWidth());
  b.stored("hidden_norm.weight", hidden);
  b.stored("norm.weight", hidden);
  images.push_back(b.finish());
  return images;
}

struct Layer final {
  ops::NormWeights input, query, key, postAttention;
  ops::Projection qkv, kv, output, gate, up, down;
};

} // namespace

struct BlockDraft::Impl final {
  Impl(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &checkpoint,
       BlockDraftLayout draftLayout, uint32_t cacheCapacity, uint32_t maximumContextRows)
      : backend(backend), layout(std::move(draftLayout)), capacity(cacheCapacity),
        contextRows(maximumContextRows),
        operators(backend.capabilities(), ops::Int8Activations::Off),
        planned(std::make_shared<Planned>(checkpoint)) {
    if (layout.format != GGUF_FMT_AF4G64 && layout.format != GGUF_FMT_Q80)
      throw std::invalid_argument("a block draft's projections are af4g64 or Q8_0");
    if (layout.hiddenSize % 64 || layout.contextWidth() % 64)
      throw std::invalid_argument("block draft widths are not af4g64 groups");
    planned->images = planImages(planned->checkpoint, layout);
    for (const auto &image : planned->images) weightBytes += image.bytes;
    loadWeights(images);
    allocateBuffers();
  }

  struct Planned final {
    explicit Planned(const std::filesystem::path &directory) : checkpoint(directory) {}
    SafetensorsCheckpoint checkpoint;
    std::vector<gguf::Image> images;
  };

  void loadWeights(WeightImages &images) {
    const auto open = [&](size_t index) {
      return loadImage(backend, images, "block-draft/", {planned, &planned->images[index]},
                       [planned = planned] { planned->checkpoint.checkUnchanged(); });
    };
    const uint32_t hidden = layout.hiddenSize, queries = layout.queryHeads * kHeadDimension;
    const uint32_t kv = layout.kvHeads * kHeadDimension, intermediate = layout.intermediateSize;
    for (uint32_t index = 0; index < layout.layers; ++index) {
      WeightFile file = open(index);
      Layer &w = layers.emplace_back();
      w.input = readNorm(file, hidden, false, "block draft input norm");
      w.qkv = readBlockProjection(file, queries + 2 * kv, hidden, "block draft QKV");
      w.kv = readBlockProjection(file, 2 * kv, hidden, "block draft context KV");
      w.query = readNorm(file, kHeadDimension, false, "block draft query norm");
      w.key = readNorm(file, kHeadDimension, false, "block draft key norm");
      w.output = readBlockProjection(file, hidden, queries, "block draft output");
      w.postAttention = readNorm(file, hidden, false, "block draft post-attention norm");
      w.gate = readBlockProjection(file, intermediate, hidden, "block draft gate");
      w.up = readBlockProjection(file, intermediate, hidden, "block draft up");
      w.down = readBlockProjection(file, hidden, intermediate, "block draft down");
      file.finish();
    }
    WeightFile file = open(layout.layers);
    contextProjection = readBlockProjection(file, hidden, layout.contextWidth(), "block draft context projection");
    hiddenNorm = readNorm(file, hidden, false, "block draft hidden norm");
    finalNorm = readNorm(file, hidden, false, "block draft final norm");
    file.finish();
    for (float &frequency : frequencies) {
      const auto pair = static_cast<float>(&frequency - frequencies.data());
      frequency = 1.0F / std::pow(layout.ropeTheta, 2.0F * pair / float(kHeadDimension));
    }
  }

  [[nodiscard]] metal::MetalBuffer allocate(uint64_t bytes, std::string_view name) {
    return backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "block-draft-" + std::string(name));
  }

  void allocateBuffers() {
    const ops::Linear &linear = operators.linear();
    const uint32_t hidden = layout.hiddenSize, block = layout.blockSize;
    const uint32_t queries = layout.queryHeads * kHeadDimension, kv = layout.kvHeads * kHeadDimension;
    // The decode tiles store up to 64 rows of a block's plans.
    blockRows = std::max(block, 64U);
    context = allocate(uint64_t{contextRows} * layout.contextWidth() * kBf16, "context");
    contextHidden = allocate(uint64_t{contextRows} * hidden * kBf16, "context-hidden");
    contextNormalized = allocate(uint64_t{contextRows} * hidden * kBf16, "context-normalized");
    contextKv = allocate(uint64_t{contextRows} * 2 * kv * kBf16, "context-kv");
    contextCosines = allocate(uint64_t{contextRows} * kHeadDimension / 2 * sizeof(float), "context-cos");
    contextSines = allocate(contextCosines.sizeBytes(), "context-sin");
    tokens = allocate(block * sizeof(uint32_t), "tokens");
    for (auto &rows : residual) rows = allocate(uint64_t{blockRows} * hidden * kBf16, "residual");
    normalized = allocate(uint64_t{blockRows} * hidden * kBf16, "normalized");
    packed = allocate(uint64_t{blockRows} * (queries + 2 * kv) * kBf16, "qkv");
    blockQueries = allocate(uint64_t{block} * queries * kBf16, "queries");
    attended = allocate(uint64_t{blockRows} * queries * kBf16, "attention");
    gateScratch = allocate(uint64_t{blockRows} * layout.intermediateSize * sizeof(float), "gate");
    intermediate = allocate(uint64_t{blockRows} * layout.intermediateSize * kBf16, "intermediate");
    blockCosines = allocate(uint64_t{block} * kHeadDimension / 2 * sizeof(float), "block-cos");
    blockSines = allocate(blockCosines.sizeBytes(), "block-sin");
    for (uint32_t layer = 0; layer < layout.layers; ++layer) {
      keys.push_back(allocate(uint64_t{layout.kvHeads} * capacity * kHeadDimension * kBf16, "keys"));
      values.push_back(allocate(keys.back().sizeBytes(), "values"));
    }
    ops::LinearScratchSize scratchSize;
    scratchSize.include(linear.scratchSize(contextProjection.shape(), contextRows));
    for (const Layer &w : layers) {
      scratchSize.include(linear.scratchSize(w.kv.shape(), contextRows));
      for (const auto *projection : {&w.qkv, &w.output, &w.gate, &w.up, &w.down})
        scratchSize.include(linear.scratchSize(projection->shape(), block));
    }
    headRows = block - 1;
    const auto buffer = [&](uint64_t bytes, std::string_view name) {
      return bytes ? allocate(bytes, name) : metal::MetalBuffer{};
    };
    scratchBytes = scratchSize;
    scratch = {buffer(scratchSize.input, "linear-input"), buffer(scratchSize.sums, "linear-sums"),
               buffer(scratchSize.partials, "linear-partials"), buffer(scratchSize.counters, "linear-counters")};
    if (scratch.counters) std::memset(scratch.counters.contents(), 0, scratch.counters.sizeBytes());
  }

  // The head's logits rows and its scratch, sized on first use: the target
  // owns the projection.
  void prepareHead(const ops::Projection &head) {
    if (logits) return;
    const ops::Linear &linear = operators.linear();
    ops::LinearScratchSize size = scratchBytes;
    size.include(linear.scratchSize(head.shape(), headRows));
    const auto grow = [&](metal::MetalBuffer &buffer, uint64_t bytes, std::string_view name) {
      if (bytes > buffer.sizeBytes()) buffer = allocate(bytes, name);
    };
    grow(scratch.input, size.input, "linear-input");
    grow(scratch.sums, size.sums, "linear-sums");
    grow(scratch.partials, size.partials, "linear-partials");
    if (size.counters > scratch.counters.sizeBytes()) {
      scratch.counters = allocate(size.counters, "linear-counters");
      std::memset(scratch.counters.contents(), 0, scratch.counters.sizeBytes());
    }
    logits = allocate(uint64_t{layout.vocabularySize} * sizeof(float) * linear.storageRows(headRows, head.shape()),
                      "logits");
  }

  void writeAngles(const metal::MetalBuffer &cosines, const metal::MetalBuffer &sines, uint32_t rows,
                   uint32_t position) const {
    auto *cos = static_cast<float *>(cosines.contents()), *sin = static_cast<float *>(sines.contents());
    for (uint32_t row = 0; row < rows; ++row)
      for (float frequency : frequencies) {
        const float angle = float(position + row) * frequency;
        *cos++ = std::cos(angle);
        *sin++ = std::sin(angle);
      }
  }

  [[nodiscard]] SplashBlockDraftRows rowsAt(uint32_t rows, uint32_t position, uint32_t queryHeads) const {
    return {rows, position, queryHeads, layout.kvHeads, capacity, layout.epsilon};
  }

  void addContext(metal::CommandGraph &graph, uint32_t rows, uint32_t position) {
    if (!rows || rows > contextRows || position + uint64_t{rows} > capacity)
      throw std::invalid_argument("block draft context rows do not fit");
    const ops::Linear &linear = operators.linear();
    writeAngles(contextCosines, contextSines, rows, position);
    linear.addProjection(graph, context, contextProjection, contextHidden, rows, scratch);
    ops::Normalization::addRms(graph, contextHidden, hiddenNorm, contextNormalized, layout.hiddenSize, rows);
    for (uint32_t layer = 0; layer < layout.layers; ++layer) {
      const Layer &w = layers[layer];
      linear.addProjection(graph, contextNormalized, w.kv, contextKv, rows, scratch);
      ops::BlockDraftAttention::addPrepare(graph, contextKv, w.key, w.key, contextCosines, contextSines, {},
                                           keys[layer], values[layer], rowsAt(rows, position, 0));
    }
  }

  void addBlock(metal::CommandGraph &graph, uint32_t anchor, uint32_t position,
                const ops::EmbeddingWeights &embedding, const ops::Projection &head) {
    const uint32_t block = layout.blockSize, hidden = layout.hiddenSize;
    if (position + uint64_t{block} > capacity) throw std::invalid_argument("a block draft past its cache");
    prepareHead(head);
    const ops::Linear &linear = operators.linear();
    auto *ids = static_cast<uint32_t *>(tokens.contents());
    ids[0] = anchor;
    std::fill(ids + 1, ids + block, layout.maskToken);
    writeAngles(blockCosines, blockSines, block, position);
    ops::Embedding::add(graph, tokens, embedding, residual[0], block);
    if (layout.inputScale != 1.0F) ops::scaleRows(graph, residual[0], hidden * block, layout.inputScale);
    const float scale = 1.0F / std::sqrt(float(kHeadDimension));
    for (uint32_t layer = 0; layer < layout.layers; ++layer) {
      const Layer &w = layers[layer];
      const bool sliding = layout.sliding(layer);
      ops::Normalization::addRms(graph, residual[0], w.input, normalized, hidden, block);
      linear.addProjection(graph, normalized, w.qkv, packed, block, scratch);
      ops::BlockDraftAttention::addPrepare(graph, packed, w.query, w.key, blockCosines, blockSines, blockQueries,
                                           keys[layer], values[layer], rowsAt(block, position, layout.queryHeads));
      ops::BlockDraftAttention::addAttention(
          graph, blockQueries, keys[layer], values[layer], attended,
          {block, position, layout.queryHeads, layout.kvHeads, capacity, sliding ? 1U : 0U,
           sliding ? layout.window : 0U, scale});
      linear.addResidualProjection(graph, attended, w.output, residual[0], residual[1], block, scratch);
      ops::Normalization::addRms(graph, residual[1], w.postAttention, normalized, hidden, block);
      linear.addSwiGlu(graph, {&w.gate, &w.up, &w.down}, {normalized, gateScratch, intermediate, scratch},
                       residual[1], residual[0], block);
    }
    auto last = backend.view(residual[0], hidden * kBf16, uint64_t{headRows} * hidden * kBf16);
    ops::Normalization::addRms(graph, std::move(last), finalNorm, normalized, hidden, headRows);
    linear.addProjection(graph, normalized, head, logits, headRows, scratch);
  }

  metal::MetalBackend &backend;
  BlockDraftLayout layout;
  uint32_t capacity, contextRows, blockRows = 0, headRows = 0;
  ops::ExecutionPlans operators;
  std::shared_ptr<Planned> planned;
  uint64_t weightBytes = 0;
  std::vector<Layer> layers;
  ops::Projection contextProjection;
  ops::NormWeights hiddenNorm, finalNorm;
  std::array<float, kHeadDimension / 2> frequencies{};
  metal::MetalBuffer context, contextHidden, contextNormalized, contextKv, contextCosines, contextSines;
  metal::MetalBuffer tokens, normalized, packed, blockQueries, attended, gateScratch, intermediate;
  metal::MetalBuffer blockCosines, blockSines, logits;
  std::array<metal::MetalBuffer, 2> residual;
  std::vector<metal::MetalBuffer> keys, values;
  ops::LinearScratchSize scratchBytes;
  ops::LinearScratch scratch;
};

BlockDraft::BlockDraft(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &checkpoint,
                       BlockDraftLayout layout, uint32_t capacity, uint32_t contextRows)
    : impl_(std::make_unique<Impl>(backend, images, checkpoint, std::move(layout), capacity, contextRows)) {}

BlockDraft::~BlockDraft() = default;

const BlockDraftLayout &BlockDraft::layout() const noexcept { return impl_->layout; }
const metal::MetalBuffer &BlockDraft::context() const noexcept { return impl_->context; }

void BlockDraft::addContext(metal::CommandGraph &graph, uint32_t rows, uint32_t position) {
  impl_->addContext(graph, rows, position);
}

void BlockDraft::addBlock(metal::CommandGraph &graph, uint32_t anchor, uint32_t position,
                          const ops::EmbeddingWeights &embedding, const ops::Projection &head) {
  impl_->addBlock(graph, anchor, position, embedding, head);
}

std::span<const float> BlockDraft::logits() const noexcept {
  if (!impl_->logits) return {};
  return {static_cast<const float *>(impl_->logits.contents()), size_t{impl_->layout.vocabularySize} * impl_->headRows};
}

uint64_t BlockDraft::bytes() const noexcept { return impl_->weightBytes; }

} // namespace splash::model
