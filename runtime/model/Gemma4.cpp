#include "Autoregressive.hpp"
#include "AwakeClock.hpp"

#include "model/BlockDraft.hpp"

#include "metal/abi/Gguf.h"
#include "model/GgufPreparation.hpp"
#include "model/LookupProposals.hpp"
#include "ops/Embedding.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/PageStorage.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/RowCopy.hpp"
#include "ops/RowMath.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace splash::model {
namespace {

constexpr uint32_t kLayers = 60, kHidden = 5376, kQueryHeads = 32, kIntermediate = 21504;
constexpr uint32_t kVocab = 262144, kGlobalHeads = 4, kGlobalDim = 512, kLocalHeads = 16, kLocalDim = 256;
constexpr uint32_t kNativeContext = 262144;
constexpr uint32_t kWindow = 1024, kEndOfSequence = 1, kEndOfTurn = 106;
constexpr uint32_t kPrefillRows = 512, kExtentPages = 4;
constexpr float kGlobalTheta = 1'000'000.0f, kLocalTheta = 10'000.0f, kSoftcap = 30.0f;
constexpr auto kQ4Type = kQuantFormats[GGUF_FMT_Q40].ggml_type;
constexpr auto kEmbeddingType = kQuantFormats[GGUF_FMT_Q6K].ggml_type;
using Clock = AwakeClock;

struct PlannedWeights final {
  std::shared_ptr<OpenGguf> source;
  kv::Layout kv;
  std::vector<gguf::Image> images;
  uint64_t bytes = 0;
};

kv::Layout inspectLayout(const GgufFile &file) {
  const auto &meta = file.metadata();
  if (meta.architecture() != "gemma4" || file.rotation() ||
      meta.stringValue("tokenizer.ggml.model") != "gemma4")
    throw GgufError("expected a Gemma 4 text GGUF with the Gemma 4 vocabulary");
  const std::pair<std::string_view, uint64_t> integers[] = {
      {"gemma4.block_count", kLayers}, {"gemma4.embedding_length", kHidden},
      {"gemma4.context_length", kNativeContext}, {"gemma4.feed_forward_length", kIntermediate},
      {"gemma4.attention.head_count", kQueryHeads}, {"gemma4.attention.sliding_window", kWindow},
      {"gemma4.attention.key_length", kGlobalDim}, {"gemma4.attention.value_length", kGlobalDim},
      {"gemma4.attention.key_length_swa", kLocalDim}, {"gemma4.attention.value_length_swa", kLocalDim},
      {"gemma4.rope.dimension_count", kGlobalDim}, {"gemma4.rope.dimension_count_swa", kLocalDim},
      {"gemma4.attention.shared_kv_layers", 0}, {"gemma4.embedding_length_per_layer_input", 0},
      {"tokenizer.ggml.eos_token_id", kEndOfSequence}, {"general.file_type", 2},
  };
  for (auto [name, value] : integers)
    if (meta.unsignedValue(name) != value) throw GgufError("unsupported Gemma 4 metadata: " + std::string(name));
  const std::pair<std::string_view, double> floats[] = {
      {"gemma4.rope.freq_base", kGlobalTheta}, {"gemma4.rope.freq_base_swa", kLocalTheta},
      {"gemma4.final_logit_softcapping", kSoftcap},
      {"gemma4.attention.layer_norm_rms_epsilon", double(float(SPLASH_RMS_EPSILON))},
  };
  for (auto [name, value] : floats)
    if (meta.floatValue(name) != value) throw GgufError("unsupported Gemma 4 metadata: " + std::string(name));
  const auto sliding = file.numericArray("gemma4.attention.sliding_window_pattern");
  const auto heads = file.numericArray("gemma4.attention.head_count_kv");
  if (!sliding || !heads || sliding->size() != kLayers || heads->size() != kLayers)
    throw GgufError("Gemma 4 needs a window and KV head count for each layer");
  uint64_t windowed = 0;
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    const double local = (*sliding)[layer];
    if ((local != 0 && local != 1) || (*heads)[layer] != (local ? kLocalHeads : kGlobalHeads))
      throw GgufError("unsupported Gemma 4 KV shape in layer " + std::to_string(layer));
    if (local) windowed |= uint64_t{1} << layer;
  }
  if (std::popcount(windowed) != 50) throw GgufError("expected 50 windowed layers in the 31B text checkpoint");
  return {.attentionLayers = kLayers, .kvHeads = kGlobalHeads, .headDimension = kGlobalDim,
          .format = kv::Format::BFloat16, .windowTokens = kWindow - 1, .windowedLayers = windowed,
          .windowKvHeads = kLocalHeads, .windowHeadDimension = kLocalDim};
}

gguf::TensorRows tensorRows(const GgufFile &file, const std::string &name, uint32_t type,
                            std::initializer_list<uint64_t> dims) {
  const GgufTensor &tensor = file.require(name);
  if (tensor.type != type || !std::ranges::equal(tensor.dims, dims))
    throw GgufError("unexpected type or shape for " + name);
  const uint64_t rows = tensor.rows();
  return {tensor.name, type, tensor.offset, rows, tensor.bytes / rows, {}, &file.source(tensor.file)};
}

PlannedWeights planWeights(const std::filesystem::path &path, GgufContents contents) {
  PlannedWeights plan;
  plan.source = std::make_shared<OpenGguf>(std::vector{path}, contents);
  const GgufFile &file = plan.source->file();
  plan.kv = inspectLayout(file);
  if (file.tensors().size() != 3 + 13 * kLayers + std::popcount(plan.kv.windowedLayers))
    throw GgufError("unexpected tensors for the dense Gemma 4 31B text checkpoint");
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    const auto shape = plan.kv.layerLayout(layer);
    const uint32_t dim = shape.headDimension, q = dim * kQueryHeads, kvWidth = dim * shape.kvHeads;
    const bool local = plan.kv.windowed(layer);
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    gguf::ImageBuilder image("layer-" + std::to_string(layer) + ".bin", layer, 1);
    const auto norm = [&](std::string_view name, uint32_t width) {
      image.copy(tensorRows(file, prefix + std::string(name), ggml::kF32, {width}));
    };
    const auto matrix = [&](std::string_view name, uint32_t input, uint32_t output) {
      const std::string tensor = prefix + std::string(name);
      auto repack = image.planes(GGUF_FMT_Q40, output, input, tensor);
      repack.sources.push_back(tensorRows(file, tensor, kQ4Type, {input, output}));
      image.repack(std::move(repack));
    };
    norm("attn_norm.weight", kHidden);
    norm("attn_q_norm.weight", dim);
    norm("attn_k_norm.weight", dim);
    norm("post_attention_norm.weight", kHidden);
    norm("ffn_norm.weight", kHidden);
    norm("post_ffw_norm.weight", kHidden);
    norm("layer_output_scale.weight", 1);
    auto qkv = image.planes(GGUF_FMT_Q40, q + kvWidth * (local ? 2 : 1), kHidden, prefix + "qkv");
    qkv.sources.push_back(tensorRows(file, prefix + "attn_q.weight", kQ4Type, {kHidden, q}));
    qkv.sources.push_back(tensorRows(file, prefix + "attn_k.weight", kQ4Type, {kHidden, kvWidth}));
    if (local) qkv.sources.push_back(tensorRows(file, prefix + "attn_v.weight", kQ4Type, {kHidden, kvWidth}));
    image.repack(std::move(qkv));
    matrix("attn_output.weight", q, kHidden);
    matrix("ffn_gate.weight", kHidden, kIntermediate);
    matrix("ffn_up.weight", kHidden, kIntermediate);
    matrix("ffn_down.weight", kIntermediate, kHidden);
    plan.images.push_back(image.finish());
  }
  gguf::ImageBuilder output("head.bin", kLayers, 2);
  output.copy(tensorRows(file, "rope_freqs.weight", ggml::kF32, {kGlobalDim / 2}));
  output.copy(tensorRows(file, "output_norm.weight", ggml::kF32, {kHidden}));
  const auto embedding = tensorRows(file, "token_embd.weight", kEmbeddingType, {kHidden, kVocab});
  output.descriptor(kEmbeddingType, kVocab, kHidden, {}, {embedding.rows * embedding.rowBytes, 0, 0}, embedding.name);
  output.copy(embedding);
  auto logits = output.planes(GGUF_FMT_Q6K, kVocab, kHidden, embedding.name);
  logits.sources.push_back(embedding);
  output.repack(std::move(logits));
  plan.images.push_back(output.finish());
  for (const auto &image : plan.images) plan.bytes += image.bytes;
  return plan;
}

struct LayerWeights final {
  ops::NormWeights attention, query, key, postAttention, ffn, postFfn;
  float outputScale;
  ops::Projection qkv, attentionOut, gate, up, down;
};

struct Weights final {
  std::vector<LayerWeights> layers;
  ops::NormWeights finalNorm;
  ops::EmbeddingWeights embedding;
  ops::Projection logits;
  std::array<float, kGlobalDim / 2> ropeFactors;
};

Weights loadWeights(metal::MetalBackend &backend, WeightImages &images, PlannedWeights &plan) {
  const auto open = [&](uint32_t index) {
    return loadImage(backend, images, "target/", std::make_shared<gguf::Image>(std::move(plan.images[index])),
                     [source = plan.source] { source->checkUnchanged(); });
  };
  Weights weights;
  weights.layers.reserve(kLayers);
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    const auto shape = plan.kv.layerLayout(layer);
    const uint32_t dim = shape.headDimension, q = kQueryHeads * dim;
    const uint32_t packed = q + shape.kvHeads * dim * (plan.kv.windowed(layer) ? 2 : 1);
    auto image = open(layer);
    auto &w = weights.layers.emplace_back();
    w.attention = readNorm(image, kHidden, true, "attention RMS norm");
    w.query = readNorm(image, dim, true, "query RMS norm");
    w.key = readNorm(image, dim, true, "key RMS norm");
    w.postAttention = readNorm(image, kHidden, true, "post-attention RMS norm");
    w.ffn = readNorm(image, kHidden, true, "pre-FFN RMS norm");
    w.postFfn = readNorm(image, kHidden, true, "post-FFN RMS norm");
    std::memcpy(&w.outputScale, image.read(sizeof(float), "layer output scale").data(), sizeof(float));
    if (!std::isfinite(w.outputScale)) throw GgufError("non-finite Gemma 4 layer output scale");
    w.qkv = readBlockProjection(image, packed, kHidden, "QKV");
    w.attentionOut = readBlockProjection(image, kHidden, q, "attention output");
    w.gate = readBlockProjection(image, kIntermediate, kHidden, "FFN gate");
    w.up = readBlockProjection(image, kIntermediate, kHidden, "FFN up");
    w.down = readBlockProjection(image, kHidden, kIntermediate, "FFN down");
    image.finish();
  }
  auto image = open(kLayers);
  std::memcpy(weights.ropeFactors.data(), image.read(sizeof(weights.ropeFactors), "global RoPE factors").data(),
              sizeof(weights.ropeFactors));
  for (float factor : weights.ropeFactors)
    if (!(factor >= 1.0f) || !std::isfinite(factor)) throw GgufError("invalid Gemma 4 global RoPE factor");
  weights.finalNorm = readNorm(image, kHidden, true, "output RMS norm");
  weights.embedding = readBlockEmbedding(image, kVocab, kHidden, "token embedding");
  weights.logits = readBlockProjection(image, kVocab, kHidden, "tied vocabulary projection");
  weights.logits.destination = ops::FloatOutput::Float32;
  image.finish();
  return weights;
}

constexpr uint64_t rowBytes(uint32_t width, uint32_t rows) { return uint64_t{width} * rows * 2; }
uint32_t ceilDiv(uint32_t count, uint32_t unit) { return (count + unit - 1) / unit; }

class Forward final {
public:
  // A forward of up to `storageRows` rows a call, the last `logitRows` of
  // which at most output logits. With a draft, every forward captures its
  // rows' outputs of the draft's target layers and adds their keys and values
  // to the draft's context.
  Forward(metal::MetalBackend &backend, const Weights &weights, kv::Layout layout, uint32_t capacity,
          uint32_t storageRows, uint32_t logitRows = 1, BlockDraft *draft = nullptr)
      : backend_(backend), weights_(weights), draft_(draft),
        operators_(backend.capabilities(), ops::Int8Activations::Off),
        pages_(backend, [](uint64_t, const auto &allocate) -> metal::AllocationResult {
          allocate();
          return {};
        }, layout, ceilDiv(ceilDiv(capacity, kv::kPageTokens), kExtentPages) * kExtentPages, kExtentPages),
        table_(allocate(pages_.pageCount() * sizeof(SplashKvPage), "prompt-table")),
        tokens_(allocate(storageRows * sizeof(uint32_t), "tokens")),
        lane_(allocate(sizeof(SplashLaneRows), "lane")), hidden_(allocate(rowBytes(kHidden, storageRows), "hidden")),
        normalized_(allocate(rowBytes(kHidden, storageRows), "normalized")),
        update_(allocate(rowBytes(kHidden, storageRows), "update")),
        packed_(allocate(rowBytes((kQueryHeads + kGlobalHeads) * kGlobalDim, storageRows), "qkv")),
        queries_(allocate(rowBytes(kQueryHeads * kGlobalDim, storageRows), "queries")),
        attended_(allocate(rowBytes(kQueryHeads * kGlobalDim, storageRows), "attention")),
        gate_(allocate(rowBytes(kIntermediate, storageRows), "ffn-gate")),
        up_(allocate(rowBytes(kIntermediate, storageRows), "ffn-up")), logitRows_(logitRows),
        logits_(allocate(uint64_t{kVocab} * sizeof(float) *
                             operators_.linear().storageRows(logitRows, weights.logits.shape()), "logits")) {
    if (logitRows == 0 || logitRows > storageRows) throw std::invalid_argument("invalid Gemma 4 logit rows");
    ops::LinearScratchSize scratch;
    for (const auto &w : weights.layers)
      for (const auto *projection : {&w.qkv, &w.attentionOut, &w.gate, &w.up, &w.down})
        scratch.include(operators_.linear().scratchSize(projection->shape(), storageRows));
    scratch.include(operators_.linear().scratchSize(weights.logits.shape(), logitRows));
    scratch_ = {allocate(scratch.input, "linear-input"), allocate(scratch.sums, "linear-sums"),
                allocate(scratch.partials, "linear-partials"), allocate(scratch.counters, "linear-counters")};
    if (scratch_.counters) std::memset(scratch_.counters.contents(), 0, scratch_.counters.sizeBytes());
    const auto global = layout.layerLayout(static_cast<uint32_t>(std::countr_zero(~layout.windowedLayers)));
    capacity_ = operators_.sequenceAttentionWorkspace(kQueryHeads, global);
    partials_ = allocate(capacity_.partialsBytes, "attention-partials");
    statistics_ = allocate(capacity_.statisticsBytes, "attention-statistics");
    for (uint32_t variant = 0; variant != rope_.size(); ++variant) {
      const uint32_t dim = variant == 0 ? kGlobalDim : kLocalDim;
      const float theta = variant == 0 ? kGlobalTheta : kLocalTheta;
      auto &rope = rope_[variant];
      rope.cosine = allocate(uint64_t{storageRows} * dim / 2 * sizeof(float), "rope-cosine");
      rope.sine = allocate(rope.cosine.sizeBytes(), "rope-sine");
      for (uint32_t i = 0; i < dim / 2; ++i) {
        const float factor = variant == 0 ? weights.ropeFactors[i] : 1.0f;
        rope.frequency.push_back(std::pow(theta, -2.0f * float(i) / float(dim)) / factor);
      }
    }
  }

  // Writes the rows' keys and values at the positions after the previous
  // call's (or rewind's), and returns the raw logits, before the softcap, of
  // the last `logitRows` rows, row-major.
  std::span<const float> run(std::span<const uint32_t> tokens, uint32_t logitRows) {
    const uint32_t rows = static_cast<uint32_t>(tokens.size());
    if (logitRows > std::min(rows, logitRows_)) throw std::invalid_argument("too many Gemma 4 logit rows");
    const uint32_t needed = ceilDiv(position_ + rows, kv::kPageTokens);
    if (needed > logicalPages_.size()) {
      for (uint32_t extent = ceilDiv(static_cast<uint32_t>(logicalPages_.size()), kExtentPages);
           extent < ceilDiv(needed, kExtentPages); ++extent)
        if (!pages_.allocateExtent(extent)) throw std::runtime_error("could not allocate the Gemma 4 KV extent");
      for (uint32_t page = static_cast<uint32_t>(logicalPages_.size()); page < needed; ++page)
        logicalPages_.push_back(page);
      pages_.writeEntries(logicalPages_, 0, table_);
    }
    std::memcpy(tokens_.contents(), tokens.data(), tokens.size_bytes());
    *static_cast<SplashLaneRows *>(lane_.contents()) = {0, rows};
    prepareRoPE(rows);
    const std::array tables{table_};
    const std::array sequences{ops::PagedAttention::chunkParams(position_, rows,
        ceilDiv(rows, kv::kPageTokens) * kv::kPageTokens, needed)};
    const ops::AttentionRows attentionRows{lane_, rows};
    const ops::Linear &linear = operators_.linear();
    metal::CommandGraph graph;
    ops::Embedding::add(graph, tokens_, weights_.embedding, hidden_, rows);
    ops::scaleRows(graph, hidden_, kHidden * rows, std::sqrt(float(kHidden)));
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      const auto &w = weights_.layers[layer];
      const auto shape = pages_.layout().layerLayout(layer);
      const auto placement = pages_.layers()[layer];
      const ops::AttentionVariant variant{.windowTokens = shape.windowTokens};
      const auto &rope = rope_[shape.windowTokens != 0];
      ops::Normalization::addRms(graph, hidden_, w.attention, normalized_, kHidden, rows);
      linear.addProjection(graph, normalized_, w.qkv, packed_, rows, scratch_);
      ops::PagedAttention::addProjection(graph, packed_, w.query, w.key, rope.cosine, rope.sine,
          queries_, placement, tables, sequences, attentionRows, kQueryHeads, shape, variant);
      const auto plan = operators_.attention(sequences, kQueryHeads, shape, capacity_, variant);
      const std::array histories{ops::AttentionHistory{placement, table_}};
      ops::PagedAttention::addAttention(graph, plan, queries_, partials_, statistics_, sequences, histories,
                                        attentionRows, {{}, attended_});
      linear.addProjection(graph, attended_, w.attentionOut, update_, rows, scratch_);
      ops::Normalization::addRms(graph, update_, w.postAttention, normalized_, kHidden, rows);
      ops::addScaledResidual(graph, hidden_, normalized_, kHidden * rows, 1.0f);

      ops::Normalization::addRms(graph, hidden_, w.ffn, normalized_, kHidden, rows);
      linear.addProjection(graph, normalized_, w.gate, gate_, rows, scratch_);
      linear.addProjection(graph, normalized_, w.up, up_, rows, scratch_);
      ops::addGeluGate(graph, gate_, up_, kIntermediate * rows);
      linear.addProjection(graph, gate_, w.down, update_, rows, scratch_);
      ops::Normalization::addRms(graph, update_, w.postFfn, normalized_, kHidden, rows);
      ops::addScaledResidual(graph, hidden_, normalized_, kHidden * rows, w.outputScale);
      if (draft_) {
        const auto &captured = draft_->layout().targetLayers;
        const auto at = std::ranges::find(captured, layer);
        if (at != captured.end())
          ops::RowCopy::add(graph, hidden_, {0, kHidden, 0}, draft_->context(),
                            {0, draft_->layout().contextWidth(), static_cast<uint32_t>(at - captured.begin()) * kHidden},
                            rows, kHidden);
      }
    }
    if (draft_) draft_->addContext(graph, rows, position_);
    if (logitRows) {
      auto last = backend_.view(hidden_, (rows - logitRows) * rowBytes(kHidden, 1), rowBytes(kHidden, logitRows));
      ops::Normalization::addRms(graph, std::move(last), weights_.finalNorm, normalized_, kHidden, logitRows);
      linear.addProjection(graph, normalized_, weights_.logits, logits_, logitRows, scratch_);
    }
    static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
    position_ += rows;
    return {static_cast<const float *>(logits_.contents()), size_t{kVocab} * logitRows};
  }

  // Drops the positions from `position` on: the next call writes over their
  // keys and values, and no attention reads past its own rows.
  void rewind(uint32_t position) {
    if (position > position_) throw std::logic_error("a Gemma 4 rewind past the written positions");
    position_ = position;
  }

  // The draft's block after `anchor`, the pending token at the position
  // after the written ones: its raw logits of the tokens after the anchor.
  std::span<const float> draft(uint32_t anchor) {
    metal::CommandGraph graph;
    draft_->addBlock(graph, anchor, position_, weights_.embedding, weights_.logits);
    static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
    return draft_->logits();
  }

  [[nodiscard]] bool drafts() const noexcept { return draft_ != nullptr; }
  [[nodiscard]] uint32_t position() const noexcept { return position_; }
  [[nodiscard]] uint64_t kvBytes() const noexcept { return pages_.actualAllocatedBytes(); }

private:
  struct Rotary final {
    metal::MetalBuffer cosine, sine;
    std::vector<float> frequency;
  };

  [[nodiscard]] metal::MetalBuffer allocate(uint64_t bytes, std::string_view name) {
    if (!bytes) return {};
    return backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, std::string(name));
  }

  void prepareRoPE(uint32_t rows) {
    for (const auto &rope : rope_) {
      auto *cos = static_cast<float *>(rope.cosine.contents()), *sin = static_cast<float *>(rope.sine.contents());
      for (uint32_t row = 0; row < rows; ++row) {
        for (float frequency : rope.frequency) {
          const float angle = float(position_ + row) * frequency;
          *cos++ = std::cos(angle);
          *sin++ = std::sin(angle);
        }
      }
    }
  }

  metal::MetalBackend &backend_;
  const Weights &weights_;
  BlockDraft *draft_;
  ops::ExecutionPlans operators_;
  kv::PageStorage pages_;
  metal::MetalBuffer table_, tokens_, lane_, hidden_, normalized_, update_, packed_, queries_, attended_, gate_, up_;
  uint32_t logitRows_;
  metal::MetalBuffer logits_, partials_, statistics_;
  ops::LinearScratch scratch_;
  ops::AttentionWorkspace capacity_;
  std::array<Rotary, 2> rope_;
  std::vector<uint32_t> logicalPages_;
  uint32_t position_ = 0;
};

// The block of the DFlash draft released for this checkpoint
// (z-lab/gemma-4-31B-it-DFlash): a forward verifies its anchor and 15 drafted
// tokens.
constexpr uint32_t kDraftBlock = 16;

BlockDraftLayout draftLayout(const std::filesystem::path &checkpoint) {
  BlockDraftLayout layout = readBlockDraftLayout(checkpoint, kLayers, kHidden, kVocab);
  if (layout.blockSize != kDraftBlock) throw std::invalid_argument("the Gemma 4 draft fills blocks of 16 tokens");
  return layout;
}

double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

float softcap(float logit) { return kSoftcap * std::tanh(logit / kSoftcap); }
bool stops(uint32_t token) { return token == kEndOfSequence || token == kEndOfTurn; }

// The token a row's softcapped logits rank first, the lowest ID among equal
// ones, without the softcap of every logit: tanh does not decrease, so a
// logit at or below a bound whose softcap is below the maximum's cannot equal
// it.
uint32_t greedy(std::span<const float> logits) {
  float top = -std::numeric_limits<float>::infinity();
  for (float logit : logits) {
    if (!std::isfinite(logit)) throw std::runtime_error("Gemma 4 produced non-finite logits");
    top = std::max(top, logit);
  }
  const float best = softcap(top);
  float margin = 1.0f / 64;
  while (softcap(top - margin) >= best) margin *= 2;
  const float bound = top - margin;
  for (size_t token = 0;; ++token)
    if (logits[token] > bound && softcap(logits[token]) == best) return static_cast<uint32_t>(token);
}

// Greedy decode from the prompt's last logits. Each forward takes the
// pending token and the tokens the forward's draft selects after it, or
// without one, with lookup on, the up to `lookupTokens` tokens that followed
// the latest earlier occurrence of the request's latest tokens
// (LookupIndex). Row i's logits select the token after row i, so the forward
// keeps the proposals its own selections repeat and the next selection after
// them; the keys and values of the rest are rewound.
void decode(Forward &forward, std::span<const float> promptLogits, std::span<const uint32_t> prompt,
            uint32_t newTokens, uint32_t lookupTokens, AutoregressiveResult &result) {
  LookupIndex lookup;
  if (lookupTokens) lookup.append(prompt);
  uint32_t token = greedy(promptLogits);
  std::vector<uint32_t> rows;
  while (!stops(token)) {
    result.tokens.push_back(token);
    const size_t left = newTokens - result.tokens.size();
    if (!left) return;
    rows.assign(1, token);
    if (forward.drafts()) {
      const auto logits = forward.draft(token);
      const size_t count = std::min(logits.size() / kVocab, left - 1);
      for (size_t row = 0; row < count; ++row) rows.push_back(greedy(logits.subspan(row * kVocab, kVocab)));
    } else if (lookupTokens) {
      lookup.append(std::span(&token, 1));
      const auto proposals = lookup.proposals(static_cast<uint32_t>(std::min<size_t>(lookupTokens, left - 1)));
      rows.insert(rows.end(), proposals.begin(), proposals.end());
    }
    const uint32_t start = forward.position(), count = static_cast<uint32_t>(rows.size());
    const auto logits = forward.run(rows, count);
    ++result.decodeCycles;
    result.decodeRows += count;
    result.proposedTokens += count - 1;
    for (uint32_t row = 0;; ++row) {
      token = greedy(logits.subspan(size_t{row} * kVocab, kVocab));
      if (row + 1 == count || token != rows[row + 1]) {
        forward.rewind(start + row + 1);
        break;
      }
      if (stops(token)) return;
      ++result.acceptedTokens;
      result.tokens.push_back(token);
      if (lookupTokens) lookup.append(std::span(&token, 1));
    }
  }
}

// Prefills the prompt in chunks of up to kPrefillRows rows and decodes up to
// `newTokens` tokens into `result`; returns the KV bytes it allocated.
uint64_t generate(metal::MetalBackend &backend, const Weights &weights, kv::Layout layout,
                  std::span<const uint32_t> prompt, uint32_t newTokens, AutoregressiveOptions options,
                  AutoregressiveResult &result, BlockDraft *draft = nullptr) {
  if (draft) options.lookupTokens = 0;
  const uint32_t capacity = static_cast<uint32_t>(prompt.size()) + newTokens;
  const uint32_t prefillRows = std::min(kPrefillRows, static_cast<uint32_t>(prompt.size()));
  const uint32_t storageRows = ceilDiv(prefillRows, GGUF_PREFILL_ROWS) * GGUF_PREFILL_ROWS;
  const uint32_t verifyRows = draft ? draft->layout().blockSize : 1 + options.lookupTokens;
  Forward forward(backend, weights, layout, capacity, storageRows, verifyRows, draft);
  const auto prefill = Clock::now();
  std::span<const float> logits;
  for (auto rest = prompt; !rest.empty();) {
    const auto chunk = rest.first(std::min<size_t>(prefillRows, rest.size()));
    rest = rest.subspan(chunk.size());
    logits = forward.run(chunk, rest.empty() ? 1 : 0);
  }
  result.prefillSeconds = elapsed(prefill);
  result.promptLogits.resize(logits.size());
  std::ranges::transform(logits, result.promptLogits.begin(), softcap);
  const auto decoding = Clock::now();
  if (newTokens) decode(forward, logits, prompt, newTokens, options.lookupTokens, result);
  result.decodeSeconds = elapsed(decoding);
  return forward.kvBytes();
}

} // namespace

void checkAutoregressiveConfiguration(const std::filesystem::path &gguf) {
  static_cast<void>(planWeights(gguf, GgufContents::Headers));
}

AutoregressiveResult generateAutoregressive(const std::filesystem::path &gguf, const std::filesystem::path &library,
    std::span<const uint32_t> prompt, uint32_t newTokens, AutoregressiveOptions options) {
  if (prompt.empty() || prompt.size() > kAutoregressiveContextLimit ||
      newTokens > kAutoregressiveContextLimit - prompt.size())
    throw std::invalid_argument("prompt and generation must fit the 4096-token autoregressive context");
  if (std::ranges::any_of(prompt, [](uint32_t id) { return id >= kVocab; }))
    throw std::invalid_argument("a prompt token is outside the Gemma 4 vocabulary");
  if (options.lookupTokens > kMaximumLookupTokens)
    throw std::invalid_argument("too many Gemma 4 lookup tokens a forward");
  const auto loadStart = Clock::now();
  auto plan = planWeights(gguf, GgufContents::Files);
  metal::MetalBackend backend(library);
  const uint32_t capacity = static_cast<uint32_t>(prompt.size()) + newTokens;
  constexpr uint64_t workspaceReserve = uint64_t{1} << 30;
  const uint32_t extents = ceilDiv(ceilDiv(capacity, kv::kPageTokens), kExtentPages);
  if (plan.bytes + extents * kExtentPages * plan.kv.bytesPerModelPage() + workspaceReserve >
      backend.capabilities().recommendedMaxWorkingSetBytes)
    throw std::runtime_error("Gemma 4 weights and KV exceed the device's recommended working set");
  WeightImages images(backend, {});
  const auto weights = loadWeights(backend, images, plan);
  std::unique_ptr<BlockDraft> draft;
  if (!options.draft.empty())
    draft = std::make_unique<BlockDraft>(backend, images, options.draft, draftLayout(options.draft), capacity + kDraftBlock,
                                         kPrefillRows);
  AutoregressiveResult result;
  result.loadSeconds = elapsed(loadStart);
  const uint64_t kvBytes = generate(backend, weights, plan.kv, prompt, newTokens, options, result, draft.get());
  result.weightAndKvBytes = plan.bytes + kvBytes + (draft ? draft->bytes() : 0);
  return result;
}

} // namespace splash::model
