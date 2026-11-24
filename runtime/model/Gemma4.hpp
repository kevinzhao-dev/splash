#pragma once

// Gemma 4 31B (GGUF architecture gemma4, Google's QAT Q4_0 checkpoint): a
// decoder of 60 layers of 5376, each an attention and a gated FFN, every one
// of them between an RMS norm of its input and one of its output. What it
// changes from the targets the runtime serves:
//  - 50 sliding layers attend the 1024 keys that end at a row through 16 KV
//    heads of 256, rotated by RoPE theta 1e4; 10 global layers attend every
//    key through 4 KV heads of 512 whose keys are their values too, the
//    values RMS-normalized without a weight, rotated by RoPE theta 1e6 over
//    frequencies its rope_freqs divide. 32 query heads; queries and keys are
//    RMS-normalized a head each, so scores are not scaled.
//  - Its FFN is GELU (tanh) of the gate times the up projection, 21504 wide;
//    a layer's output is its residual sum times the layer's output scale.
//  - Its Q6_K token table is its vocabulary projection too; a row it gathers
//    is scaled by sqrt(5376), and a logit capped at 30 tanh(x / 30).
//  - Tokens 1 and 106 end a reply.
// It runs one sequence of up to 4096 tokens with BF16 KV and greedy
// selection, apart from the engine's runtime. A DFlash v1 draft
// (BlockDraft) of 16-token blocks may propose each forward's rows.

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace splash::model {

inline constexpr uint32_t kGemma4ContextLimit = 4096;
inline constexpr uint32_t kGemma4Vocabulary = 262144;
// A forward's lookup proposals at most beside its pending token: the decode
// tiles hold its rows.
inline constexpr uint32_t kGemma4MaximumLookupTokens = 31;
// A forward of 16 rows costs 11% more than one of a single row on an M5 Pro
// after a 2,048-token prefix; 24 rows cost 37% more.
inline constexpr uint32_t kGemma4DefaultLookupTokens = 15;

// One request's greedy tokens and what they cost.
struct Gemma4Generation final {
  std::vector<uint32_t> tokens;
  // The prompt's last row's softcapped logits.
  std::vector<float> promptLogits;
  double prefillSeconds = 0;
  double decodeSeconds = 0;
  // Forwards after the prompt's, the rows they computed, and of those the
  // proposals and the ones the target selected.
  uint32_t decodeCycles = 0;
  uint32_t decodeRows = 0;
  uint32_t proposedTokens = 0;
  uint32_t acceptedTokens = 0;
  uint64_t kvBytes = 0;
};

// The checkpoint at `gguf`, and with `draft` its DFlash v1 draft checkpoint,
// loaded onto the device of `library`'s Metal backend.
class Gemma4 final {
public:
  // Checks a GGUF's metadata and tensor table, a header-only copy too.
  static void check(const std::filesystem::path &gguf);

  Gemma4(const std::filesystem::path &gguf, const std::filesystem::path &library,
         const std::filesystem::path &draft = {});
  ~Gemma4();
  Gemma4(const Gemma4 &) = delete;
  Gemma4 &operator=(const Gemma4 &) = delete;

  // The prompt and up to `newTokens` tokens, each forward verifying the
  // draft's block after its pending token or, without a draft, up to
  // `lookupTokens` prompt-lookup tokens.
  [[nodiscard]] Gemma4Generation generate(std::span<const uint32_t> prompt, uint32_t newTokens,
                                          uint32_t lookupTokens);

  // Forwards of one sequence of up to `capacity` positions, each of up to
  // the prefill chunk's rows, the last `logitRows` of them at most with
  // logits: what generate() runs, for tests and benchmarks. With `drafted`,
  // and a draft loaded, its forwards write the draft's context too.
  class Sequence final {
  public:
    ~Sequence();
    Sequence(Sequence &&) noexcept;
    // Writes the rows' keys and values after the positions written, and
    // returns the raw logits (before the softcap) of the last `logitRows`.
    std::span<const float> forward(std::span<const uint32_t> tokens, uint32_t logitRows);
    // Drops the positions from `position` on.
    void rewind(uint32_t position);
    // The draft's raw logits of the 15 tokens after `anchor`, the pending
    // token at the next position.
    std::span<const float> draft(uint32_t anchor);
    [[nodiscard]] uint32_t position() const noexcept;
    // The GPU time of the last command.
    [[nodiscard]] double gpuSeconds() const noexcept;

  private:
    friend class Gemma4;
    struct Impl;
    explicit Sequence(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };
  [[nodiscard]] Sequence sequence(uint32_t capacity, uint32_t logitRows, bool drafted = true);

  // The token softcapped logits rank first, the lowest among equal ones.
  [[nodiscard]] static uint32_t select(std::span<const float> logits);
  [[nodiscard]] bool drafts() const noexcept;
  [[nodiscard]] uint64_t weightBytes() const noexcept;
  [[nodiscard]] metal::MetalBackend &backend() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
