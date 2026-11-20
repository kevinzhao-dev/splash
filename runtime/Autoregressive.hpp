#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace splash::model {

inline constexpr uint32_t kAutoregressiveContextLimit = 4096;
// A forward's lookup proposals at most, beside its pending token: the decode
// tiles hold the rows.
inline constexpr uint32_t kMaximumLookupTokens = 31;
// The command's default: a forward of up to 16 rows costs little more than
// one of a single row on the M5 Pro (docs/gemma4-31b-port.md).
inline constexpr uint32_t kDefaultLookupTokens = 15;

struct AutoregressiveOptions final {
  // The most tokens each forward verifies from prompt lookup beside the
  // pending token; 0 decodes one token a forward.
  uint32_t lookupTokens = 0;
  // A DFlash v1 draft checkpoint (BlockDraft) whose blocks each forward
  // verifies in place of lookup proposals; empty for none.
  std::filesystem::path draft;
};

struct AutoregressiveResult final {
  std::vector<uint32_t> tokens;
  std::vector<float> promptLogits;
  double loadSeconds = 0;
  double prefillSeconds = 0;
  double decodeSeconds = 0;
  // Forwards after the prompt's, the rows they computed, and of those the
  // lookup proposals and those the target selected.
  uint32_t decodeCycles = 0;
  uint32_t decodeRows = 0;
  uint32_t proposedTokens = 0;
  uint32_t acceptedTokens = 0;
  uint64_t weightAndKvBytes = 0;
};

// Check a GGUF's metadata and tensor table, including a header-only copy.
void checkAutoregressiveConfiguration(const std::filesystem::path &gguf);
[[nodiscard]] AutoregressiveResult generateAutoregressive(const std::filesystem::path &gguf,
    const std::filesystem::path &library, std::span<const uint32_t> prompt, uint32_t newTokens,
    AutoregressiveOptions options = {});

} // namespace splash::model
