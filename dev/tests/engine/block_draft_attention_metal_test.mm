#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/BlockDraftAttention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// A DFlash block draft's attention (ops::BlockDraftAttention) through the
// production kernels: context rows' keys and values, then a block's rows,
// stored into the cache by the prepare kernel, and the block's attention
// over them causally within a window and over the whole block, each against
// an fp64 reference of the same bf16 inputs.

namespace {

using namespace splash;
using test::require;

constexpr uint32_t kDimension = ops::BlockDraftAttention::kHeadDimension, kHalf = kDimension / 2;
constexpr uint32_t kQueryHeads = 16, kKvHeads = 2, kGroup = kQueryHeads / kKvHeads;
constexpr uint32_t kContext = 71, kBlock = 16, kCapacity = 96, kWindow = 40;
constexpr float kEpsilon = 1e-6F;

uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  bits += 0x7FFF + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float fromBf16(uint16_t bits) {
  const uint32_t wide = uint32_t{bits} << 16;
  float value;
  std::memcpy(&value, &wide, 4);
  return value;
}

float roundBf16(double value) { return fromBf16(toBf16(static_cast<float>(value))); }

struct Rows final {
  uint32_t rows = 0, position = 0, queryHeads = 0;
  std::vector<uint16_t> packed;
  std::vector<float> cosines, sines;
};

Rows randomRows(std::mt19937 &random, uint32_t rows, uint32_t position, uint32_t queryHeads) {
  std::normal_distribution<float> value(0.0F, 1.5F);
  Rows r{rows, position, queryHeads, {}, {}, {}};
  r.packed.resize(uint64_t{rows} * (queryHeads + 2 * kKvHeads) * kDimension);
  for (uint16_t &bits : r.packed) bits = toBf16(value(random));
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t pair = 0; pair < kHalf; ++pair) {
      const float angle = float(position + row) / std::pow(10000.0F, 2.0F * pair / kDimension);
      r.cosines.push_back(std::cos(angle));
      r.sines.push_back(std::sin(angle));
    }
  return r;
}

// One head of a packed row as the prepare kernel writes it: RMS-normalized
// and rounded to bf16, times the bf16 weight and rounded, then rotated.
std::vector<double> prepared(const Rows &r, uint32_t row, uint32_t head, const std::vector<uint16_t> &norm) {
  const uint32_t width = (r.queryHeads + 2 * kKvHeads) * kDimension;
  const uint16_t *x = r.packed.data() + uint64_t{row} * width + head * kDimension;
  double squares = 0;
  for (uint32_t i = 0; i < kDimension; ++i) squares += double(fromBf16(x[i])) * fromBf16(x[i]);
  const float inverse = 1.0F / std::sqrt(float(squares / kDimension) + kEpsilon);
  std::vector<double> scaled(kDimension), out(kDimension);
  for (uint32_t i = 0; i < kDimension; ++i)
    scaled[i] = roundBf16(double(roundBf16(fromBf16(x[i]) * inverse)) * fromBf16(norm[i]));
  for (uint32_t i = 0; i < kHalf; ++i) {
    const double c = r.cosines[row * kHalf + i], s = r.sines[row * kHalf + i];
    out[i] = scaled[i] * c - scaled[i + kHalf] * s;
    out[i + kHalf] = scaled[i + kHalf] * c + scaled[i] * s;
  }
  return out;
}

metal::MetalBuffer upload(metal::MetalBackend &backend, const void *data, uint64_t bytes) {
  auto buffer = test::sharedBuffer(backend, bytes);
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

void runCase(metal::MetalBackend &backend, bool causal, uint32_t window) {
  std::mt19937 random(causal * 7 + window);
  std::normal_distribution<float> weight(1.0F, 0.2F);
  std::vector<uint16_t> queryNorm(kDimension), keyNorm(kDimension);
  for (auto &bits : queryNorm) bits = toBf16(weight(random));
  for (auto &bits : keyNorm) bits = toBf16(weight(random));
  const Rows context = randomRows(random, kContext, 0, 0), block = randomRows(random, kBlock, kContext, kQueryHeads);
  const uint64_t cacheBytes = uint64_t{kKvHeads} * kCapacity * kDimension * 2;
  auto keys = test::sharedBuffer(backend, cacheBytes), values = test::sharedBuffer(backend, cacheBytes);
  auto queries = test::sharedBuffer(backend, uint64_t{kBlock} * kQueryHeads * kDimension * 2);
  auto output = test::sharedBuffer(backend, queries.sizeBytes());
  const ops::NormWeights qNorm{upload(backend, queryNorm.data(), kDimension * 2)};
  const ops::NormWeights kNorm{upload(backend, keyNorm.data(), kDimension * 2)};
  metal::CommandGraph graph;
  for (const Rows *r : {&context, &block})
    ops::BlockDraftAttention::addPrepare(
        graph, upload(backend, r->packed.data(), r->packed.size() * 2), qNorm, kNorm,
        upload(backend, r->cosines.data(), r->cosines.size() * 4), upload(backend, r->sines.data(), r->sines.size() * 4),
        r->queryHeads ? queries : metal::MetalBuffer{}, keys, values,
        {r->rows, r->position, r->queryHeads, kKvHeads, kCapacity, kEpsilon});
  const float scale = 1.0F / std::sqrt(float(kDimension));
  ops::BlockDraftAttention::addAttention(graph, queries, keys, values, output,
                                         {kBlock, kContext, kQueryHeads, kKvHeads, kCapacity, causal, window, scale});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  // The reference keys and values of every position, from the bf16 values
  // the kernel stored (checked first against the prepared values).
  const auto *storedKeys = static_cast<const uint16_t *>(keys.contents());
  const auto *storedValues = static_cast<const uint16_t *>(values.contents());
  double worstKey = 0;
  for (const Rows *r : {&context, &block})
    for (uint32_t row = 0; row < r->rows; ++row)
      for (uint32_t kv = 0; kv < kKvHeads; ++kv) {
        const auto expected = prepared(*r, row, r->queryHeads + kv, keyNorm);
        const uint64_t slot = (uint64_t{kv} * kCapacity + r->position + row) * kDimension;
        const uint16_t *value = r->packed.data() + uint64_t{row} * (r->queryHeads + 2 * kKvHeads) * kDimension +
                                (r->queryHeads + kKvHeads + kv) * kDimension;
        for (uint32_t i = 0; i < kDimension; ++i) {
          worstKey = std::max(worstKey, std::abs(fromBf16(storedKeys[slot + i]) - expected[i]) /
                                            std::max(1.0, std::abs(expected[i])));
          require(storedValues[slot + i] == value[i], "a stored value differs from its packed row");
        }
      }
  require(worstKey < 1e-2, "stored keys differ from the reference by " + std::to_string(worstKey));

  const auto *attended = static_cast<const uint16_t *>(output.contents());
  double worst = 0;
  for (uint32_t row = 0; row < kBlock; ++row)
    for (uint32_t head = 0; head < kQueryHeads; ++head) {
      const auto query = prepared(block, row, head, queryNorm);
      const uint32_t position = kContext + row, kv = head / kGroup;
      const uint32_t last = causal ? position : kContext + kBlock - 1;
      const uint32_t first = window && position + 1 > window ? position + 1 - window : 0;
      std::vector<double> scores;
      for (uint32_t key = first; key <= last; ++key) {
        double dot = 0;
        for (uint32_t i = 0; i < kDimension; ++i)
          dot += double(roundBf16(query[i])) * fromBf16(storedKeys[(uint64_t{kv} * kCapacity + key) * kDimension + i]);
        scores.push_back(dot * scale);
      }
      const double top = *std::max_element(scores.begin(), scores.end());
      double total = 0;
      for (double &s : scores) total += (s = std::exp(s - top));
      for (uint32_t i = 0; i < kDimension; ++i) {
        double expected = 0;
        for (uint32_t key = first; key <= last; ++key)
          expected += scores[key - first] / total *
                      fromBf16(storedValues[(uint64_t{kv} * kCapacity + key) * kDimension + i]);
        const double actual = fromBf16(attended[(uint64_t{row} * kQueryHeads + head) * kDimension + i]);
        worst = std::max(worst, std::abs(actual - expected) / std::max(1.0, std::abs(expected)));
      }
    }
  require(worst < 2e-2, "attention differs from the reference by " + std::to_string(worst));
  std::cout << (causal ? "causal" : "whole block") << " window " << window << ": key error " << worstKey
            << ", attention error " << worst << '\n';
}

void refusals(metal::MetalBackend &backend) {
  auto buffer = test::sharedBuffer(backend, 1 << 20);
  metal::CommandGraph graph;
  test::rejects([&] {
    ops::BlockDraftAttention::addAttention(graph, buffer, buffer, buffer, buffer,
                                           {kBlock, kCapacity - kBlock + 1, kQueryHeads, kKvHeads, kCapacity, 1, 0, 1});
  }, "invalid block draft attention", "a block past the cache was accepted");
  test::rejects([&] {
    ops::BlockDraftAttention::addPrepare(graph, buffer, {buffer, true}, {buffer, true}, buffer, buffer, buffer, buffer,
                                         buffer, {kBlock, 0, kQueryHeads, kKvHeads, kCapacity, kEpsilon});
  }, "block draft norms are BF16", "F32 norms were accepted");
  require(graph.empty(), "a refused block draft call encoded a dispatch");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: block-draft-attention METALLIB");
    metal::MetalBackend backend(argv[1]);
    refusals(backend);
    runCase(backend, true, kWindow);
    runCase(backend, true, 0);
    runCase(backend, false, 0);
    std::cout << "block_draft_attention_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "block_draft_attention_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
