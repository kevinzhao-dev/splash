#include "ops/BlockDraftAttention.hpp"

#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint64_t kBf16 = sizeof(uint16_t);

void requireCache(const metal::MetalBuffer &keys, const metal::MetalBuffer &values, uint32_t kvHeads,
                  uint32_t capacity) {
  const uint64_t bytes = uint64_t{kvHeads} * capacity * BlockDraftAttention::kHeadDimension * kBf16;
  requireBytes(keys, bytes, "block draft keys");
  requireBytes(values, bytes, "block draft values");
}

} // namespace

void BlockDraftAttention::addPrepare(metal::CommandGraph &graph, metal::MetalBuffer packed,
                                     const NormWeights &queryNorm, const NormWeights &keyNorm,
                                     metal::MetalBuffer cosines, metal::MetalBuffer sines, metal::MetalBuffer queries,
                                     metal::MetalBuffer keys, metal::MetalBuffer values,
                                     const SplashBlockDraftRows &rows) {
  if (!rows.rows || !rows.kv_heads || rows.first_position + uint64_t{rows.rows} > rows.capacity)
    throw std::invalid_argument("block draft rows do not fit the cache");
  if (keyNorm.float32 || (rows.query_heads && queryNorm.float32))
    throw std::invalid_argument("block draft norms are BF16");
  const uint32_t heads = rows.query_heads + rows.kv_heads;
  requireBytes(packed, uint64_t{rows.rows} * (heads + rows.kv_heads) * kHeadDimension * kBf16, "block draft rows");
  requireBytes(keyNorm.buffer, kHeadDimension * kBf16, "block draft key norm");
  requireBytes(cosines, uint64_t{rows.rows} * kHeadDimension / 2 * sizeof(float), "block draft RoPE cosines");
  requireBytes(sines, uint64_t{rows.rows} * kHeadDimension / 2 * sizeof(float), "block draft RoPE sines");
  if (rows.query_heads) {
    requireBytes(queryNorm.buffer, kHeadDimension * kBf16, "block draft query norm");
    requireBytes(queries, uint64_t{rows.rows} * rows.query_heads * kHeadDimension * kBf16, "block draft queries");
  }
  requireCache(keys, values, rows.kv_heads, rows.capacity);
  // Context rows write no queries; a dispatch binds no empty buffer.
  if (!rows.query_heads) queries = cosines;
  graph.add("block_draft_prepare",
            {std::move(packed), rows.query_heads ? queryNorm.buffer : keyNorm.buffer, keyNorm.buffer,
             std::move(cosines), std::move(sines), std::move(queries), std::move(keys), std::move(values)},
            rows, {rows.rows, heads, 1}, {32, 1, 1});
}

void BlockDraftAttention::addAttention(metal::CommandGraph &graph, metal::MetalBuffer queries,
                                       metal::MetalBuffer keys, metal::MetalBuffer values,
                                       metal::MetalBuffer output, const SplashBlockDraftAttention &attention) {
  if (!attention.rows || !attention.kv_heads || attention.query_heads % attention.kv_heads ||
      attention.query_heads / attention.kv_heads > 32 ||
      attention.first_position + uint64_t{attention.rows} > attention.capacity)
    throw std::invalid_argument("invalid block draft attention");
  const uint64_t bytes = uint64_t{attention.rows} * attention.query_heads * kHeadDimension * kBf16;
  requireBytes(queries, bytes, "block draft queries");
  requireBytes(output, bytes, "block draft attention rows");
  requireCache(keys, values, attention.kv_heads, attention.capacity);
  graph.add("block_draft_attention", {std::move(queries), std::move(keys), std::move(values), std::move(output)},
            attention, {attention.kv_heads, attention.rows, 1},
            {32 * (attention.query_heads / attention.kv_heads), 1, 1});
}

} // namespace splash::ops
