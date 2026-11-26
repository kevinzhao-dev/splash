#include "metal/abi/BlockDraft.h"

using namespace metal;

constant uint kDimension = SPLASH_BLOCK_DRAFT_HEAD_DIMENSION;
constant uint kHalf = kDimension / 2;
// A head row's bfloat4 groups, one a lane.
constant uint kGroups = kDimension / 4;

// One simdgroup a (row, head): RMS-normalizes the head's values in float,
// rounds them to bf16 and scales them by the bf16 weights as a BF16
// checkpoint's norm does, then rotates the pairs (i, i + kHalf) by the row's
// angles. A lane holds dimensions lane and lane + 32 of each half.
kernel void block_draft_prepare(device const bfloat *packed [[buffer(0)]],
                                device const bfloat *queryNorm [[buffer(1)]],
                                device const bfloat *keyNorm [[buffer(2)]],
                                device const float *cosines [[buffer(3)]],
                                device const float *sines [[buffer(4)]],
                                device bfloat *queries [[buffer(5)]],
                                device bfloat *keys [[buffer(6)]],
                                device bfloat *values [[buffer(7)]],
                                constant SplashBlockDraftRows &params [[buffer(8)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.x, head = group.y;
  const uint heads = params.query_heads + params.kv_heads;
  if (row >= params.rows || head >= heads) return;
  const bool query = head < params.query_heads;
  device const bfloat *source = packed + (uint64_t(row) * (heads + params.kv_heads) + head) * kDimension;
  device const bfloat *weights = query ? queryNorm : keyNorm;
  const uint dimensions[4] = {lane, lane + 32, lane + kHalf, lane + kHalf + 32};
  float x[4];
  for (uint i = 0; i < 4; ++i) x[i] = float(source[dimensions[i]]);
  const float squares = simd_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
  const float inverse = rsqrt(squares / float(kDimension) + params.epsilon);
  for (uint i = 0; i < 4; ++i)
    x[i] = float(bfloat(float(bfloat(x[i] * inverse)) * float(weights[dimensions[i]])));
  device const float *cosine = cosines + uint64_t(row) * kHalf;
  device const float *sine = sines + uint64_t(row) * kHalf;
  const float c0 = cosine[lane], s0 = sine[lane], c1 = cosine[lane + 32], s1 = sine[lane + 32];
  const float rotated[4] = {x[0] * c0 - x[2] * s0, x[1] * c1 - x[3] * s1,
                            x[2] * c0 + x[0] * s0, x[3] * c1 + x[1] * s1};
  device bfloat *destination;
  if (query) {
    destination = queries + (uint64_t(row) * params.query_heads + head) * kDimension;
  } else {
    const uint64_t slot =
        (uint64_t(head - params.query_heads) * params.capacity + params.first_position + row) * kDimension;
    destination = keys + slot;
    device const bfloat *value = source + params.kv_heads * kDimension;
    for (uint i = 0; i < 4; ++i) values[slot + dimensions[i]] = value[dimensions[i]];
  }
  for (uint i = 0; i < 4; ++i) destination[dimensions[i]] = bfloat(rotated[i]);
}

// A threadgroup a (KV head, query row), a simdgroup each of the KV head's
// query heads, a lane each bfloat4 of a head row: an online softmax over the
// row's visible keys, four at a time.
kernel void block_draft_attention(device const bfloat *queries [[buffer(0)]],
                                  device const bfloat *keys [[buffer(1)]],
                                  device const bfloat *values [[buffer(2)]],
                                  device bfloat *output [[buffer(3)]],
                                  constant SplashBlockDraftAttention &params [[buffer(4)]],
                                  uint2 group [[threadgroup_position_in_grid]],
                                  uint simdgroup [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  const uint kvHead = group.x, row = group.y;
  const uint perKvHead = params.query_heads / params.kv_heads;
  if (row >= params.rows || kvHead >= params.kv_heads || simdgroup >= perKvHead) return;
  const uint head = kvHead * perKvHead + simdgroup;
  const uint position = params.first_position + row;
  const uint last = params.causal ? position : params.first_position + params.rows - 1;
  const uint first = params.window && position + 1 > params.window ? position + 1 - params.window : 0;
  const uint64_t offset = (uint64_t(row) * params.query_heads + head) * kDimension + lane * 4;
  const float4 query = float4(*(device const bfloat4 *)(queries + offset)) * (params.scale * M_LOG2E_F);
  const uint64_t history = uint64_t(kvHead) * params.capacity * kGroups + lane;
  device const bfloat4 *keyRows = (device const bfloat4 *)keys + history;
  device const bfloat4 *valueRows = (device const bfloat4 *)values + history;
  float maximum = -1.0e30f, total = 0.0f;
  float4 sum = 0.0f;
  uint key = first;
  for (; key + 4 <= last + 1; key += 4) {
    float scores[4];
    for (uint i = 0; i < 4; ++i) scores[i] = simd_sum(dot(query, float4(keyRows[uint64_t(key + i) * kGroups])));
    const float next = max(maximum, max(max(scores[0], scores[1]), max(scores[2], scores[3])));
    const float correction = exp2(maximum - next);
    total *= correction;
    sum *= correction;
    for (uint i = 0; i < 4; ++i) {
      const float weight = exp2(scores[i] - next);
      total += weight;
      sum += weight * float4(valueRows[uint64_t(key + i) * kGroups]);
    }
    maximum = next;
  }
  for (; key <= last; ++key) {
    const float score = simd_sum(dot(query, float4(keyRows[uint64_t(key) * kGroups])));
    const float next = max(maximum, score);
    const float correction = exp2(maximum - next), weight = exp2(score - next);
    total = total * correction + weight;
    sum = sum * correction + weight * float4(valueRows[uint64_t(key) * kGroups]);
    maximum = next;
  }
  *(device bfloat4 *)(output + offset) = bfloat4(sum / total);
}
