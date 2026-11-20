#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A DFlash block draft's attention (ops/BlockDraftAttention.hpp): heads of
// this many dimensions, whose RoPE rotates the pairs (i, i + half) of every
// head by a row's angles, half of them a row in its cosine and sine tables.
#define SPLASH_BLOCK_DRAFT_HEAD_DIMENSION 128u

// Packed projection rows at positions first_position on: each row's
// query_heads query heads (none for context rows), then kv_heads key heads,
// then kv_heads value heads. Their normalized, rotated queries go to the
// query rows, their normalized, rotated keys and their values to the cache,
// capacity positions a KV head.
struct SplashBlockDraftRows {
  uint32_t rows;
  uint32_t first_position;
  uint32_t query_heads;
  uint32_t kv_heads;
  uint32_t capacity;
  float epsilon;
};

// Query rows at positions first_position on, each over the cached keys and
// values from position 0 to the last query row's: up to its own position
// when causal, and with a window, only those less than window positions
// before it.
struct SplashBlockDraftAttention {
  uint32_t rows;
  uint32_t first_position;
  uint32_t query_heads;
  uint32_t kv_heads;
  uint32_t capacity;
  uint32_t causal;
  uint32_t window;
  float scale;
};
