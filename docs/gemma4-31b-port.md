# Gemma 4 31B text port

- [x] Exercise mixed BF16 paged KV on the M5 Pro with shader validation.
- [x] Pin the GGUF, check its header and SHA-256, and verify tokenization.
- [x] Verify the native forward and greedy generation on real weights.
- [x] Compare prompt logits, ranking and generated IDs against llama.cpp on short prompts and one long prompt.

The supported checkpoint is Google's `gemma-4-31B_q4_0-it.gguf` at
revision `59dde24573e7e61570dba08b18a2e1fe246955ed`. The file has
17,651,001,568 bytes and SHA-256
`179cfb99212709597eae5929112cfca677e1bbf566178b479ae1da0c4772874b`.
Its GGUF header contains 410 Q4_0 tensors, 422 F32 tensors and one Q6_K
embedding tensor shared by lookup and the vocabulary projection. The
tokenizer vocabulary has 262,144 unique strings.
[Pinned file](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf/blob/59dde24573e7e61570dba08b18a2e1fe246955ed/gemma-4-31B_q4_0-it.gguf)

## Build and verify

From the checkout root, use the mounted Apple Metal Toolchain and a local
module cache. `gguf-text` uses the installed llama.cpp and ggml libraries
for tokenization and the independent reference. The native model forward
runs through Splash's operators and kernels. `LLAMA_CPP_PREFIX` and
`GGML_PREFIX` select the local library prefixes for the optional tool.

```sh
metallib=$(xcrun --sdk macosx --find metallib)
make -j4 BUILD_ID_PYTHON=python3.13 \
  METAL="${metallib%/*}/metal -fmodules-cache-path=$PWD/build/metal-module-cache" \
  METALLIB="$metallib" all verify-build-identity build/gguf-text \
  build/engine-tests/attention-variants build/engine-tests/q8-page-storage
```

```sh
MTL_SHADER_VALIDATION=1 build/engine-tests/attention-variants \
  build/engine-tests/production-and-test.metallib mixed
MTL_SHADER_VALIDATION=1 build/engine-tests/attention-variants \
  build/engine-tests/production-and-test.metallib
MTL_SHADER_VALIDATION=1 build/engine-tests/q8-page-storage \
  build/engine-tests/metal-backend.metallib
```

The weight-free mixed test processed one sequence through `PageStorage`,
Q/K/V storage, and split/reduce kernels with 32 query heads in each layer.
Against FP64 using the GPU-stored BF16 inputs, the maximum absolute errors
were `0.000632039` for 16 KV heads × 256 dimensions with a 1024-key window
and `0.00201707` for 4 KV heads × 512 dimensions over the full history. The
test checked zero-based positions 1022–1025, 1054–1056 and 1063 across
Page32 and a physical extent boundary. At position 1056 it also used the
shared table after copying the first page into a Slim extent. The existing
attention variants, Q8 page storage, Q8 paged KV and engine cache tests passed.

```sh
model=install/models/gemma4-31b-qat-59dde24/gemma-4-31B_q4_0-it.gguf
shasum -a 256 "$model"
build/splash autoregressive-check "$model"
python3.13 -m dev.tools.gemma4_parity "$model"
./splash generate --model "$model" --prompt 'What is 17 + 25?' --max-tokens 8
```

The CLI serves a single text prompt and up to 4096 total tokens using BF16
KV and greedy decoding. Each forward verifies the 15 tokens a DFlash draft
fills after the pending one (`--draft DIR`), or without a draft up to 15
prompt-lookup tokens (`--lookup-tokens`, 0 for one token a forward). The text runner allocates Full pages for its
context. Prefill chunks take up to 512 rows, with workspace sized to the
largest chunk in the prompt and rounded to a GGUF projection tile. The
parity harness records the exact token IDs, FP32 prompt
logits and generated IDs for each backend in `build/gemma4-validation`.

## Results on M5 Pro

The same GGUF and prompt tokens ran through Splash and llama.cpp build
10450 (`ece963f41`). The comparison uses the final prompt row's 262,144
softcapped logits. The harness requires equal top-1 and greedy IDs, at
least 8 of the same top-10 IDs, and RMS logit error at most 0.15.

| Prompt | Tokens | Greedy text in both engines | Top-10 overlap | Max absolute error | RMS error |
| --- | ---: | --- | ---: | ---: | ---: |
| Japan capital | 27 | `Tokyo` | 10/10 | 0.230834 | 0.048850 |
| 17 + 25 | 29 | `42` | 10/10 | 0.227675 | 0.061981 |
| Code word before the sliding window | 1123 | `cobalt` | 10/10 | 0.417043 | 0.098820 |

The long case measured approximately 372 prefill tokens/s and 12.6
decode rows/s with 18.49 GiB of weight-image and KV allocations. The
public `splash generate` command produced a complete sentence explaining
why leaves look green. The descriptor check accepted the pinned header
and rejected a header changed to 31 query heads. Existing CPU execution
plans, model configuration, Q8 paged KV, engine cache, and the native
Qwen3.8 model check passed with the same build.

## Prefill tuning

On the same 20-core GPU, a 2,048-token prompt processed at 358.67 tokens/s
with 128-row chunks and 386.17 tokens/s with 512-row chunks. The median
prefill time fell from 5.71 to 5.30 seconds. The same loaded weights and
input IDs ran through both paths, with one warmup per chunk size and four
measured samples per size in alternating order. The 64 single-token decode
steps measured 12.47 and 12.48 tokens/s, respectively.

With 512-row chunks, all three parity cases passed the same thresholds.
The 1,123-token case retained `cobalt` and had 9/10 top-10 overlap and a
0.0811 RMS logit error. A 3,883-token version of the same prompt also
returned `cobalt` in both engines, with 10/10 overlap and a 0.0667 RMS
logit error. The mixed attention shader-validation test passed. This Mac's
samples and check results are in `build/gemma4-profile/opt`.

## Prompt-lookup decode

Each decode forward takes the pending token and, where the request's latest
three or more tokens occurred earlier, up to 15 of the tokens that followed
them there (`LookupIndex`). Row i's logits select the token after row i, so a
forward keeps the proposals its own greedy selections repeat plus the next
selection, and rewinds the position past the rest; their keys and values are
overwritten by the next forward. The runner allocates Full pages for the
whole context, so no window page is recycled under a rejected row.

One forward after a 2,048-token prefix, median of seven samples with the
row counts interleaved:

| Rows | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms | 76.7 | 77.5 | 81.1 | 80.3 | 84.4 | 85.0 | 104.9 | 109.2 |

Greedy decode of 256 tokens, two measured samples after one warmup, each
loading the weights once and alternating the widths:

| Prompt | Prompt tokens | 0 lookup tokens (tok/s) | 15 lookup tokens (tok/s) | Tokens a forward | Same IDs as 0 |
| --- | ---: | ---: | ---: | ---: | --- |
| Add type hints to a 64-line Python file | 629 | 13.34–13.42 | 29.89–32.07 | 2.45 | 256/256 |
| Summarize this document with exact numbers | 1879 | 12.14–12.37 | 18.29–18.61 | 1.65 | 256/256 |
| 20 prose records to JSON | 414 | 13.51–13.60 | 21.41–21.99 | 1.69 | 256/256 |
| 300-word short story | 34 | 14.48–14.60 | 14.35–14.43 | 1.00 | 192/256 |

Verifying at most 3 or 7 tokens a forward gained less on every prompt. The
story found 45 proposals in 254 forwards and the target selected one of
them, so lookup cost it about 1%. Its output diverged at token 192 on a
near tie (`.` against `,`): rows of a multi-row forward run another
projection tile than a single row, so their logits differ slightly. The three
llama.cpp parity cases passed with the default, unchanged. The harness,
prompts and samples are in `build/gemma4-profile/verify`.

## DFlash draft

The draft is z-lab's [gemma-4-31B-it-DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash/tree/eabd648301ce28583cc14757912e5e0f84e152e1)
at revision `eabd648301ce28583cc14757912e5e0f84e152e1`: `model.safetensors`
has 3,071,941,240 bytes and SHA-256
`b3f28580103d6e8a0a1d9389646957b9309df5982558afd754b9810357f425b7`. It is a
DFlash v1 checkpoint: five Qwen3 layers (64 query heads and 8 KV heads of
128 dimensions, four causal layers with a 2,048-token window and one that
attends the whole block) over a block of the pending token and 15 mask
tokens. Its context rows concatenate the target's outputs of layers 1, 12,
23, 35, 46 and 57, which `fc` projects to one hidden row.

`BlockDraft` reads `config.json` and quantizes the BF16 projections into
af4g64 as they load, as the DFlash2 drafts are, about 0.86 GB. Every target
forward copies those six layers' output rows into the draft's context rows
and writes their keys and values into the draft's cache at the forward's
positions, so a verify writes all 16 rows and the next block overwrites the
rejected ones. A cycle runs the draft block, selects its 15 tokens on the
host and verifies them with the pending token in one target forward. The
block's attention runs in `block_draft_attention.metal`, since the paged
kernels hold neither this head geometry nor attention over the whole block.

```sh
build/engine-tests/block-draft-attention build/splash.metallib
./splash generate --model "$model" --draft install/models/gemma4-31b-dflash-eabd648 \
  --prompt 'Write a quicksort function in Python.' --max-tokens 256
```

### Checks

- The kernel test compares the stored keys and values, and attention that
  is causal within a window, causal over all keys and over the whole block,
  with an fp64 reference of the same bf16 inputs: relative errors below
  0.004.
- An independent float32 PyTorch forward of the checkpoint, on the context
  rows Splash captured for a 34-token prompt, selected the same 15 tokens as
  Splash when its projections were rounded to af4g64 the same way (largest
  logit difference 0.48 over Splash's top 20). With BF16 projections, 12 of
  15 matched.
- Q8_0 projections changed tokens a forward by +4% on the story, 0% on the
  JSON and −4% on the code-edit prompt over 128 tokens, and decoded 4–8%
  slower.
- Capturing the layers one before or one after each listed layer changed
  tokens a forward by under 7%.
- Row 0 of verify forwards of 2 to 32 rows had RMS logit errors of 0.039 to
  0.079 against llama.cpp on the three parity prompts, as single-row
  forwards did (0.039 to 0.091), with the same top token.

### Results on M5 Pro

256 greedy tokens, two measured samples after one warmup, the weights loaded
once:

| Prompt | Prompt tokens | Without draft (tok/s) | With draft (tok/s) | Tokens a forward | Same IDs as without |
| --- | ---: | ---: | ---: | ---: | --- |
| Write a quicksort with docstring and example | 30 | 13.84–13.91 | 48.02–51.85 | 5.10 | 256/256 |
| Word problem, step by step | 67 | 13.53–13.62 | 48.21–49.21 | 5.15 | 248/248 |
| Add type hints to a 64-line Python file | 629 | 13.02–13.06 | 57.41–57.53 | 6.07 | 256/256 |
| 20 prose records to JSON | 414 | 13.26–13.37 | 97.39–97.71 | 9.81 | 256/256 |
| Summarize this document with exact numbers | 1879 | 12.88–13.39 | 37.27–42.08 | 4.32 | 256/256 |
| 300-word short story | 34 | 13.70–13.88 | 17.57–19.31 | 1.75 | 161/256 |

The document prompt is from a separate run of its own: in the six-prompt
run it ran last and measured 11.09–11.74 and 30.69–36.14 tokens/s. The story
diverged at token 161 on a near tie, as the lookup run did. After a prompt,
a cycle measured 8.2–11.0 ms for the draft block, 1.8–2.9 ms selecting its
tokens on the host and 81–91 ms for the 16-row verify; a one-row forward
measured 72–78 ms. Prefill with the draft's context took as long as without
it, within run-to-run variation. Samples and the harness are in
`build/gemma4-profile/verify`.
