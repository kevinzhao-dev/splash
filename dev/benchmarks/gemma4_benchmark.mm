// Gemma 4 31B's forwards (model/Gemma4.hpp), each measured on its own rather
// than in a workload:
//
//   gemma4-benchmark METALLIB GGUF [--draft DIR] [--samples N] SCENARIO ...
//     generate TOKEN_IDS MAX_NEW [--lookup-tokens N] [--logits FILE]
//     prefill
//     steps
//     profile
//     draft
//     teacher TOKEN_IDS REFERENCE [--rows R[,R...]]
//
// generate: a greedy answer to a file of prompt token IDs, as JSON, each
// forward verifying the draft's block or up to N prompt-lookup tokens (15),
// and with --logits the prompt's last softcapped logits as fp32.
// prefill: cold prefills of 512, 1024, 2048 and 4000 tokens in chunks of 512.
// steps: one forward of T rows, each with logits (T of 1 to 32), after
// prefixes of 128, 1024, 2048 and 4000 tokens, rewound after each: a decode
// step's cost by its rows.
// profile: each pipeline's GPU time in a forward of 1 and of 16 rows after a
// 2048-token prefix, each dispatch committed as its own command, beside the
// fused command's GPU time.
// draft: the draft's block, and a 16-row forward with and without writing the
// draft's context, after the same prefixes as steps.
// teacher: each position's top token of the softcapped logits, teacher-forced
// in forwards of R rows (32 and 1), against REFERENCE's (gguf-text teacher):
// how often they agree, and how far apart the reference's two best logits are
// where they do not.
//
// Each measurement is one JSON line: the median, least and most of --samples
// samples (5) after one warmup, the row counts of steps interleaved in an
// order that alternates each sample. The synthetic prompts are token IDs of a
// fixed linear congruential sequence over [1000, 200000), ordinary vocabulary
// that no attention or projection treats apart.

#include "AwakeClock.hpp"
#include "engine/Json.hpp"
#include "metal/BackendInstrumentation.hpp"
#include "model/Gemma4.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace splash;
using model::Gemma4;
using Clock = AwakeClock;

constexpr uint32_t kVocabulary = model::kGemma4Vocabulary;
constexpr uint32_t kChunk = 512, kBlock = 16;
constexpr std::array kPrefills{512u, 1024u, 2048u, 4000u};
constexpr std::array kContexts{128u, 1024u, 2048u, 4000u};
constexpr std::array kStepRows{1u, 2u, 4u, 8u, 12u, 16u, 24u, 32u};

double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

uint32_t parseCount(std::string_view text, std::string_view what) {
  uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument(std::string(what) + " must be an unsigned integer");
  return value;
}

std::vector<uint32_t> readIds(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("cannot read " + path.string());
  std::vector<uint32_t> ids;
  for (uint32_t id; input >> id;) ids.push_back(id);
  if (!input.eof() || ids.empty()) throw std::invalid_argument("expected decimal token IDs in " + path.string());
  return ids;
}

std::vector<uint32_t> syntheticIds(uint32_t count) {
  std::vector<uint32_t> ids(count);
  uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (uint32_t &id : ids) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    id = 1000 + static_cast<uint32_t>((state >> 33) % 199000);
  }
  return ids;
}

// The median, least and most of a measurement's samples, in milliseconds.
struct Spread final {
  double median = 0, least = 0, most = 0;
};

Spread spread(std::vector<double> values) {
  std::ranges::sort(values);
  return {values[values.size() / 2] * 1e3, values.front() * 1e3, values.back() * 1e3};
}

void writeSpread(json::Writer &out, std::string_view name, const std::vector<double> &values) {
  const Spread s = spread(values);
  out.field(std::string(name) + "_ms", s.median)
      .field(std::string(name) + "_min_ms", s.least)
      .field(std::string(name) + "_max_ms", s.most);
}

void emit(json::Writer &out) { std::cout << out.str() << std::endl; }

// Extends a sequence's prefix to `position` tokens of `ids`.
void extend(Gemma4::Sequence &sequence, std::span<const uint32_t> ids, uint32_t position) {
  while (sequence.position() < position) {
    const uint32_t first = sequence.position(), rows = std::min(kChunk, position - first);
    static_cast<void>(sequence.forward(ids.subspan(first, rows), 0));
  }
}

int generate(Gemma4 &gemma, std::span<char *> args) {
  if (args.size() < 2) throw std::invalid_argument("generate takes TOKEN_IDS MAX_NEW");
  const auto prompt = readIds(args[0]);
  const uint32_t newTokens = parseCount(args[1], "MAX_NEW");
  uint32_t lookupTokens = model::kGemma4DefaultLookupTokens;
  const char *logitsPath = nullptr;
  for (size_t i = 2; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "--lookup-tokens" && i + 1 < args.size()) lookupTokens = parseCount(args[++i], "--lookup-tokens");
    else if (arg == "--logits" && i + 1 < args.size()) logitsPath = args[++i];
    else throw std::invalid_argument("generate takes --lookup-tokens N and --logits FILE");
  }
  const auto result = gemma.generate(prompt, newTokens, lookupTokens);
  if (logitsPath) {
    std::ofstream output(logitsPath, std::ios::binary);
    output.write(reinterpret_cast<const char *>(result.promptLogits.data()),
                 static_cast<std::streamsize>(result.promptLogits.size() * sizeof(float)));
    if (!output) throw std::runtime_error("cannot write the prompt logits");
  }
  json::Writer out;
  out.object([&] {
    out.array("tokens", [&] { for (uint32_t token : result.tokens) out.value(token); });
    out.field("prompt_tokens", prompt.size())
        .field("prefill_seconds", result.prefillSeconds)
        .field("decode_seconds", result.decodeSeconds)
        .field("lookup_tokens", gemma.drafts() ? 0 : lookupTokens)
        .field("draft", gemma.drafts())
        .field("decode_cycles", result.decodeCycles)
        .field("decode_rows", result.decodeRows)
        .field("proposed_tokens", result.proposedTokens)
        .field("accepted_tokens", result.acceptedTokens)
        .field("weight_and_kv_bytes", gemma.weightBytes() + result.kvBytes);
  });
  emit(out);
  return 0;
}

void prefills(Gemma4 &gemma, uint32_t samples) {
  const auto ids = syntheticIds(kPrefills.back());
  auto sequence = gemma.sequence(kPrefills.back(), 1, false);
  for (uint32_t tokens : kPrefills) {
    std::vector<double> wall, gpu;
    for (uint32_t sample = 0; sample <= samples; ++sample) {
      double commandSeconds = 0;
      const auto start = Clock::now();
      sequence.rewind(0);
      for (uint32_t first = 0; first < tokens; first += kChunk) {
        const uint32_t rows = std::min(kChunk, tokens - first);
        static_cast<void>(sequence.forward(std::span(ids).subspan(first, rows), first + rows == tokens));
        commandSeconds += sequence.gpuSeconds();
      }
      if (!sample) continue;
      wall.push_back(seconds(start));
      gpu.push_back(commandSeconds);
    }
    json::Writer out;
    out.object([&] {
      out.field("scenario", "prefill").field("prompt_tokens", tokens);
      writeSpread(out, "wall", wall);
      writeSpread(out, "gpu", gpu);
      out.field("tokens_per_second", tokens / (spread(wall).median / 1e3));
    });
    emit(out);
  }
}

void steps(Gemma4 &gemma, uint32_t samples) {
  const auto ids = syntheticIds(kContexts.back() + kStepRows.back());
  auto sequence = gemma.sequence(kContexts.back() + kStepRows.back(), kStepRows.back(), false);
  for (uint32_t context : kContexts) {
    extend(sequence, ids, context);
    std::map<uint32_t, std::vector<double>> wall, gpu;
    for (uint32_t sample = 0; sample <= samples; ++sample)
      for (size_t i = 0; i < kStepRows.size(); ++i) {
        const uint32_t rows = kStepRows[sample % 2 ? kStepRows.size() - 1 - i : i];
        const auto start = Clock::now();
        static_cast<void>(sequence.forward(std::span(ids).subspan(context, rows), rows));
        const double elapsed = seconds(start);
        sequence.rewind(context);
        if (!sample) continue;
        wall[rows].push_back(elapsed);
        gpu[rows].push_back(sequence.gpuSeconds());
      }
    const double single = spread(gpu[1]).median;
    for (uint32_t rows : kStepRows) {
      json::Writer out;
      out.object([&] {
        out.field("scenario", "steps").field("context", context).field("rows", rows);
        writeSpread(out, "wall", wall[rows]);
        writeSpread(out, "gpu", gpu[rows]);
        out.field("gpu_over_one_row", spread(gpu[rows]).median / single);
      });
      emit(out);
    }
  }
}

void profile(Gemma4 &gemma, uint32_t samples) {
  constexpr uint32_t context = 2048;
  const auto ids = syntheticIds(context + kBlock);
  auto sequence = gemma.sequence(context + kBlock, kBlock, false);
  extend(sequence, ids, context);
  metal::MetalBackend &backend = gemma.backend();
  for (uint32_t rows : {1u, kBlock}) {
    const auto rowIds = std::span(ids).subspan(context, rows);
    std::vector<double> fused;
    for (uint32_t sample = 0; sample <= samples; ++sample) {
      static_cast<void>(sequence.forward(rowIds, rows));
      sequence.rewind(context);
      if (sample) fused.push_back(sequence.gpuSeconds());
    }
    struct Pipeline final {
      uint32_t dispatches = 0;
      double seconds = 0;
    };
    std::map<std::string, Pipeline> pipelines;
    double attributed = 0;
    metal::BackendInstrumentation::setDispatchProfiling(backend, true);
    for (uint32_t sample = 0; sample < samples; ++sample) {
      static_cast<void>(sequence.forward(rowIds, rows));
      sequence.rewind(context);
      for (const auto &timing : metal::BackendInstrumentation::takeDispatchProfile(backend)) {
        Pipeline &pipeline = pipelines[timing.pipelineName];
        ++pipeline.dispatches;
        pipeline.seconds += timing.gpuSeconds;
        attributed += timing.gpuSeconds;
      }
    }
    metal::BackendInstrumentation::setDispatchProfiling(backend, false);
    std::vector<std::pair<std::string, Pipeline>> ranked(pipelines.begin(), pipelines.end());
    std::ranges::sort(ranked, [](const auto &a, const auto &b) { return a.second.seconds > b.second.seconds; });
    json::Writer out;
    out.object([&] {
      out.field("scenario", "profile").field("context", context).field("rows", rows);
      writeSpread(out, "fused_gpu", fused);
      out.field("attributed_gpu_ms", attributed / samples * 1e3);
      out.array("pipelines", [&] {
        for (const auto &[name, pipeline] : ranked)
          out.object([&] {
            out.field("pipeline", name)
                .field("dispatches", pipeline.dispatches / samples)
                .field("gpu_ms", pipeline.seconds / samples * 1e3)
                .field("share", pipeline.seconds / attributed);
          });
      });
    });
    emit(out);
  }
}

void drafts(Gemma4 &gemma, uint32_t samples) {
  if (!gemma.drafts()) throw std::invalid_argument("the draft scenario takes --draft DIR");
  const uint32_t capacity = kContexts.back() + kBlock;
  const auto ids = syntheticIds(capacity);
  auto drafted = gemma.sequence(capacity, kBlock, true), bare = gemma.sequence(capacity, kBlock, false);
  for (uint32_t context : kContexts) {
    extend(drafted, ids, context);
    extend(bare, ids, context);
    std::vector<double> blockWall, blockGpu, withContext, without;
    for (uint32_t sample = 0; sample <= samples; ++sample) {
      const auto start = Clock::now();
      static_cast<void>(drafted.draft(ids[context]));
      const double wall = seconds(start), gpu = drafted.gpuSeconds();
      const auto rowIds = std::span(ids).subspan(context, kBlock);
      static_cast<void>(drafted.forward(rowIds, kBlock));
      const double drafting = drafted.gpuSeconds();
      drafted.rewind(context);
      static_cast<void>(bare.forward(rowIds, kBlock));
      const double plain = bare.gpuSeconds();
      bare.rewind(context);
      if (!sample) continue;
      blockWall.push_back(wall);
      blockGpu.push_back(gpu);
      withContext.push_back(drafting);
      without.push_back(plain);
    }
    json::Writer out;
    out.object([&] {
      out.field("scenario", "draft").field("context", context);
      writeSpread(out, "block_wall", blockWall);
      writeSpread(out, "block_gpu", blockGpu);
      writeSpread(out, "verify_with_context_gpu", withContext);
      writeSpread(out, "verify_without_context_gpu", without);
    });
    emit(out);
  }
}

void teacher(Gemma4 &gemma, std::span<char *> args) {
  if (args.size() < 2) throw std::invalid_argument("teacher takes TOKEN_IDS REFERENCE");
  const auto ids = readIds(args[0]);
  std::vector<uint32_t> widths{32, 1};
  for (size_t i = 2; i < args.size(); ++i) {
    if (std::string_view(args[i]) != "--rows" || i + 1 == args.size())
      throw std::invalid_argument("teacher takes --rows R[,R...]");
    widths.clear();
    std::stringstream list(args[++i]);
    for (std::string item; std::getline(list, item, ',');) widths.push_back(parseCount(item, "--rows"));
  }
  // The reference: a line a position from the first, its top token and how
  // far its best logit is above its second.
  std::ifstream input(args[1]);
  std::vector<uint32_t> referenceTop;
  std::vector<double> margins;
  for (uint32_t top; input >> top;) {
    double margin;
    if (!(input >> margin)) throw std::invalid_argument("expected TOKEN MARGIN lines in the reference");
    referenceTop.push_back(top);
    margins.push_back(margin);
  }
  if (referenceTop.size() + 1 < ids.size()) throw std::invalid_argument("the reference covers too few positions");
  const uint32_t positions = static_cast<uint32_t>(ids.size() - 1);
  for (uint32_t width : widths) {
    if (!width || width > 32) throw std::invalid_argument("teacher forwards take 1 to 32 rows");
    auto sequence = gemma.sequence(static_cast<uint32_t>(ids.size()), width, false);
    const auto start = Clock::now();
    std::vector<uint32_t> top;
    // The first chunk prefills; every forward after it outputs all its rows.
    const uint32_t lead = std::min(kChunk, positions);
    const auto first = sequence.forward(std::span(ids).first(lead), std::min(width, lead));
    const uint32_t firstRows = std::min(width, lead);
    std::vector<uint32_t> leading(lead - firstRows, UINT32_MAX);
    top.insert(top.end(), leading.begin(), leading.end());
    for (uint32_t row = 0; row < firstRows; ++row)
      top.push_back(Gemma4::select(first.subspan(size_t{row} * kVocabulary, kVocabulary)));
    while (sequence.position() < positions) {
      const uint32_t rows = std::min(width, positions - sequence.position());
      const auto logits = sequence.forward(std::span(ids).subspan(sequence.position(), rows), rows);
      for (uint32_t row = 0; row < rows; ++row)
        top.push_back(Gemma4::select(logits.subspan(size_t{row} * kVocabulary, kVocabulary)));
    }
    uint32_t compared = 0, agreed = 0;
    std::vector<double> disagreeing;
    for (uint32_t position = 0; position < positions; ++position) {
      if (top[position] == UINT32_MAX) continue;
      ++compared;
      if (top[position] == referenceTop[position]) ++agreed;
      else disagreeing.push_back(margins[position]);
    }
    std::ranges::sort(disagreeing);
    std::vector<double> all(margins.begin(), margins.begin() + positions);
    std::ranges::sort(all);
    json::Writer out;
    out.object([&] {
      out.field("scenario", "teacher")
          .field("rows", width)
          .field("positions", compared)
          .field("top1_agreement", double(agreed) / compared)
          .field("reference_margin_median", all[all.size() / 2])
          .field("disagreeing_reference_margin_median",
                 disagreeing.empty() ? 0.0 : disagreeing[disagreeing.size() / 2])
          .field("seconds", seconds(start));
    });
    emit(out);
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 4)
        throw std::invalid_argument(
            "usage: gemma4-benchmark METALLIB GGUF [--draft DIR] [--samples N] "
            "generate|prefill|steps|profile|draft|teacher ...");
      std::filesystem::path draft;
      uint32_t samples = 5;
      int arg = 3;
      for (; arg < argc && std::string_view(argv[arg]).starts_with("--"); arg += 2) {
        if (arg + 1 == argc) throw std::invalid_argument(std::string(argv[arg]) + " takes a value");
        if (std::string_view(argv[arg]) == "--draft") draft = argv[arg + 1];
        else if (std::string_view(argv[arg]) == "--samples")
          samples = std::max(1u, parseCount(argv[arg + 1], "--samples"));
        else throw std::invalid_argument(std::string("unknown option ") + argv[arg]);
      }
      if (arg == argc) throw std::invalid_argument("expected a scenario");
      const std::string_view scenario = argv[arg];
      const std::span<char *> rest(argv + arg + 1, argv + argc);
      Gemma4 gemma(argv[2], argv[1], draft);
      if (scenario == "generate") return generate(gemma, rest);
      if (scenario == "prefill") prefills(gemma, samples);
      else if (scenario == "steps") steps(gemma, samples);
      else if (scenario == "profile") profile(gemma, samples);
      else if (scenario == "draft") drafts(gemma, samples);
      else if (scenario == "teacher") teacher(gemma, rest);
      else throw std::invalid_argument("unknown scenario " + std::string(scenario));
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "gemma4-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}
