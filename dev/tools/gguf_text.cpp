#include "engine/Json.hpp"

#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kContext = 4096, kBatch = 128;
using Clock = std::chrono::steady_clock;

std::vector<llama_token> readTokens(std::istream &input, const llama_vocab *vocab) {
  std::vector<llama_token> result;
  for (llama_token token; input >> token;) {
    if (token < 0 || token >= llama_vocab_n_tokens(vocab) || result.size() >= kContext)
      throw std::runtime_error("token IDs exceed the vocabulary or context limit");
    result.push_back(token);
  }
  if (!input.eof()) throw std::runtime_error("expected decimal token IDs");
  return result;
}

void tokenize(const llama_vocab *vocab) {
  const std::string text(std::istreambuf_iterator<char>(std::cin), {});
  if (text.size() > std::numeric_limits<int32_t>::max()) throw std::runtime_error("prompt too large");
  std::vector<llama_token> tokens(kContext);
  const auto count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                                    kContext, true, true);
  if (count < 0) throw std::runtime_error("prompt exceeds the token limit");
  for (int32_t i = 0; i < count; ++i) std::cout << tokens[i] << '\n';
}

void detokenize(const llama_vocab *vocab) {
  const auto tokens = readTokens(std::cin, vocab);
  if (tokens.empty()) return;
  std::vector<char> bytes(tokens.size() * 16);
  const auto decode = [&] {
    return llama_detokenize(vocab, tokens.data(), static_cast<int32_t>(tokens.size()), bytes.data(),
                            static_cast<int32_t>(bytes.size()), true, false);
  };
  int32_t size = decode();
  if (size < 0) {
    bytes.resize(-size);
    size = decode();
  }
  if (size < 0) throw std::runtime_error("cannot detokenize the output");
  std::cout.write(bytes.data(), size);
}

void reference(llama_model *model, const llama_vocab *vocab, const char *ids, std::string_view count,
               const char *logitsFile) {
  uint32_t maximum = 0;
  const auto parsed = std::from_chars(count.data(), count.data() + count.size(), maximum);
  if (parsed.ec != std::errc{} || parsed.ptr != count.data() + count.size())
    throw std::runtime_error("expected an unsigned generation limit");
  std::ifstream input(ids);
  if (!input) throw std::runtime_error("cannot read the prompt tokens");
  const auto prompt = readTokens(input, vocab);
  if (prompt.empty() || maximum > kContext - prompt.size()) throw std::runtime_error("context limit exceeded");
  auto parameters = llama_context_default_params();
  parameters.n_ctx = kContext;
  parameters.n_batch = kBatch;
  parameters.n_ubatch = kBatch;
  parameters.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  const std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model, parameters),
                                                                    llama_free);
  if (!context) throw std::runtime_error("could not start the llama.cpp reference context");
  auto evaluate = [&](const llama_token *tokens, int32_t size) {
    if (llama_decode(context.get(), llama_batch_get_one(const_cast<llama_token *>(tokens), size)) != 0)
      throw std::runtime_error("llama.cpp decode failed");
  };
  const auto start = Clock::now();
  for (size_t first = 0; first < prompt.size(); first += kBatch)
    evaluate(prompt.data() + first, static_cast<int32_t>(std::min<size_t>(kBatch, prompt.size() - first)));
  const int32_t vocabSize = llama_vocab_n_tokens(vocab);
  const auto getLogits = [&] {
    const float *logits = llama_get_logits_ith(context.get(), -1);
    if (!logits) throw std::runtime_error("llama.cpp did not return logits");
    return logits;
  };
  const float *logits = getLogits();
  const double prefill = std::chrono::duration<double>(Clock::now() - start).count();
  {
    std::ofstream output(logitsFile, std::ios::binary);
    output.write(reinterpret_cast<const char *>(logits), vocabSize * sizeof(float));
    if (!output) throw std::runtime_error("cannot write reference logits");
  }
  std::vector<llama_token> generated;
  for (uint32_t i = 0; i < maximum; ++i) {
    const auto token = static_cast<llama_token>(std::max_element(logits, logits + vocabSize) - logits);
    if (llama_vocab_is_eog(vocab, token)) break;
    generated.push_back(token);
    if (i + 1 < maximum) {
      evaluate(&token, 1);
      logits = getLogits();
    }
  }
  splash::json::Writer output;
  output.object([&] {
    output.field("prefill_seconds", prefill);
    output.array("tokens", [&] { for (const auto token : generated) output.value(token); });
  });
  std::cout << output.str() << '\n';
}

// Each position's top token of llama.cpp's logits, teacher-forced over the
// token IDs in batches of kBatch, and how far its best logit is above its
// second: a line a position but the last.
void teacher(llama_model *model, const llama_vocab *vocab, const char *ids) {
  std::ifstream input(ids);
  if (!input) throw std::runtime_error("cannot read the token IDs");
  const auto tokens = readTokens(input, vocab);
  if (tokens.size() < 2) throw std::runtime_error("teacher forcing needs two tokens");
  auto parameters = llama_context_default_params();
  parameters.n_ctx = kContext;
  parameters.n_batch = kBatch;
  parameters.n_ubatch = kBatch;
  parameters.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  const std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model, parameters),
                                                                    llama_free);
  if (!context) throw std::runtime_error("could not start the llama.cpp reference context");
  llama_batch batch = llama_batch_init(kBatch, 0, 1);
  const int32_t vocabSize = llama_vocab_n_tokens(vocab);
  const size_t positions = tokens.size() - 1;
  for (size_t first = 0; first < positions; first += kBatch) {
    batch.n_tokens = static_cast<int32_t>(std::min<size_t>(kBatch, positions - first));
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
      batch.token[i] = tokens[first + i];
      batch.pos[i] = static_cast<llama_pos>(first + i);
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = 1;
    }
    if (llama_decode(context.get(), batch) != 0) {
      llama_batch_free(batch);
      throw std::runtime_error("llama.cpp decode failed");
    }
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
      const float *logits = llama_get_logits_ith(context.get(), i);
      if (!logits) throw std::runtime_error("llama.cpp did not return logits");
      int32_t best = 0;
      float top = logits[0], second = -std::numeric_limits<float>::infinity();
      for (int32_t token = 1; token < vocabSize; ++token) {
        if (logits[token] > top) {
          second = top;
          top = logits[token];
          best = token;
        } else if (logits[token] > second) {
          second = logits[token];
        }
      }
      std::cout << best << ' ' << top - second << '\n';
    }
  }
  llama_batch_free(batch);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view action = argc >= 2 ? argv[1] : "";
    const bool vocabOnly = action == "tokenize" || action == "detokenize";
    if (!(vocabOnly && argc == 3) && !(action == "reference" && argc == 6) && !(action == "teacher" && argc == 4))
      throw std::runtime_error("usage: gguf-text tokenize|detokenize GGUF, gguf-text reference GGUF IDS MAX LOGITS, "
                               "or gguf-text teacher GGUF IDS");
    ggml_backend_load_all();
    auto parameters = llama_model_default_params();
    parameters.vocab_only = vocabOnly;
    parameters.n_gpu_layers = vocabOnly ? 0 : -1;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[2], parameters), llama_model_free);
    if (!model) throw std::runtime_error("llama.cpp could not open the GGUF");
    const auto *vocab = llama_model_get_vocab(model.get());
    if (action == "tokenize") tokenize(vocab);
    else if (action == "detokenize") detokenize(vocab);
    else if (action == "teacher") teacher(model.get(), vocab, argv[3]);
    else reference(model.get(), vocab, argv[3], argv[4], argv[5]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gguf-text: " << error.what() << '\n';
    return 1;
  }
}
