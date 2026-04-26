// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_bitnet_inference.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

#include "common/spdlog.h"

#include <common.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <llama.h>
#include <sampling.h>

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace WasmEdge::Host::WASINN::BitNet {
namespace {
using namespace std::literals;

#define LOG_DEBUG(Debug, ...)                                                  \
  if (Debug) {                                                                 \
    spdlog::info("[WASI-NN][Debug] BitNet backend: "sv __VA_ARGS__);           \
  }

#define LOG_INFO(Info, ...)                                                    \
  if (Info) {                                                                  \
    spdlog::info("[WASI-NN] BitNet backend: "sv __VA_ARGS__);                  \
  }

#define LOG_WARN(...) spdlog::warn("[WASI-NN] BitNet backend: "sv __VA_ARGS__);

#define LOG_ERROR(...)                                                         \
  spdlog::error("[WASI-NN] BitNet backend: "sv __VA_ARGS__);

#define RET_ERROR(Error, ...)                                                  \
  spdlog::error("[WASI-NN] BitNet backend: "sv __VA_ARGS__);                   \
  return Error;

void buildOutputEmbedding(std::string &Embedding, int32_t NEmbd,
                          const float *Embeddings) noexcept {
  Embedding =
      fmt::format(R"({{"n_embedding": {}, )"
                  R"("embedding": [{:.10}]}})"sv,
                  NEmbd, fmt::join(Embeddings, Embeddings + NEmbd, ","sv));
}

void fillBatch(Span<const llama_token> Tokens, Graph &GraphRef,
               llama_batch &Batch, int &NPos, bool IsLogit = false) {
  assuming(GraphRef.Params.n_batch >= static_cast<int64_t>(Tokens.size()));
  assuming(Batch.token != nullptr);
  assuming(Batch.pos != nullptr);
  assuming(Batch.logits != nullptr);
  Batch.n_tokens = static_cast<int32_t>(Tokens.size());
  for (uint32_t I = 0; I < Tokens.size(); I++) {
    Batch.token[I] = Tokens[I];
    Batch.pos[I] = NPos + I;
    Batch.logits[I] = false;
  }

  if (IsLogit) {
    Batch.logits[Tokens.size() - 1] = true;
  }

  NPos += static_cast<int>(Tokens.size());
}

ErrNo evaluateTokens(Span<const llama_token> Tokens, Graph &GraphRef,
                     llama_batch &Batch, int &NPos,
                     bool IsLogits = false) noexcept {
  uint32_t NCtx = llama_n_ctx(GraphRef.LlamaContext.get());
  if (NPos + static_cast<uint32_t>(Tokens.size()) > NCtx) {
    LOG_INFO(
        GraphRef.EnableLog,
        "evaluateTokens: the context if full ({} / {} tokens). Please increase your "sv
        "context size."sv,
        NPos + static_cast<uint32_t>(Tokens.size()), NCtx)
    return ErrNo::ContextFull;
  }

  for (int I = 0; I < static_cast<int>(Tokens.size());
       I += static_cast<int>(GraphRef.Params.n_batch)) {
    int NEval = static_cast<int>(Tokens.size()) - I;
    if (NEval > static_cast<int>(GraphRef.Params.n_batch)) {
      NEval = static_cast<int>(GraphRef.Params.n_batch);
    }

    fillBatch(Span<const llama_token>(Tokens.begin() + I, NEval), GraphRef,
              Batch, NPos,
              IsLogits && I + NEval >= static_cast<int>(Tokens.size()));

    auto Status = llama_decode(GraphRef.LlamaContext.get(), Batch);
    if (Status == 1) {
      RET_ERROR(
          ErrNo::RuntimeError,
          "evaluateTokens: failed to llama_decode: try reducing the size of the batch "sv
          "or increasing the size of context."sv)
    }
    if (Status < 0) {
      RET_ERROR(
          ErrNo::RuntimeError,
          "evaluateTokens: failed to llama_decode: internal fatal error. Please open "sv
          "an issue on GitHub."sv)
    }
  }

  return ErrNo::Success;
}

} // namespace

std::string buildOutputMetadata(Context &CxtRef) noexcept {
  return fmt::format(R"({{"input_tokens": {}, )"
                     R"("output_tokens": {}, )"
                     R"("llama_build_number": {}, )"
                     R"("llama_commit": "{}"}})"sv,
                     CxtRef.LlamaNInputs, CxtRef.LlamaOutputTokens.size(),
                     LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
}

void clearContext(Graph &GraphRef, Context &CxtRef) noexcept {
  LOG_DEBUG(GraphRef.EnableDebugLog, "clearContext"sv)
  llama_kv_cache_clear(GraphRef.LlamaContext.get());
  common_sampler_reset(CxtRef.LlamaSampler.get());
  CxtRef.NPos = 0;
  CxtRef.LlamaOutputTokens.clear();
  CxtRef.LlamaOutputs.clear();
  LOG_DEBUG(GraphRef.EnableDebugLog, "clearContext...Done"sv)
}

ErrNo evaluateInput(Graph &GraphRef, Context &CxtRef,
                    std::string_view LogPrefix) noexcept {
  if (CxtRef.LlamaInputs.size() == 0) {
    RET_ERROR(ErrNo::InvalidArgument, "{}: llama input is not set!"sv,
              LogPrefix)
  }

  const uint64_t NCtx = llama_n_ctx(GraphRef.LlamaContext.get());
  const uint64_t MaxTokensListSize = NCtx - 4;

  if (static_cast<uint64_t>(CxtRef.LlamaInputs.size()) > MaxTokensListSize) {
    RET_ERROR(ErrNo::PromptTooLong,
              "{}: the prompt is too long. Your input has {} tokens. "sv
              "Please reduce it to {} tokens."sv,
              LogPrefix, CxtRef.LlamaInputs.size(), MaxTokensListSize)
  }

  auto ReturnCode =
      evaluateTokens(Span<const llama_token>(CxtRef.LlamaInputs.begin(),
                                             CxtRef.LlamaInputs.size()),
                     GraphRef, CxtRef.LlamaBatch.get(), CxtRef.NPos, true);
  if (ReturnCode != ErrNo::Success) {
    RET_ERROR(ReturnCode, "{}: failed to evaluate input tokens."sv, LogPrefix)
  }

  return ErrNo::Success;
}

ErrNo sampleOutput(Graph &GraphRef, Context &CxtRef,
                   bool IsSingleTokenMode) noexcept {
  const llama_token Id = common_sampler_sample(
      CxtRef.LlamaSampler.get(), GraphRef.LlamaContext.get(), /* idx */ -1);
  common_sampler_accept(CxtRef.LlamaSampler.get(), Id,
                        /* accept_grammar */ true);

  CxtRef.LlamaOutputTokens.emplace_back(Id);
  std::string OutputString =
      common_token_to_piece(GraphRef.LlamaContext.get(), Id);
  CxtRef.LlamaOutputs.insert(CxtRef.LlamaOutputs.end(), OutputString.begin(),
                             OutputString.end());
  if (!IsSingleTokenMode) {
    if (CxtRef.Conf.StreamStdout) {
      fmt::print("{}"sv,
                 common_token_to_piece(GraphRef.LlamaContext.get(), Id));
      std::fflush(stdout);
    }
    if (!CxtRef.Conf.ReversePrompt.empty() &&
        std::string(CxtRef.LlamaOutputs.begin(), CxtRef.LlamaOutputs.end())
                .find(CxtRef.Conf.ReversePrompt) != std::string::npos) {
      LOG_INFO(GraphRef.EnableLog, "sampleOutput: reverse prompt found."sv)
      return ErrNo::EndOfSequence;
    }
  }

  if (!GraphRef.Params.sparams.ignore_eos) {
    if (llama_token_is_eog(GraphRef.LlamaModel.get(), Id)) {
      LOG_INFO(GraphRef.EnableLog, "sampleOutput: EOS token found."sv)
      return ErrNo::EndOfSequence;
    }
  }
  return evaluateTokens(Span<const llama_token>(&Id, 1), GraphRef,
                        CxtRef.OutputBatch.get(), CxtRef.NPos, true);
}

Expect<ErrNo> getEmbedding(Graph &GraphRef, Context &CxtRef) noexcept {
  LOG_DEBUG(GraphRef.EnableDebugLog, "getEmbedding"sv)

  const llama_token SepTokenId = llama_token_sep(GraphRef.LlamaModel.get());
  if (SepTokenId > -1) {
    if (CxtRef.LlamaInputs.size() > 0 &&
        CxtRef.LlamaInputs.back() != SepTokenId) {
      LOG_WARN(
          "getEmbedding: last token in the prompt is not SEP, "sv
          "'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF "sv
          "header."sv)
    }
  }

  if (static_cast<int64_t>(CxtRef.LlamaInputs.size()) >
      GraphRef.Params.n_batch) {
    RET_ERROR(
        ErrNo::PromptTooLong,
        "getEmbedding: the prompt is too long. Your input has {} tokens exceeds batch "sv
        "size {}. Please reduce the input size or increase your batch-size."sv,
        CxtRef.LlamaInputs.size(), GraphRef.Params.n_batch)
  }

  auto ReturnCode = evaluateInput(GraphRef, CxtRef, "getEmbedding"sv);
  if (ReturnCode != ErrNo::Success) {
    return ReturnCode;
  }

  const struct llama_model *LlamaModel =
      llama_get_model(GraphRef.LlamaContext.get());
  const int32_t NEmbd = llama_n_embd(LlamaModel);
  std::vector<float> Embeddings(NEmbd);

  const auto &Batch = CxtRef.LlamaBatch.get();
  for (int I = 0; I < Batch.n_tokens; I++) {
    if (!Batch.logits[I]) {
      continue;
    }

    auto *Embd = llama_get_embeddings_seq(GraphRef.LlamaContext.get(),
                                          Batch.seq_id[I][0]);
    if (Embd == nullptr) {
      Embd = llama_get_embeddings_ith(GraphRef.LlamaContext.get(), I);
      if (Embd == nullptr) {
        LOG_ERROR("getEmbedding: failed to get embeddings for token {}"sv, I);
        continue;
      }
    }

    common_embd_normalize(Embd, Embeddings.data(), NEmbd,
                          static_cast<int32_t>(CxtRef.Conf.EmbdNormalize));
  }

  std::string EmbeddingString;
  buildOutputEmbedding(EmbeddingString, NEmbd, Embeddings.data());
  CxtRef.LlamaOutputs =
      std::vector<uint8_t>(EmbeddingString.begin(), EmbeddingString.end());

  if (GraphRef.EnableLog) {
    common_perf_print(GraphRef.LlamaContext.get(), /* Sampler */ nullptr);
  }

  LOG_DEBUG(GraphRef.EnableDebugLog, "getEmbedding...Done"sv)
  return ErrNo::Success;
}

} // namespace WasmEdge::Host::WASINN::BitNet

#endif
