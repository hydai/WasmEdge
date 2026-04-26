// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "plugin/plugin.h"
#include "wasinntypes.h"
#include <string>

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET
#include <common.h>
#include <llama.h>
#include <memory>
#include <sampling.h>

#include "wasinn_llama_batch.h"
#endif

namespace WasmEdge::Host::WASINN {
struct WasiNNEnvironment;
}

namespace WasmEdge::Host::WASINN::BitNet {

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

struct LlamaModelDeleter {
  void operator()(llama_model *Ptr) const {
    if (Ptr) {
      llama_free_model(Ptr);
    }
  }
};
struct LlamaContextDeleter {
  void operator()(llama_context *Ptr) const {
    if (Ptr) {
      llama_free(Ptr);
    }
  }
};
using LlamaModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;

enum class EmbdNormalizeType : int32_t {
  // From: https://github.com/ggerganov/llama.cpp/blob/master/common/common.h
  None = -1,
  MaxAbsolute = 0,
  Taxicab = 1,
  Euclidean = 2,
  PNorm = 3,
};

struct LocalConfig {
  int64_t NPredict = -1;
  bool StreamStdout = false;
  std::string ReversePrompt;
  EmbdNormalizeType EmbdNormalize = EmbdNormalizeType::Euclidean;
};

struct Graph {
  bool EnableLog = false;
  bool EnableDebugLog = false;
  common_params Params;
  LlamaModelPtr LlamaModel = nullptr;
  LlamaContextPtr LlamaContext = nullptr;
  LocalConfig Conf;
};

struct Context {
public:
  Context(uint32_t, Graph &G) noexcept : Conf(G.Conf) {}

  bool ComputeSingleStarted = false;

  int32_t NPos = 0;
  std::vector<llama_token> LlamaInputs;
  uint64_t LlamaNInputs = 0;
  std::vector<llama_token> LlamaOutputTokens;
  std::vector<uint8_t> LlamaOutputs;
  CommonSamplerPtr LlamaSampler = nullptr;
  int64_t CurrentBatchSize = 0;
  LlamaBatchHolder LlamaBatch;
  LlamaBatchHolder OutputBatch;

  LocalConfig Conf;
};

#else

struct Graph {};
struct Context {
  Context(uint32_t, Graph &) noexcept {}
};
#endif

struct Environ {};

Expect<WASINN::ErrNo> load(WASINN::WasiNNEnvironment &Env,
                           Span<const Span<uint8_t>> Builders,
                           WASINN::Device Device, uint32_t &GraphId) noexcept;

Expect<WASINN::ErrNo> initExecCtx(WASINN::WasiNNEnvironment &Env,
                                  uint32_t GraphId,
                                  uint32_t &ContextId) noexcept;

Expect<WASINN::ErrNo> setInput(WASINN::WasiNNEnvironment &Env,
                               uint32_t ContextId, uint32_t Index,
                               const TensorData &Tensor) noexcept;

Expect<WASINN::ErrNo> getOutput(WASINN::WasiNNEnvironment &Env,
                                uint32_t ContextId, uint32_t Index,
                                Span<uint8_t> OutBuffer,
                                uint32_t &BytesWritten) noexcept;

Expect<WASINN::ErrNo> compute(WASINN::WasiNNEnvironment &Env,
                              uint32_t ContextId) noexcept;

Expect<WASINN::ErrNo> getOutputSingle(WASINN::WasiNNEnvironment &Env,
                                      uint32_t ContextId, uint32_t Index,
                                      Span<uint8_t> OutBuffer,
                                      uint32_t &BytesWritten) noexcept;

Expect<WASINN::ErrNo> computeSingle(WASINN::WasiNNEnvironment &Env,
                                    uint32_t ContextId) noexcept;

Expect<WASINN::ErrNo> finiSingle(WASINN::WasiNNEnvironment &Env,
                                 uint32_t ContextId) noexcept;

Expect<WASINN::ErrNo> finalizeExecCtx(WASINN::WasiNNEnvironment &Env,
                                      uint32_t ContextId) noexcept;

Expect<WASINN::ErrNo> unload(WASINN::WasiNNEnvironment &Env,
                             uint32_t GraphId) noexcept;

} // namespace WasmEdge::Host::WASINN::BitNet
