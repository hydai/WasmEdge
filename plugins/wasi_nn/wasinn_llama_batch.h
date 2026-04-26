// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#if defined(WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET) ||                         \
    defined(WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML)
#include <llama.h>
#include <sampling.h>

#include <algorithm>
#include <cstdint>
#include <memory>

namespace WasmEdge::Host::WASINN {

struct CommonSamplerDeleter {
  void operator()(common_sampler *Ptr) const {
    if (Ptr) {
      common_sampler_free(Ptr);
    }
  }
};

using CommonSamplerPtr = std::unique_ptr<common_sampler, CommonSamplerDeleter>;

inline llama_batch allocBatch(int64_t NTokens, int64_t Embd = 0,
                              int32_t NSeqMax = 1) noexcept {
  llama_batch Batch = llama_batch_init(
      /* n_tokens_alloc */ static_cast<int32_t>(NTokens),
      /* embd */ static_cast<int32_t>(Embd),
      /* n_seq_max */ static_cast<int32_t>(NSeqMax));
  std::fill(Batch.n_seq_id, Batch.n_seq_id + NTokens,
            static_cast<int32_t>(NSeqMax));
  for (int64_t I = 0; I < NTokens; I++) {
    std::fill(Batch.seq_id[I], Batch.seq_id[I] + NSeqMax, 0);
  }
  std::fill(Batch.logits, Batch.logits + NTokens, false);
  return Batch;
}

class LlamaBatchHolder {
public:
  LlamaBatchHolder() noexcept = default;
  ~LlamaBatchHolder() noexcept { reset(); }
  LlamaBatchHolder(const LlamaBatchHolder &) = delete;
  LlamaBatchHolder &operator=(const LlamaBatchHolder &) = delete;
  LlamaBatchHolder(LlamaBatchHolder &&Other) noexcept
      : Batch(Other.Batch), Initialized(Other.Initialized) {
    Other.Batch = llama_batch{};
    Other.Initialized = false;
  }
  LlamaBatchHolder &operator=(LlamaBatchHolder &&Other) noexcept {
    if (this != &Other) {
      reset();
      Batch = Other.Batch;
      Initialized = Other.Initialized;
      Other.Batch = llama_batch{};
      Other.Initialized = false;
    }
    return *this;
  }

  llama_batch &get() noexcept { return Batch; }
  const llama_batch &get() const noexcept { return Batch; }
  void reset() noexcept {
    if (Initialized) {
      llama_batch_free(Batch);
      Batch = llama_batch{};
      Initialized = false;
    }
  }
  void reset(llama_batch NewBatch) noexcept {
    reset();
    Batch = NewBatch;
    Initialized = true;
  }

private:
  llama_batch Batch{};
  bool Initialized = false;
};

} // namespace WasmEdge::Host::WASINN
#endif
