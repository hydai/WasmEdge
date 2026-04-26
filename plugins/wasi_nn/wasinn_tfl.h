// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinntypes.h"

#include "plugin/plugin.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_TFLITE
#include "tensorflow/lite/c/c_api.h"
#include <memory>
#include <vector>
#endif

namespace WasmEdge::Host::WASINN {
struct WasiNNEnvironment;
}

namespace WasmEdge::Host::WASINN::TensorflowLite {

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_TFLITE
struct TfLiteModelDeleter {
  void operator()(TfLiteModel *Ptr) const noexcept {
    if (Ptr) {
      TfLiteModelDelete(Ptr);
    }
  }
};

struct TfLiteInterpreterDeleter {
  void operator()(TfLiteInterpreter *Ptr) const noexcept {
    if (Ptr) {
      TfLiteInterpreterDelete(Ptr);
    }
  }
};

using TfLiteModelPtr = std::unique_ptr<TfLiteModel, TfLiteModelDeleter>;
using TfLiteInterpreterPtr =
    std::unique_ptr<TfLiteInterpreter, TfLiteInterpreterDeleter>;

struct Graph {
  std::vector<unsigned char> TfLiteModData;
  TfLiteModelPtr TFLiteMod = nullptr;
};

struct Context {
public:
  Context(uint32_t, Graph &) noexcept {}
  TfLiteInterpreterPtr TFLiteInterp = nullptr;
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
} // namespace WasmEdge::Host::WASINN::TensorflowLite
