// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinntypes.h"

#include "plugin/plugin.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_CHATTTS
#include <Python.h>
#include <memory>
#include <utility>
#endif

namespace WasmEdge::Host::WASINN {
struct WasiNNEnvironment;
}

namespace WasmEdge::Host::WASINN::ChatTTS {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_CHATTTS
class GIL {
private:
  PyGILState_STATE GState;

public:
  GIL() : GState(PyGILState_Ensure()) {}
  ~GIL() { PyGILState_Release(GState); }
  GIL(const GIL &) = delete;
  GIL &operator=(const GIL &) = delete;
};

struct PyObjectDeleter {
  void operator()(PyObject *Ptr) const noexcept {
    if (Ptr != nullptr && Py_IsInitialized()) {
      Py_XDECREF(Ptr);
    }
  }
};

using PyObjectPtr = std::unique_ptr<PyObject, PyObjectDeleter>;

struct Graph {
  bool EnableDebugLog = false;
  Graph() noexcept {
    if (!Py_IsInitialized()) {
      Py_Initialize();
      if (PyGILState_Check()) {
        PyEval_SaveThread();
      }
    }
  }
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph(Graph &&) noexcept = default;
  Graph &operator=(Graph &&Other) noexcept {
    if (this != &Other) {
      release();
      EnableDebugLog = Other.EnableDebugLog;
      Chat = std::move(Other.Chat);
      ChatTTSModule = std::move(Other.ChatTTSModule);
      ParamsRefineText = std::move(Other.ParamsRefineText);
      ParamsInferCode = std::move(Other.ParamsInferCode);
    }
    return *this;
  }
  ~Graph() noexcept {
    if (Py_IsInitialized()) {
      release();
    }
  }
  void release() noexcept {
    if (!Py_IsInitialized()) {
      return;
    }
    GIL Lock;
    ParamsRefineText.reset();
    ParamsInferCode.reset();
    Chat.reset();
    ChatTTSModule.reset();
  }
  PyObjectPtr Chat = nullptr;
  PyObjectPtr ChatTTSModule = nullptr;
  PyObjectPtr ParamsRefineText = nullptr;
  PyObjectPtr ParamsInferCode = nullptr;
};
struct Context {
  Context(uint32_t, Graph &) noexcept {}
  std::string Inputs;
  std::vector<uint8_t> Outputs;
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
Expect<WASINN::ErrNo> unload(WASINN::WasiNNEnvironment &Env,
                             uint32_t GraphId) noexcept;
} // namespace WasmEdge::Host::WASINN::ChatTTS
