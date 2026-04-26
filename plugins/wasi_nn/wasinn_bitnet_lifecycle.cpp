// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_bitnet_lifecycle.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

#include "common/spdlog.h"

#include <string>
#include <string_view>

namespace WasmEdge::Host::WASINN::BitNet {
namespace {
using namespace std::literals;

template <typename ResourceT>
void resetGraphResource(ResourceT &Resource, bool IsDebugLog,
                        std::string_view Name) noexcept {
  if (Resource == nullptr) {
    return;
  }
  if (IsDebugLog) {
    spdlog::info("[WASI-NN][Debug] BitNet backend: unload: free {}"sv, Name);
  }
  Resource.reset();
  if (IsDebugLog) {
    spdlog::info("[WASI-NN][Debug] BitNet backend: unload: free {}...Done"sv,
                 Name);
  }
}

} // namespace

void llamaLogCallback(ggml_log_level LogLevel, const char *LogText,
                      void *UserData) {
  Graph &GraphRef = *reinterpret_cast<Graph *>(UserData);
  if (!GraphRef.EnableLog) {
    return;
  }
  std::string Text(LogText);
  Text = Text.erase(Text.find_last_not_of("\n") + 1);
  if (Text == ".") {
    return;
  }
  if (LogLevel == GGML_LOG_LEVEL_ERROR) {
    spdlog::error("[WASI-NN] BitNet.cpp: {}"sv, Text);
  } else if (LogLevel == GGML_LOG_LEVEL_WARN) {
    spdlog::warn("[WASI-NN] BitNet.cpp: {}"sv, Text);
  } else if (LogLevel == GGML_LOG_LEVEL_INFO) {
    spdlog::info("[WASI-NN] BitNet.cpp: {}"sv, Text);
  } else if (LogLevel == GGML_LOG_LEVEL_DEBUG) {
    spdlog::debug("[WASI-NN] BitNet.cpp: {}"sv, Text);
  }
}

void releaseGraphResources(Graph &GraphRef, bool IsDebugLog) noexcept {
  resetGraphResource(GraphRef.LlamaContext, IsDebugLog, "llama context"sv);
  resetGraphResource(GraphRef.LlamaModel, IsDebugLog, "llama model"sv);
}

void releaseContextResources(Context &CxtRef, bool IsDebugLog) noexcept {
  if (CxtRef.LlamaSampler != nullptr) {
    if (IsDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] BitNet backend: finalizeExecCtx: free sampler"sv);
    }
    CxtRef.LlamaSampler.reset();
    if (IsDebugLog) {
      spdlog::info("[WASI-NN][Debug] BitNet backend: finalizeExecCtx: free "
                   "sampler...Done"sv);
    }
  }
  CxtRef.LlamaBatch.reset();
  CxtRef.OutputBatch.reset();
}

} // namespace WasmEdge::Host::WASINN::BitNet

#endif
