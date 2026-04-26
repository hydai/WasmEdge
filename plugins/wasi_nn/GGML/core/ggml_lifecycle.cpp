// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "GGML/core/ggml_lifecycle.h"

#include "GGML/utils.h"
#include "common/spdlog.h"

#include <string>
#include <string_view>

namespace WasmEdge::Host::WASINN::GGML {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
namespace {
using namespace std::literals;

template <typename ResourceT>
void resetGraphResource(ResourceT &Resource, bool IsDebugLog,
                        std::string_view Name) noexcept {
  if (Resource == nullptr) {
    return;
  }
  if (IsDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: unload: free {}"sv, Name);
  }
  Resource.reset();
  if (IsDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: unload: free {}...Done"sv,
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
    spdlog::error("[WASI-NN] llama.cpp: {}"sv, Text);
  } else if (LogLevel == GGML_LOG_LEVEL_WARN) {
    spdlog::warn("[WASI-NN] llama.cpp: {}"sv, Text);
  } else if (LogLevel == GGML_LOG_LEVEL_INFO) {
    spdlog::info("[WASI-NN] llama.cpp: {}"sv, Text);
  } else if (LogLevel == GGML_LOG_LEVEL_DEBUG) {
    spdlog::debug("[WASI-NN] llama.cpp: {}"sv, Text);
  }
}

void releaseGraphResources(Graph &GraphRef, bool IsDebugLog) noexcept {
  resetGraphResource(GraphRef.LlamaModel, IsDebugLog, "llama model"sv);
  resetGraphResource(GraphRef.LlamaContext, IsDebugLog, "llama context"sv);
  resetGraphResource(GraphRef.VisionContext, IsDebugLog, "mtmd context"sv);
  resetGraphResource(GraphRef.VisionInputChunks, IsDebugLog, "mtmd chunks"sv);
  resetGraphResource(GraphRef.TTSModel, IsDebugLog, "TTS model"sv);
  resetGraphResource(GraphRef.TTSContext, IsDebugLog, "TTS context"sv);
  if (!GraphRef.TensorBuftOverrides.empty()) {
    LOG_DEBUG(IsDebugLog, "unload: free tensor buffer overrides"sv)
    GraphRef.TensorBuftOverrides.clear();
    LOG_DEBUG(IsDebugLog, "unload: free tensor buffer overrides...Done"sv)
  }
}

void releaseContextResources(Context &CxtRef, bool IsDebugLog) noexcept {
  if (CxtRef.LlamaSampler != nullptr) {
    LOG_DEBUG(IsDebugLog,
              "finalize_execution_context: free compute_single sampler"sv)
    CxtRef.LlamaSampler.reset();
    LOG_DEBUG(
        IsDebugLog,
        "finalize_execution_context: free compute_single sampler...Done"sv)
  }
  CxtRef.LlamaBatch.reset();
  CxtRef.OutputBatch.reset();
}

void applyDeviceSelection(common_params &Params, Device TargetDevice) noexcept {
  if (TargetDevice != Device::CPU) {
    return;
  }
  if (auto *CPUDevice = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
      CPUDevice != nullptr) {
    Params.devices.clear();
    Params.devices.push_back(CPUDevice);
  }
  Params.n_gpu_layers = 0;
  Params.no_kv_offload = true;
  Params.no_op_offload = true;
}

#endif
} // namespace WasmEdge::Host::WASINN::GGML
