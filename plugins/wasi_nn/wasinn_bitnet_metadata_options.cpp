// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_bitnet_metadata_options.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

#include "common/spdlog.h"
#include "wasinn_metadata.h"

#include <llama.h>
#include <string_view>

namespace WasmEdge::Host::WASINN::BitNet {
namespace {
using namespace std::literals;

#define RET_ERROR(Error, ...)                                                  \
  spdlog::error("[WASI-NN] BitNet backend: "sv __VA_ARGS__);                   \
  return Error;

} // namespace

ErrNo parsePluginOptions(simdjson::dom::element &Doc,
                         Graph &GraphRef) noexcept {
  if (Doc.at_key("enable-log").error() == simdjson::SUCCESS) {
    auto Err = Doc["enable-log"].get<bool>().get(GraphRef.EnableLog);
    if (Err) {
      RET_ERROR(ErrNo::InvalidArgument,
                "Unable to retrieve the enable-log option."sv)
    }
  }
  if (Doc.at_key("enable-debug-log").error() == simdjson::SUCCESS) {
    auto Err = Doc["enable-debug-log"].get<bool>().get(GraphRef.EnableDebugLog);
    if (Err) {
      RET_ERROR(ErrNo::InvalidArgument,
                "Unable to retrieve the enable-debug-log option."sv)
    }
  }
  return ErrNo::Success;
}

ErrNo parseModelOptions(simdjson::dom::element &Doc, Graph &GraphRef) noexcept {
  if (Doc.at_key("main-gpu").error() == simdjson::SUCCESS) {
    int64_t MainGPU;
    auto Err = Doc["main-gpu"].get<int64_t>().get(MainGPU);
    if (Err) {
      RET_ERROR(ErrNo::InvalidArgument,
                "Unable to retrieve the main-gpu option."sv)
    }
    GraphRef.Params.main_gpu = static_cast<int32_t>(MainGPU);
  }
  if (Doc.at_key("n-gpu-layers").error() == simdjson::SUCCESS) {
    int64_t NGPULayers;
    auto Err = Doc["n-gpu-layers"].get<int64_t>().get(NGPULayers);
    if (Err) {
      RET_ERROR(ErrNo::InvalidArgument,
                "Unable to retrieve the n-gpu-layers option."sv)
    }
    GraphRef.Params.n_gpu_layers = static_cast<int32_t>(NGPULayers);
  }
  if (Doc.at_key("tensor-split").error() == simdjson::SUCCESS) {
    std::string_view TSV;
    auto Err = Doc["tensor-split"].get<std::string_view>().get(TSV);
    if (Err) {
      RET_ERROR(ErrNo::InvalidArgument,
                "Unable to retrieve the tensor-split option."sv)
    }
    const auto TensorSplitCapacity = sizeof(GraphRef.Params.tensor_split) /
                                     sizeof(GraphRef.Params.tensor_split[0]);
    if (auto Res = parseTensorSplit(TSV, GraphRef.Params.tensor_split,
                                    TensorSplitCapacity, llama_max_devices(),
                                    "BitNet"sv);
        Res != ErrNo::Success) {
      return Res;
    }
  }
  if (Doc.at_key("embedding").error() == simdjson::SUCCESS) {
    auto Err = Doc["embedding"].get<bool>().get(GraphRef.Params.embedding);
    if (Err) {
      RET_ERROR(ErrNo::InvalidArgument,
                "Unable to retrieve the embedding option."sv)
    }
  }
  if (Doc.at_key("split-mode").error() == simdjson::SUCCESS) {
    std::string_view SplitMode;
    auto Err = Doc["split-mode"].get<std::string_view>().get(SplitMode);
    if (Err) {
      RET_ERROR(ErrNo::InvalidArgument,
                "Unable to retrieve the split-mode option."sv)
    }
    if (SplitMode == "none"sv) {
      GraphRef.Params.split_mode = LLAMA_SPLIT_MODE_NONE;
    } else if (SplitMode == "layer"sv) {
      GraphRef.Params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    } else if (SplitMode == "row"sv) {
      GraphRef.Params.split_mode = LLAMA_SPLIT_MODE_ROW;
    } else {
      RET_ERROR(ErrNo::InvalidArgument,
                "Unknown split-mode: {}. Valid: none, layer, row."sv, SplitMode)
    }
  }
  return ErrNo::Success;
}

} // namespace WasmEdge::Host::WASINN::BitNet

#endif
