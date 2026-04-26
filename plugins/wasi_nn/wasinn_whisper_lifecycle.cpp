// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_whisper_lifecycle.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_WHISPER

#include "common/spdlog.h"

using namespace std::literals;

namespace WasmEdge::Host::WASINN::Whisper {

void releaseGraphResources(Graph &GraphRef, bool IsDebugLog) noexcept {
  if (GraphRef.WhisperCtx == nullptr) {
    return;
  }
  if (IsDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] Whisper backend: unload: free whisper context"sv);
  }
  GraphRef.WhisperCtx.reset();
  if (IsDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] Whisper backend: unload: free whisper context...Done"sv);
  }
}

} // namespace WasmEdge::Host::WASINN::Whisper

#endif
