// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "GGML/core/ggml_core.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
#include <common.h>
#include <ggml-backend.h>

namespace WasmEdge::Host::WASINN::GGML {

void llamaLogCallback(ggml_log_level LogLevel, const char *LogText,
                      void *UserData);
void releaseGraphResources(Graph &GraphRef, bool IsDebugLog) noexcept;
void releaseContextResources(Context &CxtRef, bool IsDebugLog) noexcept;
void applyDeviceSelection(common_params &Params, Device TargetDevice) noexcept;

} // namespace WasmEdge::Host::WASINN::GGML
#endif
