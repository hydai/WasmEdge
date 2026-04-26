// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_bitnet.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

namespace WasmEdge::Host::WASINN::BitNet {

void llamaLogCallback(ggml_log_level LogLevel, const char *LogText,
                      void *UserData);
void releaseGraphResources(Graph &GraphRef, bool IsDebugLog) noexcept;
void releaseContextResources(Context &CxtRef, bool IsDebugLog) noexcept;

} // namespace WasmEdge::Host::WASINN::BitNet

#endif
