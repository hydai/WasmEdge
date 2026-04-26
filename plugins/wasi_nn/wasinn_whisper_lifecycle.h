// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_whisper.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_WHISPER

namespace WasmEdge::Host::WASINN::Whisper {

void releaseGraphResources(Graph &GraphRef, bool IsDebugLog) noexcept;

} // namespace WasmEdge::Host::WASINN::Whisper

#endif
