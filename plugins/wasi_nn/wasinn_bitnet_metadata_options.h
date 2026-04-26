// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_bitnet.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

#include "simdjson.h"

namespace WasmEdge::Host::WASINN::BitNet {

ErrNo parsePluginOptions(simdjson::dom::element &Doc, Graph &GraphRef) noexcept;

ErrNo parseModelOptions(simdjson::dom::element &Doc, Graph &GraphRef) noexcept;

} // namespace WasmEdge::Host::WASINN::BitNet

#endif
