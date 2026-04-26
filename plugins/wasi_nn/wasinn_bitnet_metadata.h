// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_bitnet.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

#include <string>

namespace WasmEdge::Host::WASINN::BitNet {

ErrNo parseMetadata(Graph &GraphRef, LocalConfig &ConfRef,
                    const std::string &Metadata, bool *IsModelUpdated = nullptr,
                    bool *IsContextUpdated = nullptr,
                    bool *IsSamplerUpdated = nullptr) noexcept;

} // namespace WasmEdge::Host::WASINN::BitNet

#endif
