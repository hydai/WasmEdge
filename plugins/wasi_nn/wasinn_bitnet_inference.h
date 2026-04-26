// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_bitnet.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

#include "common/expected.h"

#include <string>
#include <string_view>

namespace WasmEdge::Host::WASINN::BitNet {

std::string buildOutputMetadata(Context &CxtRef) noexcept;
void clearContext(Graph &GraphRef, Context &CxtRef) noexcept;
ErrNo evaluateInput(Graph &GraphRef, Context &CxtRef,
                    std::string_view LogPrefix) noexcept;
ErrNo sampleOutput(Graph &GraphRef, Context &CxtRef,
                   bool IsSingleTokenMode = false) noexcept;
Expect<ErrNo> getEmbedding(Graph &GraphRef, Context &CxtRef) noexcept;

} // namespace WasmEdge::Host::WASINN::BitNet

#endif
