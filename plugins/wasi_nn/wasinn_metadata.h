// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinntypes.h"

#include <cstddef>
#include <string_view>
#include <vector>

namespace WasmEdge {
namespace Host {
namespace WASINN {

ErrNo parseCommaSeparatedIntegers(std::string_view Raw, std::vector<int> &Out,
                                  std::string_view Name,
                                  std::string_view BackendName) noexcept;

ErrNo parseTensorSplit(std::string_view Raw, float *TensorSplit,
                       size_t TensorSplitCapacity, size_t MaxDevices,
                       std::string_view BackendName) noexcept;

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge
