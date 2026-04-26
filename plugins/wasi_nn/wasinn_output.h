// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinntypes.h"

#include <cstdint>
#include <string_view>

namespace WasmEdge::Host::WASINN {

ErrNo copyBytesToBuffer(Span<const uint8_t> Bytes, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten) noexcept;

ErrNo copyStringToBuffer(std::string_view Text, Span<uint8_t> OutBuffer,
                         uint32_t &BytesWritten) noexcept;

} // namespace WasmEdge::Host::WASINN
