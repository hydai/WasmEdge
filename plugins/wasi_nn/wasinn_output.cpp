// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_output.h"

#include "common/spdlog.h"

#include <algorithm>
#include <limits>

namespace WasmEdge::Host::WASINN {

ErrNo copyBytesToBuffer(Span<const uint8_t> Bytes, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten) noexcept {
  using namespace std::literals;
  if (Bytes.size() > std::numeric_limits<uint32_t>::max()) {
    spdlog::error("[WASI-NN] Output size {} is too large."sv, Bytes.size());
    BytesWritten = 0;
    return ErrNo::TooLarge;
  }
  const auto BytesToCopy = std::min(OutBuffer.size(), Bytes.size());
  std::copy_n(Bytes.begin(), BytesToCopy, OutBuffer.begin());
  BytesWritten = static_cast<uint32_t>(Bytes.size());
  return ErrNo::Success;
}

ErrNo copyStringToBuffer(std::string_view Text, Span<uint8_t> OutBuffer,
                         uint32_t &BytesWritten) noexcept {
  return copyBytesToBuffer(
      {reinterpret_cast<const uint8_t *>(Text.data()), Text.size()}, OutBuffer,
      BytesWritten);
}

} // namespace WasmEdge::Host::WASINN
