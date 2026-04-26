// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_metadata.h"

#include "common/spdlog.h"

#include <algorithm>
#include <sstream>
#include <string>

namespace WasmEdge {
namespace Host {
namespace WASINN {
namespace {
using namespace std::literals;

std::string normalizeCommaSeparated(std::string_view Raw) {
  std::string Copy(Raw);
  std::replace(Copy.begin(), Copy.end(), ',', ' ');
  return Copy;
}

} // namespace

ErrNo parseCommaSeparatedIntegers(std::string_view Raw, std::vector<int> &Out,
                                  std::string_view Name,
                                  std::string_view BackendName) noexcept {
  std::stringstream SS(normalizeCommaSeparated(Raw));
  Out.clear();
  int Value = 0;
  while (SS >> Value) {
    Out.push_back(Value);
  }
  if (!SS.eof()) {
    spdlog::error("[WASI-NN] {} backend: Invalid {} option."sv, BackendName,
                  Name);
    return ErrNo::InvalidArgument;
  }
  return ErrNo::Success;
}

ErrNo parseTensorSplit(std::string_view Raw, float *TensorSplit,
                       size_t TensorSplitCapacity, size_t MaxDevices,
                       std::string_view BackendName) noexcept {
  if (MaxDevices > TensorSplitCapacity) {
    spdlog::error("[WASI-NN] {} backend: MaxDevices exceeds tensor-split "
                  "capacity."sv,
                  BackendName);
    return ErrNo::InvalidArgument;
  }

  std::fill_n(TensorSplit, TensorSplitCapacity, 0.0f);
  std::stringstream SS(normalizeCommaSeparated(Raw));
  size_t TensorSplitSize = 0;
  float Value = 0.0f;
  while (SS >> Value) {
    if (TensorSplitSize >= MaxDevices) {
      spdlog::error(
          "[WASI-NN] {} backend: Number of Tensor-Split is larger than "
          "MaxDevices, please reduce the size of tensor-split."sv,
          BackendName);
      return ErrNo::InvalidArgument;
    }
    if (Value < 0.0f) {
      spdlog::error("[WASI-NN] {} backend: Tensor-Split must contain "
                    "non-negative values."sv,
                    BackendName);
      return ErrNo::InvalidArgument;
    }
    TensorSplit[TensorSplitSize++] = Value;
  }
  if (!SS.eof()) {
    spdlog::error("[WASI-NN] {} backend: Unable to parse the tensor-split "
                  "option."sv,
                  BackendName);
    return ErrNo::InvalidArgument;
  }
  return ErrNo::Success;
}

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge
