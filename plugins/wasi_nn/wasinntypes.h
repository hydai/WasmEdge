// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "common/errcode.h"
#include "common/span.h"
#include "common/spdlog.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace WasmEdge::Host::WASINN {

enum class ErrNo : uint32_t {
  Success = 0,              // No error occurred.
  InvalidArgument = 1,      // Caller module passed an invalid argument.
  InvalidEncoding = 2,      // Invalid encoding.
  MissingMemory = 3,        // Caller module is missing a memory export.
  Busy = 4,                 // Device or resource busy.
  RuntimeError = 5,         // Runtime Error.
  UnsupportedOperation = 6, // Unsupported Operation.
  TooLarge = 7,             // Too Large.
  NotFound = 8,             // Not Found.
  EndOfSequence = 100,      // End of Sequence Found.
  ContextFull = 101,        // Context Full.
  PromptTooLong = 102,      // Prompt Too Long.
  ModelNotFound = 103,      // Model Not Found.
};

enum class TensorType : uint8_t {
  F16 = 0,
  F32 = 1,
  F64 = 2,
  U8 = 3,
  I32 = 4,
  I64 = 5
};

enum class Device : uint32_t { CPU = 0, GPU = 1, TPU = 2, AUTO = 3 };

enum class Backend : uint8_t {
  OpenVINO = 0,
  ONNX = 1,
  Tensorflow = 2,
  PyTorch = 3,
  TensorflowLite = 4,
  Autodetect = 5,
  GGML = 6,
  NeuralSpeed = 7,
  Whisper = 9,
  MLX = 10,
  Piper = 11,
  ChatTTS = 12,
  OpenVINOGenAI = 13,
  BitNet = 14,
};

#define FOR_EACH_BACKEND(F)                                                    \
  F(OpenVINO)                                                                  \
  F(ONNX)                                                                      \
  F(Tensorflow)                                                                \
  F(PyTorch)                                                                   \
  F(TensorflowLite)                                                            \
  F(GGML)                                                                      \
  F(NeuralSpeed)                                                               \
  F(Whisper)                                                                   \
  F(Piper)                                                                     \
  F(ChatTTS)                                                                   \
  F(MLX)                                                                       \
  F(OpenVINOGenAI)                                                             \
  F(BitNet)

struct TensorData {
  Span<uint32_t> Dimension;
  WASINN::TensorType RType;
  Span<uint8_t> Tensor;
};

inline std::string_view asStringView(Span<const uint8_t> Bytes) noexcept {
  return {reinterpret_cast<const char *>(Bytes.data()), Bytes.size()};
}

inline std::string_view asStringView(Span<uint8_t> Bytes) noexcept {
  return {reinterpret_cast<const char *>(Bytes.data()), Bytes.size()};
}

inline std::string asString(Span<const uint8_t> Bytes) {
  return std::string(asStringView(Bytes));
}

inline std::string asString(Span<uint8_t> Bytes) {
  return std::string(asStringView(Bytes));
}

template <typename T>
inline void appendObjectBytes(std::vector<uint8_t> &Output, const T &Value) {
  const auto *Bytes = reinterpret_cast<const uint8_t *>(&Value);
  Output.insert(Output.end(), Bytes, Bytes + sizeof(T));
}

template <typename T>
inline void appendTypedBytes(std::vector<uint8_t> &Output, const T *Data,
                             size_t ByteSize) {
  const auto *Bytes = reinterpret_cast<const uint8_t *>(Data);
  Output.insert(Output.end(), Bytes, Bytes + ByteSize);
}

std::string_view getBackendName(Backend BE) noexcept;

std::string_view getBackendBuildOption(Backend BE) noexcept;

inline std::string_view getDeviceName(Device Target) noexcept {
  using namespace std::literals;
  switch (Target) {
  case Device::CPU:
    return "CPU"sv;
  case Device::GPU:
    return "GPU"sv;
  case Device::TPU:
    return "TPU"sv;
  case Device::AUTO:
    return "AUTO"sv;
  default:
    return "Unknown"sv;
  }
}

inline Expect<ErrNo>
reportBackendNotSupported(std::string_view BackendName) noexcept {
  using namespace std::literals;
  spdlog::error("[WASI-NN] {} backend is not supported."sv, BackendName);
  return ErrNo::InvalidArgument;
}

inline Expect<ErrNo> reportBackendNotSupported(Backend BE) noexcept {
  return reportBackendNotSupported(getBackendName(BE));
}

inline Expect<ErrNo>
reportBackendNotBuilt(std::string_view BackendName,
                      std::string_view BuildOption) noexcept {
  using namespace std::literals;
  spdlog::error("[WASI-NN] {} backend is not built. use {} to build it."sv,
                BackendName, BuildOption);
  return ErrNo::InvalidArgument;
}

inline Expect<ErrNo> reportBackendNotBuilt(Backend BE) noexcept {
  return reportBackendNotBuilt(getBackendName(BE), getBackendBuildOption(BE));
}

inline Expect<ErrNo> reportBackendRemoved(std::string_view BackendName,
                                          std::string_view Reference) noexcept {
  using namespace std::literals;
  spdlog::error("[WASI-NN] {} backend is removed due to the upstream "
                "end-of-life. Reference: {}"sv,
                BackendName, Reference);
  return ErrNo::InvalidArgument;
}

inline Expect<ErrNo> reportBackendRemoved(Backend BE,
                                          std::string_view Reference) noexcept {
  return reportBackendRemoved(getBackendName(BE), Reference);
}

inline ErrNo checkBuilderCount(Span<const Span<uint8_t>> Builders,
                               size_t Expected, Backend BE) noexcept {
  using namespace std::literals;
  if (Builders.size() == Expected) {
    return ErrNo::Success;
  }
  spdlog::error(
      "[WASI-NN] {} backend: Wrong GraphBuilder Length {}, expect {}"sv,
      getBackendName(BE), Builders.size(), Expected);
  return ErrNo::InvalidArgument;
}

inline ErrNo checkDevice(Device Actual, Device Expected, Backend BE) noexcept {
  using namespace std::literals;
  if (Actual == Expected) {
    return ErrNo::Success;
  }
  spdlog::error("[WASI-NN] {} backend only supports {} target, got {}"sv,
                getBackendName(BE), getDeviceName(Expected),
                getDeviceName(Actual));
  return ErrNo::InvalidArgument;
}

inline Expected<std::string, ErrNo>
getDeviceString(Device TargetDevice, Backend BE, bool AllowAutoAsCPU) noexcept {
  using namespace std::literals;
  switch (TargetDevice) {
  case Device::AUTO:
    if (!AllowAutoAsCPU) {
      break;
    }
    [[fallthrough]];
  case Device::CPU:
    return "CPU"s;
  case Device::GPU:
    return "GPU"s;
  default:
    break;
  }
  spdlog::error("[WASI-NN] {} backend: Unsupported device type {}"sv,
                getBackendName(BE), getDeviceName(TargetDevice));
  return Unexpected<ErrNo>(ErrNo::InvalidArgument);
}

} // namespace WasmEdge::Host::WASINN

template <>
struct fmt::formatter<WasmEdge::Host::WASINN::TensorType>
    : fmt::formatter<std::string_view> {
  fmt::format_context::iterator format(WasmEdge::Host::WASINN::TensorType RType,
                                       fmt::format_context &Ctx) const {
    using namespace std::literals;
    std::string_view Name;
    switch (RType) {
    case WasmEdge::Host::WASINN::TensorType::F16:
      Name = "F16"sv;
      break;
    case WasmEdge::Host::WASINN::TensorType::F32:
      Name = "F32"sv;
      break;
    case WasmEdge::Host::WASINN::TensorType::F64:
      Name = "F64"sv;
      break;
    case WasmEdge::Host::WASINN::TensorType::U8:
      Name = "U8"sv;
      break;
    case WasmEdge::Host::WASINN::TensorType::I32:
      Name = "I32"sv;
      break;
    case WasmEdge::Host::WASINN::TensorType::I64:
      Name = "I64"sv;
      break;
    default:
      Name = "Unknown"sv;
    }
    return fmt::formatter<std::string_view>::format(Name, Ctx);
  }
};

template <>
struct fmt::formatter<WasmEdge::Host::WASINN::Device>
    : fmt::formatter<std::string_view> {
  fmt::format_context::iterator format(WasmEdge::Host::WASINN::Device Target,
                                       fmt::format_context &Ctx) const {
    return fmt::formatter<std::string_view>::format(
        WasmEdge::Host::WASINN::getDeviceName(Target), Ctx);
  }
};
