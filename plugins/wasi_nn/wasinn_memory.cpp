// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_memory.h"

#include "common/spdlog.h"
#include "common/types.h"

namespace WasmEdge {
namespace Host {
namespace {
using namespace std::literals;

struct WasiBuilderPair {
  uint32_t Ptr;
  uint32_t Len;
};

struct WasiTensorData {
  uint32_t DimensionPtr;
  uint32_t DimensionLen;
  uint32_t RType;
  uint32_t TensorPtr;
  uint32_t TensorLen;
};

bool isKnownTensorType(WASINN::TensorType RType) noexcept {
  switch (RType) {
  case WASINN::TensorType::F16:
  case WASINN::TensorType::F32:
  case WASINN::TensorType::F64:
  case WASINN::TensorType::U8:
  case WASINN::TensorType::I32:
  case WASINN::TensorType::I64:
    return true;
  default:
    return false;
  }
}

} // namespace

Expect<Runtime::Instance::MemoryInstance *>
getMemoryInstance(const Runtime::CallingFrame &Frame) noexcept {
  auto *MemInst = Frame.getMemoryByIndex(0);
  if (MemInst == nullptr) {
    return Unexpect(ErrCode::Value::HostFuncError);
  }
  return MemInst;
}

Expect<void> checkMemoryInstance(const Runtime::CallingFrame &Frame) noexcept {
  EXPECTED_TRY(getMemoryInstance(Frame));
  return {};
}

uint32_t *getGraphIdPointer(Runtime::Instance::MemoryInstance &MemInst,
                            uint32_t GraphIdPtr) noexcept {
  auto *GraphId = MemInst.getPointer<uint32_t *>(GraphIdPtr);
  if (unlikely(GraphId == nullptr)) {
    spdlog::error(
        "[WASI-NN] Failed when accessing the return GraphID memory."sv);
  }
  return GraphId;
}

uint32_t *getContextPointer(Runtime::Instance::MemoryInstance &MemInst,
                            uint32_t ContextPtr) noexcept {
  auto *Context = MemInst.getPointer<uint32_t *>(ContextPtr);
  if (unlikely(Context == nullptr)) {
    spdlog::error("[WASI-NN] Failed when accessing the Context memory."sv);
  }
  return Context;
}

Expect<void> writeUInt32Result(Runtime::Instance::MemoryInstance &MemInst,
                               uint32_t Ptr, uint32_t Value,
                               std::string_view Name) noexcept {
  auto Res = MemInst.storeValue(Value, Ptr);
  if (!Res) {
    spdlog::error("[WASI-NN] Failed when writing the {} memory."sv, Name);
    return Unexpect(Res.error());
  }
  return {};
}

std::optional<Span<const uint8_t>>
getModelData(Runtime::Instance::MemoryInstance &MemInst, uint32_t Ptr,
             uint32_t Len, std::string_view Name) noexcept {
  auto Data = MemInst.getSpan<const uint8_t>(Ptr, Len);
  if (unlikely(Data.size() != Len)) {
    spdlog::error("[WASI-NN] Failed when accessing the {} memory."sv, Name);
    return std::nullopt;
  }
  return Data;
}

std::optional<WASINN::Device> getDevice(uint32_t Target) noexcept {
  const auto Device = static_cast<WASINN::Device>(Target);
  switch (Device) {
  case WASINN::Device::CPU:
  case WASINN::Device::GPU:
  case WASINN::Device::TPU:
  case WASINN::Device::AUTO:
    spdlog::debug("[WASI-NN] Using device: {}."sv, Device);
    return Device;
  default:
    spdlog::error("[WASI-NN] Unknown device {}."sv, Target);
    return std::nullopt;
  }
}

std::optional<std::vector<Span<uint8_t>>>
getGraphBuilders(Runtime::Instance::MemoryInstance &MemInst,
                 uint32_t BuilderPtr, uint32_t BuilderLen) noexcept {
  const auto WasiBuilders =
      MemInst.getSpan<const WasiBuilderPair>(BuilderPtr, BuilderLen);
  if (unlikely(WasiBuilders.size() != BuilderLen)) {
    spdlog::error("[WASI-NN] Failed when accessing the GraphBuilder memory."sv);
    return std::nullopt;
  }

  std::vector<Span<uint8_t>> Builders;
  Builders.reserve(BuilderLen);
  for (size_t I = 0; I < WasiBuilders.size(); ++I) {
    const auto &WasiBuilder = WasiBuilders[I];
    const auto BuilderLen = EndianValue(WasiBuilder.Len).le();
    auto Builder =
        MemInst.getSpan<uint8_t>(EndianValue(WasiBuilder.Ptr).le(), BuilderLen);
    if (unlikely(Builder.size() != BuilderLen)) {
      spdlog::error("[WASI-NN] Failed when accessing the Builder[{}] memory."sv,
                    I);
      return std::nullopt;
    }
    Builders.emplace_back(Builder);
  }
  return Builders;
}

std::optional<OutputBufferView>
getOutputBuffer(Runtime::Instance::MemoryInstance &MemInst,
                uint32_t OutBufferPtr, uint32_t OutBufferMaxSize,
                uint32_t BytesWrittenPtr) noexcept {
  const auto OutBuffer =
      MemInst.getSpan<uint8_t>(OutBufferPtr, OutBufferMaxSize);
  if (unlikely(OutBuffer.size() != OutBufferMaxSize)) {
    spdlog::error(
        "[WASI-NN] Failed when accessing the Output Buffer memory."sv);
    return std::nullopt;
  }

  auto *BytesWritten = MemInst.getPointer<uint32_t *>(BytesWrittenPtr);
  if (unlikely(BytesWritten == nullptr)) {
    spdlog::error("[WASI-NN] Failed when accessing the BytesWritten memory."sv);
    return std::nullopt;
  }

  return OutputBufferView{OutBuffer, BytesWrittenPtr};
}

std::optional<DecodedTensor>
getTensor(Runtime::Instance::MemoryInstance &MemInst,
          uint32_t TensorPtr) noexcept {
  auto *WasiTensor = MemInst.getPointer<const WasiTensorData *>(TensorPtr);
  if (unlikely(WasiTensor == nullptr)) {
    spdlog::error("[WASI-NN] Failed when accessing the Tensor memory."sv);
    return std::nullopt;
  }

  WASINN::TensorData Tensor;
  const auto DimensionLen = EndianValue(WasiTensor->DimensionLen).le();
  Tensor.Dimension = MemInst.getSpan<uint32_t>(
      EndianValue(WasiTensor->DimensionPtr).le(), DimensionLen);
  if (unlikely(Tensor.Dimension.size() != DimensionLen)) {
    spdlog::error("[WASI-NN] Failed when accessing the Dimension memory."sv);
    return std::nullopt;
  }

  const auto TensorLen = EndianValue(WasiTensor->TensorLen).le();
  Tensor.Tensor = MemInst.getSpan<uint8_t>(
      EndianValue(WasiTensor->TensorPtr).le(), TensorLen);
  if (unlikely(Tensor.Tensor.size() != TensorLen)) {
    spdlog::error("[WASI-NN] Failed when accessing the TensorData memory."sv);
    return std::nullopt;
  }

  const auto RType =
      static_cast<WASINN::TensorType>(EndianValue(WasiTensor->RType).le());
  if (!isKnownTensorType(RType)) {
    spdlog::error("[WASI-NN] Unknown tensor type {}."sv,
                  static_cast<uint32_t>(RType));
    return std::nullopt;
  }
  Tensor.RType = RType;

  return DecodedTensor{Tensor, EndianValue(WasiTensor->TensorPtr).le(),
                       TensorLen};
}

} // namespace Host
} // namespace WasmEdge
