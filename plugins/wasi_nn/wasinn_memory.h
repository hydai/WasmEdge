// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "common/errcode.h"
#include "runtime/callingframe.h"
#include "wasinntypes.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace WasmEdge {
namespace Host {

struct OutputBufferView {
  Span<uint8_t> Buffer;
  uint32_t BytesWrittenPtr;
};

struct DecodedTensor {
  WASINN::TensorData Data;
  uint32_t RawTensorPtr;
  uint32_t RawTensorLen;
};

Expect<Runtime::Instance::MemoryInstance *>
getMemoryInstance(const Runtime::CallingFrame &Frame) noexcept;

Expect<void> checkMemoryInstance(const Runtime::CallingFrame &Frame) noexcept;

uint32_t *getGraphIdPointer(Runtime::Instance::MemoryInstance &MemInst,
                            uint32_t GraphIdPtr) noexcept;

uint32_t *getContextPointer(Runtime::Instance::MemoryInstance &MemInst,
                            uint32_t ContextPtr) noexcept;

Expect<void> writeUInt32Result(Runtime::Instance::MemoryInstance &MemInst,
                               uint32_t Ptr, uint32_t Value,
                               std::string_view Name) noexcept;

std::optional<Span<const uint8_t>>
getModelData(Runtime::Instance::MemoryInstance &MemInst, uint32_t Ptr,
             uint32_t Len, std::string_view Name) noexcept;

std::optional<WASINN::Device> getDevice(uint32_t Target) noexcept;

std::optional<std::vector<Span<uint8_t>>>
getGraphBuilders(Runtime::Instance::MemoryInstance &MemInst,
                 uint32_t BuilderPtr, uint32_t BuilderLen) noexcept;

std::optional<OutputBufferView>
getOutputBuffer(Runtime::Instance::MemoryInstance &MemInst,
                uint32_t OutBufferPtr, uint32_t OutBufferMaxSize,
                uint32_t BytesWrittenPtr) noexcept;

std::optional<DecodedTensor>
getTensor(Runtime::Instance::MemoryInstance &MemInst,
          uint32_t TensorPtr) noexcept;

} // namespace Host
} // namespace WasmEdge
