// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinnfunc.h"
#include "wasinn_dispatch.h"
#include "wasinn_memory.h"
#include "wasinn_rpc_client.h"
#include "wasinnenv.h"

#include <string>
#include <utility>
#include <vector>

namespace WasmEdge {
namespace Host {

namespace {
using namespace std::literals;

struct NamedLoadMemory {
  uint32_t GraphIdPtr;
  Span<const uint8_t> Name;
};

Expected<NamedLoadMemory, WASINN::ErrNo>
getNamedLoadMemory(Runtime::Instance::MemoryInstance &MemInst, uint32_t NamePtr,
                   uint32_t NameLen, uint32_t GraphIdPtr) noexcept {
  if (unlikely(getGraphIdPointer(MemInst, GraphIdPtr) == nullptr)) {
    return Unexpected<WASINN::ErrNo>(WASINN::ErrNo::InvalidArgument);
  }

  auto Name = getModelData(MemInst, NamePtr, NameLen, "Name"sv);
  if (!Name) {
    return Unexpected<WASINN::ErrNo>(WASINN::ErrNo::InvalidArgument);
  }
  return NamedLoadMemory{GraphIdPtr, *Name};
}

template <typename CallT>
Expect<WASINN::ErrNo>
runOutputAndWriteSize(Runtime::Instance::MemoryInstance &MemInst,
                      const OutputBufferView &Output, CallT &&Call) {
  uint32_t NativeBytesWritten = 0;
  auto Res = std::forward<CallT>(Call)(Output.Buffer, NativeBytesWritten);
  if (!Res || Res.value() != WASINN::ErrNo::Success) {
    return Res;
  }
  if (auto Write = writeUInt32Result(MemInst, Output.BytesWrittenPtr,
                                     NativeBytesWritten, "BytesWritten"sv);
      !Write) {
    return WASINN::ErrNo::InvalidArgument;
  }
  return Res;
}
} // namespace

Expect<WASINN::ErrNo>
WasiNNLoad::bodyImpl(const Runtime::CallingFrame &Frame, uint32_t BuilderPtr,
                     uint32_t BuilderLen, uint32_t RawEncoding, uint32_t Target,
                     uint32_t GraphIdPtr) {
#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    return WASINN::RPC::load();
  }
#endif
  // Check memory instance from module.
  EXPECTED_TRY(auto *MemInst, getMemoryInstance(Frame));
  // Check the return value: GraphIdPtr should be valid.
  if (unlikely(getGraphIdPointer(*MemInst, GraphIdPtr) == nullptr)) {
    return WASINN::ErrNo::InvalidArgument;
  }
  auto Device = getDevice(Target);
  if (!Device) {
    return WASINN::ErrNo::InvalidArgument;
  }

  auto Builders = getGraphBuilders(*MemInst, BuilderPtr, BuilderLen);
  if (!Builders) {
    return WASINN::ErrNo::InvalidArgument;
  }

  auto Backend = static_cast<WASINN::Backend>(RawEncoding);
  uint32_t NativeGraphId = 0;
  auto Res =
      WASINN::Dispatch::load(Env, *Builders, Backend, *Device, NativeGraphId);
  if (!Res || Res.value() != WASINN::ErrNo::Success) {
    return Res;
  }
  if (auto Write =
          writeUInt32Result(*MemInst, GraphIdPtr, NativeGraphId, "GraphID"sv);
      !Write) {
    return WASINN::ErrNo::InvalidArgument;
  }
  return Res;
}

Expect<WASINN::ErrNo>
WasiNNLoadByName::bodyImpl(const Runtime::CallingFrame &Frame, uint32_t NamePtr,
                           uint32_t NameLen, uint32_t GraphIdPtr) {
  EXPECTED_TRY(auto *MemInst, getMemoryInstance(Frame));
  auto Input = getNamedLoadMemory(*MemInst, NamePtr, NameLen, GraphIdPtr);
  if (!Input) {
    return Input.error();
  }

#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    uint32_t NativeGraphId = 0;
    auto Res = WASINN::RPC::loadByName(Env, Input->Name, NativeGraphId);
    if (!Res || Res.value() != WASINN::ErrNo::Success) {
      return Res;
    }
    if (auto Write = writeUInt32Result(*MemInst, Input->GraphIdPtr,
                                       NativeGraphId, "GraphID"sv);
        !Write) {
      return WASINN::ErrNo::InvalidArgument;
    }
    return Res;
  }
#endif // ifdef WASMEDGE_BUILD_WASI_NN_RPC

  uint32_t NativeGraphId = 0;
  auto Res =
      WASINN::Dispatch::loadPreloadedModel(Env, Input->Name, NativeGraphId);
  if (!Res || Res.value() != WASINN::ErrNo::Success) {
    return Res;
  }
  if (auto Write = writeUInt32Result(*MemInst, Input->GraphIdPtr, NativeGraphId,
                                     "GraphID"sv);
      !Write) {
    return WASINN::ErrNo::InvalidArgument;
  }
  return Res;
}

Expect<WASINN::ErrNo> WasiNNLoadByNameWithConfig::bodyImpl(
    const Runtime::CallingFrame &Frame, uint32_t NamePtr, uint32_t NameLen,
    uint32_t ConfigPtr, uint32_t ConfigLen, uint32_t GraphIdPtr) {
  EXPECTED_TRY(auto *MemInst, getMemoryInstance(Frame));
  auto Input = getNamedLoadMemory(*MemInst, NamePtr, NameLen, GraphIdPtr);
  if (!Input) {
    return Input.error();
  }

  // Get the config of model
  auto Config = getModelData(*MemInst, ConfigPtr, ConfigLen, "Config"sv);
  if (!Config) {
    return WASINN::ErrNo::InvalidArgument;
  }

#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    uint32_t NativeGraphId = 0;
    auto Res = WASINN::RPC::loadByNameWithConfig(Env, Input->Name, *Config,
                                                 NativeGraphId);
    if (!Res || Res.value() != WASINN::ErrNo::Success) {
      return Res;
    }
    if (auto Write = writeUInt32Result(*MemInst, Input->GraphIdPtr,
                                       NativeGraphId, "GraphID"sv);
        !Write) {
      return WASINN::ErrNo::InvalidArgument;
    }
    return Res;
  }
#endif // ifdef WASMEDGE_BUILD_WASI_NN_RPC

  std::vector<uint8_t> ModelConfig(Config->begin(), Config->end());
  uint32_t NativeGraphId = 0;
  auto Res = WASINN::Dispatch::loadPreloadedModel(
      Env, Input->Name, NativeGraphId, std::move(ModelConfig));
  if (!Res || Res.value() != WASINN::ErrNo::Success) {
    return Res;
  }
  if (auto Write = writeUInt32Result(*MemInst, Input->GraphIdPtr, NativeGraphId,
                                     "GraphID"sv);
      !Write) {
    return WASINN::ErrNo::InvalidArgument;
  }
  return Res;
}

Expect<WASINN::ErrNo>
WasiNNInitExecCtx::bodyImpl(const Runtime::CallingFrame &Frame,
                            uint32_t GraphId, uint32_t ContextPtr) {
  EXPECTED_TRY(auto *MemInst, getMemoryInstance(Frame));

  // Check the return value: Context should be valid.
  if (unlikely(getContextPointer(*MemInst, ContextPtr) == nullptr)) {
    return WASINN::ErrNo::InvalidArgument;
  }

#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    uint32_t NativeContextId = 0;
    auto Res = WASINN::RPC::initExecCtx(Env, GraphId, NativeContextId);
    if (!Res || Res.value() != WASINN::ErrNo::Success) {
      return Res;
    }
    if (auto Write = writeUInt32Result(*MemInst, ContextPtr, NativeContextId,
                                       "Context"sv);
        !Write) {
      return WASINN::ErrNo::InvalidArgument;
    }
    return Res;
  }
#endif // ifdef WASMEDGE_BUILD_WASI_NN_RPC

  uint32_t NativeContextId = 0;
  auto Res = WASINN::Dispatch::initExecCtx(Env, GraphId, NativeContextId);
  if (!Res || Res.value() != WASINN::ErrNo::Success) {
    return Res;
  }
  if (auto Write =
          writeUInt32Result(*MemInst, ContextPtr, NativeContextId, "Context"sv);
      !Write) {
    return WASINN::ErrNo::InvalidArgument;
  }
  return Res;
}

Expect<WASINN::ErrNo>
WasiNNSetInput::bodyImpl(const Runtime::CallingFrame &Frame, uint32_t ContextId,
                         uint32_t Index, uint32_t TensorPtr) {
  EXPECTED_TRY(auto *MemInst, getMemoryInstance(Frame));

  auto Tensor = getTensor(*MemInst, TensorPtr);
  if (!Tensor) {
    return WASINN::ErrNo::InvalidArgument;
  }

#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    return WASINN::RPC::setInput(Env, *MemInst, ContextId, Index, *Tensor);
  }
#endif // ifdef WASMEDGE_BUILD_WASI_NN_RPC

  return WASINN::Dispatch::setInput(Env, ContextId, Index, Tensor->Data);
}

Expect<WASINN::ErrNo>
WasiNNGetOutput::bodyImpl(const Runtime::CallingFrame &Frame,
                          uint32_t ContextId, uint32_t Index,
                          uint32_t OutBufferPtr, uint32_t OutBufferMaxSize,
                          uint32_t BytesWrittenPtr) {
  EXPECTED_TRY(auto *MemInst, getMemoryInstance(Frame));
  auto Output = getOutputBuffer(*MemInst, OutBufferPtr, OutBufferMaxSize,
                                BytesWrittenPtr);
  if (!Output) {
    return WASINN::ErrNo::InvalidArgument;
  }

#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    return runOutputAndWriteSize(
        *MemInst, *Output,
        [&](Span<uint8_t> OutBuffer, uint32_t &NativeBytesWritten) {
          return WASINN::RPC::getOutput(Env, ContextId, Index, OutBuffer,
                                        NativeBytesWritten);
        });
  }
#endif // ifdef WASMEDGE_BUILD_WASI_NN_RPC

  return runOutputAndWriteSize(
      *MemInst, *Output,
      [&](Span<uint8_t> OutBuffer, uint32_t &NativeBytesWritten) {
        return WASINN::Dispatch::getOutput(Env, ContextId, Index, OutBuffer,
                                           NativeBytesWritten);
      });
}

Expect<WASINN::ErrNo> WasiNNGetOutputSingle::bodyImpl(
    const Runtime::CallingFrame &Frame, uint32_t ContextId, uint32_t Index,
    uint32_t OutBufferPtr, uint32_t OutBufferMaxSize,
    uint32_t BytesWrittenPtr) {
  EXPECTED_TRY(auto *MemInst, getMemoryInstance(Frame));
  auto Output = getOutputBuffer(*MemInst, OutBufferPtr, OutBufferMaxSize,
                                BytesWrittenPtr);
  if (!Output) {
    return WASINN::ErrNo::InvalidArgument;
  }

#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    return runOutputAndWriteSize(
        *MemInst, *Output,
        [&](Span<uint8_t> OutBuffer, uint32_t &NativeBytesWritten) {
          return WASINN::RPC::getOutputSingle(Env, ContextId, Index, OutBuffer,
                                              NativeBytesWritten);
        });
  }
#endif // ifdef WASMEDGE_BUILD_WASI_NN_RPC

  return runOutputAndWriteSize(
      *MemInst, *Output,
      [&](Span<uint8_t> OutBuffer, uint32_t &NativeBytesWritten) {
        return WASINN::Dispatch::getOutputSingle(Env, ContextId, Index,
                                                 OutBuffer, NativeBytesWritten);
      });
}

Expect<WASINN::ErrNo>
WasiNNCompute::bodyImpl(const Runtime::CallingFrame &Frame,
                        uint32_t ContextId) {
#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    return WASINN::RPC::compute(Env, ContextId);
  }
#endif // ifdef WASMEDGE_BUILD_WASI_NN_RPC
  EXPECTED_TRY(checkMemoryInstance(Frame));

  return WASINN::Dispatch::compute(Env, ContextId);
}

Expect<WASINN::ErrNo>
WasiNNComputeSingle::bodyImpl(const Runtime::CallingFrame &Frame,
                              uint32_t ContextId) {
#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    return WASINN::RPC::computeSingle(Env, ContextId);
  }
#endif // ifdef WASMEDGE_BUILD_WASI_NN_RPC
  EXPECTED_TRY(checkMemoryInstance(Frame));

  return WASINN::Dispatch::computeSingle(Env, ContextId);
}

Expect<WASINN::ErrNo>
WasiNNFiniSingle::bodyImpl(const Runtime::CallingFrame &Frame,
                           uint32_t ContextId) {
#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    return WASINN::RPC::finiSingle(Env, ContextId);
  }
#endif // ifdef WASMEDGE_BUILD_WASI_NN_RPC
  EXPECTED_TRY(checkMemoryInstance(Frame));

  return WASINN::Dispatch::finiSingle(Env, ContextId);
}

Expect<WASINN::ErrNo> WasiNNUnload::bodyImpl(const Runtime::CallingFrame &Frame,
                                             uint32_t GraphId) {
#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    return WASINN::RPC::unload();
  }
#endif
  EXPECTED_TRY(checkMemoryInstance(Frame));

  return WASINN::Dispatch::unload(Env, GraphId);
}

Expect<WASINN::ErrNo>
WasiNNFinalizeExecCtx::bodyImpl(const Runtime::CallingFrame &Frame,
                                uint32_t ContextId) {
#ifdef WASMEDGE_BUILD_WASI_NN_RPC
  if (Env.hasRPCChannel()) {
    return WASINN::RPC::finalizeExecCtx();
  }
#endif
  EXPECTED_TRY(checkMemoryInstance(Frame));

  return WASINN::Dispatch::finalizeExecCtx(Env, ContextId);
}

} // namespace Host
} // namespace WasmEdge
