// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_memory.h"
#include "wasinntypes.h"

#ifdef WASMEDGE_BUILD_WASI_NN_RPC

namespace WasmEdge {
namespace Host {
namespace WASINN {

struct WasiNNEnvironment;

namespace RPC {

Expect<ErrNo> load() noexcept;

Expect<ErrNo> loadByName(WasiNNEnvironment &Env, Span<const uint8_t> Name,
                         uint32_t &GraphId);

Expect<ErrNo> loadByNameWithConfig(WasiNNEnvironment &Env,
                                   Span<const uint8_t> Name,
                                   Span<const uint8_t> Config,
                                   uint32_t &GraphId);

Expect<ErrNo> initExecCtx(WasiNNEnvironment &Env, uint32_t GraphId,
                          uint32_t &ContextId);

Expect<ErrNo> setInput(WasiNNEnvironment &Env,
                       Runtime::Instance::MemoryInstance &MemInst,
                       uint32_t ContextId, uint32_t Index,
                       const DecodedTensor &Tensor);

Expect<ErrNo> getOutput(WasiNNEnvironment &Env, uint32_t ContextId,
                        uint32_t Index, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten);

Expect<ErrNo> getOutputSingle(WasiNNEnvironment &Env, uint32_t ContextId,
                              uint32_t Index, Span<uint8_t> OutBuffer,
                              uint32_t &BytesWritten);

Expect<ErrNo> compute(WasiNNEnvironment &Env, uint32_t ContextId);

Expect<ErrNo> computeSingle(WasiNNEnvironment &Env, uint32_t ContextId);

Expect<ErrNo> finiSingle(WasiNNEnvironment &Env, uint32_t ContextId);

Expect<ErrNo> unload() noexcept;

Expect<ErrNo> finalizeExecCtx() noexcept;

} // namespace RPC

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge

#endif
