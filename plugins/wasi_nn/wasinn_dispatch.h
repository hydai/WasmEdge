// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinntypes.h"

#include <string_view>
#include <vector>

namespace WasmEdge {
namespace Host {
namespace WASINN {

struct WasiNNEnvironment;

namespace Dispatch {

Expect<ErrNo> load(WasiNNEnvironment &Env, Span<const Span<uint8_t>> Builders,
                   Backend Backend, Device Device, uint32_t &GraphId);

Expect<ErrNo> loadPreloadedModel(WasiNNEnvironment &Env,
                                 Span<const uint8_t> Name, uint32_t &GraphId,
                                 std::vector<uint8_t> Config = {});

Expect<ErrNo> initExecCtx(WasiNNEnvironment &Env, uint32_t GraphId,
                          uint32_t &ContextId);

Expect<ErrNo> setInput(WasiNNEnvironment &Env, uint32_t ContextId,
                       uint32_t Index, const TensorData &Tensor);

Expect<ErrNo> getOutput(WasiNNEnvironment &Env, uint32_t ContextId,
                        uint32_t Index, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten);

Expect<ErrNo> getOutputSingle(WasiNNEnvironment &Env, uint32_t ContextId,
                              uint32_t Index, Span<uint8_t> OutBuffer,
                              uint32_t &BytesWritten);

Expect<ErrNo> compute(WasiNNEnvironment &Env, uint32_t ContextId);

Expect<ErrNo> computeSingle(WasiNNEnvironment &Env, uint32_t ContextId);

Expect<ErrNo> finiSingle(WasiNNEnvironment &Env, uint32_t ContextId);

Expect<ErrNo> unload(WasiNNEnvironment &Env, uint32_t GraphId);

Expect<ErrNo> finalizeExecCtx(WasiNNEnvironment &Env, uint32_t ContextId);

} // namespace Dispatch

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge
