// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "common/errcode.h"
#include "wasinntypes.h"

#include <string_view>

namespace WasmEdge {
namespace Host {
namespace WASINN {

struct WasiNNEnvironment;

struct BackendOperations {
  using LoadHandler = Expect<ErrNo> (*)(WasiNNEnvironment &,
                                        Span<const Span<uint8_t>>, Device,
                                        uint32_t &) noexcept;
  using InitExecCtxHandler = Expect<ErrNo> (*)(WasiNNEnvironment &, uint32_t,
                                               uint32_t &) noexcept;
  using SetInputHandler = Expect<ErrNo> (*)(WasiNNEnvironment &, uint32_t,
                                            uint32_t,
                                            const TensorData &) noexcept;
  using GetOutputHandler = Expect<ErrNo> (*)(WasiNNEnvironment &, uint32_t,
                                             uint32_t, Span<uint8_t>,
                                             uint32_t &) noexcept;
  using ComputeHandler = Expect<ErrNo> (*)(WasiNNEnvironment &,
                                           uint32_t) noexcept;
  using UnloadHandler = Expect<ErrNo> (*)(WasiNNEnvironment &,
                                          uint32_t) noexcept;

  const char *Name;
  Backend Type;
  LoadHandler Load;
  InitExecCtxHandler InitExecCtx;
  SetInputHandler SetInput;
  GetOutputHandler GetOutput;
  ComputeHandler Compute;
  GetOutputHandler GetOutputSingle;
  ComputeHandler ComputeSingle;
  ComputeHandler FiniSingle;
  UnloadHandler Unload;
  ComputeHandler FinalizeExecCtx;
};

enum class BackendPreloadMode : uint8_t {
  Bytes,
  Path,
};

struct BackendSelection {
  Backend Type;
  BackendPreloadMode Preload;
};

struct BackendDescriptor {
  Backend Type;
  std::string_view Name;
  std::string_view BuildOption;
  BackendPreloadMode Preload;
  const BackendOperations *Operations;
};

const BackendOperations *getBackendOperations(Backend Type) noexcept;
const BackendDescriptor *getBackendDescriptor(Backend Type) noexcept;
const BackendSelection *getBackendSelection(std::string_view Name) noexcept;

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge
