// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinnenv.h"

#include "common/errcode.h"
#include "runtime/hostfunc.h"

#include <utility>

namespace WasmEdge {
namespace Host {

template <typename T> class WasiNN : public Runtime::HostFunction<T> {
public:
  WasiNN(WASINN::WasiNNEnvironment &HostEnv)
      : Runtime::HostFunction<T>(0), Env(HostEnv) {}

protected:
  static constexpr uint32_t castErrNo(WASINN::ErrNo E) noexcept {
    return static_cast<uint32_t>(E);
  }

  template <typename Body, typename... Args>
  Expect<uint32_t> runBody(const Runtime::CallingFrame &Frame, Body BodyFn,
                           Args &&...BodyArgs) {
    Env.setEnviron(&Frame);
    return (static_cast<T *>(this)->*BodyFn)(Frame,
                                             std::forward<Args>(BodyArgs)...)
        .map(castErrNo);
  }

  WASINN::WasiNNEnvironment &Env;
};

} // namespace Host
} // namespace WasmEdge
