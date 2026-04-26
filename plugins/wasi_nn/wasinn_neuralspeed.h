// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinntypes.h"

namespace WasmEdge::Host::WASINN::NeuralSpeed {
struct Graph {};
struct Context {
  Context(uint32_t, Graph &) noexcept {}
};

struct Environ {};
} // namespace WasmEdge::Host::WASINN::NeuralSpeed
