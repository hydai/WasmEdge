// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2026 Second State INC

//===-- wasmedge/executor/compilation_trigger.h - Compilation trigger -----===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the CompilationTrigger interface, which lets the executor
/// request lazy compilation of a function without depending on the engine that
/// performs it.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "common/errcode.h"

namespace WasmEdge {
namespace Runtime::Instance {
class FunctionInstance;
} // namespace Runtime::Instance
namespace Executor {

/// Interface for lazily compiling a function before it executes. Implemented by
/// the owner of the lazy JIT state; the executor only holds a pointer to it.
class CompilationTrigger {
public:
  virtual ~CompilationTrigger() = default;

  /// Ensure the given function is compiled before it is executed. Implementers
  /// must treat already-compiled or non-lazy functions as a no-op, and perform
  /// any compilation-state checks under their own synchronization.
  virtual Expect<void>
  ensureCompiled(const Runtime::Instance::FunctionInstance &Func) noexcept = 0;
};

} // namespace Executor
} // namespace WasmEdge
