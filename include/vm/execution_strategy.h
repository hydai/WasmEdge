// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2026 Second State INC

//===-- wasmedge/vm/execution_strategy.h - Execution mode strategy --------===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ExecutionStrategy interface, which encapsulates the
/// per-RunMode behavior (interpreter / eager JIT / lazy JIT) so the VM does not
/// branch on the run mode throughout its lifecycle.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "ast/module.h"
#include "common/configure.h"
#include "common/errcode.h"

#include <cstdint>
#include <memory>
#include <string_view>

namespace WasmEdge {
namespace Loader {
class Loader;
} // namespace Loader
namespace Runtime::Instance {
class FunctionInstance;
class ModuleInstance;
} // namespace Runtime::Instance
namespace VM {

/// Encapsulates run-mode-specific behavior. One instance is selected per VM at
/// construction; the VM calls these hooks instead of branching on RunMode.
class ExecutionStrategy {
public:
  virtual ~ExecutionStrategy() = default;

  /// Hook run when a module is registered.
  virtual Expect<void> onModuleRegistered(AST::Module &Module) noexcept {
    (void)Module;
    return {};
  }

  /// Hook run when a module is instantiated (also covers re-instantiation).
  virtual Expect<void> onModuleInstantiated(AST::Module &Module) noexcept {
    (void)Module;
    return {};
  }

  /// Hook run when registration fails after onModuleRegistered, so any state
  /// prepared for the module can be discarded.
  virtual void onModuleRegistrationFailed(std::string_view ID) noexcept {
    (void)ID;
  }

  /// Hook run when a registered module is unregistered and no other live
  /// instance shares its ID, so per-module state for that ID can be dropped.
  virtual void onModuleUnregistered(std::string_view ID) noexcept { (void)ID; }

  /// Lazily compile a function before it executes (lazy mode only). The owning
  /// module instance identifies which per-module state to compile against; the
  /// AST module itself is held by the strategy, not passed in by the VM.
  virtual Expect<void>
  compileFunction(const Runtime::Instance::ModuleInstance &ModInst,
                  const Runtime::Instance::FunctionInstance &Func) noexcept {
    (void)ModInst;
    (void)Func;
    return {};
  }

  /// Drop any per-mode state held for compiled modules.
  virtual void cleanup() noexcept {}

  /// Number of lazily-compiled functions (lazy mode only).
  virtual uint32_t compiledFuncCount() const noexcept { return 0; }

  /// Whether the executor needs a compilation trigger installed (lazy mode).
  /// Pure virtual on purpose: a wrong default would silently disable lazy
  /// compilation (trigger never installed) with no diagnostic, so every
  /// strategy must declare its intent explicitly.
  virtual bool needsCompilationTrigger() const noexcept = 0;
};

/// Select the execution strategy for the configured run mode.
std::unique_ptr<ExecutionStrategy>
makeExecutionStrategy(const Configure &Conf, Loader::Loader &Loader) noexcept;

} // namespace VM
} // namespace WasmEdge
