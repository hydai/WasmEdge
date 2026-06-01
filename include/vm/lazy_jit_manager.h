// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2026 Second State INC

//===-- wasmedge/vm/lazy_jit_manager.h - Lazy JIT manager -----------------===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the LazyJitManager, which owns lazy-JIT state and the
/// on-demand compilation pipeline so the VM can delegate it. Module resolution
/// stays with the VM (which owns the module storage); the manager is handed the
/// resolved module instance and AST module to compile against.
///
//===----------------------------------------------------------------------===//
#pragma once

#ifdef WASMEDGE_USE_LLVM

#include "ast/module.h"
#include "common/configure.h"
#include "common/errcode.h"
#include "llvm/jit.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace WasmEdge {
namespace Loader {
class Loader;
} // namespace Loader
namespace Runtime::Instance {
class FunctionInstance;
class ModuleInstance;
} // namespace Runtime::Instance
namespace VM {

/// Owns lazy-JIT state per module and performs on-demand compilation. Each
/// tracked module keeps its own copy of the AST module, so the manager resolves
/// what to compile from its own state (under its own lock) and holds no
/// pointers into the VM's module storage.
class LazyJitManager {
public:
  LazyJitManager(const Configure &Conf, Loader::Loader &Loader) noexcept
      : Conf(Conf), LoaderEngine(Loader) {}

  /// Compile the module infrastructure and track its state by module ID.
  /// Idempotent for an already-tracked ID: the existing dylib is reused (a
  /// re-instantiated or re-registered instance upgrades its FunctionInstances on
  /// demand). discardState() untracks an ID so a later prepare() rebuilds it.
  Expect<void> prepare(AST::Module &Module) noexcept;

  /// Compile the function and its statically-reachable batch on first use. A
  /// no-op when already compiled, an import, or the module is untracked. The
  /// AST module to compile against is the copy captured by prepare(), so this
  /// reads no VM-owned storage and is safe to call without the VM lock.
  Expect<void>
  compileFunction(const Runtime::Instance::ModuleInstance &ModInst,
                  const Runtime::Instance::FunctionInstance &Func) noexcept;

  /// Drop tracked state for a single module ID (e.g. a failed registration).
  void discardState(std::string_view ID) noexcept;

  /// Total number of compiled local functions across all tracked modules.
  uint32_t compiledFuncCount() const noexcept;

  /// Drop all tracked state.
  void clear() noexcept;

private:
  /// Per-module lazy-JIT state bundled with an owned copy of the AST module, so
  /// the manager compiles against its own module instead of reading VM storage.
  struct TrackedModule {
    LLVM::LazyJITState JIT;
    std::shared_ptr<const AST::Module> ASTModule;
    /// Memoized static call-graph closures keyed by local function index. A
    /// seed's closure never changes, so re-instantiated and sibling instances
    /// reuse it instead of re-walking the call graph. Bounded by the module's
    /// own functions and dropped with the TrackedModule.
    std::unordered_map<uint32_t, std::vector<uint32_t>> ClosureCache;
    /// Per-module lock for the compilation pipeline. The global Mutex
    /// (shared_mutex) serializes map access; this mutex serializes compilation
    /// within one module so different modules can compile concurrently.
    /// Heap-allocated so TrackedModule stays movable.
    mutable std::unique_ptr<std::mutex> CompileMutex =
        std::make_unique<std::mutex>();
  };

  const Configure Conf;
  Loader::Loader &LoaderEngine;
  mutable std::shared_mutex Mutex;
  std::unordered_map<std::string, TrackedModule> States;
};

} // namespace VM
} // namespace WasmEdge

#endif // WASMEDGE_USE_LLVM
