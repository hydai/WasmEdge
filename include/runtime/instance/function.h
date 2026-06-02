// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

//===-- wasmedge/runtime/instance/function.h - Function Instance definition ==//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the function instance definition in store manager.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "ast/instruction.h"
#include "common/symbol.h"
#include "runtime/hostfunc.h"
#include "runtime/instance/composite.h"

#include <atomic>
#include <memory>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

namespace WasmEdge {
namespace Runtime {
namespace Instance {

class ModuleInstance;

class FunctionInstance : public CompositeBase {
public:
  using CompiledFunction = void;

  FunctionInstance() = delete;
  /// Move constructor.
  FunctionInstance(FunctionInstance &&Inst) noexcept
      : CompositeBase(Inst.ModInst, Inst.TypeIdx), FuncType(Inst.FuncType),
        Data(std::move(Inst.Data)),
        LazyCompiledSymbol(std::move(Inst.LazyCompiledSymbol)),
        LazyCompiledCode(
            Inst.LazyCompiledCode.load(std::memory_order_relaxed)),
        LazyCompileUnavailable(
            Inst.LazyCompileUnavailable.load(std::memory_order_relaxed)) {
    assuming(ModInst);
    // The moved-from instance no longer owns LazyCompiledSymbol; clear its
    // published code pointer too so it consistently reports "not compiled"
    // (getCompiledCodePtr() == nullptr) rather than a stale entry into code it
    // no longer owns.
    Inst.LazyCompiledCode.store(nullptr, std::memory_order_relaxed);
  }
  /// Constructor for native function.
  FunctionInstance(const ModuleInstance *Mod, const uint32_t TIdx,
                   const AST::FunctionType &Type,
                   Span<const std::pair<uint32_t, ValType>> Locs,
                   AST::InstrView Expr) noexcept
      : CompositeBase(Mod, TIdx), FuncType(Type),
        Data(std::in_place_type_t<WasmFunction>(), Locs, Expr) {
    assuming(ModInst);
  }
  /// Constructor for compiled function.
  FunctionInstance(const ModuleInstance *Mod, const uint32_t TIdx,
                   const AST::FunctionType &Type,
                   Symbol<CompiledFunction> S) noexcept
      : CompositeBase(Mod, TIdx), FuncType(Type),
        Data(std::in_place_type_t<Symbol<CompiledFunction>>(), std::move(S)) {
    assuming(ModInst);
  }
  /// Constructors for host function.
  FunctionInstance(const ModuleInstance *Mod, const uint32_t TIdx,
                   std::unique_ptr<HostFunctionBase> &&Func) noexcept
      : CompositeBase(Mod, TIdx), FuncType(Func->getFuncType()),
        Data(std::in_place_type_t<std::unique_ptr<HostFunctionBase>>(),
             std::move(Func)) {
    assuming(ModInst);
  }
  FunctionInstance(std::unique_ptr<HostFunctionBase> &&Func) noexcept
      : CompositeBase(), FuncType(Func->getFuncType()),
        Data(std::in_place_type_t<std::unique_ptr<HostFunctionBase>>(),
             std::move(Func)) {}

  /// Check whether this is a native wasm function (not a host function). In
  /// lazy-JIT mode, a wasm function may also have compiled code
  /// (getCompiledCodePtr() != nullptr) published via unsafeUpgradeToCompiled;
  /// this accessor returns true regardless of compilation status. For
  /// compiled-vs-interpreted distinction, check getCompiledCodePtr() separately.
  bool isWasmFunction() const noexcept {
    return std::holds_alternative<WasmFunction>(Data);
  }

  /// Return the AOT-compiled code pointer (non-atomic), or nullptr if this is
  /// not an ahead-of-time-compiled function. Use this for fast-path checks that
  /// do not need to consult the lazy-JIT atomic.
  CompiledFunction *getAOTCompiledCodePtr() const noexcept {
    if (auto *Sym = std::get_if<Symbol<CompiledFunction>>(&Data)) {
      return Sym->get();
    }
    return nullptr;
  }

  /// Get the executable code pointer if this function has compiled code, either
  /// an ahead-of-time symbol or a lazily JIT-compiled entry, or nullptr if it
  /// must run in the interpreter. The lazily-published entry is read with
  /// acquire ordering, so this is safe to call concurrently with
  /// unsafeUpgradeToCompiled.
  ///
  /// This is the single source of truth for whether the function has compiled
  /// code, which is split between a Symbol in Data (ahead-of-time) and the
  /// LazyCompiledCode atomic (lazy JIT). The AOT variant is checked first to
  /// avoid an atomic load on the common AOT path. Always query through this
  /// accessor; inspecting Data directly misses a lazily-published entry.
  CompiledFunction *getCompiledCodePtr() const noexcept {
    if (auto *AOT = getAOTCompiledCodePtr()) {
      return AOT;
    }
    return getLazyCompiledCodePtr();
  }

  CompiledFunction *getLazyCompiledCodePtr() const noexcept {
    if (auto *LazyCode = LazyCompiledCode.load(std::memory_order_acquire)) {
      return LazyCode;
    }
    return nullptr;
  }

  /// Whether lazy compilation has already been attempted for this function and
  /// determined it cannot be compiled (module untracked, function not found,
  /// etc.). Avoids per-call string allocation + lock overhead for functions
  /// that will never receive compiled code. Relaxed ordering suffices: a stale
  /// false just retries once; a stale true harmlessly skips a no-op trigger.
  bool isLazyCompileUnavailable() const noexcept {
    return LazyCompileUnavailable.load(std::memory_order_relaxed);
  }
  void markLazyCompileUnavailable() const noexcept {
    LazyCompileUnavailable.store(true, std::memory_order_relaxed);
  }

  /// Check whether this is a host function.
  bool isHostFunction() const noexcept {
    return std::holds_alternative<std::unique_ptr<HostFunctionBase>>(Data);
  }

  /// Getter for function type.
  const AST::FunctionType &getFuncType() const noexcept { return FuncType; }

  /// Getter for function local variables.
  Span<const std::pair<uint32_t, ValType>> getLocals() const noexcept {
    return std::get_if<WasmFunction>(&Data)->Locals;
  }

  /// Getter for function local number.
  uint32_t getLocalNum() const noexcept {
    return std::get_if<WasmFunction>(&Data)->LocalNum;
  }

  /// Getter for function body instrs.
  AST::InstrView getInstrs() const noexcept {
    if (std::holds_alternative<WasmFunction>(Data)) {
      return std::get<WasmFunction>(Data).Instrs;
    } else {
      return {};
    }
  }

  /// Getter for host function.
  HostFunctionBase &getHostFunc() const noexcept {
    return *std::get_if<std::unique_ptr<HostFunctionBase>>(&Data)->get();
  }

  /// Publish lazily JIT-compiled code for this wasm function. The compiled
  /// symbol is stored alongside the existing WasmFunction (Data is left
  /// untouched, so the interpreter's instruction view stays valid) and the code
  /// pointer is then published with release ordering. A concurrent reader
  /// therefore observes either a not-yet-compiled function or a fully-published
  /// compiled entry, never a torn state. Must be called by a single compiler
  /// thread (serialized by the lazy JIT lock). Returns false for non-wasm
  /// functions.
  bool unsafeUpgradeToCompiled(Symbol<CompiledFunction> Sym) noexcept {
    if (!isWasmFunction()) {
      return false;
    }
    LazyCompiledSymbol = std::move(Sym);
    LazyCompiledCode.store(LazyCompiledSymbol.get(), std::memory_order_release);
    return true;
  }

private:
  struct WasmFunction {
    const std::vector<std::pair<uint32_t, ValType>> Locals;
    const uint32_t LocalNum;
    AST::InstrVec Instrs;
    WasmFunction(Span<const std::pair<uint32_t, ValType>> Locs,
                 AST::InstrView Expr) noexcept
        : Locals(Locs.begin(), Locs.end()),
          LocalNum(
              std::accumulate(Locals.begin(), Locals.end(), UINT32_C(0),
                              [](uint32_t N, const auto &Pair) -> uint32_t {
                                return N + Pair.first;
                              })) {
      // FIXME: Modify the capacity to prevent connecting 2 vectors.
      Instrs.reserve(Expr.size() + 1);
      Instrs.assign(Expr.begin(), Expr.end());
    }
  };

  /// \name Data of function instance.
  /// @{

  const AST::FunctionType &FuncType;
  std::variant<WasmFunction, Symbol<CompiledFunction>,
               std::unique_ptr<HostFunctionBase>>
      Data;
  /// Lazily JIT-compiled code, published once a wasm function is compiled on
  /// demand. Kept separate from Data so that Data is never mutated after
  /// construction: every variant accessor (and the interpreter's instruction
  /// view) is then free of data races with a concurrent lazy upgrade.
  /// LazyCompiledSymbol owns the JIT library lifetime and is written once under
  /// the lazy JIT lock; LazyCompiledCode is the published entry, stored with
  /// release and loaded with acquire ordering.
  Symbol<CompiledFunction> LazyCompiledSymbol;
  std::atomic<CompiledFunction *> LazyCompiledCode{nullptr};
  mutable std::atomic<bool> LazyCompileUnavailable{false};
  /// @}
};

} // namespace Instance
} // namespace Runtime
} // namespace WasmEdge
