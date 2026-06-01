// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "executor/executor.h"

#include <cstdint>
#include <utility>

namespace WasmEdge {
namespace Executor {

// Instantiate function instance. See "include/executor/executor.h".
Expect<void> Executor::instantiate(Runtime::Instance::ModuleInstance &ModInst,
                                   const AST::FunctionSection &FuncSec,
                                   const AST::CodeSection &CodeSec) {

  // Get the function type indices.
  auto TypeIdxs = FuncSec.getContent();
  auto CodeSegs = CodeSec.getContent();

  if (CodeSegs.size() == 0) {
    return {};
  }
  // When the first code segment has a compiled symbol, check whether ALL
  // segments do. Full AOT and eager JIT always compile every function, so
  // all symbols are non-null and the fast path avoids a per-function branch.
  // Lazy JIT compiles incrementally, so after a partial compile some segments
  // have symbols and others don't; those must be constructed as WasmFunctions
  // so the lazy-compile gate (isWasmFunction()) can upgrade them on demand.
  if (CodeSegs[0].getSymbol() != false) {
    bool AllCompiled = true;
    for (uint32_t I = 1; I < CodeSegs.size(); ++I) {
      if (CodeSegs[I].getSymbol() == false) {
        AllCompiled = false;
        break;
      }
    }
    if (AllCompiled) {
      for (uint32_t I = 0; I < CodeSegs.size(); ++I) {
        auto Symbol = CodeSegs[I].getSymbol();
        ModInst.addFunc(
            TypeIdxs[I],
            (*ModInst.getType(TypeIdxs[I]))->getCompositeType().getFuncType(),
            std::move(Symbol));
      }
    } else {
      for (uint32_t I = 0; I < CodeSegs.size(); ++I) {
        auto Symbol = CodeSegs[I].getSymbol();
        if (Symbol != false) {
          ModInst.addFunc(
              TypeIdxs[I],
              (*ModInst.getType(TypeIdxs[I]))->getCompositeType().getFuncType(),
              std::move(Symbol));
        } else {
          ModInst.addFunc(
              TypeIdxs[I],
              (*ModInst.getType(TypeIdxs[I]))->getCompositeType().getFuncType(),
              CodeSegs[I].getLocals(), CodeSegs[I].getExpr().getInstrs());
        }
      }
    }
  } else {
    for (uint32_t I = 0; I < CodeSegs.size(); ++I) {
      ModInst.addFunc(
          TypeIdxs[I],
          (*ModInst.getType(TypeIdxs[I]))->getCompositeType().getFuncType(),
          CodeSegs[I].getLocals(), CodeSegs[I].getExpr().getInstrs());
    }
  }
  return {};
}

} // namespace Executor
} // namespace WasmEdge
