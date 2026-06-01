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

  for (uint32_t I = 0; I < CodeSegs.size(); ++I) {
    auto Symbol = CodeSegs[I].getSymbol();
    const auto &FuncType =
        (*ModInst.getType(TypeIdxs[I]))->getCompositeType().getFuncType();
    if (Symbol != false) {
      ModInst.addFunc(TypeIdxs[I], FuncType, std::move(Symbol));
    } else {
      ModInst.addFunc(TypeIdxs[I], FuncType, CodeSegs[I].getLocals(),
                      CodeSegs[I].getExpr().getInstrs());
    }
  }
  return {};
}

} // namespace Executor
} // namespace WasmEdge
