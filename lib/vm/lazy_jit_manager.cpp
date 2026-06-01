// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2026 Second State INC

#include "vm/lazy_jit_manager.h"

#ifdef WASMEDGE_USE_LLVM

#include "ast/instruction.h"
#include "common/span.h"
#include "executor/executor.h"
#include "llvm/compiler.h"
#include "loader/loader.h"
#include "runtime/instance/function.h"
#include "runtime/instance/module.h"

#include "spdlog/spdlog.h"

#include <fmt/format.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace WasmEdge {
namespace VM {

using namespace std::literals;

namespace {

/// Collect the transitive set of locally-defined functions reachable from a
/// seed via static call / return_call / ref.func edges. Indices are local
/// (import functions excluded). The full closure is returned regardless of
/// compile status; the caller decides which members still need compiling and
/// which are already in the dylib, because every reachable function has to be
/// wired on the calling instance.
void collectReachableClosure(uint32_t LocalSeed, const AST::Module *ModulePtr,
                             uint32_t ImportFuncCount,
                             std::vector<uint32_t> &OutSortedLocals) {
  OutSortedLocals.clear();
  if (!ModulePtr) {
    return;
  }
  const auto &CodeSec = ModulePtr->getCodeSection().getContent();
  const uint32_t DefinedCount = static_cast<uint32_t>(CodeSec.size());

  if (LocalSeed >= DefinedCount) {
    return;
  }

  std::vector<uint8_t> Visited(DefinedCount, 0);
  std::vector<uint32_t> Stack;
  Stack.reserve(64);

  Visited[LocalSeed] = 1;
  Stack.push_back(LocalSeed);
  OutSortedLocals.push_back(LocalSeed);

  while (!Stack.empty()) {
    const uint32_t L = Stack.back();
    Stack.pop_back();

    for (const auto &Instr : CodeSec[L].getExpr().getInstrs()) {
      const auto Op = Instr.getOpCode();
      if (Op == OpCode::Call || Op == OpCode::Return_call ||
          Op == OpCode::Ref__func) {
        const uint32_t Target = Instr.getTargetIndex();
        if (Target >= ImportFuncCount) {
          const uint32_t LocalIdx = Target - ImportFuncCount;
          if (LocalIdx < DefinedCount && !Visited[LocalIdx]) {
            Visited[LocalIdx] = 1;
            Stack.push_back(LocalIdx);
            OutSortedLocals.push_back(LocalIdx);
          }
        }
      }
    }
  }

  std::sort(OutSortedLocals.begin(), OutSortedLocals.end());
}

} // namespace

Expect<void> LazyJitManager::prepare(
    AST::Module &Module, std::shared_ptr<AST::Module> PreAllocated) noexcept {
  if (Module.getID().empty()) {
    // A module with no ID cannot be tracked: the ID keys per-module state and
    // the JIT symbol prefix ("m{ID}_"), so distinct empty-ID modules would all
    // collide on States[""] and be wired against each other's compiled code.
    // Such modules (caller-supplied, not assigned an ID by the lazy loader) run
    // in the interpreter, as they did before lazy JIT existed.
    return {};
  }
  {
    // Already prepared for this module ID: the same ID implies the same module
    // object, so the existing dylib is valid. A re-instantiated or re-registered
    // instance reuses it and upgrades its FunctionInstances on demand; rebuilding
    // would discard a usable dylib for no benefit. discardState() untracks an ID,
    // so a retry after a failed registration still rebuilds here.
    //
    // Still set the symbol on the incoming module even on the idempotent path:
    // the public API copies the caller's const module, so each copy needs its
    // own loadExecutable to bind the function-type wrappers from the shared
    // dylib.
    std::shared_lock Lock(Mutex);
    auto It = States.find(std::string(Module.getID()));
    if (It != States.end()) {
      if (!Module.getSymbol() && It->second.JIT.JITLib) {
        EXPECTED_TRY(LoaderEngine.loadExecutable(Module, It->second.JIT.JITLib));
      }
      return {};
    }
  }

  LLVM::Compiler ModuleCompiler(Conf);
  EXPECTED_TRY(ModuleCompiler.checkConfigure());

  auto Prefix = fmt::format("m{}_"sv, Module.getID());
  EXPECTED_TRY(auto LLModule,
               ModuleCompiler.compileInfrastructure(Module, Prefix));

  LLVM::LazyJITState State;
  State.LLData = std::move(LLModule.first);
  State.LLContext = std::move(LLModule.second);

  LLVM::JIT JITEngine(Conf);
  EXPECTED_TRY(auto Exec, JITEngine.load(State.LLData, true));
  State.JITLib = std::static_pointer_cast<LLVM::JITLibrary>(Exec);

  TrackedModule Tracked;
  Tracked.JIT = std::move(State);
  if (PreAllocated) {
    Tracked.ASTModule = std::move(PreAllocated);
  } else {
    Tracked.ASTModule = std::make_shared<const AST::Module>(Module);
  }
  Tracked.ImportFuncCount = Tracked.ASTModule->getImportedFunctionCount();

  EXPECTED_TRY(LoaderEngine.loadExecutable(Module, Tracked.JIT.JITLib));

  std::unique_lock Lock(Mutex);
  States.try_emplace(std::string(Module.getID()), std::move(Tracked));
  return {};
}

Expect<void> LazyJitManager::compileFunction(
    const Runtime::Instance::ModuleInstance &ModInst,
    const Runtime::Instance::FunctionInstance &Func) noexcept {
  const std::string ID = ModInst.getID();

  // Hold the global lock in shared mode so the States map stays alive while we
  // compile.  The per-module CompileMutex serializes compilation within one
  // module; different modules compile concurrently under the shared lock.
  // discardState/clear take the global lock in unique mode, which waits for all
  // ongoing compilations to finish before erasing.
  std::shared_lock Lock(Mutex);

  auto It = States.find(ID);
  if (It == States.end()) {
    spdlog::debug("[lazy-jit]: no JIT state for ID {}; skipping lazy "
                  "compilation"sv,
                  ID);
    return {};
  }
  std::unique_lock ModLock(*It->second.CompileMutex);
  LLVM::LazyJITState *StatePtr = &It->second.JIT;
  const AST::Module &Module = *It->second.ASTModule;
  const uint32_t ImportFuncCount = It->second.ImportFuncCount;
  LLVM::Data *LLDataPtr = &StatePtr->LLData;
  auto *LLContextPtr = &StatePtr->LLContext;

  auto FuncIdxRes = ModInst.getFuncIdx(&Func);
  if (!FuncIdxRes) {
    // The seed is not listed by its own module instance: an ownership
    // inconsistency that should not happen. Log it instead of swallowing it,
    // then fall back to the interpreter for this call.
    spdlog::error("[lazy-jit]: function not found in its module instance, "
                  "module ID: {}; running it in the interpreter"sv,
                  ID);
    return {};
  }
  const uint32_t FuncIdx = *FuncIdxRes;

  if (FuncIdx < ImportFuncCount) {
    return {};
  }
  const uint32_t LocalFuncIdx = FuncIdx - ImportFuncCount;

  // If already compiled or not a Wasm function, nothing to do.
  if (!Func.isWasmFunction() || Func.getCompiledCodePtr()) {
    return {};
  }

  if (!StatePtr->JITLib) {
    spdlog::error("[lazy-jit]: missing JIT library for module ID: {}"sv, ID);
    return Unexpect(ErrCode::Value::LazyCompilationError);
  }
  LLVM::JITLibrary &JITLib = *StatePtr->JITLib;
  LLVM::JIT JITEngine(Conf);

  // Wire the seed's entire statically-reachable closure on THIS instance, not
  // only the seed. A sibling ModuleInstance sharing this module ID may already
  // have compiled some of these functions into the shared dylib; their machine
  // code is reusable, but this instance's FunctionInstances still need upgrading
  // to it. Taking the full closure (rather than only the not-yet-compiled
  // functions) wires a reachable callee here, when its caller is compiled,
  // instead of leaving it stranded until something reaches it directly.
  // The static reachable closure of a seed never changes, so memoize it per
  // module ID; re-instantiated and sibling instances trigger the same seeds and
  // reuse the cached walk instead of re-traversing the call graph.
  auto ClosureIt = It->second.ClosureCache.find(LocalFuncIdx);
  if (ClosureIt == It->second.ClosureCache.end()) {
    std::vector<uint32_t> Computed;
    collectReachableClosure(LocalFuncIdx, &Module, ImportFuncCount, Computed);
    ClosureIt =
        It->second.ClosureCache.emplace(LocalFuncIdx, std::move(Computed)).first;
  }
  const std::vector<uint32_t> &Closure = ClosureIt->second;
  if (Closure.empty()) {
    return {};
  }

  // Split the closure into functions to compile now and functions already in the
  // dylib whose FunctionInstance on THIS module is still stranded. Resolve every
  // instance up front, before the irreversible JIT add, so the final wiring
  // cannot fail partway through.
  std::vector<uint32_t> ToCompile;
  std::vector<uint32_t> ToCompileGlobals;
  std::vector<Runtime::Instance::FunctionInstance *> ToCompileInsts;
  std::vector<uint32_t> ExistingGlobals;
  std::vector<Runtime::Instance::FunctionInstance *> ExistingInsts;
  for (uint32_t L : Closure) {
    const uint32_t WasmFuncIdx = ImportFuncCount + L;
    auto InstRes = ModInst.getFuncInst(WasmFuncIdx);
    if (!InstRes) {
      spdlog::error("[lazy-jit]: failed to get function instance for index {}, "
                    "module ID: {}"sv,
                    WasmFuncIdx, ID);
      return Unexpect(ErrCode::Value::WrongInstanceAddress);
    }
    auto *Inst = *InstRes;
    if (StatePtr->LazyCompiledFuncs.count(L) > 0) {
      // Already compiled into the dylib; wire this instance only if a sibling
      // left it stranded (its FunctionInstance has no code yet).
      if (!Inst->getCompiledCodePtr()) {
        ExistingGlobals.push_back(WasmFuncIdx);
        ExistingInsts.push_back(Inst);
      }
    } else {
      ToCompile.push_back(L);
      ToCompileGlobals.push_back(WasmFuncIdx);
      ToCompileInsts.push_back(Inst);
    }
  }

  // Resolve the already-compiled members' addresses from the existing dylib
  // before the add, keeping every fallible step ahead of the irreversible add.
  std::vector<LLVM::WasmFunctionCodeAddress> ExistingAddrs;
  if (!ExistingGlobals.empty()) {
    auto LkRes = JITEngine.lookupWasmFunctionSymbols(
        JITLib, LLDataPtr->getPrefix(), ExistingGlobals);
    if (!LkRes || LkRes->size() != ExistingGlobals.size()) {
      spdlog::error("[lazy-jit]: failed to resolve already-compiled functions, "
                    "module ID: {}"sv,
                    ID);
      return Unexpect(ErrCode::Value::LazyCompilationError);
    }
    ExistingAddrs = std::move(*LkRes);
  }

  // Compile the not-yet-compiled members as one batch and add them to the dylib.
  std::vector<LLVM::WasmFunctionCodeAddress> FreshAddrs;
  if (!ToCompile.empty()) {
    spdlog::debug("[lazy-jit]: Lazy compiling batch ({} local funcs) for wasm "
                  "entry local {}, module ID: {}"sv,
                  ToCompile.size(), LocalFuncIdx, ID);

    LLVM::Compiler BatchCompiler(Conf);
    auto ConfigResult = BatchCompiler.checkConfigure();
    if (!ConfigResult) {
      spdlog::error(
          "[lazy-jit]: Lazy JIT compiler config failed: {}, module ID: {}"sv,
          ConfigResult.error(), ID);
      return Unexpect(ConfigResult.error());
    }

    LLDataPtr->resetModule();
    if (auto CompileResult = BatchCompiler.compileFunctions(
            *LLDataPtr,
            static_cast<LLVM::Compiler::CompileContext *>(LLContextPtr->get()),
            Module, ToCompile);
        !CompileResult) {
      spdlog::error("[lazy-jit]: Lazy JIT function compilation failed: {}, "
                    "module ID: {}"sv,
                    CompileResult.error(), ID);
      return Unexpect(CompileResult.error());
    }

    // Wire the intrinsics table before the irreversible add: every step that can
    // fail must precede the add, so a later failure can never leave the dylib
    // holding symbols the compiled-set never records (which would brick the
    // function on retry, since re-adding the same symbols collides).
    if (auto IntrinsicsSymbol = JITLib.getIntrinsics()) {
      *IntrinsicsSymbol = &Executor::Executor::Intrinsics;
    } else {
      spdlog::error(
          "[lazy-jit]: failed to get intrinsics symbol, module ID: {}"sv, ID);
      return Unexpect(ErrCode::Value::LazyCompilationError);
    }

    auto AddrRes = JITEngine.add(JITLib, *LLDataPtr, ToCompileGlobals);
    if (!AddrRes) {
      spdlog::error("[lazy-jit]: Lazy JIT add failed: {}, module ID: {}"sv,
                    AddrRes.error(), ID);
      return Unexpect(AddrRes.error());
    }
    FreshAddrs = std::move(*AddrRes);
    if (FreshAddrs.size() != ToCompile.size()) {
      spdlog::error("[lazy-jit]: address count mismatch ({} vs {}), "
                    "module ID: {}"sv,
                    FreshAddrs.size(), ToCompile.size(), ID);
      return Unexpect(ErrCode::Value::LazyCompilationError);
    }
  }

  // The fallible work is done; from here nothing fails, so the dylib, the
  // upgraded FunctionInstances, and the compiled-set stay consistent. Publish
  // code into this instance's FunctionInstances, then record the freshly
  // compiled locals. Recording before upgrading would, on a later failure,
  // leave a function marked compiled yet still interpreted while a retry skipped
  // it.
  auto Wire = [&JITLib](Runtime::Instance::FunctionInstance *Inst,
                        LLVM::WasmFunctionCodeAddress Addr) {
    if (Inst->isWasmFunction()) {
      Inst->unsafeUpgradeToCompiled(JITLib.createSymbol(
          reinterpret_cast<Runtime::Instance::FunctionInstance::CompiledFunction
                               *>(Addr)));
    }
  };
  for (size_t I = 0; I < ExistingInsts.size(); ++I) {
    Wire(ExistingInsts[I], ExistingAddrs[I]);
  }
  for (size_t I = 0; I < ToCompileInsts.size(); ++I) {
    Wire(ToCompileInsts[I], FreshAddrs[I]);
  }
  for (uint32_t L : ToCompile) {
    StatePtr->LazyCompiledFuncs.insert(L);
  }

  spdlog::debug("[lazy-jit]: Lazy compilation completed ({} compiled, {} wired) "
                "for wasm entry local {}, total compiled: {}, module ID: {}"sv,
                ToCompile.size(), ExistingInsts.size(), LocalFuncIdx,
                StatePtr->LazyCompiledFuncs.size(), ID);

  return {};
}

void LazyJitManager::discardState(std::string_view ID) noexcept {
  std::unique_lock Lock(Mutex);
  States.erase(std::string(ID));
}

uint32_t LazyJitManager::compiledFuncCount() const noexcept {
  std::shared_lock Lock(Mutex);
  uint32_t Count = 0;
  for (const auto &Pair : States) {
    std::lock_guard ModLock(*Pair.second.CompileMutex);
    Count += static_cast<uint32_t>(Pair.second.JIT.LazyCompiledFuncs.size());
  }
  return Count;
}

void LazyJitManager::clear() noexcept {
  std::unique_lock Lock(Mutex);
  States.clear();
}

} // namespace VM
} // namespace WasmEdge

#endif // WASMEDGE_USE_LLVM
