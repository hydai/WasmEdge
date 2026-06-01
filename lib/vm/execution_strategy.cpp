// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2026 Second State INC

#include "vm/execution_strategy.h"

#include "spdlog/spdlog.h"

#include <memory>

#ifdef WASMEDGE_USE_LLVM
#include "llvm/compiler.h"
#include "llvm/jit.h"
#include "loader/loader.h"
#include "vm/lazy_jit_manager.h"
#endif

namespace WasmEdge {
namespace VM {

using namespace std::literals;

namespace {

/// Interpreter mode: no compilation; every other hook is a no-op default.
class InterpreterStrategy : public ExecutionStrategy {
public:
  bool needsCompilationTrigger() const noexcept override { return false; }
};

#ifdef WASMEDGE_USE_LLVM
/// Build a map_error handler for the eager-JIT pipeline: on a real error it logs
/// that the named stage failed (so the engine falls back to the interpreter) and
/// maps the error to Success, the behavior shared by every pipeline stage.
auto jitFallback(spdlog::level::level_enum Level, std::string_view Stage) {
  return [Level, Stage](uint32_t Err) {
    if (Err != ErrCode::Value::Success) {
      spdlog::log(Level,
                  "{} failed. Error code: {}, use interpreter mode instead."sv,
                  Stage, Err);
    }
    return ErrCode::Value::Success;
  };
}

/// Eager JIT mode: compile the whole module at instantiation, falling back to
/// the interpreter on any failure.
class AotJitStrategy : public ExecutionStrategy {
public:
  AotJitStrategy(const Configure &Conf, Loader::Loader &Loader) noexcept
      : Conf(Conf), LoaderEngine(Loader) {}

  Expect<void> eagerCompile(AST::Module &Module) noexcept {
    if (Module.getSymbol()) {
      return {};
    }
    LLVM::Compiler Compiler(Conf);
    Compiler.checkConfigure()
        .map_error(jitFallback(spdlog::level::err, "Compiler Configure"sv))
        .and_then([&]() { return Compiler.compile(Module); })
        .map_error(jitFallback(spdlog::level::err, "Compilation"sv))
        .and_then([&](auto LLModule) {
          LLVM::JIT JIT(Conf);
          return JIT.load(LLModule);
        })
        .map_error(jitFallback(spdlog::level::warn, "JIT"sv))
        .and_then([&](auto Exec) {
          return LoaderEngine.loadExecutable(Module, std::move(Exec));
        })
        .map_error(jitFallback(spdlog::level::warn, "Loader"sv));
    return {};
  }

  Expect<void>
  onModuleRegistered(AST::Module &Module,
                     std::shared_ptr<AST::Module>) noexcept override {
    return eagerCompile(Module);
  }

  Expect<void>
  onModuleInstantiated(AST::Module &Module,
                       std::shared_ptr<AST::Module>) noexcept override {
    return eagerCompile(Module);
  }

  bool needsModuleCopy() const noexcept override { return true; }
  bool needsCompilationTrigger() const noexcept override { return false; }

private:
  const Configure Conf;
  Loader::Loader &LoaderEngine;
};

/// Lazy JIT mode: compile module infrastructure up front and function bodies on
/// first use, delegating to a LazyJitManager.
class LazyJitStrategy : public ExecutionStrategy {
public:
  LazyJitStrategy(const Configure &Conf, Loader::Loader &Loader) noexcept
      : Manager(Conf, Loader) {
    spdlog::warn("Lazy JIT is an alpha and experimental feature, which is not "
                 "ready for production use."sv);
  }

  Expect<void>
  onModuleRegistered(AST::Module &Module,
                     std::shared_ptr<AST::Module> PreAllocated) noexcept override {
    if (Module.getSymbol()) {
      return {};
    }
    return Manager.prepare(Module, std::move(PreAllocated));
  }

  Expect<void>
  onModuleInstantiated(AST::Module &Module,
                       std::shared_ptr<AST::Module> PreAllocated) noexcept override {
    if (Module.getSymbol()) {
      return {};
    }
    return Manager.prepare(Module, std::move(PreAllocated));
  }

  void onModuleOrphaned(std::string_view ID) noexcept override {
    Manager.discardState(ID);
  }

  Expect<void>
  compileFunction(const Runtime::Instance::ModuleInstance &ModInst,
                  const Runtime::Instance::FunctionInstance &Func) noexcept
      override {
    return Manager.compileFunction(ModInst, Func);
  }

  void cleanup() noexcept override { Manager.clear(); }

  uint32_t compiledFuncCount() const noexcept override {
    return Manager.compiledFuncCount();
  }

  bool needsModuleCopy() const noexcept override { return true; }
  bool needsCompilationTrigger() const noexcept override { return true; }

private:
  LazyJitManager Manager;
};
#endif

} // namespace

std::unique_ptr<ExecutionStrategy>
makeExecutionStrategy(const Configure &Conf,
                      [[maybe_unused]] Loader::Loader &Loader) noexcept {
  const auto Mode = Conf.getRuntimeConfigure().getRunMode();
#ifdef WASMEDGE_USE_LLVM
  if (Mode == RunMode::LazyJIT) {
    return std::make_unique<LazyJitStrategy>(Conf, Loader);
  }
  if (Mode == RunMode::JIT) {
    return std::make_unique<AotJitStrategy>(Conf, Loader);
  }
#else
  if (Mode == RunMode::JIT || Mode == RunMode::LazyJIT) {
    spdlog::warn("JIT was requested but WasmEdge was built without LLVM, "
                 "falling back to interpreter."sv);
  }
#endif
  return std::make_unique<InterpreterStrategy>();
}

} // namespace VM
} // namespace WasmEdge
