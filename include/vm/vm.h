// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

//===-- wasmedge/vm/vm.h - VM execution flow class definition -------------===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the VM class.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "common/async.h"
#include "common/configure.h"
#include "common/errcode.h"
#include "common/filesystem.h"
#include "common/types.h"

#include "executor/execution_lock.h"
#include "executor/executor.h"
#include "loader/loader.h"
#include "validator/validator.h"

#include "runtime/instance/module.h"
#include "runtime/storemgr.h"
#include "vm/execution_strategy.h"

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace WasmEdge {
namespace VM {

/// VM execution flow class
class VM : public Executor::CompilationTrigger {
public:
  VM() = delete;
  VM(const Configure &Conf);
  VM(const Configure &Conf, Runtime::StoreManager &S);
  ~VM() {
    // Teardown runs through the same primitives as cleanup(). Per getExecutor()'s
    // lifetime contract the caller must ensure no borrowed executor is still
    // running on another thread at destruction, so these run unsynchronized like
    // any other single-threaded destructor.
    terminateModuleInstance(ActiveModInst.release());
    cleanupModInstContainer(RegModInsts);
    cleanupModInstContainer(BuiltInModInsts);
    cleanupModInstContainer(PlugInModInsts);
  }

  /// ======= Functions can be called before the instantiated stage. =======
  /// Register wasm modules and host modules.
  Expect<void> registerModule(std::string_view Name,
                              const std::filesystem::path &Path) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeRegisterModule(Name, Path);
  }
  Expect<void> registerModule(std::string_view Name, Span<const Byte> Code) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeRegisterModule(Name, Code);
  }
  /// Register a module by mutable reference (zero-copy). In JIT/lazy-JIT mode
  /// the module is mutated in place (compiled symbols are embedded via
  /// loadExecutable). Use the const overload to leave the caller's module
  /// untouched.
  Expect<void> registerModule(std::string_view Name, AST::Module &Module) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeRegisterModule(Name, Module);
  }
  Expect<void> registerModule(std::string_view Name,
                              const AST::Module &Module) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    if (Strategy->needsModuleCopy()) {
      auto ModPtr = std::make_shared<AST::Module>(Module);
      return unsafeRegisterModule(Name, *ModPtr, std::move(ModPtr));
    }
    return unsafeRegisterModule(Name,
                                const_cast<AST::Module &>(Module)); // NOLINT
  }
  Expect<void>
  registerModule(const Runtime::Instance::ModuleInstance &ModInst) {
    return registerModule(ModInst.getModuleName(), ModInst);
  }
  Expect<void>
  registerModule(std::string_view Name,
                 const Runtime::Instance::ModuleInstance &ModInst) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeRegisterModule(Name, ModInst);
  }

  /// Unregister a named module instance.
  Expect<void> unregisterModule(std::string_view Name) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeUnregisterModule(Name);
  }

  /// Rapidly load, validate, instantiate, and run wasm function.
  Expect<std::vector<std::pair<ValVariant, ValType>>>
  runWasmFile(const std::filesystem::path &Path, std::string_view Func,
              Span<const ValVariant> Params = {},
              Span<const ValType> ParamTypes = {}) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeRunWasmFile(Path, Func, Params, ParamTypes);
  }
  Expect<std::vector<std::pair<ValVariant, ValType>>>
  runWasmFile(Span<const Byte> Code, std::string_view Func,
              Span<const ValVariant> Params = {},
              Span<const ValType> ParamTypes = {}) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeRunWasmFile(Code, Func, Params, ParamTypes);
  }
  /// Run a wasm function from a module by mutable reference (zero-copy). In
  /// JIT/lazy-JIT mode the module is mutated in place (compiled symbols are
  /// embedded via loadExecutable). Use the const overload to leave the caller's
  /// module untouched.
  Expect<std::vector<std::pair<ValVariant, ValType>>>
  runWasmFile(AST::Module &Module, std::string_view Func,
              Span<const ValVariant> Params = {},
              Span<const ValType> ParamTypes = {}) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeRunWasmFile(Module, Func, Params, ParamTypes);
  }
  Expect<std::vector<std::pair<ValVariant, ValType>>>
  runWasmFile(const AST::Module &Module, std::string_view Func,
              Span<const ValVariant> Params = {},
              Span<const ValType> ParamTypes = {}) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    if (Strategy->needsModuleCopy()) {
      auto ModPtr = std::make_shared<AST::Module>(Module);
      return unsafeRunWasmFile(*ModPtr, Func, Params, ParamTypes,
                               std::move(ModPtr));
    }
    return unsafeRunWasmFile(const_cast<AST::Module &>(Module), // NOLINT
                             Func, Params, ParamTypes);
  }

  Async<Expect<std::vector<std::pair<ValVariant, ValType>>>>
  asyncRunWasmFile(const std::filesystem::path &Path, std::string_view Func,
                   Span<const ValVariant> Params = {},
                   Span<const ValType> ParamTypes = {});
  Async<Expect<std::vector<std::pair<ValVariant, ValType>>>>
  asyncRunWasmFile(Span<const Byte> Code, std::string_view Func,
                   Span<const ValVariant> Params = {},
                   Span<const ValType> ParamTypes = {});
  Async<Expect<std::vector<std::pair<ValVariant, ValType>>>>
  asyncRunWasmFile(const AST::Module &Module, std::string_view Func,
                   Span<const ValVariant> Params = {},
                   Span<const ValType> ParamTypes = {});

  /// Let runtimeTool check which arguments should be prepared.
  bool holdsModule() {
    if (ActiveModInst) {
      return true;
    }
    return false;
  }
  bool holdsComponent() {
    if (ActiveCompInst) {
      return true;
    }
    return false;
  }

  /// Load given wasm file, wasm bytecode, or wasm module.
  Expect<void> loadWasm(const std::filesystem::path &Path) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeLoadWasm(Path);
  }
  Expect<void> loadWasm(Span<const Byte> Code) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeLoadWasm(Code);
  }
  Expect<void> loadWasm(const AST::Module &Module) {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeLoadWasm(Module);
  }

  /// ======= Functions can be called after the loaded stage. =======
  /// Validate loaded wasm module.
  Expect<void> validate() {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeValidate();
  }

  /// Force-set the VM stage to Validated for a loaded component without
  /// actually running validation. Only used in spec tests and will be removed
  /// when component-model is fully supported.
  /// Returns failure if no component is loaded.
  Expect<void> forceValidateForComponent() {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    if (Stage < VMStage::Loaded || !Comp) {
      return Unexpect(ErrCode::Value::WrongVMWorkflow);
    }
    Comp->setIsValidated();
    Stage = VMStage::Validated;
    return {};
  }

  /// ======= Functions can be called after the validated stage. =======
  /// Instantiate validated wasm module.
  Expect<void> instantiate() {
    EXPECTED_TRY(auto Lock, lockExclusiveForMutation());
    return unsafeInstantiate();
  }

  /// ======= Functions can be called after the instantiated stage. =======
  /// Execute wasm with given input.
  Expect<std::vector<std::pair<ValVariant, ValType>>>
  execute(std::string_view Func, Span<const ValVariant> Params = {},
          Span<const ValType> ParamTypes = {}) {
    Executor::SharedExecutionLock Lock(&Mutex);
    return unsafeExecute(Func, Params, ParamTypes);
  }

  /// Execute a function of a registered module with the given input.
  Expect<std::vector<std::pair<ValVariant, ValType>>>
  execute(std::string_view ModName, std::string_view Func,
          Span<const ValVariant> Params = {},
          Span<const ValType> ParamTypes = {}) {
    Executor::SharedExecutionLock Lock(&Mutex);
    return unsafeExecute(ModName, Func, Params, ParamTypes);
  }

  /// Execute a component function with the given input.
  Expect<std::vector<std::pair<ComponentValVariant, ComponentValType>>>
  executeComponent(std::string_view Func,
                   Span<const ComponentValVariant> Params = {},
                   Span<const ComponentValType> ParamTypes = {}) {
    Executor::SharedExecutionLock Lock(&Mutex);
    return unsafeExecuteComponent(Func, Params, ParamTypes);
  }

  /// Execute a function of a registered component with the given input.
  Expect<std::vector<std::pair<ComponentValVariant, ComponentValType>>>
  executeComponent(std::string_view CompName, std::string_view Func,
                   Span<const ComponentValVariant> Params = {},
                   Span<const ComponentValType> ParamTypes = {}) {
    Executor::SharedExecutionLock Lock(&Mutex);
    return unsafeExecuteComponent(CompName, Func, Params, ParamTypes);
  }

  /// Asynchronously execute Wasm with the given input.
  Async<Expect<std::vector<std::pair<ValVariant, ValType>>>>
  asyncExecute(std::string_view Func, Span<const ValVariant> Params = {},
               Span<const ValType> ParamTypes = {});

  /// Asynchronously execute a function of a registered module with the given
  /// input.
  Async<Expect<std::vector<std::pair<ValVariant, ValType>>>>
  asyncExecute(std::string_view ModName, std::string_view Func,
               Span<const ValVariant> Params = {},
               Span<const ValType> ParamTypes = {});

  /// Asynchronously execute a component function with the given input.
  Async<Expect<std::vector<std::pair<ComponentValVariant, ComponentValType>>>>
  asyncExecuteComponent(std::string_view Func,
                        Span<const ComponentValVariant> Params = {},
                        Span<const ComponentValType> ParamTypes = {});

  /// Asynchronously execute a function of a registered component with the given
  /// input.
  Async<Expect<std::vector<std::pair<ComponentValVariant, ComponentValType>>>>
  asyncExecuteComponent(std::string_view ModName, std::string_view Func,
                        Span<const ComponentValVariant> Params = {},
                        Span<const ComponentValType> ParamTypes = {});

  /// Stop execution
  void stop() noexcept { ExecutorEngine.stop(); }

  /// ======= Functions which are stageless. =======
  /// Clean up VM status
  void cleanup() {
    auto Lock = lockExclusiveForMutation();
    if (!Lock) {
      return;
    }
    return unsafeCleanup();
  }

  /// Get list of callable functions and corresponding function types.
  std::vector<std::pair<std::string, const AST::FunctionType &>>
  getFunctionList() const {
    Executor::SharedExecutionLock Lock(&Mutex);
    return unsafeGetFunctionList();
  }

  /// Get list of callable component functions and corresponding function types.
  std::vector<std::pair<std::string, const AST::Component::FuncType &>>
  getComponentFunctionList() const {
    Executor::SharedExecutionLock Lock(&Mutex);
    return unsafeGetComponentFunctionList();
  }

  /// Get pre-registered module instance by configuration.
  Runtime::Instance::ModuleInstance *
  getImportModule(const HostRegistration Type) const {
    Executor::SharedExecutionLock Lock(&Mutex);
    return unsafeGetImportModule(Type);
  }

  /// Get current instantiated module instance.
  const Runtime::Instance::ModuleInstance *getActiveModule() const {
    Executor::SharedExecutionLock Lock(&Mutex);
    return unsafeGetActiveModule();
  }

  /// Getter for the store set in the VM.
  Runtime::StoreManager &getStoreManager() noexcept { return StoreRef; }
  const Runtime::StoreManager &getStoreManager() const noexcept {
    return StoreRef;
  }

  /// Getter for the loader in the VM.
  Loader::Loader &getLoader() noexcept { return LoaderEngine; }

  /// Getter for the validator in the VM.
  Validator::Validator &getValidator() noexcept { return ValidatorEngine; }

  /// Getter for the executor in the VM.
  ///
  /// Thread safety: the VM has a single execution lock (Mutex). Every executor
  /// call (invoke) and every VM read entry point holds it in shared mode for the
  /// call's duration; every VM state mutation (registerModule, unregisterModule,
  /// instantiate, runWasmFile, cleanup) holds it in unique mode for the whole
  /// operation. A borrowed executor can therefore run concurrently with reads
  /// and other invokes on other threads, and a mutation waits for in-flight
  /// invokes to drain before it frees any module instance — so an in-flight call
  /// never has its module instances, or the lazy-JIT compiled-code callbacks
  /// that read them, freed underneath it. The caller still owns the lifetime of
  /// any FunctionInstance* it passes in: do not invoke with a handle obtained
  /// before a mutation that removed it.
  ///
  /// Lifetime contract: do not destroy the VM while a borrowed executor call is
  /// still running on another thread. ~VM() tears down module instances without
  /// taking the lock — the executor and the lock are themselves being destroyed
  /// — so concurrent use during destruction is undefined.
  ///
  /// Reentrancy: a host function reached through a borrowed-executor invoke()
  /// already holds the execution lock in shared mode on this thread. It may
  /// re-enter this VM's read entry points (execute, getFunctionList, ...), which
  /// reuse the held lock instead of re-locking. It may not re-enter a state
  /// mutation: that would need to upgrade the held shared lock to unique, so the
  /// call fails fast with WrongVMWorkflow rather than deadlocking.
  Executor::Executor &getExecutor() noexcept { return ExecutorEngine; }

  /// Getter for statistics.
  Statistics::Statistics &getStatistics() noexcept { return Stat; }

  uint32_t getLazyCompiledFuncCount() const noexcept {
    Executor::SharedExecutionLock Lock(&Mutex);
    return unsafeGetLazyCompiledFuncCount();
  }

private:
  uint32_t unsafeGetLazyCompiledFuncCount() const noexcept {
    return Strategy->compiledFuncCount();
  }

  /// Notify the strategy that a module ID has no live instances left, so
  /// per-module state (lazy-JIT dylib, AST copy) can be discarded. Skips
  /// empty IDs (untrackable) and IDs that still have a live instance.
  void discardOrphanedModuleState(std::string_view ID) noexcept {
    if (!ID.empty() && !hasLiveInstanceWithID(ID)) {
      Strategy->onModuleOrphaned(ID);
    }
  }

  /// Acquire the execution lock (Mutex) in unique mode for a state-mutating
  /// entry point, rejecting a call reached reentrantly from inside an invoke or
  /// a read on this thread (which already holds the lock in shared mode). A
  /// std::shared_mutex cannot upgrade a shared hold to unique, so such a
  /// reentrant mutation could never take the lock; it fails fast with
  /// WrongVMWorkflow instead. Read entry points take the lock shared through
  /// SharedExecutionLock, which reuses a hold the thread already has, so
  /// reentrant reads from a host function remain allowed.
  Expect<Executor::UniqueExecutionLock> lockExclusiveForMutation();

  /// Terminate a single already-released module instance. Every state mutation
  /// holds the execution lock in unique mode for its whole duration, so no
  /// borrowed executor can be running a function in the instance while it is
  /// freed; this helper takes no lock of its own. \p ModInst may be null.
  void terminateModuleInstance(Runtime::Instance::ModuleInstance *ModInst) {
    if (ModInst) {
      ModInst->terminate();
    }
  }

  /// Terminate and clear every module instance in \p Container. Like
  /// terminateModuleInstance, this runs under the caller's unique execution lock
  /// (or single-threaded construction/destruction), so it takes no lock itself.
  void cleanupModInstContainer(
      std::vector<std::unique_ptr<Runtime::Instance::ModuleInstance>>
          &Container) {
    for (auto &Item : Container) {
      if (auto *ModInst = Item.release()) {
        ModInst->terminate();
      }
    }
    Container.clear();
  }

  void cleanupModInstContainer(
      std::unordered_map<std::string,
                         std::unique_ptr<Runtime::Instance::ModuleInstance>>
          &Container) {
    for (auto &Item : Container) {
      if (auto *ModInst = Item.second.release()) {
        ModInst->terminate();
      }
    }
    Container.clear();
  }

  void cleanupModInstContainer(
      std::unordered_map<HostRegistration,
                         std::unique_ptr<Runtime::Instance::ModuleInstance>>
          &Container) {
    for (auto &Item : Container) {
      if (auto *ModInst = Item.second.release()) {
        ModInst->terminate();
      }
    }
    Container.clear();
  }

  Expect<void> unsafeRegisterModule(std::string_view Name,
                                    const std::filesystem::path &Path);
  Expect<void> unsafeRegisterModule(std::string_view Name,
                                    Span<const Byte> Code);
  Expect<void>
  unsafeRegisterModule(std::string_view Name, AST::Module &Module,
                       std::shared_ptr<AST::Module> PreAllocated = nullptr);
  Expect<void>
  unsafeRegisterModule(std::string_view Name,
                       const Runtime::Instance::ModuleInstance &ModInst);

  Expect<void> unsafeUnregisterModule(std::string_view Name);

  Expect<std::vector<std::pair<ValVariant, ValType>>>
  unsafeRunWasmFile(const std::filesystem::path &Path, std::string_view Func,
                    Span<const ValVariant> Params = {},
                    Span<const ValType> ParamTypes = {});
  Expect<std::vector<std::pair<ValVariant, ValType>>>
  unsafeRunWasmFile(Span<const Byte> Code, std::string_view Func,
                    Span<const ValVariant> Params = {},
                    Span<const ValType> ParamTypes = {});
  Expect<std::vector<std::pair<ValVariant, ValType>>>
  unsafeRunWasmFile(AST::Module &Module, std::string_view Func,
                    Span<const ValVariant> Params = {},
                    Span<const ValType> ParamTypes = {},
                    std::shared_ptr<AST::Module> PreAllocated = nullptr);
  Expect<std::vector<std::pair<ValVariant, ValType>>>
  unsafeRunWasmFile(const AST::Component::Component &Component,
                    std::string_view Func, Span<const ValVariant> Params = {},
                    Span<const ValType> ParamTypes = {});

  Expect<void> unsafeLoadWasm(const std::filesystem::path &Path);
  Expect<void> unsafeLoadWasm(Span<const Byte> Code);
  Expect<void> unsafeLoadWasm(const AST::Module &Module);

  Expect<void> unsafeValidate();

  Expect<void> unsafeInstantiate();

  Expect<std::vector<std::pair<ValVariant, ValType>>>
  unsafeExecute(std::string_view Func, Span<const ValVariant> Params = {},
                Span<const ValType> ParamTypes = {});
  Expect<std::vector<std::pair<ValVariant, ValType>>>

  unsafeExecute(std::string_view Mod, std::string_view Func,
                Span<const ValVariant> Params = {},
                Span<const ValType> ParamTypes = {});

  Expect<std::vector<std::pair<ComponentValVariant, ComponentValType>>>
  unsafeExecuteComponent(std::string_view Func,
                         Span<const ComponentValVariant> Params = {},
                         Span<const ComponentValType> ParamTypes = {});

  Expect<std::vector<std::pair<ComponentValVariant, ComponentValType>>>
  unsafeExecuteComponent(std::string_view Comp, std::string_view Func,
                         Span<const ComponentValVariant> Params = {},
                         Span<const ComponentValType> ParamTypes = {});

  void unsafeCleanup();

  std::vector<std::pair<std::string, const AST::FunctionType &>>
  unsafeGetFunctionList() const;

  std::vector<std::pair<std::string, const AST::Component::FuncType &>>
  unsafeGetComponentFunctionList() const;

  Runtime::Instance::ModuleInstance *
  unsafeGetImportModule(const HostRegistration Type) const;

  const Runtime::Instance::ModuleInstance *unsafeGetActiveModule() const;

  enum class VMStage : uint8_t { Inited, Loaded, Validated, Instantiated };

  void unsafeInitVM();
  void unsafeLoadBuiltInHosts();
  void unsafeLoadPlugInHosts();
  void unsafeRegisterBuiltInHosts();
  void unsafeRegisterPlugInHosts();

  /// Helper function for execution.
  Expect<std::vector<std::pair<ValVariant, ValType>>>
  unsafeExecute(const Runtime::Instance::ModuleInstance *ModInst,
                std::string_view Func, Span<const ValVariant> Params = {},
                Span<const ValType> ParamTypes = {});

  Expect<std::vector<std::pair<ComponentValVariant, ComponentValType>>>
  unsafeExecuteComponent(const Runtime::Instance::ComponentInstance *CompInst,
                         std::string_view Func,
                         Span<const ComponentValVariant> Params = {},
                         Span<const ComponentValType> ParamTypes = {});

  /// \name VM environment.
  /// @{
  const Configure Conf;
  Statistics::Statistics Stat;
  VMStage Stage;
  /// The VM's single execution lock. Read entry points and the executor's
  /// invoke() hold it shared; state mutations hold it unique for their whole
  /// duration. Declared before ExecutorEngine so it outlives the executor's
  /// pointer to it (installed via setExecutionMutex).
  mutable std::shared_mutex Mutex;
  /// @}

  /// \name VM components.
  /// @{
  Loader::Loader LoaderEngine;
  Validator::Validator ValidatorEngine;
  Executor::Executor ExecutorEngine;
  std::unique_ptr<ExecutionStrategy> Strategy;
  /// @}

  /// \name VM Storage.
  /// @{
  /// Loaded AST module.
  std::unique_ptr<AST::Module> Mod;
  std::unique_ptr<AST::Component::Component> Comp;
  /// Active module instance.
  std::unique_ptr<Runtime::Instance::ModuleInstance> ActiveModInst;
  std::unique_ptr<Runtime::Instance::ComponentInstance> ActiveCompInst;
  /// Registered module instances by user.
  std::vector<std::unique_ptr<Runtime::Instance::ModuleInstance>> RegModInsts;
  /// Built-in module instances mapped to the configurations. For WASI.
  std::unordered_map<HostRegistration,
                     std::unique_ptr<Runtime::Instance::ModuleInstance>>
      BuiltInModInsts;
  /// Loaded module instances from plug-ins.
  std::vector<std::unique_ptr<Runtime::Instance::ModuleInstance>>
      PlugInModInsts;
  std::vector<std::unique_ptr<Runtime::Instance::ComponentInstance>>
      PlugInCompInsts;
  /// Self-owned store (nullptr if an outside store is assigned in constructor).
  std::unique_ptr<Runtime::StoreManager> Store;
  /// Reference to the store.
  Runtime::StoreManager &StoreRef;
  /// @}

  /// Ensure a function is lazily compiled before execution
  /// (CompilationTrigger). A no-op unless lazy JIT is active.
  Expect<void> ensureCompiled(
      const Runtime::Instance::FunctionInstance &Func) noexcept override;

  /// Whether any live instance (active or registered) still has module ID
  /// \p ID, so per-ID lazy-JIT state must be kept rather than discarded when a
  /// registration fails or a module is unregistered.
  bool hasLiveInstanceWithID(std::string_view ID) const noexcept;
};

} // namespace VM
} // namespace WasmEdge
