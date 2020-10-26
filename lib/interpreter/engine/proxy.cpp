// SPDX-License-Identifier: Apache-2.0
#include "interpreter/interpreter.h"

namespace WasmEdge {
namespace Interpreter {

namespace {
static uint32_t AliveInstanceCount = 0;
static std::mutex SignalMutex;
} // namespace

thread_local Interpreter *Interpreter::This = nullptr;

struct Interpreter::HostContext {
  HostContext(Interpreter *T) noexcept : _(T) { _->leaveCompiledContext(); }
  ~HostContext() noexcept { _->enterCompiledContext(); }
  Interpreter *_;
};

void Interpreter::InstanceIncrease() {
  std::unique_lock Lock(SignalMutex);
  ++AliveInstanceCount;
}
void Interpreter::InstanceDecrease() {
  std::unique_lock Lock(SignalMutex);
  assert(AliveInstanceCount > 0);
  --AliveInstanceCount;
  if (AliveInstanceCount == 0) {
    std::signal(SIGFPE, SIG_DFL);
    std::signal(SIGSEGV, SIG_DFL);
  }
}

template <typename RetT, typename... ArgsT>
struct Interpreter::ProxyHelper<Expect<RetT> (Interpreter::*)(
    Runtime::StoreManager &, ArgsT...) noexcept> {
  template <Expect<RetT> (Interpreter::*Func)(Runtime::StoreManager &,
                                              ArgsT...) noexcept>
  static RetT proxy(ArgsT... Args) noexcept {
    auto *_ = This;
    HostContext Host(_);
    if (auto Res = (_->*Func)(*_->CurrentStore, Args...); unlikely(!Res)) {
      siglongjmp(*_->TrapJump, uint8_t(Res.error()));
    } else if constexpr (!std::is_void_v<RetT>) {
      return *Res;
    }
  }
};

#if defined(__clang_major__) && __clang_major__ >= 10
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-designator"
#endif

/// Intrinsics table
extern "C" {
__attribute__((visibility("default")))
AST::Module::IntrinsicsTable intrinsics = {
#define ENTRY(NAME, FUNC)                                                      \
  [uint8_t(AST::Module::Intrinsics::NAME)] = reinterpret_cast<void *>(         \
      &Interpreter::ProxyHelper<decltype(                                      \
          &Interpreter::FUNC)>::proxy<&Interpreter::FUNC>)
    ENTRY(kTrap, trap),
    ENTRY(kCall, call),
    ENTRY(kCallIndirect, callIndirect),
    ENTRY(kMemCopy, memCopy),
    ENTRY(kMemFill, memFill),
    ENTRY(kMemGrow, memGrow),
    ENTRY(kMemSize, memSize),
    ENTRY(kMemInit, memInit),
    ENTRY(kDataDrop, dataDrop),
    ENTRY(kTableGet, tableGet),
    ENTRY(kTableSet, tableSet),
    ENTRY(kTableCopy, tableCopy),
    ENTRY(kTableFill, tableFill),
    ENTRY(kTableGrow, tableGrow),
    ENTRY(kTableSize, tableSize),
    ENTRY(kTableInit, tableInit),
    ENTRY(kElemDrop, elemDrop),
    ENTRY(kRefFunc, refFunc),
#undef ENTRY
};
}

#if defined(__clang_major__) && __clang_major__ >= 10
#pragma clang diagnostic pop
#endif

/// SIGBUS, SIGFPE, SIGILL, and SIGSEGV will send to the same thread that
/// generated it.

void Interpreter::signalHandler(int Signal, siginfo_t *Siginfo,
                                void *) noexcept {
  if (This == nullptr) {
    return;
  }
  auto *_ = This;
  HostContext Host(_);
  int Status;
  switch (Signal) {
  case SIGSEGV:
    Status = uint8_t(ErrCode::MemoryOutOfBounds);
    break;
  case SIGFPE:
    assert(Siginfo->si_code == FPE_INTDIV);
    Status = uint8_t(ErrCode::DivideByZero);
    break;
  default:
    __builtin_unreachable();
  }
  siglongjmp(*_->TrapJump, Status);
}

void Interpreter::enterCompiledContext() noexcept {
  assert(This == nullptr);
  This = this;
  struct sigaction Action {};
  Action.sa_sigaction = &signalHandler;
  Action.sa_flags = SA_SIGINFO | SA_NODEFER;
  sigemptyset(&Action.sa_mask);
  sigaction(SIGFPE, &Action, nullptr);
  sigaction(SIGSEGV, &Action, nullptr);
}

void Interpreter::leaveCompiledContext() noexcept {
  assert(This != nullptr);
  This = nullptr;
}

Expect<void> Interpreter::trap(Runtime::StoreManager &StoreMgr,
                               const uint8_t Code) noexcept {
  return Unexpect(ErrCode(Code));
}

Expect<void> Interpreter::call(Runtime::StoreManager &StoreMgr,
                               const uint32_t FuncIndex, const ValVariant *Args,
                               ValVariant *Rets) noexcept {
  const auto *ModInst = *StoreMgr.getModule(StackMgr.getModuleAddr());
  const uint32_t FuncAddr = *ModInst->getFuncAddr(FuncIndex);
  const auto *FuncInst = *StoreMgr.getFunction(FuncAddr);
  const auto &FuncType = FuncInst->getFuncType();
  const unsigned ParamsSize = FuncType.Params.size();
  const unsigned ReturnsSize = FuncType.Returns.size();

  for (unsigned I = 0; I < ParamsSize; ++I) {
    StackMgr.push(Args[I]);
  }

  auto Instrs = FuncInst->getInstrs();
  AST::InstrView::iterator StartIt;
  if (auto Res = enterFunction(StoreMgr, *FuncInst, Instrs.end() - 1)) {
    StartIt = *Res;
  } else {
    return Unexpect(Res);
  }
  if (auto Res = execute(StoreMgr, StartIt, Instrs.end()); unlikely(!Res)) {
    return Unexpect(Res);
  }

  for (unsigned I = 0; I < ReturnsSize; ++I) {
    Rets[ReturnsSize - 1 - I] = StackMgr.pop();
  }

  return {};
}

Expect<void> Interpreter::callIndirect(Runtime::StoreManager &StoreMgr,
                                       const uint32_t TableIndex,
                                       const uint32_t FuncTypeIndex,
                                       const uint32_t FuncIndex,
                                       const ValVariant *Args,
                                       ValVariant *Rets) noexcept {
  const auto *TabInst = getTabInstByIdx(StoreMgr, TableIndex);

  if (unlikely(FuncIndex >= TabInst->getSize())) {
    return Unexpect(ErrCode::UndefinedElement);
  }

  ValVariant Ref = *TabInst->getRefAddr(FuncIndex);
  if (unlikely(isNullRef(Ref))) {
    return Unexpect(ErrCode::UninitializedElement);
  }
  const auto FuncAddr = retrieveFuncIdx(Ref);

  const auto *ModInst = *StoreMgr.getModule(StackMgr.getModuleAddr());
  const auto *TargetFuncType = *ModInst->getFuncType(FuncTypeIndex);
  const auto *FuncInst = *StoreMgr.getFunction(FuncAddr);
  const auto &FuncType = FuncInst->getFuncType();
  if (unlikely(*TargetFuncType != FuncType)) {
    return Unexpect(ErrCode::IndirectCallTypeMismatch);
  }

  const unsigned ParamsSize = FuncType.Params.size();
  const unsigned ReturnsSize = FuncType.Returns.size();

  for (unsigned I = 0; I < ParamsSize; ++I) {
    StackMgr.push(Args[I]);
  }

  auto Instrs = FuncInst->getInstrs();
  AST::InstrView::iterator StartIt;
  if (auto Res = enterFunction(StoreMgr, *FuncInst, Instrs.end() - 1)) {
    StartIt = *Res;
  } else {
    return Unexpect(Res);
  }
  if (auto Res = execute(StoreMgr, StartIt, Instrs.end()); unlikely(!Res)) {
    return Unexpect(Res);
  }

  for (unsigned I = 0; I < ReturnsSize; ++I) {
    Rets[ReturnsSize - 1 - I] = StackMgr.pop();
  }

  return {};
}

Expect<uint32_t> Interpreter::memGrow(Runtime::StoreManager &StoreMgr,
                                      const uint32_t NewSize) noexcept {
  auto &MemInst = *getMemInstByIdx(StoreMgr, 0);
  const uint32_t CurrPageSize = MemInst.getDataPageSize();
  if (MemInst.growPage(NewSize)) {
    return CurrPageSize;
  } else {
    return -1;
  }
}

Expect<uint32_t>
Interpreter::memSize(Runtime::StoreManager &StoreMgr) noexcept {
  auto &MemInst = *getMemInstByIdx(StoreMgr, 0);
  return MemInst.getDataPageSize();
}

Expect<void> Interpreter::memCopy(Runtime::StoreManager &StoreMgr,
                                  const uint32_t Dst, const uint32_t Src,
                                  const uint32_t Len) noexcept {
  auto &MemInst = *getMemInstByIdx(StoreMgr, 0);

  if (auto Data = MemInst.getBytes(Src, Len); unlikely(!Data)) {
    return Unexpect(Data);
  } else {
    if (auto Res = MemInst.setBytes(*Data, Dst, 0, Len); unlikely(!Res)) {
      return Unexpect(Res);
    }
  }

  return {};
}

Expect<void> Interpreter::memFill(Runtime::StoreManager &StoreMgr,
                                  const uint32_t Off, const uint8_t Val,
                                  const uint32_t Len) noexcept {
  auto &MemInst = *getMemInstByIdx(StoreMgr, 0);
  if (auto Res = MemInst.fillBytes(Val, Off, Len); unlikely(!Res)) {
    return Unexpect(Res);
  }

  return {};
}

Expect<void> Interpreter::memInit(Runtime::StoreManager &StoreMgr,
                                  const uint32_t DataIdx, const uint32_t Dst,
                                  const uint32_t Src,
                                  const uint32_t Len) noexcept {
  auto &MemInst = *getMemInstByIdx(StoreMgr, 0);
  auto &DataInst = *getDataInstByIdx(StoreMgr, DataIdx);

  if (auto Res = MemInst.setBytes(DataInst.getData(), Dst, Src, Len);
      unlikely(!Res)) {
    return Unexpect(Res);
  }

  return {};
}

Expect<void> Interpreter::dataDrop(Runtime::StoreManager &StoreMgr,
                                   const uint32_t DataIdx) noexcept {
  auto &DataInst = *getDataInstByIdx(StoreMgr, DataIdx);
  DataInst.clear();

  return {};
}

Expect<RefVariant> Interpreter::tableGet(Runtime::StoreManager &StoreMgr,
                                         const uint32_t TableIndex,
                                         const uint32_t Idx) noexcept {
  auto &TabInst = *getTabInstByIdx(StoreMgr, TableIndex);
  if (auto Res = TabInst.getRefAddr(Idx); unlikely(!Res)) {
    return Unexpect(Res);
  } else {
    return *Res;
  }

  return {};
}

Expect<void> Interpreter::tableSet(Runtime::StoreManager &StoreMgr,
                                   const uint32_t TableIndex,
                                   const uint32_t Idx,
                                   const RefVariant Ref) noexcept {
  auto &TabInst = *getTabInstByIdx(StoreMgr, TableIndex);
  if (auto Res = TabInst.setRefAddr(Idx, Ref); unlikely(!Res)) {
    return Unexpect(Res);
  }

  return {};
}

Expect<void> Interpreter::tableCopy(Runtime::StoreManager &StoreMgr,
                                    const uint32_t TableIndexSrc,
                                    const uint32_t TableIndexDst,
                                    const uint32_t Dst, const uint32_t Src,
                                    const uint32_t Len) noexcept {
  auto &TabInstSrc = *getTabInstByIdx(StoreMgr, TableIndexSrc);
  auto &TabInstDst = *getTabInstByIdx(StoreMgr, TableIndexDst);
  if (auto Refs = TabInstSrc.getRefs(Src, Len); unlikely(!Refs)) {
    return Unexpect(Refs);
  } else {
    if (auto Res = TabInstDst.setRefs(*Refs, Dst, 0, Len); unlikely(!Res)) {
      return Unexpect(Res);
    }
  }

  return {};
}

Expect<uint32_t> Interpreter::tableGrow(Runtime::StoreManager &StoreMgr,
                                        const uint32_t TableIndex,
                                        const RefVariant Val,
                                        const uint32_t NewSize) noexcept {
  auto &TabInst = *getTabInstByIdx(StoreMgr, TableIndex);
  const uint32_t CurrTableSize = TabInst.getSize();
  if (likely(TabInst.growTable(NewSize, Val))) {
    return CurrTableSize;
  } else {
    return -1;
  }
}

Expect<uint32_t> Interpreter::tableSize(Runtime::StoreManager &StoreMgr,
                                        const uint32_t TableIndex) noexcept {
  auto &TabInst = *getTabInstByIdx(StoreMgr, TableIndex);
  return TabInst.getSize();
}

Expect<void> Interpreter::tableFill(Runtime::StoreManager &StoreMgr,
                                    const uint32_t TableIndex,
                                    const uint32_t Off, const RefVariant Ref,
                                    const uint32_t Len) noexcept {
  auto &TabInst = *getTabInstByIdx(StoreMgr, TableIndex);
  if (auto Res = TabInst.fillRefs(Ref, Off, Len); unlikely(!Res)) {
    return Unexpect(Res);
  }

  return {};
}

Expect<void> Interpreter::tableInit(Runtime::StoreManager &StoreMgr,
                                    const uint32_t TableIndex,
                                    const uint32_t ElementIndex,
                                    const uint32_t Dst, const uint32_t Src,
                                    const uint32_t Len) noexcept {
  auto &TabInst = *getTabInstByIdx(StoreMgr, TableIndex);
  auto &ElemInst = *getElemInstByIdx(StoreMgr, ElementIndex);
  if (auto Res = TabInst.setRefs(ElemInst.getRefs(), Dst, Src, Len);
      unlikely(!Res)) {
    return Unexpect(Res);
  }

  return {};
}

Expect<void> Interpreter::elemDrop(Runtime::StoreManager &StoreMgr,
                                   const uint32_t ElementIndex) noexcept {
  auto &ElemInst = *getElemInstByIdx(StoreMgr, ElementIndex);
  ElemInst.clear();

  return {};
}

Expect<RefVariant> Interpreter::refFunc(Runtime::StoreManager &StoreMgr,
                                        const uint32_t FuncIndex) noexcept {
  const auto *ModInst = *StoreMgr.getModule(StackMgr.getModuleAddr());
  const uint32_t FuncAddr = *ModInst->getFuncAddr(FuncIndex);
  return genFuncRef(FuncAddr);
}

} // namespace Interpreter
} // namespace WasmEdge
