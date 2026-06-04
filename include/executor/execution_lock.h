// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

//===-- wasmedge/executor/execution_lock.h - Execution lock guard ---------===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the shared-mode guard for the VM's single execution lock.
/// Both Executor::invoke() and the owning VM's read entry points take the lock
/// through this guard, so they share one per-thread recursion view and never
/// re-lock the non-recursive std::shared_mutex on a single thread.
///
//===----------------------------------------------------------------------===//
#pragma once

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace WasmEdge {
namespace Executor {

namespace detail {
/// Execution mutexes currently held by the calling thread, in shared OR unique
/// mode, innermost last. Tracked per mutex (not as a depth) so a nested acquire
/// on a different mutex still locks it, while a nested shared acquire on a mutex
/// this thread already holds skips re-acquisition: std::shared_mutex is neither
/// recursive nor shared-upgradable, so without this a read re-entered from an
/// invoke — or a mutation that holds the lock unique and then runs wasm through
/// invoke() — would self-deadlock.
inline thread_local std::vector<std::shared_mutex *> HeldExecutionMutexes;

inline bool isExecutionMutexHeld(std::shared_mutex *ExecMutex) noexcept {
  return std::find(HeldExecutionMutexes.begin(), HeldExecutionMutexes.end(),
                   ExecMutex) != HeldExecutionMutexes.end();
}
} // namespace detail

/// RAII guard that holds \p ExecMutex in shared mode for its scope, unless the
/// calling thread already holds it (shared or unique) — then it is a no-op, so
/// the non-recursive std::shared_mutex is never re-locked on one thread. That
/// would otherwise self-deadlock against a pending writer, or deadlock a
/// mutation that holds the lock unique and then runs wasm through invoke(). A
/// null \p ExecMutex (a standalone executor with no owning VM) is also a no-op.
/// Both Executor::invoke() and the VM's read entry points acquire through this
/// guard, so a host function that re-enters the VM from inside an invoke reuses
/// the outer hold instead of re-locking.
class SharedExecutionLock {
public:
  explicit SharedExecutionLock(std::shared_mutex *ExecMutex) noexcept
      : Mutex(ExecMutex) {
    if (Mutex != nullptr) {
      if (!detail::isExecutionMutexHeld(Mutex)) {
        Lock = std::shared_lock<std::shared_mutex>(*Mutex);
      }
      detail::HeldExecutionMutexes.push_back(Mutex);
    }
  }
  ~SharedExecutionLock() noexcept {
    if (Mutex != nullptr) {
      detail::HeldExecutionMutexes.pop_back();
    }
  }
  SharedExecutionLock(const SharedExecutionLock &) = delete;
  SharedExecutionLock &operator=(const SharedExecutionLock &) = delete;

private:
  std::shared_mutex *Mutex;
  std::shared_lock<std::shared_mutex> Lock;
};

/// RAII guard that holds the VM's execution lock in unique mode for a state
/// mutation, and records the hold so a nested SharedExecutionLock on this thread
/// (the wasm a mutation runs through invoke(), e.g. an instantiate start
/// function) reuses it instead of trying — and failing — to acquire it shared.
/// Move-only, so it can be returned through Expect like a std::unique_lock; a
/// moved-from instance owns nothing and untracks nothing.
class UniqueExecutionLock {
public:
  UniqueExecutionLock() noexcept = default;
  explicit UniqueExecutionLock(std::shared_mutex &ExecMutex)
      : Lock(ExecMutex), Mutex(&ExecMutex) {
    detail::HeldExecutionMutexes.push_back(Mutex);
  }
  UniqueExecutionLock(UniqueExecutionLock &&Other) noexcept
      : Lock(std::move(Other.Lock)), Mutex(Other.Mutex) {
    Other.Mutex = nullptr;
  }
  UniqueExecutionLock &operator=(UniqueExecutionLock &&Other) noexcept {
    if (this != &Other) {
      untrack();
      Lock = std::move(Other.Lock);
      Mutex = Other.Mutex;
      Other.Mutex = nullptr;
    }
    return *this;
  }
  ~UniqueExecutionLock() { untrack(); }
  UniqueExecutionLock(const UniqueExecutionLock &) = delete;
  UniqueExecutionLock &operator=(const UniqueExecutionLock &) = delete;

private:
  void untrack() noexcept {
    if (Mutex != nullptr) {
      detail::HeldExecutionMutexes.pop_back();
      Mutex = nullptr;
    }
  }
  std::unique_lock<std::shared_mutex> Lock;
  std::shared_mutex *Mutex = nullptr;
};

} // namespace Executor
} // namespace WasmEdge
