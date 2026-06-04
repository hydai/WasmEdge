// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

//===-- wasmedge/llvm/data.h - Data class definition ----------------------===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the Data class.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "ast/module.h"
#include "common/configure.h"
#include "common/errcode.h"
#include "common/filesystem.h"
#include "common/span.h"

#include <cassert>
#include <cstdlib>
#include <mutex>

namespace WasmEdge::LLVM {
class OrcThreadSafeContext;

/// Holds llvm-relative runtime data, like llvm::Context, llvm::Module, etc.
class Data {
public:
  struct DataContext;
  Data() noexcept;
  ~Data() noexcept;
  Data(Data &&) noexcept;
  Data &operator=(Data &&) noexcept;
  DataContext &extract() noexcept {
    // extract() returns a reference and so cannot fall back to a safe value
    // like the other accessors; callers must hold a valid (non-moved-from)
    // Data. A moved-from Data here is a caller bug: fail with a defined abort
    // (assert message in debug) instead of the release-build UB that
    // assuming() would produce when dereferencing a null Context.
    if (!isValid()) {
      assert(false && "LLVM::Data::extract() called on a moved-from Data");
      std::abort();
    }
    return *Context;
  }
  bool isValid() const noexcept { return static_cast<bool>(Context); }
  bool hasModule() const noexcept;
  void resetModule() noexcept;
  void setPrefix(std::string_view P) noexcept;
  std::string_view getPrefix() const noexcept;

private:
  std::unique_ptr<DataContext> Context;
  const Configure Conf;
};

} // namespace WasmEdge::LLVM
