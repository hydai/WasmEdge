// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2026 Second State INC

//===-- test/llvm/LazyJITTest.cpp - Lazy JIT compilation tests ------------===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains tests for the lazy JIT (per-function) compilation
/// feature. It verifies that:
/// 1. Lazy JIT mode can be enabled via configuration
/// 2. Functions are compiled on-demand rather than upfront
/// 3. The behavior is correct compared to eager compilation
///
//===----------------------------------------------------------------------===//

#include "common/configure.h"
#include "common/spdlog.h"
#include "common/types.h"
#include "runtime/callingframe.h"
#include "runtime/instance/module.h"
#include "vm/vm.h"
#include "llvm/jit.h"
#include <gtest/gtest.h>
#include <thread>

namespace {

using namespace std::literals;
using namespace WasmEdge;

// Module with 6 functions: add, mul, sub, const42, unused1, unused2
std::vector<uint8_t> SimpleWasm = {
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x17, 0x04, 0x60,
    0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x60, 0x00, 0x01, 0x7f, 0x60, 0x01, 0x7f,
    0x01, 0x7f, 0x60, 0x03, 0x7f, 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x07, 0x06,
    0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x07, 0x31, 0x06, 0x03, 0x61, 0x64,
    0x64, 0x00, 0x00, 0x03, 0x6d, 0x75, 0x6c, 0x00, 0x01, 0x03, 0x73, 0x75,
    0x62, 0x00, 0x02, 0x07, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x34, 0x32, 0x00,
    0x03, 0x07, 0x75, 0x6e, 0x75, 0x73, 0x65, 0x64, 0x31, 0x00, 0x04, 0x07,
    0x75, 0x6e, 0x75, 0x73, 0x65, 0x64, 0x32, 0x00, 0x05, 0x0a, 0x32, 0x06,
    0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b, 0x07, 0x00, 0x20, 0x00,
    0x20, 0x01, 0x6c, 0x0b, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6b, 0x0b,
    0x04, 0x00, 0x41, 0x2a, 0x0b, 0x08, 0x00, 0x20, 0x00, 0x41, 0xe4, 0x00,
    0x6a, 0x0b, 0x0a, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x20, 0x02, 0x6c,
    0x0b};

std::vector<uint8_t> FibonacciWasm = {
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01,
    0x60, 0x01, 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07,
    0x01, 0x03, 0x66, 0x69, 0x62, 0x00, 0x00, 0x0a, 0x1e, 0x01, 0x1c,
    0x00, 0x20, 0x00, 0x41, 0x02, 0x48, 0x04, 0x7f, 0x41, 0x01, 0x05,
    0x20, 0x00, 0x41, 0x02, 0x6b, 0x10, 0x00, 0x20, 0x00, 0x41, 0x01,
    0x6b, 0x10, 0x00, 0x6a, 0x0b, 0x0b};

// Module with 1 imported function and 3 local functions:
//   import "env" "host_add" (func (param i32 i32) (result i32))
//   export "call_host"  -> calls host_add
//   export "double_add" -> calls host_add twice, adds results
//   export "local_mul"  -> pure local i32.mul
std::vector<uint8_t> ImportWasm = {
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60,
    0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x02, 0x10, 0x01, 0x03, 0x65, 0x6e, 0x76,
    0x08, 0x68, 0x6f, 0x73, 0x74, 0x5f, 0x61, 0x64, 0x64, 0x00, 0x00, 0x03,
    0x04, 0x03, 0x00, 0x00, 0x00, 0x07, 0x26, 0x03, 0x09, 0x63, 0x61, 0x6c,
    0x6c, 0x5f, 0x68, 0x6f, 0x73, 0x74, 0x00, 0x01, 0x0a, 0x64, 0x6f, 0x75,
    0x62, 0x6c, 0x65, 0x5f, 0x61, 0x64, 0x64, 0x00, 0x02, 0x09, 0x6c, 0x6f,
    0x63, 0x61, 0x6c, 0x5f, 0x6d, 0x75, 0x6c, 0x00, 0x03, 0x0a, 0x22, 0x03,
    0x08, 0x00, 0x20, 0x00, 0x20, 0x01, 0x10, 0x00, 0x0b, 0x0f, 0x00, 0x20,
    0x00, 0x20, 0x01, 0x10, 0x00, 0x20, 0x00, 0x20, 0x01, 0x10, 0x00, 0x6a,
    0x0b, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6c, 0x0b};

// Library module exporting "add" and "mul" (i32,i32)->i32
std::vector<uint8_t> MathLibWasm = {
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60,
    0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x03, 0x02, 0x00, 0x00, 0x07, 0x0d,
    0x02, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00, 0x03, 0x6d, 0x75, 0x6c, 0x00,
    0x01, 0x0a, 0x11, 0x02, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
    0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6c, 0x0b};

// Consumer module importing "math"."add" and "math"."mul",
// exporting "add_and_square" = (a+b)*(a+b) and "sum_of_squares" = a*a+b*b
std::vector<uint8_t> MathConsumerWasm = {
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01,
    0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x02, 0x17, 0x02, 0x04, 0x6d,
    0x61, 0x74, 0x68, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00, 0x04, 0x6d,
    0x61, 0x74, 0x68, 0x03, 0x6d, 0x75, 0x6c, 0x00, 0x00, 0x03, 0x03,
    0x02, 0x00, 0x00, 0x07, 0x23, 0x02, 0x0e, 0x61, 0x64, 0x64, 0x5f,
    0x61, 0x6e, 0x64, 0x5f, 0x73, 0x71, 0x75, 0x61, 0x72, 0x65, 0x00,
    0x02, 0x0e, 0x73, 0x75, 0x6d, 0x5f, 0x6f, 0x66, 0x5f, 0x73, 0x71,
    0x75, 0x61, 0x72, 0x65, 0x73, 0x00, 0x03, 0x0a, 0x23, 0x02, 0x10,
    0x00, 0x20, 0x00, 0x20, 0x01, 0x10, 0x00, 0x20, 0x00, 0x20, 0x01,
    0x10, 0x00, 0x10, 0x01, 0x0b, 0x10, 0x00, 0x20, 0x00, 0x20, 0x00,
    0x10, 0x01, 0x20, 0x01, 0x20, 0x01, 0x10, 0x01, 0x10, 0x00, 0x0b};

// Module with two local functions, where "f" (func 1) calls "g" (func 0):
//   (func $g (param i32) (result i32) local.get 0  i32.const 1  i32.add)
//   (func $f (param i32) (result i32) local.get 0  call $g)
//   (export "g" (func $g)) (export "f" (func $f))
std::vector<uint8_t> CallerCalleeWasm = {
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60,
    0x01, 0x7f, 0x01, 0x7f, 0x03, 0x03, 0x02, 0x00, 0x00, 0x07, 0x09, 0x02,
    0x01, 0x67, 0x00, 0x00, 0x01, 0x66, 0x00, 0x01, 0x0a, 0x10, 0x02, 0x07,
    0x00, 0x20, 0x00, 0x41, 0x01, 0x6a, 0x0b, 0x06, 0x00, 0x20, 0x00, 0x10,
    0x00, 0x0b};

// Like CallerCalleeWasm but "f" (func 1) calls "g" (func 0) only on the taken
// branch, so f reaches g statically while a zero argument skips the call at
// run time:
//   (func $g (param i32) (result i32) local.get 0  i32.const 1  i32.add)
//   (func $f (param i32) (result i32)
//     local.get 0  (if (result i32) (then local.get 0  call $g)
//                                    (else local.get 0)))
//   (export "g" (func $g)) (export "f" (func $f))
std::vector<uint8_t> CondCallWasm = {
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60,
    0x01, 0x7f, 0x01, 0x7f, 0x03, 0x03, 0x02, 0x00, 0x00, 0x07, 0x09, 0x02,
    0x01, 0x67, 0x00, 0x00, 0x01, 0x66, 0x00, 0x01, 0x0a, 0x18, 0x02, 0x07,
    0x00, 0x20, 0x00, 0x41, 0x01, 0x6a, 0x0b, 0x0e, 0x00, 0x20, 0x00, 0x04,
    0x7f, 0x20, 0x00, 0x10, 0x00, 0x05, 0x20, 0x00, 0x0b, 0x0b};

class HostAdd : public Runtime::HostFunction<HostAdd> {
public:
  Expect<uint32_t> body(const Runtime::CallingFrame &, uint32_t A, uint32_t B) {
    return A + B;
  }
};

class TestEnvModule : public Runtime::Instance::ModuleInstance {
public:
  TestEnvModule() : ModuleInstance("env") {
    addHostFunc("host_add", std::make_unique<HostAdd>());
  }
};

class LazyJITTest : public ::testing::Test {
protected:
  // Helper to create a VM with eager JIT
  std::unique_ptr<VM::VM> createEagerJITVM() {
    Configure Conf;
    Conf.getRuntimeConfigure().setRunMode(RunMode::JIT);
    Conf.getCompilerConfigure().setOptimizationLevel(
        CompilerConfigure::OptimizationLevel::O1);
    return std::make_unique<VM::VM>(Conf);
  }

  // Helper to create a VM with lazy JIT
  std::unique_ptr<VM::VM> createLazyJITVM() {
    Configure Conf;
    Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
    Conf.getCompilerConfigure().setOptimizationLevel(
        CompilerConfigure::OptimizationLevel::O1);
    return std::make_unique<VM::VM>(Conf);
  }
};

TEST_F(LazyJITTest, ConfigurationDefaultDisabled) {
  Configure Conf;
  EXPECT_EQ(Conf.getRuntimeConfigure().getRunMode(), RunMode::Interpreter);
}

TEST_F(LazyJITTest, ConfigurationEnableDisable) {
  Configure Conf;

  // Enable lazy JIT
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  EXPECT_EQ(Conf.getRuntimeConfigure().getRunMode(), RunMode::LazyJIT);

  // Disable lazy JIT (back to interpreter)
  Conf.getRuntimeConfigure().setRunMode(RunMode::Interpreter);
  EXPECT_NE(Conf.getRuntimeConfigure().getRunMode(), RunMode::LazyJIT);
}

TEST_F(LazyJITTest, RuntimeConfigureIndependent) {
  RuntimeConfigure RConf1;
  RuntimeConfigure RConf2;

  RConf1.setRunMode(RunMode::LazyJIT);
  EXPECT_EQ(RConf1.getRunMode(), RunMode::LazyJIT);
  EXPECT_NE(RConf2.getRunMode(), RunMode::LazyJIT);
}

TEST_F(LazyJITTest, LazyJITCorrectness) {
  auto VM = createLazyJITVM();

  ASSERT_TRUE(VM->loadWasm(SimpleWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};

  // Test add function - should trigger lazy compilation
  std::vector<ValVariant> AddP = {10U, 5U};
  auto AddResult = VM->execute("add", AddP, Types);
  ASSERT_TRUE(AddResult);
  EXPECT_EQ((*AddResult)[0].first.get<uint32_t>(), 15U);

  // Test mul function - should trigger lazy compilation
  std::vector<ValVariant> MulP = {7U, 6U};
  auto MulResult = VM->execute("mul", MulP, Types);
  ASSERT_TRUE(MulResult);
  EXPECT_EQ((*MulResult)[0].first.get<uint32_t>(), 42U);

  // Test sub function - should trigger lazy compilation
  std::vector<ValVariant> SubP = {20U, 8U};
  auto SubResult = VM->execute("sub", SubP, Types);
  ASSERT_TRUE(SubResult);
  EXPECT_EQ((*SubResult)[0].first.get<uint32_t>(), 12U);

  // Test const42 function
  std::vector<ValVariant> EmptyP = {};
  std::vector<ValType> EmptyT = {};
  auto ConstResult = VM->execute("const42", EmptyP, EmptyT);
  ASSERT_TRUE(ConstResult);
  EXPECT_EQ((*ConstResult)[0].first.get<uint32_t>(), 42U);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITRepeatCalls) {
  auto VM = createLazyJITVM();

  ASSERT_TRUE(VM->loadWasm(SimpleWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 0U);

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};

  // First call - triggers lazy compilation
  std::vector<ValVariant> P1 = {10U, 5U};
  auto Result1 = VM->execute("add", P1, Types);
  ASSERT_TRUE(Result1);
  EXPECT_EQ((*Result1)[0].first.get<uint32_t>(), 15U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  // Second call - uses cached compiled code
  std::vector<ValVariant> P2 = {100U, 200U};
  auto Result2 = VM->execute("add", P2, Types);
  ASSERT_TRUE(Result2);
  EXPECT_EQ((*Result2)[0].first.get<uint32_t>(), 300U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  // Third call - different values
  std::vector<ValVariant> P3 = {1U, 1U};
  auto Result3 = VM->execute("add", P3, Types);
  ASSERT_TRUE(Result3);
  EXPECT_EQ((*Result3)[0].first.get<uint32_t>(), 2U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITFibonacci) {
  auto VM = createLazyJITVM();

  ASSERT_TRUE(VM->loadWasm(FibonacciWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  std::vector<ValType> Types = {ValType(TypeCode::I32)};

  // Test fibonacci function with various inputs
  // fib(0) = 1, fib(1) = 1, fib(2) = 2, fib(3) = 3, fib(4) = 5, fib(5) =8
  std::vector<std::pair<uint32_t, uint32_t>> TestCases = {
      {0, 1}, {1, 1}, {2, 2}, {3, 3}, {4, 5}, {5, 8}, {10, 89}};

  for (const auto &[Input, Expected] : TestCases) {
    std::vector<ValVariant> Params = {Input};
    auto Result = VM->execute("fib", Params, Types);
    ASSERT_TRUE(Result) << "fib(" << Input << ") failed";
    EXPECT_EQ((*Result)[0].first.get<uint32_t>(), Expected)
        << "fib(" << Input << ") = " << Expected;
  }

  VM->cleanup();
}

TEST_F(LazyJITTest, EagerVsLazyResultsMatch) {
  auto EagerVM = createEagerJITVM();
  auto LazyVM = createLazyJITVM();

  ASSERT_TRUE(EagerVM->loadWasm(SimpleWasm));
  ASSERT_TRUE(EagerVM->validate());
  ASSERT_TRUE(EagerVM->instantiate());

  ASSERT_TRUE(LazyVM->loadWasm(SimpleWasm));
  ASSERT_TRUE(LazyVM->validate());
  ASSERT_TRUE(LazyVM->instantiate());

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};

  // Test with various inputs
  std::vector<std::pair<uint32_t, uint32_t>> TestInputs = {
      {10, 5}, {0, 0}, {100, 200}, {7, 6}, {0, 100}};

  for (const auto &[A, B] : TestInputs) {
    std::vector<ValVariant> Params = {A, B};

    auto EagerAdd = EagerVM->execute("add", Params, Types);
    auto LazyAdd = LazyVM->execute("add", Params, Types);

    ASSERT_TRUE(EagerAdd);
    ASSERT_TRUE(LazyAdd);
    EXPECT_EQ((*EagerAdd)[0].first.get<uint32_t>(),
              (*LazyAdd)[0].first.get<uint32_t>())
        << "add(" << A << ", " << B << ") mismatch";

    auto EagerMul = EagerVM->execute("mul", Params, Types);
    auto LazyMul = LazyVM->execute("mul", Params, Types);

    ASSERT_TRUE(EagerMul);
    ASSERT_TRUE(LazyMul);
    EXPECT_EQ((*EagerMul)[0].first.get<uint32_t>(),
              (*LazyMul)[0].first.get<uint32_t>())
        << "mul(" << A << ", " << B << ") mismatch";
  }

  EagerVM->cleanup();
  LazyVM->cleanup();
}

TEST_F(LazyJITTest, EagerVsLazyFibonacciMatch) {
  auto EagerVM = createEagerJITVM();
  auto LazyVM = createLazyJITVM();

  ASSERT_TRUE(EagerVM->loadWasm(FibonacciWasm));
  ASSERT_TRUE(EagerVM->validate());
  ASSERT_TRUE(EagerVM->instantiate());

  ASSERT_TRUE(LazyVM->loadWasm(FibonacciWasm));
  ASSERT_TRUE(LazyVM->validate());
  ASSERT_TRUE(LazyVM->instantiate());

  std::vector<ValType> Types = {ValType(TypeCode::I32)};

  for (uint32_t N = 0; N <= 15; ++N) {
    std::vector<ValVariant> Params = {N};
    auto EagerResult = EagerVM->execute("fib", Params, Types);
    auto LazyResult = LazyVM->execute("fib", Params, Types);

    ASSERT_TRUE(EagerResult) << "Eager fib(" << N << ") failed";
    ASSERT_TRUE(LazyResult) << "Lazy fib(" << N << ") failed";
    EXPECT_EQ((*EagerResult)[0].first.get<uint32_t>(),
              (*LazyResult)[0].first.get<uint32_t>())
        << "fib(" << N << ") mismatch between eager and lazy JIT";
  }

  EagerVM->cleanup();
  LazyVM->cleanup();
}

TEST_F(LazyJITTest, LazyJITCallNonExistent) {
  auto VM = createLazyJITVM();

  ASSERT_TRUE(VM->loadWasm(SimpleWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  // Try to call a non-existent function
  std::vector<ValVariant> EmptyP = {};
  std::vector<ValType> EmptyT = {};
  auto Result = VM->execute("nonexistent", EmptyP, EmptyT);
  EXPECT_FALSE(Result);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITMultipleInstantiations) {
  for (int Idx = 0; Idx < 3; ++Idx) {
    auto VM = createLazyJITVM();

    ASSERT_TRUE(VM->loadWasm(SimpleWasm));
    ASSERT_TRUE(VM->validate());
    ASSERT_TRUE(VM->instantiate());

    std::vector<ValVariant> Params = {static_cast<uint32_t>(Idx),
                                      static_cast<uint32_t>(Idx * 10)};
    std::vector<ValType> Types = {ValType(TypeCode::I32),
                                  ValType(TypeCode::I32)};
    auto Result = VM->execute("add", Params, Types);

    ASSERT_TRUE(Result);
    EXPECT_EQ((*Result)[0].first.get<uint32_t>(),
              static_cast<uint32_t>(Idx + Idx * 10));

    VM->cleanup();
  }
}

TEST_F(LazyJITTest, LazyJITReinstantiateSameVM) {
  auto VM = createLazyJITVM();
  ASSERT_TRUE(VM->loadWasm(SimpleWasm));
  ASSERT_TRUE(VM->validate());

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {7U, 10U};

  ASSERT_TRUE(VM->instantiate());
  auto R1 = VM->execute("add", Params, Types);
  ASSERT_TRUE(R1);
  EXPECT_EQ((*R1)[0].first.get<uint32_t>(), 17U);

  // Re-instantiate the same module on the same VM without cleanup().
  ASSERT_TRUE(VM->instantiate());
  auto R2 = VM->execute("mul", Params, Types);
  ASSERT_TRUE(R2);
  EXPECT_EQ((*R2)[0].first.get<uint32_t>(), 70U);

  auto R3 = VM->execute("add", Params, Types);
  ASSERT_TRUE(R3);
  EXPECT_EQ((*R3)[0].first.get<uint32_t>(), 17U);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITReinstantiateUncompiledFunctionStillWorks) {
  auto VM = createLazyJITVM();
  ASSERT_TRUE(VM->loadWasm(SimpleWasm));
  ASSERT_TRUE(VM->validate());

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {3U, 4U};

  ASSERT_TRUE(VM->instantiate());
  auto R1 = VM->execute("add", Params, Types);
  ASSERT_TRUE(R1);
  EXPECT_EQ((*R1)[0].first.get<uint32_t>(), 7U);

  ASSERT_TRUE(VM->instantiate());

  auto R2 = VM->execute("sub", Params, Types);
  ASSERT_TRUE(R2);
  EXPECT_EQ((*R2)[0].first.get<uint32_t>(), static_cast<uint32_t>(-1));

  auto R3 = VM->execute("mul", Params, Types);
  ASSERT_TRUE(R3);
  EXPECT_EQ((*R3)[0].first.get<uint32_t>(), 12U);

  auto R4 = VM->execute("add", Params, Types);
  ASSERT_TRUE(R4);
  EXPECT_EQ((*R4)[0].first.get<uint32_t>(), 7U);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITOnlySomeFunctionsCalled) {
  auto VM = createLazyJITVM();

  ASSERT_TRUE(VM->loadWasm(SimpleWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  auto *AddFunc = VM->getActiveModule()->findFuncExports("add");
  auto *MulFunc = VM->getActiveModule()->findFuncExports("mul");
  auto *SubFunc = VM->getActiveModule()->findFuncExports("sub");

  ASSERT_NE(AddFunc, nullptr);
  ASSERT_NE(MulFunc, nullptr);
  ASSERT_NE(SubFunc, nullptr);

  // Initially none should be compiled
  EXPECT_EQ(AddFunc->getCompiledCodePtr(), nullptr);
  EXPECT_EQ(MulFunc->getCompiledCodePtr(), nullptr);
  EXPECT_EQ(SubFunc->getCompiledCodePtr(), nullptr);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 0U);

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};

  // Call 'add' should trigger compilation of function 0
  std::vector<ValVariant> AddP = {1U, 2U};
  auto AddResult = VM->execute("add", AddP, Types);
  ASSERT_TRUE(AddResult);
  EXPECT_EQ((*AddResult)[0].first.get<uint32_t>(), 3U);

  // Check state: add compiled, others not
  EXPECT_NE(AddFunc->getCompiledCodePtr(), nullptr);
  EXPECT_EQ(MulFunc->getCompiledCodePtr(), nullptr);
  EXPECT_EQ(SubFunc->getCompiledCodePtr(), nullptr);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  // Call 'mul' should trigger compilation of function 1
  std::vector<ValVariant> MulP = {3U, 4U};
  auto MulResult = VM->execute("mul", MulP, Types);
  ASSERT_TRUE(MulResult);
  EXPECT_EQ((*MulResult)[0].first.get<uint32_t>(), 12U);

  // Check state: add and mul compiled, sub not
  EXPECT_NE(AddFunc->getCompiledCodePtr(), nullptr);
  EXPECT_NE(MulFunc->getCompiledCodePtr(), nullptr);
  EXPECT_EQ(SubFunc->getCompiledCodePtr(), nullptr);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 2U);

  // Call 'add' again should NOT increase compiled count (already compiled)
  std::vector<ValVariant> AddP2 = {5U, 6U};
  auto AddResult2 = VM->execute("add", AddP2, Types);
  ASSERT_TRUE(AddResult2);
  EXPECT_EQ((*AddResult2)[0].first.get<uint32_t>(), 11U);

  // Check state: add and mul compiled, sub not
  EXPECT_NE(AddFunc->getCompiledCodePtr(), nullptr);
  EXPECT_NE(MulFunc->getCompiledCodePtr(), nullptr);
  EXPECT_EQ(SubFunc->getCompiledCodePtr(), nullptr);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 2U);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITWithImports) {
  auto VM = createLazyJITVM();

  TestEnvModule HostMod;
  ASSERT_TRUE(VM->registerModule(HostMod));

  ASSERT_TRUE(VM->loadWasm(ImportWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 0U);

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};

  // call_host delegates to the imported host_add; only the local wrapper
  // should be lazy-compiled, not the import.
  std::vector<ValVariant> P1 = {3U, 7U};
  auto R1 = VM->execute("call_host", P1, Types);
  ASSERT_TRUE(R1);
  EXPECT_EQ((*R1)[0].first.get<uint32_t>(), 10U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  // local_mul is a pure local function with no import usage.
  std::vector<ValVariant> P2 = {4U, 5U};
  auto R2 = VM->execute("local_mul", P2, Types);
  ASSERT_TRUE(R2);
  EXPECT_EQ((*R2)[0].first.get<uint32_t>(), 20U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 2U);

  // double_add calls host_add twice and adds the results.
  std::vector<ValVariant> P3 = {6U, 8U};
  auto R3 = VM->execute("double_add", P3, Types);
  ASSERT_TRUE(R3);
  EXPECT_EQ((*R3)[0].first.get<uint32_t>(), 28U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 3U);

  // Repeat call_host – count must not change (cached).
  std::vector<ValVariant> P4 = {100U, 200U};
  auto R4 = VM->execute("call_host", P4, Types);
  ASSERT_TRUE(R4);
  EXPECT_EQ((*R4)[0].first.get<uint32_t>(), 300U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 3U);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITWasmImportsWasm) {
  auto VM = createLazyJITVM();

  ASSERT_TRUE(VM->registerModule("math", MathLibWasm));

  ASSERT_TRUE(VM->loadWasm(MathConsumerWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 0U);

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};

  std::vector<ValVariant> P1 = {3U, 4U};
  auto R1 = VM->execute("add_and_square", P1, Types);
  ASSERT_TRUE(R1);
  EXPECT_EQ((*R1)[0].first.get<uint32_t>(), 49U);
  uint32_t CountAfterFirst = VM->getLazyCompiledFuncCount();
  EXPECT_EQ(CountAfterFirst, 3U);

  std::vector<ValVariant> P2 = {3U, 4U};
  auto R2 = VM->execute("sum_of_squares", P2, Types);
  ASSERT_TRUE(R2);
  EXPECT_EQ((*R2)[0].first.get<uint32_t>(), 25U);
  uint32_t CountAfterSecond = VM->getLazyCompiledFuncCount();
  EXPECT_EQ(CountAfterSecond, 4U);

  std::vector<ValVariant> P3 = {10U, 20U};
  auto R3 = VM->execute("add_and_square", P3, Types);
  ASSERT_TRUE(R3);
  EXPECT_EQ((*R3)[0].first.get<uint32_t>(), 900U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), CountAfterSecond);

  std::vector<ValVariant> P4 = {5U, 12U};
  auto R4 = VM->execute("sum_of_squares", P4, Types);
  ASSERT_TRUE(R4);
  EXPECT_EQ((*R4)[0].first.get<uint32_t>(), 169U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), CountAfterSecond);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITReferenceModuleNotReleasedOnCleanup) {
  bool Destroyed = false;
  auto Deleter = [&Destroyed](AST::Module *M) {
    Destroyed = true;
    delete M;
  };

  Configure Conf;
  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(SimpleWasm);
  ASSERT_TRUE(ModOrErr);
  auto &RawMod = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);
  auto Module = std::shared_ptr<AST::Module>(RawMod.release(), Deleter);

  auto VM = createLazyJITVM();
  ASSERT_TRUE(VM->registerModule("test", *Module));

  VM->cleanup();
  EXPECT_FALSE(Destroyed);

  Module.reset();
  EXPECT_TRUE(Destroyed);
}

TEST_F(LazyJITTest, LazyJITReferenceModuleNotReleasedOnVMDestruction) {
  bool Destroyed = false;
  auto Deleter = [&Destroyed](AST::Module *M) {
    Destroyed = true;
    delete M;
  };

  Configure Conf;
  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(SimpleWasm);
  ASSERT_TRUE(ModOrErr);
  auto &RawMod = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);
  auto Module = std::shared_ptr<AST::Module>(RawMod.release(), Deleter);

  {
    auto VM = createLazyJITVM();
    ASSERT_TRUE(VM->registerModule("test", *Module));
  }
  EXPECT_FALSE(Destroyed);

  Module.reset();
  EXPECT_TRUE(Destroyed);
}

TEST_F(LazyJITTest, JITAddLookupFailure) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Conf.getCompilerConfigure().setOptimizationLevel(
      CompilerConfigure::OptimizationLevel::O1);

  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(SimpleWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  auto ValRes = ValidatorEngine.validate(*Module);
  ASSERT_TRUE(ValRes);

  LLVM::Compiler Compiler(Conf);
  ASSERT_TRUE(Compiler.checkConfigure());

  auto Prefix = "test_prefix_";
  auto CompileRes = Compiler.compileInfrastructure(*Module, Prefix);
  ASSERT_TRUE(CompileRes);

  auto &LLData = CompileRes->first;
  auto &CompileCtx = CompileRes->second;

  LLVM::JIT JIT(Conf);
  auto ExecRes = JIT.load(LLData, true);
  ASSERT_TRUE(ExecRes);

  auto JITLib = std::static_pointer_cast<LLVM::JITLibrary>(*ExecRes);

  std::vector<uint32_t> CompileLocals = {0};
  LLData.resetModule();
  auto FuncCompileRes = Compiler.compileFunctions(LLData, CompileCtx.get(),
                                                  *Module, CompileLocals);
  ASSERT_TRUE(FuncCompileRes);

  std::vector<uint32_t> InvalidIndices = {999};

  auto AddResult = JIT.add(*JITLib, LLData, InvalidIndices);
  EXPECT_FALSE(AddResult);
  EXPECT_EQ(AddResult.error(), ErrCode::Value::LazyCompilationError);
}

TEST_F(LazyJITTest, JITLookupWasmFunctionSymbolsFailure) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Conf.getCompilerConfigure().setOptimizationLevel(
      CompilerConfigure::OptimizationLevel::O1);

  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(SimpleWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  auto ValRes = ValidatorEngine.validate(*Module);
  ASSERT_TRUE(ValRes);

  LLVM::Compiler Compiler(Conf);
  ASSERT_TRUE(Compiler.checkConfigure());

  auto Prefix = "test_prefix_";
  auto CompileRes = Compiler.compileInfrastructure(*Module, Prefix);
  ASSERT_TRUE(CompileRes);

  auto &LLData = CompileRes->first;

  LLVM::JIT JIT(Conf);
  auto ExecRes = JIT.load(LLData, true);
  ASSERT_TRUE(ExecRes);

  auto JITLib = std::static_pointer_cast<LLVM::JITLibrary>(*ExecRes);

  std::vector<uint32_t> InvalidIndices = {999};

  auto LookupRes =
      JIT.lookupWasmFunctionSymbols(*JITLib, Prefix, InvalidIndices);
  EXPECT_FALSE(LookupRes);
  EXPECT_EQ(LookupRes.error(), ErrCode::Value::LazyCompilationError);
}

TEST_F(LazyJITTest, JITResolveSymbolsEmptyBatch) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Conf.getCompilerConfigure().setOptimizationLevel(
      CompilerConfigure::OptimizationLevel::O1);

  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(SimpleWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*Module));

  LLVM::Compiler Compiler(Conf);
  ASSERT_TRUE(Compiler.checkConfigure());

  auto Prefix = "test_prefix_";
  auto CompileRes = Compiler.compileInfrastructure(*Module, Prefix);
  ASSERT_TRUE(CompileRes);

  auto &LLData = CompileRes->first;

  LLVM::JIT JIT(Conf);
  auto ExecRes = JIT.load(LLData, true);
  ASSERT_TRUE(ExecRes);

  auto JITLib = std::static_pointer_cast<LLVM::JITLibrary>(*ExecRes);

  std::vector<uint32_t> EmptyIndices;
  auto LookupRes = JIT.lookupWasmFunctionSymbols(*JITLib, Prefix, EmptyIndices);
  EXPECT_TRUE(LookupRes);
  if (LookupRes) {
    EXPECT_TRUE(LookupRes->empty());
  }
}

TEST_F(LazyJITTest, CompileFunctionsFailureKeepsDataValid) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Conf.getCompilerConfigure().setOptimizationLevel(
      CompilerConfigure::OptimizationLevel::O1);

  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(SimpleWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*Module));

  LLVM::Compiler Compiler(Conf);
  ASSERT_TRUE(Compiler.checkConfigure());

  auto CompileRes = Compiler.compileInfrastructure(*Module, "test_prefix_");
  ASSERT_TRUE(CompileRes);
  auto &LLData = CompileRes->first;
  auto &CompileCtx = CompileRes->second;

  std::vector<uint32_t> BadLocals = {999};
  auto BadRes =
      Compiler.compileFunctions(LLData, CompileCtx.get(), *Module, BadLocals);
  EXPECT_FALSE(BadRes);

  EXPECT_TRUE(LLData.isValid());
  EXPECT_TRUE(LLData.hasModule());
}

TEST_F(LazyJITTest, DataAccessorsOnInvalidDataAreSafe) {
  LLVM::Data D1;
  EXPECT_TRUE(D1.isValid());

  LLVM::Data D2(std::move(D1));
  EXPECT_TRUE(D2.isValid());
  EXPECT_FALSE(D1.isValid());
  EXPECT_FALSE(D1.hasModule());
  EXPECT_TRUE(D1.getPrefix().empty());
  D1.resetModule();
  EXPECT_FALSE(D1.hasModule());
}

// Unlike the value-returning accessors above, extract() returns a reference and
// cannot fall back to a safe default for a moved-from Data. It must fail with a
// defined abort rather than dereferencing a null Context (release-build UB).
TEST_F(LazyJITTest, DataExtractOnMovedFromAborts) {
  LLVM::Data D1;
  ASSERT_TRUE(D1.isValid());
  LLVM::Data D2(std::move(D1));
  ASSERT_FALSE(D1.isValid());

  EXPECT_DEATH({ (void)D1.extract(); }, "");
}

TEST_F(LazyJITTest, JITAddDuplicateModuleKeepsLibraryUsable) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Conf.getCompilerConfigure().setOptimizationLevel(
      CompilerConfigure::OptimizationLevel::O1);

  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(SimpleWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*Module));

  LLVM::Compiler Compiler(Conf);
  ASSERT_TRUE(Compiler.checkConfigure());

  auto Prefix = "dup_prefix_";
  auto CompileRes = Compiler.compileInfrastructure(*Module, Prefix);
  ASSERT_TRUE(CompileRes);
  auto &LLData = CompileRes->first;
  auto &CompileCtx = CompileRes->second;

  LLVM::JIT JIT(Conf);
  auto ExecRes = JIT.load(LLData, true);
  ASSERT_TRUE(ExecRes);
  auto JITLib = std::static_pointer_cast<LLVM::JITLibrary>(*ExecRes);

  std::vector<uint32_t> Locals = {0};
  std::vector<uint32_t> Globals = {0};

  LLData.resetModule();
  ASSERT_TRUE(Compiler.compileFunctions(LLData, CompileCtx.get(), *Module,
                                        Locals));
  auto Add1 = JIT.add(*JITLib, LLData, Globals);
  ASSERT_TRUE(Add1);

  // Re-adding the same symbol collides; addLLVMIRModuleWithRT fails and must
  // remove its tracker without disturbing the already-resolved definition.
  LLData.resetModule();
  ASSERT_TRUE(Compiler.compileFunctions(LLData, CompileCtx.get(), *Module,
                                        Locals));
  auto Add2 = JIT.add(*JITLib, LLData, Globals);
  EXPECT_FALSE(Add2);

  auto Lookup = JIT.lookupWasmFunctionSymbols(*JITLib, Prefix, Globals);
  EXPECT_TRUE(Lookup);
}

TEST_F(LazyJITTest, LazyJITConcurrentSameFunction) {
  auto VM = createLazyJITVM();

  ASSERT_TRUE(VM->loadWasm(SimpleWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};
  std::vector<std::thread> Threads;
  const size_t NumThreads = 8;
  Threads.reserve(NumThreads);

  for (size_t Idx = 0; Idx < NumThreads; ++Idx) {
    Threads.emplace_back([&VM, &Types, Idx]() {
      std::vector<ValVariant> Params = {static_cast<uint32_t>(Idx), 10U};
      auto Result = VM->execute("add", Params, Types);
      EXPECT_TRUE(Result);
      if (Result) {
        EXPECT_EQ((*Result)[0].first.get<uint32_t>(),
                  static_cast<uint32_t>(Idx + 10));
      }
    });
  }

  for (auto &T : Threads) {
    T.join();
  }
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITConcurrentDifferentFunctions) {
  auto VM = createLazyJITVM();

  ASSERT_TRUE(VM->loadWasm(SimpleWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  std::vector<ValType> Types2 = {ValType(TypeCode::I32),
                                 ValType(TypeCode::I32)};
  std::vector<ValType> Types0 = {};

  std::vector<std::thread> Threads;
  Threads.emplace_back([&VM, &Types2]() {
    std::vector<ValVariant> Params = {10U, 20U};
    auto Result = VM->execute("add", Params, Types2);
    EXPECT_TRUE(Result);
    if (Result) {
      EXPECT_EQ((*Result)[0].first.get<uint32_t>(), 30U);
    }
  });
  Threads.emplace_back([&VM, &Types2]() {
    std::vector<ValVariant> Params = {5U, 6U};
    auto Result = VM->execute("mul", Params, Types2);
    EXPECT_TRUE(Result);
    if (Result) {
      EXPECT_EQ((*Result)[0].first.get<uint32_t>(), 30U);
    }
  });
  Threads.emplace_back([&VM, &Types2]() {
    std::vector<ValVariant> Params = {50U, 20U};
    auto Result = VM->execute("sub", Params, Types2);
    EXPECT_TRUE(Result);
    if (Result) {
      EXPECT_EQ((*Result)[0].first.get<uint32_t>(), 30U);
    }
  });
  Threads.emplace_back([&VM, &Types0]() {
    std::vector<ValVariant> Params = {};
    auto Result = VM->execute("const42", Params, Types0);
    EXPECT_TRUE(Result);
    if (Result) {
      EXPECT_EQ((*Result)[0].first.get<uint32_t>(), 42U);
    }
  });

  for (auto &T : Threads) {
    T.join();
  }
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 4U);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITConcurrentFibonacci) {
  auto VM = createLazyJITVM();

  ASSERT_TRUE(VM->loadWasm(FibonacciWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  std::vector<ValType> Types = {ValType(TypeCode::I32)};
  std::vector<std::thread> Threads;
  const size_t NumThreads = 8;
  Threads.reserve(NumThreads);

  for (size_t Idx = 0; Idx < NumThreads; ++Idx) {
    Threads.emplace_back([&VM, &Types, Idx]() {
      std::vector<ValVariant> Params = {static_cast<uint32_t>(Idx)};
      auto Result = VM->execute("fib", Params, Types);
      EXPECT_TRUE(Result);
      if (Result) {
        uint32_t Expected = 0;
        if (Idx == 0 || Idx == 1) {
          Expected = 1;
        } else {
          uint32_t Prev2 = 1;
          uint32_t Prev1 = 1;
          for (size_t Jdx = 2; Jdx <= Idx; ++Jdx) {
            Expected = Prev2 + Prev1;
            Prev2 = Prev1;
            Prev1 = Expected;
          }
        }
        EXPECT_EQ((*Result)[0].first.get<uint32_t>(), Expected);
      }
    });
  }

  for (auto &T : Threads) {
    T.join();
  }
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  VM->cleanup();
}

// Registering the same AST::Module object under two names creates two
// ModuleInstances sharing one module ID. Each registration must get its own
// fresh lazy-JIT state so the second instance compiles independently instead of
// reusing the first instance's stale compiled-function set. Calling "f" (which
// calls "g") on the re-registered instance must execute correctly: before the
// fix the stale set skipped "g", so it was never upgraded on the new instance
// and the inline cache resolved it to null.
TEST_F(LazyJITTest, LazyJITReRegisterSameModuleObjectCompilesIndependently) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(CallerCalleeWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*Module));

  auto VM = createLazyJITVM();

  std::vector<ValType> Types = {ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {7U};

  // First registration; calling "g" compiles it on the first instance.
  ASSERT_TRUE(VM->registerModule("a", *Module));
  auto Rg = VM->execute("a", "g", Params, Types);
  ASSERT_TRUE(Rg);
  EXPECT_EQ((*Rg)[0].first.get<uint32_t>(), 8U);

  // Re-register the SAME module object; the second instance must not inherit
  // the first instance's stale compiled-function set.
  ASSERT_TRUE(VM->registerModule("b", *Module));

  // "f" calls "g"; on the fresh instance both compile, so the call succeeds.
  auto Rf = VM->execute("b", "f", Params, Types);
  ASSERT_TRUE(Rf);
  EXPECT_EQ((*Rf)[0].first.get<uint32_t>(), 8U);

  VM->cleanup();
}

// Unregistering a module must drop its lazy-JIT state, not leak it (and not
// keep counting its compiled functions) until cleanup().
TEST_F(LazyJITTest, LazyJITUnregisterDropsCompiledState) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(SimpleWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*Module));

  auto VM = createLazyJITVM();
  ASSERT_TRUE(VM->registerModule("m", *Module));

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {3U, 4U};
  auto R = VM->execute("m", "add", Params, Types);
  ASSERT_TRUE(R);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  ASSERT_TRUE(VM->unregisterModule("m"));
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 0U);

  VM->cleanup();
}

// Two registrations of the same module object share one ID; unregistering one
// must not drop the lazy-JIT state the surviving instance still uses.
TEST_F(LazyJITTest, LazyJITUnregisterKeepsStateForOtherSameIdInstance) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(CallerCalleeWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*Module));

  auto VM = createLazyJITVM();
  ASSERT_TRUE(VM->registerModule("a", *Module));
  ASSERT_TRUE(VM->registerModule("b", *Module));

  std::vector<ValType> Types = {ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {7U};

  // Unregister "a"; "b" shares the module ID and must keep compiling/running.
  ASSERT_TRUE(VM->unregisterModule("a"));
  auto R = VM->execute("b", "f", Params, Types);
  ASSERT_TRUE(R);
  EXPECT_EQ((*R)[0].first.get<uint32_t>(), 8U);
  EXPECT_GT(VM->getLazyCompiledFuncCount(), 0U);

  VM->cleanup();
}

// Regression test for finding #1: per-ID compiled-function set strands a sibling instance.
// When the same AST::Module is registered twice (two live instances, one shared ID),
// compiling a callee on instance 'a' inserts it into the shared set; the batch collector
// then excludes that callee when compiling a caller on instance 'b', leaving b's callee
// uncompiled. The compiled caller on 'b' hits an inline-cache miss and traps with
// LazyCompilationError.
//
// This test reproduces the exact failing scenario:
// 1. Register "a" and "b" with the same module (they share module ID)
// 2. Execute "a.g" (compiles g, adds local-idx(g) to shared LazyCompiledFuncs)
// 3. Execute "b.f" (f calls g, but g is excluded from batch because it's in the set)
//    → b's g FunctionInstance is never upgraded
//    → when b.f's compiled code hits the call to g, the inline cache resolves to null
//    → emitLazyCall traps with LazyCompilationError (Code 0x00f)
TEST_F(LazyJITTest, LazyJITSameModuleIdStrandsCalleeOnSiblingInstance) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(CallerCalleeWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*Module));

  auto VM = createLazyJITVM();

  std::vector<ValType> Types = {ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {7U};

  // Both instances register the same module object (shared ID).
  ASSERT_TRUE(VM->registerModule("a", *Module));
  ASSERT_TRUE(VM->registerModule("b", *Module));

  // Execute "a.g" first, which compiles g on instance 'a' and inserts it into the
  // shared LazyCompiledFuncs set (keyed per module ID).
  auto Rg = VM->execute("a", "g", Params, Types);
  ASSERT_TRUE(Rg) << "First execution of a.g should succeed";
  EXPECT_EQ((*Rg)[0].first.get<uint32_t>(), 8U);

  // Now execute "b.f" where f calls g. Before the fix, the shared LazyCompiledFuncs set
  // marked g compiled (from instance 'a'), so b's g was never upgraded and b.f's compiled
  // code trapped resolving g to null. The fix resolves g's already-present symbol from the
  // shared dylib and upgrades b's g instance on demand instead of leaving it stranded.
  auto Rf = VM->execute("b", "f", Params, Types);
  ASSERT_TRUE(Rf) << "Second execution of b.f should succeed (but currently traps with "
                     "LazyCompilationError due to b.g being stranded)";
  EXPECT_EQ((*Rf)[0].first.get<uint32_t>(), 8U);

  // Verify that both functions are compiled on both instances.
  // After the fix, both instances should have compiled both functions.
  EXPECT_GT(VM->getLazyCompiledFuncCount(), 0U);

  VM->cleanup();
}

// A registration runs prepare() before ExecutorEngine.registerModule, so a
// registration that fails afterwards (here, a duplicate module name) leaves the
// module's symbol set while discardState() drops its tracking. A later
// registration of the same module object must still prepare it for lazy JIT,
// not mistake the leftover symbol for externally-provided (AOT) code and skip.
TEST_F(LazyJITTest, LazyJITReregisterAfterFailedRegistrationStillCompiles) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Loader::Loader LoaderEngine(Conf);

  auto ModA = LoaderEngine.parseWasmUnit(SimpleWasm);
  ASSERT_TRUE(ModA);
  auto &ModuleA = std::get<std::unique_ptr<AST::Module>>(*ModA);
  auto ModB = LoaderEngine.parseWasmUnit(CallerCalleeWasm);
  ASSERT_TRUE(ModB);
  auto &ModuleB = std::get<std::unique_ptr<AST::Module>>(*ModB);

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*ModuleA));
  ASSERT_TRUE(ValidatorEngine.validate(*ModuleB));

  auto VM = createLazyJITVM();

  // Occupy name "n" with module A.
  ASSERT_TRUE(VM->registerModule("n", *ModuleA));

  // Registering B under the same name fails the name-conflict check, but only
  // after prepare(B) already ran and set B's symbol; the failure path then
  // discards B's tracked state.
  ASSERT_FALSE(VM->registerModule("n", *ModuleB));

  // Re-register the same B object under a fresh name. It must be lazily prepared
  // despite the leftover symbol.
  ASSERT_TRUE(VM->registerModule("n2", *ModuleB));

  std::vector<ValType> Types = {ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {7U};
  auto R = VM->execute("n2", "g", Params, Types);
  ASSERT_TRUE(R);
  EXPECT_EQ((*R)[0].first.get<uint32_t>(), 8U);

  // Discriminator: a JIT-prepared module compiles g on demand. If prepare() were
  // skipped (the bug), g would run in the interpreter and the count stay 0.
  EXPECT_GT(VM->getLazyCompiledFuncCount(), 0U);

  VM->cleanup();
}

// Re-instantiating the same module reuses its lazy-JIT dylib instead of
// rebuilding it: prepare() is idempotent for an already-tracked module ID. The
// previously compiled functions stay compiled (the new instance upgrades them on
// demand), so the compiled-function count persists across re-instantiation.
TEST_F(LazyJITTest, LazyJITReinstantiateReusesCompiledState) {
  auto VM = createLazyJITVM();
  ASSERT_TRUE(VM->loadWasm(SimpleWasm));
  ASSERT_TRUE(VM->validate());

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {7U, 10U};

  ASSERT_TRUE(VM->instantiate());
  ASSERT_TRUE(VM->execute("add", Params, Types));
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  // Re-instantiate the same module. With an idempotent prepare(), the dylib and
  // its compiled-function set are reused rather than rebuilt from scratch.
  ASSERT_TRUE(VM->instantiate());
  EXPECT_GT(VM->getLazyCompiledFuncCount(), 0U);

  // The reused instance still runs correctly (add upgraded on demand).
  auto R = VM->execute("add", Params, Types);
  ASSERT_TRUE(R);
  EXPECT_EQ((*R)[0].first.get<uint32_t>(), 17U);

  VM->cleanup();
}

// runWasmFile is the one-shot load+validate+instantiate+execute API. In lazy
// JIT mode it must engage the JIT exactly like the explicit instantiate() path:
// the run path has to invoke the execution strategy's instantiation hook so the
// module is prepared and its functions compile on demand. Before the fix this
// path skipped the hook, so the whole runWasmFile family silently ran in the
// interpreter and never compiled anything.
TEST_F(LazyJITTest, LazyJITRunWasmFileCompilesOnDemand) {
  auto VM = createLazyJITVM();

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {7U, 10U};

  auto R = VM->runWasmFile(SimpleWasm, "add", Params, Types);
  ASSERT_TRUE(R);
  EXPECT_EQ((*R)[0].first.get<uint32_t>(), 17U);

  // Discriminator: a JIT-prepared module compiles "add" on demand. If the run
  // path skipped preparation, "add" would run interpreted and the count stay 0.
  EXPECT_GT(VM->getLazyCompiledFuncCount(), 0U);

  VM->cleanup();
}

// Finding #2: compiling a function wires its whole statically-reachable closure
// on THIS instance, not just the functions freshly compiled in the batch. With
// two instances sharing a module ID, compiling callee "g" on instance "a" puts
// it in the shared per-ID compiled-set; later compiling caller "f" on instance
// "b" must still wire b's own "g" FunctionInstance, even though this call to f
// (argument 0, else branch) never dynamically reaches g. Before the fix b's "g"
// was stranded (null compiled-code pointer) until something happened to trigger
// it, so an indirect call or stack-trace lookup observed an un-upgraded instance.
TEST_F(LazyJITTest, LazyJITSiblingWiresUncalledReachableCallee) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(CondCallWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*Module));

  auto VM = createLazyJITVM();
  ASSERT_TRUE(VM->registerModule("a", *Module));
  ASSERT_TRUE(VM->registerModule("b", *Module));

  std::vector<ValType> Types = {ValType(TypeCode::I32)};

  // Compile "g" on instance "a"; it enters the shared per-ID compiled-set.
  std::vector<ValVariant> CallG = {7U};
  auto Rg = VM->execute("a", "g", CallG, Types);
  ASSERT_TRUE(Rg);
  EXPECT_EQ((*Rg)[0].first.get<uint32_t>(), 8U);

  // Call "f" on instance "b" with 0: f reaches g statically but skips the call
  // at run time. Compiling f must still wire b's own "g".
  std::vector<ValVariant> NoCallG = {0U};
  auto Rf = VM->execute("b", "f", NoCallG, Types);
  ASSERT_TRUE(Rf);
  EXPECT_EQ((*Rf)[0].first.get<uint32_t>(), 0U);

  const auto *BInst = VM->getStoreManager().findModule("b");
  ASSERT_NE(BInst, nullptr);
  const auto *BG = BInst->findFuncExports("g");
  ASSERT_NE(BG, nullptr);
  EXPECT_NE(BG->getCompiledCodePtr(), nullptr);

  VM->cleanup();
}

// Two DIFFERENT modules with empty IDs must not share lazy-JIT state. Module
// IDs are assigned only by the lazy loader, so a caller-supplied AST::Module
// keeps an empty ID. Keying per-module state on an empty ID collapses every
// such module onto States[""]: the second module reuses the first's dylib/AST
// and its FunctionInstances get wired against the first module's compiled code.
// Empty-ID modules must instead run in the interpreter, as they did before lazy
// JIT existed.
TEST_F(LazyJITTest, LazyJITEmptyModuleIdRunsInterpretedWithoutCollision) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Loader::Loader LoaderEngine(Conf);

  // CallerCallee: f(x) = g(x) = x + 1. CondCall: f(x) = x ? g(x) : x.
  auto ModA = LoaderEngine.parseWasmUnit(CallerCalleeWasm);
  ASSERT_TRUE(ModA);
  auto &ModuleA = std::get<std::unique_ptr<AST::Module>>(*ModA);
  ModuleA->setID("");

  auto ModB = LoaderEngine.parseWasmUnit(CondCallWasm);
  ASSERT_TRUE(ModB);
  auto &ModuleB = std::get<std::unique_ptr<AST::Module>>(*ModB);
  ModuleB->setID("");

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*ModuleA));
  ASSERT_TRUE(ValidatorEngine.validate(*ModuleB));

  auto VM = createLazyJITVM();
  ASSERT_TRUE(VM->registerModule("a", *ModuleA));
  ASSERT_TRUE(VM->registerModule("b", *ModuleB));

  std::vector<ValType> Types = {ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {0U};

  // f(0): CallerCallee returns 1, CondCall returns 0. If b were wired against
  // a's compiled f (the collision bug), b.f(0) would return 1 instead of 0.
  auto Ra = VM->execute("a", "f", Params, Types);
  ASSERT_TRUE(Ra);
  EXPECT_EQ((*Ra)[0].first.get<uint32_t>(), 1U);

  auto Rb = VM->execute("b", "f", Params, Types);
  ASSERT_TRUE(Rb);
  EXPECT_EQ((*Rb)[0].first.get<uint32_t>(), 0U);

  // Empty-ID modules are not tracked, so nothing is JIT-compiled.
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 0U);

  VM->cleanup();
}

// A registration runs prepare() before ExecutorEngine.registerModule, so a
// registration that fails afterwards (here, re-registering the same module
// under a name that is already taken) must not discard the lazy-JIT state a
// still-live instance with that module ID depends on. prepare() early-returns
// for the already-tracked ID, so the failed attempt added nothing; dropping the
// shared state would strand the surviving instance (its functions could never
// compile and getLazyCompiledFuncCount() would fall to zero).
TEST_F(LazyJITTest, LazyJITFailedReregisterKeepsStateForLiveSibling) {
  Configure Conf;
  Conf.getRuntimeConfigure().setRunMode(RunMode::LazyJIT);
  Loader::Loader LoaderEngine(Conf);
  auto ModOrErr = LoaderEngine.parseWasmUnit(SimpleWasm);
  ASSERT_TRUE(ModOrErr);
  auto &Module = std::get<std::unique_ptr<AST::Module>>(*ModOrErr);

  Validator::Validator ValidatorEngine(Conf);
  ASSERT_TRUE(ValidatorEngine.validate(*Module));

  auto VM = createLazyJITVM();
  ASSERT_TRUE(VM->registerModule("a", *Module));

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {3U, 4U};

  // Compile "add" on the registered instance.
  auto RAdd = VM->execute("a", "add", Params, Types);
  ASSERT_TRUE(RAdd);
  EXPECT_EQ((*RAdd)[0].first.get<uint32_t>(), 7U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  // Re-register the SAME module under the SAME name: registration fails on the
  // name conflict after prepare() already ran (and early-returned for the
  // already-tracked ID). The failure must not discard the state "a" still uses.
  ASSERT_FALSE(VM->registerModule("a", *Module));

  // State preserved: the previously compiled count is intact...
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 1U);

  // ...and a not-yet-compiled function still compiles on demand.
  auto RMul = VM->execute("a", "mul", Params, Types);
  ASSERT_TRUE(RMul);
  EXPECT_EQ((*RMul)[0].first.get<uint32_t>(), 12U);
  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 2U);

  VM->cleanup();
}

// runWasmFile is the one-shot load+validate+instantiate+execute API. In eager
// JIT mode it must AOT-compile the module through the execution strategy's
// instantiation hook, exactly like the explicit instantiate() path, so the
// executed function runs as compiled code instead of the interpreter. This is
// the eager-JIT companion to LazyJITRunWasmFileCompilesOnDemand; before the run
// path invoked the hook, the whole runWasmFile family silently ran interpreted
// under eager JIT too.
TEST_F(LazyJITTest, EagerJITRunWasmFileCompilesModule) {
  auto VM = createEagerJITVM();

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {3U, 4U};

  auto R = VM->runWasmFile(SimpleWasm, "add", Params, Types);
  ASSERT_TRUE(R);
  EXPECT_EQ((*R)[0].first.get<uint32_t>(), 7U);

  // Discriminator: eager JIT must have AOT-compiled the module, so the executed
  // function carries compiled code. If the run path skipped compilation, "add"
  // would run interpreted with a null compiled-code pointer.
  const auto *ActiveMod = VM->getActiveModule();
  ASSERT_NE(ActiveMod, nullptr);
  const auto *AddFunc = ActiveMod->findFuncExports("add");
  ASSERT_NE(AddFunc, nullptr);
  EXPECT_NE(AddFunc->getCompiledCodePtr(), nullptr);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITFailedInstantiationCleansUpState) {
  auto VM = createLazyJITVM();

  // ImportWasm requires "env"/"host_add" — instantiation will fail when the
  // host module is not registered, triggering the failure-cleanup path.
  ASSERT_TRUE(VM->loadWasm(ImportWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_FALSE(VM->instantiate());

  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 0U);

  // Now load a self-contained module. If the failed instantiation leaked state,
  // this would either collide with the stale entry or leave the manager in an
  // inconsistent state.
  ASSERT_TRUE(VM->loadWasm(SimpleWasm));
  ASSERT_TRUE(VM->validate());
  ASSERT_TRUE(VM->instantiate());

  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {3U, 4U};
  auto R = VM->execute("add", Params, Types);
  ASSERT_TRUE(R);
  EXPECT_EQ((*R)[0].first.get<uint32_t>(), 7U);
  EXPECT_GT(VM->getLazyCompiledFuncCount(), 0U);

  VM->cleanup();
}

TEST_F(LazyJITTest, LazyJITFailedRunWasmFileCleansUpState) {
  auto VM = createLazyJITVM();

  // runWasmFile with ImportWasm will fail at instantiation (missing host
  // module), exercising the unsafeRunWasmFile failure-cleanup path.
  std::vector<ValType> Types = {ValType(TypeCode::I32), ValType(TypeCode::I32)};
  std::vector<ValVariant> Params = {3U, 7U};
  auto R1 = VM->runWasmFile(ImportWasm, "call_host", Params, Types);
  ASSERT_FALSE(R1);

  EXPECT_EQ(VM->getLazyCompiledFuncCount(), 0U);

  // A subsequent runWasmFile with a self-contained module must succeed and
  // compile on demand, proving the failed run's state was cleaned up.
  auto R2 = VM->runWasmFile(SimpleWasm, "add", Params, Types);
  ASSERT_TRUE(R2);
  EXPECT_EQ((*R2)[0].first.get<uint32_t>(), 10U);
  EXPECT_GT(VM->getLazyCompiledFuncCount(), 0U);

  VM->cleanup();
}

} // namespace

GTEST_API_ int main(int argc, char **argv) {
  WasmEdge::Log::setErrorLoggingLevel();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
