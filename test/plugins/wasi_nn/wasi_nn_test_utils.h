// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinnfunc.h"
#include "wasinnmodule.h"

#include "common/types.h"
#include "runtime/callingframe.h"
#include "runtime/instance/memory.h"
#include "runtime/instance/module.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace WasmEdge::Host::WASINN::Testing {

using namespace std::literals;
using WasmEdge::Host::WASINN::Backend;
using WasmEdge::Host::WASINN::Device;
using WasmEdge::Host::WASINN::ErrNo;
using WasmEdge::Host::WASINN::TensorType;

template <typename T, typename U>
inline std::unique_ptr<T> dynamicPointerCast(std::unique_ptr<U> &&R) noexcept {
  static_assert(std::has_virtual_destructor_v<T>);
  T *P = dynamic_cast<T *>(R.get());
  if (P) {
    R.release();
  }
  return std::unique_ptr<T>(P);
}

inline std::unique_ptr<WasmEdge::Host::WasiNNModule>
createModule(std::string_view NNRPCURI = "") {
  WasmEdge::Plugin::Plugin::load(
      std::filesystem::u8path("../../../plugins/wasi_nn/" WASMEDGE_LIB_PREFIX
                              "wasmedgePluginWasiNN" WASMEDGE_LIB_EXTENSION));
  if (const auto *Plugin = WasmEdge::Plugin::Plugin::find("wasi_nn"sv)) {
    WasmEdge::PO::ArgumentParser Parser;
    Plugin->registerOptions(Parser);
    if (NNRPCURI != "") {
      Parser.set_raw_value<std::string>("nn-rpc-uri"sv, std::string(NNRPCURI));
    }
    if (const auto *Module = Plugin->findModule("wasi_nn"sv)) {
      return dynamicPointerCast<WasmEdge::Host::WasiNNModule>(Module->create());
    }
  }
  return {};
}

class WasiNNTestContext {
public:
  explicit WasiNNTestContext(uint32_t MemoryPageCount,
                             std::string_view NNRPCURI = "")
      : NNMod(createModule(NNRPCURI)), Mod(""), CallFrame(nullptr, &Mod) {
    if (NNMod) {
      Mod.addHostMemory(
          "memory",
          std::make_unique<WasmEdge::Runtime::Instance::MemoryInstance>(
              WasmEdge::AST::MemoryType(MemoryPageCount)));
      MemInst = Mod.findMemoryExports("memory");
    }
  }

  bool isValid() const noexcept {
    return NNMod != nullptr && MemInst != nullptr;
  }

  WasmEdge::Host::WasiNNModule &module() noexcept {
    assuming(NNMod != nullptr);
    return *NNMod;
  }

  WasmEdge::Runtime::Instance::MemoryInstance &memory() noexcept {
    assuming(MemInst != nullptr);
    return *MemInst;
  }

  WasmEdge::Runtime::CallingFrame &frame() noexcept { return CallFrame; }

  template <typename T> T &hostFunc(std::string_view Name) {
    auto *FuncInst = module().findFuncExports(Name);
    EXPECT_NE(FuncInst, nullptr);
    assuming(FuncInst != nullptr);
    EXPECT_TRUE(FuncInst->isHostFunction());
    return dynamic_cast<T &>(FuncInst->getHostFunc());
  }

private:
  std::unique_ptr<WasmEdge::Host::WasiNNModule> NNMod;
  WasmEdge::Runtime::Instance::ModuleInstance Mod;
  WasmEdge::Runtime::CallingFrame CallFrame;
  WasmEdge::Runtime::Instance::MemoryInstance *MemInst = nullptr;
};

template <typename T>
T &getHostFunc(WasmEdge::Host::WasiNNModule &Module, std::string_view Name) {
  auto *FuncInst = Module.findFuncExports(Name);
  EXPECT_NE(FuncInst, nullptr);
  assuming(FuncInst != nullptr);
  EXPECT_TRUE(FuncInst->isHostFunction());
  return dynamic_cast<T &>(FuncInst->getHostFunc());
}

inline void expectErrNo(const std::array<WasmEdge::ValVariant, 1> &Result,
                        ErrNo Expected) noexcept {
  EXPECT_EQ(static_cast<uint32_t>(Result[0].get<int32_t>()),
            static_cast<uint32_t>(Expected));
}

inline std::vector<uint8_t> readEntireFile(const std::string &Path) {
  std::ifstream Fin(Path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!Fin) {
    return {};
  }
  std::vector<uint8_t> Buf(static_cast<std::size_t>(Fin.tellg()));
  Fin.seekg(0, std::ios::beg);
  if (!Fin.read(reinterpret_cast<char *>(Buf.data()),
                static_cast<std::streamsize>(Buf.size()))) {
    return {};
  }
  Fin.close();
  return Buf;
}

template <typename T>
void writeBinaries(WasmEdge::Runtime::Instance::MemoryInstance &MemInst,
                   WasmEdge::Span<const T> Binaries, uint32_t Ptr) noexcept {
  std::copy(Binaries.begin(), Binaries.end(), MemInst.getPointer<T *>(Ptr));
}

inline void writeUInt32(WasmEdge::Runtime::Instance::MemoryInstance &MemInst,
                        uint32_t Value, uint32_t &Ptr) {
  MemInst.storeValue(Value, Ptr);
  Ptr += 4;
}

inline void
writeFatPointer(WasmEdge::Runtime::Instance::MemoryInstance &MemInst,
                uint32_t PtrVal, uint32_t PtrSize, uint32_t &Ptr) {
  writeUInt32(MemInst, PtrVal, Ptr);
  writeUInt32(MemInst, PtrSize, Ptr);
}

inline uint32_t
writeGraphBuilderEntry(WasmEdge::Runtime::Instance::MemoryInstance &MemInst,
                       uint32_t &BuilderPtr, uint32_t DataPtr,
                       WasmEdge::Span<const uint8_t> Data) {
  writeFatPointer(MemInst, DataPtr, static_cast<uint32_t>(Data.size()),
                  BuilderPtr);
  writeBinaries<uint8_t>(MemInst, Data, DataPtr);
  return DataPtr + static_cast<uint32_t>(Data.size());
}

inline uint32_t
writeTensor(WasmEdge::Runtime::Instance::MemoryInstance &MemInst,
            uint32_t TensorPtr, uint32_t &NextTensorPtr, uint32_t DataPtr,
            WasmEdge::Span<const uint32_t> Dimensions, TensorType Type,
            WasmEdge::Span<const uint8_t> TensorData) {
  const auto TensorDataPtr =
      DataPtr + static_cast<uint32_t>(Dimensions.size() * sizeof(uint32_t));
  NextTensorPtr = TensorPtr;
  writeFatPointer(MemInst, DataPtr, static_cast<uint32_t>(Dimensions.size()),
                  NextTensorPtr);
  writeUInt32(MemInst, static_cast<uint32_t>(Type), NextTensorPtr);
  writeFatPointer(MemInst, TensorDataPtr,
                  static_cast<uint32_t>(TensorData.size()), NextTensorPtr);
  writeBinaries<uint32_t>(MemInst, Dimensions, DataPtr);
  writeBinaries<uint8_t>(MemInst, TensorData, TensorDataPtr);
  return TensorDataPtr + static_cast<uint32_t>(TensorData.size());
}

class MemoryWriter {
public:
  MemoryWriter(WasmEdge::Runtime::Instance::MemoryInstance &M,
               uint32_t StartPtr) noexcept
      : MemInst(M), NextPtr(StartPtr) {}

  uint32_t ptr() const noexcept { return NextPtr; }
  void reset(uint32_t Ptr) noexcept { NextPtr = Ptr; }

  template <typename T> uint32_t write(WasmEdge::Span<const T> Data) noexcept {
    const uint32_t Ptr = NextPtr;
    writeBinaries<T>(MemInst, Data, Ptr);
    NextPtr += static_cast<uint32_t>(Data.size() * sizeof(T));
    return Ptr;
  }

  uint32_t writeGraphBuilderEntry(uint32_t &BuilderPtr,
                                  WasmEdge::Span<const uint8_t> Data) {
    const uint32_t Ptr = NextPtr;
    NextPtr = ::WasmEdge::Host::WASINN::Testing::writeGraphBuilderEntry(
        MemInst, BuilderPtr, NextPtr, Data);
    return Ptr;
  }

  uint32_t writeTensor(uint32_t TensorPtr, uint32_t &NextTensorPtr,
                       WasmEdge::Span<const uint32_t> Dimensions,
                       TensorType Type,
                       WasmEdge::Span<const uint8_t> TensorData) {
    const uint32_t Ptr = NextPtr;
    NextPtr = ::WasmEdge::Host::WASINN::Testing::writeTensor(
        MemInst, TensorPtr, NextTensorPtr, NextPtr, Dimensions, Type,
        TensorData);
    return Ptr;
  }

private:
  WasmEdge::Runtime::Instance::MemoryInstance &MemInst;
  uint32_t NextPtr;
};

template <typename T>
std::vector<size_t> classSort(WasmEdge::Span<const T> Array) {
  std::vector<size_t> Indices(Array.size());
  std::iota(Indices.begin(), Indices.end(), 0);
  std::sort(Indices.begin(), Indices.end(),
            [&Array](size_t Left, size_t Right) -> bool {
              return Array[Left] > Array[Right];
            });
  return Indices;
}

} // namespace WasmEdge::Host::WASINN::Testing
