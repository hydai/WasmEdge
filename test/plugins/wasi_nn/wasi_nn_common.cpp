// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasi_nn_test_utils.h"
#include "wasinn_backend.h"
#include "wasinn_dispatch.h"
#include "wasinn_memory.h"
#include "wasinn_metadata.h"
#include "wasinn_output.h"
#include "wasinnenv.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <limits>
#include <optional>
#include <string_view>
#include <thread>
#include <vector>

using WasmEdge::Span;
using WasmEdge::Host::WASINN::Backend;
using WasmEdge::Host::WASINN::BackendPreloadMode;
using WasmEdge::Host::WASINN::Device;
using WasmEdge::Host::WASINN::ErrNo;
using WasmEdge::Host::WASINN::TensorType;
using namespace std::literals;

TEST(WasiNNCommonTest, BackendDescriptors) {
  constexpr std::array<Backend, 14> Backends{
      Backend::OpenVINO,      Backend::ONNX,           Backend::Tensorflow,
      Backend::PyTorch,       Backend::TensorflowLite, Backend::Autodetect,
      Backend::GGML,          Backend::NeuralSpeed,    Backend::Whisper,
      Backend::MLX,           Backend::Piper,          Backend::ChatTTS,
      Backend::OpenVINOGenAI, Backend::BitNet};
  for (const auto BackendType : Backends) {
    SCOPED_TRACE(static_cast<uint32_t>(BackendType));
    const auto *Descriptor =
        WasmEdge::Host::WASINN::getBackendDescriptor(BackendType);
    ASSERT_NE(Descriptor, nullptr);
    EXPECT_EQ(Descriptor->Type, BackendType);
    EXPECT_FALSE(Descriptor->Name.empty());
    ASSERT_NE(Descriptor->Operations, nullptr);
    EXPECT_EQ(Descriptor->Operations->Type, BackendType);
    EXPECT_EQ(WasmEdge::Host::WASINN::getBackendOperations(BackendType),
              Descriptor->Operations);
  }

  const auto *GGML =
      WasmEdge::Host::WASINN::getBackendDescriptor(Backend::GGML);
  ASSERT_NE(GGML, nullptr);
  EXPECT_EQ(GGML->Type, Backend::GGML);
  EXPECT_EQ(GGML->Preload, BackendPreloadMode::Path);
  EXPECT_EQ(WasmEdge::Host::WASINN::getBackendName(Backend::GGML), "ggml"sv);
  EXPECT_EQ(WasmEdge::Host::WASINN::getBackendBuildOption(Backend::BitNet),
            "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"BitNet\""sv);

  const auto *Selection =
      WasmEdge::Host::WASINN::getBackendSelection("pytorchaoti");
  ASSERT_NE(Selection, nullptr);
  EXPECT_EQ(Selection->Type, Backend::PyTorch);
  EXPECT_EQ(Selection->Preload, BackendPreloadMode::Path);

  EXPECT_EQ(
      WasmEdge::Host::WASINN::getBackendOperations(static_cast<Backend>(255)),
      nullptr);
}

TEST(WasiNNCommonTest, UnavailableBackendReportsInvalidArgument) {
  auto *Ops = WasmEdge::Host::WASINN::getBackendOperations(Backend::ONNX);
  ASSERT_NE(Ops, nullptr);

  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  std::vector<Span<uint8_t>> Builders;
  uint32_t GraphId = 0;
  auto Res = Ops->Load(Env, Builders, Device::CPU, GraphId);
  ASSERT_TRUE(Res);
  EXPECT_EQ(Res.value(), ErrNo::InvalidArgument);
}

TEST(WasiNNCommonTest, DeviceAutoIsAcceptedByHostDecoder) {
  auto DeviceValue =
      WasmEdge::Host::getDevice(static_cast<uint32_t>(Device::AUTO));

  ASSERT_TRUE(DeviceValue);
  EXPECT_EQ(*DeviceValue, Device::AUTO);
}

TEST(WasiNNCommonTest, GraphGuardRollsBackUncommittedGraph) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  uint32_t GraphId = 0;
  {
    auto Graph = Env.newGraphGuard(Backend::ONNX);
    GraphId = Graph.id();
    EXPECT_TRUE(Env.hasGraph(GraphId));
  }

  EXPECT_FALSE(Env.hasGraph(GraphId));
}

TEST(WasiNNCommonTest, GraphAndContextLifecycle) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  auto Graph = Env.newGraphGuard(Backend::ONNX);
  const auto GraphId = Graph.commit();
  EXPECT_TRUE(Env.hasGraph(GraphId));
  EXPECT_TRUE(Env.isGraphReady(GraphId));

  auto Context = Env.newContextGuard(GraphId);
  const auto ContextId = Context.commit();
  EXPECT_TRUE(Env.isContextReady(ContextId));
  EXPECT_TRUE(Env.isContextGraphReady(ContextId));

  Env.deleteGraph(GraphId);
  EXPECT_TRUE(Env.hasGraph(GraphId));
  EXPECT_TRUE(Env.isGraphFinalized(GraphId));

  Env.deleteContext(ContextId);
  EXPECT_FALSE(Env.hasContext(ContextId));
  EXPECT_FALSE(Env.hasGraph(GraphId));
}

TEST(WasiNNCommonTest, GraphAndContextDeleteAreIdempotent) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  auto Graph = Env.newGraphGuard(Backend::ONNX);
  const auto GraphId = Graph.commit();
  auto Context = Env.newContextGuard(GraphId);
  const auto ContextId = Context.commit();

  Env.deleteContext(ContextId);
  Env.deleteContext(ContextId);
  EXPECT_FALSE(Env.hasContext(ContextId));
  EXPECT_TRUE(Env.hasGraph(GraphId));

  Env.deleteGraph(GraphId);
  Env.deleteGraph(GraphId);
  EXPECT_FALSE(Env.hasGraph(GraphId));
}

TEST(WasiNNCommonTest, ContextGuardRejectsUnknownGraph) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  auto Context = Env.newContextGuard(42);

  EXPECT_FALSE(Context.valid());
  EXPECT_EQ(Context.id(), WasmEdge::Host::WASINN::GraphStore::InvalidId);
  EXPECT_EQ(Context.commit(), WasmEdge::Host::WASINN::GraphStore::InvalidId);
}

TEST(WasiNNCommonTest, ContextGuardRejectsFinalizedGraph) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  auto Graph = Env.newGraphGuard(Backend::ONNX);
  const auto GraphId = Graph.commit();
  auto ExistingContext = Env.newContextGuard(GraphId);
  const auto ExistingContextId = ExistingContext.commit();

  Env.deleteGraph(GraphId);
  EXPECT_TRUE(Env.hasGraph(GraphId));
  EXPECT_TRUE(Env.isGraphFinalized(GraphId));

  auto RejectedContext = Env.newContextGuard(GraphId);
  EXPECT_FALSE(RejectedContext.valid());
  EXPECT_EQ(RejectedContext.id(),
            WasmEdge::Host::WASINN::GraphStore::InvalidId);

  Env.deleteContext(ExistingContextId);
  EXPECT_FALSE(Env.hasGraph(GraphId));
}

TEST(WasiNNCommonTest, RecycledNonTailGraphIsNotAddressable) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  auto FirstGraph = Env.newGraphGuard(Backend::ONNX);
  const auto FirstGraphId = FirstGraph.commit();
  auto SecondGraph = Env.newGraphGuard(Backend::ONNX);
  const auto SecondGraphId = SecondGraph.commit();

  Env.deleteGraph(FirstGraphId);
  EXPECT_FALSE(Env.hasGraph(FirstGraphId));
  EXPECT_TRUE(Env.hasGraph(SecondGraphId));

  auto ReusedGraph = Env.newGraphGuard(Backend::GGML);
  EXPECT_EQ(ReusedGraph.id(), FirstGraphId);
  const auto ReusedGraphId = ReusedGraph.commit();
  EXPECT_TRUE(Env.hasGraph(ReusedGraphId));
  auto GraphBackend = Env.getGraphBackend(ReusedGraphId);
  ASSERT_TRUE(GraphBackend);
  EXPECT_EQ(*GraphBackend, Backend::GGML);
}

TEST(WasiNNCommonTest, RecycledNonTailContextIsNotAddressable) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  auto Graph = Env.newGraphGuard(Backend::ONNX);
  const auto GraphId = Graph.commit();
  auto FirstContext = Env.newContextGuard(GraphId);
  const auto FirstContextId = FirstContext.commit();
  auto SecondContext = Env.newContextGuard(GraphId);
  const auto SecondContextId = SecondContext.commit();

  Env.deleteContext(FirstContextId);
  EXPECT_FALSE(Env.hasContext(FirstContextId));
  EXPECT_TRUE(Env.hasContext(SecondContextId));

  auto ReusedContext = Env.newContextGuard(GraphId);
  EXPECT_EQ(ReusedContext.id(), FirstContextId);
  const auto ReusedContextId = ReusedContext.commit();
  EXPECT_TRUE(Env.hasContext(ReusedContextId));
  auto ContextGraphId = Env.getContextGraphId(ReusedContextId);
  ASSERT_TRUE(ContextGraphId);
  EXPECT_EQ(*ContextGraphId, GraphId);
}

TEST(WasiNNCommonTest, GraphDeleteWaitsForActiveContext) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  auto FirstGraph = Env.newGraphGuard(Backend::ONNX);
  const auto FirstGraphId = FirstGraph.commit();
  auto SecondGraph = Env.newGraphGuard(Backend::ONNX);
  const auto SecondGraphId = SecondGraph.commit();
  auto Context = Env.newContextGuard(FirstGraphId);
  const auto ContextId = Context.commit();

  Env.deleteGraph(FirstGraphId);
  EXPECT_TRUE(Env.hasGraph(FirstGraphId));
  EXPECT_TRUE(Env.isGraphFinalized(FirstGraphId));
  EXPECT_TRUE(Env.isContextReady(ContextId));
  EXPECT_TRUE(Env.hasGraph(SecondGraphId));

  auto Res = WasmEdge::Host::WASINN::Dispatch::compute(Env, ContextId);
  ASSERT_TRUE(Res);
  EXPECT_EQ(Res.value(), ErrNo::InvalidArgument);

  Env.deleteContext(ContextId);
  EXPECT_FALSE(Env.hasContext(ContextId));
  EXPECT_FALSE(Env.hasGraph(FirstGraphId));
  EXPECT_TRUE(Env.hasGraph(SecondGraphId));
}

TEST(WasiNNCommonTest, DispatchRejectsStaleContextId) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  auto Graph = Env.newGraphGuard(Backend::ONNX);
  const auto GraphId = Graph.commit();
  auto Context = Env.newContextGuard(GraphId);
  const auto ContextId = Context.commit();
  Env.deleteContext(ContextId);

  auto Res = WasmEdge::Host::WASINN::Dispatch::compute(Env, ContextId);
  ASSERT_TRUE(Res);
  EXPECT_EQ(Res.value(), ErrNo::InvalidArgument);
}

TEST(WasiNNCommonTest, PreloadCacheRemovesGraphId) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  Env.registerPreloadModelForTesting("model",
                                     {{uint8_t{1}, uint8_t{2}, uint8_t{3}}},
                                     Backend::ONNX, Device::CPU);

  uint32_t GraphId = 0;
  auto Res = Env.mdBuild(
      "model", GraphId,
      [](WasmEdge::Host::WASINN::WasiNNEnvironment &LoadEnv,
         Span<const Span<uint8_t>> Builders, Backend Encoding, Device Target,
         uint32_t &LoadedGraphId) -> WasmEdge::Expect<ErrNo> {
        EXPECT_EQ(Builders.size(), 1U);
        EXPECT_EQ(Encoding, Backend::ONNX);
        EXPECT_EQ(Target, Device::CPU);
        auto Graph = LoadEnv.newGraphGuard(Encoding);
        LoadedGraphId = Graph.commit();
        return ErrNo::Success;
      });
  ASSERT_TRUE(Res);
  EXPECT_EQ(Res.value(), ErrNo::Success);

  uint32_t CachedGraphId = 0;
  EXPECT_TRUE(Env.mdGet("model", CachedGraphId));
  EXPECT_EQ(CachedGraphId, GraphId);

  Env.mdRemoveById(GraphId);
  EXPECT_FALSE(Env.mdGet("model", CachedGraphId));
  Env.deleteGraph(GraphId);
}

TEST(WasiNNCommonTest, DispatchUnloadKeepsPreloadCacheOnFailure) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  Env.registerPreloadModelForTesting("model",
                                     {{uint8_t{1}, uint8_t{2}, uint8_t{3}}},
                                     Backend::ONNX, Device::CPU);

  uint32_t GraphId = 0;
  auto BuildRes =
      Env.mdBuild("model", GraphId,
                  [](WasmEdge::Host::WASINN::WasiNNEnvironment &LoadEnv,
                     Span<const Span<uint8_t>>, Backend Encoding, Device,
                     uint32_t &LoadedGraphId) -> WasmEdge::Expect<ErrNo> {
                    auto Graph = LoadEnv.newGraphGuard(Encoding);
                    LoadedGraphId = Graph.commit();
                    return ErrNo::Success;
                  });
  ASSERT_TRUE(BuildRes);
  EXPECT_EQ(BuildRes.value(), ErrNo::Success);

  uint32_t CachedGraphId = 0;
  EXPECT_TRUE(Env.mdGet("model", CachedGraphId));
  EXPECT_EQ(CachedGraphId, GraphId);

  auto UnloadRes = WasmEdge::Host::WASINN::Dispatch::unload(Env, GraphId);
  ASSERT_TRUE(UnloadRes);
  EXPECT_EQ(UnloadRes.value(), ErrNo::InvalidArgument);
  EXPECT_TRUE(Env.mdGet("model", CachedGraphId));
  EXPECT_EQ(CachedGraphId, GraphId);

  Env.mdRemoveById(GraphId);
  Env.deleteGraph(GraphId);
}

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
TEST(WasiNNCommonTest, DispatchUnloadRemovesPreloadCacheOnSuccess) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  Env.registerPreloadModelForTesting("model",
                                     {{uint8_t{1}, uint8_t{2}, uint8_t{3}}},
                                     Backend::GGML, Device::CPU);

  uint32_t GraphId = 0;
  auto BuildRes =
      Env.mdBuild("model", GraphId,
                  [](WasmEdge::Host::WASINN::WasiNNEnvironment &LoadEnv,
                     Span<const Span<uint8_t>>, Backend Encoding, Device,
                     uint32_t &LoadedGraphId) -> WasmEdge::Expect<ErrNo> {
                    auto Graph = LoadEnv.newGraphGuard(Encoding);
                    LoadedGraphId = Graph.commit();
                    return ErrNo::Success;
                  });
  ASSERT_TRUE(BuildRes);
  EXPECT_EQ(BuildRes.value(), ErrNo::Success);

  uint32_t CachedGraphId = 0;
  EXPECT_TRUE(Env.mdGet("model", CachedGraphId));
  EXPECT_EQ(CachedGraphId, GraphId);

  auto UnloadRes = WasmEdge::Host::WASINN::Dispatch::unload(Env, GraphId);
  ASSERT_TRUE(UnloadRes);
  EXPECT_EQ(UnloadRes.value(), ErrNo::Success);
  EXPECT_FALSE(Env.mdGet("model", CachedGraphId));
  EXPECT_FALSE(Env.hasGraph(GraphId));
}
#endif

TEST(WasiNNCommonTest, PreloadBuildIsSharedAcrossConcurrentCallers) {
  WasmEdge::Host::WASINN::WasiNNEnvironment Env;
  Env.registerPreloadModelForTesting("model",
                                     {{uint8_t{1}, uint8_t{2}, uint8_t{3}}},
                                     Backend::ONNX, Device::CPU);

  std::atomic<uint32_t> LoadCount = 0;
  std::atomic<bool> ReleaseLoad = false;
  uint32_t FirstGraphId = 0;
  uint32_t SecondGraphId = 0;
  std::optional<ErrNo> FirstResult;
  std::optional<ErrNo> SecondResult;
  auto Load = [&](WasmEdge::Host::WASINN::WasiNNEnvironment &LoadEnv,
                  Span<const Span<uint8_t>> Builders, Backend Encoding,
                  Device Target,
                  uint32_t &LoadedGraphId) -> WasmEdge::Expect<ErrNo> {
    EXPECT_EQ(Builders.size(), 1U);
    EXPECT_EQ(Encoding, Backend::ONNX);
    EXPECT_EQ(Target, Device::CPU);
    LoadCount.fetch_add(1, std::memory_order_acq_rel);
    while (!ReleaseLoad.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    auto Graph = LoadEnv.newGraphGuard(Encoding);
    LoadedGraphId = Graph.commit();
    return ErrNo::Success;
  };

  std::thread FirstThread([&]() {
    auto Res = Env.mdBuild("model", FirstGraphId, Load);
    if (Res) {
      FirstResult = Res.value();
    }
  });

  while (LoadCount.load(std::memory_order_acquire) == 0) {
    std::this_thread::yield();
  }

  std::thread SecondThread([&]() {
    auto Res = Env.mdBuild("model", SecondGraphId, Load);
    if (Res) {
      SecondResult = Res.value();
    }
  });

  ReleaseLoad.store(true, std::memory_order_release);
  FirstThread.join();
  SecondThread.join();

  ASSERT_TRUE(FirstResult);
  ASSERT_TRUE(SecondResult);
  EXPECT_EQ(*FirstResult, ErrNo::Success);
  EXPECT_EQ(*SecondResult, ErrNo::Success);
  EXPECT_EQ(FirstGraphId, SecondGraphId);
  EXPECT_EQ(LoadCount.load(std::memory_order_acquire), 1U);

  Env.deleteGraph(FirstGraphId);
}

TEST(WasiNNCommonTest, CopyBytesReportsRequiredByteCount) {
  const std::array<uint8_t, 3> Source{uint8_t{1}, uint8_t{2}, uint8_t{3}};
  std::array<uint8_t, 2> Output{};
  uint32_t BytesWritten = 0;

  const auto Res = WasmEdge::Host::WASINN::copyBytesToBuffer(
      {Source.data(), Source.size()}, {Output.data(), Output.size()},
      BytesWritten);

  EXPECT_EQ(Res, ErrNo::Success);
  EXPECT_EQ(Output[0], Source[0]);
  EXPECT_EQ(Output[1], Source[1]);
  EXPECT_EQ(BytesWritten, Source.size());
}

TEST(WasiNNCommonTest, CopyBytesRejectsRequiredByteCountOverflow) {
  uint32_t BytesWritten = 7;
  auto Res = WasmEdge::Host::WASINN::copyBytesToBuffer(
      {static_cast<const uint8_t *>(nullptr),
       static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1U},
      {}, BytesWritten);

  EXPECT_EQ(Res, ErrNo::TooLarge);
  EXPECT_EQ(BytesWritten, 0U);
}

TEST(WasiNNCommonTest, OutputBufferAcceptsZeroLengthBuffer) {
  WasmEdge::Runtime::Instance::MemoryInstance MemInst(
      WasmEdge::AST::MemoryType(1));
  const uint32_t BytesWrittenPtr = 16;

  auto Output = WasmEdge::Host::getOutputBuffer(MemInst, UINT32_C(0xFFFFFFFF),
                                                0, BytesWrittenPtr);

  ASSERT_TRUE(Output);
  EXPECT_EQ(Output->Buffer.size(), 0U);
  EXPECT_EQ(Output->BytesWrittenPtr, BytesWrittenPtr);
}

TEST(WasiNNCommonTest, OutputBufferRejectsInvalidBytesWrittenPointer) {
  WasmEdge::Runtime::Instance::MemoryInstance MemInst(
      WasmEdge::AST::MemoryType(1));

  auto Output =
      WasmEdge::Host::getOutputBuffer(MemInst, 0, 0, UINT32_C(0xFFFFFFFF));

  EXPECT_FALSE(Output);
}

TEST(WasiNNCommonTest, GraphBuilderDecoderRejectsInvalidBuilderSpan) {
  WasmEdge::Runtime::Instance::MemoryInstance MemInst(
      WasmEdge::AST::MemoryType(1));
  uint32_t BuilderEntryPtr = 0;
  WasmEdge::Host::WASINN::Testing::writeFatPointer(
      MemInst, UINT32_C(0xFFFFFFFF), 16, BuilderEntryPtr);

  auto Builders = WasmEdge::Host::getGraphBuilders(MemInst, 0, 1);

  EXPECT_FALSE(Builders);
}

TEST(WasiNNCommonTest, UInt32ResultWriterStoresLittleEndianValue) {
  WasmEdge::Runtime::Instance::MemoryInstance MemInst(
      WasmEdge::AST::MemoryType(1));
  const uint32_t Ptr = 16;

  auto Res =
      WasmEdge::Host::writeUInt32Result(MemInst, Ptr, 0x01020304U, "test"sv);
  ASSERT_TRUE(Res);

  auto Bytes = MemInst.getSpan<uint8_t>(Ptr, sizeof(uint32_t));
  ASSERT_EQ(Bytes.size(), sizeof(uint32_t));
  EXPECT_EQ(Bytes[0], uint8_t{0x04});
  EXPECT_EQ(Bytes[1], uint8_t{0x03});
  EXPECT_EQ(Bytes[2], uint8_t{0x02});
  EXPECT_EQ(Bytes[3], uint8_t{0x01});
}

TEST(WasiNNCommonTest, MetadataParsesCommaSeparatedIntegers) {
  std::vector<int> Values;
  auto Res = WasmEdge::Host::WASINN::parseCommaSeparatedIntegers(
      "1,2, 3"sv, Values, "value"sv, "test"sv);

  EXPECT_EQ(Res, ErrNo::Success);
  ASSERT_EQ(Values.size(), 3U);
  EXPECT_EQ(Values[0], 1);
  EXPECT_EQ(Values[1], 2);
  EXPECT_EQ(Values[2], 3);
}

TEST(WasiNNCommonTest, MetadataRejectsInvalidCommaSeparatedIntegers) {
  std::vector<int> Values;
  auto Res = WasmEdge::Host::WASINN::parseCommaSeparatedIntegers(
      "1,bad"sv, Values, "value"sv, "test"sv);

  EXPECT_EQ(Res, ErrNo::InvalidArgument);
}

TEST(WasiNNCommonTest, MetadataParsesTensorSplit) {
  std::array<float, 4> TensorSplit{};
  auto Res = WasmEdge::Host::WASINN::parseTensorSplit(
      "3,2"sv, TensorSplit.data(), TensorSplit.size(), 3, "test"sv);

  EXPECT_EQ(Res, ErrNo::Success);
  EXPECT_FLOAT_EQ(TensorSplit[0], 3.0f);
  EXPECT_FLOAT_EQ(TensorSplit[1], 2.0f);
  EXPECT_FLOAT_EQ(TensorSplit[2], 0.0f);
  EXPECT_FLOAT_EQ(TensorSplit[3], 0.0f);
}

TEST(WasiNNCommonTest, MetadataRejectsInvalidTensorSplit) {
  std::array<float, 2> TensorSplit{};

  auto TooMany = WasmEdge::Host::WASINN::parseTensorSplit(
      "1,2,3"sv, TensorSplit.data(), TensorSplit.size(), 2, "test"sv);
  EXPECT_EQ(TooMany, ErrNo::InvalidArgument);

  auto Negative = WasmEdge::Host::WASINN::parseTensorSplit(
      "1,-2"sv, TensorSplit.data(), TensorSplit.size(), 2, "test"sv);
  EXPECT_EQ(Negative, ErrNo::InvalidArgument);

  auto Invalid = WasmEdge::Host::WASINN::parseTensorSplit(
      "1,bad"sv, TensorSplit.data(), TensorSplit.size(), 2, "test"sv);
  EXPECT_EQ(Invalid, ErrNo::InvalidArgument);
}

TEST(WasiNNCommonTest, TensorDecoderAcceptsAllKnownTensorTypes) {
  WasmEdge::Runtime::Instance::MemoryInstance MemInst(
      WasmEdge::AST::MemoryType(1));
  const std::vector<uint32_t> TensorDim{1};
  const std::vector<uint8_t> TensorData{uint8_t{0}};
  const uint32_t TensorPtr = 0;
  const uint32_t DataPtr = 64;
  WasmEdge::Host::WASINN::Testing::MemoryWriter Writer(MemInst, DataPtr);
  std::array<TensorType, 6> TensorTypes{TensorType::F16, TensorType::F32,
                                        TensorType::F64, TensorType::U8,
                                        TensorType::I32, TensorType::I64};

  for (const auto RType : TensorTypes) {
    uint32_t BuilderPtr = TensorPtr;
    Writer.reset(DataPtr);
    Writer.writeTensor(TensorPtr, BuilderPtr, TensorDim, RType, TensorData);

    auto Tensor = WasmEdge::Host::getTensor(MemInst, TensorPtr);
    ASSERT_TRUE(Tensor);
    EXPECT_EQ(Tensor->Data.RType, RType);
  }
}

TEST(WasiNNCommonTest, TensorDecoderRejectsUnknownTensorType) {
  WasmEdge::Runtime::Instance::MemoryInstance MemInst(
      WasmEdge::AST::MemoryType(1));
  const std::vector<uint32_t> TensorDim{1};
  const std::vector<uint8_t> TensorData{uint8_t{0}};
  const uint32_t TensorPtr = 0;
  const uint32_t DataPtr = 64;
  uint32_t NextTensorPtr = TensorPtr;

  WasmEdge::Host::WASINN::Testing::writeFatPointer(
      MemInst, DataPtr, static_cast<uint32_t>(TensorDim.size()), NextTensorPtr);
  WasmEdge::Host::WASINN::Testing::writeUInt32(MemInst, UINT32_C(255),
                                               NextTensorPtr);
  WasmEdge::Host::WASINN::Testing::writeFatPointer(
      MemInst, DataPtr + static_cast<uint32_t>(sizeof(uint32_t)),
      static_cast<uint32_t>(TensorData.size()), NextTensorPtr);
  WasmEdge::Host::WASINN::Testing::writeBinaries<uint32_t>(MemInst, TensorDim,
                                                           DataPtr);
  WasmEdge::Host::WASINN::Testing::writeBinaries<uint8_t>(
      MemInst, TensorData, DataPtr + static_cast<uint32_t>(sizeof(uint32_t)));

  auto Tensor = WasmEdge::Host::getTensor(MemInst, TensorPtr);

  EXPECT_FALSE(Tensor);
}
