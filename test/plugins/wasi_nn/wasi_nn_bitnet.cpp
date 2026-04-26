// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasi_nn_test_utils.h"

using namespace WasmEdge::Host::WASINN::Testing;

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET
TEST(WasiNNTest, BitNetBackend) {
  WasiNNTestContext Ctx(60000);
  ASSERT_TRUE(Ctx.isValid());
  auto &MemInst = Ctx.memory();
  auto &CallFrame = Ctx.frame();

  auto &HostFuncLoad = Ctx.hostFunc<WasmEdge::Host::WasiNNLoad>("load");
  auto &HostFuncInit =
      Ctx.hostFunc<WasmEdge::Host::WasiNNInitExecCtx>("init_execution_context");
  auto &HostFuncSetInput =
      Ctx.hostFunc<WasmEdge::Host::WasiNNSetInput>("set_input");
  auto &HostFuncGetOutput =
      Ctx.hostFunc<WasmEdge::Host::WasiNNGetOutput>("get_output");
  auto &HostFuncCompute =
      Ctx.hostFunc<WasmEdge::Host::WasiNNCompute>("compute");
  auto &HostFuncUnload = Ctx.hostFunc<WasmEdge::Host::WasiNNUnload>("unload");
  auto &HostFuncComputeSingle =
      Ctx.hostFunc<WasmEdge::Host::WasiNNComputeSingle>("compute_single");
  auto &HostFuncGetOutputSingle =
      Ctx.hostFunc<WasmEdge::Host::WasiNNGetOutputSingle>("get_output_single");
  auto &HostFuncFiniSingle =
      Ctx.hostFunc<WasmEdge::Host::WasiNNFiniSingle>("fini_single");

  // --- Test Data & Pointer Setup ---
  const std::string ModelPath = "./wasinn_bitnet_fixtures/ggml-model-i2_s.gguf";
  const std::string ModelPreloadStr = "preload:" + ModelPath;
  const std::string MetadataStr = R"({"n-predict": 128})";
  const std::string Prompt = "Once upon a time, ";

  uint32_t BuilderPtr = 0;
  uint32_t LoadEntryPtr = 0;
  uint32_t SetInputEntryPtr = 0;
  uint32_t StorePtr = 65536;
  const uint32_t OutBoundPtr = UINT32_C(61000) * UINT32_C(65536);
  std::array<WasmEdge::ValVariant, 1> Errno = {0};
  uint32_t GraphId = 0;
  uint32_t CtxId = 0;

  // BitNet WASI-NN load tests
  // Test: load -- empty builder array.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, 0, static_cast<uint32_t>(Backend::BitNet),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: load -- graph builder ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            OutBoundPtr, 1, static_cast<uint32_t>(Backend::BitNet),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: load -- model bin ptr out of bounds.
  {
    BuilderPtr = LoadEntryPtr;
    writeFatPointer(MemInst, OutBoundPtr, 10, BuilderPtr);
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, 1, static_cast<uint32_t>(Backend::BitNet),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: load -- invalid metadata encoding.
  {
    const std::string InvalidMetadataStr = R"({"n-predict": "not-a-number")";
    std::vector<uint8_t> ModelPreloadVec(ModelPreloadStr.begin(),
                                         ModelPreloadStr.end());
    std::vector<uint8_t> InvalidMetadataVec(InvalidMetadataStr.begin(),
                                            InvalidMetadataStr.end());
    BuilderPtr = LoadEntryPtr;
    writeFatPointer(MemInst, StorePtr, ModelPreloadVec.size(), BuilderPtr);
    writeFatPointer(MemInst, StorePtr + ModelPreloadVec.size(),
                    InvalidMetadataVec.size(), BuilderPtr);
    writeBinaries<uint8_t>(MemInst, ModelPreloadVec, StorePtr);
    writeBinaries<uint8_t>(MemInst, InvalidMetadataVec,
                           StorePtr + ModelPreloadVec.size());
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, 2, static_cast<uint32_t>(Backend::BitNet),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidEncoding);
  }
  // Test: load -- load successfully.
  {
    std::vector<uint8_t> ModelPreloadVec(ModelPreloadStr.begin(),
                                         ModelPreloadStr.end());
    std::vector<uint8_t> MetadataVec(MetadataStr.begin(), MetadataStr.end());
    BuilderPtr = LoadEntryPtr;
    writeFatPointer(MemInst, StorePtr, ModelPreloadVec.size(), BuilderPtr);
    writeFatPointer(MemInst, StorePtr + ModelPreloadVec.size(),
                    MetadataVec.size(), BuilderPtr);
    writeBinaries<uint8_t>(MemInst, ModelPreloadVec, StorePtr);
    writeBinaries<uint8_t>(MemInst, MetadataVec,
                           StorePtr + ModelPreloadVec.size());
    StorePtr += ModelPreloadVec.size() + MetadataVec.size();
    ASSERT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, 2, static_cast<uint32_t>(Backend::BitNet),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno))
        << "Load failed. Ensure model file exists at: " << ModelPath;
    ASSERT_EQ(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
    GraphId = *MemInst.getPointer<uint32_t *>(BuilderPtr);
    BuilderPtr += 4;
  }

  // BitNet WASI-NN init_execution_context tests
  // Test: init_execution_context -- graph id invalid.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{999, BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: init_execution_context -- init context successfully.
  {
    ASSERT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{GraphId, BuilderPtr},
        Errno));
    ASSERT_EQ(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
    CtxId = *MemInst.getPointer<uint32_t *>(BuilderPtr);
    BuilderPtr += 4;
  }

  // BitNet WASI-NN set_input tests
  SetInputEntryPtr = BuilderPtr;
  {
    std::vector<uint8_t> PromptData(Prompt.begin(), Prompt.end());
    std::vector<uint32_t> PromptDim = {
        static_cast<uint32_t>(PromptData.size())};

    writeFatPointer(MemInst, StorePtr, PromptDim.size(), BuilderPtr);
    writeUInt32(MemInst, static_cast<uint32_t>(TensorType::U8), BuilderPtr);
    writeFatPointer(MemInst, StorePtr + PromptDim.size() * 4, PromptData.size(),
                    BuilderPtr);
    writeBinaries<uint32_t>(MemInst, PromptDim, StorePtr);
    writeBinaries<uint8_t>(MemInst, PromptData,
                           StorePtr + PromptDim.size() * 4);
  }
  // Test: set_input -- invalid context id.
  {
    EXPECT_TRUE(HostFuncSetInput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{999, 0, SetInputEntryPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: set_input -- invalid tensor index.
  {
    EXPECT_TRUE(HostFuncSetInput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{CtxId, 2, SetInputEntryPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: set_input -- set input successfully.
  {
    ASSERT_TRUE(HostFuncSetInput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{CtxId, 0, SetInputEntryPtr},
        Errno));
    ASSERT_EQ(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
  }

  // BitNet WASI-NN compute and get_output tests
  // Test: compute -- invalid context ID.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{999}, Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: compute -- compute successfully.
  {
    ASSERT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{CtxId}, Errno));
    ASSERT_EQ(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
  }
  // Test: get_output -- output buffer pointer out of bounds.
  {
    EXPECT_TRUE(
        HostFuncGetOutput.run(CallFrame,
                              std::initializer_list<WasmEdge::ValVariant>{
                                  CtxId, 0, OutBoundPtr, 5, BuilderPtr},
                              Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: get_output -- bytes written pointer out of bounds.
  {
    EXPECT_TRUE(
        HostFuncGetOutput.run(CallFrame,
                              std::initializer_list<WasmEdge::ValVariant>{
                                  CtxId, 0, StorePtr, 5, OutBoundPtr},
                              Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: get_output -- get output successfully.
  {
    uint32_t BytesNeeded = 0;
    ASSERT_TRUE(
        HostFuncGetOutput.run(CallFrame,
                              std::initializer_list<WasmEdge::ValVariant>{
                                  CtxId, 0, StorePtr, 0, BuilderPtr},
                              Errno));
    ASSERT_EQ(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
    BytesNeeded = *MemInst.getPointer<uint32_t *>(BuilderPtr);
    EXPECT_GT(BytesNeeded, 10);
  }

  // BitNet compute_single tests
  {
    std::string FullStreamedOutput = "";
    const int MaxStreamTokens = 20;

    // Test: set_input -- set prompt to start a new streaming sequence.
    {
      ASSERT_TRUE(
          HostFuncSetInput.run(CallFrame,
                               std::initializer_list<WasmEdge::ValVariant>{
                                   CtxId, 0, SetInputEntryPtr},
                               Errno));
      ASSERT_EQ(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
    }

    // Test: compute_single and get_output_single in a loop.
    for (int i = 0; i < MaxStreamTokens; ++i) {
      ASSERT_TRUE(HostFuncComputeSingle.run(
          CallFrame, std::initializer_list<WasmEdge::ValVariant>{CtxId},
          Errno));
      if (Errno[0].get<int32_t>() ==
          static_cast<uint32_t>(ErrNo::EndOfSequence)) {
        break;
      }
      ASSERT_EQ(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));

      uint32_t SingleTokenBytes = 0;
      ASSERT_TRUE(HostFuncGetOutputSingle.run(
          CallFrame,
          std::initializer_list<WasmEdge::ValVariant>{CtxId, 0, StorePtr, 32,
                                                      BuilderPtr},
          Errno));
      ASSERT_EQ(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
      SingleTokenBytes = *MemInst.getPointer<uint32_t *>(BuilderPtr);
      if (SingleTokenBytes > 0) {
        auto TokenSpan = *MemInst.getBytes(StorePtr, SingleTokenBytes);
        FullStreamedOutput += std::string(
            reinterpret_cast<const char *>(TokenSpan.data()), TokenSpan.size());
      }
    }
    EXPECT_GT(FullStreamedOutput.length(), 10);

    // Test: fini_single -- finalize the streaming session.
    {
      ASSERT_TRUE(HostFuncFiniSingle.run(
          CallFrame, std::initializer_list<WasmEdge::ValVariant>{CtxId},
          Errno));
      ASSERT_EQ(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
    }
  }

  // BitNet WASI-NN unload tests.
  // Test: unload -- invalid graph id.
  {
    EXPECT_TRUE(HostFuncUnload.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{999}, Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: unload -- unload successfully and verify.
  {
    ASSERT_TRUE(HostFuncUnload.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{GraphId},
        Errno));
    ASSERT_EQ(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));

    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{GraphId, BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
}
#endif // WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET
