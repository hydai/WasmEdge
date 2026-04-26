// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasi_nn_test_utils.h"

using namespace WasmEdge::Host::WASINN::Testing;

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
TEST(WasiNNTest, GGMLBackend) {
  WasiNNTestContext Ctx(60000);
  ASSERT_TRUE(Ctx.isValid());
  auto &MemInst = Ctx.memory();
  auto &CallFrame = Ctx.frame();

  // Load the files.
  std::string Prompt = "Once upon a time, ";
  std::vector<uint8_t> TensorData(Prompt.begin(), Prompt.end());
  std::string Model = WasmEdge::Endian::native == WasmEdge::Endian::little
                          ? "./wasinn_ggml_fixtures/orca_mini.gguf"
                          : "./wasinn_ggml_fixtures/granite-3.gguf";
  std::vector<uint8_t> WeightRead = readEntireFile(Model);

  std::vector<uint32_t> TensorDim{1};
  uint32_t BuilderPtr = UINT32_C(0);
  uint32_t LoadEntryPtr = UINT32_C(0);
  uint32_t SetInputEntryPtr = UINT32_C(0);
  uint32_t OutBoundPtr = UINT32_C(61000 * 65536);
  uint32_t StorePtr = UINT32_C(65536);
  MemoryWriter Store(MemInst, StorePtr);

  // Return value.
  std::array<WasmEdge::ValVariant, 1> Errno = {UINT32_C(0)};

  auto &HostFuncLoad = Ctx.hostFunc<WasmEdge::Host::WasiNNLoad>("load");
  auto &HostFuncInit =
      Ctx.hostFunc<WasmEdge::Host::WasiNNInitExecCtx>("init_execution_context");
  auto &HostFuncSetInput =
      Ctx.hostFunc<WasmEdge::Host::WasiNNSetInput>("set_input");
  auto &HostFuncGetOutput =
      Ctx.hostFunc<WasmEdge::Host::WasiNNGetOutput>("get_output");
  auto &HostFuncCompute =
      Ctx.hostFunc<WasmEdge::Host::WasiNNCompute>("compute");

  // GGML WASI-NN load tests.
  // Test: load -- meaningless binaries.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(1), static_cast<uint32_t>(Backend::GGML),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- graph id ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(1), static_cast<uint32_t>(Backend::GGML),
            static_cast<uint32_t>(Device::CPU), OutBoundPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- graph builder ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            OutBoundPtr, UINT32_C(1), static_cast<uint32_t>(Backend::GGML),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- GGML model bin ptr out of bounds.
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, OutBoundPtr,
                  static_cast<uint32_t>(WeightRead.size()), BuilderPtr);
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(1), static_cast<uint32_t>(Backend::GGML),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- wrong metadata encoding when builders length > 1.
  BuilderPtr = LoadEntryPtr;
  Store.writeGraphBuilderEntry(BuilderPtr, WeightRead);
  StorePtr = Store.ptr();
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(2), static_cast<uint32_t>(Backend::GGML),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidEncoding);
  }

  // Test: load -- load successfully.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(1), static_cast<uint32_t>(Backend::GGML),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // GGML WASI-NN init_execution_context tests.
  // Test: init_execution_context -- graph id invalid.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(2), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: init_execution_context -- init context successfully.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // GGML WASI-NN set_input tests.
  SetInputEntryPtr = BuilderPtr;
  Store.writeTensor(SetInputEntryPtr, BuilderPtr, TensorDim, TensorType::F32,
                    TensorData);
  StorePtr = Store.ptr();

  // Test: set_input -- context id exceeds.
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(3), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: set_input -- set input successfully.
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(0), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::Success);
  }

  const std::string Metadata = R"({"n-predict":8})";
  const std::vector<uint8_t> MetadataData(Metadata.begin(), Metadata.end());
  SetInputEntryPtr = BuilderPtr;
  Store.writeTensor(SetInputEntryPtr, BuilderPtr, TensorDim, TensorType::U8,
                    MetadataData);
  StorePtr = Store.ptr();
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(0), UINT32_C(1), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::Success);
  }

  // GGML WASI-NN compute tests.
  // Test: compute -- context id exceeds.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(3)},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: compute -- compute until finish or context full.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0)},
        Errno));
    EXPECT_TRUE(
        Errno[0].get<int32_t>() == static_cast<uint32_t>(ErrNo::Success) ||
        Errno[0].get<int32_t>() == static_cast<uint32_t>(ErrNo::ContextFull));
  }

  // GGML WASI-NN get_output tests.
  // Test: get_output -- output bytes ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncGetOutput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(0), UINT32_C(0), StorePtr, 65532, OutBoundPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: get_output -- output buffer ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncGetOutput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(0), UINT32_C(0), OutBoundPtr, 65532, BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: get_output -- get output successfully.
  {
    EXPECT_TRUE(HostFuncGetOutput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(0), UINT32_C(0), StorePtr, 65532, BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    auto BytesWritten = *MemInst.getPointer<uint32_t *>(BuilderPtr);
    EXPECT_GT(BytesWritten, 0U);
  }
}
#ifdef WASMEDGE_BUILD_WASI_NN_RPC
TEST(WasiNNTest, GGMLBackendWithRPC) {
  // wasi_nn_rpcserver has to be started outside this test,
  // and the URI has to be set to $WASI_NN_RPC_TEST_URI.
  // nn-preload has to be specified for "default".
  /*
    DIR=/tmp/build
    export WASI_NN_RPC_TEST_URI=unix://${DIR}/wasi_nn_rpc.sock
    export WASMEDGE_PLUGIN_PATH=${DIR}/plugins/wasi_nn
    ${DIR}/tools/wasmedge/wasi_nn_rpcserver \
      --nn-rpc-uri=$WASI_NN_RPC_TEST_URI \
      --nn-preload=default:GGML:AUTO:${DIR}/test/plugins/wasi_nn/wasinn_ggml_fixtures/orca_mini.gguf
  */
  const auto NNRPCURI = ::getenv("WASI_NN_RPC_TEST_URI");
  if (NNRPCURI == nullptr) {
    GTEST_SKIP() << "WASI_NN_RPC_TEST_URI is unset";
  }

  WasiNNTestContext Ctx(60000, NNRPCURI);
  ASSERT_TRUE(Ctx.isValid());
  auto &MemInst = Ctx.memory();
  auto &CallFrame = Ctx.frame();

  std::string Prompt = "Once upon a time, ";
  std::vector<uint8_t> TensorData(Prompt.begin(), Prompt.end());

  std::vector<uint32_t> TensorDim{1};
  uint32_t BuilderPtr = UINT32_C(0);
  uint32_t LoadEntryPtr = UINT32_C(0);
  uint32_t SetInputEntryPtr = UINT32_C(0);
  uint32_t OutBoundPtr = UINT32_C(61000) * UINT32_C(65536);
  uint32_t StorePtr = UINT32_C(65536);
  MemoryWriter Store(MemInst, StorePtr);

  // Return value.
  std::array<WasmEdge::ValVariant, 1> Errno = {UINT32_C(0)};

  auto &HostFuncLoadByName =
      Ctx.hostFunc<WasmEdge::Host::WasiNNLoadByName>("load_by_name");
  auto &HostFuncLoadByNameWithConfig =
      Ctx.hostFunc<WasmEdge::Host::WasiNNLoadByNameWithConfig>(
          "load_by_name_with_config");
  auto &HostFuncInit =
      Ctx.hostFunc<WasmEdge::Host::WasiNNInitExecCtx>("init_execution_context");
  auto &HostFuncSetInput =
      Ctx.hostFunc<WasmEdge::Host::WasiNNSetInput>("set_input");
  auto &HostFuncGetOutput =
      Ctx.hostFunc<WasmEdge::Host::WasiNNGetOutput>("get_output");
  auto &HostFuncCompute =
      Ctx.hostFunc<WasmEdge::Host::WasiNNCompute>("compute");

  // Test: load_by_name -- load successfully.
  {
    std::string Name = "default";
    std::vector<char> NameVec(Name.begin(), Name.end());
    writeBinaries<char>(MemInst, NameVec, LoadEntryPtr);
    EXPECT_TRUE(HostFuncLoadByName.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, static_cast<uint32_t>(NameVec.size()), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // Test: load_by_name_with_config -- load successfully.
  {
    std::string Name = "default";
    std::string Config = "{}";
    std::vector<char> NameVec(Name.begin(), Name.end());
    std::vector<char> ConfigVec(Config.begin(), Config.end());
    uint32_t ConfigPtr = LoadEntryPtr + NameVec.size();
    writeBinaries<char>(MemInst, NameVec, LoadEntryPtr);
    writeBinaries<char>(MemInst, ConfigVec, ConfigPtr);
    EXPECT_TRUE(HostFuncLoadByNameWithConfig.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, static_cast<uint32_t>(NameVec.size()), ConfigPtr,
            static_cast<uint32_t>(ConfigVec.size()), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // GGML WASI-NN init_execution_context tests.
  // Test: init_execution_context -- graph id invalid.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(2), BuilderPtr},
        Errno));
    EXPECT_NE(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
  }

  // Test: init_execution_context -- init context successfully.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // GGML WASI-NN set_input tests.
  SetInputEntryPtr = BuilderPtr;
  Store.writeTensor(SetInputEntryPtr, BuilderPtr, TensorDim, TensorType::F32,
                    TensorData);
  StorePtr = Store.ptr();

  // Test: set_input -- context id exceeds.
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(3), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    EXPECT_NE(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
  }

  // Test: set_input -- set input successfully.
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(0), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::Success);
  }

  // GGML WASI-NN compute tests.
  // Test: compute -- context id exceeds.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(3)},
        Errno));
    EXPECT_NE(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
  }

  // Test: compute -- compute until finish or context full.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0)},
        Errno));
    // FIXME: ErrNo propagation is not supported yet
    //    EXPECT_TRUE(
    //        Errno[0].get<int32_t>() == static_cast<uint32_t>(ErrNo::Success)
    //        || Errno[0].get<int32_t>() ==
    //        static_cast<uint32_t>(ErrNo::ContextFull));
  }

  // GGML WASI-NN get_output tests.
  // Test: get_output -- output bytes ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncGetOutput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(0), UINT32_C(0), StorePtr, 65532, OutBoundPtr},
        Errno));
    EXPECT_NE(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
  }

  // Test: get_output -- output buffer ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncGetOutput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(0), UINT32_C(0), OutBoundPtr, 65532, BuilderPtr},
        Errno));
    EXPECT_NE(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
  }

  // Test: get_output -- get output successfully.
  {
    EXPECT_TRUE(HostFuncGetOutput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(0), UINT32_C(0), StorePtr, 65532, BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    // Should output more than 50 bytes.
    auto BytesWritten = *MemInst.getPointer<uint32_t *>(BuilderPtr);
    EXPECT_GE(BytesWritten, 50);
  }
}

TEST(WasiNNTest, GGMLBackendComputeSingleWithRPC) {
  // wasi_nn_rpcserver has to be started outside this test,
  // and the URI has to be set to $WASI_NN_RPC_TEST_URI.
  // nn-preload has to be specified for "default".
  /*
    DIR=/tmp/build
    export WASI_NN_RPC_TEST_URI=unix://${DIR}/wasi_nn_rpc.sock
    export WASMEDGE_PLUGIN_PATH=${DIR}/plugins/wasi_nn
    ${DIR}/tools/wasmedge/wasi_nn_rpcserver \
      --nn-rpc-uri=$WASI_NN_RPC_TEST_URI \
      --nn-preload=default:GGML:AUTO:${DIR}/test/plugins/wasi_nn/wasinn_ggml_fixtures/orca_mini.gguf
  */
  const auto NNRPCURI = ::getenv("WASI_NN_RPC_TEST_URI");
  if (NNRPCURI == nullptr) {
    GTEST_SKIP() << "WASI_NN_RPC_TEST_URI is unset";
  }

  WasiNNTestContext Ctx(60000, NNRPCURI);
  ASSERT_TRUE(Ctx.isValid());
  auto &MemInst = Ctx.memory();
  auto &CallFrame = Ctx.frame();

  std::string Prompt = "Once upon a time, ";
  std::vector<uint8_t> TensorData(Prompt.begin(), Prompt.end());

  std::vector<uint32_t> TensorDim{1};
  uint32_t BuilderPtr = UINT32_C(0);
  uint32_t LoadEntryPtr = UINT32_C(0);
  uint32_t SetInputEntryPtr = UINT32_C(0);
  uint32_t OutBoundPtr = UINT32_C(61000) * UINT32_C(65536);
  uint32_t StorePtr = UINT32_C(65536);
  MemoryWriter Store(MemInst, StorePtr);

  // Return value.
  std::array<WasmEdge::ValVariant, 1> Errno = {UINT32_C(0)};

  auto &HostFuncLoadByName =
      Ctx.hostFunc<WasmEdge::Host::WasiNNLoadByName>("load_by_name");
  auto &HostFuncLoadByNameWithConfig =
      Ctx.hostFunc<WasmEdge::Host::WasiNNLoadByNameWithConfig>(
          "load_by_name_with_config");
  auto &HostFuncInit =
      Ctx.hostFunc<WasmEdge::Host::WasiNNInitExecCtx>("init_execution_context");
  auto &HostFuncSetInput =
      Ctx.hostFunc<WasmEdge::Host::WasiNNSetInput>("set_input");
  auto &HostFuncGetOutputSingle =
      Ctx.hostFunc<WasmEdge::Host::WasiNNGetOutputSingle>("get_output_single");
  auto &HostFuncComputeSingle =
      Ctx.hostFunc<WasmEdge::Host::WasiNNComputeSingle>("compute_single");

  // Test: load_by_name -- load successfully.
  {
    std::string Name = "default";
    std::vector<char> NameVec(Name.begin(), Name.end());
    writeBinaries<char>(MemInst, NameVec, LoadEntryPtr);
    EXPECT_TRUE(HostFuncLoadByName.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, static_cast<uint32_t>(NameVec.size()), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // Test: load_by_name_with_config -- load successfully.
  {
    std::string Name = "default";
    std::string Config = "{}";
    std::vector<char> NameVec(Name.begin(), Name.end());
    std::vector<char> ConfigVec(Config.begin(), Config.end());
    uint32_t ConfigPtr = LoadEntryPtr + NameVec.size();
    writeBinaries<char>(MemInst, NameVec, LoadEntryPtr);
    writeBinaries<char>(MemInst, ConfigVec, ConfigPtr);
    EXPECT_TRUE(HostFuncLoadByNameWithConfig.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, static_cast<uint32_t>(NameVec.size()), ConfigPtr,
            static_cast<uint32_t>(ConfigVec.size()), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // Test: init_execution_context -- init context successfully.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // GGML WASI-NN set_input tests.
  SetInputEntryPtr = BuilderPtr;
  Store.writeTensor(SetInputEntryPtr, BuilderPtr, TensorDim, TensorType::F32,
                    TensorData);
  StorePtr = Store.ptr();

  // Test: set_input -- set input successfully.
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(0), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::Success);
  }

  // GGML WASI-NN compute_single tests.
  // Test: compute_single -- context id exceeds.
  {
    EXPECT_TRUE(HostFuncComputeSingle.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(3)},
        Errno));
    EXPECT_NE(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
  }

  // Test: compute_single -- call compute_single once follow by a
  // get_output_single.
  {
    // compute_single
    EXPECT_TRUE(HostFuncComputeSingle.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0)},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    // get_output_single
    EXPECT_TRUE(HostFuncGetOutputSingle.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(0), UINT32_C(0), StorePtr, 65532, BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
  }

  // GGML WASI-NN get_output_single tests.
  // Test: get_output_single -- output bytes ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncGetOutputSingle.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(0), UINT32_C(0), StorePtr, 65532, OutBoundPtr},
        Errno));
    EXPECT_NE(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
  }

  // Test: get_output -- output buffer ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncGetOutputSingle.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(0), UINT32_C(0), OutBoundPtr, 65532, BuilderPtr},
        Errno));
    EXPECT_NE(Errno[0].get<int32_t>(), static_cast<uint32_t>(ErrNo::Success));
  }
}
#endif // WASMEDGE_BUILD_WASI_NN_RPC
#endif // WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
