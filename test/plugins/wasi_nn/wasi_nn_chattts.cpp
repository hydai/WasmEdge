// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasi_nn_test_utils.h"

using namespace WasmEdge::Host::WASINN::Testing;

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_CHATTTS
TEST(WasiNNTest, ChatTTSBackend) {
  WasiNNTestContext Ctx(60000);
  ASSERT_TRUE(Ctx.isValid());
  auto &MemInst = Ctx.memory();
  auto &CallFrame = Ctx.frame();

  // Load the files.
  std::string Prompt = "This is test prompt.";
  std::vector<uint8_t> TensorData(Prompt.begin(), Prompt.end());
  std::string config =
      "{\"prompt\":\"[oral_2][laugh_0][break_6]\",\"spk_emb\":\"random\","
      "\"temperature\":0.5,\"top_k\":0,\"top_p\":0.9}";
  std::vector<uint8_t> ConfigData(config.begin(), config.end());

  std::vector<uint32_t> TensorDim{1};
  uint32_t BuilderPtr = UINT32_C(0);
  uint32_t LoadEntryPtr = UINT32_C(0);
  uint32_t SetInputEntryPtr = UINT32_C(0);
  uint32_t OutBoundPtr = UINT32_C(61000) * UINT32_C(65536);
  uint32_t StorePtr = UINT32_C(65536);

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
  auto &HostFuncUnload = Ctx.hostFunc<WasmEdge::Host::WasiNNUnload>("unload");

  // ChatTTS WASI-NN load tests.
  // Test: load -- graph id ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(CallFrame,
                                 std::initializer_list<WasmEdge::ValVariant>{
                                     LoadEntryPtr, UINT32_C(1),
                                     static_cast<uint32_t>(Backend::ChatTTS),
                                     UINT32_C(0), OutBoundPtr},
                                 Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- graph builder ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(CallFrame,
                                 std::initializer_list<WasmEdge::ValVariant>{
                                     OutBoundPtr, UINT32_C(1),
                                     static_cast<uint32_t>(Backend::ChatTTS),
                                     UINT32_C(0), BuilderPtr},
                                 Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- load successfully.
  BuilderPtr = LoadEntryPtr;
  {
    EXPECT_TRUE(HostFuncLoad.run(CallFrame,
                                 std::initializer_list<WasmEdge::ValVariant>{
                                     LoadEntryPtr, UINT32_C(1),
                                     static_cast<uint32_t>(Backend::ChatTTS),
                                     UINT32_C(0), BuilderPtr},
                                 Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }
  // ChatTTS WASI-NN init_execution_context tests.
  // Test: init_execution_context -- graph id invalid.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(2), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: init_execution_context -- init second context.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // ChatTTS WASI-NN set_input tests.
  SetInputEntryPtr = BuilderPtr;
  writeFatPointer(MemInst, StorePtr, TensorDim.size(), BuilderPtr);
  writeUInt32(MemInst, static_cast<uint32_t>(TensorType::U8), BuilderPtr);
  writeFatPointer(MemInst, StorePtr + TensorDim.size() * 4, TensorData.size(),
                  BuilderPtr);
  writeBinaries<uint32_t>(MemInst, TensorDim, StorePtr);
  writeBinaries<uint8_t>(MemInst, TensorData, StorePtr + TensorDim.size() * 4);

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
  StorePtr += (TensorDim.size() * 4 + TensorData.size());

  // ChatTTS WASI-NN compute tests.
  // Test: compute -- context id exceeds.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(3)},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: compute -- compute successfully.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0)},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
  }

  // Test: setInput -- set metadata successfully.
  SetInputEntryPtr = BuilderPtr;
  writeFatPointer(MemInst, StorePtr, TensorDim.size(), BuilderPtr);
  writeUInt32(MemInst, static_cast<uint32_t>(TensorType::U8), BuilderPtr);
  writeFatPointer(MemInst, StorePtr + TensorDim.size() * 4, ConfigData.size(),
                  BuilderPtr);
  writeBinaries<uint32_t>(MemInst, TensorDim, StorePtr);
  writeBinaries<uint8_t>(MemInst, ConfigData, StorePtr + TensorDim.size() * 4);
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(0), UINT32_C(1), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::Success);
  }
  StorePtr += (TensorDim.size() * 4 + ConfigData.size());

  // Test: compute -- compute successfully.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0)},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
  }

  // ChatTTS WASI-NN get_output tests.
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
    // Should output more than 50 bytes.
    auto BytesWritten = *MemInst.getPointer<uint32_t *>(BuilderPtr);
    EXPECT_GE(BytesWritten, 50);
  }

  // ChatTTS WASI-NN unload tests.
  // Test: unload -- unload successfully.
  {
    EXPECT_TRUE(HostFuncUnload.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0)},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
  }
}
#endif // WASMEDGE_PLUGIN_WASI_NN_BACKEND_CHATTTS
