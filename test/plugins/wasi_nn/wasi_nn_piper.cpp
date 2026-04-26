// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasi_nn_test_utils.h"

using namespace WasmEdge::Host::WASINN::Testing;

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_PIPER
TEST(WasiNNTest, PiperBackend) {
  WasiNNTestContext Ctx(400);
  ASSERT_TRUE(Ctx.isValid());
  auto &MemInst = Ctx.memory();
  auto &CallFrame = Ctx.frame();

  // Load the files.
  std::string Text = "This is a test.";
  std::vector<uint8_t> TensorData(Text.begin(), Text.end());

  std::vector<uint32_t> TensorDim{1};
  uint32_t BuilderPtr = UINT32_C(0);
  uint32_t LoadEntryPtr = UINT32_C(0);
  uint32_t SetInputEntryPtr = UINT32_C(0);
  uint32_t OutBoundPtr = UINT32_C(410 * 65536);
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

  // Piper WASI-NN load tests.
  // Test: load -- graph id ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(CallFrame,
                                 std::initializer_list<WasmEdge::ValVariant>{
                                     LoadEntryPtr, UINT32_C(1),
                                     static_cast<uint32_t>(Backend::Piper),
                                     UINT32_C(0), OutBoundPtr},
                                 Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- graph builder ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(CallFrame,
                                 std::initializer_list<WasmEdge::ValVariant>{
                                     OutBoundPtr, UINT32_C(1),
                                     static_cast<uint32_t>(Backend::Piper),
                                     UINT32_C(0), BuilderPtr},
                                 Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- Piper config ptr out of bounds.
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, OutBoundPtr, 1, BuilderPtr);
  {
    EXPECT_TRUE(HostFuncLoad.run(CallFrame,
                                 std::initializer_list<WasmEdge::ValVariant>{
                                     LoadEntryPtr, UINT32_C(1),
                                     static_cast<uint32_t>(Backend::Piper),
                                     UINT32_C(0), BuilderPtr},
                                 Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- wrong config encoding.
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, StorePtr, 0, BuilderPtr);
  {
    EXPECT_TRUE(HostFuncLoad.run(CallFrame,
                                 std::initializer_list<WasmEdge::ValVariant>{
                                     LoadEntryPtr, UINT32_C(1),
                                     static_cast<uint32_t>(Backend::Piper),
                                     UINT32_C(0), BuilderPtr},
                                 Errno));
    expectErrNo(Errno, ErrNo::InvalidEncoding);
  }

  // Test: load -- load successfully.
  std::string Config =
      "{\"model\": \"./wasinn_piper_fixtures/test_voice.onnx\", "
      "\"espeak_data\": \"./wasinn_piper_fixtures/piper/espeak-ng-data\"}";
  std::vector<uint8_t> ConfigData(Config.begin(), Config.end());
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, StorePtr, ConfigData.size(), BuilderPtr);
  writeBinaries<uint8_t>(MemInst, ConfigData, StorePtr);
  {
    EXPECT_TRUE(HostFuncLoad.run(CallFrame,
                                 std::initializer_list<WasmEdge::ValVariant>{
                                     LoadEntryPtr, UINT32_C(1),
                                     static_cast<uint32_t>(Backend::Piper),
                                     UINT32_C(0), BuilderPtr},
                                 Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // Piper WASI-NN init_execution_context tests.
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

  // Piper WASI-NN set_input tests.
  SetInputEntryPtr = BuilderPtr;
  writeFatPointer(MemInst, StorePtr, TensorDim.size(), BuilderPtr);
  writeUInt32(MemInst, 3, BuilderPtr);
  writeFatPointer(MemInst,
                  StorePtr + TensorDim.size() *
                                 sizeof(decltype(TensorDim)::value_type),
                  TensorData.size(), BuilderPtr);
  writeBinaries<uint32_t>(MemInst, TensorDim, StorePtr);
  writeBinaries<uint8_t>(
      MemInst, TensorData,
      StorePtr + TensorDim.size() * sizeof(decltype(TensorDim)::value_type));

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
  StorePtr += TensorDim.size() * sizeof(decltype(TensorDim)::value_type) +
              TensorData.size();

  // Piper WASI-NN compute tests.
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

  // Piper WASI-NN get_output tests.
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
    // Should output more than 10000 bytes.
    auto BytesWritten = *MemInst.getPointer<uint32_t *>(BuilderPtr);
    EXPECT_GE(BytesWritten, 10000);
  }

  // Piper json_input tests.
  // Test: load -- load successfully.
  Config =
      "{\"model\": "
      "\"./wasinn_piper_fixtures/test_voice.onnx\",\"espeak_data\": "
      "\"./wasinn_piper_fixtures/piper/espeak-ng-data\",\"json_input\":true}";
  ConfigData = {Config.begin(), Config.end()};
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, StorePtr, ConfigData.size(), BuilderPtr);
  writeBinaries<uint8_t>(MemInst, ConfigData, StorePtr);
  {
    EXPECT_TRUE(HostFuncLoad.run(CallFrame,
                                 std::initializer_list<WasmEdge::ValVariant>{
                                     LoadEntryPtr, UINT32_C(1),
                                     static_cast<uint32_t>(Backend::Piper),
                                     UINT32_C(0), BuilderPtr},
                                 Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 1);
    BuilderPtr += 4;
  }

  // Test: init_execution_context -- init context successfully.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(1), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 1);
    BuilderPtr += 4;
  }

  // First json input with parameters overridden
  Text = "{\"text\": \"This is a test.\", \"noise_scale\": 0.0, "
         "\"length_scale\": 2.0, \"noise_w\": 0.0}";
  TensorData = {Text.begin(), Text.end()};
  SetInputEntryPtr = BuilderPtr;
  writeFatPointer(MemInst, StorePtr, TensorDim.size(), BuilderPtr);
  writeUInt32(MemInst, 3, BuilderPtr);
  writeFatPointer(MemInst,
                  StorePtr + TensorDim.size() *
                                 sizeof(decltype(TensorDim)::value_type),
                  TensorData.size(), BuilderPtr);
  writeBinaries<uint32_t>(MemInst, TensorDim, StorePtr);
  writeBinaries<uint8_t>(
      MemInst, TensorData,
      StorePtr + TensorDim.size() * sizeof(decltype(TensorDim)::value_type));

  // Test: set_input -- set input successfully.
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(1), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::Success);
  }
  StorePtr += TensorDim.size() * sizeof(decltype(TensorDim)::value_type) +
              TensorData.size();

  // Test: compute -- compute successfully.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(1)},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
  }

  // Test: get_output -- get output successfully.
  {
    EXPECT_TRUE(HostFuncGetOutput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(1), UINT32_C(0), StorePtr, 65532, BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    auto BytesWritten = *MemInst.getPointer<uint32_t *>(BuilderPtr);
    // Should output more than 40000 bytes.
    EXPECT_GE(BytesWritten, 40000);
  }

  // Second json input to check if one-time overriding is working properly
  Text = "{\"text\": \"This is a test.\", \"output_type\": \"raw\", "
         "\"noise_scale\": 0.0, \"noise_w\": 0.0}";
  TensorData = {Text.begin(), Text.end()};
  SetInputEntryPtr = BuilderPtr;
  writeFatPointer(MemInst, StorePtr, TensorDim.size(), BuilderPtr);
  writeUInt32(MemInst, 3, BuilderPtr);
  writeFatPointer(MemInst,
                  StorePtr + TensorDim.size() *
                                 sizeof(decltype(TensorDim)::value_type),
                  TensorData.size(), BuilderPtr);
  writeBinaries<uint32_t>(MemInst, TensorDim, StorePtr);
  writeBinaries<uint8_t>(
      MemInst, TensorData,
      StorePtr + TensorDim.size() * sizeof(decltype(TensorDim)::value_type));

  // Test: set_input -- set input successfully.
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(1), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::Success);
  }
  StorePtr += TensorDim.size() * sizeof(decltype(TensorDim)::value_type) +
              TensorData.size();

  // Test: compute -- compute successfully.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(1)},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
  }

  // Test: get_output -- get output successfully.
  {
    EXPECT_TRUE(HostFuncGetOutput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(1), UINT32_C(0), StorePtr, 65532, BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    auto BytesWritten = *MemInst.getPointer<uint32_t *>(BuilderPtr);
    EXPECT_GE(BytesWritten, 30000);
    // Should output less than 50000 bytes.
    EXPECT_LT(BytesWritten, 50000);
    EXPECT_EQ(BytesWritten, 44100);
  }
}
#endif // WASMEDGE_PLUGIN_WASI_NN_BACKEND_PIPER
