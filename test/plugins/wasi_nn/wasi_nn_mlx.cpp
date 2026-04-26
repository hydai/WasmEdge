// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasi_nn_test_utils.h"

using namespace WasmEdge::Host::WASINN::Testing;

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_MLX
TEST(WasiNNTest, MLXBackend) {
  WasiNNTestContext Ctx(60000);
  ASSERT_TRUE(Ctx.isValid());
  auto &MemInst = Ctx.memory();
  auto &CallFrame = Ctx.frame();

  // Load the files.
  std::string Prompt = "How are you?";
  std::string Tokenizer = "./wasinn_mlx_fixtures/tokenizer.json";
  std::vector<uint8_t> TensorData(Prompt.begin(), Prompt.end());
  std::vector<uint8_t> WeightRead =
      readEntireFile("./wasinn_mlx_fixtures/model.safetensors");

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

  // MLX WASI-NN load tests.
  // Test: load -- meaningless binaries.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(1), static_cast<uint32_t>(Backend::MLX),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: load -- graph id ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(1), static_cast<uint32_t>(Backend::MLX),
            static_cast<uint32_t>(Device::CPU), OutBoundPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- graph builder ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            OutBoundPtr, UINT32_C(1), static_cast<uint32_t>(Backend::MLX),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- MLX model bin ptr out of bounds.
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, OutBoundPtr,
                  static_cast<uint32_t>(WeightRead.size()), BuilderPtr);
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(1), static_cast<uint32_t>(Backend::MLX),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: load -- load successfully.
  std::string Config =
      "{\"model_type\":\"tiny_llama_1.1B_chat_v1.0\", "
      "\"tokenizer\":\"" +
      Tokenizer +
      "\", \"q_bits\": 4, \"group_size\": 128, \"is_quantized\": false}";
  std::vector<uint8_t> ConfigData(Config.begin(), Config.end());
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, StorePtr, WeightRead.size(), BuilderPtr);
  writeFatPointer(MemInst, StorePtr + WeightRead.size(), ConfigData.size(),
                  BuilderPtr);
  writeBinaries<uint8_t>(MemInst, WeightRead, StorePtr);
  writeBinaries<uint8_t>(MemInst, ConfigData, StorePtr + WeightRead.size());
  StorePtr += WeightRead.size() + ConfigData.size();
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(2), static_cast<uint32_t>(Backend::MLX),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // MLX WASI-NN init_execution_context tests.
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

  // MLX WASI-NN set_input tests.
  SetInputEntryPtr = BuilderPtr;
  writeFatPointer(MemInst, StorePtr, TensorDim.size(), BuilderPtr);
  writeUInt32(MemInst, UINT32_C(1), BuilderPtr);
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

  // MLX WASI-NN compute tests.
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

  // MLX WASI-NN get_output tests.
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
}
#endif // WASMEDGE_PLUGIN_WASI_NN_BACKEND_MLX
