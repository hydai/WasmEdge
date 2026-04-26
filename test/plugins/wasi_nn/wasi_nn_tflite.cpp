// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasi_nn_test_utils.h"

using namespace WasmEdge::Host::WASINN::Testing;

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_TFLITE
TEST(WasiNNTest, TFLiteBackend) {
  WasiNNTestContext Ctx(400);
  ASSERT_TRUE(Ctx.isValid());
  auto *NNMod = &Ctx.module();
  auto &MemInst = Ctx.memory();
  auto &CallFrame = Ctx.frame();

  // Load the files.
  std::vector<uint8_t> TensorData =
      readEntireFile("./wasinn_tflite_fixtures/birdx224x224x3.rgb");
  std::vector<uint8_t> WeightRead =
      readEntireFile("./wasinn_tflite_fixtures/"
                     "lite-model_aiy_vision_classifier_birds_V1_3.tflite");
  spdlog::info("Read {}"sv, TensorData.size());
  std::vector<uint32_t> TensorDim{1, 224, 224, 3};
  uint32_t BuilderPtr = UINT32_C(0);
  uint32_t LoadEntryPtr = UINT32_C(0);
  uint32_t SetInputEntryPtr = UINT32_C(0);
  uint32_t OutBoundPtr = UINT32_C(410 * 65536);
  uint32_t StorePtr = UINT32_C(65536);

  // Return value.
  std::array<WasmEdge::ValVariant, 1> Errno = {UINT32_C(0)};

  // Temp. values.
  std::deque<WasmEdge::Host::WASINN::Graph> NNGraphTmp;
  std::deque<WasmEdge::Host::WASINN::Context> NNContextTmp;

  auto &HostFuncLoad = Ctx.hostFunc<WasmEdge::Host::WasiNNLoad>("load");
  auto &HostFuncInit =
      Ctx.hostFunc<WasmEdge::Host::WasiNNInitExecCtx>("init_execution_context");
  auto &HostFuncSetInput =
      Ctx.hostFunc<WasmEdge::Host::WasiNNSetInput>("set_input");
  auto &HostFuncGetOutput =
      Ctx.hostFunc<WasmEdge::Host::WasiNNGetOutput>("get_output");
  auto &HostFuncCompute =
      Ctx.hostFunc<WasmEdge::Host::WasiNNCompute>("compute");

  // Torch WASI-NN load tests.
  // Test: load -- meaningless binaries.
  {
    EXPECT_TRUE(
        HostFuncLoad.run(CallFrame,
                         std::initializer_list<WasmEdge::ValVariant>{
                             LoadEntryPtr, UINT32_C(1),
                             static_cast<uint32_t>(Backend::TensorflowLite),
                             static_cast<uint32_t>(Device::CPU), BuilderPtr},
                         Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- graph id ptr out of bounds.
  {
    EXPECT_TRUE(
        HostFuncLoad.run(CallFrame,
                         std::initializer_list<WasmEdge::ValVariant>{
                             LoadEntryPtr, UINT32_C(1),
                             static_cast<uint32_t>(Backend::TensorflowLite),
                             static_cast<uint32_t>(Device::CPU), OutBoundPtr},
                         Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- graph builder ptr out of bounds.
  {
    EXPECT_TRUE(
        HostFuncLoad.run(CallFrame,
                         std::initializer_list<WasmEdge::ValVariant>{
                             OutBoundPtr, UINT32_C(1),
                             static_cast<uint32_t>(Backend::TensorflowLite),
                             static_cast<uint32_t>(Device::CPU), BuilderPtr},
                         Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }
  // Test: load -- model bin ptr out of bounds.
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, OutBoundPtr,
                  static_cast<uint32_t>(WeightRead.size()), BuilderPtr);
  {
    EXPECT_TRUE(
        HostFuncLoad.run(CallFrame,
                         std::initializer_list<WasmEdge::ValVariant>{
                             LoadEntryPtr, UINT32_C(1),
                             static_cast<uint32_t>(Backend::TensorflowLite),
                             static_cast<uint32_t>(Device::CPU), BuilderPtr},
                         Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- wrong builders' length.
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, StorePtr, static_cast<uint32_t>(WeightRead.size()),
                  BuilderPtr);
  writeBinaries<uint8_t>(MemInst, WeightRead, StorePtr);
  StorePtr += WeightRead.size();
  {
    EXPECT_TRUE(
        HostFuncLoad.run(CallFrame,
                         std::initializer_list<WasmEdge::ValVariant>{
                             LoadEntryPtr, UINT32_C(2),
                             static_cast<uint32_t>(Backend::TensorflowLite),
                             static_cast<uint32_t>(Device::CPU), BuilderPtr},
                         Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- unsupported device. CPU 0, GPU 1, TPU 2
  {
    EXPECT_TRUE(
        HostFuncLoad.run(CallFrame,
                         std::initializer_list<WasmEdge::ValVariant>{
                             LoadEntryPtr, UINT32_C(1),
                             static_cast<uint32_t>(Backend::TensorflowLite),
                             static_cast<uint32_t>(Device::AUTO), BuilderPtr},
                         Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- load successfully.
  {
    EXPECT_TRUE(
        HostFuncLoad.run(CallFrame,
                         std::initializer_list<WasmEdge::ValVariant>{
                             LoadEntryPtr, UINT32_C(1),
                             static_cast<uint32_t>(Backend::TensorflowLite),
                             static_cast<uint32_t>(Device::CPU), BuilderPtr},
                         Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // Test: load -- load second graph.
  {
    EXPECT_TRUE(
        HostFuncLoad.run(CallFrame,
                         std::initializer_list<WasmEdge::ValVariant>{
                             LoadEntryPtr, UINT32_C(1),
                             static_cast<uint32_t>(Backend::TensorflowLite),
                             static_cast<uint32_t>(Device::CPU), BuilderPtr},
                         Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 1);
    BuilderPtr += 4;
  }

  // WASI-NN init_execution_context tests.
  // Test: init_execution_context -- graph id invalid.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(2), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Swap to the tmp. env.
  // Test: init_execution_context -- graph id exceeds.
  NNGraphTmp.emplace_back(Backend::TensorflowLite);
  NNGraphTmp.back().setReady();
  NNMod->getEnv().swapGraphContextForTesting(NNGraphTmp, NNContextTmp);
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::MissingMemory);
  }
  // Swap back.
  NNMod->getEnv().swapGraphContextForTesting(NNGraphTmp, NNContextTmp);

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

  // Test: init_execution_context -- init second context.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(1), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 1);
    BuilderPtr += 4;
  }

  // Torch WASI-NN set_input tests.
  SetInputEntryPtr = BuilderPtr;
  writeFatPointer(MemInst, StorePtr, static_cast<uint32_t>(TensorDim.size()),
                  BuilderPtr);
  writeUInt32(MemInst, UINT32_C(1), BuilderPtr);
  writeFatPointer(MemInst,
                  StorePtr + static_cast<uint32_t>(TensorDim.size()) * 4,
                  static_cast<uint32_t>(TensorData.size()), BuilderPtr);
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

  NNContextTmp.emplace_back(0, NNGraphTmp[0]);
  NNContextTmp.back().setReady();

  // Test: set_input -- set input successfully.
  BuilderPtr = SetInputEntryPtr;
  writeFatPointer(MemInst, StorePtr, static_cast<uint32_t>(TensorDim.size()),
                  BuilderPtr);
  // Tensor type U8
  writeUInt32(MemInst, UINT32_C(3), BuilderPtr);
  writeFatPointer(MemInst,
                  StorePtr + static_cast<uint32_t>(TensorDim.size()) * 4,
                  static_cast<uint32_t>(TensorData.size()), BuilderPtr);
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(1), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::Success);
  }
  StorePtr += (TensorDim.size() * 4 + TensorData.size());

  // Torch WASI-NN compute tests.
  // Test: compute -- context id exceeds.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(3)},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Swap to the tmp. env.
  NNMod->getEnv().swapGraphContextForTesting(NNGraphTmp, NNContextTmp);
  // Test: compute -- empty context.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0)},
        Errno));
    expectErrNo(Errno, ErrNo::MissingMemory);
  }
  // Swap back.
  NNMod->getEnv().swapGraphContextForTesting(NNGraphTmp, NNContextTmp);

  // Test: compute -- compute successfully.
  {
    EXPECT_TRUE(HostFuncCompute.run(
        CallFrame, std::initializer_list<WasmEdge::ValVariant>{UINT32_C(1)},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
  }
  // WASI-NN get_output tests.
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

  // Test: get_output -- output index exceeds.
  {
    EXPECT_TRUE(HostFuncGetOutput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(0), UINT32_C(10), StorePtr, 65532, BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: get_output -- get output successfully.
  {
    EXPECT_TRUE(HostFuncGetOutput.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            UINT32_C(1), UINT32_C(0), StorePtr, 65532, BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), UINT32_C(965));
    const auto OutputClassification =
        MemInst.getSpan<const uint8_t>(StorePtr, 965);
    std::vector<size_t> SortedIndex, CorrectClasses{166, 158, 34, 778, 819};
    // FIXME: classSort causing segmentation fault
    SortedIndex = classSort<uint8_t>(OutputClassification);

    // The probability of class i is placed at buffer[i].
    for (size_t I = 0; I < CorrectClasses.size(); I++) {
      EXPECT_EQ(OutputClassification[SortedIndex[I]],
                OutputClassification[CorrectClasses[I]]);
    }
  }
}
#endif // WASMEDGE_PLUGIN_WASI_NN_BACKEND_TFLITE
