// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasi_nn_test_utils.h"

using namespace WasmEdge::Host::WASINN::Testing;

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_OPENVINO
TEST(WasiNNTest, OpenVINOBackend) {
  WasiNNTestContext Ctx(400);
  ASSERT_TRUE(Ctx.isValid());
  auto *NNMod = &Ctx.module();
  auto &MemInst = Ctx.memory();
  auto &CallFrame = Ctx.frame();

  // Load the files.
  std::vector<uint8_t> TensorDataLegecy =
      readEntireFile("./wasinn_openvino_fixtures/tensor-1x224x224x3-f32.bgr");
  std::vector<uint8_t> XmlRead =
      readEntireFile("./wasinn_openvino_fixtures/mobilenet.xml");
  std::vector<uint8_t> WeightRead =
      readEntireFile("./wasinn_openvino_fixtures/mobilenet.bin");

  // Convert the NHWC to NCHW format.
  // For historical reasons, the OpenVINO model expects the input tensor in
  // NCHW format, while the input tensor is in NHWC format.
  // https://github.com/intel/openvino-rs/blob/v0.3.3/crates/openvino/tests/fixtures/mobilenet/build.sh#L39
  // https://github.com/intel/openvino-rs/blob/v0.8.0/crates/openvino/tests/classify-mobilenet.rs#L34
  ASSERT_EQ(TensorDataLegecy.size(), 3 * 224 * 224 * 4);
  std::vector<uint8_t> TensorData(TensorDataLegecy.size());

  for (size_t C = 0; C < 3; ++C) {
    for (size_t H = 0; H < 224; ++H) {
      for (size_t W = 0; W < 224; ++W) {
        size_t Loc = H * 224 + W;
        for (size_t B = 0; B < 4; ++B) {
          TensorData[(C * 224 * 224 + Loc) * 4 + B] =
              TensorDataLegecy[(3 * Loc + C) * 4 + B];
        }
      }
    }
  }

  std::vector<uint32_t> TensorDim{1, 3, 224, 224};
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

  // OpenVINO WASI-NN load tests.
  // Test: load -- meaningless binaries.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(2), static_cast<uint32_t>(Backend::OpenVINO),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::RuntimeError);
  }

  // Test: load -- graph id ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(2), static_cast<uint32_t>(Backend::OpenVINO),
            static_cast<uint32_t>(Device::CPU), OutBoundPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- graph builder ptr out of bounds.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            OutBoundPtr, UINT32_C(2), static_cast<uint32_t>(Backend::OpenVINO),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- OpenVINO model xml ptr out of bounds.
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, OutBoundPtr, static_cast<uint32_t>(XmlRead.size()),
                  BuilderPtr);
  writeFatPointer(MemInst, StorePtr + static_cast<uint32_t>(XmlRead.size()),
                  static_cast<uint32_t>(WeightRead.size()), BuilderPtr);
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(2), static_cast<uint32_t>(Backend::OpenVINO),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- OpenVINO model bin ptr out of bounds.
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, StorePtr, static_cast<uint32_t>(XmlRead.size()),
                  BuilderPtr);
  writeFatPointer(MemInst, OutBoundPtr,
                  static_cast<uint32_t>(WeightRead.size()), BuilderPtr);
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(2), static_cast<uint32_t>(Backend::OpenVINO),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- wrong builders' length.
  BuilderPtr = LoadEntryPtr;
  writeFatPointer(MemInst, StorePtr, static_cast<uint32_t>(XmlRead.size()),
                  BuilderPtr);
  writeFatPointer(MemInst, StorePtr + static_cast<uint32_t>(XmlRead.size()),
                  static_cast<uint32_t>(WeightRead.size()), BuilderPtr);
  writeBinaries<uint8_t>(MemInst, XmlRead, StorePtr);
  writeBinaries<uint8_t>(MemInst, WeightRead, StorePtr + XmlRead.size());
  StorePtr += (XmlRead.size() + WeightRead.size());
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(4), static_cast<uint32_t>(Backend::OpenVINO),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- unsupported device. CPU 0, GPU 1, TPU 2
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(2), static_cast<uint32_t>(Backend::OpenVINO),
            static_cast<uint32_t>(Device::AUTO), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: load -- load successfully.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(2), static_cast<uint32_t>(Backend::OpenVINO),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 0);
    BuilderPtr += 4;
  }

  // Test: load -- load second graph.
  {
    EXPECT_TRUE(HostFuncLoad.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{
            LoadEntryPtr, UINT32_C(2), static_cast<uint32_t>(Backend::OpenVINO),
            static_cast<uint32_t>(Device::CPU), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::Success);
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), 1);
    BuilderPtr += 4;
  }

  // OpenVINO WASI-NN init_execution_context tests.
  // Test: init_execution_context -- graph id invalid.
  {
    EXPECT_TRUE(HostFuncInit.run(
        CallFrame,
        std::initializer_list<WasmEdge::ValVariant>{UINT32_C(2), BuilderPtr},
        Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Swap to the tmp. env.
  NNGraphTmp.emplace_back(Backend::OpenVINO);
  NNGraphTmp.back().setReady();
  NNMod->getEnv().swapGraphContextForTesting(NNGraphTmp, NNContextTmp);
  // Test: init_execution_context -- graph id exceeds.
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

  // OpenVINO WASI-NN set_input tests.
  SetInputEntryPtr = BuilderPtr;
  writeFatPointer(MemInst, StorePtr, static_cast<uint32_t>(TensorDim.size()),
                  BuilderPtr);
  writeUInt32(MemInst, UINT32_C(1), BuilderPtr);
  writeFatPointer(MemInst,
                  StorePtr + static_cast<uint32_t>(TensorDim.size()) * 4,
                  static_cast<uint32_t>(TensorData.size()), BuilderPtr);
  writeBinaries<uint32_t>(MemInst, TensorDim, StorePtr);
  writeBinaries<uint8_t>(MemInst, TensorData, StorePtr + TensorDim.size() * 4);

  // Swap to the tmp. env.
  NNContextTmp.emplace_back(0, NNGraphTmp[0]);
  NNContextTmp.back().setReady();
  NNMod->getEnv().swapGraphContextForTesting(NNGraphTmp, NNContextTmp);
  // Test: set_input -- context id exceeds.
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(3), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: set_input -- empty context.
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(0), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::MissingMemory);
  }
  // Swap back.
  NNMod->getEnv().swapGraphContextForTesting(NNGraphTmp, NNContextTmp);

  // Test: set_input -- input index exceeds.
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(0), UINT32_C(10), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: set_input -- tensor type not FP32.
  BuilderPtr = SetInputEntryPtr;
  writeFatPointer(MemInst, StorePtr, static_cast<uint32_t>(TensorDim.size()),
                  BuilderPtr);
  writeUInt32(MemInst, UINT32_C(2), BuilderPtr);
  writeFatPointer(MemInst,
                  StorePtr + static_cast<uint32_t>(TensorDim.size()) * 4,
                  static_cast<uint32_t>(TensorData.size()), BuilderPtr);
  {
    EXPECT_TRUE(
        HostFuncSetInput.run(CallFrame,
                             std::initializer_list<WasmEdge::ValVariant>{
                                 UINT32_C(0), UINT32_C(0), SetInputEntryPtr},
                             Errno));
    expectErrNo(Errno, ErrNo::InvalidArgument);
  }

  // Test: set_input -- set input successfully.
  BuilderPtr = SetInputEntryPtr;
  writeFatPointer(MemInst, StorePtr, static_cast<uint32_t>(TensorDim.size()),
                  BuilderPtr);
  writeUInt32(MemInst, UINT32_C(1), BuilderPtr);
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

  // OpenVINO WASI-NN compute tests.
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
    expectErrNo(Errno, ErrNo::RuntimeError);
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

  // OpenVINO WASI-NN get_output tests.
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
    EXPECT_EQ(*MemInst.getPointer<uint32_t *>(BuilderPtr), UINT32_C(4004));
    const auto OutputClassification =
        MemInst.getSpan<const float>(StorePtr, 1001).subspan(1);
    std::vector<size_t> SortedIndex, CorrectClasses{963, 762, 909, 926, 567};
    SortedIndex = classSort<float>(OutputClassification);
    // The probability of class i is placed at buffer[i].
    for (size_t I = 0; I < CorrectClasses.size(); I++) {
      EXPECT_EQ(SortedIndex[I], CorrectClasses[I]);
    }
  }
}
#endif // WASMEDGE_PLUGIN_WASI_NN_BACKEND_OPENVINO
