// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_tfl.h"
#include "wasinn_output.h"
#include "wasinnenv.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_TFLITE
#include "tensorflow/lite/c/common.h"

#include <limits>
#endif

using namespace std::literals;

namespace WasmEdge::Host::WASINN::TensorflowLite {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_TFLITE
namespace {
struct TfLiteInterpreterOptionsDeleter {
  void operator()(TfLiteInterpreterOptions *Ptr) const noexcept {
    if (Ptr) {
      TfLiteInterpreterOptionsDelete(Ptr);
    }
  }
};

using TfLiteInterpreterOptionsPtr =
    std::unique_ptr<TfLiteInterpreterOptions, TfLiteInterpreterOptionsDeleter>;
} // namespace

Expect<WASINN::ErrNo> load(WASINN::WasiNNEnvironment &Env,
                           Span<const Span<uint8_t>> Builders,
                           WASINN::Device Device, uint32_t &GraphId) noexcept {
  if (auto Res = checkDevice(Device, Device::CPU, Backend::TensorflowLite);
      Res != WASINN::ErrNo::Success) {
    return Res;
  }
  if (auto Res = checkBuilderCount(Builders, 1, Backend::TensorflowLite);
      Res != WASINN::ErrNo::Success) {
    return Res;
  }
  // Add a new graph.
  auto Graph = Env.newGraphGuard(Backend::TensorflowLite);
  auto &GraphRef = Graph.get<Backend::TensorflowLite>();

  // Copy graph builder data to TfLiteModData and create a new TfLiteModel.
  GraphRef.TfLiteModData.assign(Builders[0].begin(), Builders[0].end());
  GraphRef.TFLiteMod.reset(TfLiteModelCreate(GraphRef.TfLiteModData.data(),
                                             GraphRef.TfLiteModData.size()));
  if (unlikely(GraphRef.TFLiteMod == nullptr)) {
    spdlog::error("[WASI-NN] Cannot import TFLite model"sv);
    return WASINN::ErrNo::InvalidArgument;
  }

  // Store the loaded graph.
  GraphId = Graph.commit();
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> initExecCtx(WASINN::WasiNNEnvironment &Env,
                                  uint32_t GraphId,
                                  uint32_t &ContextId) noexcept {
  // Check the network and the execution network with the graph ID.
  auto GraphInst = Env.getBackendGraphOrError<Backend::TensorflowLite>(
      GraphId, "init_execution_context"sv);
  if (!GraphInst) {
    return GraphInst.error();
  }
  auto &GraphRef = **GraphInst;
  if (GraphRef.TFLiteMod == nullptr) {
    spdlog::error("[WASI-NN] Model for Graph:{} is missing!"sv, GraphId);
    return WASINN::ErrNo::MissingMemory;
  }

  // Create context.
  auto Context = Env.newContextGuard(GraphId);
  auto State = Env.getBackendContextGraphOrError<Backend::TensorflowLite>(
      Context.id(), "init_execution_context"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &ContextGraphRef = State->graph();
  TfLiteInterpreterOptionsPtr TFLiteOps(TfLiteInterpreterOptionsCreate());
  TfLiteInterpreterOptionsSetNumThreads(TFLiteOps.get(), 2);
  CxtRef.TFLiteInterp.reset(TfLiteInterpreterCreate(
      ContextGraphRef.TFLiteMod.get(), TFLiteOps.get()));
  if (unlikely(CxtRef.TFLiteInterp == nullptr)) {
    spdlog::error("[WASI-NN] Cannot create TFLite interpreter."sv);
    return WASINN::ErrNo::Busy;
  }
  TfLiteInterpreterAllocateTensors(CxtRef.TFLiteInterp.get());

  ContextId = Context.commit();
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> setInput(WASINN::WasiNNEnvironment &Env,
                               uint32_t ContextId, uint32_t Index,
                               const WASINN::TensorData &Tensor) noexcept {
  auto CxtInst = Env.getBackendContextOrError<Backend::TensorflowLite>(
      ContextId, "set_input"sv);
  if (!CxtInst) {
    return CxtInst.error();
  }
  auto &CxtRef = **CxtInst;
  uint32_t InCnt =
      TfLiteInterpreterGetInputTensorCount(CxtRef.TFLiteInterp.get());
  if (Index >= InCnt) {
    spdlog::error("[WASI-NN] Invalid index id {} for the input, only {} "
                  "inputs are allowed",
                  Index, InCnt);
    return WASINN::ErrNo::InvalidArgument;
  }

  auto *HoldTensor =
      TfLiteInterpreterGetInputTensor(CxtRef.TFLiteInterp.get(), Index);
  // Check the input data size.
  const auto HoldTensorByteSize = TfLiteTensorByteSize(HoldTensor);
  if (HoldTensorByteSize != Tensor.Tensor.size()) {
    spdlog::error("[WASI-NN] Expect tensor byte size {}, but got {}"sv,
                  HoldTensorByteSize, Tensor.Tensor.size());
    return WASINN::ErrNo::InvalidArgument;
  }
  // Check the input tensor dimensions.
  const auto HoldTensorNumDims = TfLiteTensorNumDims(HoldTensor);
  if (static_cast<size_t>(HoldTensorNumDims) != Tensor.Dimension.size()) {
    spdlog::error(
        "[WASI-NN] Expect tensor number of dimensions {}, but got {}"sv,
        HoldTensorNumDims, Tensor.Dimension.size());
    return WASINN::ErrNo::InvalidArgument;
  }
  for (uint32_t I = 0; I < Tensor.Dimension.size(); I++) {
    const auto HoldTensorDim = TfLiteTensorDim(HoldTensor, I);
    if (static_cast<uint32_t>(HoldTensorDim) != Tensor.Dimension[I]) {
      spdlog::error("[WASI-NN] Expect tensor dimension[{}] = {}, but got {}"sv,
                    I, HoldTensorDim, Tensor.Dimension[I]);
      return WASINN::ErrNo::InvalidArgument;
    }
  }
  // Check the input tensor type.
  WASINN::TensorType LiteType;
  switch (const auto Type = TfLiteTensorType(HoldTensor)) {
  case TfLiteType::kTfLiteUInt8:
    LiteType = WASINN::TensorType::U8;
    break;
  case TfLiteType::kTfLiteFloat16:
    LiteType = WASINN::TensorType::F16;
    break;
  case TfLiteType::kTfLiteFloat32:
    LiteType = WASINN::TensorType::F32;
    break;
  case TfLiteType::kTfLiteInt32:
    LiteType = WASINN::TensorType::I32;
    break;
  default:
    spdlog::error("[WASI-NN] Unsupported TFLite type: {}"sv,
                  TfLiteTypeGetName(Type));
    return WASINN::ErrNo::InvalidArgument;
  }

  if (unlikely(LiteType != Tensor.RType)) {
    spdlog::error("[WASI-NN] Expect tensor type {}, but got {}"sv, LiteType,
                  Tensor.RType);
    return WASINN::ErrNo::InvalidArgument;
  }
  TfLiteStatus Stat = TfLiteTensorCopyFromBuffer(
      HoldTensor, Tensor.Tensor.data(), Tensor.Tensor.size());
  if (unlikely(Stat != TfLiteStatus::kTfLiteOk)) {
    spdlog::error("[WASI-NN] Copy tensor memory failed"sv);
    return WASINN::ErrNo::Busy;
  }

  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> getOutput(WASINN::WasiNNEnvironment &Env,
                                uint32_t ContextId, uint32_t Index,
                                Span<uint8_t> OutBuffer,
                                uint32_t &BytesWritten) noexcept {
  auto CxtInst = Env.getBackendContextOrError<Backend::TensorflowLite>(
      ContextId, "get_output"sv);
  if (!CxtInst) {
    return CxtInst.error();
  }
  auto &CxtRef = **CxtInst;
  uint32_t OutCnt =
      TfLiteInterpreterGetOutputTensorCount(CxtRef.TFLiteInterp.get());
  if (Index >= OutCnt) {
    spdlog::error("[WASI-NN] Invalid index id {} for the input, only {} "
                  "outputs are allowed"sv,
                  Index, OutCnt);
    return WASINN::ErrNo::InvalidArgument;
  }
  const TfLiteTensor *HoldTensor =
      TfLiteInterpreterGetOutputTensor(CxtRef.TFLiteInterp.get(), Index);
  const size_t TensorByteSize = TfLiteTensorByteSize(HoldTensor);
  if (TensorByteSize > std::numeric_limits<uint32_t>::max()) {
    spdlog::error(
        "[WASI-NN] TensorFlow Lite backend: Output size {} is greater than std::numeric_limits<uint32_t>::max() {}."sv,
        TensorByteSize, std::numeric_limits<uint32_t>::max());
    return WASINN::ErrNo::InvalidArgument;
  }
  std::vector<uint8_t> Output(TensorByteSize);
  const TfLiteStatus Stat =
      TfLiteTensorCopyToBuffer(HoldTensor, Output.data(), TensorByteSize);
  if (unlikely(Stat != TfLiteStatus::kTfLiteOk)) {
    spdlog::error("[WASI-NN] Copy tensor memory failed"sv);
    return WASINN::ErrNo::Busy;
  }
  return copyBytesToBuffer(Output, OutBuffer, BytesWritten);
}

Expect<WASINN::ErrNo> compute(WASINN::WasiNNEnvironment &Env,
                              uint32_t ContextId) noexcept {
  auto CxtInst = Env.getBackendContextOrError<Backend::TensorflowLite>(
      ContextId, "compute"sv);
  if (!CxtInst) {
    return CxtInst.error();
  }
  auto &CxtRef = **CxtInst;
  // Run session
  if (unlikely(CxtRef.TFLiteInterp == nullptr)) {
    spdlog::error("[WASI-NN] Tensorflow Lite context empty"sv);
    return WASINN::ErrNo::MissingMemory;
  }
  TfLiteStatus Stat = TfLiteInterpreterInvoke(CxtRef.TFLiteInterp.get());
  if (unlikely(Stat != TfLiteStatus::kTfLiteOk)) {
    spdlog::error("[WASI-NN] Invocation failed."sv);
    return WASINN::ErrNo::Busy;
  }
  return WASINN::ErrNo::Success;
}
#endif
} // namespace WasmEdge::Host::WASINN::TensorflowLite
