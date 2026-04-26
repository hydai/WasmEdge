// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_openvino.h"
#include "wasinn_output.h"
#include "wasinnenv.h"

#include <algorithm>

using namespace std::literals;

namespace WasmEdge::Host::WASINN::OpenVINO {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_OPENVINO

namespace {

WASINN::ErrNo
checkModelReady(const std::shared_ptr<ov::Model> &Model) noexcept {
  if (Model == nullptr) {
    spdlog::error("[WASI-NN] OpenVINO backend: Model is empty."sv);
    return WASINN::ErrNo::MissingMemory;
  }
  return WASINN::ErrNo::Success;
}

} // namespace

Expect<WASINN::ErrNo> load(WASINN::WasiNNEnvironment &Env,
                           Span<const Span<uint8_t>> Builders,
                           WASINN::Device Device, uint32_t &GraphId) noexcept {
  if (auto Res = checkBuilderCount(Builders, 2, Backend::OpenVINO);
      Res != WASINN::ErrNo::Success) {
    return Res;
  }

  // Get the XML and Weight raw buffer.
  //   Builder-0: the XML string
  //   Builder-1: the Weight binary
  auto XML = Builders[0];
  auto Weight = Builders[1];

  // Add a new graph.
  auto Graph = Env.newGraphGuard(Backend::OpenVINO);
  auto &GraphRef = Graph.get<Backend::OpenVINO>();

  // Store device information
  GraphRef.TargetDevice = Device;

  try {
    auto ModelString = asString(XML);
    GraphRef.OpenVINOIWeightTensor =
        ov::Tensor(ov::element::Type_t::u8, {Weight.size()});
    std::copy_n(Weight.data(), Weight.size(),
                static_cast<uint8_t *>(GraphRef.OpenVINOIWeightTensor.data()));
    GraphRef.OpenVINOModel = Env.OpenVINOCore.read_model(
        ModelString, GraphRef.OpenVINOIWeightTensor);
  } catch (const std::exception &EX) {
    spdlog::error("[WASI-NN] Model Load Exception: {}"sv, EX.what());
    return WASINN::ErrNo::RuntimeError;
  }
  // Store the loaded graph.
  GraphId = Graph.commit();
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> initExecCtx(WASINN::WasiNNEnvironment &Env,
                                  uint32_t GraphId,
                                  uint32_t &ContextId) noexcept {
  // Check the network and the execution network with the graph ID.
  auto GraphInst = Env.getBackendGraphOrError<Backend::OpenVINO>(
      GraphId, "init_execution_context"sv);
  if (!GraphInst) {
    return GraphInst.error();
  }
  auto &GraphRef = **GraphInst;
  if (auto Res = checkModelReady(GraphRef.OpenVINOModel);
      Res != WASINN::ErrNo::Success) {
    return Res;
  }
  ContextId = Env.newReadyContext(GraphId);
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> setInput(WASINN::WasiNNEnvironment &Env,
                               uint32_t ContextId, uint32_t Index,
                               const TensorData &Tensor) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::OpenVINO>(
      ContextId, "set_input"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();

  if (auto Res = checkModelReady(GraphRef.OpenVINOModel);
      Res != WASINN::ErrNo::Success) {
    return Res;
  }

  if (Tensor.Dimension.size() > 8) {
    spdlog::error("[WASI-NN] Tensor dimension is out of range, expect it under "
                  "8-dim, but got {}-dim."sv,
                  Tensor.Dimension.size());
    return WASINN::ErrNo::InvalidArgument;
  }
  if (Tensor.RType != WASINN::TensorType::F32) {
    spdlog::error(
        "[WASI-NN] Only F32 inputs and outputs are supported for now."sv);
    return WASINN::ErrNo::InvalidArgument;
  }

  // Check the input index.
  if (GraphRef.OpenVINOModel->inputs().size() <= Index) {
    spdlog::error(
        "[WASI-NN] The input index {} exceeds the inputs number {}."sv, Index,
        GraphRef.OpenVINOModel->inputs().size());
    return WASINN::ErrNo::InvalidArgument;
  }

  try {
    ov::element::Type InputType = ov::element::f32;
    ov::Shape InputShape(Tensor.Dimension.data(),
                         Tensor.Dimension.data() + Tensor.Dimension.size());
    ov::Tensor InputTensor =
        ov::Tensor(InputType, InputShape, Tensor.Tensor.data());
    auto Device =
        getDeviceString(GraphRef.TargetDevice, Backend::OpenVINO, true);
    if (!Device) {
      return Device.error();
    }

    ov::CompiledModel CompiledModel =
        Env.OpenVINOCore.compile_model(GraphRef.OpenVINOModel, Device.value());
    CxtRef.OpenVINOInferRequest = CompiledModel.create_infer_request();
    CxtRef.OpenVINOInferRequest.set_input_tensor(Index, InputTensor);
  } catch (const std::exception &EX) {
    spdlog::error("[WASI-NN] Set Input Exception: {}"sv, EX.what());
    return WASINN::ErrNo::RuntimeError;
  }
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> getOutput(WASINN::WasiNNEnvironment &Env,
                                uint32_t ContextId, uint32_t Index,
                                Span<uint8_t> OutBuffer,
                                uint32_t &BytesWritten) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::OpenVINO>(
      ContextId, "get_output"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();

  if (auto Res = checkModelReady(GraphRef.OpenVINOModel);
      Res != WASINN::ErrNo::Success) {
    return Res;
  }

  // Check the output index.
  if (GraphRef.OpenVINOModel->outputs().size() <= Index) {
    spdlog::error(
        "[WASI-NN] The output index {} exceeds the outputs number {}."sv, Index,
        GraphRef.OpenVINOModel->outputs().size());
    return WASINN::ErrNo::InvalidArgument;
  }

  try {
    const ov::Tensor &OutputTensor =
        CxtRef.OpenVINOInferRequest.get_output_tensor(Index);
    if (auto Res = copyBytesToBuffer(
            {static_cast<const uint8_t *>(OutputTensor.data()),
             OutputTensor.get_byte_size()},
            OutBuffer, BytesWritten);
        Res != WASINN::ErrNo::Success) {
      return Res;
    }
  } catch (const std::exception &EX) {
    spdlog::error("[WASI-NN] Get Output Exception: {}"sv, EX.what());
    return WASINN::ErrNo::RuntimeError;
  }
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> compute(WASINN::WasiNNEnvironment &Env,
                              uint32_t ContextId) noexcept {
  auto CxtInst =
      Env.getBackendContextOrError<Backend::OpenVINO>(ContextId, "compute"sv);
  if (!CxtInst) {
    return CxtInst.error();
  }
  auto &CxtRef = **CxtInst;
  try {
    CxtRef.OpenVINOInferRequest.infer();
  } catch (const std::exception &EX) {
    spdlog::error("[WASI-NN] Infer Request Exception: {}"sv, EX.what());
    return WASINN::ErrNo::RuntimeError;
  }
  return WASINN::ErrNo::Success;
}
#endif
} // namespace WasmEdge::Host::WASINN::OpenVINO
