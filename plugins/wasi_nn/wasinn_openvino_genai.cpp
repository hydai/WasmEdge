// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_openvino_genai.h"
#include "wasinn_output.h"
#include "wasinnenv.h"

#include <algorithm>

using namespace std::literals;

namespace WasmEdge::Host::WASINN::OpenVINOGenAI {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_OPENVINOGENAI

namespace {

WASINN::ErrNo
checkModelReady(const std::unique_ptr<OpenVINOGenAIBackend> &Model) noexcept {
  if (Model == nullptr) {
    spdlog::error("[WASI-NN] OpenVINO GenAI backend: Model is empty."sv);
    return WASINN::ErrNo::MissingMemory;
  }
  return WASINN::ErrNo::Success;
}

} // namespace

Expect<WASINN::ErrNo> isStringTensor(const TensorData &Tensor) noexcept {
  if (Tensor.RType != WASINN::TensorType::U8) {
    spdlog::warn(
        "[WASI-NN] Only STRING (u8) inputs and outputs are supported for "
        "now. Input Type: {}"sv,
        Tensor.RType);
    // return WASINN::ErrNo::InvalidArgument;
  }
  if (Tensor.Dimension.size() != 1) {
    spdlog::error("[WASI-NN] Tensor dimension is out of range, expect it under "
                  "1-dim, but got {}-dim."sv,
                  Tensor.Dimension.size());
    return WASINN::ErrNo::InvalidArgument;
  }
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo>
LLMPipelineBackend::SetContextInput(Context &CxtRef, uint32_t Index,
                                    const TensorData &Tensor) {

  if (Index != 0) {
    spdlog::error("[WASI-NN] The input index {} is out of range."sv, Index);
    return WASINN::ErrNo::InvalidArgument;
  }

  if (auto Res = isStringTensor(Tensor); !Res) {
    return Res;
  }

  try {
    CxtRef.StringInput = asString(Tensor.Tensor);
  } catch (const std::exception &EX) {
    spdlog::error("[WASI-NN] Set Input Exception: {}"sv, EX.what());
    return WASINN::ErrNo::RuntimeError;
  }
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> LLMPipelineBackend::Generate(Context &CxtRef) {
  try {
    // TODO: let the user to set the generation config.
    spdlog::warn("[WASI-NN] The generation config is not supported for now."sv);
    spdlog::warn("[WASI-NN] Maximum token limit is set to 100."sv);
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    CxtRef.StringOutput = Model->generate(CxtRef.StringInput, config);
  } catch (const std::exception &EX) {
    spdlog::error("[WASI-NN] Generate Exception: {}"sv, EX.what());
    return WASINN::ErrNo::RuntimeError;
  }
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo>
LLMPipelineBackend::GetContextOutput(Context &CxtRef, uint32_t Index,
                                     Span<uint8_t> OutBuffer,
                                     uint32_t &BytesWritten) {
  if (Index != 0) {
    spdlog::error("[WASI-NN] The output index {} is out of range."sv, Index);
    return WASINN::ErrNo::InvalidArgument;
  }

  try {
    if (auto Res =
            copyStringToBuffer(CxtRef.StringOutput, OutBuffer, BytesWritten);
        Res != ErrNo::Success) {
      return Res;
    }
  } catch (const std::exception &EX) {
    spdlog::error("[WASI-NN] Get Output Exception: {}"sv, EX.what());
    return WASINN::ErrNo::RuntimeError;
  }
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> load(WASINN::WasiNNEnvironment &Env,
                           Span<const Span<uint8_t>> Builders,
                           WASINN::Device Device, uint32_t &GraphId) noexcept {
  if (auto Res = checkBuilderCount(Builders, 3, Backend::OpenVINOGenAI);
      Res != WASINN::ErrNo::Success) {
    return Res;
  }

  // Get the XML and Weight raw buffer.
  //   Builder-0: Reserved, the string "LLMPipeline"
  //   Builder-1: Path to the dir model xml/bin files
  //   Builder-2: Empty for now (reserved for future use)

  // There are 4 types (text or img) x (text or img), we assume the input is 0
  // for now.
  auto ModelType = asString(Builders[0]);
  auto ModelPath = asString(Builders[1]);
  // TODO: Support extra model information. (ex. enable kv cache)
  [[maybe_unused]] auto ModelExtra = asString(Builders[2]);

  // Add a new graph.
  auto Graph = Env.newGraphGuard(Backend::OpenVINOGenAI);
  auto &GraphRef = Graph.get<Backend::OpenVINOGenAI>();

  // Store device information
  GraphRef.TargetDevice = Device;
  auto DeviceString = getDeviceString(Device, Backend::OpenVINOGenAI, false);
  if (!DeviceString) {
    return DeviceString.error();
  }

  try {
    // Create the OpenVINO GenAI Backend.
    // Currently, we only support LLMPipeline.
    if (ModelType == "LLMPipeline") {
      GraphRef.OpenVINOGenAI =
          std::make_unique<LLMPipelineBackend>(ModelPath, DeviceString.value());
    } else {
      spdlog::error("[WASI-NN] Unsupported model type: {}"sv, ModelType);
      return WASINN::ErrNo::InvalidArgument;
    }

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
  auto GraphInst = Env.getBackendGraphOrError<Backend::OpenVINOGenAI>(
      GraphId, "init_execution_context"sv);
  if (!GraphInst) {
    return GraphInst.error();
  }
  auto &GraphRef = **GraphInst;
  if (auto Res = checkModelReady(GraphRef.OpenVINOGenAI);
      Res != WASINN::ErrNo::Success) {
    return Res;
  }
  ContextId = Env.newReadyContext(GraphId);
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> setInput(WASINN::WasiNNEnvironment &Env,
                               uint32_t ContextId, uint32_t Index,
                               const TensorData &Tensor) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::OpenVINOGenAI>(
      ContextId, "set_input"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();

  if (auto Res = checkModelReady(GraphRef.OpenVINOGenAI);
      Res != WASINN::ErrNo::Success) {
    return Res;
  }

  return GraphRef.OpenVINOGenAI->SetContextInput(CxtRef, Index, Tensor);
}

Expect<WASINN::ErrNo> getOutput(WASINN::WasiNNEnvironment &Env,
                                uint32_t ContextId, uint32_t Index,
                                Span<uint8_t> OutBuffer,
                                uint32_t &BytesWritten) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::OpenVINOGenAI>(
      ContextId, "get_output"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();

  if (auto Res = checkModelReady(GraphRef.OpenVINOGenAI);
      Res != WASINN::ErrNo::Success) {
    return Res;
  }

  return GraphRef.OpenVINOGenAI->GetContextOutput(CxtRef, Index, OutBuffer,
                                                  BytesWritten);
}

Expect<WASINN::ErrNo> compute(WASINN::WasiNNEnvironment &Env,
                              uint32_t ContextId) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::OpenVINOGenAI>(
      ContextId, "compute"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();
  try {
    GraphRef.OpenVINOGenAI->Generate(CxtRef);
  } catch (const std::exception &EX) {
    spdlog::error("[WASI-NN] Infer Request Exception: {}"sv, EX.what());
    return WASINN::ErrNo::RuntimeError;
  }
  return WASINN::ErrNo::Success;
}
#endif
} // namespace WasmEdge::Host::WASINN::OpenVINOGenAI
