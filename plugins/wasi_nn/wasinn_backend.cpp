// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_backend.h"

#include "GGML/core/ggml_core.h"
#include "wasinn_bitnet.h"
#include "wasinn_chattts.h"
#include "wasinn_mlx.h"
#include "wasinn_openvino.h"
#include "wasinn_openvino_genai.h"
#include "wasinn_piper.h"
#include "wasinn_tfl.h"
#include "wasinn_torch.h"
#include "wasinn_whisper.h"

#include <array>
#include <cstddef>

namespace WasmEdge {
namespace Host {
namespace WASINN {
namespace {
using namespace std::literals;

constexpr BackendOperations backendOperations(
    const char *Name, Backend Type, BackendOperations::LoadHandler Load,
    BackendOperations::InitExecCtxHandler InitExecCtx,
    BackendOperations::SetInputHandler SetInput,
    BackendOperations::GetOutputHandler GetOutput,
    BackendOperations::ComputeHandler Compute,
    BackendOperations::GetOutputHandler GetOutputSingle = nullptr,
    BackendOperations::ComputeHandler ComputeSingle = nullptr,
    BackendOperations::ComputeHandler FiniSingle = nullptr,
    BackendOperations::UnloadHandler Unload = nullptr,
    BackendOperations::ComputeHandler FinalizeExecCtx = nullptr) noexcept {
  return {Name,          Type,       Load,    InitExecCtx,
          SetInput,      GetOutput,  Compute, GetOutputSingle,
          ComputeSingle, FiniSingle, Unload,  FinalizeExecCtx};
}

constexpr BackendOperations
basicBackendOperations(const char *Name, Backend Type,
                       BackendOperations::LoadHandler Load,
                       BackendOperations::InitExecCtxHandler InitExecCtx,
                       BackendOperations::SetInputHandler SetInput,
                       BackendOperations::GetOutputHandler GetOutput,
                       BackendOperations::ComputeHandler Compute) noexcept {
  return backendOperations(Name, Type, Load, InitExecCtx, SetInput, GetOutput,
                           Compute);
}

constexpr BackendOperations unloadBackendOperations(
    const char *Name, Backend Type, BackendOperations::LoadHandler Load,
    BackendOperations::InitExecCtxHandler InitExecCtx,
    BackendOperations::SetInputHandler SetInput,
    BackendOperations::GetOutputHandler GetOutput,
    BackendOperations::ComputeHandler Compute,
    BackendOperations::UnloadHandler Unload,
    BackendOperations::ComputeHandler FinalizeExecCtx = nullptr) noexcept {
  return backendOperations(Name, Type, Load, InitExecCtx, SetInput, GetOutput,
                           Compute, nullptr, nullptr, nullptr, Unload,
                           FinalizeExecCtx);
}

constexpr BackendOperations singleBackendOperations(
    const char *Name, Backend Type, BackendOperations::LoadHandler Load,
    BackendOperations::InitExecCtxHandler InitExecCtx,
    BackendOperations::SetInputHandler SetInput,
    BackendOperations::GetOutputHandler GetOutput,
    BackendOperations::ComputeHandler Compute,
    BackendOperations::GetOutputHandler GetOutputSingle,
    BackendOperations::ComputeHandler ComputeSingle,
    BackendOperations::ComputeHandler FiniSingle,
    BackendOperations::UnloadHandler Unload,
    BackendOperations::ComputeHandler FinalizeExecCtx) noexcept {
  return backendOperations(Name, Type, Load, InitExecCtx, SetInput, GetOutput,
                           Compute, GetOutputSingle, ComputeSingle, FiniSingle,
                           Unload, FinalizeExecCtx);
}

constexpr BackendOperations
unsupportedBackendOperations(const char *Name, Backend Type) noexcept {
  return backendOperations(Name, Type, nullptr, nullptr, nullptr, nullptr,
                           nullptr);
}

template <Backend B> Expect<ErrNo> reportUnavailableBackend() noexcept {
  return reportBackendNotSupported(B);
}

template <>
Expect<ErrNo> reportUnavailableBackend<Backend::NeuralSpeed>() noexcept {
  return reportBackendRemoved(Backend::NeuralSpeed,
                              "https://github.com/intel/neural-speed");
}

template <Backend B>
Expect<ErrNo> unavailableLoad(WasiNNEnvironment &, Span<const Span<uint8_t>>,
                              Device, uint32_t &) noexcept {
  return reportUnavailableBackend<B>();
}

template <Backend B>
Expect<ErrNo> unavailableInitExecCtx(WasiNNEnvironment &, uint32_t,
                                     uint32_t &) noexcept {
  return reportUnavailableBackend<B>();
}

template <Backend B>
Expect<ErrNo> unavailableSetInput(WasiNNEnvironment &, uint32_t, uint32_t,
                                  const TensorData &) noexcept {
  return reportUnavailableBackend<B>();
}

template <Backend B>
Expect<ErrNo> unavailableGetOutput(WasiNNEnvironment &, uint32_t, uint32_t,
                                   Span<uint8_t>, uint32_t &) noexcept {
  return reportUnavailableBackend<B>();
}

template <Backend B>
Expect<ErrNo> unavailableCompute(WasiNNEnvironment &, uint32_t) noexcept {
  return reportUnavailableBackend<B>();
}

template <Backend B>
Expect<ErrNo> notBuiltLoad(WasiNNEnvironment &, Span<const Span<uint8_t>>,
                           Device, uint32_t &) noexcept {
  return reportBackendNotBuilt(B);
}

template <Backend B>
Expect<ErrNo> notBuiltInitExecCtx(WasiNNEnvironment &, uint32_t,
                                  uint32_t &) noexcept {
  return reportBackendNotBuilt(B);
}

template <Backend B>
Expect<ErrNo> notBuiltSetInput(WasiNNEnvironment &, uint32_t, uint32_t,
                               const TensorData &) noexcept {
  return reportBackendNotBuilt(B);
}

template <Backend B>
Expect<ErrNo> notBuiltGetOutput(WasiNNEnvironment &, uint32_t, uint32_t,
                                Span<uint8_t>, uint32_t &) noexcept {
  return reportBackendNotBuilt(B);
}

template <Backend B>
Expect<ErrNo> notBuiltCompute(WasiNNEnvironment &, uint32_t) noexcept {
  return reportBackendNotBuilt(B);
}

template <Backend B>
constexpr BackendOperations
notBuiltBasicBackendOperations(const char *Name, Backend Type) noexcept {
  return basicBackendOperations(Name, Type, notBuiltLoad<B>,
                                notBuiltInitExecCtx<B>, notBuiltSetInput<B>,
                                notBuiltGetOutput<B>, notBuiltCompute<B>);
}

template <Backend B>
constexpr BackendOperations
notBuiltUnloadBackendOperations(const char *Name, Backend Type) noexcept {
  return unloadBackendOperations(Name, Type, notBuiltLoad<B>,
                                 notBuiltInitExecCtx<B>, notBuiltSetInput<B>,
                                 notBuiltGetOutput<B>, notBuiltCompute<B>,
                                 notBuiltCompute<B>, notBuiltCompute<B>);
}

template <Backend B>
constexpr BackendOperations
notBuiltSingleBackendOperations(const char *Name, Backend Type) noexcept {
  return singleBackendOperations(
      Name, Type, notBuiltLoad<B>, notBuiltInitExecCtx<B>, notBuiltSetInput<B>,
      notBuiltGetOutput<B>, notBuiltCompute<B>, notBuiltGetOutput<B>,
      notBuiltCompute<B>, notBuiltCompute<B>, notBuiltCompute<B>,
      notBuiltCompute<B>);
}

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_OPENVINO
const BackendOperations OpenVINOOperations = basicBackendOperations(
    "OpenVINO", Backend::OpenVINO, OpenVINO::load, OpenVINO::initExecCtx,
    OpenVINO::setInput, OpenVINO::getOutput, OpenVINO::compute);
#else
const BackendOperations OpenVINOOperations =
    notBuiltBasicBackendOperations<Backend::OpenVINO>("OpenVINO",
                                                      Backend::OpenVINO);
#endif

const BackendOperations ONNXOperations = basicBackendOperations(
    "ONNX", Backend::ONNX, unavailableLoad<Backend::ONNX>,
    unavailableInitExecCtx<Backend::ONNX>, unavailableSetInput<Backend::ONNX>,
    unavailableGetOutput<Backend::ONNX>, unavailableCompute<Backend::ONNX>);

const BackendOperations TensorflowOperations = basicBackendOperations(
    "Tensorflow", Backend::Tensorflow, unavailableLoad<Backend::Tensorflow>,
    unavailableInitExecCtx<Backend::Tensorflow>,
    unavailableSetInput<Backend::Tensorflow>,
    unavailableGetOutput<Backend::Tensorflow>,
    unavailableCompute<Backend::Tensorflow>);

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_TORCH
const BackendOperations PyTorchOperations = basicBackendOperations(
    "PyTorch", Backend::PyTorch, PyTorch::load, PyTorch::initExecCtx,
    PyTorch::setInput, PyTorch::getOutput, PyTorch::compute);
#else
const BackendOperations PyTorchOperations =
    notBuiltBasicBackendOperations<Backend::PyTorch>("PyTorch",
                                                     Backend::PyTorch);
#endif

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_TFLITE
const BackendOperations TensorflowLiteOperations = basicBackendOperations(
    "TensorflowLite", Backend::TensorflowLite, TensorflowLite::load,
    TensorflowLite::initExecCtx, TensorflowLite::setInput,
    TensorflowLite::getOutput, TensorflowLite::compute);
#else
const BackendOperations TensorflowLiteOperations =
    notBuiltBasicBackendOperations<Backend::TensorflowLite>(
        "TensorflowLite", Backend::TensorflowLite);
#endif

const BackendOperations AutodetectOperations =
    unsupportedBackendOperations("Autodetect", Backend::Autodetect);

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
const BackendOperations GGMLOperations = singleBackendOperations(
    "ggml", Backend::GGML, GGML::load, GGML::initExecCtx, GGML::setInput,
    GGML::getOutput, GGML::compute, GGML::getOutputSingle, GGML::computeSingle,
    GGML::finiSingle, GGML::unload, GGML::finalizeExecCtx);
#else
const BackendOperations GGMLOperations =
    notBuiltSingleBackendOperations<Backend::GGML>("ggml", Backend::GGML);
#endif

const BackendOperations NeuralSpeedOperations = basicBackendOperations(
    "Neural Speed", Backend::NeuralSpeed, unavailableLoad<Backend::NeuralSpeed>,
    unavailableInitExecCtx<Backend::NeuralSpeed>,
    unavailableSetInput<Backend::NeuralSpeed>,
    unavailableGetOutput<Backend::NeuralSpeed>,
    unavailableCompute<Backend::NeuralSpeed>);

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_WHISPER
const BackendOperations WhisperOperations = unloadBackendOperations(
    "Whisper", Backend::Whisper, Whisper::load, Whisper::initExecCtx,
    Whisper::setInput, Whisper::getOutput, Whisper::compute, Whisper::unload,
    Whisper::finalizeExecCtx);
#else
const BackendOperations WhisperOperations =
    notBuiltUnloadBackendOperations<Backend::Whisper>("Whisper",
                                                      Backend::Whisper);
#endif

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_PIPER
const BackendOperations PiperOperations = basicBackendOperations(
    "Piper", Backend::Piper, Piper::load, Piper::initExecCtx, Piper::setInput,
    Piper::getOutput, Piper::compute);
#else
const BackendOperations PiperOperations =
    notBuiltBasicBackendOperations<Backend::Piper>("Piper", Backend::Piper);
#endif

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_CHATTTS
const BackendOperations ChatTTSOperations = unloadBackendOperations(
    "ChatTTS", Backend::ChatTTS, ChatTTS::load, ChatTTS::initExecCtx,
    ChatTTS::setInput, ChatTTS::getOutput, ChatTTS::compute, ChatTTS::unload);
#else
const BackendOperations ChatTTSOperations =
    notBuiltUnloadBackendOperations<Backend::ChatTTS>("ChatTTS",
                                                      Backend::ChatTTS);
#endif

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_MLX
const BackendOperations MLXOperations =
    basicBackendOperations("MLX", Backend::MLX, MLX::load, MLX::initExecCtx,
                           MLX::setInput, MLX::getOutput, MLX::compute);
#else
const BackendOperations MLXOperations =
    notBuiltBasicBackendOperations<Backend::MLX>("MLX", Backend::MLX);
#endif

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_OPENVINOGENAI
const BackendOperations OpenVINOGenAIOperations = basicBackendOperations(
    "OpenVINO GenAI", Backend::OpenVINOGenAI, OpenVINOGenAI::load,
    OpenVINOGenAI::initExecCtx, OpenVINOGenAI::setInput,
    OpenVINOGenAI::getOutput, OpenVINOGenAI::compute);
#else
const BackendOperations OpenVINOGenAIOperations =
    notBuiltBasicBackendOperations<Backend::OpenVINOGenAI>(
        "OpenVINO GenAI", Backend::OpenVINOGenAI);
#endif

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET
const BackendOperations BitNetOperations = singleBackendOperations(
    "BitNet", Backend::BitNet, BitNet::load, BitNet::initExecCtx,
    BitNet::setInput, BitNet::getOutput, BitNet::compute,
    BitNet::getOutputSingle, BitNet::computeSingle, BitNet::finiSingle,
    BitNet::unload, BitNet::finalizeExecCtx);
#else
const BackendOperations BitNetOperations =
    notBuiltSingleBackendOperations<Backend::BitNet>("BitNet", Backend::BitNet);
#endif

struct BackendAlias {
  std::string_view Name;
  BackendSelection Selection;
};

struct BackendSpec {
  BackendDescriptor Descriptor;
  std::array<BackendAlias, 2> Aliases;
  size_t AliasCount;
};

constexpr BackendAlias backendAlias(std::string_view Name, Backend Type,
                                    BackendPreloadMode Preload) noexcept {
  return BackendAlias{Name, {Type, Preload}};
}

constexpr BackendSpec backendSpec(const BackendOperations &Ops,
                                  std::string_view BuildOption,
                                  BackendPreloadMode Preload,
                                  std::string_view Alias) noexcept {
  return BackendSpec{
      BackendDescriptor{Ops.Type, Ops.Name, BuildOption, Preload, &Ops},
      {backendAlias(Alias, Ops.Type, Preload), BackendAlias{}},
      1};
}

constexpr BackendSpec
backendSpec(const BackendOperations &Ops, std::string_view BuildOption,
            BackendPreloadMode Preload, std::string_view Alias,
            std::string_view SecondaryAlias,
            BackendPreloadMode SecondaryPreload) noexcept {
  return BackendSpec{
      BackendDescriptor{Ops.Type, Ops.Name, BuildOption, Preload, &Ops},
      {backendAlias(Alias, Ops.Type, Preload),
       backendAlias(SecondaryAlias, Ops.Type, SecondaryPreload)},
      2};
}

const std::array BackendSpecs{
    backendSpec(OpenVINOOperations,
                "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"OpenVINO\""sv,
                BackendPreloadMode::Bytes, "openvino"sv),
    backendSpec(ONNXOperations, ""sv, BackendPreloadMode::Bytes, "onnx"sv),
    backendSpec(TensorflowOperations, ""sv, BackendPreloadMode::Bytes,
                "tensorflow"sv),
    backendSpec(PyTorchOperations,
                "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"PyTorch\""sv,
                BackendPreloadMode::Bytes, "pytorch"sv, "pytorchaoti"sv,
                BackendPreloadMode::Path),
    backendSpec(TensorflowLiteOperations,
                "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"TensorflowLite\""sv,
                BackendPreloadMode::Bytes, "tensorflowlite"sv),
    backendSpec(AutodetectOperations, ""sv, BackendPreloadMode::Bytes,
                "autodetect"sv),
    backendSpec(GGMLOperations, "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"GGML\""sv,
                BackendPreloadMode::Path, "ggml"sv),
    backendSpec(NeuralSpeedOperations, ""sv, BackendPreloadMode::Bytes,
                "neuralspeed"sv),
    backendSpec(WhisperOperations,
                "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"Whisper\""sv,
                BackendPreloadMode::Bytes, "whisper"sv),
    backendSpec(PiperOperations,
                "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"Piper\""sv,
                BackendPreloadMode::Bytes, "piper"sv),
    backendSpec(ChatTTSOperations,
                "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"ChatTTS\""sv,
                BackendPreloadMode::Bytes, "chattts"sv),
    backendSpec(MLXOperations, "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"MLX\""sv,
                BackendPreloadMode::Bytes, "mlx"sv),
    backendSpec(OpenVINOGenAIOperations,
                "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"OpenVINOGenAI\""sv,
                BackendPreloadMode::Bytes, "openvinogenai"sv),
    backendSpec(BitNetOperations,
                "-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=\"BitNet\""sv,
                BackendPreloadMode::Path, "bitnet"sv),
};

} // namespace

const BackendOperations *getBackendOperations(Backend Type) noexcept {
  if (const auto *Descriptor = getBackendDescriptor(Type)) {
    return Descriptor->Operations;
  }
  return nullptr;
}

const BackendDescriptor *getBackendDescriptor(Backend Type) noexcept {
  for (const auto &Spec : BackendSpecs) {
    if (Spec.Descriptor.Type == Type) {
      return &Spec.Descriptor;
    }
  }
  return nullptr;
}

const BackendSelection *getBackendSelection(std::string_view Name) noexcept {
  for (const auto &Spec : BackendSpecs) {
    for (size_t I = 0; I < Spec.AliasCount; ++I) {
      const auto &Alias = Spec.Aliases[I];
      if (Alias.Name == Name) {
        return &Alias.Selection;
      }
    }
  }
  return nullptr;
}

std::string_view getBackendName(Backend BE) noexcept {
  if (const auto *Descriptor = getBackendDescriptor(BE)) {
    return Descriptor->Name;
  }
  return "Unknown"sv;
}

std::string_view getBackendBuildOption(Backend BE) noexcept {
  if (const auto *Descriptor = getBackendDescriptor(BE)) {
    return Descriptor->BuildOption;
  }
  return ""sv;
}

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge
