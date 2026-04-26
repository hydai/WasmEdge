// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_torch.h"
#include "wasinn_output.h"
#include "wasinnenv.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_TORCH
#include <limits>
#include <torch/torch.h>
#endif

namespace WasmEdge::Host::WASINN::PyTorch {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_TORCH

Expect<ErrNo> TorchScript::setDevice(Device Device) {
  if (Device == Device::CPU) {
    TorchDevice = at::kCPU;
    return ErrNo::Success;
  } else if (Device == Device::GPU) {
    if (!torch::cuda::is_available()) {
      spdlog::error("[WASI-NN] Torch: CUDA Unavailable. Please check if the "
                    "installed Torch version or platform supports CUDA."sv);
      return ErrNo::InvalidArgument;
    }
    TorchDevice = at::kCUDA;
    return ErrNo::Success;
  }

  spdlog::error("[WASI-NN] Torch: Unknown target device. We currently support "
                "only CPU and GPU targets."sv);
  return ErrNo::InvalidArgument;
}

Expect<ErrNo> TorchScript::loadFromBinary(std::istream &In, Device Device) {
  if (auto Err = setDevice(Device); Err != ErrNo::Success) {
    return Err;
  }
  TorchModel = torch::jit::load(In);
  return ErrNo::Success;
}

Expect<ErrNo> TorchScript::loadFromPath(const std::string &Path,
                                        Device Device) {
  if (auto Err = setDevice(Device); Err != ErrNo::Success) {
    return Err;
  }
  TorchModel = torch::jit::load(Path);
  return ErrNo::Success;
}

Expect<ErrNo> TorchScript::run(std::vector<at::Tensor> In,
                               std::vector<at::Tensor> &Out) {
  std::vector<torch::jit::IValue> Inputs;
  std::vector<at::Tensor> Outputs;
  for (auto &OneOf : In) {
    Inputs.push_back(OneOf);
  }
  auto RawOutput = TorchModel.forward(Inputs);
  if (RawOutput.isTensorList()) {
    auto OutTensors = RawOutput.toTensorVector();
    for (auto &OneOf : OutTensors) {
      Out.push_back(OneOf.clone());
    }
  } else if (RawOutput.isTuple()) {
    auto OutTensorsTuple = RawOutput.toTuple()->elements();
    for (auto &OneOf : OutTensorsTuple) {
      Out.push_back(OneOf.toTensor().clone());
    }
  } else if (RawOutput.isTensor()) {
    auto OutTensor = RawOutput.toTensor();
    Out.push_back(OutTensor.clone());
  } else {
    spdlog::error(
        "[WASI-NN] Torch: The output can only be one of the following tensor "
        "types: a tensor, a list of tensors, or a tuple of tensors."sv);
    return ErrNo::InvalidArgument;
  }
  return ErrNo::Success;
}

AOTInductor::AOTInductor() : TorchModel(nullptr) {
#if defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI == 1
  spdlog::warn(
      "[WASI-NN] Torch: AOTInductor build by pip default is not supported in "
      "_GLIBCXX_USE_CXX11_ABI=1. Please rebuild the WasmEdge with "
      "_GLIBCXX_USE_CXX11_ABI=0."sv);
#endif
}

Expect<ErrNo> AOTInductor::setDevice(Device Device) {
  if (Device == Device::CPU) {
    TorchDevice = at::kCPU;
    return ErrNo::Success;
  } else if (Device == Device::GPU) {
#ifdef TORCHAOTI_USE_CUDA
    TorchDevice = at::kCUDA;
    return ErrNo::Success;
#else
    spdlog::error("[WASI-NN] Torch: Please rebuild the plugin with AOTInductor "
                  "CUDA support."sv);
    return ErrNo::InvalidArgument;
#endif
  }

  spdlog::error("[WASI-NN] Torch: Unknown target device. We currently support "
                "only CPU and GPU targets."sv);
  return ErrNo::InvalidArgument;
}

Expect<ErrNo> AOTInductor::loadFromBinary(std::istream &, Device) {
  spdlog::error(
      "[WASI-NN] Torch: AOTInductor cannot load by binary data. Please "
      "pass the shared library name (*.so) in nn-preload"sv);
  return ErrNo::InvalidArgument;
}

Expect<ErrNo> AOTInductor::loadFromPath(const std::string &Path,
                                        Device Device) {
  if (auto Err = setDevice(Device); Err != ErrNo::Success) {
    return Err;
  }
  if (TorchDevice == at::kCPU) {
    TorchModel = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        Path.c_str());
  } else if (TorchDevice == at::kCUDA) {
#ifdef TORCHAOTI_USE_CUDA
    TorchModel =
        std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
            Path.c_str());
#else
    spdlog::error("[WASI-NN] Torch: Please rebuild the plugin with AOTInductor "
                  "CUDA support."sv);
    return ErrNo::InvalidArgument;
#endif
  } else {
    spdlog::error("[WASI-NN] Torch: Cannot load the AOTInductor."sv);
    return ErrNo::InvalidArgument;
  }
  return ErrNo::Success;
}

Expect<ErrNo> AOTInductor::run(std::vector<at::Tensor> In,
                               std::vector<at::Tensor> &Out) {
  std::vector<at::Tensor> RawOutput = TorchModel->run(In);

  for (auto &OneOf : RawOutput) {
    Out.push_back(OneOf.clone());
  }
  return ErrNo::Success;
}

PyModelBackend guessPyModelBackendType(const std::string_view &Model) {
  // TODO: Add more model type detection when we support more OS.
  // ex .dll, .dylib, etc.
  if (Model.substr(0, 8) == "preload:"sv) {
    if (Model.substr(Model.size() - 3, 3) == ".so"sv) {
      // AOTInductor only accept the shared library.
      return PyModelBackend::AOTInductor;
    }
  }

  // ELF Header: 0x7f 'E' 'L' 'F'
  if (Model.substr(0, 4) == "\x7f\x45\x4c\x46"sv) {
    return PyModelBackend::AOTInductor;
  }

  // Fall back to TorchScript if the model type is not set.
  // This keep the compatibility with the old version.
  return PyModelBackend::TorchScript;
}

Expect<ErrNo> load(WasiNNEnvironment &Env, Span<const Span<uint8_t>> Builders,
                   Device Device, uint32_t &GraphId) noexcept {
  if (auto Res = checkBuilderCount(Builders, 1, Backend::PyTorch);
      Res != ErrNo::Success) {
    return Res;
  }

  auto Weight = Builders[0];
  // Add a new graph.
  auto Graph = Env.newGraphGuard(Backend::PyTorch);
  auto &GraphRef = Graph.get<Backend::PyTorch>();

  // Load the model from the binary data.
  // Note: Pytorch use try catch to handle the error.
  try {
    const std::string_view BinModel = asStringView(Weight);
    PyModelBackend ModelType = guessPyModelBackendType(BinModel);

    if (ModelType == PyModelBackend::TorchScript) {
      GraphRef.Model = std::make_unique<TorchScript>();
    } else if (ModelType == PyModelBackend::AOTInductor) {
      GraphRef.Model = std::make_unique<AOTInductor>();
    } else {
      spdlog::error("[WASI-NN] Torch: Unknown model type."sv);
      return ErrNo::InvalidArgument;
    }

    if (BinModel.substr(0, 8) == "preload:"sv) {
      const std::string ModelFilePath(BinModel.substr(8));
      if (auto Res = GraphRef.Model->loadFromPath(ModelFilePath, Device);
          Res != ErrNo::Success) {
        return Res;
      }
    } else {
      std::istringstream BinRead{std::string(BinModel)};
      // std::istringstream BinRead(BinModel); // Need C++26...
      if (auto Res = GraphRef.Model->loadFromBinary(BinRead, Device);
          Res != ErrNo::Success) {
        return Res;
      }
    }
  } catch (const c10::Error &e) {
    spdlog::error("[WASI-NN] Torch: Failed when load the TorchScript model."sv);
    return ErrNo::InvalidArgument;
  }

  GraphId = Graph.commit();
  return ErrNo::Success;
}

Expect<ErrNo> initExecCtx(WasiNNEnvironment &Env, uint32_t GraphId,
                          uint32_t &ContextId) noexcept {
  ContextId = Env.newReadyContext(GraphId);
  return ErrNo::Success;
}

Expect<ErrNo> setInput(WasiNNEnvironment &Env, uint32_t ContextId,
                       uint32_t Index, const TensorData &Tensor) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::PyTorch>(
      ContextId, "set_input"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();
  if (Index >= CxtRef.TorchInputs.size()) {
    CxtRef.TorchInputs.resize(Index + 1);
  }
  if (Tensor.RType != TensorType::F32) {
    spdlog::error(
        "[WASI-NN] Torch: Only F32 inputs and outputs are supported for now."sv);
    return ErrNo::InvalidArgument;
  }
  auto Options =
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
  std::vector<int64_t> Dims;
  for (size_t I = 0; I < Tensor.Dimension.size(); I++) {
    Dims.push_back(static_cast<int64_t>(Tensor.Dimension[I]));
  }
  torch::Tensor InTensor =
      torch::from_blob(reinterpret_cast<float *>(Tensor.Tensor.data()), Dims,
                       Options)
          .to(GraphRef.Model->getDevice());

  CxtRef.TorchInputs[Index] = InTensor.clone();
  return ErrNo::Success;
}

Expect<ErrNo> getOutput(WasiNNEnvironment &Env, uint32_t ContextId,
                        uint32_t Index, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten) noexcept {
  auto CxtInst =
      Env.getBackendContextOrError<Backend::PyTorch>(ContextId, "get_output"sv);
  if (!CxtInst) {
    return CxtInst.error();
  }
  auto &CxtRef = **CxtInst;
  if (CxtRef.TorchOutputs.size() <= Index) {
    spdlog::error(
        "[WASI-NN] Torch: The output index {} exceeds the outputs number {}."sv,
        Index, CxtRef.TorchOutputs.size());
    return ErrNo::InvalidArgument;
  }
  torch::Tensor OutTensor =
      CxtRef.TorchOutputs[Index].to(at::kCPU).toType(torch::kFloat32);
  float *TensorBuffer = OutTensor.data_ptr<float>();

  size_t BlobSize = 1;
  for (auto I : OutTensor.sizes()) {
    const auto Dimension = static_cast<size_t>(I);
    if (Dimension != 0 &&
        BlobSize > std::numeric_limits<size_t>::max() / Dimension) {
      spdlog::error("[WASI-NN] Torch: Output tensor size is too large."sv);
      return ErrNo::InvalidArgument;
    }
    BlobSize *= Dimension;
  }
  if (BlobSize > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) /
                     sizeof(float)) {
    spdlog::error("[WASI-NN] Torch: Output tensor size is too large."sv);
    return ErrNo::InvalidArgument;
  }

  const size_t TensorByteSize = BlobSize * sizeof(float);
  return copyBytesToBuffer(
      {reinterpret_cast<const uint8_t *>(TensorBuffer), TensorByteSize},
      OutBuffer, BytesWritten);
}

Expect<ErrNo> compute(WasiNNEnvironment &Env, uint32_t ContextId) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::PyTorch>(ContextId,
                                                                   "compute"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();
  if (CxtRef.TorchInputs.size() == 0) {
    spdlog::error("[WASI-NN] Torch: Input is not set!"sv);
    return ErrNo::InvalidArgument;
  }
  for (size_t I = 0; I < CxtRef.TorchInputs.size(); I++) {
    torch::jit::IValue InTensor = CxtRef.TorchInputs[I];
    if (InTensor.isNone()) {
      spdlog::error("[WASI-NN] Torch: Input [{}] is not set!"sv, I);
      return ErrNo::InvalidArgument;
    }
  }
  return GraphRef.Model->run(CxtRef.TorchInputs, CxtRef.TorchOutputs);
}

Expect<ErrNo> unload(WasiNNEnvironment &Env, uint32_t GraphId) noexcept {
  auto GraphInst =
      Env.getBackendGraphOrError<Backend::PyTorch>(GraphId, "unload"sv);
  if (!GraphInst) {
    return GraphInst.error();
  }
  auto &GraphRef = **GraphInst;
  GraphRef.Model.reset();
  return ErrNo::Success;
}
#endif
} // namespace WasmEdge::Host::WASINN::PyTorch
