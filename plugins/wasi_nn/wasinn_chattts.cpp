// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_chattts.h"
#include "wasinn_output.h"
#include "wasinnenv.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_CHATTTS
#include "simdjson.h"

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__WIN32__) &&             \
    !defined(__TOS_WIN__) && !defined(__WINDOWS__)
#include <dlfcn.h>
#endif
#include <chrono>
#include <time.h>
#endif

namespace WasmEdge::Host::WASINN::ChatTTS {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_CHATTTS
#if defined(_WIN32) || defined(_WIN64) || defined(__WIN32__) ||                \
    defined(__TOS_WIN__) || defined(__WINDOWS__)
HINSTANCE SharedLib = LoadLibrary(PYTHON_LIB_PATH);
#else
void *SharedLib = dlopen(PYTHON_LIB_PATH, RTLD_GLOBAL | RTLD_NOW);
#endif
Expect<WASINN::ErrNo> load(WASINN::WasiNNEnvironment &Env,
                           Span<const Span<uint8_t>>, WASINN::Device,
                           uint32_t &GraphId) noexcept {
  // Add a new graph.
  auto Graph = Env.newGraphGuard(Backend::ChatTTS);
  auto &GraphRef = Graph.get<Backend::ChatTTS>();
  // Initialize the plugin parameters.
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN] ChatTTS backend: Load."sv);
  }

  // Create Model class
  if (!Py_IsInitialized()) {
    Py_Initialize();
    if (PyGILState_Check()) {
      PyEval_SaveThread();
    }
  }
  GIL Lock;
  if (GraphRef.ChatTTSModule == nullptr) {
    GraphRef.ChatTTSModule.reset(PyImport_ImportModule("ChatTTS"));
    if (GraphRef.ChatTTSModule == nullptr) {
      spdlog::error(
          "[WASI-NN] ChatTTS backend: Cannot find ChatTTS library."sv);
      return WASINN::ErrNo::RuntimeError;
    }
  }
  if (GraphRef.Chat == nullptr) {
    PyObjectPtr ChatFunction(
        PyObject_GetAttrString(GraphRef.ChatTTSModule.get(), "Chat"));
    if (ChatFunction == nullptr || !PyCallable_Check(ChatFunction.get())) {
      spdlog::error(
          "[WASI-NN] ChatTTS backend: Cannot find Chat class in ChatTTS."sv);
      return WASINN::ErrNo::RuntimeError;
    }
    GraphRef.Chat.reset(PyObject_CallObject(ChatFunction.get(), nullptr));
    if (GraphRef.Chat == nullptr) {
      spdlog::error("[WASI-NN] ChatTTS backend: Cannot create chat."sv);
      return WASINN::ErrNo::RuntimeError;
    }
    PyObjectPtr LoadMethod(PyObject_GetAttrString(GraphRef.Chat.get(), "load"));
    if (LoadMethod == nullptr || !PyCallable_Check(LoadMethod.get())) {
      spdlog::error("[WASI-NN] ChatTTS backend: Cannot load chat."sv);
      return WASINN::ErrNo::RuntimeError;
    }
    PyObjectPtr Value(PyObject_CallObject(LoadMethod.get(), nullptr));
  }
  // Store the loaded graph.
  GraphId = Graph.commit();

  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> initExecCtx(WasiNNEnvironment &Env, uint32_t GraphId,
                                  uint32_t &ContextId) noexcept {
  if (!Py_IsInitialized()) {
    spdlog::error(
        "[WASI-NN] ChatTTS backend: Model has been released, please reload it."sv);
    return WASINN::ErrNo::RuntimeError;
  }
  ContextId = Env.newReadyContext(GraphId);
  return ErrNo::Success;
}

Expect<WASINN::ErrNo> setInput(WasiNNEnvironment &Env, uint32_t ContextId,
                               uint32_t Index,
                               const TensorData &Tensor) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::ChatTTS>(
      ContextId, "set_input"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();
  if (!Py_IsInitialized()) {
    spdlog::error(
        "[WASI-NN] ChatTTS backend: Model has been released, please reload it."sv);
    return WASINN::ErrNo::RuntimeError;
  }
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN] ChatTTS backend: setInput"sv);
  }
  if (Index == 0) {
    // Set the input.
    std::string Prompt(reinterpret_cast<char *>(Tensor.Tensor.data()),
                       Tensor.Tensor.size());
    CxtRef.Inputs.clear();
    CxtRef.Inputs = Prompt;
    return WASINN::ErrNo::Success;
  } else if (Index == 1) {
    // Set metadata.
    std::string Metadata = asString(Tensor.Tensor);
    simdjson::dom::parser Parser;
    simdjson::dom::element Doc;
    auto ParseError = Parser.parse(Metadata).get(Doc);
    if (ParseError) {
      spdlog::error("[WASI-NN] ChatTTS backend: Parse metadata error"sv);
      return ErrNo::InvalidEncoding;
    }
    GIL Lock;
    // Handle Refine Text Params
    PyObjectPtr PromptObj = nullptr;
    if (Doc.at_key("prompt").error() == simdjson::SUCCESS) {
      std::string_view PromptView;
      auto Err = Doc["prompt"].get<std::string_view>().get(PromptView);
      if (Err) {
        spdlog::error(
            "[WASI-NN] ChatTTS backend: Unable to retrieve the prompt option."sv);
        return ErrNo::InvalidArgument;
      }
      PromptObj.reset(PyUnicode_FromString(std::string(PromptView).c_str()));
    }
    if (PromptObj != nullptr) {
      PyObjectPtr Args(PyTuple_New(0));
      PyObjectPtr Kwargs(PyDict_New());
      PyDict_SetItemString(Kwargs.get(), "prompt", PromptObj.get());
      PyObjectPtr RefineTextParamsFun(
          PyObject_GetAttrString(GraphRef.Chat.get(), "RefineTextParams"));
      GraphRef.ParamsRefineText.reset(
          PyObject_Call(RefineTextParamsFun.get(), Args.get(), Kwargs.get()));
    }
    // Handle Infer Code Params
    PyObjectPtr InferKwargs(PyDict_New());
    if (Doc.at_key("temperature").error() == simdjson::SUCCESS) {
      double Temperature;
      auto Err = Doc["temperature"].get<double>().get(Temperature);
      if (Err) {
        spdlog::error(
            "[WASI-NN] ChatTTS backend: Unable to retrieve the temperature option."sv);
        return ErrNo::InvalidArgument;
      }
      PyObjectPtr TemperatureObject(PyFloat_FromDouble(Temperature));
      PyDict_SetItemString(InferKwargs.get(), "temperature",
                           TemperatureObject.get());
    }
    if (Doc.at_key("top_K").error() == simdjson::SUCCESS) {
      double TopK;
      auto Err = Doc["top_K"].get<double>().get(TopK);
      if (Err) {
        spdlog::error(
            "[WASI-NN] ChatTTS backend: Unable to retrieve the topK option."sv);
        return ErrNo::InvalidArgument;
      }
      PyObjectPtr TopKObject(PyFloat_FromDouble(TopK));
      PyDict_SetItemString(InferKwargs.get(), "top_K", TopKObject.get());
    }
    if (Doc.at_key("top_P").error() == simdjson::SUCCESS) {
      double TopP;
      auto Err = Doc["top_P"].get<double>().get(TopP);
      if (Err) {
        spdlog::error(
            "[WASI-NN] ChatTTS backend: Unable to retrieve the temperature option."sv);
        return ErrNo::InvalidArgument;
      }
      PyObjectPtr TopPObject(PyFloat_FromDouble(TopP));
      PyDict_SetItemString(InferKwargs.get(), "top_P", TopPObject.get());
    }
    if (Doc.at_key("spk_emb").error() == simdjson::SUCCESS) {
      std::string_view SpkEmb;
      auto Err = Doc["spk_emb"].get<std::string_view>().get(SpkEmb);
      if (Err) {
        spdlog::error(
            "[WASI-NN] ChatTTS backend: Unable to retrieve the spk_emb option."sv);
        return ErrNo::InvalidArgument;
      }
      if (SpkEmb == "random") {
        PyObjectPtr SampleRandomSpeaker(PyObject_GetAttrString(
            GraphRef.Chat.get(), "sample_random_speaker"));
        PyObjectPtr Spk(PyObject_CallNoArgs(SampleRandomSpeaker.get()));
        PyDict_SetItemString(InferKwargs.get(), "spk_emb", Spk.get());
      } else {
        PyObjectPtr Spk(PyUnicode_FromString(std::string(SpkEmb).c_str()));
        PyDict_SetItemString(InferKwargs.get(), "spk_emb", Spk.get());
      }
    }
    if (PyDict_Size(InferKwargs.get()) != 0) {
      PyObjectPtr Args(PyTuple_New(0));
      PyObjectPtr InferCodeParams(
          PyObject_GetAttrString(GraphRef.Chat.get(), "InferCodeParams"));
      GraphRef.ParamsInferCode.reset(
          PyObject_Call(InferCodeParams.get(), Args.get(), InferKwargs.get()));
    }
    return WASINN::ErrNo::Success;
  }
  return WASINN::ErrNo::InvalidArgument;
}

Expect<WASINN::ErrNo> getOutput(WasiNNEnvironment &Env, uint32_t ContextId,
                                uint32_t Index, Span<uint8_t> OutBuffer,
                                uint32_t &BytesWritten) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::ChatTTS>(
      ContextId, "get_output"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN] ChatTTS backend: getOutput"sv);
  }
  if (Index == 0) {
    return copyBytesToBuffer(CxtRef.Outputs, OutBuffer, BytesWritten);
  }
  return WASINN::ErrNo::InvalidArgument;
}

Expect<WASINN::ErrNo> compute(WasiNNEnvironment &Env,
                              uint32_t ContextId) noexcept {
  if (!Py_IsInitialized()) {
    spdlog::error(
        "[WASI-NN] ChatTTS backend: Model has been released, please reload it."sv);
    return WASINN::ErrNo::RuntimeError;
  }
  auto State = Env.getBackendContextGraphOrError<Backend::ChatTTS>(ContextId,
                                                                   "compute"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN] ChatTTS backend: compute"sv);
  }
  if (CxtRef.Inputs.size() == 0) {
    spdlog::error("[WASI-NN] ChatTTS backend: Input is not set!"sv);
    return ErrNo::InvalidArgument;
  }
  GIL Lock;
  PyObjectPtr InputStr(PyUnicode_FromString(CxtRef.Inputs.c_str()));
  PyObjectPtr InferMethod(PyObject_GetAttrString(GraphRef.Chat.get(), "infer"));
  PyObjectPtr Result = nullptr;
  if (InferMethod == nullptr || !PyCallable_Check(InferMethod.get())) {
    spdlog::error(
        "[WASI-NN] ChatTTS backend: Cannot find infer method in Chat."sv);
    PyErr_Print();
    return WASINN::ErrNo::RuntimeError;
  }
  if (GraphRef.ParamsRefineText == nullptr &&
      GraphRef.ParamsInferCode == nullptr) {
    PyObjectPtr Args(PyTuple_Pack(1, InputStr.get()));
    Result.reset(PyObject_CallObject(InferMethod.get(), Args.get()));
  } else {
    PyObjectPtr Args(PyTuple_New(0));
    PyObjectPtr Kwargs(PyDict_New());
    PyDict_SetItemString(Kwargs.get(), "text", InputStr.get());
    if (GraphRef.ParamsRefineText != nullptr) {
      PyDict_SetItemString(Kwargs.get(), "params_refine_text",
                           GraphRef.ParamsRefineText.get());
    }
    if (GraphRef.ParamsInferCode != nullptr) {
      PyDict_SetItemString(Kwargs.get(), "params_infer_code",
                           GraphRef.ParamsInferCode.get());
    }
    Result.reset(PyObject_Call(InferMethod.get(), Args.get(), Kwargs.get()));
  }
  if (Result != nullptr) {
    PyObjectPtr Index(PyLong_FromLong(0));
    PyObjectPtr Wav0(PyObject_GetItem(Result.get(), Index.get()));
    PyObjectPtr BytesObj(PyObject_CallMethod(Wav0.get(), "tobytes", nullptr));
    char *Bytes = PyBytes_AsString(BytesObj.get());
    Py_ssize_t Size = PyBytes_Size(BytesObj.get());
    CxtRef.Outputs = std::vector<uint8_t>(Bytes, Bytes + Size);
  } else {
    spdlog::error(
        "[WASI-NN] ChatTTS backend: Cannot get output from infer method."sv);
    return WASINN::ErrNo::RuntimeError;
  }
  return WASINN::ErrNo::Success;
}

Expect<WASINN::ErrNo> unload(WASINN::WasiNNEnvironment &Env,
                             uint32_t GraphId) noexcept {
  auto GraphInst =
      Env.getBackendGraphOrError<Backend::ChatTTS>(GraphId, "unload"sv);
  if (!GraphInst) {
    return GraphInst.error();
  }
  auto &GraphRef = **GraphInst;
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN] ChatTTS backend: start unload."sv);
  }
  GraphRef.release();
  Env.deleteGraph(GraphId);
  return WASINN::ErrNo::Success;
}
#endif

} // namespace WasmEdge::Host::WASINN::ChatTTS
