// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_whisper.h"
#include "wasinn_output.h"
#include "wasinn_whisper_audio.h"
#include "wasinn_whisper_config.h"
#include "wasinn_whisper_lifecycle.h"
#include "wasinn_whisper_output.h"
#include "wasinnenv.h"
#include <cstdint>
#include <vector>

using namespace std::literals;

namespace WasmEdge::Host::WASINN::Whisper {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_WHISPER

Expect<ErrNo> load(WasiNNEnvironment &Env, Span<const Span<uint8_t>> Builders,
                   [[maybe_unused]] Device Device, uint32_t &GraphId) noexcept {
  // Add a new graph.
  auto Graph = Env.newGraphGuard(Backend::Whisper);
  auto &GraphRef = Graph.get<Backend::Whisper>();

  // Initialize the parameters.
  auto CParam = whisper_context_default_params();
  GraphRef.ModelFilePath = ""sv;
  GraphRef.WhisperConfig.SpokenLanguage = "en"sv;
  GraphRef.UseGPU = CParam.use_gpu;
  GraphRef.MainGPU = CParam.gpu_device;

  // Set whisper log callback.
  whisper_log_set(WhisperLogCallback, &GraphRef);

  // If the graph builder length > 1, the data of builder[1] is the metadata.
  if (Builders.size() > 1) {
    const std::string Metadata = asString(Builders[1]);
    // Ignore context or model updates when initializing the graph.
    auto Res = parseMetadata(GraphRef.WhisperConfig, Metadata);
    if (Res != ErrNo::Success) {
      spdlog::error("[WASI-NN] Whisper backend: Failed to parse metadata."sv);
      return Res;
    }
  }

  // Handle the model path.
  if (GraphRef.WhisperConfig.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] Whisper backend: Handling model path."sv);
  }
  auto Weight = Builders[0];
  const std::string_view BinModel = asStringView(Weight);
  if (BinModel.substr(0, 8) == "preload:"sv) {
    GraphRef.ModelFilePath = BinModel.substr(8);
  }

  // Initialize whisper context from model file with parameters.
  if (GraphRef.WhisperConfig.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] Whisper backend: Initialize whisper context with "
        "given parameters"sv);
  }
  if (GraphRef.ModelFilePath == ""sv) {
    GraphRef.WhisperCtx.reset(whisper_init_from_buffer_with_params(
        Weight.data(), Weight.size(), CParam));
  } else {
    GraphRef.WhisperCtx.reset(whisper_init_from_file_with_params(
        GraphRef.ModelFilePath.c_str(), CParam));
  }
  if (GraphRef.WhisperCtx == nullptr) {
    spdlog::error(
        "[WASI-NN] Whisper backend: Error: unable to init whisper context from "
        "model."sv);
    return ErrNo::InvalidArgument;
  }
  if (GraphRef.WhisperConfig.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] Whisper backend: Initialize whisper context with "
        "given parameters...Done"sv);
  }

  auto ResTranslateConfig = handleTranslationConfig(GraphRef.WhisperCtx.get(),
                                                    GraphRef.WhisperConfig);
  if (ResTranslateConfig != ErrNo::Success) {
    return ResTranslateConfig;
  }

  // Store the loaded graph.
  GraphId = Graph.commit();

  return ErrNo::Success;
}

Expect<ErrNo> initExecCtx(WasiNNEnvironment &Env, uint32_t GraphId,
                          uint32_t &ContextId) noexcept {
  auto GraphInst = Env.getBackendGraphOrError<Backend::Whisper>(
      GraphId, "init_execution_context"sv);
  if (!GraphInst) {
    return GraphInst.error();
  }
  auto &GraphRef = **GraphInst;
  if (GraphRef.WhisperConfig.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] Whisper backend: initExecCtx"sv);
  }
  auto Context = Env.newContextGuard(GraphId);
  auto &CxtRef = Context.get<Backend::Whisper>();
  CxtRef.WhisperParams = whisper_full_default_params(
      whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH);
  setWhisperParams(CxtRef);
  if (GraphRef.WhisperConfig.EnableLog) {
    spdlog::info("[WASI-NN] Whisper backend: whisper_system_info: {}"sv,
                 whisper_print_system_info());
  }
  ContextId = Context.commit();
  if (GraphRef.WhisperConfig.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] Whisper backend: initExecCtx...Done"sv);
  }
  return ErrNo::Success;
}

Expect<ErrNo> setInput(WasiNNEnvironment &Env, uint32_t ContextId,
                       uint32_t Index [[maybe_unused]],
                       const TensorData &Tensor) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::Whisper>(
      ContextId, "set_input"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();
  if (CxtRef.WhisperConfig.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] Whisper backend: setInput"sv);
  }

  // Use index 1 for metadata.
  if (Index == 1) {
    if (CxtRef.WhisperConfig.EnableDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] Whisper backend: found Metadata, processing"sv);
    }
    // Set the whisper config of this context as the graph default first.
    // This will reset the config and inherit settings from the graph metadata.
    CxtRef.WhisperConfig = GraphRef.WhisperConfig;
    const std::string Metadata(reinterpret_cast<char *>(Tensor.Tensor.data()),
                               Tensor.Tensor.size());
    auto Res = parseMetadata(CxtRef.WhisperConfig, Metadata);
    if (Res != ErrNo::Success) {
      spdlog::error("[WASI-NN] Whisper backend: Failed to parse metadata."sv);
      return Res;
    }
    Res = handleTranslationConfig(GraphRef.WhisperCtx.get(),
                                  CxtRef.WhisperConfig);
    if (Res != ErrNo::Success) {
      return Res;
    }
    setWhisperParams(CxtRef);
    if (CxtRef.WhisperConfig.EnableDebugLog) {
      spdlog::info("[WASI-NN][Debug] Whisper backend: found Metadata, "
                   "processing...Done"sv);
    }
    return ErrNo::Success;
  }

  if (Tensor.Dimension.size() != 2) {
    spdlog::error("[WASI-NN] Tensor dimension is out of range, expect 2-dim, "
                  "but got {}-dim."sv,
                  Tensor.Dimension.size());
    return WASINN::ErrNo::InvalidArgument;
  }
  if (Tensor.Dimension[0] != 1) {
    spdlog::error("[WASI-NN] Only 1 channel supported for now."sv);
    return WASINN::ErrNo::InvalidArgument;
  }

  // Tensor type not used here. Not to check this.

  // Check the input audio file format and load. Currently WAV supported.
  if (!checkAudioRIFF(
          std::string_view(reinterpret_cast<char *>(Tensor.Tensor.data()),
                           Tensor.Tensor.size()),
          "WAVE"sv)) {
    spdlog::error("[WASI-NN] Only WAV format supported now."sv);
    return WASINN::ErrNo::InvalidArgument;
  }
  if (!loadWAV(Tensor.Tensor, CxtRef.InputPCM, CxtRef.InputPCMs,
               CxtRef.WhisperConfig.Diarize)) {
    return WASINN::ErrNo::InvalidArgument;
  }

  if (CxtRef.WhisperConfig.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] Whisper backend: setInput...Done"sv);
  }
  return ErrNo::Success;
}

Expect<ErrNo> getOutput(WasiNNEnvironment &Env, uint32_t ContextId,
                        uint32_t Index, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten) noexcept {
  return Env.withBackendState<Backend::Whisper>(
      ContextId, "get_output"sv,
      [&](Context &CxtRef, Graph &GraphRef) -> Expect<ErrNo> {
        if (CxtRef.WhisperConfig.EnableDebugLog) {
          spdlog::info(
              "[WASI-NN][Debug] Whisper backend: getOutput with Index {}"sv,
              Index);
        }

        if (auto Res =
                copyStringToBuffer(CxtRef.Outputs, OutBuffer, BytesWritten);
            Res != ErrNo::Success) {
          return Res;
        }
        if (CxtRef.WhisperConfig.EnableDebugLog) {
          spdlog::info("[WASI-NN][Debug] Whisper backend: getOutput with Index "
                       "{}...Done"sv,
                       Index);
        }

        if (CxtRef.WhisperConfig.OutputSrt) {
          const auto Fname = CxtRef.WhisperConfig.FileName + ".srt";
          outputSrt(Env, GraphRef.WhisperCtx.get(), Fname, CxtRef.WhisperConfig,
                    CxtRef.InputPCMs);
        }

        if (CxtRef.WhisperConfig.OutputLrc) {
          const auto Fname = CxtRef.WhisperConfig.FileName + ".lrc";
          outputLrc(Env, GraphRef.WhisperCtx.get(), Fname, CxtRef.WhisperConfig,
                    CxtRef.InputPCMs);
        }

        if (CxtRef.WhisperConfig.OutputJson) {
          const auto Fname = CxtRef.WhisperConfig.FileName + ".json";
          outputJson(Env, GraphRef.WhisperCtx.get(), Fname,
                     CxtRef.WhisperConfig, CxtRef.InputPCMs,
                     CxtRef.WhisperConfig.OutputJsonFull);
        }

        return ErrNo::Success;
      });
}

Expect<ErrNo> compute(WasiNNEnvironment &Env, uint32_t ContextId) noexcept {
  return Env.withBackendState<Backend::Whisper>(
      ContextId, "compute"sv,
      [](Context &CxtRef, Graph &GraphRef) -> Expect<ErrNo> {
        if (CxtRef.WhisperConfig.EnableDebugLog) {
          spdlog::info("[WASI-NN][Debug] Whisper backend: compute"sv);
        }

        CxtRef.Outputs.clear();
        if (whisper_full_parallel(GraphRef.WhisperCtx.get(),
                                  CxtRef.WhisperParams, CxtRef.InputPCM.data(),
                                  CxtRef.InputPCM.size(),
                                  CxtRef.WhisperConfig.ProcessorsNum) != 0) {
          spdlog::error(
              "[WASI-NN] Whisper backend: Error: failed to process audio."sv);
          return ErrNo::RuntimeError;
        }

        if (CxtRef.WhisperConfig.EnableDebugLog) {
          spdlog::info("[WASI-NN][Debug] Whisper backend: compute...Done"sv);
        }
        return ErrNo::Success;
      });
}

Expect<ErrNo> unload(WasiNNEnvironment &Env, uint32_t GraphId) noexcept {
  return Env.withBackendGraph<Backend::Whisper>(
      GraphId, "unload"sv, [&](Graph &GraphRef) -> Expect<ErrNo> {
        const bool IsDebugLog = GraphRef.WhisperConfig.EnableDebugLog;
        if (IsDebugLog) {
          spdlog::info("[WASI-NN][Debug] Whisper backend: unload"sv);
        }
        releaseGraphResources(GraphRef, IsDebugLog);
        Env.deleteGraph(GraphId);
        if (IsDebugLog) {
          spdlog::info("[WASI-NN][Debug] Whisper backend: unload...Done"sv);
        }
        return ErrNo::Success;
      });
}

Expect<ErrNo> finalizeExecCtx(WasiNNEnvironment &Env,
                              uint32_t ContextId) noexcept {
  return Env.withBackendContext<Backend::Whisper>(
      ContextId, "finalizeExecCtx"sv, [&](Context &CxtRef) -> Expect<ErrNo> {
        const bool IsDebugLog = CxtRef.WhisperConfig.EnableDebugLog;
        if (IsDebugLog) {
          spdlog::info(
              "[WASI-NN][Debug] Whisper backend: finalize_execution_context"sv);
        }
        Env.deleteContext(ContextId);
        if (IsDebugLog) {
          spdlog::info("[WASI-NN][Debug] Whisper backend: "
                       "finalize_execution_context...Done"sv);
        }
        return ErrNo::Success;
      });
}
#endif
} // Namespace WasmEdge::Host::WASINN::Whisper
