// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_bitnet.h"
#include "wasinn_bitnet_inference.h"
#include "wasinn_bitnet_lifecycle.h"
#include "wasinn_bitnet_metadata.h"
#include "wasinn_output.h"
#include "wasinnenv.h"
#include <cstdint>

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET
#include <common.h>
#include <filesystem>
#include <fstream>
#include <llama.h>
#include <sampling.h>
#include <string_view>
#endif

namespace WasmEdge::Host::WASINN::BitNet {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_BITNET

namespace {

// Macro for logging debug message.
#define LOG_DEBUG(Debug, ...)                                                  \
  if (Debug) {                                                                 \
    spdlog::info("[WASI-NN][Debug] BitNet backend: "sv __VA_ARGS__);           \
  }

// Macro for logging info message.
#define LOG_INFO(Info, ...)                                                    \
  if (Info) {                                                                  \
    spdlog::info("[WASI-NN] BitNet backend: "sv __VA_ARGS__);                  \
  }

// Macro for logging warning message.
#define LOG_WARN(...) spdlog::warn("[WASI-NN] BitNet backend: "sv __VA_ARGS__);

// Macro for logging error message.
#define LOG_ERROR(...)                                                         \
  spdlog::error("[WASI-NN] BitNet backend: "sv __VA_ARGS__);

// Macro for logging error message and return.
#define RET_ERROR(Error, ...)                                                  \
  spdlog::error("[WASI-NN] BitNet backend: "sv __VA_ARGS__);                   \
  return Error;

} // namespace

Expect<ErrNo> load(WasiNNEnvironment &Env, Span<const Span<uint8_t>> Builders,
                   [[maybe_unused]] Device Device, uint32_t &GraphId) noexcept {
  if (Builders.empty()) {
    RET_ERROR(ErrNo::InvalidArgument,
              "Invalid builders size, builders size must be > 0.");
  }

  // Add a graph
  auto Graph = Env.newGraphGuard(Backend::BitNet);
  auto &GraphRef = Graph.get<Backend::BitNet>();

  // Initialize the plugin parameters.
  GraphRef.EnableLog = false;
  GraphRef.EnableDebugLog = false;
  const common_params CommonParamsDefault;
  GraphRef.Params = CommonParamsDefault;
  GraphRef.Params.n_keep = 0;
  GraphRef.Params.n_chunks = -1;
  GraphRef.Params.n_parallel = 1;
  GraphRef.Params.grp_attn_n = 1;
  GraphRef.Params.grp_attn_w = 512;
  GraphRef.Params.n_print = -1;
  GraphRef.Params.split_mode = llama_split_mode::LLAMA_SPLIT_MODE_LAYER;
  // Initialize the model parameters.
  llama_model_params ModelParamsDefault = llama_model_default_params();
  GraphRef.Params.n_gpu_layers = ModelParamsDefault.n_gpu_layers;
  GraphRef.Params.mmproj = ""sv;
  GraphRef.Params.warmup = false;
  // Initialize the context parameters.
  llama_context_params ContextParamsDefault = llama_context_default_params();
  GraphRef.Params.n_ctx = ContextParamsDefault.n_ctx;
  GraphRef.Params.n_batch = ContextParamsDefault.n_batch;
  GraphRef.Params.n_ubatch = ContextParamsDefault.n_ubatch;
  GraphRef.Params.cpuparams.n_threads = ContextParamsDefault.n_threads_batch;
  GraphRef.Params.cpuparams_batch.n_threads =
      ContextParamsDefault.n_threads_batch;
  GraphRef.Params.rope_scaling_type = ContextParamsDefault.rope_scaling_type;
  GraphRef.Params.pooling_type = ContextParamsDefault.pooling_type;
  GraphRef.Params.attention_type = ContextParamsDefault.attention_type;
  GraphRef.Params.rope_freq_base = ContextParamsDefault.rope_freq_base;
  GraphRef.Params.rope_freq_scale = ContextParamsDefault.rope_freq_scale;
  GraphRef.Params.yarn_ext_factor = ContextParamsDefault.yarn_ext_factor;
  GraphRef.Params.yarn_attn_factor = ContextParamsDefault.yarn_attn_factor;
  GraphRef.Params.yarn_beta_fast = ContextParamsDefault.yarn_beta_fast;
  GraphRef.Params.yarn_beta_slow = ContextParamsDefault.yarn_beta_slow;
  GraphRef.Params.yarn_orig_ctx = ContextParamsDefault.yarn_orig_ctx;
  GraphRef.Params.defrag_thold = ContextParamsDefault.defrag_thold;
  GraphRef.Params.cb_eval = ContextParamsDefault.cb_eval;
  GraphRef.Params.cb_eval_user_data = ContextParamsDefault.cb_eval_user_data;
  GraphRef.Params.embedding = ContextParamsDefault.embeddings;
  GraphRef.Params.no_kv_offload = !ContextParamsDefault.offload_kqv;
  GraphRef.Params.flash_attn = ContextParamsDefault.flash_attn;
  GraphRef.Params.no_perf = ContextParamsDefault.no_perf;

  // Initialize the sampling parameters.
  const common_sampler_params SamplerParamsDefault;
  GraphRef.Params.sparams = SamplerParamsDefault;

  // Initialize the config parameters.
  GraphRef.Conf.StreamStdout = false;
  GraphRef.Conf.EmbdNormalize =
      static_cast<EmbdNormalizeType>(CommonParamsDefault.embd_normalize);
  GraphRef.Conf.NPredict = ContextParamsDefault.n_ctx;
  GraphRef.Conf.ReversePrompt = ""sv;

  // Set llama log callback.
  llama_log_set(llamaLogCallback, &GraphRef);
  LOG_DEBUG(GraphRef.EnableDebugLog, "load start."sv)

  // If the graph builder length > 1, the data of builder[1] is the metadata.
  if (Builders.size() > 1) {
    const std::string Metadata = asString(Builders[1]);
    // Ignore context or model updates when initializing the graph.
    auto Res = parseMetadata(GraphRef, GraphRef.Conf, Metadata);
    if (Res != ErrNo::Success) {
      RET_ERROR(Res, "load: Failed to parse metadata."sv);
    }
  }

  LOG_INFO(GraphRef.EnableLog, "LLAMA_COMMIT {}"sv, LLAMA_COMMIT)
  LOG_INFO(GraphRef.EnableLog, "LLAMA_BUILD_NUMBER {}"sv, LLAMA_BUILD_NUMBER)

  LOG_DEBUG(GraphRef.EnableDebugLog, "load: handling model path."sv)
  const auto &Weight = Builders[0];
  const std::string_view BinModel = asStringView(Weight);

  if (BinModel.substr(0, 8) == "preload:"sv) {
    GraphRef.Params.model = std::string(BinModel.substr(8));
  } else {
    LOG_DEBUG(GraphRef.EnableDebugLog,
              "load: Model path not found in nn-preload, write model into "sv
              "a tmpfile."sv)
    GraphRef.Params.model = "bitnet-model.bin"sv;
    std::ofstream TempFile(GraphRef.Params.model,
                           std::ios::out | std::ios::binary | std::ios::trunc);
    if (!TempFile) {
      RET_ERROR(ErrNo::InvalidArgument, "Failed to create temp model file."sv)
    }
    TempFile.write(BinModel.data(), BinModel.size());
    TempFile.close();
    LOG_DEBUG(GraphRef.EnableDebugLog,
              "load: Write model into a tmpfile...Done"sv)
  }
  LOG_DEBUG(GraphRef.EnableDebugLog, "load: handling model path...Done"sv)

  // Check if the model exists.
  if (!std::filesystem::exists(
          std::filesystem::u8path(GraphRef.Params.model))) {
    RET_ERROR(ErrNo::ModelNotFound,
              "load: Model file not found at path: '{}'."sv,
              GraphRef.Params.model)
  }

  LOG_INFO(GraphRef.EnableLog, "load: Loading model from '{}'."sv,
           GraphRef.Params.model)

  // Initialize model parameters.
  LOG_DEBUG(GraphRef.EnableDebugLog,
            "load: initialize model with given parameters."sv)

  llama_backend_init();
  llama_numa_init(GraphRef.Params.numa);

  // Initialize the llama model and context.
  common_init_result LlamaInit = common_init_from_params(GraphRef.Params);
  GraphRef.LlamaModel.reset(LlamaInit.model);
  GraphRef.LlamaContext.reset(LlamaInit.context);

  if (GraphRef.LlamaModel == nullptr) {
    RET_ERROR(ErrNo::InvalidArgument, "load: Unable to init model."sv)
  }
  if (GraphRef.LlamaContext == nullptr) {
    RET_ERROR(ErrNo::InvalidArgument, "load: Unable to init context."sv)
  }

  LOG_DEBUG(GraphRef.EnableDebugLog,
            "load: initialize model with given parameters...Done"sv)

  // Store the loaded graph.
  GraphId = Graph.commit();

  LOG_DEBUG(GraphRef.EnableDebugLog, "load...Done"sv)
  return ErrNo::Success;
}

Expect<ErrNo> initExecCtx(WasiNNEnvironment &Env, uint32_t GraphId,
                          uint32_t &ContextId) noexcept {
  auto GraphInst = Env.getBackendGraphOrError<Backend::BitNet>(
      GraphId, "init_execution_context"sv);
  if (!GraphInst) {
    return GraphInst.error();
  }
  auto &GraphRef = **GraphInst;
  LOG_DEBUG(GraphRef.EnableDebugLog, "init_execution_context"sv)
  auto Context = Env.newContextGuard(GraphId);
  auto &CxtRef = Context.get<Backend::BitNet>();

  LOG_INFO(GraphRef.EnableLog, "llama_system_info: {}"sv,
           llama_print_system_info())

  // Allocate the batch for input string prompt tokens.
  CxtRef.LlamaBatch.reset(allocBatch(GraphRef.Params.n_batch));
  CxtRef.CurrentBatchSize = GraphRef.Params.n_batch;

  // Allocate the batch for single-token output sampling.
  CxtRef.OutputBatch.reset(allocBatch(1));

  // Allocate the sampler
  CxtRef.LlamaSampler.reset(
      common_sampler_init(GraphRef.LlamaModel.get(), GraphRef.Params.sparams));
  if (CxtRef.LlamaSampler == nullptr) {
    releaseContextResources(CxtRef, GraphRef.EnableDebugLog);
    RET_ERROR(ErrNo::InvalidArgument, "initExecCtx: unable to init sampler."sv)
  }

  ContextId = Context.commit();
  LOG_DEBUG(GraphRef.EnableDebugLog, "initExecCtx...Done"sv)
  return ErrNo::Success;
}

Expect<ErrNo> setInput(WasiNNEnvironment &Env, uint32_t ContextId,
                       uint32_t Index, const TensorData &Tensor) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::BitNet>(
      ContextId, "set_input"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();
  LOG_DEBUG(GraphRef.EnableDebugLog, "set_input"sv)

  // Handle Metadata at Index 1
  if (Index == 1) {
    LOG_DEBUG(GraphRef.EnableDebugLog, "setInput: found Metadata, processing"sv)
    bool IsModelUpdated = false;
    bool IsContextUpdated = false;
    bool IsSamplerUpdated = false;
    const auto Metadata = asString(Tensor.Tensor);

    auto Res = parseMetadata(GraphRef, CxtRef.Conf, Metadata, &IsModelUpdated,
                             &IsContextUpdated, &IsSamplerUpdated);
    if (Res != ErrNo::Success) {
      RET_ERROR(Res, "setInput: failed to parse metadata."sv)
    }

    if (IsModelUpdated || GraphRef.LlamaModel == nullptr) {
      // The llama model may be nullptr if set_input with updated model params
      // last time. Therefore besides the model params updated, we should
      // reload the llama model if the model is nullptr.
      LOG_INFO(GraphRef.EnableLog,
               "setInput: Reloading model due to parameter change"sv)

      // Prepare model parameters for the reload.
      llama_model_params ModelParams = llama_model_default_params();
      ModelParams.n_gpu_layers =
          static_cast<int32_t>(GraphRef.Params.n_gpu_layers);
      ModelParams.main_gpu = static_cast<int32_t>(GraphRef.Params.main_gpu);

      // Free all resources that depend on the old model.
      GraphRef.LlamaModel.reset();

      // Due to the model change, the context and sampler should also be
      // reloaded. The new context and sampler will be created in the next
      // block.
      GraphRef.LlamaContext.reset();
      CxtRef.LlamaSampler.reset();

      // Attempt to load the model from file with new parameters.
      GraphRef.LlamaModel.reset(llama_load_model_from_file(
          GraphRef.Params.model.c_str(), ModelParams));
      if (GraphRef.LlamaModel == nullptr) {
        Env.setContextGraphInvalid(ContextId);
        RET_ERROR(ErrNo::InvalidArgument, "setInput: unable to init model."sv)
      }
    }

    // Reload context if its parameters changed OR if it was cleared by a model
    // reload.
    if (IsContextUpdated || GraphRef.LlamaContext == nullptr) {
      LOG_INFO(GraphRef.EnableLog,
               "setInput: Reloading llama context due to parameter change."sv)
      GraphRef.LlamaContext.reset();
      llama_context_params CtxParams =
          common_context_params_to_llama(GraphRef.Params);
      GraphRef.LlamaContext.reset(
          llama_new_context_with_model(GraphRef.LlamaModel.get(), CtxParams));
      if (GraphRef.LlamaContext == nullptr) {
        Env.setContextGraphInvalid(ContextId);
        RET_ERROR(ErrNo::InvalidArgument, "setInput: unable to init context."sv)
      }
    }

    // Re-initialize sampler if its parameters changed OR if it was cleared.
    if (IsSamplerUpdated || CxtRef.LlamaSampler == nullptr) {
      LOG_INFO(GraphRef.EnableLog,
               "setInput: Re-initializing sampler due to parameter change."sv);
      CxtRef.LlamaSampler.reset(common_sampler_init(GraphRef.LlamaModel.get(),
                                                    GraphRef.Params.sparams));
      if (CxtRef.LlamaSampler == nullptr) {
        Env.setContextGraphInvalid(ContextId);
        RET_ERROR(ErrNo::InvalidArgument, "setInput: unable to init sampler."sv)
      }
    }

    // Re-allocate batch if the batch size changed.
    if (CxtRef.CurrentBatchSize != GraphRef.Params.n_batch) {
      LOG_INFO(GraphRef.EnableLog,
               "Re-allocating batch due to n_batch change.");
      CxtRef.LlamaBatch.reset(allocBatch(GraphRef.Params.n_batch));
      if (!CxtRef.LlamaBatch.get().token) {
        RET_ERROR(ErrNo::InvalidArgument, "Failed to re-allocate llama_batch.");
      }
      CxtRef.CurrentBatchSize = GraphRef.Params.n_batch;
    }

    Env.setContextGraphReady(ContextId);
    LOG_DEBUG(GraphRef.EnableDebugLog,
              "setInput: metadata processing...Done"sv);
    return ErrNo::Success;
  }

  if (Index != 0) {
    RET_ERROR(ErrNo::InvalidArgument,
              "Only prompt (index 0) and metadata (index 1) are supported.");
  }

  // Check the graph is valid after reloading during previous set_input.
  if (!Env.isContextGraphReady(ContextId)) {
    RET_ERROR(
        ErrNo::InvalidArgument,
        "setInput: Graph is invalid. Please reload again by passing metadata "sv
        "in set_input or unload graph."sv)
  }

  LOG_DEBUG(GraphRef.EnableLog, "setInput: Clearing KV cache for new prompt."sv)
  llama_kv_cache_clear(GraphRef.LlamaContext.get());
  LOG_DEBUG(GraphRef.EnableLog,
            "setInput: Clearing KV cache for new prompt...done"sv)

  // Check tensor type.
  if (Tensor.RType != TensorType::U8) {
    RET_ERROR(ErrNo::InvalidArgument,
              "Input tensor must be a UTF-8 string (U8).");
  }

  // Tokenize the new prompt.
  const auto Prompt = asString(Tensor.Tensor);
  LOG_DEBUG(GraphRef.EnableDebugLog, "setInput: tokenize text prompt"sv)
  CxtRef.LlamaInputs =
      common_tokenize(GraphRef.LlamaContext.get(), Prompt,
                      llama_add_bos_token(GraphRef.LlamaModel.get()), true);
  LOG_DEBUG(GraphRef.EnableDebugLog, "setInput: tokenize text prompt...Done"sv)

  // Get the number of input tokens (for the metadata).
  CxtRef.LlamaNInputs = CxtRef.LlamaInputs.size();

  // Reset state for the compute loop.
  CxtRef.ComputeSingleStarted = false;
  LOG_DEBUG(GraphRef.EnableDebugLog, "setInput...Done"sv)
  return ErrNo::Success;
}

Expect<ErrNo> getOutput(WasiNNEnvironment &Env, uint32_t ContextId,
                        uint32_t Index, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten) noexcept {
  return Env.withBackendState<Backend::BitNet>(
      ContextId, "get_output"sv,
      [&](Context &CxtRef, Graph &GraphRef) -> Expect<ErrNo> {
        LOG_DEBUG(GraphRef.EnableDebugLog, "getOutput: with Index {}"sv, Index)

        // Handle Metadata Output at Index 1
        if (Index == 1) {
          const std::string Metadata = buildOutputMetadata(CxtRef);

          LOG_DEBUG(GraphRef.EnableDebugLog,
                    "getOutput: Metadata (Index 1)...Done"sv)
          return copyStringToBuffer(Metadata, OutBuffer, BytesWritten);
        }

        LOG_DEBUG(GraphRef.EnableDebugLog, "getOutput: Text (Index 0)...Done"sv)
        return copyBytesToBuffer(CxtRef.LlamaOutputs, OutBuffer, BytesWritten);
      });
}

Expect<ErrNo> compute(WasiNNEnvironment &Env, uint32_t ContextId) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::BitNet>(ContextId,
                                                                  "compute"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();
  LOG_DEBUG(GraphRef.EnableDebugLog, "compute"sv);

  // Clear the context and reset the sampler.
  clearContext(GraphRef, CxtRef);

  if (GraphRef.Params.embedding) {
    return getEmbedding(GraphRef, CxtRef);
  }

  // Evaluate the input tokens.
  auto ReturnCode = evaluateInput(GraphRef, CxtRef, "compute"sv);
  if (ReturnCode != ErrNo::Success) {
    return ReturnCode;
  }

  // Main prediction loop.
  LOG_DEBUG(GraphRef.EnableDebugLog, "compute: enter main prediction loop"sv)
  int64_t NPredict = CxtRef.Conf.NPredict;
  if (NPredict < 0) {
    NPredict = INT32_MAX;
  }

  while (NPredict > 0) {
    ReturnCode = sampleOutput(GraphRef, CxtRef);
    if (ReturnCode != ErrNo::Success) {
      break;
    }
    NPredict--;
  }

  if (ReturnCode == ErrNo::EndOfSequence || ReturnCode == ErrNo::ContextFull) {
    LOG_INFO(GraphRef.EnableLog, "compute finished with status: {}."sv,
             static_cast<uint32_t>(ReturnCode))
    return ErrNo::Success;
  }

  LOG_DEBUG(GraphRef.EnableDebugLog,
            "compute: enter main prediction loop...Done"sv)

  if (GraphRef.EnableLog) {
    common_perf_print(GraphRef.LlamaContext.get(), CxtRef.LlamaSampler.get());
  }

  LOG_DEBUG(GraphRef.EnableDebugLog, "compute...Done"sv)
  return ReturnCode;
}

Expect<ErrNo> getOutputSingle(WasiNNEnvironment &Env, uint32_t ContextId,
                              uint32_t Index, Span<uint8_t> OutBuffer,
                              uint32_t &BytesWritten) noexcept {
  return Env.withBackendState<Backend::BitNet>(
      ContextId, "get_output_single"sv,
      [&](Context &CxtRef, Graph &GraphRef) -> Expect<ErrNo> {
        LOG_DEBUG(GraphRef.EnableDebugLog, "getOutputSingle: with Index {}"sv,
                  Index)

        // Metadata Output at Index 1
        if (Index == 1) {
          const std::string Metadata = buildOutputMetadata(CxtRef);

          LOG_DEBUG(GraphRef.EnableDebugLog,
                    "getOutputSingle: Metadata (Index 1)...Done"sv)
          return copyStringToBuffer(Metadata, OutBuffer, BytesWritten);
        }

        if (CxtRef.LlamaOutputTokens.empty()) {
          BytesWritten = 0;
          return ErrNo::Success;
        }

        const std::string LastTokenStr = common_token_to_piece(
            GraphRef.LlamaContext.get(), CxtRef.LlamaOutputTokens.back());

        LOG_DEBUG(GraphRef.EnableDebugLog,
                  "getOutputSingle: Text (Index 0)...Done"sv)
        return copyStringToBuffer(LastTokenStr, OutBuffer, BytesWritten);
      });
}

Expect<ErrNo> computeSingle(WasiNNEnvironment &Env,
                            uint32_t ContextId) noexcept {
  auto State = Env.getBackendContextGraphOrError<Backend::BitNet>(
      ContextId, "compute_single"sv);
  if (!State) {
    return State.error();
  }
  auto &CxtRef = State->context();
  auto &GraphRef = State->graph();
  LOG_DEBUG(GraphRef.EnableDebugLog, "compute_single"sv)

  auto ReturnCode = ErrNo::Success;
  if (!CxtRef.ComputeSingleStarted) {
    // Clear the context and reset the sampler.
    clearContext(GraphRef, CxtRef);
    ReturnCode = evaluateInput(GraphRef, CxtRef, "compute_single"sv);
    if (ReturnCode != ErrNo::Success) {
      return ReturnCode;
    }

    CxtRef.ComputeSingleStarted = true;
  }

  // Main prediction process.
  LOG_DEBUG(GraphRef.EnableDebugLog,
            "computeSingle: enter main prediction process"sv)
  ReturnCode = sampleOutput(GraphRef, CxtRef, /* IsSingleTokenMode */ true);
  if (ReturnCode != ErrNo::Success) {
    CxtRef.ComputeSingleStarted = false;
  }
  LOG_DEBUG(GraphRef.EnableDebugLog,
            "computeSingle: enter main prediction process...Done"sv)
  // End of main predict process.

  LOG_DEBUG(GraphRef.EnableDebugLog, "computeSingle...Done"sv)
  return ReturnCode;
}

Expect<ErrNo> finiSingle(WasiNNEnvironment &Env, uint32_t ContextId) noexcept {
  return Env.withBackendState<Backend::BitNet>(
      ContextId, "fini_single"sv,
      [](Context &CxtRef, Graph &GraphRef) -> Expect<ErrNo> {
        LOG_DEBUG(GraphRef.EnableDebugLog, "fini_single"sv);

        if (GraphRef.EnableLog) {
          common_perf_print(GraphRef.LlamaContext.get(),
                            CxtRef.LlamaSampler.get());
        }

        // Reset the llama sampler.
        common_sampler_reset(CxtRef.LlamaSampler.get());

        // Clear the outputs.
        LOG_DEBUG(GraphRef.EnableDebugLog,
                  "finiSingle: clear the previous output and tokens"sv)
        CxtRef.LlamaOutputs.clear();
        CxtRef.LlamaOutputTokens.clear();
        LOG_DEBUG(GraphRef.EnableDebugLog,
                  "finiSingle: clear the previous output and tokens...Done"sv)

        CxtRef.NPos = 0;
        CxtRef.ComputeSingleStarted = false;

        LOG_DEBUG(GraphRef.EnableDebugLog, "finiSingle...Done"sv)
        return ErrNo::Success;
      });
}

Expect<ErrNo> unload(WasiNNEnvironment &Env, uint32_t GraphId) noexcept {
  return Env.withBackendGraph<Backend::BitNet>(
      GraphId, "unload"sv, [&](Graph &GraphRef) -> Expect<ErrNo> {
        const bool IsDebugLog = GraphRef.EnableDebugLog;

        LOG_DEBUG(IsDebugLog, "unload"sv)

        releaseGraphResources(GraphRef, IsDebugLog);
        Env.deleteGraph(GraphId);

        LOG_DEBUG(IsDebugLog, "unload...Done"sv)
        return ErrNo::Success;
      });
}

Expect<ErrNo> finalizeExecCtx(WasiNNEnvironment &Env,
                              uint32_t ContextId) noexcept {
  return Env.withBackendState<Backend::BitNet>(
      ContextId, "finalizeExecCtx"sv,
      [&](Context &CxtRef, Graph &GraphRef) -> Expect<ErrNo> {
        LOG_DEBUG(GraphRef.EnableDebugLog, "finalizeExecCtx"sv)

        releaseContextResources(CxtRef, GraphRef.EnableDebugLog);
        Env.deleteContext(ContextId);

        LOG_DEBUG(GraphRef.EnableDebugLog, "finalizeExecCtx...Done"sv)
        return ErrNo::Success;
      });
}
#endif
} // namespace WasmEdge::Host::WASINN::BitNet
