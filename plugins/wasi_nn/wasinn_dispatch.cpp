// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_dispatch.h"

#include "wasinn_backend.h"
#include "wasinnenv.h"

#include "common/spdlog.h"

#include <string>
#include <string_view>
#include <utility>

namespace WasmEdge {
namespace Host {
namespace WASINN {
namespace {
using namespace std::literals;

constexpr std::string_view GetOutputSingleUnsupported =
    "[WASI-NN] get_output_single: Only GGML and BitNet backends support "
    "get_output_single."sv;
constexpr std::string_view ComputeSingleUnsupported =
    "[WASI-NN] compute_single: Only GGML and BitNet backends support "
    "compute_single."sv;
constexpr std::string_view FiniSingleUnsupported =
    "[WASI-NN] fini_single: Only GGML and BitNet backends support "
    "fini_single."sv;

void reportUnknownBackend(Backend BE) noexcept {
  spdlog::error("[WASI-NN] Unknown backend {}."sv, static_cast<uint32_t>(BE));
}

const BackendOperations *getBackendOperationsOrReport(Backend BE) noexcept {
  const auto *Ops = getBackendOperations(BE);
  if (Ops == nullptr) {
    reportUnknownBackend(BE);
  }
  return Ops;
}

template <typename Handler>
Expected<Handler, ErrNo>
requireBackendOperation(Backend BE,
                        Handler BackendOperations::*Operation) noexcept {
  const auto *Ops = getBackendOperationsOrReport(BE);
  if (Ops == nullptr) {
    return Unexpected<ErrNo>(ErrNo::InvalidEncoding);
  }

  auto HandlerPtr = Ops->*Operation;
  if (HandlerPtr == nullptr) {
    reportUnknownBackend(BE);
    return Unexpected<ErrNo>(ErrNo::InvalidEncoding);
  }
  return HandlerPtr;
}

template <typename Handler>
Expected<Handler, ErrNo>
requireBackendFeature(Backend BE, Handler BackendOperations::*Operation,
                      std::string_view UnsupportedMessage) noexcept {
  const auto *Ops = getBackendOperationsOrReport(BE);
  if (Ops == nullptr) {
    return Unexpected<ErrNo>(ErrNo::InvalidEncoding);
  }

  auto HandlerPtr = Ops->*Operation;
  if (HandlerPtr == nullptr) {
    spdlog::error("{}"sv, UnsupportedMessage);
    return Unexpected<ErrNo>(ErrNo::InvalidArgument);
  }
  return HandlerPtr;
}

Expected<Backend, ErrNo>
getReadyGraphBackendForInitExecCtx(WasiNNEnvironment &Env,
                                   uint32_t GraphId) noexcept {
  auto BE = Env.getGraphBackend(GraphId);
  if (!BE || Env.isGraphFinalized(GraphId)) {
    spdlog::error("[WASI-NN] init_execution_context: Graph ID {} does not "sv
                  "exist or is unloaded."sv,
                  GraphId);
    return Unexpected<ErrNo>(ErrNo::InvalidArgument);
  }
  if (!Env.isGraphReady(GraphId)) {
    spdlog::error("[WASI-NN] init_execution_context: Graph ID {} is invalid. "sv
                  "Please reload or unload this graph."sv,
                  GraphId);
    return Unexpected<ErrNo>(ErrNo::InvalidArgument);
  }
  return *BE;
}

Expected<Backend, ErrNo>
getContextBackendOrReport(WasiNNEnvironment &Env, uint32_t ContextId,
                          std::string_view FunctionName) noexcept {
  auto BE = Env.getContextBackend(ContextId);
  if (!BE) {
    spdlog::error("[WASI-NN] {}: Context ID {} does not exist."sv, FunctionName,
                  ContextId);
    return Unexpected<ErrNo>(ErrNo::InvalidArgument);
  }
  return *BE;
}

Expected<Backend, ErrNo>
getReadyContextBackendOrReport(WasiNNEnvironment &Env, uint32_t ContextId,
                               std::string_view FunctionName) noexcept {
  auto BE = Env.getContextBackend(ContextId);
  if (!BE || !Env.isContextReady(ContextId)) {
    spdlog::error("[WASI-NN] {}: Context ID {} does not exist."sv, FunctionName,
                  ContextId);
    return Unexpected<ErrNo>(ErrNo::InvalidArgument);
  }
  return *BE;
}

Expected<Backend, ErrNo>
getReadyContextBackendWithGraph(WasiNNEnvironment &Env, uint32_t ContextId,
                                std::string_view FunctionName) noexcept {
  auto BE = getReadyContextBackendOrReport(Env, ContextId, FunctionName);
  if (!BE) {
    return Unexpected<ErrNo>(BE.error());
  }

  auto GraphId = Env.getContextGraphId(ContextId);
  assuming(GraphId.has_value());
  if (!GraphId || !Env.isGraphReady(*GraphId)) {
    spdlog::error("[WASI-NN] {}: Graph ID {} for context ID {} does not "sv
                  "exist or has released."sv,
                  FunctionName, GraphId.value_or(0), ContextId);
    return Unexpected<ErrNo>(ErrNo::InvalidArgument);
  }
  return *BE;
}

} // namespace

namespace Dispatch {

Expect<ErrNo> load(WasiNNEnvironment &Env, Span<const Span<uint8_t>> Builders,
                   Backend BE, Device Target, uint32_t &GraphId) {
  return Env.withGraphOperation([&]() -> Expect<ErrNo> {
    auto Handler = requireBackendOperation(BE, &BackendOperations::Load);
    if (!Handler) {
      return Handler.error();
    }
    return Handler.value()(Env, Builders, Target, GraphId);
  });
}

Expect<ErrNo> loadPreloadedModel(WasiNNEnvironment &Env,
                                 Span<const uint8_t> Name, uint32_t &GraphId,
                                 std::vector<uint8_t> Config) {
  auto ModelName = asString(Name);
  return Env.mdBuild(ModelName, GraphId, load, std::move(Config));
}

Expect<ErrNo> initExecCtx(WasiNNEnvironment &Env, uint32_t GraphId,
                          uint32_t &ContextId) {
  return Env.withGraphOperation([&]() -> Expect<ErrNo> {
    auto BE = getReadyGraphBackendForInitExecCtx(Env, GraphId);
    if (!BE) {
      return BE.error();
    }

    auto Handler =
        requireBackendOperation(*BE, &BackendOperations::InitExecCtx);
    if (!Handler) {
      return Handler.error();
    }
    return Handler.value()(Env, GraphId, ContextId);
  });
}

Expect<ErrNo> setInput(WasiNNEnvironment &Env, uint32_t ContextId,
                       uint32_t Index, const TensorData &Tensor) {
  return Env.withGraphOperation([&]() -> Expect<ErrNo> {
    auto BE = getReadyContextBackendOrReport(Env, ContextId, "set_input"sv);
    if (!BE) {
      return BE.error();
    }

    auto Handler = requireBackendOperation(*BE, &BackendOperations::SetInput);
    if (!Handler) {
      return Handler.error();
    }
    return Handler.value()(Env, ContextId, Index, Tensor);
  });
}

Expect<ErrNo> getOutput(WasiNNEnvironment &Env, uint32_t ContextId,
                        uint32_t Index, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten) {
  return Env.withGraphOperation([&]() -> Expect<ErrNo> {
    auto BE = getReadyContextBackendOrReport(Env, ContextId, "get_output"sv);
    if (!BE) {
      return BE.error();
    }

    auto Handler = requireBackendOperation(*BE, &BackendOperations::GetOutput);
    if (!Handler) {
      return Handler.error();
    }
    return Handler.value()(Env, ContextId, Index, OutBuffer, BytesWritten);
  });
}

Expect<ErrNo> getOutputSingle(WasiNNEnvironment &Env, uint32_t ContextId,
                              uint32_t Index, Span<uint8_t> OutBuffer,
                              uint32_t &BytesWritten) {
  return Env.withGraphOperation([&]() -> Expect<ErrNo> {
    auto BE =
        getReadyContextBackendOrReport(Env, ContextId, "get_output_single"sv);
    if (!BE) {
      return BE.error();
    }

    auto Handler = requireBackendFeature(
        *BE, &BackendOperations::GetOutputSingle, GetOutputSingleUnsupported);
    if (!Handler) {
      return Handler.error();
    }
    return Handler.value()(Env, ContextId, Index, OutBuffer, BytesWritten);
  });
}

Expect<ErrNo> compute(WasiNNEnvironment &Env, uint32_t ContextId) {
  return Env.withGraphOperation([&]() -> Expect<ErrNo> {
    auto BE = getReadyContextBackendWithGraph(Env, ContextId, "compute"sv);
    if (!BE) {
      return BE.error();
    }

    auto Handler = requireBackendOperation(*BE, &BackendOperations::Compute);
    if (!Handler) {
      return Handler.error();
    }
    return Handler.value()(Env, ContextId);
  });
}

Expect<ErrNo> computeSingle(WasiNNEnvironment &Env, uint32_t ContextId) {
  return Env.withGraphOperation([&]() -> Expect<ErrNo> {
    auto BE =
        getReadyContextBackendWithGraph(Env, ContextId, "compute_single"sv);
    if (!BE) {
      return BE.error();
    }

    auto Handler = requireBackendFeature(*BE, &BackendOperations::ComputeSingle,
                                         ComputeSingleUnsupported);
    if (!Handler) {
      return Handler.error();
    }
    return Handler.value()(Env, ContextId);
  });
}

Expect<ErrNo> finiSingle(WasiNNEnvironment &Env, uint32_t ContextId) {
  return Env.withGraphOperation([&]() -> Expect<ErrNo> {
    auto BE = getReadyContextBackendOrReport(Env, ContextId, "fini_single"sv);
    if (!BE) {
      return BE.error();
    }

    auto Handler = requireBackendFeature(*BE, &BackendOperations::FiniSingle,
                                         FiniSingleUnsupported);
    if (!Handler) {
      return Handler.error();
    }
    return Handler.value()(Env, ContextId);
  });
}

Expect<ErrNo> unload(WasiNNEnvironment &Env, uint32_t GraphId) {
  return Env.withGraphOperation([&]() -> Expect<ErrNo> {
    auto BE = Env.getGraphBackend(GraphId);
    if (!BE) {
      spdlog::error("[WASI-NN] unload: GraphId {} does not exist."sv, GraphId);
      return ErrNo::InvalidArgument;
    }

    auto Handler =
        requireBackendFeature(*BE, &BackendOperations::Unload,
                              "[WASI-NN] unload: Only GGML, Whisper, ChatTTS "
                              "and BitNet backends support unload."sv);
    if (!Handler) {
      return Handler.error();
    }
    auto Res = Handler.value()(Env, GraphId);
    if (Res && Res.value() == ErrNo::Success) {
      Env.mdRemoveById(GraphId);
    }
    return Res;
  });
}

Expect<ErrNo> finalizeExecCtx(WasiNNEnvironment &Env, uint32_t ContextId) {
  return Env.withGraphOperation([&]() -> Expect<ErrNo> {
    auto BE = getContextBackendOrReport(Env, ContextId,
                                        "finalize_execution_context"sv);
    if (!BE) {
      return BE.error();
    }

    auto Handler = requireBackendFeature(
        *BE, &BackendOperations::FinalizeExecCtx,
        "[WASI-NN] finalize_execution_context: Only GGML, BitNet and Whisper "
        "backends support finalize_execution_context."sv);
    if (!Handler) {
      return Handler.error();
    }
    return Handler.value()(Env, ContextId);
  });
}

} // namespace Dispatch

} // namespace WASINN
} // namespace Host
} // namespace WasmEdge
