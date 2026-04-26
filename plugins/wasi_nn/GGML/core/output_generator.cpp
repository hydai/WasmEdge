// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "ggml_core.h"
#include "wasinn_output.h"

namespace WasmEdge::Host::WASINN::GGML {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
namespace {
// Generate output metadata.
std::string buildOutputMetadata(Context &CxtRef) noexcept {
  return fmt::format(R"({{"input_tokens": {}, )"
                     R"("output_tokens": {}, )"
                     R"("llama_build_number": {}, )"
                     R"("llama_commit": "{}"}})"sv,
                     CxtRef.LlamaNInputs, CxtRef.LlamaOutputTokens.size(),
                     LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
}
} // namespace

Expect<ErrNo> getOutputSingle(WasiNNEnvironment &Env, uint32_t ContextId,
                              uint32_t Index, Span<uint8_t> OutBuffer,
                              uint32_t &BytesWritten) noexcept {
  return Env.withBackendState<Backend::GGML>(
      ContextId, "get_output_single"sv,
      [&](Context &CxtRef, Graph &GraphRef) -> Expect<ErrNo> {
        LOG_DEBUG(GraphRef.EnableDebugLog, "getOutputSingle: with Index {}"sv,
                  Index)

        // Use index 1 for the metadata of the outputs.
        if (Index == 1) {
          std::string Metadata = buildOutputMetadata(CxtRef);
          LOG_DEBUG(GraphRef.EnableDebugLog,
                    "getOutputSingle: with Index {} a.k.a Metadata...Done"sv,
                    Index)
          return copyStringToBuffer(Metadata, OutBuffer, BytesWritten);
        }

        std::string LastToken = common_token_to_piece(
            GraphRef.LlamaContext.get(), CxtRef.LlamaOutputTokens.back());
        LOG_DEBUG(GraphRef.EnableDebugLog,
                  "getOutputSingle: with Index {}...Done"sv, Index)
        return copyStringToBuffer(LastToken, OutBuffer, BytesWritten);
      });
}

Expect<ErrNo> getOutput(WasiNNEnvironment &Env, uint32_t ContextId,
                        uint32_t Index, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten) noexcept {
  return Env.withBackendState<Backend::GGML>(
      ContextId, "get_output"sv,
      [&](Context &CxtRef, Graph &GraphRef) -> Expect<ErrNo> {
        LOG_DEBUG(GraphRef.EnableDebugLog, "getOutput: with Index {}"sv, Index)

        // Use index 1 for the metadata of the outputs.
        if (Index == 1) {
          std::string Metadata = buildOutputMetadata(CxtRef);
          LOG_DEBUG(GraphRef.EnableDebugLog,
                    "getOutput: with Index {} a.k.a Metadata ...Done"sv, Index)
          return copyStringToBuffer(Metadata, OutBuffer, BytesWritten);
        }

        LOG_DEBUG(GraphRef.EnableDebugLog, "getOutput: with Index {}...Done"sv,
                  Index)
        return copyBytesToBuffer(CxtRef.LlamaOutputs, OutBuffer, BytesWritten);
      });
}

#endif
} // namespace WasmEdge::Host::WASINN::GGML
