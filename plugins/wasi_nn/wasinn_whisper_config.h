// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_whisper.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_WHISPER

#include <string>

namespace WasmEdge::Host::WASINN::Whisper {

void WhisperLogCallback(ggml_log_level LogLevel, const char *LogText,
                        void *UserData);
void setWhisperParams(Context &CxtRef) noexcept;
Expect<ErrNo> parseMetadata(Config &ConfigRef,
                            const std::string &Metadata) noexcept;
Expect<ErrNo> handleTranslationConfig(whisper_context *WhisperCtx,
                                      Config &ConfigRef) noexcept;

} // namespace WasmEdge::Host::WASINN::Whisper

#endif
