// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_whisper.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_WHISPER

#include <string>
#include <vector>

namespace WasmEdge::Host::WASINN::Whisper {

bool outputSrt(WasiNNEnvironment &Env, whisper_context *Ctx,
               const std::string &Fname, const Config &Params,
               const std::vector<std::vector<float>> &PCMF32s);
bool outputLrc(WasiNNEnvironment &Env, whisper_context *Ctx,
               const std::string &Fname, const Config &Params,
               const std::vector<std::vector<float>> &PCMF32s);
bool outputJson(WasiNNEnvironment &Env, whisper_context *Ctx,
                const std::string &Fname, const Config &Params,
                const std::vector<std::vector<float>> &PCMF32s, bool Full);
void WhisperOutputSegmentCallback(struct whisper_context *WhisperCtx,
                                  struct whisper_state *State, int NewN,
                                  void *UserData);

} // namespace WasmEdge::Host::WASINN::Whisper

#endif
