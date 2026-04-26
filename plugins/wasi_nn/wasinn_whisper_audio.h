// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "wasinn_whisper.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_WHISPER

#include <string_view>
#include <vector>

namespace WasmEdge::Host::WASINN::Whisper {

bool checkAudioRIFF(std::string_view Buf, std::string_view Format);
bool loadWAV(Span<const uint8_t> Buf, std::vector<float> &PCMF32,
             std::vector<std::vector<float>> &PCMF32s, bool Stereo);

} // namespace WasmEdge::Host::WASINN::Whisper

#endif
