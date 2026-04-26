// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_whisper_audio.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_WHISPER

#define DR_WAV_IMPLEMENTATION
#include "common/spdlog.h"
#include <examples/dr_wav.h>

#include <cstdint>
#include <string_view>
#include <vector>

using namespace std::literals;

namespace WasmEdge::Host::WASINN::Whisper {
bool checkAudioRIFF(const std::string_view Buf, const std::string_view Format) {
  if (Buf.size() < 12 || Buf.substr(0, 4) != "RIFF"sv) {
    return false;
  }
  if (Buf.substr(8, 4) != Format) {
    return false;
  }
  uint32_t ChunkSize = *reinterpret_cast<const uint32_t *>(Buf.data() + 4);
  if (ChunkSize + 8 != Buf.size()) {
    return false;
  }
  return true;
}

bool loadWAV(Span<const uint8_t> Buf, std::vector<float> &PCMF32,
             std::vector<std::vector<float>> &PCMF32s, bool Stereo) {
  // Not to use the helper function in examples of whisper.cpp to prevent from
  // copy.
  drwav WAV;
  const uint32_t ConstSampleRate = 16000;

  if (!drwav_init_memory(&WAV, Buf.data(), Buf.size(), nullptr)) {
    spdlog::error("[WASI-NN] Whisper backend: load WAV failed."sv);
    return false;
  }

  if (WAV.channels != 1 && WAV.channels != 2) {
    spdlog::error("[WASI-NN] Whisper backend: WAV must be mono or stereo."sv);
    drwav_uninit(&WAV);
    return false;
  }

  if (WAV.sampleRate != ConstSampleRate) {
    spdlog::error("[WASI-NN] Whisper backend: WAV must be {} kHz."sv,
                  ConstSampleRate / 1000);
    drwav_uninit(&WAV);
    return false;
  }

  if (WAV.bitsPerSample != 16) {
    spdlog::error("[WASI-NN] Whisper backend: WAV must be 16-bit."sv);
    drwav_uninit(&WAV);
    return false;
  }

  const uint32_t N = WAV.totalPCMFrameCount;
  std::vector<int16_t> PCM16(N * WAV.channels);
  drwav_read_pcm_frames_s16(&WAV, N, PCM16.data());
  drwav_uninit(&WAV);

  PCMF32.resize(N);
  if (WAV.channels == 1) {
    for (uint64_t I = 0; I < N; I++) {
      PCMF32[I] = static_cast<float>(PCM16[I]) / 32768.0f;
    }
  } else {
    for (uint64_t I = 0; I < N; I++) {
      PCMF32[I] =
          static_cast<float>(PCM16[2 * I] + PCM16[2 * I + 1]) / 65536.0f;
    }
  }
  if (Stereo) {
    PCMF32s.resize(2);

    PCMF32s[0].resize(N);
    PCMF32s[1].resize(N);
    for (uint64_t I = 0; I < N; I++) {
      PCMF32s[0][I] = float(PCM16[2 * I]) / 32768.0f;
      PCMF32s[1][I] = float(PCM16[2 * I + 1]) / 32768.0f;
    }
  }
  return true;
}

} // namespace WasmEdge::Host::WASINN::Whisper

#endif
