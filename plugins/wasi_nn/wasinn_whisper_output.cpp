// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_whisper_output.h"
#include "host/wasi/vfs_io.h"
#include "wasinnenv.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_WHISPER

#include "common/spdlog.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

using namespace std::literals;

namespace WasmEdge::Host::WASINN::Whisper {
int timestampToSample(int64_t T, int NSamples, int WhisperSampleRate) {
  return std::max(0, std::min(static_cast<int>(NSamples) - 1,
                              static_cast<int>((T * WhisperSampleRate) / 100)));
}
std::string toTimestamp(int64_t T, bool Comma) {
  int64_t Msec = T * 10;
  int64_t Hr = Msec / (1000 * 60 * 60);
  Msec = Msec - Hr * (1000 * 60 * 60);
  int64_t Min = Msec / (1000 * 60);
  Msec = Msec - Min * (1000 * 60);
  int64_t Sec = Msec / 1000;
  Msec = Msec - Sec * 1000;

  char Buf[32] = {};
  snprintf(Buf, sizeof(Buf), "%02d:%02d:%02d%s%03d", static_cast<int>(Hr),
           static_cast<int>(Min), static_cast<int>(Sec), Comma ? "," : ".",
           static_cast<int>(Msec));

  return std::string(Buf);
}

std::string
estimateDiarizationSpeaker(const std::vector<std::vector<float>> PCMF32s,
                           int64_t T0, int64_t T1, bool IdOnly = false) {
  std::string Speaker = "";
  const int64_t NSamples = PCMF32s[0].size();

  const int64_t Is0 = timestampToSample(T0, NSamples, WHISPER_SAMPLE_RATE);
  const int64_t Is1 = timestampToSample(T1, NSamples, WHISPER_SAMPLE_RATE);

  double Energy0 = 0.0f;
  double Energy1 = 0.0f;

  for (int64_t I = Is0; I < Is1; I++) {
    Energy0 += fabs(PCMF32s[0][I]);
    Energy1 += fabs(PCMF32s[1][I]);
  }

  if (Energy0 > 1.1 * Energy1) {
    Speaker = "0";
  } else if (Energy1 > 1.1 * Energy0) {
    Speaker = "1";
  } else {
    Speaker = "?";
  }

  if (!IdOnly) {
    Speaker.insert(0, "(speaker ");
    Speaker.append(")");
  }

  return Speaker;
}

bool outputSrt(WasiNNEnvironment &Env, whisper_context *Ctx,
               const std::string &Fname, const Config &Params,
               const std::vector<std::vector<float>> &PCMF32s) {
  WasmEdge::FStream::OFStream Fout(Fname, Env.getEnv());
  if (!Fout.is_open()) {
    spdlog::error("[WASI-NN] Whisper backend: failed to open {} for writing."sv,
                  Fname);
    return false;
  }
  spdlog::info("[WASI-NN] Whisper backend: saving srt output to {}."sv, Fname);

  const int NSegments = whisper_full_n_segments(Ctx);
  for (int I = 0; I < NSegments; ++I) {
    const std::string &Text = whisper_full_get_segment_text(Ctx, I);
    const int64_t T0 = whisper_full_get_segment_t0(Ctx, I);
    const int64_t T1 = whisper_full_get_segment_t1(Ctx, I);
    std::string Speaker = "";

    if (Params.Diarize && PCMF32s.size() == 2) {
      Speaker = estimateDiarizationSpeaker(PCMF32s, T0, T1);
    }

    Fout << I + 1 + Params.OffsetN << "\n";
    Fout << toTimestamp(T0, true) << " --> " << toTimestamp(T1, true) << "\n";
    Fout << Speaker << Text << "\n\n";
  }
  return true;
}

bool outputLrc(WasiNNEnvironment &Env, whisper_context *Ctx,
               const std::string &Fname, const Config &Params,
               const std::vector<std::vector<float>> &PCMF32s) {
  WasmEdge::FStream::OFStream Fout(Fname, Env.getEnv());
  if (!Fout.is_open()) {
    spdlog::error("[WASI-NN] Whisper backend: failed to open {} for writing."sv,
                  Fname);
    return false;
  }

  spdlog::info("[WASI-NN] Whisper backend: saving lrc output to {}."sv, Fname);

  Fout << "[by:whisper.cpp]\n";

  const int NSegments = whisper_full_n_segments(Ctx);
  for (int I = 0; I < NSegments; ++I) {
    const std::string &text = whisper_full_get_segment_text(Ctx, I);
    const int64_t T = whisper_full_get_segment_t0(Ctx, I);

    int64_t Msec = T * 10;
    int64_t Min = Msec / (1000 * 60);
    Msec = Msec - Min * (1000 * 60);
    int64_t Sec = Msec / 1000;
    Msec = Msec - Sec * 1000;

    char Buf[16];
    snprintf(Buf, sizeof(Buf), "%02d:%02d.%02d", static_cast<int>(Min),
             static_cast<int>(Sec), static_cast<int>((Msec / 10)));
    std::string TimestampLrc = std::string(Buf);
    std::string Speaker = "";

    if (Params.Diarize && PCMF32s.size() == 2) {
      const int64_t t0 = whisper_full_get_segment_t0(Ctx, I);
      const int64_t t1 = whisper_full_get_segment_t1(Ctx, I);
      Speaker = estimateDiarizationSpeaker(PCMF32s, t0, t1);
    }

    Fout << '[' << TimestampLrc << ']' << Speaker << text << "\n";
  }

  return true;
}

std::string escapeDoubleQuotesAndBackslashes(const std::string &Str) {
  std::string Escaped;
  for (auto W : Str) {
    if (W == '"' || W == '\\') {
      Escaped += '\\';
    }
    Escaped += W;
  }
  return Escaped;
}

bool outputJson(WasiNNEnvironment &Env, whisper_context *Ctx,
                const std::string &Fname, const Config &Params,
                const std::vector<std::vector<float>> &PCMF32s, bool Full) {
  WasmEdge::FStream::OFStream Fout(Fname, Env.getEnv());
  int Indent = 0;

  auto Doindent = [&]() {
    for (int i = 0; i < Indent; i++)
      Fout << "\t";
  };

  auto StartArr = [&](const char *Name) {
    Doindent();
    Fout << "\"" << Name << "\": [\n";
    Indent++;
  };

  auto EndArr = [&](bool End) {
    Indent--;
    Doindent();
    Fout << (End ? "]\n" : "],\n");
  };

  auto StartObj = [&](const char *Name) {
    Doindent();
    if (Name) {
      Fout << "\"" << Name << "\": {\n";
    } else {
      Fout << "{\n";
    }
    Indent++;
  };

  auto EndObj = [&](bool End) {
    Indent--;
    Doindent();
    Fout << (End ? "}\n" : "},\n");
  };

  auto StartValue = [&](const char *Name) {
    Doindent();
    Fout << "\"" << Name << "\": ";
  };

  auto ValueS = [&](const char *Name, const std::string &Val, bool End) {
    StartValue(Name);
    std::string ValEscaped = escapeDoubleQuotesAndBackslashes(Val);
    Fout << "\"" << ValEscaped << (End ? "\"\n" : "\",\n");
  };

  auto EndValue = [&](bool End) { Fout << (End ? "\n" : ",\n"); };

  auto ValueI = [&](const char *Name, const int64_t Val, bool End) {
    StartValue(Name);
    Fout << Val;
    EndValue(End);
  };

  auto ValueF = [&](const char *Name, const float Val, bool End) {
    StartValue(Name);
    Fout << Val;
    EndValue(End);
  };

  auto ValueB = [&](const char *Name, const bool Val, bool End) {
    StartValue(Name);
    Fout << (Val ? "true" : "false");
    EndValue(End);
  };

  auto TimesO = [&](int64_t T0, int64_t T1, bool End) {
    StartObj("timestamps");
    ValueS("from", toTimestamp(T0, true), false);
    ValueS("to", toTimestamp(T1, true), true);
    EndObj(false);
    StartObj("offsets");
    ValueI("from", T0 * 10, false);
    ValueI("to", T1 * 10, true);
    EndObj(End);
  };

  if (!Fout.is_open()) {
    spdlog::error("[WASI-NN] Whisper backend: failed to open {} for writing."sv,
                  Fname);
    return false;
  }

  spdlog::info("[WASI-NN] Whisper backend: saving json output to {}."sv, Fname);

  StartObj(nullptr);
  ValueS("systeminfo", whisper_print_system_info(), false);
  StartObj("model");
  ValueS("type", whisper_model_type_readable(Ctx), false);
  ValueB("multilingual", whisper_is_multilingual(Ctx), false);
  ValueI("vocab", whisper_model_n_vocab(Ctx), false);
  StartObj("audio");
  ValueI("ctx", whisper_model_n_audio_ctx(Ctx), false);
  ValueI("state", whisper_model_n_audio_state(Ctx), false);
  ValueI("head", whisper_model_n_audio_head(Ctx), false);
  ValueI("layer", whisper_model_n_audio_layer(Ctx), true);
  EndObj(false);
  StartObj("text");
  ValueI("ctx", whisper_model_n_text_ctx(Ctx), false);
  ValueI("state", whisper_model_n_text_state(Ctx), false);
  ValueI("head", whisper_model_n_text_head(Ctx), false);
  ValueI("layer", whisper_model_n_text_layer(Ctx), true);
  EndObj(false);
  ValueI("mels", whisper_model_n_mels(Ctx), false);
  ValueI("ftype", whisper_model_ftype(Ctx), true);
  EndObj(false);
  StartObj("params");
  ValueS("model", "Wasi-nn preload", false);
  ValueS("language", Params.SpokenLanguage, false);
  ValueB("translate", Params.Translate, true);
  EndObj(false);
  StartObj("result");
  ValueS("language", whisper_lang_str(whisper_full_lang_id(Ctx)), true);
  EndObj(false);
  StartArr("transcription");

  const int NSegments = whisper_full_n_segments(Ctx);
  for (int I = 0; I < NSegments; ++I) {
    const std::string &Text = whisper_full_get_segment_text(Ctx, I);

    const int64_t T0 = whisper_full_get_segment_t0(Ctx, I);
    const int64_t T1 = whisper_full_get_segment_t1(Ctx, I);

    StartObj(nullptr);
    TimesO(T0, T1, false);
    ValueS("text", Text, !Params.Diarize && !Params.TinyDiarize && !Full);

    if (Full) {
      StartArr("tokens");
      const int n = whisper_full_n_tokens(Ctx, I);
      for (int j = 0; j < n; ++j) {
        auto token = whisper_full_get_token_data(Ctx, I, j);
        StartObj(nullptr);
        ValueS("text", whisper_token_to_str(Ctx, token.id), false);
        if (token.t0 > -1 && token.t1 > -1) {
          // If we have per-token timestamps, write them out
          TimesO(token.t0, token.t1, false);
        }
        ValueI("id", token.id, false);
        ValueF("p", token.p, false);
        ValueF("t_dtw", token.t_dtw, true);
        EndObj(j == (n - 1));
      }
      EndArr(!Params.Diarize && !Params.TinyDiarize);
    }

    if (Params.Diarize && PCMF32s.size() == 2) {
      ValueS("speaker", estimateDiarizationSpeaker(PCMF32s, T0, T1, true),
             true);
    }

    if (Params.TinyDiarize) {
      ValueB("speaker_turn_next",
             whisper_full_get_segment_speaker_turn_next(Ctx, I), true);
    }
    EndObj(I == (NSegments - 1));
  }

  EndArr(true);
  EndObj(true);
  return true;
}

void WhisperOutputSegmentCallback(struct whisper_context *WhisperCtx,
                                  struct whisper_state * /* state */, int NewN,
                                  void *UserData) {
  auto &CxtRef = *reinterpret_cast<Context *>(UserData);
  const int SegN = whisper_full_n_segments(WhisperCtx);

  std::string Speaker = "";
  for (int I = SegN - NewN; I < SegN; I++) {
    int64_t T0 = 0;
    int64_t T1 = 0;
    if (!CxtRef.WhisperConfig.NoTimestamps) {
      T0 = whisper_full_get_segment_t0(WhisperCtx, I);
      T1 = whisper_full_get_segment_t1(WhisperCtx, I);
      CxtRef.Outputs += "[";
      CxtRef.Outputs += toTimestamp(T0, false);
      CxtRef.Outputs += " --> ";
      CxtRef.Outputs += toTimestamp(T1, false);
      CxtRef.Outputs += "] ";
    }
    if (CxtRef.WhisperConfig.Diarize && CxtRef.InputPCMs.size() == 2) {
      Speaker = estimateDiarizationSpeaker(CxtRef.InputPCMs, T0, T1);
    }
    CxtRef.Outputs += Speaker + whisper_full_get_segment_text(WhisperCtx, I);
    if (!CxtRef.WhisperConfig.NoTimestamps || CxtRef.WhisperConfig.Diarize) {
      CxtRef.Outputs += "\n";
    }
  }
}

} // namespace WasmEdge::Host::WASINN::Whisper

#endif
