// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "wasinn_whisper_config.h"
#include "wasinn_whisper_output.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_WHISPER

#include "common/spdlog.h"
#include "simdjson.h"

#include <string>
#include <string_view>

using namespace std::literals;

namespace WasmEdge::Host::WASINN::Whisper {
void WhisperLogCallback(ggml_log_level LogLevel, const char *LogText,
                        void *UserData) {
  const Graph &GraphRef = *reinterpret_cast<Graph *>(UserData);
  if (!GraphRef.WhisperConfig.EnableLog) {
    return;
  }
  std::string Text(LogText);
  // Remove the trailing newlines.
  Text = Text.erase(Text.find_last_not_of("\n") + 1);
  // Skip for "."
  if (Text == ".") {
    return;
  }
  if (LogLevel == GGML_LOG_LEVEL_ERROR) {
    spdlog::error("[WASI-NN] whisper.cpp: {}"sv, Text);
  } else if (LogLevel == GGML_LOG_LEVEL_WARN) {
    spdlog::warn("[WASI-NN] whisper.cpp: {}"sv, Text);
  } else if (LogLevel == GGML_LOG_LEVEL_INFO) {
    spdlog::info("[WASI-NN] whisper.cpp: {}"sv, Text);
  } else if (LogLevel == GGML_LOG_LEVEL_DEBUG) {
    spdlog::debug("[WASI-NN] whisper.cpp: {}"sv, Text);
  }
}

void setWhisperParams(Context &CxtRef) noexcept {
  auto &WParam = CxtRef.WhisperParams;
  auto &ConfigRef = CxtRef.WhisperConfig;
  WParam.n_threads = ConfigRef.ThreadsNum;
  WParam.n_max_text_ctx = ConfigRef.MaxTokenContext;
  WParam.offset_ms = ConfigRef.TimeOffsetMS;
  WParam.duration_ms = ConfigRef.DurationMS;
  WParam.print_progress = false;
  WParam.thold_pt = ConfigRef.WordThreshold;
  WParam.max_len = ConfigRef.MaxSegmentLength;
  WParam.token_timestamps = (WParam.max_len > 0);
  WParam.split_on_word = ConfigRef.SplitOnWord;
  WParam.translate = ConfigRef.Translate;
  WParam.language = ConfigRef.SpokenLanguage.c_str();
  WParam.detect_language = ConfigRef.DetectLanguage;
  WParam.initial_prompt = ConfigRef.InitialPrompt.c_str();
  WParam.temperature_inc = ConfigRef.TemperatureInc;
  WParam.temperature = ConfigRef.Temperature;
  WParam.entropy_thold = ConfigRef.EntropyThreshold;
  WParam.logprob_thold = ConfigRef.LogprobThreshold;
  WParam.grammar_penalty = ConfigRef.GrammarPenalty;
  WParam.new_segment_callback = WhisperOutputSegmentCallback;
  WParam.new_segment_callback_user_data = &CxtRef;
  WParam.greedy.best_of = ConfigRef.BestOf;
  WParam.print_timestamps = !ConfigRef.NoTimestamps;
  WParam.no_timestamps = ConfigRef.NoTimestamps;
  WParam.audio_ctx = ConfigRef.AudioCtx;
  WParam.strategy =
      (ConfigRef.BeamSize > 1)
          ? whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH
          : whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY;
  WParam.beam_search.beam_size = ConfigRef.BeamSize;

  if (ConfigRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] Whisper backend: Config: threads: {}"sv,
                 ConfigRef.ThreadsNum);
    spdlog::info("[WASI-NN][Debug] Whisper backend: Config: processors: {}"sv,
                 ConfigRef.ProcessorsNum);
    spdlog::info("[WASI-NN][Debug] Whisper backend: Config: max-context: {}"sv,
                 ConfigRef.MaxTokenContext);
    spdlog::info("[WASI-NN][Debug] Whisper backend: Config: offset-t: {}"sv,
                 ConfigRef.TimeOffsetMS);
    spdlog::info("[WASI-NN][Debug] Whisper backend: Config: duration: {}"sv,
                 ConfigRef.DurationMS);
    spdlog::info("[WASI-NN][Debug] Whisper backend: Config: max-len: {}"sv,
                 ConfigRef.MaxSegmentLength);
    spdlog::info(
        "[WASI-NN][Debug] Whisper backend: Config: split-on-word : {}"sv,
        ConfigRef.SplitOnWord);
    spdlog::info("[WASI-NN][Debug] Whisper backend: Config: translate: {}"sv,
                 ConfigRef.Translate);
    spdlog::info("[WASI-NN][Debug] Whisper backend: Config: language: \"{}\""sv,
                 ConfigRef.SpokenLanguage);
    spdlog::info(
        "[WASI-NN][Debug] Whisper backend: Config: detect-language: {}"sv,
        ConfigRef.DetectLanguage);
    spdlog::info("[WASI-NN][Debug] Whisper backend: Config: temperature: {}"sv,
                 ConfigRef.Temperature);
    spdlog::info("[WASI-NN][Debug] Whisper backend: Config: prompt: \"{}\""sv,
                 ConfigRef.InitialPrompt);
  }
}

Expect<ErrNo> parseMetadata(Config &ConfigRef,
                            const std::string &Metadata) noexcept {
  simdjson::dom::parser Parser;
  simdjson::dom::element Doc;
  auto ParseError = Parser.parse(Metadata).get(Doc);
  if (ParseError) {
    spdlog::error("[WASI-NN] Whisper backend: Parse metadata error."sv);
    return ErrNo::InvalidEncoding;
  }

  auto PrintParsedOption = [&](std::string_view Name, const auto &Val) {
    if (ConfigRef.EnableDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] Whisper backend: Parsed metadata -- {}:{}"sv, Name,
          Val);
    }
  };

  // Get metadata from the json.
  // Currently supported metadata:
  // Plugin parameters (used by this plugin):
  //   enable-log: bool
  //   enable-debug-log: bool
  //   threads: uint32_t
  //   processors: uint32_t
  //   offset-t: uint32_t
  //   duration: uint32_t
  //   max-context: uint32_t
  //   max-len: uint32_t
  //   split-on-word: bool
  //   translate: bool
  //   language: string
  //   detect-language: bool
  //   temperature: float
  //   prompt: string

  // The plugin parameters.
  if (Doc.at_key("enable-log").error() == simdjson::SUCCESS) {
    auto Err = Doc["enable-log"].get<bool>().get(ConfigRef.EnableLog);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the enable-log "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("enable-debug-log").error() == simdjson::SUCCESS) {
    auto Err =
        Doc["enable-debug-log"].get<bool>().get(ConfigRef.EnableDebugLog);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the enable-debug-log "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("threads").error() == simdjson::SUCCESS) {
    auto Err = Doc["threads"].get<uint64_t>().get(ConfigRef.ThreadsNum);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the threads option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("threads"sv, ConfigRef.ThreadsNum);
  }
  if (Doc.at_key("processors").error() == simdjson::SUCCESS) {
    auto Err = Doc["processors"].get<uint64_t>().get(ConfigRef.ProcessorsNum);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the processors option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("processors"sv, ConfigRef.ProcessorsNum);
  }
  if (Doc.at_key("offset-t").error() == simdjson::SUCCESS) {
    auto Err = Doc["offset-t"].get<uint64_t>().get(ConfigRef.TimeOffsetMS);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the offset-t option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("offset-t"sv, ConfigRef.TimeOffsetMS);
  }
  if (Doc.at_key("duration").error() == simdjson::SUCCESS) {
    auto Err = Doc["duration"].get<uint64_t>().get(ConfigRef.DurationMS);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the duration option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("duration"sv, ConfigRef.DurationMS);
  }
  if (Doc.at_key("max-context").error() == simdjson::SUCCESS) {
    int64_t MaxContext = 0;
    auto Err = Doc["max-context"].get<int64_t>().get(MaxContext);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the max-context option."sv);
      return ErrNo::InvalidArgument;
    }
    if (MaxContext >= 0) {
      ConfigRef.MaxTokenContext = static_cast<uint64_t>(MaxContext);
      PrintParsedOption("max-context"sv, ConfigRef.MaxTokenContext);
    }
  }
  if (Doc.at_key("max-len").error() == simdjson::SUCCESS) {
    auto Err = Doc["max-len"].get<uint64_t>().get(ConfigRef.MaxSegmentLength);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the max-len option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("max-len"sv, ConfigRef.MaxSegmentLength);
  }
  if (Doc.at_key("split-on-word").error() == simdjson::SUCCESS) {
    auto Err = Doc["split-on-word"].get<bool>().get(ConfigRef.SplitOnWord);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the split-on-word "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("split-on-word"sv, ConfigRef.SplitOnWord);
  }
  if (Doc.at_key("translate").error() == simdjson::SUCCESS) {
    auto Err = Doc["translate"].get<bool>().get(ConfigRef.Translate);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the translate "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("translate"sv, ConfigRef.Translate);
  }
  if (Doc.at_key("language").error() == simdjson::SUCCESS) {
    std::string_view Language;
    auto Err = Doc["language"].get<std::string_view>().get(Language);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the language "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    ConfigRef.SpokenLanguage = Language;
    PrintParsedOption("language"sv, ConfigRef.SpokenLanguage);
  }
  if (Doc.at_key("detect-language").error() == simdjson::SUCCESS) {
    auto Err = Doc["detect-language"].get<bool>().get(ConfigRef.DetectLanguage);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the detect-language "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("detect-language"sv, ConfigRef.DetectLanguage);
  }
  if (Doc.at_key("temperature").error() == simdjson::SUCCESS) {
    double Temperature;
    auto Err = Doc["temperature"].get<double>().get(Temperature);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the temperature option."sv);
      return ErrNo::InvalidArgument;
    }
    ConfigRef.Temperature = static_cast<float>(Temperature);
    PrintParsedOption("temperature"sv, ConfigRef.Temperature);
  }
  if (Doc.at_key("prompt").error() == simdjson::SUCCESS) {
    std::string_view Prompt;
    auto Err = Doc["prompt"].get<std::string_view>().get(Prompt);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the prompt option."sv);
      return ErrNo::InvalidArgument;
    }
    ConfigRef.InitialPrompt = Prompt;
    PrintParsedOption("prompt"sv, ConfigRef.InitialPrompt);
  }
  if (Doc.at_key("best-of").error() == simdjson::SUCCESS) {
    auto Err = Doc["best-of"].get<uint64_t>().get(ConfigRef.BestOf);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the best-of option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("best-of"sv, ConfigRef.BestOf);
  }
  if (Doc.at_key("beam-size").error() == simdjson::SUCCESS) {
    auto Err = Doc["beam-size"].get<uint64_t>().get(ConfigRef.BeamSize);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the beam-size option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("beam-size"sv, ConfigRef.BeamSize);
  }
  if (Doc.at_key("output-srt").error() == simdjson::SUCCESS) {
    auto Err = Doc["output-srt"].get<bool>().get(ConfigRef.OutputSrt);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the output-srt "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("output-srt"sv, ConfigRef.OutputLrc);
  }
  if (Doc.at_key("output-lrc").error() == simdjson::SUCCESS) {
    auto Err = Doc["output-lrc"].get<bool>().get(ConfigRef.OutputLrc);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the output-lrc"
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("output-lrc"sv, ConfigRef.OutputLrc);
  }
  if (Doc.at_key("output-json").error() == simdjson::SUCCESS) {
    auto Err = Doc["output-json"].get<bool>().get(ConfigRef.OutputJson);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the output-json "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("output-json"sv, ConfigRef.OutputJson);
  }
  if (Doc.at_key("output-json-full").error() == simdjson::SUCCESS) {
    auto Err =
        Doc["output-json-full"].get<bool>().get(ConfigRef.OutputJsonFull);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the output-json-full "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("output-json-full"sv, ConfigRef.OutputJsonFull);
  }
  if (Doc.at_key("no-timestamps").error() == simdjson::SUCCESS) {
    auto Err = Doc["no-timestamps"].get<bool>().get(ConfigRef.NoTimestamps);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the no-timestamps "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("no-timestamps"sv, ConfigRef.NoTimestamps);
  }
  if (Doc.at_key("output-file").error() == simdjson::SUCCESS) {
    std::string_view FileName;
    auto Err = Doc["output-file"].get<std::string_view>().get(FileName);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the output file"
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    ConfigRef.FileName = FileName;
    PrintParsedOption("output-file"sv, ConfigRef.FileName);
  }
  if (Doc.at_key("audio-ctx").error() == simdjson::SUCCESS) {
    auto Err = Doc["audio-ctx"].get<uint64_t>().get(ConfigRef.AudioCtx);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the audio-ctx "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("audio-ctx"sv, ConfigRef.AudioCtx);
  }
  if (Doc.at_key("diarize").error() == simdjson::SUCCESS) {
    auto Err = Doc["diarize"].get<bool>().get(ConfigRef.Diarize);
    if (Err) {
      spdlog::error("[WASI-NN] Whisper backend: Unable to retrieve the diarize "
                    "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("diarize"sv, ConfigRef.Diarize);
  }
  if (Doc.at_key("offset-n").error() == simdjson::SUCCESS) {
    auto Err = Doc["offset-n"].get<uint64_t>().get(ConfigRef.OffsetN);
    if (Err) {
      spdlog::error(
          "[WASI-NN] Whisper backend: Unable to retrieve the offset-n "
          "option."sv);
      return ErrNo::InvalidArgument;
    }
    PrintParsedOption("offset-n"sv, ConfigRef.OffsetN);
  }

  return ErrNo::Success;
}

Expect<ErrNo> handleTranslationConfig(whisper_context *WhisperCtx,
                                      Config &ConfigRef) noexcept {
  assuming(WhisperCtx);

  // Check the language.
  if (ConfigRef.SpokenLanguage != "auto"sv &&
      whisper_lang_id(ConfigRef.SpokenLanguage.c_str()) == -1) {
    spdlog::error("[WASI-NN] Whisper backend: Error: unknown language {}."sv,
                  ConfigRef.SpokenLanguage);
    return ErrNo::InvalidArgument;
  }

  // Check the translate option.
  if (!whisper_is_multilingual(WhisperCtx)) {
    if (ConfigRef.SpokenLanguage != "en"sv || ConfigRef.Translate) {
      ConfigRef.SpokenLanguage = "en"sv;
      ConfigRef.Translate = false;
      if (ConfigRef.EnableLog) {
        spdlog::info(
            "[WASI-NN] Whisper backend: Model is not multilingual. Ignoring "
            "language and translation options"sv);
      }
    }
  }
  if (ConfigRef.DetectLanguage) {
    ConfigRef.SpokenLanguage = "auto"sv;
  }
  return ErrNo::Success;
}

} // namespace WasmEdge::Host::WASINN::Whisper

#endif
