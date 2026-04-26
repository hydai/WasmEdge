# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2019-2024 Second State INC

set(WASMEDGE_WASI_NN_BACKEND_openvino_PLUGIN_SOURCES
  wasinn_openvino.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_openvinogenai_PLUGIN_SOURCES
  wasinn_openvino_genai.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_pytorch_PLUGIN_SOURCES
  wasinn_torch.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_tensorflowlite_PLUGIN_SOURCES
  wasinn_tfl.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_mlx_PLUGIN_SOURCES
  MLX/prompt/prompt.cpp
  MLX/model/llm/transformer.cpp
  MLX/model/llm/registry.cpp
  MLX/model/gemma3/language.cpp
  MLX/model/gemma3/vision.cpp
  MLX/model/gemma3/gemma3.cpp
  MLX/model/converter.cpp
  MLX/model/utils.cpp
  MLX/model/vlm_base.cpp
  MLX/model/vlm_sampling.cpp
  MLX/model/whisper/whisper.cpp
  MLX/model/whisper/tokenizer.cpp
  MLX/model/whisper/decoding.cpp
  MLX/model/whisper_transcribe.cpp
  MLX/mlx/base.cpp
  MLX/mlx/linear.cpp
  MLX/mlx/convolution.cpp
  MLX/mlx/positional_encoding.cpp
  MLX/mlx/activations.cpp
  MLX/mlx/embedding.cpp
  MLX/mlx/normalization.cpp
  MLX/mlx/transformer.cpp
  MLX/mlx/pooling.cpp
  MLX/mlx/quantized.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_ggml_PLUGIN_SOURCES
  GGML/core/ggml_core.cpp
  GGML/core/input_processor.cpp
  GGML/core/ggml_lifecycle.cpp
  GGML/core/output_generator.cpp
  GGML/metadata/metadata_parser.cpp
  GGML/compute/compute_engine.cpp
  GGML/compute/inference_manager.cpp
  GGML/tts/tts_core.cpp
  GGML/utils.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_piper_PLUGIN_SOURCES
  wasinn_piper.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_whisper_PLUGIN_SOURCES
  wasinn_whisper.cpp
  wasinn_whisper_audio.cpp
  wasinn_whisper_config.cpp
  wasinn_whisper_lifecycle.cpp
  wasinn_whisper_output.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_chattts_PLUGIN_SOURCES
  wasinn_chattts.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_bitnet_PLUGIN_SOURCES
  wasinn_bitnet.cpp
  wasinn_bitnet_inference.cpp
  wasinn_bitnet_lifecycle.cpp
  wasinn_bitnet_metadata.cpp
  wasinn_bitnet_metadata_options.cpp
  wasinn_bitnet_metadata_update.cpp
)

set(WASMEDGE_WASI_NN_BACKEND_openvino_TEST_SOURCES
  wasi_nn_openvino.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_pytorch_TEST_SOURCES
  wasi_nn_pytorch.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_tensorflowlite_TEST_SOURCES
  wasi_nn_tflite.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_mlx_TEST_SOURCES
  wasi_nn_mlx.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_ggml_TEST_SOURCES
  wasi_nn_ggml.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_piper_TEST_SOURCES
  wasi_nn_piper.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_whisper_TEST_SOURCES
  wasi_nn_whisper.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_chattts_TEST_SOURCES
  wasi_nn_chattts.cpp
)
set(WASMEDGE_WASI_NN_BACKEND_bitnet_TEST_SOURCES
  wasi_nn_bitnet.cpp
)

function(wasmedge_wasi_nn_normalize_backend OUT BACKEND)
  string(TOLOWER "${BACKEND}" NORMALIZED)
  set(${OUT} "${NORMALIZED}" PARENT_SCOPE)
endfunction()

function(wasmedge_wasi_nn_get_backend_sources OUT BACKEND KIND)
  wasmedge_wasi_nn_normalize_backend(NORMALIZED "${BACKEND}")
  set(SOURCES_VARIABLE WASMEDGE_WASI_NN_BACKEND_${NORMALIZED}_${KIND}_SOURCES)
  if(DEFINED ${SOURCES_VARIABLE})
    set(${OUT} ${${SOURCES_VARIABLE}} PARENT_SCOPE)
  elseif(KIND STREQUAL "TEST")
    set(${OUT} "" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "Unsupported WASI-NN backend '${BACKEND}' in WASMEDGE_PLUGIN_WASI_NN_BACKEND")
  endif()
endfunction()

function(wasmedge_wasi_nn_add_backend_sources TARGET BACKEND KIND)
  wasmedge_wasi_nn_get_backend_sources(BACKEND_SOURCES "${BACKEND}" "${KIND}")
  if(BACKEND_SOURCES)
    target_sources(${TARGET}
      PRIVATE
      ${BACKEND_SOURCES}
    )
  endif()
endfunction()
