# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2019-2024 Second State INC

add_library(wasmedgePluginWasiNNCore
  OBJECT
  wasinn_backend.cpp
  wasinn_dispatch.cpp
  wasinn_graph.cpp
  wasinn_memory.cpp
  wasinn_metadata.cpp
  wasinn_output.cpp
  wasinn_rpc_client.cpp
  wasinnenv.cpp
  wasinnfunc.cpp
  wasinnmodule.cpp
)
wasmedge_setup_target(wasmedgePluginWasiNNCore)

include(WASINNDeps)
include(${CMAKE_CURRENT_LIST_DIR}/wasi_nn_backends.cmake)
wasmedge_setup_wasinn_target(wasmedgePluginWasiNNCore PLUGINLIB)

wasmedge_add_library(wasmedgePluginWasiNN
  SHARED
  $<TARGET_OBJECTS:wasmedgePluginWasiNNCore>
)

set(WASMEDGE_WASI_NN_VERSION "0.1.34" CACHE STRING "WasmEdge WASI-NN library version")
set(WASMEDGE_WASI_NN_SOVERSION "0" CACHE STRING "WasmEdge WASI-NN library soversion")

# Handle the version of the WASI-NN plugin
string(REPLACE "." ";" WASI_NN_VERSION_LIST ${WASMEDGE_WASI_NN_VERSION})
list(GET WASI_NN_VERSION_LIST 0 WASI_NN_VERSION_MAJOR)
list(GET WASI_NN_VERSION_LIST 1 WASI_NN_VERSION_MINOR)
list(GET WASI_NN_VERSION_LIST 2 WASI_NN_VERSION_PATCH)

target_compile_definitions(wasmedgePluginWasiNNCore PRIVATE
  WASI_NN_VERSION_MAJOR=${WASI_NN_VERSION_MAJOR}
  WASI_NN_VERSION_MINOR=${WASI_NN_VERSION_MINOR}
  WASI_NN_VERSION_PATCH=${WASI_NN_VERSION_PATCH}
)

if(WASMEDGE_BUILD_TESTS)
  target_compile_definitions(wasmedgePluginWasiNNCore
    PUBLIC
    WASMEDGE_WASI_NN_TESTING
  )
endif()

foreach(BACKEND ${WASMEDGE_PLUGIN_WASI_NN_BACKEND})
  wasmedge_wasi_nn_normalize_backend(BACKEND "${BACKEND}")
  wasmedge_wasi_nn_add_backend_sources(wasmedgePluginWasiNNCore "${BACKEND}" PLUGIN)
  if(BACKEND STREQUAL "ggml")
    if(WASMEDGE_PLUGIN_WASI_NN_GGML_LLAMA_HIP)
      find_package(hip REQUIRED)
      find_package(hipblas REQUIRED)
      set(GGML_HIP ON CACHE BOOL "Build GGML with HIP" FORCE)
      if(DEFINED ENV{GPU_TARGETS})
        set(GPU_TARGETS $ENV{GPU_TARGETS} CACHE STRING "HIP GPU targets")
      else()
        set(GPU_TARGETS "gfx90c" CACHE STRING "Default for Vega iGPU")
      endif()
      message(STATUS "Enabling HIP for ggml backend with targets: ${GPU_TARGETS}")
      target_compile_definitions(wasmedgePluginWasiNNCore PRIVATE GGML_HIP=${GGML_HIP} GPU_TARGETS=${GPU_TARGETS})
      target_link_libraries(wasmedgePluginWasiNN PRIVATE hip::host roc::hipblas)
    endif()
  elseif(BACKEND STREQUAL "piper")
    if(DEFINED PIPER_ROOT)
      find_library(ESPEAK_NG_LIB
          NAMES espeak-ng libespeak-ng
          PATHS /usr/local/lib /usr/local/lib64
          NO_DEFAULT_PATH
      )
      if (NOT ESPEAK_NG_LIB)
          find_library(ESPEAK_NG_LIB NAMES espeak-ng libespeak-ng)
      endif()

      find_library(UCD_LIB
          NAMES ucd libucd
          PATHS /usr/local/lib /usr/local/lib64
          NO_DEFAULT_PATH
      )
      if (NOT UCD_LIB)
          find_library(UCD_LIB NAMES ucd libucd)
      endif()

      set(ESPEAK_TARGETS ${ESPEAK_NG_LIB} ${UCD_LIB})
    else()
      set(ESPEAK_TARGETS "")
    endif()

    target_link_libraries(wasmedgePluginWasiNN
      PRIVATE
      onnxruntime
      ${ESPEAK_TARGETS}
    )
  endif()
endforeach()

target_compile_definitions(wasmedgePluginWasiNNCore
  PUBLIC
  WASMEDGE_PLUGIN
)

target_include_directories(wasmedgePluginWasiNNCore
  PUBLIC
  $<TARGET_PROPERTY:wasmedgePlugin,INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:wasmedgeHostModuleWasi,INCLUDE_DIRECTORIES>
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(wasmedgePluginWasiNN
  PUBLIC
  $<TARGET_PROPERTY:wasmedgePluginWasiNNCore,INCLUDE_DIRECTORIES>
)

wasmedge_link_wasinn_backend_dependencies(wasmedgePluginWasiNN wasmedgePluginWasiNNCore)

if(WASMEDGE_BUILD_WASI_NN_RPC)
  target_compile_definitions(wasmedgePluginWasiNNCore
    PUBLIC
    WASMEDGE_BUILD_WASI_NN_RPC
  )
  target_include_directories(wasmedgePluginWasiNNCore
    SYSTEM BEFORE PUBLIC ${Protobuf_INCLUDE_DIR}
  )
  target_link_libraries(wasmedgePluginWasiNN
    PRIVATE
    wasiNNRPC
  )
endif()

if(WASMEDGE_LINK_PLUGINS_STATIC)
  target_link_libraries(wasmedgePluginWasiNN
    PRIVATE
    wasmedgeCAPI
  )
else()
  target_link_libraries(wasmedgePluginWasiNN
    PRIVATE
    wasmedge_shared
  )
endif()

install(
  TARGETS wasmedgePluginWasiNN
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/wasmedge
  COMPONENT WasmEdge
)
