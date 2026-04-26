# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2019-2024 Second State INC

include(WASINNDeps)
include(${CMAKE_SOURCE_DIR}/plugins/wasi_nn/wasi_nn_backends.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/wasi_nn_fixtures.cmake)

function(wasmedge_wasi_nn_setup_test_target TARGET)
  target_compile_definitions(${TARGET}
    PRIVATE
    WASMEDGE_WASI_NN_TESTING
  )

  add_dependencies(${TARGET}
    wasmedgePluginWasiNN
  )

  wasmedge_setup_wasinn_target(${TARGET})

  target_include_directories(${TARGET}
    PUBLIC
    $<TARGET_PROPERTY:wasmedgePlugin,INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:wasmedgePluginWasiNNCore,INCLUDE_DIRECTORIES>
  )

  target_link_libraries(${TARGET}
    PRIVATE
    wasmedgePluginWasiNNCore
    ${GTEST_BOTH_LIBRARIES}
  )
  wasmedge_link_wasinn_backend_dependencies(${TARGET} wasmedgePluginWasiNNCore)

  if(TARGET build_info)
    target_link_libraries(${TARGET}
      PRIVATE
      build_info
    )
  endif()

  if(WASMEDGE_LINK_PLUGINS_STATIC)
    target_link_libraries(${TARGET}
      PRIVATE
      wasmedgeCAPI
    )
  else()
    target_link_libraries(${TARGET}
      PRIVATE
      wasmedge_shared
    )
  endif()

  if(WASMEDGE_BUILD_WASI_NN_RPC)
    target_compile_definitions(${TARGET}
      PRIVATE
      WASMEDGE_BUILD_WASI_NN_RPC
    )
    target_link_libraries(${TARGET}
      PRIVATE
      wasiNNRPC
    )
  endif()
endfunction()

wasmedge_add_executable(wasiNNCommonTests
  wasi_nn_common.cpp
)
wasmedge_wasi_nn_setup_test_target(wasiNNCommonTests)
add_test(wasiNNCommonTests wasiNNCommonTests)

if(WASMEDGE_PLUGIN_WASI_NN_BACKEND)
  set(WASMEDGE_WASI_NN_BACKEND_TEST_SOURCES)
  foreach(BACKEND ${WASMEDGE_PLUGIN_WASI_NN_BACKEND})
    wasmedge_wasi_nn_get_backend_sources(BACKEND_TEST_SOURCES "${BACKEND}" TEST)
    list(APPEND WASMEDGE_WASI_NN_BACKEND_TEST_SOURCES ${BACKEND_TEST_SOURCES})
  endforeach()

  if(WASMEDGE_WASI_NN_BACKEND_TEST_SOURCES)
    wasmedge_add_executable(wasiNNTests
      ${WASMEDGE_WASI_NN_BACKEND_TEST_SOURCES}
    )
    wasmedge_wasi_nn_setup_test_target(wasiNNTests)

    foreach(BACKEND ${WASMEDGE_PLUGIN_WASI_NN_BACKEND})
      wasmedge_wasi_nn_prepare_backend_fixtures(wasiNNTests "${BACKEND}")
    endforeach()

    add_test(wasiNNTests wasiNNTests)
  endif()
endif()
