# Copyright 2020-2026 Laurynas Biveinis

set(micro_benchmark_key_prefix_quick_arg "") # Benchmark is quick as-is
set(micro_benchmark_n4_quick_arg "--benchmark_filter=\"/16$$|/25|/100\"")
set(micro_benchmark_n16_quick_arg "--benchmark_filter=\"/64\"")
set(micro_benchmark_n48_quick_arg "--benchmark_filter=\"/8$$|/128|/192\"")
set(micro_benchmark_n256_quick_arg "--benchmark_filter=\"/8|/128|/192\"")
set(micro_benchmark_quick_arg
  "--benchmark_filter=\".*/100$$|.*/1000/.*:800$$|.*/100/.*:0$$\"")
set(micro_benchmark_mutex_quick_arg "--benchmark_filter=\"/4/70000/\"")
set(micro_benchmark_olc_quick_arg "--benchmark_filter=\"/4/70000/\"")

add_custom_target(benchmarks
  env ${SANITIZER_ENV} ./micro_benchmark_key_prefix
  COMMAND env ${SANITIZER_ENV} ./micro_benchmark_n4
  COMMAND env ${SANITIZER_ENV} ./micro_benchmark_n16
  COMMAND env ${SANITIZER_ENV} ./micro_benchmark_n48
  COMMAND env ${SANITIZER_ENV} ./micro_benchmark_n256
  COMMAND env ${SANITIZER_ENV} ./micro_benchmark
  COMMAND env ${SANITIZER_ENV} ./micro_benchmark_mutex
  COMMAND env ${SANITIZER_ENV} ./micro_benchmark_olc)

add_custom_target(quick_benchmarks
  env ${SANITIZER_ENV}
  ./micro_benchmark_key_prefix ${micro_benchmark_key_prefix_quick_arg}
  COMMAND env ${SANITIZER_ENV}
  ./micro_benchmark_n4 ${micro_benchmark_n4_quick_arg}
  COMMAND env ${SANITIZER_ENV}
  ./micro_benchmark_n16 ${micro_benchmark_n16_quick_arg}
  COMMAND env ${SANITIZER_ENV}
  ./micro_benchmark_n48 ${micro_benchmark_n48_quick_arg}
  COMMAND env ${SANITIZER_ENV}
  ./micro_benchmark_n256 ${micro_benchmark_n256_quick_arg}
  COMMAND env ${SANITIZER_ENV}
  ./micro_benchmark ${micro_benchmark_quick_arg}
  COMMAND env ${SANITIZER_ENV}
  ./micro_benchmark_mutex ${micro_benchmark_mutex_quick_arg}
  COMMAND env ${SANITIZER_ENV}
  ./micro_benchmark_olc ${micro_benchmark_olc_quick_arg})

add_custom_target(valgrind_benchmarks
  COMMAND ${VALGRIND_COMMAND} ./micro_benchmark_key_prefix
  ${micro_benchmark_key_prefix_quick_arg}
  COMMAND ${VALGRIND_COMMAND} ./micro_benchmark_n4
  ${micro_benchmark_n4_quick_arg}
  COMMAND ${VALGRIND_COMMAND} ./micro_benchmark_n16
  ${micro_benchmark_n16_quick_arg}
  COMMAND ${VALGRIND_COMMAND} ./micro_benchmark_n48
  ${micro_benchmark_n48_quick_arg}
  COMMAND ${VALGRIND_COMMAND} ./micro_benchmark_n256
  ${micro_benchmark_n256_quick_arg}
  COMMAND ${VALGRIND_COMMAND} ./micro_benchmark ${micro_benchmark_quick_arg}
  COMMAND ${VALGRIND_COMMAND} ./micro_benchmark_mutex
  ${micro_benchmark_mutex_quick_arg}
  COMMAND ${VALGRIND_COMMAND} ./micro_benchmark_olc
  ${micro_benchmark_olc_quick_arg})

add_library(micro_benchmark_utils STATIC micro_benchmark_utils.cpp
  micro_benchmark_utils.hpp)
common_target_properties(micro_benchmark_utils)
target_compile_definitions(micro_benchmark_utils PRIVATE
  BENCHMARK_STATIC_DEFINE)
target_include_directories(micro_benchmark_utils PUBLIC ".")
target_link_libraries(micro_benchmark_utils PUBLIC unodb)
target_link_libraries(micro_benchmark_utils PUBLIC benchmark::benchmark)
target_include_directories(micro_benchmark_utils SYSTEM PUBLIC
  ${benchmark_include_dirs})
set_clang_tidy_options(micro_benchmark_utils "${DO_CLANG_TIDY}")

function(ADD_BENCHMARK_TARGET TARGET)
  add_executable("${TARGET}" "${TARGET}.cpp")
  common_target_properties("${TARGET}")
  target_compile_definitions("${TARGET}" PRIVATE BENCHMARK_STATIC_DEFINE)
  target_link_libraries("${TARGET}" PRIVATE micro_benchmark_utils)
  add_dependencies(benchmarks "${TARGET}")
  add_dependencies(quick_benchmarks "${TARGET}")
  add_dependencies(valgrind_benchmarks "${TARGET}")
endfunction()

function(ADD_NODE_BENCHMARK_TARGET TARGET)
  add_benchmark_target("${TARGET}")
  target_sources("${TARGET}" PRIVATE micro_benchmark_node_utils.hpp)
endfunction()

function(ADD_CONCURRENT_BENCHMARK_TARGET TARGET)
  add_benchmark_target("${TARGET}")
  target_sources("${TARGET}" PRIVATE micro_benchmark_concurrency.hpp)
endfunction()

add_benchmark_target(micro_benchmark_key_prefix)
add_benchmark_target(micro_benchmark_key_view)
add_node_benchmark_target(micro_benchmark_n4)
add_node_benchmark_target(micro_benchmark_n16)
add_node_benchmark_target(micro_benchmark_n48)
add_node_benchmark_target(micro_benchmark_n256)
add_node_benchmark_target(micro_benchmark)
add_concurrent_benchmark_target(micro_benchmark_mutex)
add_concurrent_benchmark_target(micro_benchmark_olc)
