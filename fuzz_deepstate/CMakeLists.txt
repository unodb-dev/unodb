# Copyright 2021-2026 Laurynas Biveinis

enable_testing()

function(COMMON_DEEPSTATE_TARGET_PROPERTIES TARGET)
  cmake_parse_arguments(PARSE_ARGV 1 CDTP "WITH_LIBFUZZER" "" "")
  if(CDTP_WITH_LIBFUZZER)
    common_target_properties(${TARGET} SKIP_CHECKS)
    target_link_libraries(${TARGET} PRIVATE deepstate_lf)
    target_compile_options(${TARGET} PRIVATE "-fsanitize=fuzzer-no-link")
    target_link_options(${TARGET} PRIVATE "-fsanitize=fuzzer")
  else()
    common_target_properties(${TARGET})
    target_link_libraries(${TARGET} PRIVATE deepstate)
    set_clang_tidy_options(${TARGET} "${DO_CLANG_TIDY}")
  endif()
endfunction()

set(DEEPSTATE_FAIL_DIR "deepstate_fails")
set(DEEPSTATE_OPTIONS "--fuzz" "--no_fork" "--output_test_dir"
  "${DEEPSTATE_FAIL_DIR}" "--timeout")
# The shortest run is special in that it is a smoke test, and a part of the
# default testsuite. On a completely broken code stop immediately instead of
# accumulating millions of testcases.
set(DEEPSTATE_2S_OPTIONS "--abort_on_fail" ${DEEPSTATE_OPTIONS})

if(LIBFUZZER_AVAILABLE)
  set(DEEPSTATE_LF_OPTIONS "-use_value_profile=1" "-detect_leaks=0")
endif()

function(ADD_FUZZ_DEEPSTATE_TARGET ID)
  set(TEST_NAME "test_${ID}_fuzz_deepstate")
  set(TEST_2S "deepstate_${ID}_2s")
  set(TEST_1M "deepstate_${ID}_1m")
  set(TEST_20M "deepstate_${ID}_20m")
  set(TEST_8H "deepstate_${ID}_8h")
  set(VALGRIND_TARGET "valgrind_${ID}_deepstate")

  add_executable(${TEST_NAME} "${TEST_NAME}.cpp")
  common_deepstate_target_properties(${TEST_NAME})
  add_unodb_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
    ${DEEPSTATE_2S_OPTIONS} 2)

  add_custom_target(${TEST_2S} DEPENDS ${TEST_NAME}
    COMMAND env ${SANITIZER_ENV} "./${TEST_NAME}" ${DEEPSTATE_2S_OPTIONS} 2)
  add_custom_target(${TEST_1M}
    COMMAND env ${SANITIZER_ENV} "./${TEST_NAME}" ${DEEPSTATE_OPTIONS} 60)
  add_custom_target(${TEST_20M}
    COMMAND env ${SANITIZER_ENV} "./${TEST_NAME}" ${DEEPSTATE_OPTIONS} 1200)
  add_custom_target(${TEST_8H}
    COMMAND env ${SANITIZER_ENV} "./${TEST_NAME}" ${DEEPSTATE_OPTIONS} 28800)

  add_dependencies(deepstate_2s ${TEST_2S})
  add_dependencies(deepstate_1m ${TEST_1M})
  add_dependencies(deepstate_20m ${TEST_20M})
  add_dependencies(deepstate_8h ${TEST_8H})

  add_custom_target(${VALGRIND_TARGET} COMMAND ${VALGRIND_COMMAND}
    ./${TEST_NAME} ${DEEPSTATE_OPTIONS} 60)

  add_dependencies(valgrind_deepstate ${VALGRIND_TARGET})

  if(LIBFUZZER_AVAILABLE)
    set(SOURCE_NAME "test_${ID}_fuzz_deepstate.cpp")
    set(TEST_NAME "test_${ID}_fuzz_deepstate_lf")
    set(CORPUS_DIR "deepstate_lf_corpus_${ID}")
    set(INVOCATION "./${TEST_NAME}" ${CORPUS_DIR} ${DEEPSTATE_LF_OPTIONS})
    set(TEST_2S "deepstate_lf_${ID}_2s")
    set(TEST_1M "deepstate_lf_${ID}_1m")
    set(TEST_20M "deepstate_lf_${ID}_20m")
    set(TEST_8H "deepstate_lf_${ID}_8h")
    set(VALGRIND_TARGET "valgrind_${ID}_deepstate_lf")

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CORPUS_DIR})

    add_executable(${TEST_NAME} "${SOURCE_NAME}")
    common_deepstate_target_properties(${TEST_NAME} WITH_LIBFUZZER)
    add_unodb_test(NAME ${TEST_NAME} COMMAND ${INVOCATION}
      "-max_total_time=2")

    add_custom_target(${TEST_2S} DEPENDS ${TEST_NAME}
      COMMAND env ${SANITIZER_ENV} ${INVOCATION} "-max-total-time=2")
    add_custom_target(${TEST_1M}
      COMMAND env ${SANITIZER_ENV} ${INVOCATION} "-max_total_time=60")
    add_custom_target(${TEST_20M}
      COMMAND env ${SANITIZER_ENV} ${INVOCATION} "-max_total_time=1200")
    add_custom_target(${TEST_8H}
      COMMAND env ${SANITIZER_ENV} ${INVOCATION} "-max_total_time=28800")

    add_dependencies(deepstate_lf_2s ${TEST_2S})
    add_dependencies(deepstate_lf_1m ${TEST_1M})
    add_dependencies(deepstate_lf_20m ${TEST_20M})
    add_dependencies(deepstate_lf_8h ${TEST_8H})

    add_custom_target(${VALGRIND_TARGET} COMMAND ${VALGRIND_COMMAND}
      ./${INVOCATION} "-max_total_time=60")

    add_dependencies(valgrind_deepstate_lf ${VALGRIND_TARGET})
  endif()
endfunction()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DEEPSTATE_FAIL_DIR})

add_custom_target(deepstate_2s)
add_custom_target(deepstate_1m)
add_custom_target(deepstate_20m)
add_custom_target(deepstate_8h)
add_custom_target(valgrind_deepstate)

if(LIBFUZZER_AVAILABLE)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DEEPSTATE_LF_CORPUS_DIR})
  add_custom_target(deepstate_lf_2s)
  add_custom_target(deepstate_lf_1m)
  add_custom_target(deepstate_lf_20m)
  add_custom_target(deepstate_lf_8h)
  add_custom_target(valgrind_deepstate_lf)
endif()

add_fuzz_deepstate_target(art)
target_link_libraries(test_art_fuzz_deepstate PRIVATE unodb)
if(LIBFUZZER_AVAILABLE)
  target_link_libraries(test_art_fuzz_deepstate_lf PRIVATE unodb_lf)
endif()

add_fuzz_deepstate_target(qsbr)
# QSBR fault injection requires unodb_test's replacement operator new.
target_link_libraries(test_qsbr_fuzz_deepstate PRIVATE unodb_qsbr unodb_test)
if(LIBFUZZER_AVAILABLE)
  target_link_libraries(test_qsbr_fuzz_deepstate_lf
    PRIVATE unodb_qsbr_lf unodb_test_lf)
endif()
