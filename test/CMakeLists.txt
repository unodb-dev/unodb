# Copyright 2021-2026 UnoDB contributors

enable_testing()

function(ADD_COVERAGE_TARGET)
  set(fn_options "")
  set(fn_one_value_args TARGET DEPENDENCY)
  set(fn_multi_value_args "")
  cmake_parse_arguments(ARG "${fn_options}" "${fn_one_value_args}"
    "${fn_multi_value_args}" ${ARGN})
  set(COV_DATA "coverage-${ARG_TARGET}.info")
  add_custom_target(${ARG_TARGET}
    COMMAND lcov ${LCOV_GCOV_ARG} --capture --directory .. --output-file
    ${COV_DATA}
    COMMAND lcov ${LCOV_GCOV_ARG} --remove ${COV_DATA} '/usr/*' --output-file
    ${COV_DATA}
    COMMAND lcov ${LCOV_GCOV_ARG} --remove ${COV_DATA} '/Library/Developer/*'
    --output-file ${COV_DATA}
    COMMAND lcov ${LCOV_GCOV_ARG} --remove ${COV_DATA} '/Applications/Xcode*'
    --output-file ${COV_DATA}
    COMMAND lcov ${LCOV_GCOV_ARG} --remove ${COV_DATA} '*/unodb/3rd_party/*'
    --output-file ${COV_DATA}
    COMMAND lcov ${LCOV_GCOV_ARG} --list ${COV_DATA}
    DEPENDS ${ARG_DEPENDENCY})
endfunction()

add_library(db_test_utils STATIC db_test_utils.hpp db_test_utils.cpp)
common_target_properties(db_test_utils)
target_link_libraries(db_test_utils
  PUBLIC unodb GTest::gtest_main GTest::gmock_main)
set_clang_tidy_options(db_test_utils "${DO_CLANG_TIDY}")

add_library(qsbr_test_utils STATIC qsbr_test_utils.hpp qsbr_test_utils.cpp
  qsbr_gtest_utils.hpp qsbr_gtest_utils.cpp)
common_target_properties(qsbr_test_utils)
target_link_libraries(qsbr_test_utils PUBLIC GTest::gtest_main)
target_link_libraries(qsbr_test_utils PRIVATE unodb_qsbr)
set_clang_tidy_options(qsbr_test_utils "${DO_CLANG_TIDY}")

function(ADD_TEST_TARGET TARGET)
  add_executable("${TARGET}" "${TARGET}.cpp")
  common_target_properties("${TARGET}")
  target_link_libraries("${TARGET}"
    PRIVATE unodb_qsbr unodb_test GTest::gtest_main)
  set_clang_tidy_options("${TARGET}" "${DO_CLANG_TIDY}")
  add_sanitized_test(NAME "${TARGET}" COMMAND "${TARGET}")
endfunction()

function(ADD_DB_TEST_TARGET TARGET)
  add_test_target("${TARGET}")
  target_link_libraries("${TARGET}" PRIVATE unodb db_test_utils)
endfunction()

add_test_target(test_qsbr_ptr)
add_test_target(test_qsbr)
target_link_libraries(test_qsbr PRIVATE qsbr_test_utils)
add_db_test_target(test_key_encode_decode)

add_db_test_target(test_art)
target_compile_options(test_art PRIVATE "$<${is_msvc}:/bigobj>")

add_db_test_target(test_art_bulk_load)
add_db_test_target(test_art_bulk_load_structural)
add_db_test_target(test_art_bulk_load_ops)
target_link_libraries(test_art_bulk_load_ops PRIVATE qsbr_test_utils)

add_db_test_target(test_art_key_view)
add_db_test_target(test_art_key_view_full_chain)
add_db_test_target(test_art_iter)
add_db_test_target(test_art_scan)
add_db_test_target(test_art_concurrency)
add_db_test_target(test_art_upsert)
target_compile_options(test_art_upsert PRIVATE "$<${is_msvc}:/bigobj>")
add_db_test_target(test_art_allocator)
# - Google Test with MSVC standard library tries to allocate memory in the
# exception-thrown-as-expected-path.
# - clang analyzer diagnoses potential memory leak in Google Test matcher
# - ASan/TSan do not work with replaced global new/delete:
# https://github.com/llvm/llvm-project/issues/20034
if (NOT MSVC AND NOT SANITIZE_ADDRESS AND NOT SANITIZE_THREAD
    AND NOT (is_any_clang AND STATIC_ANALYSIS))
  add_db_test_target(test_art_oom)
  add_test_target(test_qsbr_oom)
  target_link_libraries(test_qsbr_oom PRIVATE qsbr_test_utils)
endif()
target_link_libraries(test_art_concurrency PRIVATE qsbr_test_utils)
target_link_libraries(test_art_upsert PRIVATE qsbr_test_utils)

# Regression test: olc_db with custom allocator must link without libunodb_qsbr.
# Intentionally links only unodb_util (header-only) and Threads, NOT unodb or
# unodb_qsbr.  Does NOT use common_target_properties (which defines
# UNODB_DETAIL_QSBR_DEBUG).  If this fails to link, the decoupling is broken.
add_executable(test_olc_no_qsbr test_olc_no_qsbr.cpp)
target_compile_features(test_olc_no_qsbr PUBLIC cxx_std_20)
set_target_properties(test_olc_no_qsbr PROPERTIES CXX_EXTENSIONS OFF)
target_compile_definitions(test_olc_no_qsbr PRIVATE
  "UNODB_SPINLOCK_LOOP_VALUE=${SPINLOCK_LOOP_VALUE}")
target_compile_options(test_olc_no_qsbr PRIVATE
  "$<${is_x86_64_any_msvc}:$<IF:${has_avx2},/arch:AVX2,/arch:AVX>>"
  "$<${is_not_windows_x86_64}:$<IF:${has_avx2},-mavx2,-msse4.1>>")
target_link_libraries(test_olc_no_qsbr PRIVATE unodb_util Threads::Threads)
add_test(NAME test_olc_no_qsbr COMMAND test_olc_no_qsbr)

if(COVERAGE)
  add_custom_target(tests_for_coverage ctest
    DEPENDS test_key_encode_decode test_art test_art_key_view test_art_key_view_full_chain test_art_iter test_art_scan test_art_concurrency test_art_allocator test_art_upsert test_olc_no_qsbr test_qsbr_ptr test_qsbr test_art_oom
            test_qsbr_oom test_art_bulk_load test_art_bulk_load_structural test_art_bulk_load_ops)
  add_coverage_target(TARGET coverage DEPENDENCY tests_for_coverage)
endif()

add_custom_target(valgrind_tests
  COMMAND ${VALGRIND_COMMAND} ./test_qsbr_ptr;
  # The death tests will print their SIGABRT stacktrace under Valgrind. I have
  # not found a way to disable it.
  COMMAND ${VALGRIND_COMMAND} ./test_qsbr;
  COMMAND ${VALGRIND_COMMAND} ./test_key_encode_decode;
  COMMAND ${VALGRIND_COMMAND} ./test_art;
  COMMAND ${VALGRIND_COMMAND} ./test_art_key_view;
  COMMAND ${VALGRIND_COMMAND} ./test_art_key_view_full_chain;
  COMMAND ${VALGRIND_COMMAND} ./test_art_iter;
  COMMAND ${VALGRIND_COMMAND} ./test_art_scan;
  COMMAND ${VALGRIND_COMMAND} ./test_art_concurrency;
  COMMAND ${VALGRIND_COMMAND} ./test_art_upsert;
  COMMAND ${VALGRIND_COMMAND} ./test_art_bulk_load;
  COMMAND ${VALGRIND_COMMAND} ./test_art_bulk_load_structural;
  COMMAND ${VALGRIND_COMMAND} ./test_art_bulk_load_ops
  DEPENDS test_qsbr_ptr test_qsbr test_key_encode_decode test_art test_art_key_view test_art_key_view_full_chain test_art_iter test_art_scan test_art_concurrency test_art_upsert test_art_bulk_load test_art_bulk_load_structural test_art_bulk_load_ops)
