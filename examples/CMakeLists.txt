# Copyright 2024-2025 Laurynas Biveinis
cmake_minimum_required(VERSION 3.16)

# This is a top-level CMake script and not an include of ../CMakeLists.txt. In
# fact, this one includes the one in the parent directory to show how another
# project can integrate UnoDB in its build.

project(unodb-examples VERSION 0.1
  DESCRIPTION "unodb key-value store library examples"
  HOMEPAGE_URL "https://github.com/laurynas-biveinis/unodb" LANGUAGES CXX)

option(TESTS "Find Google Test in the parent project for UnoDB to use" ON)
option(BENCHMARKS "Find Google Benchmark in the parent project for UnoDB to use"
  ON)

# Search for Google Test and Google Benchmark before including UnoDB
if(TESTS)
  find_package(GTest)
endif()
if(BENCHMARKS)
  find_package(benchmark)
endif()

# The TESTS and BENCHMARKS options will be passed to UnoDB too to configure the
# corresponding build parts. The parent project may also have different names or
# completely different logic, in which case the UnoDB options have to be set up
# explicitly.
add_subdirectory(.. unodb)

add_executable(example_art example_art.cpp)
target_link_libraries(example_art PRIVATE unodb)

add_executable(example_art_stats example_art_stats.cpp)
target_link_libraries(example_art_stats PRIVATE unodb)

add_executable(example_olc_art example_olc_art.cpp)
target_link_libraries(example_olc_art PRIVATE unodb)

add_executable(example_upsert example_upsert.cpp)
target_link_libraries(example_upsert PRIVATE unodb)

add_custom_target(examples
  ./example_art
  COMMAND ./example_art_stats
  COMMAND ./example_olc_art
  COMMAND ./example_upsert
  DEPENDS example_art example_art_stats example_olc_art example_upsert)

set(VALGRIND_COMMAND "valgrind" "--error-exitcode=1" "--leak-check=full"
  "--trace-children=yes" "-v")

add_custom_target(valgrind_examples
  COMMAND ${VALGRIND_COMMAND} ./example_art
  COMMAND ${VALGRIND_COMMAND} ./example_art_stats
  COMMAND ${VALGRIND_COMMAND} ./example_olc_art
  COMMAND ${VALGRIND_COMMAND} ./example_upsert
  DEPENDS example_art example_art_stats example_olc_art example_upsert)
