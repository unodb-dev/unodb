# Copyright 2019-2025 Laurynas Biveinis
cmake_minimum_required(VERSION 3.16)

# Set to new once Boost 1.70 is the minimum oldest version
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

project(unodb VERSION 0.1
  DESCRIPTION "unodb key-value store library"
  HOMEPAGE_URL "https://github.com/laurynas-biveinis/unodb" LANGUAGES CXX)

# https://stackoverflow.com/a/62488003/80458
macro(set_bool var)
  if(${ARGN})
    set(${var} ON)
  else()
    set(${var} OFF)
  endif()
endmacro()

# Condition helpers
set_bool(is_debug CMAKE_BUILD_TYPE STREQUAL "Debug")
set_bool(is_release CMAKE_BUILD_TYPE STREQUAL "Release")
set_bool(is_darwin CMAKE_SYSTEM_NAME STREQUAL "Darwin")
set_bool(is_windows CMAKE_SYSTEM_NAME STREQUAL "Windows")
set_bool(is_x86_64 "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "AMD64"
  OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
set_bool(is_apple_clang "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
set_bool(is_clang "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
set_bool(is_clang11 ${is_clang}
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
set_bool(is_gxx "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
set_bool(is_any_clang ${is_apple_clang} OR ${is_clang})

# Cannot push the top level condition to set_bool because an empty variable
# would expand to an empty token
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT)
  set_bool(is_msvc_clang ${is_clang}
    AND ("${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC"))
else()
  set(is_msvc_clang OFF)
endif()

set_bool(is_not_msvc_clang ${is_clang} AND NOT ${is_msvc_clang})

option(STANDALONE
  "Build UnoDB as a standalone project. Enables extra global checks, builds \
benchmarks by default.")

option(MAINTAINER_MODE "Maintainer mode: compiler warnings are fatal")
if(MAINTAINER_MODE)
  message(STATUS "Warning diagnostics are fatal")
else()
  message(STATUS "Warning diagnostics are not fatal")
endif()

set(CLANG_CXX_WARNING_FLAGS
  # Warning groups
  "-Wall" "-Wextra" "-Wconversion" "-Wdelete-non-virtual-dtor" "-Wdeprecated"
  "-Wgnu" "-Wimplicit" "-Wloop-analysis" "-Wparentheses" "-Wpedantic"
  "-Wpragmas"
  "-Wself-assign" "-Wshadow-all"
  # Individual warnings
  "-Wabstract-vbase-init" "-Warray-bounds-pointer-arithmetic" "-Wassign-enum"
  "-Watomic-implicit-seq-cst" "-Wbad-function-cast"
  "-Wcast-align" "-Wcast-qual" "-Wclass-varargs" "-Wcomma"
  "-Wconditional-uninitialized" "-Wcovered-switch-default" "-Wdate-time"
  "-Wdeprecated-implementations" "-Wdisabled-macro-expansion"
  "-Wdouble-promotion" "-Wduplicate-decl-specifier" "-Wduplicate-enum"
  "-Wduplicate-method-arg" "-Wduplicate-method-match" "-Wextra-semi-stmt"
  "-Wfloat-equal" "-Wformat-pedantic" "-Wformat=2" "-Wheader-hygiene"
  "-Widiomatic-parentheses" "-Wimplicit-fallthrough" "-Wmain"
  "-Wmethod-signatures" "-Wmissing-noreturn" "-Wmissing-prototypes"
  "-Wmissing-variable-declarations" "-Wnewline-eof" "-Wnon-virtual-dtor"
  "-Wnonportable-system-include-path" "-Wold-style-cast" "-Wover-aligned"
  "-Wpacked" "-Wpointer-arith" "-Wprofile-instr-missing" "-Wredundant-parens"
  "-Wshift-sign-overflow" "-Wstatic-in-inline" "-Wstrict-prototypes"
  "-Wsuper-class-method-mismatch" "-Wswitch-enum" "-Wtautological-compare"
  "-Wtautological-constant-in-range-compare" "-Wundef"
  "-Wundefined-func-template" "-Wundefined-reinterpret-cast"
  "-Wunreachable-code-aggressive" "-Wunused-exception-parameter"
  "-Wunused-macros" "-Wunused-member-function" "-Wunused-template"
  "-Wused-but-marked-unused" "-Wvector-conversion" "-Wvla"
  "-Wweak-template-vtables" "-Wweak-vtables" "-Wzero-as-null-pointer-constant")
set(CLANG_LT_13_CXX_WARNING_FLAGS "-Wreserved-id-macro")
set(CLANG_GE_13_CXX_WARNING_FLAGS "-Wreserved-identifier")
# Do not enable clang 16+ -Wunsafe-buffer-usage because it warns on all memcpy
# calls and our pointer- and span-like class implementations.
set(CLANG_GE_21_CXX_WARNING_FLAGS "-Wnrvo" "-Wunique-object-duplication"
  "-Wshift-bool" "-Wc99-designator" "-Wms-bitfield-padding")

set(GCC_CXX_WARNING_FLAGS
  # Warning groups
  "-Wall" "-Wextra" "-Wpedantic" "-Wunused" "-Wparentheses" "-Wconversion"
  # Individual warnings
  "-Wabi-tag" "-Wcast-align=strict" "-Wcast-qual" "-Wcatch-value=3"
  "-Wctor-dtor-privacy" "-Wdouble-promotion" "-Wduplicated-branches"
  "-Wduplicated-cond" "-Wextra-semi" "-Wfloat-equal" "-Wformat-overflow=2"
  "-Wformat-signedness" "-Wformat-truncation=2" "-Wformat=2"
  "-Wimplicit-fallthrough=5" "-Winvalid-pch" "-Wlogical-op" "-Wmismatched-tags"
  "-Wmissing-include-dirs" "-Wnoexcept"
  "-Wnon-virtual-dtor" "-Wnull-dereference"  "-Wold-style-cast"
  "-Woverloaded-virtual" "-Wpacked" "-Wplacement-new=2" "-Wredundant-decls"
  "-Wshadow=global" "-Wsign-conversion" "-Wsign-promo" "-Wstrict-null-sentinel"
  "-Wstringop-truncation" "-Wsuggest-attribute=cold" "-Wsuggest-attribute=const"
  "-Wsuggest-attribute=format" "-Wsuggest-attribute=malloc"
  "-Wsuggest-attribute=noreturn" "-Wsuggest-attribute=pure"
  "-Wsuggest-final-methods" "-Wsuggest-final-types" "-Wsuggest-override"
  "-Wswitch-enum" "-Wtrampolines" "-Wundef" "-Wuninitialized"
  "-Wunsafe-loop-optimizations" "-Wunused-const-variable=2" "-Wunused-macros"
  "-Wuseless-cast" "-Wvector-operation-performance" "-Wvla"
  "-Wzero-as-null-pointer-constant" "-Wattribute-alias=2" "-Warray-bounds=2"
  "-Wredundant-tags")
set(GCC_GE_11_CXX_WARNING_FLAGS
  "-Wctad-maybe-unsupported" "-Wdeprecated-enum-enum-conversion"
  "-Wdeprecated-enum-float-conversion" "-Wvexing-parse")
set(GCC_GE_12_CXX_WARNING_FLAGS "-Winterference-size")
set(GCC_GE_14_CXX_WARNING_FLAGS "-Wnrvo" "-Welaborated-enum-base"
  "-Wdangling-reference")
set(GCC_GE_15_CXX_WARNING_FLAGS "-Wleading-whitespace=spaces"
  "-Wtrailing-whitespace=any")

set(UNIX_CXX_FLAGS "-g")

# clang 14 produces DWARF-5 by default, which Valgrind does not support yet
set(CLANG_GE_14_CXX_FLAGS "-gdwarf-4")

option(AVX2 "Enable AVX2 instructions on x86_64" ON)
if(AVX2)
  message(STATUS "Using AVX2 instructions on x86_64")
else()
  message(STATUS "Using SSE4.1 instructions on x86_64")
endif()

set(SPINLOCK_LOOP "PAUSE" CACHE STRING "Spinlock loop implementation type")
set_property(CACHE SPINLOCK_LOOP PROPERTY STRINGS PAUSE EMPTY)
if(SPINLOCK_LOOP STREQUAL "PAUSE")
  set(SPINLOCK_LOOP_VALUE 1)
elseif(SPINLOCK_LOOP STREQUAL "EMPTY")
  set(SPINLOCK_LOOP_VALUE 2)
else()
  message(FATAL_ERROR "SPINLOCK_LOOP must be one of: PAUSE, EMPTY")
endif()

option(STATS "Whether to compile in the statistics counters" ON)

# Disable the following warnings for MSVC:
# - "C4324: '...': structure was padded due to alignment specifier"
# - "C5030: attribute '...' is not recognized"
# - "C5072, ASAN enabled without debug information emission" - triggered by
#   msvc-release-asan preset
# - "C5246: '...': the initialization of a subobject should be wrapped in
#   braces": billions of std::array false positives with 17.2.2.
# - The other disabled warnings only activate on /Wall
# - "C5264: 'const' variable is not used": billions of false positives with 17.5
# - "C4714": Google Benchmark 1.9.1 fails to inline __forceinline functions
# - "C4514: unreferenced inline function has been removed"
set(MSVC_CXX_WARNING_FLAGS "/Wall" "/wd4324" "/wd5030" "/wd5072" "/wd4623"
  "/wd4625" "/wd4626" "/wd4582" "/wd4820" "/wd4710" "/wd4711" "/wd4868"
  "/wd5026" "/wd5027" "/wd5045" "/wd5246" "/wd5264" "/wd4714" "/wd4514")

# Flags for both MSVC cl and clang-cl compilers
set(ANY_MSVC_CXX_FLAGS "/permissive-")

# clang-cl warning flags that cl does not understand
set(MSVC_CLANG_WARNING_FLAGS "-Wno-c++20-compat" "-Wno-c++98-compat"
  "-Wno-c++98-compat-pedantic" "-Wno-exit-time-destructors"
  "-Wno-ctad-maybe-unsupported" "-Wno-global-constructors"
  "-Wno-unsafe-buffer-usage" "-Wno-switch-default")

# cl flags that clang-cl does not understand
set(MSVC_CXX_FLAGS "/external:anglebrackets" "/external:W0")

if(NOT DEFINED VSINSTALLDIR)
  set(VSINSTALLDIR NOTFOUND)
endif()

set(MSVC_STATIC_ANALYSIS_FLAGS "/analyze" "/analyze:external-"
  "/analyze:pluginEspXEngine.dll" "/analyze:rulesetdirectory$ENV{VSINSTALLDIR}"
  "/analyze:rulesetAllRules.ruleset"
  "/analyze:log:format:sarif" "/analyze:logmsvc.sarif"
  "/analyze:log:compilerwarnings"
  # Won't fix: "Symbol '...' is never tested for nullness, it can be marked as
  # not_null (f.23)"
  "/wd26429"
  # Won't fix: "Prefer to use gsl::at() instead of unchecked subscript operator
  # (bounds.4)"
  "/wd26446"
  # Won't fix: "Only index into arrays using constant expressions (bounds.2)"
  "/wd26482"
  # Won't fix: "Don't use a static_cast for arithmetic conversions. Use brace
  # initialization, gsl::narrow_cast or gsl::narrow (type.1)"
  "/wd26472"
  # Won't fix: "For '...', consider using gsl::span instead of std::span to
  # guarantee runtime bounds safety"
  "/wd26821")

option(COVERAGE "Enable code coverage reporting")
if(COVERAGE)
  if(MSVC)
    message(FATAL_ERROR "MSVC is incompatible with code code coverage reporting")
  endif()
  option(GCOV_PATH "gcov tool location to be used by lcov")
  if(GCOV_PATH)
    set(LCOV_GCOV_ARG "--gcov-tool" "${GCOV_PATH}")
    message(STATUS "Code coverage reporting enabled with gcov at ${GCOV_PATH}")
  else()
    set(LCOV_GCOV_ARG "")
    message(STATUS "Code coverage reporting enabled with default gcov path")
  endif()
else()
  set(GCOV_PATH NOTFOUND)
  message(STATUS "Code coverage reporting not enabled")
endif()

set(SANITIZER_CXX_FLAGS "")
set(SANITIZER_LD_FLAGS "")

macro(ADD_TO_GNU_SANITIZER_FLAGS)
  if(is_gxx)
    list(APPEND SANITIZER_CXX_FLAGS ${ARGV})
    list(APPEND SANITIZER_LD_FLAGS ${ARGV})
  endif()
endmacro()

macro(SET_COMMON_SANITIZER_FLAGS)
  if(NOT MSVC)
    list(APPEND SANITIZER_CXX_FLAGS "-fno-omit-frame-pointer"
        "-fno-optimize-sibling-calls")
  endif()
endmacro()

set(SANITIZER_ENV "")

option(SANITIZE_ADDRESS "Enable AddressSanitizer runtime checks")
if(SANITIZE_ADDRESS)
  set_common_sanitizer_flags()
  if(MSVC)
    list(APPEND SANITIZER_CXX_FLAGS "/fsanitize=address")
    # Documented to be incompatible with ASan
    string(REGEX REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG 
        "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REGEX REPLACE "/INCREMENTAL" "/INCREMENTAL:NO"
        CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
    string(REGEX REPLACE "/INCREMENTAL" "/INCREMENTAL:NO" 
        CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")
    string(REGEX REPLACE "/INCREMENTAL" "/INCREMENTAL:NO" 
        CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
    string(REGEX REPLACE "/INCREMENTAL" "/INCREMENTAL:NO" 
        CMAKE_STATIC_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG}")
  else()
    list(APPEND SANITIZER_CXX_FLAGS "-fsanitize=address")
    list(APPEND SANITIZER_LD_FLAGS "-fsanitize=address")
    add_to_gnu_sanitizer_flags("-fsanitize=leak"
        "-fsanitize-address-use-after-scope" "-fsanitize=pointer-compare"
        "-fsanitize=pointer-subtract")
  endif()
  string(CONCAT ASAN_ENV "ASAN_OPTIONS="
    "check_initialization_order=true:detect_stack_use_after_return=true:"
    "alloc_dealloc_mismatch=true:strict_string_checks=true")
  if(NOT is_gxx)
    # False positive on std::vector::capacity with GCC 12
    string(APPEND ASAN_ENV ":detect_invalid_pointer_pairs=2")
  endif()
  if(is_clang OR is_gxx)
    string(APPEND ASAN_ENV ":detect_leaks=1")
  endif()
  list(APPEND SANITIZER_ENV ${ASAN_ENV})
  unset(ASAN_ENV)
else()
  if(MSVC AND NOT is_clang)
    string(REGEX REPLACE "/Zi" "/ZI" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  endif()
endif()

option(SANITIZE_THREAD "Enable ThreadSanitizer runtime checks")
if(SANITIZE_THREAD)
  if(MSVC)
    message(FATAL_ERROR "MSVC is incompatible with ThreadSanitizer")
  endif()
  set_common_sanitizer_flags()
  list(APPEND SANITIZER_CXX_FLAGS "-fsanitize=thread")
  list(APPEND SANITIZER_LD_FLAGS "-fsanitize=thread")
endif()

option(SANITIZE_UB "Enable UndefinedBehaviorSanitizer runtime checks")
if(SANITIZE_UB)
  if(MSVC)
    message(FATAL_ERROR "MSVC is incompatible with UndefinedBehaviorSanitizer")
  endif()
  set_common_sanitizer_flags()
  list(APPEND SANITIZER_CXX_FLAGS "-fsanitize=undefined")
  if(is_any_clang)
    # It was interesting to try -fsanitize=implicit-conversion, but the expected
    # conversions do not appear to be fully suppressable
    list(APPEND SANITIZER_CXX_FLAGS "-fsanitize=local-bounds"
      "-fsanitize=nullability")
    if(is_clang AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.0)
      list(APPEND SANITIZER_CXX_FLAGS "-fsanitize=implicit-bitfield-conversion")
    endif()
    if(is_clang AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 21.0)
      list(APPEND SANITIZER_CXX_FLAGS "-fsanitize=vptr")
    endif()
  endif()

  set(SANITIZER_LD_FLAGS "-fsanitize=undefined")

  if(is_clang AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 21.0)
    list(APPEND SANITIZER_LD_FLAGS "-fsanitize=vptr")
  endif()

  string(CONCAT UBSAN_ENV "UBSAN_OPTIONS="
    "print_stacktrace=1:halt_on_error=1:abort_on_error=1")

  # clang 11 as packaged in Ubuntu 22.04 needs a workaround for
  # https://github.com/llvm/llvm-project/issues/55785
  if(is_clang11)
    string(APPEND UBSAN_ENV ":use_sigaltstack=false")
  endif()

  list(APPEND SANITIZER_ENV ${UBSAN_ENV})
  unset(UBSAN_ENV)
elseif(is_clang11)
  # Even though the sanitizer is not enabled here, DeepState will enable it
  # itself, and clang 11 as packaged in Ubuntu 22.04, needs a workaround for
  # https://github.com/llvm/llvm-project/issues/55785
  list(APPEND SANITIZER_ENV "UBSAN_OPTIONS=use_sigaltstack=false")
endif()

message(STATUS "SANITIZER_ENV: ${SANITIZER_ENV}")

option(STATIC_ANALYSIS "Enable compiler static analysis")

if(STANDALONE)
  if(MSVC AND is_clang)
    message(STATUS "Not enabling IPO/LTO due to MSVC LLVM not supporting it")
  elseif(is_clang AND SANITIZE_ADDRESS AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 17.0 AND
    CMAKE_CXX_COMPILER_VERSION VERSION_LESS 21.0)
    # /usr/bin/ld: error: Failed to link module
    # ../libunodb_qsbr.a.llvm.3080.qsbr.cpp: Expected at most one ThinLTO module
    # per bitcode file
    # ...
    message(STATUS
      "Not enabling IPO/LTO due to LLVM clang 17-20 w/ AddressSanitizer bug")
  else()
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_SUPPORT_ERROR LANGUAGES
        CXX)
    if(IPO_SUPPORTED)
      message(STATUS "Enabling IPO/LTO for release config")
    else()
      message(STATUS "IPO/LTO is not supported: ${IPO_SUPPORT_ERROR}")
    endif()
  endif()
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

find_package(Boost
  OPTIONAL_COMPONENTS stacktrace_basic stacktrace_backtrace stacktrace_windbg)

if(STATS AND NOT Boost_FOUND)
  message(FATAL_ERROR
    "Building with stats requires Boost, install it or use -DSTATS=OFF")
endif()

if(Boost_STACKTRACE_BACKTRACE_FOUND)
  message(STATUS "Boost.Stacktrace built with libbacktrace support, using it.")
  set(USE_BOOST_STACKTRACE ON)
elseif(Boost_STACKTRACE_WINDBG_FOUND)
  message(STATUS "Boost.Stacktrace built with WinDbg support, using it.")
  set(USE_BOOST_STACKTRACE ON)
elseif(Boost_STACKTRACE_BASIC_FOUND)
  if (is_gxx AND NOT is_darwin)
    message(STATUS "Boost.Stacktrace built without libbacktrace support, not using it.")
    set(USE_BOOST_STACKTRACE OFF)
  else()
    message(STATUS "Using Boost.Stacktrace without libbacktrace support.")
    set(USE_BOOST_STACKTRACE ON)
  endif()
else()
  set(USE_BOOST_STACKTRACE OFF)
endif()

# Generator expression helpers
# Platform
set(is_release_genex "$<CONFIG:Release>")
set(is_not_release_genex "$<NOT:${is_release_genex}>")
set(is_not_windows_x86_64 "$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>")
set(is_windows_x86_64 "$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},AMD64>")
set(is_windows_genex "$<PLATFORM_ID:Windows>")
set(is_not_windows "$<NOT:${is_windows_genex}>")
set(is_linux "$<PLATFORM_ID:Linux>")
# Compiler
set(is_gxx_genex "$<CXX_COMPILER_ID:GNU>")
set(is_clang_genex "$<CXX_COMPILER_ID:Clang>")
set(is_apple_clang_genex "$<CXX_COMPILER_ID:AppleClang>")
set(is_any_clang_genex "$<CXX_COMPILER_ID:AppleClang,Clang>")
set(is_msvc "$<CXX_COMPILER_ID:MSVC>")
# This condition is incorrect due to clang-cl having clang compiler ID but
# taking MSVC flags. Change to use $<CXX_COMPILER_FRONTEND_VARIANT> on 3.30
set(is_clang_cl "$<AND:${is_windows_genex},${is_clang_genex}>")
set(is_any_msvc "$<OR:${is_msvc},${is_clang_cl}>")
set(is_x86_64_any_msvc "$<AND:${is_any_msvc},${is_windows_x86_64}>")
set(is_clang_not_windows "$<AND:${is_clang_genex},${is_not_windows}>")
# Compiler version
set(cxx_ge_11 "$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,11.0>")
set(cxx_ge_12 "$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,12.0>")
set(cxx_lt_13 "$<VERSION_LESS:$<CXX_COMPILER_VERSION>,13.0>")
set(cxx_ge_13 "$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,13.0>")
set(cxx_ge_14 "$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,14.0>")
set(cxx_ge_15 "$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,15.0>")
set(cxx_ge_21 "$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,21.0>")
set(is_clang_lt_13_not_windows "$<AND:${is_clang_not_windows},${cxx_lt_13}>")
set(is_clang_ge_13_not_windows "$<AND:${is_clang_not_windows},${cxx_ge_13}>")
set(is_clang_ge_14_not_windows "$<AND:${is_clang_not_windows},${cxx_ge_14}>")
set(is_clang_ge_21_not_windows "$<AND:${is_clang_not_windows},${cxx_ge_21}>")
set(is_darwin_clang_ge_21_x86_64 "$<AND:$<PLATFORM_ID:Darwin>,${is_clang_ge_21_not_windows},${is_not_windows_x86_64}>")
set(is_darwin_clang_ge_21_arm64 "$<AND:$<PLATFORM_ID:Darwin>,${is_clang_ge_21_not_windows},$<NOT:${is_not_windows_x86_64}>>")
set(is_gxx_ge_11 "$<AND:${is_gxx_genex},${cxx_ge_11}>")
set(is_gxx_ge_12 "$<AND:${is_gxx_genex},${cxx_ge_12}>")
set(is_gxx_ge_14 "$<AND:${is_gxx_genex},${cxx_ge_14}>")
set(is_gxx_ge_15 "$<AND:${is_gxx_genex},${cxx_ge_15}>")
# Configuration
set(has_avx2 "$<BOOL:${AVX2}>")
set(use_boost_stacktrace "$<BOOL:${USE_BOOST_STACKTRACE}>")
set(with_stats "$<BOOL:${STATS}>")
set(fatal_warnings_on "$<BOOL:${MAINTAINER_MODE}>")
set(coverage_on "$<BOOL:${COVERAGE}>")
set(is_standalone "$<BOOL:${STANDALONE}>")
set(asan_on "$<BOOL:${SANITIZE_ADDRESS}>")
# Checks across multiple categories
set(is_standalone_not_release "$<AND:${is_standalone},${is_not_release_genex}>")
set(is_standalone_asan "$<AND:${is_standalone},${asan_on}>")

option(TESTS "Build tests, including fuzz ones if DeepState is available" ON)

option(BENCHMARKS "Build benchmarks" ${STANDALONE})

if((STANDALONE OR NOT TARGET GTest::gtest_main
    OR NOT TARGET benchmark::benchmark) AND (TESTS OR BENCHMARKS))
  string(REPLACE ";" " " CXX_FLAGS_FOR_SUBDIR_STR "${SANITIZER_CXX_FLAGS}")
  if(STANDALONE AND MSVC)
    string(REPLACE ";" " " MSVC_WARNING_FLAGS_FOR_SUBDIR_STR_TMP
      "${MSVC_CXX_WARNING_FLAGS}")
    string(REPLACE "/Wall" "" MSVC_WARNING_FLAGS_FOR_SUBDIR_STR
      "${MSVC_WARNING_FLAGS_FOR_SUBDIR_STR_TMP}")
  endif()
  string(REPLACE ";" " " LD_FLAGS_FOR_SUBDIR_STR "${SANITIZER_LD_FLAGS}")

  macro(ADD_CXX_FLAGS_FOR_SUBDIR)
    set(ORIG_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(ORIG_CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})
    set(ORIG_CMAKE_MODULE_LINKER_FLAGS ${CMAKE_MODULE_LINKER_FLAGS})
    set(ORIG_CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS})
    string(APPEND CMAKE_CXX_FLAGS " " "${CXX_FLAGS_FOR_SUBDIR_STR}")
    if(MSVC)
      string(APPEND CMAKE_CXX_FLAGS " " "${MSVC_WARNING_FLAGS_FOR_SUBDIR_STR}")
    endif()
    if(is_not_msvc_clang
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0)
      string(APPEND CMAKE_CXX_FLAGS " " "${CLANG_GE_14_CXX_FLAGS}")
    endif()
    string(APPEND CMAKE_EXE_LINKER_FLAGS "${LD_FLAGS_FOR_SUBDIR_STR}")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS "${LD_FLAGS_FOR_SUBDIR_STR}")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS "${LD_FLAGS_FOR_SUBDIR_STR}")
  endmacro()

  macro(RESTORE_CXX_FLAGS_FOR_SUBDIR)
    set(CMAKE_CXX_FLAGS ${ORIG_CMAKE_CXX_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS ${ORIG_CMAKE_EXE_LINKER_FLAGS})
    set(CMAKE_MODULE_LINKER_FLAGS ${ORIG_CMAKE_MODULE_LINKER_FLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS ${ORIG_CMAKE_SHARED_LINKER_FLAGS})
  endmacro()
endif()

# C++-related properties that must be set on every compiled target. Applies to
# dependencies too in the case of standalone build.
function(GLOBAL_CXX_TARGET_PROPERTIES TARGET)
  target_compile_definitions(${TARGET} PUBLIC
    "$<${is_standalone_not_release}:_GLIBCXX_DEBUG>"
    "$<${is_standalone_not_release}:_GLIBCXX_DEBUG_PEDANTIC>"
    "$<${is_standalone_asan}:_GLIBCXX_SANITIZE_VECTOR>"
    "$<$<AND:${is_any_msvc},${is_standalone_not_release}>:_MSVC_STL_HARDENING=1>"
    "$<$<AND:${is_any_msvc},${is_standalone_not_release}>:_MSVC_STL_DESTRUCTOR_TOMBSTONES=1>")
endfunction()

if(TESTS)
  if (STANDALONE OR NOT TARGET GTest::gtest_main)
    # For Windows: Prevent overriding the parent project's compiler/linker
    # settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    ADD_CXX_FLAGS_FOR_SUBDIR()
    if(is_gxx)
      string(APPEND CMAKE_CXX_FLAGS " -Wno-maybe-uninitialized")
    endif()
    if(is_clang)
      string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-declarations")
    endif()
    add_subdirectory(3rd_party/googletest)
    RESTORE_CXX_FLAGS_FOR_SUBDIR()

    global_cxx_target_properties(gtest)
    global_cxx_target_properties(gtest_main)
  endif()

  # Do not build DeepState:
  # - under Windows as it's not supported
  # - on anything else than x86_64
  # - if 32-bit build is not possible
  # - with GCC under macOS due to
  #   https://github.com/trailofbits/deepstate/issues/374
  # - under macOS with ASan or TSan enabled
  if(STANDALONE
      AND NOT (is_darwin AND (is_gxx OR SANITIZE_ADDRESS OR SANITIZE_THREAD))
      AND NOT is_windows AND is_x86_64)
    CHECK_INCLUDE_FILE(stdio.h CAN_BUILD_32BIT -m32)

    if(CAN_BUILD_32BIT)
      message(STATUS "32-bit compilation supported, building DeepState")

      # ThreadSanitizer is not compatible with libfuzzer and LLVM 11 linker
      # crashes on libfuzzer release build
      if(is_clang AND NOT(is_release) AND NOT(SANITIZE_THREAD))
        set(LIBFUZZER_AVAILABLE TRUE)
        set(BUILD_DEEPSTATE_LIBFUZZER "-DDEEPSTATE_LIBFUZZER=ON")
      else()
        set(LIBFUZZER_AVAILABLE FALSE)
        set(BUILD_DEEPSTATE_LIBFUZZER "-DDEEPSTATE_LIBFUZZER=OFF")
      endif()

      include(ExternalProject)
      ExternalProject_Add(3rd_party_deepstate
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rd_party/deepstate"
        BINARY_DIR "${CMAKE_BINARY_DIR}/3rd_party/deepstate"
        CMAKE_ARGS "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" "-Wno-dev"
        "--no-warn-unused-cli"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_C_FLAGS=-w -Wno-implicit-function-declaration"
        "-DCMAKE_CXX_FLAGS=-w" "${BUILD_DEEPSTATE_LIBFUZZER}"
        INSTALL_COMMAND "")
      ExternalProject_Get_property(3rd_party_deepstate SOURCE_DIR)
      ExternalProject_Get_property(3rd_party_deepstate BINARY_DIR)

      add_library(deepstate STATIC IMPORTED)
      add_dependencies(deepstate 3rd_party_deepstate)
      target_include_directories(deepstate INTERFACE "${SOURCE_DIR}/src/include/")
      set_target_properties(deepstate PROPERTIES IMPORTED_LOCATION
        "${BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}deepstate${CMAKE_STATIC_LIBRARY_SUFFIX}")

      if(LIBFUZZER_AVAILABLE)
        add_library(deepstate_lf STATIC IMPORTED)
        add_dependencies(deepstate_lf 3rd_party_deepstate)
        target_include_directories(deepstate_lf INTERFACE
          "${SOURCE_DIR}/src/include/")
        set_target_properties(deepstate_lf PROPERTIES IMPORTED_LOCATION
          "${BINARY_DIR}/libdeepstate_LF.a")
      endif()
    else()
      message(STATUS
        "32-bit compilation not supported, skipping DeepState build")
    endif()
  endif()
endif()

if(BENCHMARKS)
  if(STANDALONE OR NOT TARGET benchmark::benchmark)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
      "Suppressing Google Benchmark tests" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL
      "Suppressing Google Benchmark installation" FORCE)

    if(IPO_SUPPORTED)
      if(is_debug)
        # It seems that Google Benchmark does not support multi-configuration
        # generators if LTO is enabled
        message(STATUS "Disabling LTO for Google Benchmark due to debug build")
      elseif(is_apple_clang)
        message(STATUS
          "Disabling LTO for Google Benchmark because Apple clang is not \
           supported")
      else()
        set(BENCHMARK_ENABLE_LTO ON CACHE BOOL
          "Enabling LTO for Google Benchmark" FORCE)
        message(STATUS "Enabling LTO for Google Benchmark")
      endif()
    endif()

    if(is_clang11)
      # clang 11 as packaged in Ubuntu 22.04 needs a configure-time workaround
      # for https://github.com/llvm/llvm-project/issues/55785
      if(SANITIZE_ADDRESS)
        set(ENV{ASAN_OPTIONS} "use_sigaltstack=false")
      endif()
      if(SANITIZE_THREAD)
        set(ENV{TSAN_OPTIONS} "use_sigaltstack=false")
      endif()
      if(SANITIZE_UB)
        set(ENV{UBSAN_OPTIONS} "use_sigaltstack=false")
      endif()
    endif()

    ADD_CXX_FLAGS_FOR_SUBDIR()
    add_subdirectory(3rd_party/benchmark)
    RESTORE_CXX_FLAGS_FOR_SUBDIR()

    global_cxx_target_properties(benchmark)
    global_cxx_target_properties(benchmark_main)
  endif()

  # Add benchmark_include_dirs by target_include_directories(... SYSTEM ...)
  # before target_link_libraries so that benchmark headers are included through
  # -isystem not -I, resulting in build-breaking diagnostics.
  get_target_property(benchmark_include_dirs benchmark::benchmark
    INTERFACE_INCLUDE_DIRECTORIES)
endif()

if(NOT is_any_clang)
  message(STATUS "Not using clang-tidy due to non-clang compiler being used")
  set(DO_CLANG_TIDY "")
elseif(MSVC)
  message(STATUS "Not using clang-tidy due to broken (?) MSVC support")
else()
  find_program(CLANG_TIDY_EXE NAMES "clang-tidy"
    DOC "Path to clang-tidy executable")
  if(NOT CLANG_TIDY_EXE)
    message(STATUS "clang-tidy not found")
    set(DO_CLANG_TIDY "")
  else()
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-p=${CMAKE_BINARY_DIR}")
    # Add suppression for unknown warning options when clang-tidy might not
    # recognize all compiler warning flags (e.g., when scan-build from LLVM 21
    # wraps the build)
    list(APPEND DO_CLANG_TIDY "--extra-arg=-Wno-unknown-warning-option")
  endif()
endif()

set(CPPCHECK_ARGS
  # False positive on Google Test TEST macro and compilers are much better
  # for syntax checking anyway
  "--suppress=syntaxError"
  # Non-actionable informational message that even fails the build before
  # cppcheck 2.11.
  "--suppress=normalCheckLevelMaxBranches")

option(CPPCHECK_AGGRESSIVE "Enable inconclusive cppcheck checks")
if(CPPCHECK_AGGRESSIVE)
  list(APPEND CPPCHECK_ARGS "--inconclusive")
endif()
if(MAINTAINER_MODE)
  list(APPEND CPPCHECK_ARGS "--error-exitcode=2")
else()
  list(APPEND CPPCHECK_ARGS "--error-exitcode=0")
endif()

find_program(CPPCHECK_EXE NAMES "cppcheck" DOC "Path to cppcheck executable")
if(NOT CPPCHECK_EXE)
  message(STATUS "cppcheck not found")
else()
  execute_process(COMMAND "${CPPCHECK_EXE}" "--version" OUTPUT_VARIABLE
    CPPCHECK_VERSION_OUTPUT)
  message(STATUS
    "cppcheck found: ${CPPCHECK_EXE}, --version: ${CPPCHECK_VERSION_OUTPUT}")
  set(DO_CPPCHECK "${CPPCHECK_EXE}"
    "--enable=warning,style,performance,portability" "-D__x86_64" "-D__GLIBCXX__"
    "--inline-suppr" "--quiet")
  list(APPEND DO_CPPCHECK "${CPPCHECK_ARGS}")
endif()

find_program(CPPLINT_EXE NAMES "cpplint" DOC "Path to cpplint executable")
if(NOT CPPLINT_EXE)
  message(STATUS "cpplint not found")
else()
  message(STATUS "cpplint found: ${CPPLINT_EXE}")
  list(APPEND CPPLINT_EXE "--quiet")
endif()

option(IWYU "Enable include-what-you-use checking")
if(IWYU)
  if(NOT is_any_clang OR is_windows)
    message(STATUS
      "Not using include-what-you-use due to non-clang compiler or Windows being used")
  else()
    find_program(IWYU_EXE NAMES "include-what-you-use"
      DOC "Path to include-what-you-use executable")
    if(NOT IWYU_EXE)
      message(FATAL_ERROR "include-what-you-use not found")
    else()
      execute_process(COMMAND "${IWYU_EXE}" "--version" OUTPUT_VARIABLE
        IWYU_VERSION_OUTPUT)
      message(STATUS
        "include-what-you-use found: ${IWYU_EXE}, --version: ${IWYU_VERSION_OUTPUT}")
      # IWYU is not necessarily based on the same version clang is and might not
      # understand newer options
      set(DO_IWYU "${IWYU_EXE};-Wno-unknown-warning-option")
    endif()
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(COMMON_TARGET_PROPERTIES TARGET)
  cmake_parse_arguments(PARSE_ARGV 1 CTP "SKIP_CHECKS" "" "")
  target_compile_features(${TARGET} PUBLIC cxx_std_20)
  set_target_properties(${TARGET} PROPERTIES CXX_EXTENSIONS OFF)
  global_cxx_target_properties(${TARGET})
  target_compile_definitions(${TARGET} PUBLIC
    # Features
    "$<${is_standalone}:UNODB_DETAIL_STANDALONE>"
    "$<${use_boost_stacktrace}:UNODB_DETAIL_BOOST_STACKTRACE>"
    "$<${with_stats}:UNODB_DETAIL_WITH_STATS>"
    "UNODB_SPINLOCK_LOOP_VALUE=${SPINLOCK_LOOP_VALUE}")
  target_compile_options(${TARGET} PUBLIC
    # Architecture
    "$<${is_x86_64_any_msvc}:$<IF:${has_avx2},/arch:AVX2,/arch:AVX>>"
    "$<${is_not_windows_x86_64}:$<IF:${has_avx2},-mavx2,-msse4.1>>")
  target_compile_options(${TARGET} PRIVATE
    "${SANITIZER_CXX_FLAGS}"
    "$<${is_msvc}:${MSVC_CXX_FLAGS}>"
    "$<${is_any_msvc}:${ANY_MSVC_CXX_FLAGS}>"
    "$<${is_not_windows}:${UNIX_CXX_FLAGS}>"
    "$<${is_clang_ge_14_not_windows}:${CLANG_GE_14_CXX_FLAGS}>"
    # Warnings
    "$<${fatal_warnings_on}:$<IF:${is_any_msvc},/WX,-Werror>>"
    "$<$<AND:${is_standalone},${is_any_msvc}>:${MSVC_CXX_WARNING_FLAGS}>"
    "$<$<AND:${is_standalone},${is_clang_cl}>:${MSVC_CLANG_WARNING_FLAGS}>"
    "$<$<AND:${is_standalone},${is_any_clang_genex},${is_not_windows}>:${CLANG_CXX_WARNING_FLAGS}>"
    "$<$<AND:${is_standalone},${is_clang_lt_13_not_windows}>:${CLANG_LT_13_CXX_WARNING_FLAGS}>"
    "$<$<AND:${is_standalone},${is_clang_ge_13_not_windows}>:${CLANG_GE_13_CXX_WARNING_FLAGS}>"
    "$<$<AND:${is_standalone},${is_clang_ge_21_not_windows}>:${CLANG_GE_21_CXX_WARNING_FLAGS}>"
    "$<$<AND:${is_standalone},${is_gxx_genex}>:${GCC_CXX_WARNING_FLAGS}>"
    "$<$<AND:${is_standalone},${is_gxx_ge_11}>:${GCC_GE_11_CXX_WARNING_FLAGS}>"
    "$<$<AND:${is_standalone},${is_gxx_ge_12}>:${GCC_GE_12_CXX_WARNING_FLAGS}>"
    "$<$<AND:${is_standalone},${is_gxx_ge_14}>:${GCC_GE_14_CXX_WARNING_FLAGS}>"
    "$<$<AND:${is_standalone},${is_gxx_ge_15}>:${GCC_GE_15_CXX_WARNING_FLAGS}>"
    # Optimization
    "$<${is_not_release_genex}:$<IF:${is_windows_genex},/Od,-O0>>"
    "$<$<AND:${is_release_genex},${is_not_windows}>:$<IF:${coverage_on},-O0,-O3>>"
    # Misc
    "$<${coverage_on}:--coverage>")
  target_link_options(${TARGET} INTERFACE "$<${coverage_on}:--coverage>")
  target_link_options(${TARGET} PRIVATE
    "$<${is_apple_clang_genex}:-Wl,-no_warn_duplicate_libraries>"
    "$<${is_darwin_clang_ge_21_x86_64}:-L/usr/local/opt/llvm/lib/c++>"
    "$<${is_darwin_clang_ge_21_x86_64}:-Wl,-rpath,/usr/local/opt/llvm/lib/c++>"
    "$<${is_darwin_clang_ge_21_arm64}:-L/opt/homebrew/opt/llvm/lib/c++>"
    "$<${is_darwin_clang_ge_21_arm64}:-Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++>"
    "${SANITIZER_LD_FLAGS}"
  )
  target_link_libraries(${TARGET} PRIVATE
    "$<${is_linux}:${CMAKE_DL_LIBS}>"
    "$<$<AND:${is_linux},${is_gxx_genex},${use_boost_stacktrace}>:backtrace>")
  if(IPO_SUPPORTED)
    set_target_properties(${TARGET} PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
  if(NOT CTP_SKIP_CHECKS)
    if(CPPCHECK_EXE)
      set_target_properties(${TARGET} PROPERTIES CXX_CPPCHECK "${DO_CPPCHECK}")
    endif()
    if(CPPLINT_EXE)
      set_target_properties(${TARGET} PROPERTIES CXX_CPPLINT "${CPPLINT_EXE}")
    endif()
    if(IWYU_EXE)
      set_target_properties(${TARGET} PROPERTIES CXX_INCLUDE_WHAT_YOU_USE "${DO_IWYU}")
    endif()
    if(STATIC_ANALYSIS)
      # The condition below is incorrect due to clang-cl having clang compiler
      # ID but taking MSVC flags. Change to $<CXX_COMPILER_FRONTEND_VARIANT>
      # on 3.30.
      target_compile_options(${TARGET} PRIVATE
        "$<IF:${is_msvc},${MSVC_STATIC_ANALYSIS_FLAGS},-fanalyzer>"
        # GCC 12 -fanalyzer not fully ready for C++ yet
        "$<${is_gxx_ge_11}:-Wno-analyzer-null-dereference>"
        "$<${is_gxx_ge_11}:-Wno-analyzer-possible-null-argument>"
        "$<${is_gxx_ge_11}:-Wno-analyzer-possible-null-dereference>"
        "$<${is_gxx_ge_11}:-Wno-analyzer-malloc-leak>"
        "$<${is_gxx_ge_11}:-Wno-analyzer-null-argument>")
    endif()
  endif()
endfunction()

function(SET_CLANG_TIDY_OPTIONS TARGET COMMAND)
  if(DO_CLANG_TIDY)
    set_target_properties(${TARGET} PROPERTIES CXX_CLANG_TIDY "${COMMAND}")
  endif()
endfunction()

function(ADD_UNODB_LIBRARY LIB)
  add_library(${LIB} ${ARGN})
  common_target_properties(${LIB})
  set_clang_tidy_options(${LIB} "${DO_CLANG_TIDY}")

  if(LIBFUZZER_AVAILABLE)
    set(LIB_LF "${LIB}_lf")
    add_library(${LIB_LF} ${ARGN})
    common_target_properties(${LIB_LF} SKIP_CHECKS)
    target_compile_options(${LIB_LF} PRIVATE "-fsanitize=fuzzer-no-link")
  endif()
endfunction()

add_library(unodb_util INTERFACE)
target_include_directories(unodb_util INTERFACE ".")
if(Boost_FOUND)
  target_include_directories(unodb_util SYSTEM INTERFACE ${Boost_INCLUDE_DIRS})
endif()

add_unodb_library(unodb_qsbr qsbr.cpp qsbr.hpp qsbr_ptr.cpp qsbr_ptr.hpp)
if(Boost_FOUND)
  target_include_directories(unodb_qsbr SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
  target_link_libraries(unodb_qsbr PRIVATE ${Boost_LIBRARIES})
endif()
target_link_libraries(unodb_qsbr PUBLIC unodb_util Threads::Threads)
if(LIBFUZZER_AVAILABLE)
  if(Boost_FOUND)
    target_include_directories(unodb_qsbr_lf SYSTEM PUBLIC
      ${Boost_INCLUDE_DIRS})
    target_link_libraries(unodb_qsbr_lf PRIVATE ${Boost_LIBRARIES})
  endif()
  target_link_libraries(unodb_qsbr_lf PUBLIC unodb_util Threads::Threads)
endif()

add_unodb_library(unodb art.hpp art_common.hpp mutex_art.hpp optimistic_lock.hpp
  art_internal_impl.hpp olc_art.hpp art_internal.hpp art_internal.cpp
  node_type.hpp duckdb_encode_decode.hpp)
target_link_libraries(unodb PUBLIC unodb_util unodb_qsbr)
if(LIBFUZZER_AVAILABLE)
  target_link_libraries(unodb_lf PUBLIC unodb_util unodb_qsbr_lf)
endif()

set(VALGRIND_COMMAND "valgrind" "--error-exitcode=1" "--leak-check=full"
  "--trace-children=yes" "-v")

# If tests and/or benchmarks are enabled, their Valgrind targets will become
# dependencies of the main Valgrind target.
add_custom_target(valgrind)

if(TESTS)
  add_dependencies(valgrind valgrind_tests)
  enable_testing()

  add_unodb_library(unodb_test test_heap.cpp)
  target_link_libraries(unodb_test PUBLIC unodb_util)
  target_link_libraries(unodb_test PRIVATE ${Boost_LIBRARIES})

  function(add_sanitized_test)
    cmake_parse_arguments(AST "" "NAME" "COMMAND" ${ARGN})
    add_test(NAME "${AST_NAME}" COMMAND ${AST_COMMAND})
    # Set Sanitizer environment on a sanitizer-less build too, if using clang
    # 11 and DeepState is built, because it enables sanitizers itself. clang 11
    # handling is to workaround
    # https://github.com/llvm/llvm-project/issues/55785 on Ubuntu 22.04.
    if(SANITIZE_ADDRESS OR SANITIZE_THREAD OR SANITIZE_UB
        OR (TARGET deepstate AND is_clang11))
      set_property(TEST "${AST_NAME}" APPEND PROPERTY ENVIRONMENT
        "${SANITIZER_ENV}")
    endif()
  endfunction()

  if(TARGET deepstate)
    add_subdirectory(fuzz_deepstate)
    add_dependencies(valgrind valgrind_deepstate)
  endif()
  add_subdirectory(test)
endif()

if(BENCHMARKS)
  add_dependencies(valgrind valgrind_benchmarks)
  add_subdirectory(benchmark)
endif()

if(STANDALONE)
  set(BIN_DIR_CDB "${CMAKE_BINARY_DIR}/compile_commands.json")
  set(SRC_DIR_CDB "${CMAKE_SOURCE_DIR}/compile_commands.json")

  # cmake -E create_symlink has Windows implementation, but additional
  # permissions are required.
  if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${BIN_DIR_CDB} ${SRC_DIR_CDB})
  endif()
endif()

message(STATUS "Build configuration and user-set CMake options:")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "STANDALONE: ${STANDALONE}")
message(STATUS "AVX2: ${AVX2}")
message(STATUS "SPINLOCK_LOOP: ${SPINLOCK_LOOP}")
message(STATUS "STATS: ${STATS}")
message(STATUS "TESTS: ${TESTS}")
message(STATUS "BENCHMARKS: ${BENCHMARKS}")
message(STATUS "MAINTAINER_MODE: ${MAINTAINER_MODE}")
message(STATUS "COVERAGE: ${COVERAGE}")
message(STATUS "GCOV_PATH: ${GCOV_PATH}")
message(STATUS "SANITIZE_ADDRESS: ${SANITIZE_ADDRESS}")
message(STATUS "SANITIZE_THREAD: ${SANITIZE_THREAD}")
message(STATUS "SANITIZE_UB: ${SANITIZE_UB}")
message(STATUS "STATIC_ANALYSIS: ${STATIC_ANALYSIS}")
message(STATUS "CPPCHECK_AGGRESSIVE: ${CPPCHECK_AGGRESSIVE}")
message(STATUS "IWYU: ${IWYU}")
