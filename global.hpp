// Copyright 2019-2026 UnoDB contributors
#ifndef UNODB_DETAIL_GLOBAL_HPP
#define UNODB_DETAIL_GLOBAL_HPP

/// \file
/// Global defines that must precede every other include directive in all the
/// source files.
/// \ingroup internal

/// \addtogroup internal
/// \{

// Macros that have multiple definitions are documented once.

/// \def UNODB_DETAIL_BUILTIN_ASSUME(condition)
/// \hideinitializer
/// Low-level macro to hint the compiler that the \a condition is true.
/// Should not be used directly: use UNODB_DETAIL_ASSUME() in assert.hpp
/// instead.

/// \def UNODB_DETAIL_LIFETIMEBOUND
/// Indicate that a reference parameter lifetime is bound to that of the return
/// value.

/// \def UNODB_DETAIL_C_STRING_ARG(x)
/// Mark a parameter as a null-terminated C string.
/// \param x The 1-based parameter index to be marked

/// \def UNODB_DETAIL_LIKELY(condition)
/// \hideinitializer
/// Hint the compiler that the \a condition is likely true.
/// \note Once clang 12 is the minimum supported version, most uses should be
/// migrated to C++20 `[[likely]]`.

/// \def UNODB_DETAIL_UNLIKELY(condition)
/// \hideinitializer
/// Hint the compiler that the \a condition is likely false.
/// \note Once clang 12 is the minimum supported version, most uses should be
/// migrated to C++20 `[[unlikely]]`.

/// \def UNODB_DETAIL_UNUSED
/// \hideinitializer
/// Mark a declaration as intentionally unused to suppress compiler warnings

/// \def UNODB_DETAIL_FORCE_INLINE
/// \hideinitializer
/// Provide the strongest available hint to the compiler to inline the function

/// \def UNODB_DETAIL_NOINLINE
/// \hideinitializer
/// Ask the compiler to not inline the function

/// \def UNODB_DETAIL_UNREACHABLE
/// \hideinitializer
/// Low-level macro to indicate an unreachable code.
/// Should not be used directly: use #UNODB_DETAIL_CANNOT_HAPPEN in
/// assert.hpp instead.

/// \def UNODB_DETAIL_CONSTEXPR_NOT_MSVC
/// \hideinitializer
/// Expand to `constexpr` with all compilers except for MSVC.
/// Use to mark `constexpr` declarations that are not supported by MSVC.

/// \def UNODB_DETAIL_ADDRESS_SANITIZER
/// Defined when compiling with AddressSanitizer

/// \def UNODB_DETAIL_THREAD_SANITIZER
/// Defined when compiling with ThreadSanitizer

/// \def UNODB_DETAIL_DO_PRAGMA(x)
/// Helper macro for creating pragma directives

/// \def UNODB_DETAIL_DISABLE_WARNING(x)
/// Disable a GCC or clang compiler warning \a x until
/// UNODB_DETAIL_RESTORE_WARNINGS().

/// \def UNODB_DETAIL_RESTORE_WARNINGS()
/// Re-enable the warning that was previously disabled with
/// UNODB_DETAIL_DISABLE_WARNING().

/// \def UNODB_DETAIL_DISABLE_MSVC_WARNING(x)
/// Disable an MSVC warning \a x until UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// \def UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
/// Re-enable the warning that was previously disabled with
/// UNODB_DETAIL_DISABLE_MSVC_WARNING().

/// \def UNODB_DETAIL_DISABLE_CLANG_WARNING(x)
/// Disable a clang warning \a x until UNODB_DETAIL_RESTORE_CLANG_WARNINGS()

/// \def UNODB_DETAIL_RESTORE_CLANG_WARNINGS()
/// Re-enable the warning that was previously disabled with
/// UNODB_DETAIL_DISABLE_CLANG_WARNING().

/// \def UNODB_DETAIL_DISABLE_CLANG_21_WARNING(x)
/// Disable a clang version 21 or later warning \a x until
/// UNODB_DETAIL_RESTORE_CLANG_21_WARNINGS().

/// \def UNODB_DETAIL_RESTORE_CLANG_21_WARNINGS()
/// Re-enable the warning that was previously disabled with
/// UNODB_DETAIL_DISABLE_CLANG_21_WARNING().

/// \def UNODB_DETAIL_DISABLE_GCC_WARNING(x)
/// Disable a GCC warning \a x until UNODB_DETAIL_RESTORE_GCC_WARNINGS()

/// \def UNODB_DETAIL_RESTORE_GCC_WARNINGS()
/// Re-enable the warning that was previously disabled with
/// UNODB_DETAIL_DISABLE_GCC_WARNING().

/// \def UNODB_DETAIL_DISABLE_GCC_10_WARNING(x)
/// Disable a GCC version 10 warning \a x until
/// UNODB_DETAIL_RESTORE_GCC_WARNINGS().

/// \def UNODB_DETAIL_RESTORE_GCC_10_WARNINGS()
/// Re-enable the warning that was previously disabled with
/// UNODB_DETAIL_DISABLE_GCC_10_WARNING().

/// \def UNODB_DETAIL_DISABLE_GCC_11_WARNING(x)
/// Disable a GCC version 11 or later warning \a x until
/// UNODB_DETAIL_RESTORE_GCC_WARNINGS().

/// \def UNODB_DETAIL_RESTORE_GCC_11_WARNINGS()
/// Re-enable the warning that was previously disabled with
/// UNODB_DETAIL_DISABLE_GCC_11_WARNING().

/// \def UNODB_DETAIL_RELEASE_CONSTEXPR
/// Expands to `constexpr` in release builds, empty in debug builds

/// \def UNODB_DETAIL_RELEASE_CONST
/// Expands to `const` in release builds, empty in debug builds

/// \def UNODB_DETAIL_RELEASE_EXPLICIT
/// Expands to `explicit` in release builds, empty in debug builds

/// \def UNODB_DETAIL_USED_IN_DEBUG
/// Marks a declaration as intentionally unused in release builds, but used in
/// debug ones. Suppresses a compiler warning.

/// \name CMake macros
/// Macros set by CMake.
/// \{
// This section is never compiled in, only processed by Doxygen
#ifdef UNODB_DETAIL_DOXYGEN

/// Defined when UnoDB is built as a standalone project rather than as a part of
/// another project.
#define UNODB_DETAIL_STANDALONE

/// Defined when UnoDB is compiled with the statistics counters.
#define UNODB_DETAIL_WITH_STATS

/// Defined when UnoDB is compiled with Boost.Stacktrace.
#define UNODB_DETAIL_BOOST_STACKTRACE

/// Defined to the selected value of the optimistic spin lock wait algorithm
/// implementation for unodb::spin_wait_loop_body(). It is also used by the OLC
/// ART algorithms restarting close to the start of their execution. The
/// possible values are #UNODB_DETAIL_SPINLOCK_LOOP_PAUSE and
/// #UNODB_DETAIL_SPINLOCK_LOOP_EMPTY.
#define UNODB_DETAIL_SPINLOCK_LOOP_VALUE

/// Enables the libstdc++ debug mode when compiling in the standalone debug
/// configuration.
#define _GLIBCXX_DEBUG

/// Enables erroring on the use of libstdc++-specific behaviors and extensions
/// when compiling in the standalone debug configuration.
#define _GLIBCXX_DEBUG_PEDANTIC

/// Annotates `std::vector` in libstdc++ for AddressSanitizer when compiling
/// with AddressSanitizer.
#define _GLIBCXX_SANITIZE_VECTOR

/// Enables the MSVC STL debug checks (iterator debugging, bounds checking)
/// when compiling in the standalone debug configuration.
#define _MSVC_STL_HARDENING

/// Enables the MSVC STL destructor tombstone feature to catch use-after-
/// destruction bugs when compiling in the standalone debug configuration.
#define _MSVC_STL_DESTRUCTOR_TOMBSTONES

#endif  // UNODB_DETAIL_DOXYGEN
/// \}

#ifdef _MSC_VER
/// Defined under MSVC to stop redefining `min` and `max` if windows.h is
/// included later.
#define NOMINMAX
#if !defined(NDEBUG) && defined(__SANITIZE_ADDRESS__)
// Workaround bug of _aligned_free not being hooked for ASan under MSVC debug
// build -
// https://developercommunity.visualstudio.com/t/asan-check-failed-using-aligned-free-in-debug-buil/1406956
#undef _CRTDBG_MAP_ALLOC
#endif
#endif

/// \name Architecture macros
/// Macros for the CPU architecture and instruction set level.
/// \{

#if defined(_MSC_VER) && defined(_M_X64)
/// Defined when compiling with MSVC on x86-64
#define UNODB_DETAIL_MSVC_X86_64
#endif

#if defined(__x86_64) || defined(UNODB_DETAIL_MSVC_X86_64)
/// Defined when compiling on x86_64
#define UNODB_DETAIL_X86_64
#endif

#ifdef UNODB_DETAIL_X86_64
#ifdef __AVX2__
/// Defined when compiling with AVX2 instructions on x86-64
#define UNODB_DETAIL_AVX2
#else
/// Defined when compiling with SSE4.2 and not AVX2 instructions on x86-64
#define UNODB_DETAIL_SSE4_2
#endif
#endif

/// \}

/// \name Compiler macros
/// Macros to hide compiler specifics
/// \{

#if defined(_MSC_VER)
#if !defined(__clang__)
/// Defined on MSVC with the MSVC frontend, not the LLVM one
#define UNODB_DETAIL_MSVC
#else  // #if !defined(__clang__)
/// Defined on MSVC with the LLVM frontend, not the MSVC one
#define UNODB_DETAIL_MSVC_CLANG
#endif  // #if !defined(__clang__)
#endif  // #if defined(_MSC_VER)

#ifndef UNODB_DETAIL_MSVC

#ifdef __clang__

#define UNODB_DETAIL_BUILTIN_ASSUME(condition) __builtin_assume(condition)
#define UNODB_DETAIL_LIFETIMEBOUND [[clang::lifetimebound]]
#define UNODB_DETAIL_C_STRING_ARG(x)

#else

#define UNODB_DETAIL_BUILTIN_ASSUME(condition) \
  do {                                         \
    if (!(condition)) __builtin_unreachable(); \
  } while (0)

#define UNODB_DETAIL_LIFETIMEBOUND

#if __GNUG__ >= 14

#define UNODB_DETAIL_C_STRING_ARG(x) \
  __attribute__((null_terminated_string_arg(x)))

#else

#define UNODB_DETAIL_C_STRING_ARG(x)

#endif

#endif

#define UNODB_DETAIL_LIKELY(condition) __builtin_expect(condition, 1)
#define UNODB_DETAIL_UNLIKELY(condition) __builtin_expect(condition, 0)

#define UNODB_DETAIL_UNUSED [[gnu::unused]]
#define UNODB_DETAIL_FORCE_INLINE __attribute__((always_inline))
#define UNODB_DETAIL_NOINLINE __attribute__((noinline))
#define UNODB_DETAIL_UNREACHABLE() __builtin_unreachable()
#define UNODB_DETAIL_CONSTEXPR_NOT_MSVC constexpr

#else  // #ifndef UNODB_DETAIL_MSVC

#define UNODB_DETAIL_BUILTIN_ASSUME(condition) __assume(condition)

#define UNODB_DETAIL_LIKELY(condition) (!!(condition))
#define UNODB_DETAIL_UNLIKELY(condition) (!!(condition))

#define UNODB_DETAIL_UNUSED [[maybe_unused]]
#define UNODB_DETAIL_FORCE_INLINE __forceinline
#define UNODB_DETAIL_NOINLINE __declspec(noinline)
#define UNODB_DETAIL_UNREACHABLE() __assume(0)
#define UNODB_DETAIL_CONSTEXPR_NOT_MSVC inline
#define UNODB_DETAIL_LIFETIMEBOUND [[msvc::lifetimebound]]
#define UNODB_DETAIL_C_STRING_ARG(x)

#endif  // #ifndef UNODB_DETAIL_MSVC

/// \}

/// Declaration specifier for a function in a header file that should not be
/// inlined.
/// \hideinitializer
/// This is a pair of two seemingly conflicting intentions: "noinline" and
/// "inline". However, only the "noinline" has anything to do with inlining. The
/// "inline" has nothing to do with inlining and is required for functions
/// declared in headers.
#define UNODB_DETAIL_HEADER_NOINLINE UNODB_DETAIL_NOINLINE inline

/// \name Sanitizer macros
/// \{

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define UNODB_DETAIL_ADDRESS_SANITIZER
#endif
#if __has_feature(thread_sanitizer)
#define UNODB_DETAIL_THREAD_SANITIZER
#endif
#else
#ifdef __SANITIZE_ADDRESS__
#define UNODB_DETAIL_ADDRESS_SANITIZER
#endif
#ifdef __SANITIZE_THREAD__
#define UNODB_DETAIL_THREAD_SANITIZER
#endif
#endif

/// Suppress TSan for functions with benign races (e.g., OLC optimistic reads).
#ifdef UNODB_DETAIL_THREAD_SANITIZER
#define UNODB_DETAIL_DISABLE_TSAN \
  __attribute__((no_sanitize("thread"), noinline))
#else
#define UNODB_DETAIL_DISABLE_TSAN
#endif

/// \}

#define UNODB_DETAIL_DO_PRAGMA(x) _Pragma(#x)

/// \name Warning macros
/// \{

#ifndef UNODB_DETAIL_MSVC

#define UNODB_DETAIL_DISABLE_WARNING(x) \
  _Pragma("GCC diagnostic push")        \
      UNODB_DETAIL_DO_PRAGMA(GCC diagnostic ignored x)

#define UNODB_DETAIL_RESTORE_WARNINGS() _Pragma("GCC diagnostic pop")

#define UNODB_DETAIL_DISABLE_MSVC_WARNING(x)
#define UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

#else  // #ifndef UNODB_DETAIL_MSVC

#define UNODB_DETAIL_DISABLE_MSVC_WARNING(x) \
  _Pragma("warning(push)") UNODB_DETAIL_DO_PRAGMA(warning(disable : x))

#define UNODB_DETAIL_RESTORE_MSVC_WARNINGS() _Pragma("warning(pop)")

#endif  // #ifndef UNODB_DETAIL_MSVC

#ifdef __clang__
#define UNODB_DETAIL_DISABLE_CLANG_WARNING(x) UNODB_DETAIL_DISABLE_WARNING(x)
#define UNODB_DETAIL_RESTORE_CLANG_WARNINGS() UNODB_DETAIL_RESTORE_WARNINGS()
#else
#define UNODB_DETAIL_DISABLE_CLANG_WARNING(x)
#define UNODB_DETAIL_RESTORE_CLANG_WARNINGS()
#endif

#if defined(__clang__) && __clang_major__ >= 21
#define UNODB_DETAIL_DISABLE_CLANG_21_WARNING(x) UNODB_DETAIL_DISABLE_WARNING(x)
#define UNODB_DETAIL_RESTORE_CLANG_21_WARNINGS() UNODB_DETAIL_RESTORE_WARNINGS()
#else
#define UNODB_DETAIL_DISABLE_CLANG_21_WARNING(x)
#define UNODB_DETAIL_RESTORE_CLANG_21_WARNINGS()
#endif

#if defined(__GNUG__) && !defined(__clang__)
#define UNODB_DETAIL_DISABLE_GCC_WARNING(x) UNODB_DETAIL_DISABLE_WARNING(x)
#define UNODB_DETAIL_RESTORE_GCC_WARNINGS() UNODB_DETAIL_RESTORE_WARNINGS()
#if __GNUG__ == 10
#define UNODB_DETAIL_DISABLE_GCC_10_WARNING(x) UNODB_DETAIL_DISABLE_WARNING(x)
#define UNODB_DETAIL_RESTORE_GCC_10_WARNINGS() UNODB_DETAIL_RESTORE_WARNINGS()
#else  // __GNUG__ == 10
#define UNODB_DETAIL_DISABLE_GCC_10_WARNING(x)
#define UNODB_DETAIL_RESTORE_GCC_10_WARNINGS()
#endif  // __GNUG__ == 10
#if __GNUG__ >= 11
#define UNODB_DETAIL_DISABLE_GCC_11_WARNING(x) UNODB_DETAIL_DISABLE_WARNING(x)
#define UNODB_DETAIL_RESTORE_GCC_11_WARNINGS() UNODB_DETAIL_RESTORE_WARNINGS()
#else  // __GNUG__ >= 11
#define UNODB_DETAIL_DISABLE_GCC_11_WARNING(x)
#define UNODB_DETAIL_RESTORE_GCC_11_WARNINGS()
#endif  // __GNUG__ >= 11
#else   // defined(__GNUG__) && !defined(__clang__)
#define UNODB_DETAIL_DISABLE_GCC_WARNING(x)
#define UNODB_DETAIL_RESTORE_GCC_WARNINGS()
#define UNODB_DETAIL_DISABLE_GCC_10_WARNING(x)
#define UNODB_DETAIL_RESTORE_GCC_10_WARNINGS()
#define UNODB_DETAIL_DISABLE_GCC_11_WARNING(x)
#define UNODB_DETAIL_RESTORE_GCC_11_WARNINGS()
#endif  // defined(__GNUG__) && !defined(__clang__)

/// \}

/// \name Debug or release build macros
/// Definitions conditional on the build type
/// \{

#ifdef NDEBUG
#define UNODB_DETAIL_RELEASE_CONSTEXPR constexpr
#define UNODB_DETAIL_RELEASE_CONST const
#define UNODB_DETAIL_RELEASE_EXPLICIT explicit
#define UNODB_DETAIL_USED_IN_DEBUG UNODB_DETAIL_UNUSED
#else
#define UNODB_DETAIL_RELEASE_CONSTEXPR
#define UNODB_DETAIL_RELEASE_CONST
#define UNODB_DETAIL_RELEASE_EXPLICIT
#define UNODB_DETAIL_USED_IN_DEBUG
#endif

/// \}

/// \name Feature macros
/// Compile-time feature selection macros
/// \{

/// Spin lock wait loops use x86_64 PAUSE instruction or its closest equivalent
/// on other architectures.
#define UNODB_DETAIL_SPINLOCK_LOOP_PAUSE 1

/// Spin lock loop wait loops are empty, causing aggressive spinning.
#define UNODB_DETAIL_SPINLOCK_LOOP_EMPTY 2

/// \}

/// \}

#endif  // UNODB_DETAIL_GLOBAL_HPP
