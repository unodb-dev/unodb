// Copyright 2026 UnoDB contributors
#ifndef UNODB_DETAIL_PORTABILITY_EXECUTION_HPP
#define UNODB_DETAIL_PORTABILITY_EXECUTION_HPP

/// \file
/// Portability shim for C++17 execution policies (`<execution>`).
///
/// Apple libc++ and some older standard libraries lack execution policy
/// support.  This header detects availability and provides a minimal
/// sequential-only fallback when the real header is absent.

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

#if __has_include(<execution>)
#include <execution>  // IWYU pragma: export
#endif

// Detect usable execution policies via the library feature-test macro.
// __cpp_lib_execution >= 201603 means parallel_policy is available.
#if defined(__cpp_lib_execution) && __cpp_lib_execution >= 201603L
#define UNODB_DETAIL_HAS_EXECUTION_POLICIES 1
#else
#define UNODB_DETAIL_HAS_EXECUTION_POLICIES 0
#endif

#if !UNODB_DETAIL_HAS_EXECUTION_POLICIES

/// Minimal fallback: sequential execution tag only.
namespace std::execution {  // NOLINT(cert-dcl58-cpp)

struct sequenced_policy {};
inline constexpr sequenced_policy seq{};

struct parallel_policy {};
inline constexpr parallel_policy par{};

struct parallel_unsequenced_policy {};
inline constexpr parallel_unsequenced_policy par_unseq{};

}  // namespace std::execution

#endif  // !UNODB_DETAIL_HAS_EXECUTION_POLICIES

#endif  // UNODB_DETAIL_PORTABILITY_EXECUTION_HPP
