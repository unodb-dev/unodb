// Copyright 2022-2025 UnoDB contributors
#ifndef UNODB_DETAIL_PORTABILITY_BUILTINS_HPP
#define UNODB_DETAIL_PORTABILITY_BUILTINS_HPP

/// \file
/// Definitions to abstract differences between different compiler builtins and
/// intrinsics.
/// \ingroup internal

// Should be the first include
#include "global.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>

#ifdef UNODB_DETAIL_MSVC
#include <intrin.h>
#endif

namespace unodb::detail {

/// Reverse the order of bytes in \a x.
template <typename T>
[[nodiscard, gnu::const]] constexpr T bswap(T x) noexcept {
#ifdef UNODB_DETAIL_MSVC
  static_assert(sizeof(std::uint32_t) ==
                sizeof(unsigned long));               // NOLINT(runtime/int)
  static_assert(std::is_same_v<unsigned short, T> ||  // NOLINT(runtime/int)
                std::is_same_v<std::uint32_t, T> ||
                std::is_same_v<std::uint64_t, T>);
  if constexpr (std::is_same_v<unsigned short, T>) {  // NOLINT(runtime/int)
    return _byteswap_ushort(x);
  }
  if constexpr (std::is_same_v<std::uint32_t, T>) {
    return _byteswap_ulong(x);
  }
  if constexpr (std::is_same_v<std::uint64_t, T>) {
    return _byteswap_uint64(x);
  }
#else   // UNODB_DETAIL_MSVC
  static_assert(std::is_same_v<std::uint16_t, T> ||
                std::is_same_v<std::uint32_t, T> ||
                std::is_same_v<std::uint64_t, T>);
  if constexpr (std::is_same_v<std::uint16_t, T>) {
    return __builtin_bswap16(x);
  }
  if constexpr (std::is_same_v<std::uint32_t, T>) {
    return __builtin_bswap32(x);
  }
  if constexpr (std::is_same_v<std::uint64_t, T>) {
    return __builtin_bswap64(x);
  }
#endif  // UNODB_DETAIL_MSVC
}

/// Reinterpret object representation as a different type.
///
/// A replacement for `std::bit_cast` until it is provided by all supported
/// compilers.
///
/// \tparam To Destination type for the conversion
/// \tparam From Source type of the input value
/// \param input Object to convert
/// \return Object of type \a To with the same bit pattern as \a input.
///
/// \pre `sizeof(To) == sizeof(From)`
// TODO(laurynas) We can use std::bit_cast at GCC 11 clang 14 minimums.
template <typename To, typename From>
[[nodiscard, gnu::const]] constexpr To bit_cast(From input) noexcept {
  // See https://github.com/jfbastien/bit_cast for an implementation using
  // memcpy (MIT license).
  static_assert(sizeof(To) == sizeof(From));
  typename std::aligned_storage<sizeof(To), alignof(To)>::type tmp;
  std::memcpy(&tmp, &input, sizeof(To));
  return reinterpret_cast<const To&>(tmp);
}

}  // namespace unodb::detail

/// Workaround for #700: union type-pun causes stale reads under clang -O3.
/// Forces the compiler to re-read the object through a volatile pointer.
/// FIXME(@laurynas-biveinis): remove when #700 is fixed.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define UNODB_DETAIL_RELOAD(obj)       \
  (*const_cast<                        \
      std::remove_cvref_t<decltype(obj)>*>( \
      static_cast<volatile const      \
          std::remove_cvref_t<decltype(obj)>*>(&(obj))))

#endif
