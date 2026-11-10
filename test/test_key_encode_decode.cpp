// Copyright 2024-2025 UnoDB contributors

// Should be the first include
#include "global.hpp"

// IWYU pragma: no_include <__cstddef/byte.h>
// IWYU pragma: no_include <__math/traits.h>
// IWYU pragma: no_include <__ostream/basic_ostream.h>
// IWYU pragma: no_include <__vector/vector.h>
// IWYU pragma: no_include <_string.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>  // IWYU pragma: keep
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>  // IWYU pragma: keep

#ifndef NDEBUG
#include <iostream>  // IWYU pragma: keep
#endif

#include <gtest/gtest.h>

#include "art_common.hpp"
#include "art_internal.hpp"
#include "gtest_utils.hpp"
#include "portability_builtins.hpp"

#ifndef NDEBUG
#include "test_utils.hpp"
#endif

namespace {

/// Test suite for key encoding, decoding, and the lexicographic
/// ordering obtained from the encoded keys.
template <class Db>
class ARTKeyEncodeDecodeTest : public ::testing::Test {
 public:
  using Test::Test;
};

using unodb::detail::compare;

constexpr auto INITIAL_CAPACITY = unodb::detail::INITIAL_BUFFER_CAPACITY;

/// Test helper verifies that [ekey1] < [ekey2].
///
/// @param ekey1 An external key of some type.
///
/// @param ekey2 Another external key of the same type.
template <typename T>
void do_encode_decode_lt_test(const T ekey1, const T ekey2) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    // Note: floating point +0 and -0 compare as equal, so we do not
    // compare the keys for non-quality if one of the keys is zero.
    if (std::fpclassify(ekey1) != FP_ZERO) {
      UNODB_EXPECT_NE(ekey1, ekey2);  // not the same ekey.
    }
  } else {
    UNODB_EXPECT_NE(ekey1, ekey2);  // not the same ekey.
  }
  unodb::key_encoder enc1{};
  unodb::key_encoder enc2{};  // separate decoder (backed by different span).
  const auto ikey1 = enc1.encode(ekey1).get_key_view();  // into encoder buf!
  const auto ikey2 = enc2.encode(ekey2).get_key_view();  // into encoder buf!
  UNODB_EXPECT_EQ(compare(ikey1, ikey1), 0);             // compare w/ self
  UNODB_EXPECT_EQ(compare(ikey2, ikey2), 0);             // compare w/ self
  UNODB_EXPECT_NE(compare(ikey1, ikey2), 0);             // not the same ikey.
  // Check the core assertion for this test helper. The internal keys
  // (after encoding) obey the asserted ordering over the external
  // keys (before encoding).
  if (!(compare(ikey1, ikey2) < 0)) {
    // LCOV_EXCL_START
    std::stringstream ss1;
    std::stringstream ss2;
    unodb::detail::dump_key(ss1, ikey1);
    unodb::detail::dump_key(ss2, ikey2);
    FAIL() << "ikey1 < ikey2"
           << ": ekey1(" << ekey1 << ")[" << ss1.str() << "]"
           << ", ekey2(" << ekey2 << ")[" << ss2.str() << "]";
    // LCOV_EXCL_START
  }
  // Verify key2 > key1
  // NOLINTNEXTLINE(readability/check)
  UNODB_EXPECT_GT(compare(ikey2, ikey1), 0);
  // Verify that we can round trip both values.
  unodb::key_decoder dec1{ikey1};
  unodb::key_decoder dec2{ikey2};
  T akey1;
  T akey2;
  dec1.decode(akey1);
  dec2.decode(akey2);
  UNODB_EXPECT_EQ(ekey1, akey1);
  UNODB_EXPECT_EQ(ekey2, akey2);
}

// basic memory management - initial buffer case.
UNODB_TEST(ARTKeyEncodeDecodeTest, C00001) {
  unodb::key_encoder enc{};
  UNODB_EXPECT_EQ(enc.capacity(), INITIAL_CAPACITY);
  UNODB_EXPECT_EQ(enc.size_bytes(), 0);
  // ensure some space is available w/o change in encoder.
  enc.ensure_available(INITIAL_CAPACITY - 1);  // edge case
  UNODB_EXPECT_EQ(enc.capacity(), INITIAL_CAPACITY);
  UNODB_EXPECT_EQ(enc.size_bytes(), 0);
  // ensure some space is available w/o change in encoder.
  enc.ensure_available(INITIAL_CAPACITY);  // edge case
  UNODB_EXPECT_EQ(enc.capacity(), INITIAL_CAPACITY);
  UNODB_EXPECT_EQ(enc.size_bytes(), 0);
  // reset -- nothing changes.
  enc.reset();
  UNODB_EXPECT_EQ(enc.capacity(), INITIAL_CAPACITY);
  UNODB_EXPECT_EQ(enc.size_bytes(), 0);
  // key_view is empty
  const auto kv = enc.get_key_view();
  UNODB_EXPECT_EQ(kv.size_bytes(), 0);
}

// basic memory management -- buffer extension case.
UNODB_TEST(ARTKeyEncodeDecodeTest, C00002) {
  unodb::key_encoder enc{};
  UNODB_EXPECT_EQ(enc.capacity(), INITIAL_CAPACITY);
  UNODB_EXPECT_EQ(enc.size_bytes(), 0);
  // ensure some space is available w/o change in encoder.
  enc.ensure_available(INITIAL_CAPACITY + 1);  // edge case.
  UNODB_EXPECT_EQ(enc.capacity(),
                  INITIAL_CAPACITY * 2);  // assumes power of two
  UNODB_EXPECT_EQ(enc.size_bytes(), 0);
  UNODB_EXPECT_EQ(enc.get_key_view().size_bytes(), 0);  // key_view is empty
  // reset.
  enc.reset();
  UNODB_EXPECT_EQ(enc.capacity(), INITIAL_CAPACITY * 2);  // unchanged
  UNODB_EXPECT_EQ(enc.size_bytes(), 0);                   // reset
  UNODB_EXPECT_EQ(enc.get_key_view().size_bytes(), 0);    // key_view is empty
}

// Encode/decode round trip test.
template <typename T>
void do_encode_decode_test(const T ekey) {
  unodb::key_encoder enc{};
  enc.encode(ekey);
  const unodb::key_view kv = enc.get_key_view();   // encode
  UNODB_EXPECT_EQ(kv.size_bytes(), sizeof(ekey));  // check size
  // decode check
  unodb::key_decoder dec{kv};
  T akey;
  dec.decode(akey);
  UNODB_EXPECT_EQ(akey, ekey);
}

UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)

// Encode/decode round trip test which also verifies the encoded byte sequence.
template <typename T>
void do_encode_decode_test(const T ekey,
                           const std::array<const std::byte, sizeof(T)> ikey) {
  unodb::key_encoder enc{};
  enc.encode(ekey);
  const unodb::key_view kv = enc.get_key_view();   // encode
  UNODB_EXPECT_EQ(kv.size_bytes(), sizeof(ekey));  // check size
  // check order.
  size_t i = 0;
  for (const auto byte : ikey) {
    UNODB_EXPECT_EQ(byte, kv[i++]);
  }
  // decode check
  unodb::key_decoder dec{kv};
  T akey;
  dec.decode(akey);
  UNODB_EXPECT_EQ(akey, ekey);
}

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

UNODB_TEST(ARTKeyEncodeDecodeTest, UInt8C00010) {
  using T = std::uint8_t;
  constexpr auto one = static_cast<T>(1);
  // Check the encoder byte order.
  constexpr std::array<const std::byte, sizeof(T)> ikey{
      static_cast<std::byte>(0x01)};
  do_encode_decode_test(static_cast<T>(0x01ULL), ikey);
  // round-trip tests.
  do_encode_decode_test(static_cast<T>(0));
  do_encode_decode_test(static_cast<T>(0 + 1));
  do_encode_decode_test(static_cast<T>(0 - 1));
  do_encode_decode_test(std::numeric_limits<T>::min());
  do_encode_decode_test(std::numeric_limits<T>::max());
  do_encode_decode_test(std::numeric_limits<T>::min() + one);
  do_encode_decode_test(std::numeric_limits<T>::max() - one);
  // check lexicographic ordering for std::uint8_t pairs.
  do_encode_decode_lt_test(static_cast<T>(0x01ULL), static_cast<T>(0x09ULL));
  do_encode_decode_lt_test(static_cast<T>(0), static_cast<T>(1));
  do_encode_decode_lt_test(static_cast<T>(0x7FULL), static_cast<T>(0x80ULL));
  do_encode_decode_lt_test(static_cast<T>(0xFEULL), static_cast<T>(~0ULL));
}

UNODB_TEST(ARTKeyEncodeDecodeTest, Int8C00010) {
  using T = std::int8_t;
  constexpr auto one = static_cast<T>(1);
  // Check the encoder byte order.
  constexpr std::array<const std::byte, sizeof(T)> ikey{
      static_cast<std::byte>(0x81)};
  do_encode_decode_test(static_cast<T>(0x01LL), ikey);
  // round-trip tests.
  do_encode_decode_test(static_cast<T>(0));
  do_encode_decode_test(static_cast<T>(0 + 1));
  do_encode_decode_test(static_cast<T>(0 - 1));
  do_encode_decode_test(std::numeric_limits<T>::min());
  do_encode_decode_test(std::numeric_limits<T>::max());
  do_encode_decode_test(std::numeric_limits<T>::min() + one);
  do_encode_decode_test(std::numeric_limits<T>::max() - one);
  // check lexicographic ordering for std::uint8_t pairs.
  do_encode_decode_lt_test(static_cast<T>(0), static_cast<T>(1));
  do_encode_decode_lt_test(static_cast<T>(5), static_cast<T>(7));
  do_encode_decode_lt_test(std::numeric_limits<T>::min(),
                           static_cast<T>(std::numeric_limits<T>::min() + one));
  do_encode_decode_lt_test(static_cast<T>(std::numeric_limits<T>::max() - one),
                           std::numeric_limits<T>::max());
}

UNODB_TEST(ARTKeyEncodeDecodeTest, UInt16C00010) {
  using T = std::uint16_t;
  constexpr auto one = static_cast<T>(1);
  // Check the encoder byte order.
  constexpr std::array<const std::byte, sizeof(T)> ikey{
      static_cast<std::byte>(0x01), static_cast<std::byte>(0x02)};
  do_encode_decode_test(static_cast<T>(0x0102ULL), ikey);
  // round-trip tests.
  do_encode_decode_test(static_cast<T>(0));
  do_encode_decode_test(static_cast<T>(0 + 1));
  do_encode_decode_test(static_cast<T>(0 - 1));
  do_encode_decode_test(std::numeric_limits<T>::min());
  do_encode_decode_test(std::numeric_limits<T>::max());
  do_encode_decode_test(std::numeric_limits<T>::min() + one);
  do_encode_decode_test(std::numeric_limits<T>::max() - one);
  // check lexicographic ordering for std::uint16_t pairs.
  do_encode_decode_lt_test(static_cast<T>(0x0102ULL),
                           static_cast<T>(0x090AULL));
  do_encode_decode_lt_test(static_cast<T>(0), static_cast<T>(1));
  do_encode_decode_lt_test(static_cast<T>(0x7FFFULL),
                           static_cast<T>(0x8000ULL));
  do_encode_decode_lt_test(static_cast<T>(0xFFFEULL), static_cast<T>(~0ULL));
}

UNODB_TEST(ARTKeyEncodeDecodeTest, Int16C00010) {
  using T = std::int16_t;
  constexpr auto one = static_cast<T>(1);
  // Check the encoder byte order.
  constexpr std::array<const std::byte, sizeof(T)> ikey{
      static_cast<std::byte>(0x81), static_cast<std::byte>(0x02)};
  do_encode_decode_test(static_cast<T>(0x0102LL), ikey);
  // round-trip tests.
  do_encode_decode_test(static_cast<T>(0));
  do_encode_decode_test(static_cast<T>(0 + 1));
  do_encode_decode_test(static_cast<T>(0 - 1));
  do_encode_decode_test(std::numeric_limits<T>::min());
  do_encode_decode_test(std::numeric_limits<T>::max());
  do_encode_decode_test(std::numeric_limits<T>::min() + one);
  do_encode_decode_test(std::numeric_limits<T>::max() - one);
  // check lexicographic ordering for std::uint16_t pairs.
  do_encode_decode_lt_test(static_cast<T>(0), static_cast<T>(1));
  do_encode_decode_lt_test(static_cast<T>(5), static_cast<T>(7));
  do_encode_decode_lt_test(std::numeric_limits<T>::min(),
                           static_cast<T>(std::numeric_limits<T>::min() + one));
  do_encode_decode_lt_test(static_cast<T>(std::numeric_limits<T>::max() - one),
                           std::numeric_limits<T>::max());
}

UNODB_TEST(ARTKeyEncodeDecodeTest, Uint32C00010) {
  using T = std::uint32_t;
  constexpr auto one = static_cast<T>(1);
  // Check the encoder byte order.
  constexpr std::array<const std::byte, sizeof(T)> ikey{
      static_cast<std::byte>(0x01), static_cast<std::byte>(0x02),
      static_cast<std::byte>(0x03), static_cast<std::byte>(0x04)};
  do_encode_decode_test(static_cast<T>(0x01020304), ikey);
  // round-trip tests.
  do_encode_decode_test(static_cast<T>(0));
  do_encode_decode_test(static_cast<T>(0 + 1));
  do_encode_decode_test(static_cast<T>(0 - 1));
  do_encode_decode_test(std::numeric_limits<T>::min());
  do_encode_decode_test(std::numeric_limits<T>::max());
  do_encode_decode_test(std::numeric_limits<T>::min() + one);
  do_encode_decode_test(std::numeric_limits<T>::max() - one);
  // check lexicographic ordering for std::uint32_t pairs.
  do_encode_decode_lt_test(static_cast<T>(0x01020304ULL),
                           static_cast<T>(0x090A0B0CULL));
  do_encode_decode_lt_test(static_cast<T>(0), static_cast<T>(1));
  do_encode_decode_lt_test(static_cast<T>(0x7FFFFFFFULL),
                           static_cast<T>(0x80000000ULL));
  do_encode_decode_lt_test(static_cast<T>(0xFFFFFFFEULL),
                           static_cast<T>(~0ULL));
}

UNODB_TEST(ARTKeyEncodeDecodeTest, Int32C00010) {
  using T = std::int32_t;
  constexpr auto one = 1;  // useless cast:: static_cast<T>(1);
  // Check the encoder byte order.
  constexpr std::array<const std::byte, sizeof(T)> ikey{
      static_cast<std::byte>(0x81), static_cast<std::byte>(0x02),
      static_cast<std::byte>(0x03), static_cast<std::byte>(0x04)};
  do_encode_decode_test(static_cast<T>(0x01020304LL), ikey);
  // round-trip tests.
  //
  // Note: 0, 1, ~0, etc. are already std::int32_t.  If that is not
  // true universally, then we need conditional compilation here to
  // avoid "warning useless cast" errors in the compiler.
  do_encode_decode_test(0);
  do_encode_decode_test(0 + 1);
  do_encode_decode_test(0 - 1);
  do_encode_decode_test(std::numeric_limits<T>::min());
  do_encode_decode_test(std::numeric_limits<T>::min() + one);
  do_encode_decode_test(std::numeric_limits<T>::max());
  do_encode_decode_test(std::numeric_limits<T>::max() - one);
  // check lexicographic ordering for std::uint32_t pairs.
  do_encode_decode_lt_test(0, 1);
  do_encode_decode_lt_test(5, 7);
  do_encode_decode_lt_test(std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::min() + one);
  do_encode_decode_lt_test(std::numeric_limits<T>::max() - one,
                           std::numeric_limits<T>::max());
}

UNODB_TEST(ARTKeyEncodeDecodeTest, UInt64C00010) {
  using T = std::uint64_t;
  constexpr auto one = static_cast<T>(1);
  // Check the encoder byte order.
  constexpr std::array<const std::byte, sizeof(T)> ikey{
      static_cast<std::byte>(0x01), static_cast<std::byte>(0x02),
      static_cast<std::byte>(0x03), static_cast<std::byte>(0x04),
      static_cast<std::byte>(0x05), static_cast<std::byte>(0x06),
      static_cast<std::byte>(0x07), static_cast<std::byte>(0x08)};
  do_encode_decode_test(static_cast<T>(0x0102030405060708), ikey);
  // round-trip tests.
  do_encode_decode_test(static_cast<T>(0));
  do_encode_decode_test(static_cast<T>(0 + 1));
  do_encode_decode_test(static_cast<T>(0 - 1));
  do_encode_decode_test(std::numeric_limits<T>::min());
  do_encode_decode_test(std::numeric_limits<T>::max());
  do_encode_decode_test(std::numeric_limits<T>::min() + one);
  do_encode_decode_test(std::numeric_limits<T>::max() - one);
  // check lexicographic ordering for std::uint64_t pairs.
  do_encode_decode_lt_test<T>(0x0102030405060708ULL, 0x090A0B0C0D0F1011ULL);
  do_encode_decode_lt_test(static_cast<T>(0), static_cast<T>(1));
  do_encode_decode_lt_test<T>(0x7FFFFFFFFFFFFFFFULL, 0x8000000000000000ULL);
  do_encode_decode_lt_test<T>(0xFFFFFFFFFFFFFFFEULL, ~0ULL);
}

UNODB_TEST(ARTKeyEncodeDecodeTest, Int64C00010) {
  using T = std::int64_t;
  constexpr auto one = static_cast<T>(1);
  // Check the encoder byte order.
  constexpr std::array<const std::byte, sizeof(T)> ikey{
      static_cast<std::byte>(0x81), static_cast<std::byte>(0x02),
      static_cast<std::byte>(0x03), static_cast<std::byte>(0x04),
      static_cast<std::byte>(0x05), static_cast<std::byte>(0x06),
      static_cast<std::byte>(0x07), static_cast<std::byte>(0x08)};
  do_encode_decode_test<T>(0x0102030405060708LL, ikey);
  // round-trip tests.
  do_encode_decode_test(static_cast<T>(0));
  do_encode_decode_test(static_cast<T>(0 + 1));
  do_encode_decode_test(static_cast<T>(0 - 1));
  do_encode_decode_test(std::numeric_limits<T>::min());
  do_encode_decode_test(std::numeric_limits<T>::max());
  do_encode_decode_test(std::numeric_limits<T>::min() + one);
  do_encode_decode_test(std::numeric_limits<T>::max() - one);
  // check lexicographic ordering for std::uint64_t pairs.
  do_encode_decode_lt_test(static_cast<T>(0), static_cast<T>(1));
  do_encode_decode_lt_test(static_cast<T>(5), static_cast<T>(7));
  do_encode_decode_lt_test(std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::min() + one);
  do_encode_decode_lt_test(std::numeric_limits<T>::max() - one,
                           std::numeric_limits<T>::max());
}

//
// float & double tests.
//

/// Can be used for anything (handles NaN as a special case).
void do_encode_decode_float_test(const float expected) {
  using U = std::uint32_t;
  using F = float;
  // encode
  unodb::key_encoder enc;
  enc.reset().encode(expected);
  // Check decode as float (round trip).
  F actual;
  {
    unodb::key_decoder dec{enc.get_key_view()};
    dec.decode(actual);
  }
  if (std::isnan(expected)) {
    // Verify canonical NaN.
    UNODB_EXPECT_TRUE(std::isnan(actual));
    U u;
    unodb::key_decoder dec{enc.get_key_view()};
    dec.decode(u);
  } else {
    UNODB_EXPECT_EQ(actual, expected);
  }
}

/// Test encode/decode of various floating point values.
UNODB_TEST(ARTKeyEncodeDecodeTest, FloatC0001) {
  using F = float;
  constexpr auto pzero = 0.F;
  constexpr auto nzero = -0.F;
  UNODB_EXPECT_FALSE(std::signbit(pzero));
  UNODB_EXPECT_TRUE(std::signbit(nzero));
  do_encode_decode_float_test(pzero);
  do_encode_decode_float_test(nzero);
  do_encode_decode_float_test(10.001F);
  do_encode_decode_float_test(-10.001F);
  do_encode_decode_float_test(std::numeric_limits<F>::min());
  do_encode_decode_float_test(std::numeric_limits<F>::lowest());
  do_encode_decode_float_test(std::numeric_limits<F>::max());
  do_encode_decode_float_test(std::numeric_limits<F>::epsilon());
  do_encode_decode_float_test(std::numeric_limits<F>::denorm_min());
}

/// inf
UNODB_TEST(ARTKeyEncodeDecodeTest, FloatC0002Infinity) {
  using F = float;
  using U = std::uint32_t;
  constexpr auto inf = std::numeric_limits<F>::infinity();
  UNODB_EXPECT_EQ(unodb::detail::bit_cast<const U>(inf), 0x7f800000U);
  do_encode_decode_float_test(inf);
}

/// -inf
UNODB_TEST(ARTKeyEncodeDecodeTest, FloatC0003NegInfinity) {
  using F = float;
  using U = std::uint32_t;
  constexpr auto ninf = -std::numeric_limits<F>::infinity();
  UNODB_EXPECT_EQ(sizeof(ninf), sizeof(float));
  UNODB_EXPECT_TRUE(std::numeric_limits<float>::is_iec559)
      << "IEEE 754 required";
  UNODB_EXPECT_LT(ninf, std::numeric_limits<float>::lowest());
  UNODB_EXPECT_TRUE(std::isinf(ninf));
  UNODB_EXPECT_FALSE(std::isnan(ninf));
  UNODB_EXPECT_EQ(unodb::detail::bit_cast<const U>(ninf), 0xff800000U);
  do_encode_decode_float_test(ninf);
}

/// quiet_NaN
UNODB_TEST(ARTKeyEncodeDecodeTest, FloatC0004QuietNaN) {
  using F = float;
  constexpr F f{std::numeric_limits<F>::quiet_NaN()};
  UNODB_EXPECT_TRUE(std::isnan(f));
  do_encode_decode_float_test(f);
}

/// signaling_NaN
UNODB_TEST(ARTKeyEncodeDecodeTest, FloatC0005SignalingNan) {
  using F = float;
  constexpr F f{std::numeric_limits<F>::signaling_NaN()};
  UNODB_EXPECT_TRUE(std::isnan(f));
  do_encode_decode_float_test(f);
}

/// NaN can be formed for any floating point value using std::nanf().
UNODB_TEST(ARTKeyEncodeDecodeTest, FloatC0006NumericNaN) {
  do_encode_decode_float_test(std::nanf("-1"));
  do_encode_decode_float_test(std::nanf("1"));
  do_encode_decode_float_test(std::nanf("100.1"));
  do_encode_decode_float_test(std::nanf("-100.1"));
}

/// Verify the ordering over various floating point pairs.
UNODB_TEST(ARTKeyEncodeDecodeTest, FloatC0007Order) {
  using F = float;
  constexpr auto pzero = 0.F;
  constexpr auto nzero = -0.F;
  UNODB_EXPECT_FALSE(std::signbit(pzero));
  UNODB_EXPECT_TRUE(std::signbit(nzero));
  constexpr auto minf = std::numeric_limits<F>::min();
  constexpr auto maxf = std::numeric_limits<F>::max();
  constexpr auto inf = std::numeric_limits<F>::infinity();
  constexpr auto ninf = -std::numeric_limits<F>::infinity();
  constexpr auto lowest = std::numeric_limits<F>::lowest();
  do_encode_decode_lt_test(-10.01F, -1.01F);
  do_encode_decode_lt_test(-1.F, pzero);
  do_encode_decode_lt_test(nzero, pzero);
  do_encode_decode_lt_test(pzero, 1.0F);
  do_encode_decode_lt_test(1.01F, 10.01F);
  do_encode_decode_lt_test(ninf, lowest);
  do_encode_decode_lt_test(0.F, minf);
  do_encode_decode_lt_test(maxf, inf);
}

/// Can be used for anything (handles NaN as a special case).
void do_encode_decode_double_test(const double expected) {
  using U = std::uint64_t;
  using F = double;
  // encode
  unodb::key_encoder enc;
  enc.reset().encode(expected);
  // Check decode as double (round trip).
  F actual;
  {
    unodb::key_decoder dec{enc.get_key_view()};
    dec.decode(actual);
  }
  if (std::isnan(expected)) {
    // Verify canonical NaN.
    UNODB_EXPECT_TRUE(std::isnan(actual));
    U u;
    unodb::key_decoder dec{enc.get_key_view()};
    dec.decode(u);
  } else {
    UNODB_EXPECT_EQ(actual, expected);
  }
}

/// Test encode/decode of various double precisions floating point
/// values.
UNODB_TEST(ARTKeyEncodeDecodeTest, DoubleC0001) {
  using F = double;
  constexpr auto pzero = 0.F;
  constexpr auto nzero = -0.F;
  UNODB_EXPECT_FALSE(std::signbit(pzero));
  UNODB_EXPECT_TRUE(std::signbit(nzero));
  do_encode_decode_float_test(pzero);
  do_encode_decode_float_test(nzero);
  do_encode_decode_double_test(10.001);
  do_encode_decode_double_test(-10.001);
  do_encode_decode_double_test(std::numeric_limits<F>::min());
  do_encode_decode_double_test(std::numeric_limits<F>::lowest());
  do_encode_decode_double_test(std::numeric_limits<F>::max());
  do_encode_decode_double_test(std::numeric_limits<F>::epsilon());
  do_encode_decode_double_test(std::numeric_limits<F>::denorm_min());
}

/// inf
UNODB_TEST(ARTKeyEncodeDecodeTest, DoubleC0002Infinity) {
  using F = double;
  using U = std::uint64_t;
  constexpr auto inf = std::numeric_limits<F>::infinity();
  UNODB_EXPECT_EQ(unodb::detail::bit_cast<const U>(inf), 0x7ff0000000000000ULL);
  do_encode_decode_double_test(inf);
}

/// -inf
UNODB_TEST(ARTKeyEncodeDecodeTest, DoubleC0003NegInfinity) {
  using F = double;
  using U = std::uint64_t;
  constexpr auto ninf = -std::numeric_limits<F>::infinity();
  UNODB_EXPECT_EQ(sizeof(ninf), sizeof(double));
  UNODB_EXPECT_TRUE(std::numeric_limits<double>::is_iec559)
      << "IEEE 754 required";
  UNODB_EXPECT_LT(ninf, std::numeric_limits<double>::lowest());
  UNODB_EXPECT_TRUE(std::isinf(ninf));
  UNODB_EXPECT_FALSE(std::isnan(ninf));
  UNODB_EXPECT_EQ(unodb::detail::bit_cast<const U>(ninf),
                  0xfff0000000000000ULL);
  do_encode_decode_double_test(ninf);
}

/// quiet_NaN
UNODB_TEST(ARTKeyEncodeDecodeTest, DoubleC0004QuietNaN) {
  using F = double;
  constexpr F f{std::numeric_limits<F>::quiet_NaN()};
  UNODB_EXPECT_TRUE(std::isnan(f));
  do_encode_decode_double_test(f);
}

/// signaling_NaN
UNODB_TEST(ARTKeyEncodeDecodeTest, DoubleC0005SignalingNan) {
  using F = double;
  constexpr F f{std::numeric_limits<F>::signaling_NaN()};
  UNODB_EXPECT_TRUE(std::isnan(f));
  do_encode_decode_double_test(f);
}

/// NaN can be formed for any double precision floating point value
/// using std::nanf().
UNODB_TEST(ARTKeyEncodeDecodeTest, DoubleC0006NumericNaN) {
  do_encode_decode_double_test(std::nan("-1"));
  do_encode_decode_double_test(std::nan("1"));
  do_encode_decode_double_test(std::nan("100.1"));
  do_encode_decode_double_test(std::nan("-100.1"));
}

/// Verify the ordering over various double precision floating point
/// pairs.
UNODB_TEST(ARTKeyEncodeDecodeTest, DoubleC0007Order) {
  using F = double;
  constexpr auto pzero = 0.;
  constexpr auto nzero = -0.;
  UNODB_EXPECT_FALSE(std::signbit(pzero));
  UNODB_EXPECT_TRUE(std::signbit(nzero));
  constexpr auto minf = std::numeric_limits<F>::min();
  constexpr auto maxf = std::numeric_limits<F>::max();
  constexpr auto inf = std::numeric_limits<F>::infinity();
  constexpr auto ninf = -std::numeric_limits<F>::infinity();
  constexpr auto lowest = std::numeric_limits<F>::lowest();
  do_encode_decode_lt_test(-10.01, -1.01);
  do_encode_decode_lt_test(-1., pzero);
  do_encode_decode_lt_test(nzero, pzero);
  do_encode_decode_lt_test(pzero, 1.0);
  do_encode_decode_lt_test(1.01, 10.01);
  do_encode_decode_lt_test(ninf, lowest);
  do_encode_decode_lt_test(0., minf);
  do_encode_decode_lt_test(maxf, inf);
}

//
// Append span<const::byte> (aka unodb::key_view).
//

void do_encode_bytes_test(std::span<const std::byte> a) {
  unodb::key_encoder enc;
  const auto sz = a.size();
  enc.append_bytes(a);
  const auto cmp = std::memcmp(enc.get_key_view().data(), a.data(), sz);
  UNODB_EXPECT_EQ(0, cmp);
  UNODB_EXPECT_EQ(sz, enc.size_bytes());
}

/// Unit test look at the simple case of appending a sequence of bytes
/// to the key_encoder.
UNODB_TEST(ARTKeyEncodeDecodeTest, AppendSpanConstByteC0001) {
  constexpr auto test_data_0 = std::array<const std::byte, 3>{
      std::byte{0x02}, std::byte{0x05}, std::byte{0x05}};
  constexpr auto test_data_1 = std::array<const std::byte, 3>{
      std::byte{0x03}, std::byte{0x00}, std::byte{0x05}};
  constexpr auto test_data_2 = std::array<const std::byte, 3>{
      std::byte{0x03}, std::byte{0x00}, std::byte{0x10}};
  constexpr auto test_data_3 = std::array<const std::byte, 3>{
      std::byte{0x03}, std::byte{0x05}, std::byte{0x05}};
  constexpr auto test_data_4 = std::array<const std::byte, 3>{
      std::byte{0x03}, std::byte{0x05}, std::byte{0x10}};
  constexpr auto test_data_5 = std::array<const std::byte, 3>{
      std::byte{0x03}, std::byte{0x10}, std::byte{0x05}};
  constexpr auto test_data_6 = std::array<const std::byte, 3>{
      std::byte{0x04}, std::byte{0x05}, std::byte{0x10}};
  constexpr auto test_data_7 = std::array<const std::byte, 3>{
      std::byte{0x04}, std::byte{0x10}, std::byte{0x05}};

  do_encode_bytes_test(std::span<const std::byte>(test_data_0));
  do_encode_bytes_test(std::span<const std::byte>(test_data_1));
  do_encode_bytes_test(std::span<const std::byte>(test_data_2));
  do_encode_bytes_test(std::span<const std::byte>(test_data_3));
  do_encode_bytes_test(std::span<const std::byte>(test_data_4));
  do_encode_bytes_test(std::span<const std::byte>(test_data_5));
  do_encode_bytes_test(std::span<const std::byte>(test_data_6));
  do_encode_bytes_test(std::span<const std::byte>(test_data_7));
}

/// key_buffer::assign_from() copies contents of each length around its
/// 32-byte chunks and the inline capacity, from and into buffers that are
/// inline or have grown past it.
UNODB_TEST(KeyBufferTest, AssignFromCopiesContents) {
  unodb::detail::key_buffer src;
  unodb::detail::key_buffer dst;
  // Not a std::vector: an empty one's data() may be null, and push() passes
  // it to memcpy, which must not get a null pointer even to copy nothing.
  std::array<std::byte, 300> storage{};
  for (const std::size_t n :
       {0U, 1U, 31U, 32U, 33U, 255U, 256U, 257U, 300U, 40U}) {
    const std::span<std::byte> bytes{storage.data(), n};
    for (std::size_t i = 0; i < n; ++i)
      bytes[i] = static_cast<std::byte>(i + n);
    src.reset();
    src.push(unodb::key_view{bytes});
    dst.assign_from(src);
    UNODB_EXPECT_TRUE(std::ranges::equal(dst.get_key_view(), bytes)) << n;
  }
}

#ifndef NDEBUG

/// key_buffer::assign_from() leaves the destination's contents as they were
/// if growing it to hold contents over the inline capacity fails.
UNODB_TEST(KeyBufferTest, AssignFromKeepsContentsOnBadAlloc) {
  const std::vector<std::byte> kept(40, std::byte{0x2A});
  const std::vector<std::byte> grown(300, std::byte{0x55});
  unodb::detail::key_buffer src;
  src.push(unodb::key_view{grown});
  unodb::detail::key_buffer dst;
  dst.push(unodb::key_view{kept});
  UNODB_ASSERT_TRUE(
      unodb::test::throws_bad_alloc(1, [&] { dst.assign_from(src); }));
  UNODB_ASSERT_TRUE(std::ranges::equal(dst.get_key_view(), kept));
}

#endif  // !NDEBUG

//
// Encoding of text fields (optionally truncated to maxlen and padded
// out to maxlen via run length encoding).
//

UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)

// Helper class to hold copies of key_views from a key_encoder.  We
// need to make copies because the key_view is backed by the data in
// the encoder. So we copy the data out into a new allocation and
// return a key_view backed by that allocation. The set of those
// allocations is held by this factory object and they go out of scope
// together.
class key_factory {
 public:
  /// Used to retain arrays backing unodb::key_views.
  std::vector<std::vector<std::byte>> key_views;

  /// Copy the data from the encoder into a new entry in
  /// test::key_factor::key_views.
  unodb::key_view make_key_view(const unodb::key_encoder& enc) {
    const auto kv{enc.get_key_view()};
    const auto sz{kv.size()};
    key_views.emplace_back(sz);
    auto& a = key_views.back();        // a *reference* to data emplaced_back.
    std::ranges::copy(kv, a.begin());  // copy data to inner vec
    return {a.data(), sz};             // view of inner vec's data.
  }
};

void do_simple_pad_test(unodb::key_encoder& enc, std::string_view sv) {
  using st = unodb::key_encoder::size_type;
  const auto len = sv.size();                         // text length.
  const auto sz = (len > unodb::key_encoder::maxlen)  // truncated len.
                      ? unodb::key_encoder::maxlen
                      : len;
  const auto kv = enc.reset().encode_text(sv).get_key_view();
  // Check expected resulting key length.
  UNODB_EXPECT_EQ(kv.size(), sz + sizeof(unodb::key_encoder::pad) + sizeof(st))
      << "text(" << sz << ")[" << (sz < 100 ? sv : "...") << "]";
  // Verify that the first N bytes are the same as the given text.
  UNODB_EXPECT_EQ(std::memcmp(sv.data(), kv.data(), sz), 0)
      << "text(" << sz << ")[" << (sz < 100 ? sv : "...") << "]";
  // Check for the pad byte.
  UNODB_EXPECT_EQ(kv[sz], unodb::key_encoder::pad)
      << "text(" << sz << ")[" << (sz < 100 ? sv : "...") << "]";
  // Check the pad length.
  const st padlen{static_cast<st>(unodb::key_encoder::maxlen - sz)};
  st tmp;
  std::memcpy(&tmp, &kv[sz + 1], sizeof(st));  // copy out pad length.
  const st tmp2 = unodb::detail::bswap(tmp);   // decode.
  UNODB_EXPECT_EQ(tmp2, padlen)
      << "text(" << sz << ")[" << (sz < 100 ? sv : "...") << "]";
}

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// Helper generates a large string and feeds it into
/// do_simple_pad_test().
void do_pad_test_large_string(unodb::key_encoder& enc, size_t nbytes,
                              bool expect_truncation = false) {
  std::string buf(nbytes, 'a');
  buf.push_back('\0');
  do_simple_pad_test(enc, std::string_view(buf.data(), nbytes));
  if (expect_truncation) {
    const auto kv = enc.get_key_view();
    constexpr auto max_key_size = unodb::key_encoder::maxlen +
                                  sizeof(unodb::key_encoder::pad) +
                                  sizeof(unodb::key_encoder::size_type);
    UNODB_EXPECT_EQ(kv.size(), max_key_size);
  }
}

/// Verify proper padding to maxlen.
UNODB_TEST(ARTKeyEncodeDecodeTest, EncodeTextC0001) {
  unodb::key_encoder enc;
  do_simple_pad_test(enc, "");
  do_simple_pad_test(enc, "abc");
  do_simple_pad_test(enc, "brown");
  do_simple_pad_test(enc, "banana");
}

/// Unit test variant examines truncation for a key whose length is
/// maxlen - 1.
UNODB_TEST(ARTKeyEncodeDecodeTest, EncodeTextC0012) {
  unodb::key_encoder enc;
  do_pad_test_large_string(enc, unodb::key_encoder::maxlen - 1);
}

/// Unit test variant examines truncation for a key whose length is
/// exactly maxlen.
UNODB_TEST(ARTKeyEncodeDecodeTest, EncodeTextC0013) {
  unodb::key_encoder enc;
  do_pad_test_large_string(enc, unodb::key_encoder::maxlen);
}

/// Unit test where the key is truncated.
UNODB_TEST(ARTKeyEncodeDecodeTest, EncodeTextC0014) {
  unodb::key_encoder enc;
  do_pad_test_large_string(enc, unodb::key_encoder::maxlen + 1, true);
}

/// Unit test where the key is truncated.
UNODB_TEST(ARTKeyEncodeDecodeTest, EncodeTextC0015) {
  unodb::key_encoder enc;
  do_pad_test_large_string(enc, unodb::key_encoder::maxlen + 2, true);
}

/// Verify the lexicographic sort order obtained for {bro, brown,
/// break, bre}, including verifying that the pad byte causes a prefix
/// such as "bro" to sort before a term which extends that prefix,
/// such as "brown".
UNODB_TEST(ARTKeyEncodeDecodeTest, EncodeTextC0020) {
  key_factory fac;
  unodb::key_encoder enc;
  const auto k0 = fac.make_key_view(enc.reset().encode_text("brown"));
  const auto k1 = fac.make_key_view(enc.reset().encode_text("bro"));
  const auto k2 = fac.make_key_view(enc.reset().encode_text("break"));
  const auto k3 = fac.make_key_view(enc.reset().encode_text("bre"));
#ifndef NDEBUG
  std::cerr << "k0=";
  unodb::detail::dump_key(std::cerr, k0);
  std::cerr << "\n";
  std::cerr << "k1=";
  unodb::detail::dump_key(std::cerr, k1);
  std::cerr << "\n";
  std::cerr << "k2=";
  unodb::detail::dump_key(std::cerr, k2);
  std::cerr << "\n";
  std::cerr << "k3=";
  unodb::detail::dump_key(std::cerr, k3);
  std::cerr << "\n";
#endif
  // Inspect the implied sort order without sorting.
  //
  UNODB_EXPECT_LT(compare(k3, k2), 0);  // bre < break
  UNODB_EXPECT_LT(compare(k2, k1), 0);  // break < bro
  UNODB_EXPECT_LT(compare(k1, k0), 0);  // bro < brown
}

/// Verify that trailing nul (0x00) bytes are removed as part of the
/// truncation and logical padding logic.
UNODB_TEST(ARTKeyEncodeDecodeTest, EncodeTextC0021) {
  key_factory fac;
  unodb::key_encoder enc;
  // Use std::array rather than "C" strings since the nul would
  // otherwise be interpreted as the end of the C string.
  constexpr auto a1 = std::array<const std::byte, 5>{
      std::byte{'b'}, std::byte{'r'}, std::byte{'o'}, std::byte{'w'},
      std::byte{'n'}};
  constexpr auto a2 = std::array<const std::byte, 6>{
      std::byte{'b'}, std::byte{'r'}, std::byte{'o'},
      std::byte{'w'}, std::byte{'n'}, std::byte{0x00}};
  using S = std::span<const std::byte>;
  const auto k1 = fac.make_key_view(enc.reset().encode_text(S(a1)));
  const auto k2 = fac.make_key_view(enc.reset().encode_text(S(a2)));
  UNODB_EXPECT_EQ(compare(k1, k2), 0);    // same sort order.
  UNODB_EXPECT_EQ(k1.size(), k2.size());  // same number of bytes.
  UNODB_EXPECT_EQ(k1.size_bytes(),
                  a1.size() + 1 + sizeof(unodb::key_encoder::size_type));
#ifndef NDEBUG
  std::cerr << "k1=";
  unodb::detail::dump_key(std::cerr, k1);
  std::cerr << "\n";
  std::cerr << "k2=";
  unodb::detail::dump_key(std::cerr, k2);
  std::cerr << "\n";
#endif
}

/// Verifies that an embedded nul byte is supported.
UNODB_TEST(ARTKeyEncodeDecodeTest, EncodeTextC0022) {
  key_factory fac;
  unodb::key_encoder enc;
  // Use std::array rather than "C" strings since the nul would
  // otherwise be interpreted as the end of the C string.
  constexpr auto a1 = std::array<const std::byte, 5>{
      std::byte{'b'}, std::byte{'r'}, std::byte{0}, std::byte{'w'},
      std::byte{'n'}};
  using S = std::span<const std::byte>;
  const auto k1 = fac.make_key_view(enc.reset().encode_text(S(a1)));
  UNODB_EXPECT_EQ(k1.size_bytes(),
                  a1.size() + 1 + sizeof(unodb::key_encoder::size_type));
#ifndef NDEBUG
  std::cerr << "k1=";
  unodb::detail::dump_key(std::cerr, k1);
  std::cerr << "\n";
#endif
}

}  // namespace
