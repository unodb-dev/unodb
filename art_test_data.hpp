// Copyright 2026 UnoDB contributors
#ifndef UNODB_DETAIL_ART_TEST_DATA_HPP
#define UNODB_DETAIL_ART_TEST_DATA_HPP

/// \file
/// Shared ART test data and key encoding helpers.
///
/// \ingroup test-internals
///
/// Pure test data (values, key byte patterns), `unodb::key_encoder`-based key
/// construction helpers, and a helper for copying encoded keys into
/// caller-owned storage, shared between test and benchmark executables.
/// Deliberately free of Google Test and database template dependencies so that
/// any target can include it without linking extra libraries.
///
/// \warning The `make_*` builders below return a view into the caller's
/// `unodb::key_encoder` buffer, not into storage of their own. Such a view is
/// valid only until the next non-`const` call on that same encoder — any
/// `make_*` helper, `reset()`, `encode()`, `encode_text()`, `append_bytes()`
/// or `ensure_available()` — or until that encoder is destroyed. Re-encoding
/// after `reset()` overwrites the bytes under the view, and outgrowing the
/// inline buffer moves them out from under it. Either consume the view inside
/// the full expression that produced it, copy it out with
/// \ref unodb::test_data::copy_key, or give each simultaneously live key its
/// own encoder.
///
/// The chain-shape notes on the `make_key_*` builders count single-child chain
/// levels and name every branching inode by the key byte it dispatches on.
/// They give no absolute depths on purpose: chains are built bottom-up from
/// the key end, so a level's depth follows the key length and shifts again
/// whenever a divergence higher up splits an ancestor, while its dispatch byte
/// does not.

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

#include "art_common.hpp"
#include "assert.hpp"

namespace unodb::test_data {

// The byte arrays below are deliberately not `inline`, unlike every other
// constant in this header. `test_values` constant-initializes a `std::span`
// over each of them, and under `_GLIBCXX_DEBUG` plus UBSan, GCC 14.2's
// constexpr evaluator cannot fold the sanitizer-instrumented `ptr + count`
// inside `__glibcxx_requires_valid_range` when `ptr` is the address of an
// inline (vague-linkage) variable - "is not a constant expression" - while it
// folds the same expression over an internal-linkage array. Reproduced with
// Ubuntu 24.04's g++-14 14.2.0-4ubuntu2~24.04.1; Homebrew GCC 14.4 accepts
// both. micro_benchmark_utils.hpp's `values` relies on the same shape.

/// Test value of one zero byte.
constexpr auto test_value_1 = std::array<std::byte, 1>{std::byte{0x00}};
/// Two-byte test value.
constexpr auto test_value_2 =
    std::array<std::byte, 2>{std::byte{0x00}, std::byte{0x02}};
/// Three-byte test value.
constexpr auto test_value_3 =
    std::array<std::byte, 3>{std::byte{0x03}, std::byte{0x00}, std::byte{0x01}};
/// Four-byte test value.
constexpr auto test_value_4 = std::array<std::byte, 4>{
    std::byte{0x04}, std::byte{0x01}, std::byte{0x00}, std::byte{0x02}};
/// Five-byte test value.
constexpr auto test_value_5 =
    std::array<std::byte, 5>{std::byte{0x05}, std::byte{0xF4}, std::byte{0xFF},
                             std::byte{0x00}, std::byte{0x01}};
/// Empty test value.
constexpr auto empty_test_value = std::array<std::byte, 0>{};

/// Views over the test values above, for `unodb::value_view` databases.
inline constexpr std::array<unodb::value_view, 6> test_values = {
    unodb::value_view{test_value_1},     // [0] { 00              }
    unodb::value_view{test_value_2},     // [1] { 00 02           }
    unodb::value_view{test_value_3},     // [2] { 03 00 01        }
    unodb::value_view{test_value_4},     // [3] { 04 01 00 02     }
    unodb::value_view{test_value_5},     // [4] { 05 F4 FF 00 01  }
    unodb::value_view{empty_test_value}  // [5] {                 }
};

/// Test values for `std::uint64_t` value databases.
inline constexpr std::array<std::uint64_t, 6> test_values_u64 = {0, 1, 2,
                                                                 3, 4, 5};

/// Byte backing the oversize rejection views below.
inline constexpr std::byte too_long_view_byte{0x00};

/// Length one greater than the maximum supported key or value view size.
inline constexpr auto too_long_view_length =
    std::max(static_cast<std::uint64_t>(
                 std::numeric_limits<unodb::key_size_type>::max()),
             static_cast<std::uint64_t>(
                 std::numeric_limits<unodb::value_size_type>::max())) +
    1U;
static_assert(too_long_view_length >
                  static_cast<std::uint64_t>(
                      std::numeric_limits<unodb::value_size_type>::max()),
              "No representable over-limit view length exists any more: the "
              "rejection tests need rethinking, not a bigger constant");

/// A key view too long to be stored in any tree, for rejection tests.
/// \warning Only the length is real — the view spans a single byte. Never read
/// through it; pass it only to APIs that reject it by size before touching the
/// data (currently only insert paths check the size). Deliberately a function
/// rather than a `constexpr` variable, despite CONTRIBUTING.md's rule: under
/// `_GLIBCXX_DEBUG` libstdc++'s `std::span` constructor forms `ptr + count`,
/// and constant-initializing that offset past a single-byte object is not a
/// core constant expression, which Clang rejects outright.
[[nodiscard]] inline unodb::key_view too_long_key_view() noexcept {
  return unodb::key_view{&too_long_view_byte, too_long_view_length};
}

/// The same view as \ref too_long_key_view, spelled as a value view for
/// value-rejection tests; the warning there applies here too.
[[nodiscard]] inline unodb::value_view too_long_value_view() noexcept {
  return too_long_key_view();
}

/// String keys exercising text encoding, in insertion order; deliberately not
/// sorted: inserting `ostritch` (0x6F) after `yellow` (0x79) forces the
/// mid-array shift branch of `basic_inode_16::add_to_nonfull`.
inline constexpr std::array<std::string_view, 8> encoded_text_keys{
    "", "a", "abba", "banana", "camel", "yellow", "ostritch", "zebra"};

/// Byte that every byte of \ref chain_key_filler repeats.
inline constexpr std::uint8_t chain_key_filler_byte = 0x42;

/// Filler word giving consecutive key bytes a shared pattern, forcing
/// key-prefix chains: keys sharing more than key_prefix_capacity (7) bytes
/// need a chain of internal nodes because the dispatch byte after the stored
/// prefix is the same for all of them. The zero words in \ref make_key_18 and
/// \ref make_key_26 force chains the same way. 0x42 keeps filler bytes apart
/// from those zero words and from \ref chain_key_filler_alt in a dump. The
/// OLCRemoveDescentVsSplit tests rely on all of its bytes being equal and
/// restrict its value; see there.
inline constexpr auto chain_key_filler =
    std::uint64_t{0x0101'0101'0101'0101ULL * chain_key_filler_byte};

/// A second filler word, for keys that must diverge from \ref
/// chain_key_filler at the first byte of the corresponding key word.
inline constexpr auto chain_key_filler_alt =
    std::uint64_t{0x4343434343434343ULL};

/// Encode a 9-byte key (uint8 + uint64).
/// Same tag byte → 8 shared bytes when uint64 values are small.
[[nodiscard]] inline unodb::key_view make_key(unodb::key_encoder& enc
                                              UNODB_DETAIL_LIFETIMEBOUND,
                                              std::uint8_t tag,
                                              std::uint64_t v) {
  return enc.reset().encode(tag).encode(v).get_key_view();
}

/// Encode a 1-byte key (uint8 only).
/// Diverges at byte 0 from any key with a different first byte.
/// Its single byte is consumed by one dispatch byte, so the entry sits
/// directly in the root inode's slot — leaf pointer, or packed value in
/// can_eliminate_leaf trees — with no key-prefix chain above it.
[[nodiscard]] inline unodb::key_view make_short_key(unodb::key_encoder& enc
                                                    UNODB_DETAIL_LIFETIMEBOUND,
                                                    std::uint8_t tag) {
  return enc.reset().encode(tag).get_key_view();
}

/// Encode a 10-byte key (uint8 + uint64 + uint8).
/// When used with the same tag and v as \ref make_key, the 9-byte key is a
/// prefix of this 10-byte key — which ART does not support.  Use
/// different v values to avoid prefix relationships.
/// Both lengths (9 and 10) exceed key_prefix_capacity + 1 = 8.
[[nodiscard]] inline unodb::key_view make_long_key(unodb::key_encoder& enc
                                                   UNODB_DETAIL_LIFETIMEBOUND,
                                                   std::uint8_t tag,
                                                   std::uint64_t v,
                                                   std::uint8_t suffix) {
  return enc.reset().encode(tag).encode(v).encode(suffix).get_key_view();
}

/// Copy an encoded key into caller-owned storage; the returned view is
/// valid for the lifetime of `buf`.
/// \pre `kv.size() <= buf.size()`. The copy is bounded by `kv`, not by `buf`,
/// and the returned view's length is `kv.size()`, so an oversize key both
/// overruns `buf` and yields a view past its end. Enforced only by an assert,
/// which `NDEBUG` compiles out.
[[nodiscard]] constexpr unodb::key_view copy_key(
    unodb::key_view kv,
    std::span<std::byte> buf UNODB_DETAIL_LIFETIMEBOUND) noexcept {
  UNODB_DETAIL_ASSERT(kv.size() <= buf.size());
  std::ranges::copy(kv, buf.begin());
  return {buf.data(), kv.size()};
}

/// Encode a 17-byte key: 0xAA × 10, then byte10, then 0xAA × 5, then last.
/// Two keys sharing byte10 differ only at byte 16: two single-child chain
/// levels above the inode that branches on byte 16. A key with a different
/// byte10 diverges inside that inode's prefix and splits it, leaving an inode
/// branching on byte10 with one byte-16 inode per byte10 value below it.
[[nodiscard]] inline unodb::key_view make_key_17_byte10(
    unodb::key_encoder& enc UNODB_DETAIL_LIFETIMEBOUND, std::uint8_t byte10,
    std::uint8_t last) {
  enc.reset();
  for (unsigned i = 0; i < 10; ++i) enc.encode(std::uint8_t{0xAA});
  enc.encode(byte10);
  for (unsigned i = 11; i < 16; ++i) enc.encode(std::uint8_t{0xAA});
  enc.encode(last);
  return enc.get_key_view();
}

/// Encode a 17-byte key: 0xAA × 16, then last — \ref make_key_17_byte10 with
/// the uniform 0xAA prefix. Kept a distinct name rather than an overload so
/// that an argument-count slip is a compile error instead of a different but
/// still valid key shape.
[[nodiscard]] inline unodb::key_view make_key_17(unodb::key_encoder& enc
                                                 UNODB_DETAIL_LIFETIMEBOUND,
                                                 std::uint8_t last) {
  return make_key_17_byte10(enc, std::uint8_t{0xAA}, last);
}

/// Encode an 18-byte key (uint64 + uint8 + uint64 + uint8).
/// All keys share bytes [0..7], and within a mid group also bytes [9..16]:
/// one single-child chain level above the inode that branches on byte 8 =
/// mid, and one more above each inode that branches on byte 17 = bottom.
[[nodiscard]] inline unodb::key_view make_key_18(unodb::key_encoder& enc
                                                 UNODB_DETAIL_LIFETIMEBOUND,
                                                 std::uint8_t mid,
                                                 std::uint8_t bottom) {
  return enc.reset()
      .encode(chain_key_filler)
      .encode(mid)
      .encode(std::uint64_t{0})
      .encode(bottom)
      .get_key_view();
}

/// Encode an 11-byte key (uint8 + uint64 + zero uint8 + uint8).
/// Keys sharing tag share bytes [0..9] → one single-child chain level above
/// the inode that branches on byte 10 = bottom (a CD=1 chain cut shape).
[[nodiscard]] inline unodb::key_view make_key_11(unodb::key_encoder& enc
                                                 UNODB_DETAIL_LIFETIMEBOUND,
                                                 std::uint8_t tag,
                                                 std::uint8_t bottom) {
  return enc.reset()
      .encode(tag)
      .encode(chain_key_filler)
      .encode(std::uint8_t{0})
      .encode(bottom)
      .get_key_view();
}

/// Encode a 26-byte key (uint8 + uint64 × 3 + uint8).
/// Two keys sharing tag share 25 bytes: three single-child chain levels above
/// the inode that branches on byte 25 = bottom. A key with a different tag
/// diverges at byte 0, splitting the topmost level and adding an inode that
/// branches on byte 0 above it.
[[nodiscard]] inline unodb::key_view make_key_26(unodb::key_encoder& enc
                                                 UNODB_DETAIL_LIFETIMEBOUND,
                                                 std::uint8_t tag,
                                                 std::uint8_t bottom) {
  return enc.reset()
      .encode(tag)
      .encode(chain_key_filler)
      .encode(std::uint64_t{0})
      .encode(std::uint64_t{0})
      .encode(bottom)
      .get_key_view();
}

/// Encode a 34-byte key (uint8 + uint64 v1 + filler uint64 × 3 + uint8).
/// Two keys sharing tag and v1 share 33 bytes → four chain levels after the
/// sibling removal; pass `chain_key_filler` as v1 for the shared chain and
/// `chain_key_filler_alt` for a key diverging at the first chain level.
[[nodiscard]] inline unodb::key_view make_key_34(unodb::key_encoder& enc
                                                 UNODB_DETAIL_LIFETIMEBOUND,
                                                 std::uint8_t tag,
                                                 std::uint64_t v1,
                                                 std::uint8_t bottom) {
  return enc.reset()
      .encode(tag)
      .encode(v1)
      .encode(chain_key_filler)
      .encode(chain_key_filler)
      .encode(chain_key_filler)
      .encode(bottom)
      .get_key_view();
}

/// Decode a `std::uint64_t` key from its encoded form.
/// \pre `akey.size() >= sizeof(std::uint64_t)`. Only the leading 8 bytes are
/// read, and `unodb::key_decoder` stores its capacity but never checks it, so
/// a shorter view is a silent out-of-bounds read.
[[nodiscard]] inline std::uint64_t decode(unodb::key_view akey) noexcept {
  UNODB_DETAIL_ASSERT(akey.size() >= sizeof(std::uint64_t));
  unodb::key_decoder dec{akey};
  std::uint64_t k;
  dec.decode(k);
  return k;
}

}  // namespace unodb::test_data

#endif  // UNODB_DETAIL_ART_TEST_DATA_HPP
