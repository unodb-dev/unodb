// Copyright 2025 UnoDB contributors

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

// IWYU pragma: no_include <__cstddef/byte.h>
// IWYU pragma: no_include <string>
// IWYU pragma: no_include <string_view>

#include <array>
#include <cstddef>  // IWYU pragma: keep
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "art_common.hpp"
#include "assert.hpp"  // UNODB_DETAIL_ASSERT
#include "db_test_utils.hpp"
#include "gtest_utils.hpp"

namespace {

template <class Db>
class ARTKeyViewCorrectnessTest : public ::testing::Test {
 public:
  using Test::Test;
};

using ARTTypes =
    ::testing::Types<unodb::test::key_view_db, unodb::test::key_view_mutex_db,
                     unodb::test::key_view_olc_db>;

UNODB_TYPED_TEST_SUITE(ARTKeyViewCorrectnessTest, ARTTypes)

/// Unit test of correct rejection of a key which is too large to be
/// stored in the tree.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, TooLongKey) {
  constexpr std::byte fake_val{0x00};
  const unodb::key_view too_long{
      &fake_val,
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) +
          1U};

  unodb::test::tree_verifier<TypeParam> verifier;

  UNODB_ASSERT_THROW(std::ignore = verifier.get_db().insert(too_long, {}),
                     std::length_error);

  verifier.assert_empty();

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_growing_inodes({0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Unit test inserts several string keys with proper encoding and
/// validates the tree.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, EncodedTextKeys) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];
  verifier.insert(enc.reset().encode_text("").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("a").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("abba").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("banana").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("camel").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("yellow").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("ostritch").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("zebra").get_key_view(), val);
  verifier.check_present_values();  // checks keys and key ordering.
}

// ===================================================================
// key_view tests for the dispatch byte collision bug.
//
// key_prefix_capacity is 7 bytes.  When two key_view keys share more
// than 7 bytes of common prefix, the leaf-to-inode4 split must create
// a chain of internal nodes rather than a single inode4, because the
// dispatch byte (the byte immediately after the stored prefix) is the
// same for both keys.
//
// Keys are 9 bytes: (uint8_t tag, uint64_t value).  Same tag + small
// values → 8 shared bytes, triggering the bug.  10-byte keys
// (uint8_t tag, uint64_t value, uint8_t suffix) are used for
// mixed-length tests.  Both lengths exceed key_prefix_capacity + 1.
//
// Test plan groups:
//   0. Bug reproduction — minimal cases
//   1. Prefix boundary cases — validate various shared-prefix lengths
//   2. Node growth — inode4 -> inode16 -> inode48 -> inode256
//   3. Removal & shrinkage — chained inodes collapse correctly
//   4. Mixed key lengths — different-length keys with long shared prefix
//   5. Duplicate & edge cases — duplicate insert, get-missing
//   6. Stats verification — node counts at intermediate states
//     6a. Insert stats — verify chain structure after insert
//     6b. Partial remove — bottom inode shrinks, chain preserved
//         I16→I4, I48→I16, I256→I48
//     6c. Full remove — remove all keys through each bottom inode size
//         I4, I16, I48, I256
//     6d. Cascade — chain under parent at min_size, removing chain
//         triggers parent shrinkage: I4→leaf, I16→I4, I48→I16, I256→I48
//     6e. Multi-level chain — 17-byte keys, 2 chain levels
// ===================================================================

// Helper: encode a 9-byte key (uint8 + uint64).
// Same tag byte → 8 shared bytes when uint64 values are small.
inline unodb::key_view make_key(unodb::key_encoder& enc, std::uint8_t tag,
                                std::uint64_t v) {
  return enc.reset().encode(tag).encode(v).get_key_view();
}

// Helper: encode a 1-byte key (uint8 only).
// Diverges at byte 0 from any key with a different first byte.
// Used in cascade tests where we need direct-leaf children of the root
// alongside a chain subtree.
[[maybe_unused]] inline unodb::key_view make_short_key(unodb::key_encoder& enc,
                                                       std::uint8_t tag) {
  return enc.reset().encode(tag).get_key_view();
}

// Helper: encode a 10-byte key (uint8 + uint64 + uint8).
// When used with the same tag and v as make_key, the 9-byte key is a
// prefix of this 10-byte key — which ART does not support.  Use
// different v values to avoid prefix relationships.
// Both lengths (9 and 10) exceed key_prefix_capacity + 1 = 8.
inline unodb::key_view make_long_key(unodb::key_encoder& enc, std::uint8_t tag,
                                     std::uint64_t v, std::uint8_t suffix) {
  return enc.reset().encode(tag).encode(v).encode(suffix).get_key_view();
}

// -------------------------------------------------------------------
// Group 0: Original bug reproduction tests
// -------------------------------------------------------------------

/// Two 9-byte keys sharing 8 bytes — dispatch byte collision.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CompoundKeysLongSharedPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 (prefix=7, 1 child) + bottom I4 (2 children) + 2 leaves
  verifier.assert_node_counts({2, 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Three keys with the same tag byte and small uint64 values.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ThreeCompoundKeysLongSharedPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_key(enc, 0x42, 3), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + bottom I4 (3 children) + 3 leaves
  verifier.assert_node_counts({3, 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// 9-byte keys sharing 8 bytes — minimal collision case.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, NineByteCompoundKeysLongPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({2, 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

// -------------------------------------------------------------------
// Group 1: Prefix boundary cases
// -------------------------------------------------------------------

/// Keys identical except last byte — maximum chaining depth for 9-byte keys.
/// 9-byte keys sharing 8 bytes, differing only at byte 8.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest,
                 CompoundKeysIdenticalExceptLastByte) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_key(enc, 0x42, 3), val);
  verifier.insert(make_key(enc, 0x42, 4), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + bottom I4 (4 children, at capacity) + 4 leaves
  verifier.assert_node_counts({4, 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Two 17-byte keys sharing 16 bytes — forces two consecutive chain
/// nodes (depth 0→8 and depth 8→16) before the normal 2-child split.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, MultiLevelChain) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // 17 bytes: 0xAA × 16, then 0x01 vs 0x02.
  auto make17 = [&](std::uint8_t last) {
    enc.reset();
    for (unsigned i = 0; i < 16; ++i) enc.encode(std::uint8_t{0xAA});
    enc.encode(last);
    return enc.get_key_view();
  };
  verifier.insert(make17(0x01), val);
  verifier.insert(make17(0x02), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // 2 chain I4s (depth 0→8, 8→16) + bottom I4 (2 children) + 2 leaves
  verifier.assert_node_counts({2, 3, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Three 17-byte keys: A and B share 16 bytes, C diverges at byte 10.
/// After inserting A and B (two chain levels), inserting C splits the
/// second chain node mid-prefix.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest,
                 InsertDivergingAtIntermediateChainDepth) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make17 = [&](std::uint8_t byte10, std::uint8_t last) {
    enc.reset();
    for (unsigned i = 0; i < 10; ++i) enc.encode(std::uint8_t{0xAA});
    enc.encode(byte10);
    for (unsigned i = 11; i < 16; ++i) enc.encode(std::uint8_t{0xAA});
    enc.encode(last);
    return enc.get_key_view();
  };
  // A and B share 16 bytes (byte10=0xAA), differ at byte 16.
  verifier.insert(make17(0xAA, 0x01), val);
  verifier.insert(make17(0xAA, 0x02), val);
  // C diverges at byte 10 — splits the second chain node.
  verifier.insert(make17(0xBB, 0x03), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 (bytes 0-7) + split I4 (at byte 10) + chain I4 (bytes 11-15)
  // + bottom I4 (A,B) + 3 leaves
  verifier.assert_node_counts({3, 4, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

// -------------------------------------------------------------------
// Group 2: Node growth (inode4 -> inode16 -> inode48 -> inode256)
// -------------------------------------------------------------------

/// 5 keys with same 8-byte prefix — forces inode4 -> inode16 growth.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CompoundKeysFiveChildren) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 5; ++i) {
    verifier.insert(make_key(enc, 0x42, i), val);
  }
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + I16 (5 children) + 5 leaves
  verifier.assert_node_counts({5, 1, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// 17 keys — forces inode16 -> inode48 growth.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CompoundKeysSeventeenChildren) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 17; ++i) {
    verifier.insert(make_key(enc, 0x42, i), val);
  }
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + I48 (17 children) + 17 leaves
  verifier.assert_node_counts({17, 1, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// 50 keys — forces inode48 -> inode256 growth.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CompoundKeysFiftyChildren) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 50; ++i) {
    verifier.insert(make_key(enc, 0x42, i), val);
  }
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + I256 (50 children) + 50 leaves
  verifier.assert_node_counts({50, 1, 0, 0, 1});
#endif  // UNODB_DETAIL_WITH_STATS
}

// -------------------------------------------------------------------
// Group 3: Removal & shrinkage
// -------------------------------------------------------------------

// Group 3a: Chain collapse scenarios

/// Insert 2 colliding keys, remove one, verify the other is still found.
/// The chain of inode_4s should collapse to a single leaf.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CompoundKeysInsertThenRemove) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // Bottom I4 collapsed via leave_last_child.  Chain I4 remains with
  // 1 child (the surviving leaf).
  verifier.assert_node_counts({1, 1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Insert 3 keys: two sharing 8 bytes, one diverging earlier.
/// Remove one of the 8-byte-shared pair.  The surviving structure
/// should be an inode with 2 children (the remaining keys).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, RemoveFromChainLeavesInode) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // Keys 1 and 2 share 8 bytes; key 3 diverges at byte 5.
  // key3 uint64 = 0x0000000100000000 differs at byte 5 (overall).
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_key(enc, 0x42, 0x0000000100000000ULL), val);
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // Root I4 (2 children: key3 leaf + chain) + chain I4 + 2 leaves
  verifier.assert_node_counts({2, 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Insert 3 colliding keys, remove in reverse order, assert empty.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, RemoveAllFromChainReverseOrder) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_key(enc, 0x42, 3), val);
  verifier.remove(make_key(enc, 0x42, 3));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.assert_empty();
}

/// Insert 3 colliding keys, remove in forward order, assert empty.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, RemoveAllFromChainForwardOrder) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_key(enc, 0x42, 3), val);
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.remove(make_key(enc, 0x42, 3));
  verifier.assert_empty();
}

// Group 3b: Shrinkage at chain terminal

/// Insert 5 keys (-> inode16 at chain terminal), remove 3 (-> inode4).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ShrinkInode16InChain) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 5; ++i) {
    verifier.insert(make_key(enc, 0x42, i), val);
  }
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.remove(make_key(enc, 0x42, 3));
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // I16(5) shrinks to I4(4) on first remove, then 2 more removes → I4(2).
  verifier.assert_node_counts({2, 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Insert 17 keys (-> inode48), remove 13 (-> shrink to inode4).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ShrinkInode48InChain) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 17; ++i) {
    verifier.insert(make_key(enc, 0x42, i), val);
  }
  for (std::uint64_t i = 1; i <= 13; ++i) {
    verifier.remove(make_key(enc, 0x42, i));
  }
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // I48(17) shrinks to I16(16) on first remove, then I16(5) shrinks
  // to I4(4) on 13th remove.
  verifier.assert_node_counts({4, 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Insert 5 keys (-> inode16), remove all 5, assert empty.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ShrinkToEmptyFromInode16InChain) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 5; ++i) {
    verifier.insert(make_key(enc, 0x42, i), val);
  }
  for (std::uint64_t i = 1; i <= 5; ++i) {
    verifier.remove(make_key(enc, 0x42, i));
  }
  verifier.assert_empty();
}

// Group 3c: Mixed-depth removal

/// Insert a 10-byte and 9-byte key sharing 8 bytes (same tag, different
/// uint64 values).  Remove the 10-byte key, verify 9-byte key.  Then
/// remove it too, assert empty.
///
/// Note: the two keys must NOT be in a prefix relationship — ART does
/// not support one key being a prefix of another.  Using different
/// uint64 values ensures they diverge within the shared bytes.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, RemoveMixedLengthFromChain) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_long_key(enc, 0x42, 1, 0xFF), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.remove(make_long_key(enc, 0x42, 1, 0xFF));
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // Chain I4 + 1 surviving leaf
  verifier.assert_node_counts({1, 1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.assert_empty();
}

// Group 3d: Stress removal

/// Insert 24 keys (divergence at positions 7..18), remove every other
/// key, verify remaining.  Then remove all, assert empty.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, StressInsertRemoveAtEveryPosition) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  constexpr unsigned key_len = 20;
  constexpr unsigned prefix_cap = 7;

  // Insert all 24 keys: for each divergence position d (7..18), two
  for (unsigned d = prefix_cap; d < key_len - 1; ++d) {
    for (unsigned variant = 1; variant <= 2; ++variant) {
      enc.reset();
      enc.encode(static_cast<std::uint8_t>(d));
      for (unsigned i = 1; i < key_len; ++i) {
        if (i < prefix_cap) {
          enc.encode(std::uint8_t{0xAA});
        } else if (i == d) {
          enc.encode(static_cast<std::uint8_t>(variant));
        } else {
          enc.encode(std::uint8_t{0x00});
        }
      }
      verifier.insert(enc.get_key_view(), val);
    }
  }
  verifier.check_present_values();

  // Remove variant=1 keys (one from each pair).
  for (unsigned d = prefix_cap; d < key_len - 1; ++d) {
    enc.reset();
    enc.encode(static_cast<std::uint8_t>(d));
    for (unsigned i = 1; i < key_len; ++i) {
      if (i < prefix_cap) {
        enc.encode(std::uint8_t{0xAA});
      } else if (i == d) {
        enc.encode(std::uint8_t{1});
      } else {
        enc.encode(std::uint8_t{0x00});
      }
    }
    verifier.remove(enc.get_key_view());
  }
  verifier.check_present_values();

  // Remove remaining variant=2 keys.
  for (unsigned d = prefix_cap; d < key_len - 1; ++d) {
    enc.reset();
    enc.encode(static_cast<std::uint8_t>(d));
    for (unsigned i = 1; i < key_len; ++i) {
      if (i < prefix_cap) {
        enc.encode(std::uint8_t{0xAA});
      } else if (i == d) {
        enc.encode(std::uint8_t{2});
      } else {
        enc.encode(std::uint8_t{0x00});
      }
    }
    verifier.remove(enc.get_key_view());
  }
  verifier.assert_empty();
}

// -------------------------------------------------------------------
// Group 4: Mixed key lengths
// -------------------------------------------------------------------

/// A 10-byte key and a 9-byte key sharing 8 bytes (same tag, different
/// uint64 values).  The keys must not be in a prefix relationship.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, MixedLengthKeysLongPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_long_key(enc, 0x42, 1, 0xFF), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  constexpr std::uint64_t I = unodb::test::is_olc_db<TypeParam> ? 2 : 3;
  verifier.assert_node_counts({2, I, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

// -------------------------------------------------------------------
// Group 5: Duplicate & edge cases
// -------------------------------------------------------------------

/// Inserting the same 9-byte key twice returns false on the second insert.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CompoundKeyDuplicateInsert) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  // Second insert of same key should fail.
  UNODB_ASSERT_FALSE(verifier.get_db().insert(make_key(enc, 0x42, 1), val));
  verifier.check_present_values();
}

/// Get with a key sharing 8 bytes but differing at the last byte
/// should return empty when only one key is present.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CompoundKeyGetMissing) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  const auto result = verifier.get_db().get(make_key(enc, 0x42, 2));
  UNODB_ASSERT_FALSE(TypeParam::key_found(result));
  verifier.check_present_values();
}

#ifdef UNODB_DETAIL_WITH_STATS

// ===================================================================
// Group 6: Stats verification — node counts at intermediate states
//
// These tests verify that the tree's internal bookkeeping (node counts,
// memory use) is correct after insert and remove operations involving
// single-child chain I4 nodes.
//
// Inode size thresholds:
//   I4:   min=2, capacity=4   (shrinks via leave_last_child)
//   I16:  min=5, capacity=16  (shrinks to I4)
//   I48:  min=17, capacity=48 (shrinks to I16)
//   I256: min=49, capacity=256 (shrinks to I48)
//
// For keys make_key(enc, 0x42, v) with small v:
//   9 bytes: [0x42, 0,0,0,0,0,0, 0, v]
//   All share 8 bytes, differ only at byte 8.
//   Tree: chain I4 (prefix=7, dispatch=0x00) → bottom inode (children
//   keyed by v).
// ===================================================================

// -------------------------------------------------------------------
// Group 6b: Partial remove — bottom inode shrinks, chain preserved
// -------------------------------------------------------------------

/// Insert 3 keys, remove 1.  Chain I4 + bottom I4 (2 children).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainPartialRemoveI4) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 3; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({3, 2, 0, 0, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
  // I4(3) → remove → I4(2).  Chain I4 unchanged.
  verifier.assert_node_counts({2, 2, 0, 0, 0});
}

/// Insert 5 keys (→I16), remove 1 (I16 at min_size → shrink to I4).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainShrinkI16ToI4) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 5; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({5, 1, 1, 0, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
  // I16(5) at min_size → shrink to I4(4).  Chain I4 + bottom I4.
  verifier.assert_node_counts({4, 2, 0, 0, 0});
  verifier.assert_shrinking_inodes({0, 1, 0, 0});
}

/// Insert 17 keys (→I48), remove 1 (I48 at min_size → shrink to I16).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainShrinkI48ToI16) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 17; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({17, 1, 0, 1, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
  // I48(17) at min_size → shrink to I16(16).  Chain I4 + I16.
  verifier.assert_node_counts({16, 1, 1, 0, 0});
  verifier.assert_shrinking_inodes({0, 0, 1, 0});
}

/// Insert 49 keys (→I256), remove 1 (I256 at min_size → shrink to I48).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainShrinkI256ToI48) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 49; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({49, 1, 0, 0, 1});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
  // I256(49) at min_size → shrink to I48(48).  Chain I4 + I48.
  verifier.assert_node_counts({48, 1, 0, 1, 0});
  verifier.assert_shrinking_inodes({0, 0, 0, 1});
}

// -------------------------------------------------------------------
// Group 6c: Full remove — all keys removed through each bottom inode
// -------------------------------------------------------------------

/// Insert 17 keys into chain (bottom I48), remove all.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainRemoveAllFromI48) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 17; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  for (std::uint64_t i = 1; i <= 17; ++i)
    verifier.remove(make_key(enc, 0x42, i));
  verifier.assert_empty();
}

/// Insert 49 keys into chain (bottom I256), remove all.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainRemoveAllFromI256) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  for (std::uint64_t i = 1; i <= 49; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  for (std::uint64_t i = 1; i <= 49; ++i)
    verifier.remove(make_key(enc, 0x42, i));
  verifier.assert_empty();
}

// -------------------------------------------------------------------
// Group 6d: Cascade — chain under parent at min_size, removing chain
// triggers parent shrinkage
//
// Each test creates a parent inode at its min_size, where one child
// slot points to a chain (2 keys with tag=0x42 sharing 8 bytes) and
// the remaining slots are direct leaves (distinct tag bytes).
// Removing both chain keys should: reclaim the chain, then shrink
// the parent through the normal shrinkage path.
// -------------------------------------------------------------------

/// Chain under I4(2 children).  Remove chain → I4 collapses via
/// leave_last_child → root becomes the surviving leaf.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CascadeChainUnderI4) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // Insert chain keys first so the chain forms at root level, then
  // insert a short key that splits the root prefix → parent I4.
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_short_key(enc, 0x01), val);
  // root I4(2: 0x42→chain, 0x01→leaf) + chain I4 + bottom I4 + 3 leaves
  constexpr std::uint64_t I3 = unodb::test::is_olc_db<TypeParam> ? 3 : 2;
  verifier.assert_node_counts({3, I3, 0, 0, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  // Bottom I4 collapsed via leave_last_child.  Chain I4 has 1 child (leaf).
  // Root I4 still has 2 children.
  constexpr std::uint64_t I2 = unodb::test::is_olc_db<TypeParam> ? 2 : 1;
  verifier.assert_node_counts({2, I2, 0, 0, 0});

  verifier.remove(make_key(enc, 0x42, 2));
  verifier.check_present_values();
  // Chain fully reclaimed.  Root I4 at min_size(2) loses a child →
  // leave_last_child → root becomes the surviving leaf.
  verifier.assert_node_counts({1, 0, 0, 0, 0});
}

/// Chain under I16(5 children).  Remove chain → I16 shrinks to I4.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CascadeChainUnderI16) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // Chain keys first, then 4 short keys → root I16(5 children).
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x04; ++t)
    verifier.insert(make_short_key(enc, t), val);
  constexpr std::uint64_t IC16 = unodb::test::is_olc_db<TypeParam> ? 2 : 1;
  verifier.assert_node_counts({6, IC16, 1, 0, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.check_present_values();
  // Chain reclaimed.  I16(5) → remove chain slot → is_min_size →
  // shrink to I4(4).
  verifier.assert_node_counts({4, 1, 0, 0, 0});
}

/// Chain under I48(17 children).  Remove chain → I48 shrinks to I16.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CascadeChainUnderI48) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // Chain keys first, then 16 short keys → root I48(17 children).
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x10; ++t)
    verifier.insert(make_short_key(enc, t), val);
  constexpr std::uint64_t IC48 = unodb::test::is_olc_db<TypeParam> ? 2 : 1;
  verifier.assert_node_counts({18, IC48, 0, 1, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.check_present_values();
  // Chain reclaimed.  I48(17) → remove chain slot → shrink to I16(16).
  verifier.assert_node_counts({16, 0, 1, 0, 0});
}

/// Chain under I256(49 children).  Remove chain → I256 shrinks to I48.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, CascadeChainUnderI256) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // Chain keys first, then 48 short keys → root I256(49 children).
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x30; ++t)
    verifier.insert(make_short_key(enc, t), val);
  constexpr std::uint64_t IC256 = unodb::test::is_olc_db<TypeParam> ? 2 : 1;
  verifier.assert_node_counts({50, IC256, 0, 0, 1});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.check_present_values();
  // Chain reclaimed.  I256(49) → remove chain slot → shrink to I48(48).
  verifier.assert_node_counts({48, 0, 0, 1, 0});
}

// Chain under I48(18 children, above min_size).  Remove chain child
// via remove_child_entry (no shrink).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainRemoveFromI48AboveMin) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x11; ++t)  // 17 short keys → I48(18)
    verifier.insert(make_short_key(enc, t), val);

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.check_present_values();
}

// Chain under I256(50 children, above min_size).  Remove chain child
// via remove_child_entry (no shrink).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainRemoveFromI256AboveMin) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x31; ++t)  // 49 short keys → I256(50)
    verifier.insert(make_short_key(enc, t), val);

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.check_present_values();
}

// -------------------------------------------------------------------
// Group 6e: Multi-level chain removal
// -------------------------------------------------------------------

/// Two 17-byte keys (2 chain I4s + bottom I4), remove both.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, MultiLevelChainRemoveAll) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make17 = [&](std::uint8_t last) {
    enc.reset();
    for (unsigned i = 0; i < 16; ++i) enc.encode(std::uint8_t{0xAA});
    enc.encode(last);
    return enc.get_key_view();
  };
  verifier.insert(make17(0x01), val);
  verifier.insert(make17(0x02), val);
  verifier.assert_node_counts({2, 3, 0, 0, 0});

  verifier.remove(make17(0x01));
  // Bottom I4 collapsed.  Two chain I4s remain, last one has 1 child (leaf).
  verifier.assert_node_counts({1, 2, 0, 0, 0});

  verifier.remove(make17(0x02));
  verifier.assert_empty();
}

// -------------------------------------------------------------------
// Group 7: Parent inode growth with chain children (GAP A)
//
// The root inode grows I4→I16→I48→I256 where one child is a chain
// subtree.  Existing cascade tests (6d) only cover parent shrinkage.
// -------------------------------------------------------------------

/// Parent I4→I16 growth with a chain child.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainParentGrowthI4ToI16) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // Chain subtree under tag=0x42.
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  // 4 short keys to grow parent past I4 capacity.
  for (std::uint8_t t = 0x01; t <= 0x04; ++t)
    verifier.insert(make_short_key(enc, t), val);
  verifier.check_present_values();
  constexpr std::uint64_t IG16 = unodb::test::is_olc_db<TypeParam> ? 2 : 1;
  verifier.assert_node_counts({6, IG16, 1, 0, 0});
}

/// Parent I16→I48 growth with a chain child.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainParentGrowthI16ToI48) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x10; ++t)
    verifier.insert(make_short_key(enc, t), val);
  verifier.check_present_values();
  constexpr std::uint64_t IG48 = unodb::test::is_olc_db<TypeParam> ? 2 : 1;
  verifier.assert_node_counts({18, IG48, 0, 1, 0});
}

/// Parent I48→I256 growth with a chain child.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainParentGrowthI48ToI256) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x30; ++t)
    verifier.insert(make_short_key(enc, t), val);
  verifier.check_present_values();
  constexpr std::uint64_t IG256 = unodb::test::is_olc_db<TypeParam> ? 2 : 1;
  verifier.assert_node_counts({50, IG256, 0, 0, 1});
}

// -------------------------------------------------------------------
// Group 8: Bottom inode growth with cumulative stats (GAP C)
// -------------------------------------------------------------------

/// Verify growing_inodes through I4→I16→I48→I256 under a chain.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainBottomGrowthStats) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // Insert 4 keys: chain-I4 + bottom-I4(4).
  for (std::uint64_t i = 1; i <= 4; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({4, 2, 0, 0, 0});
  // chain-I4 creation + bottom-I4 creation = 2 I4 grows.
  verifier.assert_growing_inodes({2, 0, 0, 0});

  // 5th key: bottom I4→I16.
  verifier.insert(make_key(enc, 0x42, 5), val);
  verifier.assert_node_counts({5, 1, 1, 0, 0});
  verifier.assert_growing_inodes({2, 1, 0, 0});

  // 17th key: bottom I16→I48.
  for (std::uint64_t i = 6; i <= 17; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({17, 1, 0, 1, 0});
  verifier.assert_growing_inodes({2, 1, 1, 0});

  // 49th key: bottom I48→I256.
  for (std::uint64_t i = 18; i <= 49; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({49, 1, 0, 0, 1});
  verifier.assert_growing_inodes({2, 1, 1, 1});

  verifier.check_present_values();
}

// -------------------------------------------------------------------
// Group 9: Multi-level chain with fat bottom inode (GAP D)
// -------------------------------------------------------------------

/// Two chain levels + I16 at bottom (5 keys, 17 bytes each).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, MultiLevelChainFatBottom) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make17 = [&](std::uint8_t last) {
    enc.reset();
    for (unsigned i = 0; i < 16; ++i) enc.encode(std::uint8_t{0xAA});
    enc.encode(last);
    return enc.get_key_view();
  };

  for (std::uint8_t i = 1; i <= 5; ++i) verifier.insert(make17(i), val);
  verifier.check_present_values();
  // 2 chain-I4s + bottom I16(5) + 5 leaves
  verifier.assert_node_counts({5, 2, 1, 0, 0});
  verifier.assert_growing_inodes({3, 1, 0, 0});

  // Remove 1 → I16 at min_size → shrink to I4.
  verifier.remove(make17(1));
  verifier.assert_node_counts({4, 3, 0, 0, 0});
  verifier.assert_shrinking_inodes({0, 1, 0, 0});

  // Remove remaining → chains collapse.
  for (std::uint8_t i = 2; i <= 5; ++i) verifier.remove(make17(i));
  verifier.assert_empty();
}

// -------------------------------------------------------------------
// Group 10: Mid-level inode between chain levels (GAP B)
//
// 18-byte keys: encode(u64).encode(u8_mid).encode(u64).encode(u8)
// = 8+1+8+1 = 18 bytes.
// All keys share bytes[0..7] → chain at depth 0.
// byte[8] = mid → mid-level inode at depth 8.
// Within each mid group, bytes[9..16] shared → chain at depth 9.
// byte[17] = bottom → bottom inode at depth 17.
// Structure: chain(0) → mid-inode(8) → {chain(9) → bottom-inode(17)}*
// -------------------------------------------------------------------

/// Mid-level inode growth with chains above and below.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, MidLevelInodeGrowth) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make18 = [&](std::uint8_t mid, std::uint8_t bottom) {
    return enc.reset()
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(mid)
        .encode(std::uint64_t{0})
        .encode(bottom)
        .get_key_view();
  };

  // 5 mid groups × 2 bottom keys = 10 keys.
  // Mid-inode at depth 8 grows to I16(5).
  for (std::uint8_t m = 1; m <= 5; ++m)
    for (std::uint8_t b = 1; b <= 2; ++b) verifier.insert(make18(m, b), val);
  verifier.check_present_values();
  // chain(0) → mid-I16(5) → 5×(chain(9) → bottom-I4(2) → 2 leaves)
  // I4: 1 top-chain + 5 depth-9-chains + 5 bottoms = 11
  verifier.assert_node_counts({10, 11, 1, 0, 0});
}

/// Mid-level inode shrinkage with chains above and below.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, MidLevelInodeShrink) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make18 = [&](std::uint8_t mid, std::uint8_t bottom) {
    return enc.reset()
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(mid)
        .encode(std::uint64_t{0})
        .encode(bottom)
        .get_key_view();
  };

  // 5 mid groups × 2 bottom keys = 10 keys → mid I16(5).
  for (std::uint8_t m = 1; m <= 5; ++m)
    for (std::uint8_t b = 1; b <= 2; ++b) verifier.insert(make18(m, b), val);

  // Remove both keys for mid=1 → bottom-I4(2) collapses (I4 shrink),
  // chain(9) collapses (I4 shrink), mid-I16(5) at min_size → I4(4)
  // (I16 shrink).
  verifier.remove(make18(1, 1));
  verifier.remove(make18(1, 2));
  verifier.check_present_values();
  // chain(0) → mid-I4(4) → 4×(chain(9) → bottom-I4(2) → 2 leaves)
  // I4: 1 top-chain + 4 depth-9-chains + 4 bottoms + 1 mid = 10
  verifier.assert_node_counts({8, 10, 0, 0, 0});
  verifier.assert_shrinking_inodes({2, 1, 0, 0});
}

// -------------------------------------------------------------------
// Group 10a: Chain remove — key not found / mismatch coverage
// -------------------------------------------------------------------

// Remove non-existent key where chain I4's find_child returns nullptr.
// The chain has one child at dispatch byte 0x01; we try dispatch 0x03.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainRemoveKeyNotFound) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.remove(make_key(enc, 0x42, 2));
  // Chain I4 has 1 child at dispatch 0x01.  v=3 → dispatch 0x03.
  UNODB_ASSERT_FALSE(verifier.get_db().remove(make_key(enc, 0x42, 3)));
  verifier.check_present_values();
}

// Remove non-existent key when root is a single leaf.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, RemoveMissRootIsLeaf) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  UNODB_ASSERT_FALSE(verifier.get_db().remove(make_key(enc, 0x42, 2)));
  verifier.check_present_values();
}

// Remove key where chain I4's child is a leaf that doesn't match.
// Use a 10-byte key that shares the chain prefix and dispatch byte
// but differs at the leaf level.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainRemoveLeafMismatch) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.remove(make_key(enc, 0x42, 2));
  // Chain I4 child is leaf with key [0x42, 0,...,0x01] (9 bytes).
  // Try removing a 10-byte key with same first 9 bytes + suffix.
  // This matches prefix and dispatch but leaf->matches() fails.
  UNODB_ASSERT_FALSE(
      verifier.get_db().remove(make_long_key(enc, 0x42, 1, 0xFF)));
  verifier.check_present_values();
}

// -------------------------------------------------------------------
// Group 10b: Atomic chain cut — structural gap tests
// -------------------------------------------------------------------

// T5: I4(2) collapse with CD=1 chain, remaining child is leaf.
// Tree: root-I4(2: chain→chain→leaf(A), leaf(C))
// Remove A → chain cut, I4 collapses, root becomes leaf(C).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainCutCD1CollapseToLeaf) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make18 = [&](std::uint8_t tag, std::uint8_t mid, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(mid)
        .encode(bottom)
        .get_key_view();
  };

  // A and B share tag + 8 bytes → chain at depth 0, chain at depth 8.
  // C has different tag → sibling at root.
  verifier.insert(make18(0x01, 0x00, 0x01), val);  // A
  verifier.insert(make18(0x01, 0x00, 0x02), val);  // B
  verifier.insert(make18(0x02, 0x00, 0x01), val);  // C
  verifier.check_present_values();

  // Remove B: bottom I4(2→1), chain extends.
  verifier.remove(make18(0x01, 0x00, 0x02));
  verifier.check_present_values();

  // Remove A: chain cut (CD=1). Root I4(2→1) collapses to leaf(C).
  verifier.remove(make18(0x01, 0x00, 0x01));
  verifier.check_present_values();

  // Remove C: tree empty.
  verifier.remove(make18(0x02, 0x00, 0x01));
  verifier.assert_empty();
}

// T7: I4(2) collapse with CD=0 chain, remaining child is inode.
// Remove A → chain cut, I4 collapses, remaining child promoted.
// Use 1-byte keys for B,C so remaining child has short prefix.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainCutCD0CollapseToInode) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // A: 9-byte key with tag=0x10 → chain at depth 0.
  // B,C: 1-byte keys with tag=0x20, 0x30 → I4(2) at root, no prefix.
  // Root: I4(3: 0x10→chain→leaf(A), 0x20→leaf(B), 0x30→leaf(C))
  // Wait — that's I4(3), not I4(2). For I4(2) collapse we need
  // exactly 2 children. Use B as a subtree:
  // B1,B2: 1-byte keys → but 1-byte keys can't form a subtree.
  //
  // Simpler: use 2-byte keys for B,C sharing first byte.
  // B: [0x20, 0x01], C: [0x20, 0x02] → I4(2: leaf(B), leaf(C))
  // under dispatch 0x20. Root: I4(2: 0x10→chain, 0x20→I4(2: B, C))
  // Root prefix is empty. Remaining child I4(B,C) has empty prefix.
  // Merge: 0 + 1 + 0 = 1 byte → fits.
  verifier.insert(make_key(enc, 0x10, 1), val);  // A (9 bytes)
  verifier.insert(enc.reset()
                      .encode(std::uint8_t{0x20})
                      .encode(std::uint8_t{0x01})
                      .get_key_view(),
                  val);  // B (2 bytes)
  verifier.insert(enc.reset()
                      .encode(std::uint8_t{0x20})
                      .encode(std::uint8_t{0x02})
                      .get_key_view(),
                  val);  // C (2 bytes)
  verifier.check_present_values();

  // Remove A: chain cut (CD=0). Root I4(2→1) collapses.
  verifier.remove(make_key(enc, 0x10, 1));
  verifier.check_present_values();
}

// T8: I4(2) collapse with CD=1 chain, remaining child is inode.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainCutCD1CollapseToInode) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make18 = [&](std::uint8_t tag, std::uint8_t mid, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(mid)
        .encode(bottom)
        .get_key_view();
  };

  // A,B share tag=0x01 + 8 bytes → depth-1 chain → I4(2: A, B).
  // D,E have tag=0x02 → I4(2: D, E) subtree.
  // Root: I4(2: 0x01→chain→chain→I4(A,B), 0x02→chain→I4(D,E))
  verifier.insert(make18(0x01, 0x00, 0x01), val);  // A
  verifier.insert(make18(0x01, 0x00, 0x02), val);  // B
  verifier.insert(make18(0x02, 0x00, 0x01), val);  // D
  verifier.insert(make18(0x02, 0x00, 0x02), val);  // E
  verifier.check_present_values();

  // Remove B: bottom I4(2→1) under tag=0x01, chain extends.
  verifier.remove(make18(0x01, 0x00, 0x02));
  verifier.check_present_values();

  // Remove A: chain cut (CD=1). Root I4(2→1) collapses.
  // Remaining child is the tag=0x02 subtree — prefix merge needed.
  verifier.remove(make18(0x01, 0x00, 0x01));
  verifier.check_present_values();
}

// T12: I4(3) with CD=1 chain, just remove entry.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainCutCD1RemoveFromI4) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make18 = [&](std::uint8_t tag, std::uint8_t mid, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(mid)
        .encode(bottom)
        .get_key_view();
  };

  // 3 tag groups: 0x01 (chain target), 0x02, 0x03.
  // Root: I4(3: chain→...A, leaf(D), leaf(E))
  verifier.insert(make18(0x01, 0x00, 0x01), val);  // A
  verifier.insert(make18(0x01, 0x00, 0x02), val);  // B
  verifier.insert(make18(0x02, 0x00, 0x01), val);  // D
  verifier.insert(make18(0x03, 0x00, 0x01), val);  // E
  verifier.check_present_values();

  // Remove B, then A: chain cut from I4(3→2).
  verifier.remove(make18(0x01, 0x00, 0x02));
  verifier.remove(make18(0x01, 0x00, 0x01));
  verifier.check_present_values();
}

// T8b: I4(2) collapse with CD=1 chain, remaining child is inode,
// prefix merge fits.  Uses 2-byte sibling keys so the remaining
// child has empty prefix → merge = 0 + 1 + 0 = 1 ≤ 7.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest,
                 ChainCutCD1CollapseToInodeShortPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make18 = [&](std::uint8_t tag, std::uint8_t mid, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(mid)
        .encode(bottom)
        .get_key_view();
  };

  // A,B: tag=0x01, 18-byte keys → depth-1 chain → I4(2: A, B).
  verifier.insert(make18(0x01, 0x00, 0x01), val);  // A
  verifier.insert(make18(0x01, 0x00, 0x02), val);  // B
  // C,D: tag=0x02, 2-byte keys → I4(2: C, D) with empty prefix.
  verifier.insert(enc.reset()
                      .encode(std::uint8_t{0x02})
                      .encode(std::uint8_t{0x01})
                      .get_key_view(),
                  val);  // C
  verifier.insert(enc.reset()
                      .encode(std::uint8_t{0x02})
                      .encode(std::uint8_t{0x02})
                      .get_key_view(),
                  val);  // D
  verifier.check_present_values();

  // Remove B: bottom I4(2→1) under tag=0x01, chain extends.
  verifier.remove(make18(0x01, 0x00, 0x02));
  verifier.check_present_values();

  // Remove A: chain cut (CD=1). Root I4(2→1) collapses.
  // Remaining child is I4(C,D) with empty prefix.
  // Merge: root prefix(0) + dispatch(0x02) + child prefix(0) = 1 ≤ 7.
  verifier.remove(make18(0x01, 0x00, 0x01));
  verifier.check_present_values();
}

// T6: I4(2) collapse with CD=2 chain, remaining child is leaf.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainCutCD2CollapseToLeaf) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // 26-byte keys: tag + uint64 + uint64 + uint64 + uint8.
  // Two keys sharing 25 bytes → 3 chain levels (depth 0,8,16).
  auto make26 = [&](std::uint8_t tag, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(std::uint64_t{0})
        .encode(bottom)
        .get_key_view();
  };

  verifier.insert(make26(0x01, 0x01), val);  // A
  verifier.insert(make26(0x01, 0x02), val);  // B
  verifier.insert(make26(0x02, 0x01), val);  // sibling
  verifier.check_present_values();

  // Remove B → chain extends. Remove A → CD=2 chain cut.
  verifier.remove(make26(0x01, 0x02));
  verifier.check_present_values();
  verifier.remove(make26(0x01, 0x01));
  verifier.check_present_values();

  // Sibling subtree remains: chain(depth 0)→chain(depth 8)→chain(depth 16)
  //   →I4(2: leaf(sib_01), leaf(sib_02))... but we only inserted one sibling.
  // Actually just: root single-child I4 → chain → chain → leaf(sib).
  // Prefix overflow prevents collapse at each level.

  // Remove sibling → empty.
  verifier.remove(make26(0x02, 0x01));
  verifier.assert_empty();
}

// T9: I4(2) collapse with CD=0, prefix merge overflows.
// The I4 should NOT collapse — remains as single-child I4.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainCutCD0PrefixOverflow) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // Both children are 9-byte keys with different tags.
  // Root I4 has empty prefix. Each child chain has 7-byte prefix.
  // Collapse would need 0 + 1 + 7 = 8 bytes → overflow.
  verifier.insert(make_key(enc, 0x10, 1), val);  // A (chain→leaf)
  verifier.insert(make_key(enc, 0x10, 2), val);  // B (chain→I4(A,B))
  verifier.insert(make_key(enc, 0x20, 1), val);  // C (chain→leaf)
  verifier.insert(make_key(enc, 0x20, 2), val);  // D (chain→I4(C,D))
  verifier.check_present_values();

  // Remove B → chain under 0x10 extends to leaf(A).
  verifier.remove(make_key(enc, 0x10, 2));
  verifier.check_present_values();

  // Remove A → chain cut. Root I4(2→1). Remaining child is 0x20 chain
  // with 7-byte prefix. Prefix merge overflows → no collapse.
  verifier.remove(make_key(enc, 0x10, 1));
  verifier.check_present_values();

  // C and D still accessible.
  verifier.remove(make_key(enc, 0x20, 1));
  verifier.remove(make_key(enc, 0x20, 2));
  verifier.assert_empty();
}

// T10: I4(2) collapse with CD=1, prefix merge overflows.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainCutCD1PrefixOverflow) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make18 = [&](std::uint8_t tag, std::uint8_t mid, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(mid)
        .encode(bottom)
        .get_key_view();
  };

  // tag=0x01: 2 keys sharing 10 bytes → CD=1 chain.
  // tag=0x02: 2 keys → chain with 7-byte prefix.
  // Root I4 has empty prefix. 0x02 child has 7-byte prefix.
  // Collapse: 0 + 1 + 7 = 8 → overflow.
  verifier.insert(make18(0x01, 0x00, 0x01), val);
  verifier.insert(make18(0x01, 0x00, 0x02), val);
  verifier.insert(make18(0x02, 0x00, 0x01), val);
  verifier.insert(make18(0x02, 0x00, 0x02), val);
  verifier.check_present_values();

  // Remove both under tag=0x01 → CD=1 chain cut.
  verifier.remove(make18(0x01, 0x00, 0x02));
  verifier.remove(make18(0x01, 0x00, 0x01));
  verifier.check_present_values();

  // tag=0x02 keys still accessible.
  verifier.remove(make18(0x02, 0x00, 0x01));
  verifier.remove(make18(0x02, 0x00, 0x02));
  verifier.assert_empty();
}

// T14: I16(min) shrink with CD=1 chain.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainCutCD1ShrinkI16) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make18 = [&](std::uint8_t tag, std::uint8_t mid, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(mid)
        .encode(bottom)
        .get_key_view();
  };

  // 5 tag groups → I16(5) at root level (after chain at depth 0).
  // Under tag=0x01: 2 keys → CD=1 chain.
  for (std::uint8_t t = 1; t <= 5; ++t)
    for (std::uint8_t b = 1; b <= 2; ++b)
      verifier.insert(make18(t, 0x00, b), val);
  verifier.check_present_values();

  // Remove both under tag=0x01 → CD=1 chain cut from I16(5→4) → I4.
  verifier.remove(make18(0x01, 0x00, 0x02));
  verifier.remove(make18(0x01, 0x00, 0x01));
  verifier.check_present_values();
  // 8 leaves, 4 tag groups each with chain(depth 0)→chain(depth 8)→I4(2).
  // Top-level I4(4) + 4×(chain + bottom-I4) = 1 + 8 = 9 I4s.
  verifier.assert_node_counts({8, 9, 0, 0, 0});
  verifier.assert_shrinking_inodes({2, 1, 0, 0});
}

// T16: I48(min) shrink with CD=1 chain.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainCutCD1ShrinkI48) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make18 = [&](std::uint8_t tag, std::uint8_t mid, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(mid)
        .encode(bottom)
        .get_key_view();
  };

  // 17 tag groups → I48(17).
  for (std::uint8_t t = 1; t <= 17; ++t)
    for (std::uint8_t b = 1; b <= 2; ++b)
      verifier.insert(make18(t, 0x00, b), val);
  verifier.check_present_values();

  // Remove both under tag=0x01 → CD=1 chain cut from I48(17→16) → I16.
  verifier.remove(make18(0x01, 0x00, 0x02));
  verifier.remove(make18(0x01, 0x00, 0x01));
  verifier.check_present_values();
  // 32 leaves, 16 tag groups each with chain→chain→I4(2).
  // Top-level I16(16) + 16×(chain + bottom-I4) = 0 + 32 = 32 I4s.
  verifier.assert_node_counts({32, 32, 1, 0, 0});
  verifier.assert_shrinking_inodes({2, 0, 1, 0});
}

// T18: I256(min) shrink with CD=1 chain.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ChainCutCD1ShrinkI256) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make18 = [&](std::uint8_t tag, std::uint8_t mid, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(mid)
        .encode(bottom)
        .get_key_view();
  };

  // 49 tag groups → I256(49).
  for (std::uint8_t t = 1; t <= 49; ++t)
    for (std::uint8_t b = 1; b <= 2; ++b)
      verifier.insert(make18(t, 0x00, b), val);
  verifier.check_present_values();

  // Remove both under tag=0x01 → CD=1 chain cut from I256(49→48) → I48.
  verifier.remove(make18(0x01, 0x00, 0x02));
  verifier.remove(make18(0x01, 0x00, 0x01));
  verifier.check_present_values();
  // 96 leaves, 48 tag groups each with chain→chain→I4(2).
  // Top-level I48(48) + 48×(chain + bottom-I4) = 0 + 96 = 96 I4s.
  verifier.assert_node_counts({96, 96, 0, 1, 0});
  verifier.assert_shrinking_inodes({2, 0, 0, 1});
}

// -------------------------------------------------------------------
// Verify tree structures used by concurrent chain cut tests (CT1-CT4).
// These confirm the chain depth and that insert/remove work correctly
// on these key patterns before we add concurrency.

// CT1/CT3 tree: 26-byte keys, 3 chain levels after removing B.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ConcurrentTestTree26Byte) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  auto make26 = [&](std::uint8_t tag, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(std::uint64_t{0})
        .encode(std::uint64_t{0})
        .encode(bottom)
        .get_key_view();
  };

  verifier.insert(make26(0x10, 0x01), val);  // A
  verifier.insert(make26(0x10, 0x02), val);  // B
  verifier.insert(make26(0x20, 0x01), val);  // sib
  verifier.check_present_values();

  verifier.remove(make26(0x10, 0x02));  // remove B
  verifier.check_present_values();

  verifier.remove(make26(0x10, 0x01));  // remove A (chain cut)
  verifier.check_present_values();

  // Insert a new key at root level (what CT1's T2 does).
  verifier.insert(make26(0x30, 0x01), val);
  verifier.check_present_values();
}

// CT2/CT4 tree: 34-byte keys, 4 chain levels after removing B.
// Also verify that inserting a key diverging at chain[0] works.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ConcurrentTestTree34Byte) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  constexpr auto X = std::uint64_t{0x4242424242424242ULL};
  constexpr auto Z = std::uint64_t{0x4343434343434343ULL};

  auto make34 = [&](std::uint8_t tag, std::uint64_t v1, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(v1)
        .encode(X)
        .encode(X)
        .encode(X)
        .encode(bottom)
        .get_key_view();
  };

  verifier.insert(make34(0x10, X, 0x01), val);  // A
  verifier.insert(make34(0x10, X, 0x02), val);  // B
  verifier.insert(make34(0x20, X, 0x01), val);  // sib
  verifier.check_present_values();

  verifier.remove(make34(0x10, X, 0x02));  // remove B
  verifier.check_present_values();

  // Insert T2's key (diverges at chain[0] level, different v1).
  verifier.insert(make34(0x10, Z, 0x01), val);
  verifier.check_present_values();

  // Remove A (chain cut with T2's key already in tree).
  verifier.remove(make34(0x10, X, 0x01));
  verifier.check_present_values();
}

// Group 11: Scan through chain with mixed-length keys (GAP E)
// -------------------------------------------------------------------

/// Iterator walks through chain subtree with 9-byte and 10-byte keys.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ScanChainMixedLengths) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::test_values[0];

  // 9-byte and 10-byte keys in the same chain subtree.
  // make_key(0x42, v) = 9 bytes.  make_long_key(0x42, v, s) = 10 bytes.
  // Keys must not be in a prefix relationship — use different v values.
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_long_key(enc, 0x42, 3, 0x01), val);
  verifier.insert(make_long_key(enc, 0x42, 4, 0x01), val);
  // check_present_values does a full scan + per-key probe.
  verifier.check_present_values();
  verifier.assert_node_counts({4, 2, 0, 0, 0});

  // Remove a 10-byte key, verify scan still works.
  verifier.remove(make_long_key(enc, 0x42, 3, 0x01));
  verifier.check_present_values();

  // Remove a 9-byte key.
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
}

#endif  // UNODB_DETAIL_WITH_STATS

<<<<<<< HEAD
// Regression test: basic_art_key<key_view>::cmp() must compare actual
// key data, not the raw std::span struct bytes (pointer + size).
// The bug caused scan_range to pick the wrong direction when key
// buffers were at addresses that disagreed with key data ordering.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ScanRangeReversedPointerOrder) {
  unodb::test::tree_verifier<TypeParam> verifier;
  const auto val = unodb::test::test_values[0];

  // Insert 3 keys: 0x01, 0x02, 0x03.
  std::array<std::byte, 1> buf_a{std::byte{0x01}};
  std::array<std::byte, 1> buf_b{std::byte{0x02}};
  std::array<std::byte, 1> buf_c{std::byte{0x03}};
  const auto ka = unodb::key_view{buf_a.data(), 1};
  const auto kb = unodb::key_view{buf_b.data(), 1};
  const auto kc = unodb::key_view{buf_c.data(), 1};
  verifier.insert(ka, val);
  verifier.insert(kb, val);
  verifier.insert(kc, val);

  // Construct from/to keys in separate buffers where the "larger" key
  // data (0x03) is at a LOWER address than the "smaller" key (0x01).
  // This triggers the bug: cmp() compared pointer values, not key data.
  std::array<std::byte, 256> mem{};
  mem[0] = std::byte{0x03};    // larger key at lower address
  mem[128] = std::byte{0x01};  // smaller key at higher address
  const auto mem_span = std::span{mem};
  const auto from_key = unodb::key_view{mem_span.subspan(128, 1)};  // 0x01
  const auto to_key = unodb::key_view{mem_span.subspan(0, 1)};      // 0x03

  // scan_range(0x01, 0x03) should visit 0x01 and 0x02 (forward scan,
  // exclusive upper bound).
  std::vector<std::byte> visited;
  verifier.get_db().scan_range(from_key, to_key,
                               [&visited](const auto& visitor) {
                                 const auto k = visitor.get_key();
                                 UNODB_DETAIL_ASSERT(k.size() == 1);
                                 visited.push_back(k[0]);
                                 return false;  // continue
                               });
  UNODB_ASSERT_EQ(visited.size(), 2U);
  UNODB_EXPECT_EQ(visited[0], std::byte{0x01});
  UNODB_EXPECT_EQ(visited[1], std::byte{0x02});
}

// -------------------------------------------------------------------
// Zero-length key
// -------------------------------------------------------------------

/// A zero-length key_view should work as a root leaf (no prefix bytes
/// to encode in an inode chain).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, ZeroLengthKey) {
  unodb::test::tree_verifier<TypeParam> verifier;
  const auto empty_key = unodb::key_view{};
  constexpr auto val = unodb::test::test_values[0];

  verifier.insert(empty_key, val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({1, 0, 0, 0, 0});
#endif

  verifier.remove(empty_key);
  verifier.assert_empty();
#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({0, 0, 0, 0, 0});
#endif
}

// -------------------------------------------------------------------
// build_chain correctness tests (no stats assertions).
//
// Verify that the first key_view insert creates a correct inode chain
// for various key lengths.  key_prefix_capacity = 7, so each chain I4
// consumes up to 7 prefix + 1 dispatch = 8 bytes.
//
// Key lengths tested: 1, 7, 8, 9, 15, 16, 17, 18
// (0-byte key already tested in ZeroLengthKey above)
// -------------------------------------------------------------------

namespace {

/// Build a raw key of \a len bytes.  All bytes are 0x42 except the
/// last byte which is \a last.  \a buf must have at least \a len bytes.
inline unodb::key_view make_raw_key(std::byte* buf, std::size_t len,
                                    std::byte last = std::byte{0x01}) {
  std::fill_n(buf, len, std::byte{0x42});
  if (len > 0) buf[len - 1] = last;
  return unodb::key_view{buf, len};
}

}  // namespace

// T1: Single key insert/get/remove for each key length.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, BuildChainSingleKey) {
  constexpr auto val = unodb::test::test_values[0];
  constexpr std::size_t lengths[] = {1, 7, 8, 9, 15, 16, 17, 18};
  for (auto len : lengths) {
    unodb::test::tree_verifier<TypeParam> verifier;
    std::array<std::byte, 32> buf{};
    const auto k = make_raw_key(buf.data(), len);
    verifier.insert(k, val);
    verifier.check_present_values();
    // Duplicate insert must fail.
    UNODB_ASSERT_FALSE(verifier.get_db().insert(k, val));
    verifier.remove(k);
    verifier.assert_empty();
  }
}

// T2: Two keys diverging at the last byte.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, BuildChainTwoKeysDivergeAtEnd) {
  constexpr auto val = unodb::test::test_values[0];
  constexpr std::size_t lengths[] = {1, 8, 9, 16, 17, 18};
  for (auto len : lengths) {
    unodb::test::tree_verifier<TypeParam> verifier;
    std::array<std::byte, 32> buf_a{};
    std::array<std::byte, 32> buf_b{};
    const auto ka = make_raw_key(buf_a.data(), len, std::byte{0x01});
    const auto kb = make_raw_key(buf_b.data(), len, std::byte{0x02});
    verifier.insert(ka, val);
    verifier.insert(kb, val);
    verifier.check_present_values();
    verifier.remove(ka);
    verifier.check_present_values();
    verifier.remove(kb);
    verifier.assert_empty();
  }
}

// T3: Two keys diverging at an intermediate byte (byte 0).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, BuildChainDivergeAtStart) {
  constexpr auto val = unodb::test::test_values[0];
  constexpr std::size_t lengths[] = {9, 17, 18};
  for (auto len : lengths) {
    unodb::test::tree_verifier<TypeParam> verifier;
    std::array<std::byte, 32> buf_a{};
    std::array<std::byte, 32> buf_b{};
    const auto ka = make_raw_key(buf_a.data(), len, std::byte{0x01});
    // Diverge at byte 0.
    std::fill_n(buf_b.data(), len, std::byte{0x42});
    buf_b[0] = std::byte{0x10};
    buf_b[len - 1] = std::byte{0x01};
    const auto kb = unodb::key_view{buf_b.data(), len};
    verifier.insert(ka, val);
    verifier.insert(kb, val);
    verifier.check_present_values();
    verifier.remove(ka);
    verifier.remove(kb);
    verifier.assert_empty();
  }
}

// T4: Scan over chain keys.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, BuildChainScan) {
  unodb::test::tree_verifier<TypeParam> verifier;
  constexpr auto val = unodb::test::test_values[0];
  std::array<std::byte, 9> buf1{}, buf2{}, buf3{};
  const auto k1 = make_raw_key(buf1.data(), 9, std::byte{0x01});
  const auto k2 = make_raw_key(buf2.data(), 9, std::byte{0x02});
  const auto k3 = make_raw_key(buf3.data(), 9, std::byte{0x03});
  verifier.insert(k1, val);
  verifier.insert(k2, val);
  verifier.insert(k3, val);

  // Forward scan — collect keys.
  std::vector<std::vector<std::byte>> keys;
  verifier.get_db().scan([&keys](auto visitor) {
    auto kv = visitor.get_key();
    keys.emplace_back(kv.begin(), kv.end());
    return false;  // continue
  });
  UNODB_ASSERT_EQ(keys.size(), 3U);
  // Keys should be in lexicographic order.
  UNODB_ASSERT_TRUE(keys[0] < keys[1]);
  UNODB_ASSERT_TRUE(keys[1] < keys[2]);
}

// T5: Chain + non-chain sibling (different first byte).
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, BuildChainWithSibling) {
  unodb::test::tree_verifier<TypeParam> verifier;
  constexpr auto val = unodb::test::test_values[0];
  // 9-byte chain key.
  std::array<std::byte, 9> buf_chain{};
  const auto k_chain = make_raw_key(buf_chain.data(), 9, std::byte{0x01});
  // 8-byte key with different first byte.
  std::array<std::byte, 8> buf_short{};
  std::fill(buf_short.begin(), buf_short.end(), std::byte{0x10});
  const auto k_short = unodb::key_view{buf_short.data(), 8};
  verifier.insert(k_chain, val);
  verifier.insert(k_short, val);
  verifier.check_present_values();
  verifier.remove(k_chain);
  verifier.check_present_values();
  verifier.remove(k_short);
  verifier.assert_empty();
}

// T6: Prefix overflow on I4 collapse after chain removal.
UNODB_TYPED_TEST(ARTKeyViewCorrectnessTest, BuildChainPrefixOverflow) {
  unodb::test::tree_verifier<TypeParam> verifier;
  constexpr auto val = unodb::test::test_values[0];
  unodb::key_encoder enc;
  // Two 9-byte keys under tag=0x10.
  verifier.insert(make_key(enc, 0x10, 1), val);
  verifier.insert(make_key(enc, 0x10, 2), val);
  // Two 9-byte keys under tag=0x20.
  verifier.insert(make_key(enc, 0x20, 1), val);
  verifier.insert(make_key(enc, 0x20, 2), val);
  verifier.check_present_values();
  // Remove both tag=0x10 keys.
  verifier.remove(make_key(enc, 0x10, 1));
  verifier.remove(make_key(enc, 0x10, 2));
  // tag=0x20 keys must still be accessible.
  verifier.check_present_values();
  // Remove remaining.
  verifier.remove(make_key(enc, 0x20, 1));
  verifier.remove(make_key(enc, 0x20, 2));
  verifier.assert_empty();
}

}  // namespace
