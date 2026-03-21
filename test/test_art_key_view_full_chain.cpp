// Copyright 2025-2026 UnoDB contributors

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

// IWYU pragma: no_include <__cstddef/byte.h>
// IWYU pragma: no_include <span>
// IWYU pragma: no_include <string>
// IWYU pragma: no_include <string_view>

#include <algorithm>
#include <array>
#include <cstddef>  // IWYU pragma: keep
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "art_common.hpp"
#include "db_test_utils.hpp"
#include "gtest_utils.hpp"
#include "node_type.hpp"

namespace {

template <class Db>
class ARTKeyViewFullChainTest : public ::testing::Test {
 public:
  using Test::Test;

  static constexpr std::uint64_t LC(std::uint64_t n) {
    if constexpr (!std::is_same_v<typename Db::value_type, unodb::value_view>)
      return 0;
    else
      return n;
  }
};

using ARTTypes = ::testing::Types<unodb::test::key_view_u64val_db,
                                  unodb::test::key_view_u64val_mutex_db,
                                  unodb::test::key_view_u64val_olc_db>;

UNODB_TYPED_TEST_SUITE(ARTKeyViewFullChainTest, ARTTypes)

/// Unit test of correct rejection of a key which is too large to be
/// stored in the tree.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, TooLongKey) {
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

/// Minimal reproducer: two text keys into a keyless-leaf tree.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, TwoKeyMinimalRepro) {
  TypeParam db;
  unodb::key_encoder enc;

  auto k1 = enc.reset().encode_text("").get_key_view();
  UNODB_ASSERT_TRUE(db.insert(k1, 100));

  auto k2 = enc.reset().encode_text("a").get_key_view();
  UNODB_ASSERT_TRUE(db.insert(k2, 200));
}

/// Unit test inserts several string keys with proper encoding and
/// validates the tree.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, EncodedTextKeys) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);
  verifier.insert(enc.reset().encode_text("").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("a").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("abba").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("banana").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("camel").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("yellow").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("ostritch").get_key_view(), val);
  verifier.insert(enc.reset().encode_text("zebra").get_key_view(), val);
  verifier.check_present_values();
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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CompoundKeysLongSharedPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 (prefix=7, 1 child) + bottom I4 (2 children) + 2 leaves
  verifier.assert_node_counts({TestFixture::LC(2), 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Three keys with the same tag byte and small uint64 values.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ThreeCompoundKeysLongSharedPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_key(enc, 0x42, 3), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + bottom I4 (3 children) + 3 leaves
  verifier.assert_node_counts({TestFixture::LC(3), 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// 9-byte keys sharing 8 bytes — minimal collision case.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, NineByteCompoundKeysLongPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({TestFixture::LC(2), 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

// -------------------------------------------------------------------
// Group 1: Prefix boundary cases
// -------------------------------------------------------------------

/// Keys identical except last byte — maximum chaining depth for 9-byte keys.
/// 9-byte keys sharing 8 bytes, differing only at byte 8.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CompoundKeysIdenticalExceptLastByte) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_key(enc, 0x42, 3), val);
  verifier.insert(make_key(enc, 0x42, 4), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + bottom I4 (4 children, at capacity) + 4 leaves
  verifier.assert_node_counts({TestFixture::LC(4), 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Two 17-byte keys sharing 16 bytes — forces two consecutive chain
/// nodes (depth 0→8 and depth 8→16) before the normal 2-child split.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, MultiLevelChain) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
  verifier.assert_node_counts({TestFixture::LC(2), 3, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Three 17-byte keys: A and B share 16 bytes, C diverges at byte 10.
/// After inserting A and B (two chain levels), inserting C splits the
/// second chain node mid-prefix.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest,
                 InsertDivergingAtIntermediateChainDepth) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
  // + bottom I4 (A,B) + chain I4 for C's suffix + 3 leaves
  verifier.assert_node_counts({TestFixture::LC(3), 5, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

// -------------------------------------------------------------------
// Group 2: Node growth (inode4 -> inode16 -> inode48 -> inode256)
// -------------------------------------------------------------------

/// 5 keys with same 8-byte prefix — forces inode4 -> inode16 growth.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CompoundKeysFiveChildren) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  for (std::uint64_t i = 1; i <= 5; ++i) {
    verifier.insert(make_key(enc, 0x42, i), val);
  }
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + I16 (5 children) + 5 leaves
  verifier.assert_node_counts({TestFixture::LC(5), 1, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// 17 keys — forces inode16 -> inode48 growth.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CompoundKeysSeventeenChildren) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  for (std::uint64_t i = 1; i <= 17; ++i) {
    verifier.insert(make_key(enc, 0x42, i), val);
  }
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + I48 (17 children) + 17 leaves
  verifier.assert_node_counts({TestFixture::LC(17), 1, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// 50 keys — forces inode48 -> inode256 growth.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CompoundKeysFiftyChildren) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  for (std::uint64_t i = 1; i <= 50; ++i) {
    verifier.insert(make_key(enc, 0x42, i), val);
  }
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + I256 (50 children) + 50 leaves
  verifier.assert_node_counts({TestFixture::LC(50), 1, 0, 0, 1});
#endif  // UNODB_DETAIL_WITH_STATS
}

// -------------------------------------------------------------------
// Group 3: Removal & shrinkage
// -------------------------------------------------------------------

// Group 3a: Chain collapse scenarios

/// Insert 2 colliding keys, remove one, verify the other is still found.
/// The chain of inode_4s should collapse to a single leaf.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CompoundKeysInsertThenRemove) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // For keyless leaves, the chain I4 above the leaf is NOT collapsed
  // (collapsing would lose key bytes from the inode path).
  verifier.assert_node_counts({TestFixture::LC(1), 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Insert 3 keys: two sharing 8 bytes, one diverging earlier.
/// Remove one of the 8-byte-shared pair.  The surviving structure
/// should be an inode with 2 children (the remaining keys).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, RemoveFromChainLeavesInode) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  // Keys 1 and 2 share 8 bytes; key 3 diverges at byte 5.
  // key3 uint64 = 0x0000000100000000 differs at byte 5 (overall).
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_key(enc, 0x42, 0x0000000100000000ULL), val);
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // Root I4 (2 children: key3 chain + key2 chain) + chain I4s for
  // each key's suffix + 2 leaves.  Extra I4 because keyless leaf
  // prevents collapse of the chain node above key2's leaf.
  verifier.assert_node_counts({TestFixture::LC(2), 4, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Insert 3 colliding keys, remove in reverse order, assert empty.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, RemoveAllFromChainReverseOrder) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_key(enc, 0x42, 3), val);
  verifier.remove(make_key(enc, 0x42, 3));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.assert_empty();
}

/// Insert 3 colliding keys, remove in forward order, assert empty.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, RemoveAllFromChainForwardOrder) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ShrinkInode16InChain) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  for (std::uint64_t i = 1; i <= 5; ++i) {
    verifier.insert(make_key(enc, 0x42, i), val);
  }
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.remove(make_key(enc, 0x42, 3));
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // I16(5) shrinks to I4(4) on first remove, then 2 more removes → I4(2).
  verifier.assert_node_counts({TestFixture::LC(2), 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Insert 17 keys (-> inode48), remove 13 (-> shrink to inode4).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ShrinkInode48InChain) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
  verifier.assert_node_counts({TestFixture::LC(4), 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Insert 5 keys (-> inode16), remove all 5, assert empty.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ShrinkToEmptyFromInode16InChain) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, RemoveMixedLengthFromChain) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_long_key(enc, 0x42, 1, 0xFF), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.remove(make_long_key(enc, 0x42, 1, 0xFF));
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // Chain I4 + 1 surviving leaf.  Extra I4 for keyless leaf no-collapse.
  verifier.assert_node_counts({TestFixture::LC(1), 2, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.assert_empty();
}

// Group 3d: Stress removal

/// Insert 24 keys (divergence at positions 7..18), remove every other
/// key, verify remaining.  Then remove all, assert empty.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, StressInsertRemoveAtEveryPosition) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, MixedLengthKeysLongPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_long_key(enc, 0x42, 1, 0xFF), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.check_present_values();
#ifdef UNODB_DETAIL_WITH_STATS
  // chain I4 + divergence I4 + chain I4 for shorter key's suffix + 2 leaves
  verifier.assert_node_counts({TestFixture::LC(2), 3, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

// -------------------------------------------------------------------
// Group 5: Duplicate & edge cases
// -------------------------------------------------------------------

/// Inserting the same 9-byte key twice returns false on the second insert.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CompoundKeyDuplicateInsert) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  // Second insert of same key should fail.
  UNODB_ASSERT_FALSE(verifier.get_db().insert(make_key(enc, 0x42, 1), val));
  verifier.check_present_values();
}

/// Get with a key sharing 8 bytes but differing at the last byte
/// should return empty when only one key is present.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CompoundKeyGetMissing) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainPartialRemoveI4) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  for (std::uint64_t i = 1; i <= 3; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({TestFixture::LC(3), 2, 0, 0, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
  // I4(3) → remove → I4(2).  Chain I4 unchanged.
  verifier.assert_node_counts({TestFixture::LC(2), 2, 0, 0, 0});
}

/// Insert 5 keys (→I16), remove 1 (I16 at min_size → shrink to I4).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainShrinkI16ToI4) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  for (std::uint64_t i = 1; i <= 5; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({TestFixture::LC(5), 1, 1, 0, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
  // I16(5) at min_size → shrink to I4(4).  Chain I4 + bottom I4.
  verifier.assert_node_counts({TestFixture::LC(4), 2, 0, 0, 0});
  verifier.assert_shrinking_inodes({0, 1, 0, 0});
}

/// Insert 17 keys (→I48), remove 1 (I48 at min_size → shrink to I16).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainShrinkI48ToI16) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  for (std::uint64_t i = 1; i <= 17; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({TestFixture::LC(17), 1, 0, 1, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
  // I48(17) at min_size → shrink to I16(16).  Chain I4 + I16.
  verifier.assert_node_counts({TestFixture::LC(16), 1, 1, 0, 0});
  verifier.assert_shrinking_inodes({0, 0, 1, 0});
}

/// Insert 49 keys (→I256), remove 1 (I256 at min_size → shrink to I48).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainShrinkI256ToI48) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  for (std::uint64_t i = 1; i <= 49; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({TestFixture::LC(49), 1, 0, 0, 1});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
  // I256(49) at min_size → shrink to I48(48).  Chain I4 + I48.
  verifier.assert_node_counts({TestFixture::LC(48), 1, 0, 1, 0});
  verifier.assert_shrinking_inodes({0, 0, 0, 1});
}

// -------------------------------------------------------------------
// Group 6c: Full remove — all keys removed through each bottom inode
// -------------------------------------------------------------------

/// Insert 17 keys into chain (bottom I48), remove all.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainRemoveAllFromI48) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  for (std::uint64_t i = 1; i <= 17; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  for (std::uint64_t i = 1; i <= 17; ++i)
    verifier.remove(make_key(enc, 0x42, i));
  verifier.assert_empty();
}

/// Insert 49 keys into chain (bottom I256), remove all.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainRemoveAllFromI256) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CascadeChainUnderI4) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  // Insert chain keys first so the chain forms at root level, then
  // insert a short key that splits the root prefix → parent I4.
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_short_key(enc, 0x01), val);
  // root I4(2: 0x42→chain, 0x01→bare leaf) + chain I4 + bottom I4 + 3 leaves
  // Short key (1 byte) has no suffix → no chain wrapper.
  verifier.assert_node_counts({TestFixture::LC(3), 2, 0, 0, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  // Keyless leaf prevents collapse.  Chain I4 above surviving leaf stays.
  verifier.assert_node_counts({TestFixture::LC(2), 2, 0, 0, 0});

  verifier.remove(make_key(enc, 0x42, 2));
  verifier.check_present_values();
  // Short key's leaf is under its own chain I4 (keyless, no collapse).
  verifier.assert_node_counts({TestFixture::LC(1), 1, 0, 0, 0});
}

/// Chain under I16(5 children).  Remove chain → I16 shrinks to I4.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CascadeChainUnderI16) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  // Chain keys first, then 4 short keys → root I16(5 children).
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x04; ++t)
    verifier.insert(make_short_key(enc, t), val);
  // root I16(5) + chain I4 + bottom I4 + 6 leaves
  // Short keys (1 byte) have no suffix → no chain wrappers → I4=1 not 2.
  verifier.assert_node_counts({TestFixture::LC(6), 1, 1, 0, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.check_present_values();
  // Chain reclaimed.  I16(5) → remove chain slot → is_min_size →
  // shrink to I4(4).
  verifier.assert_node_counts({TestFixture::LC(4), 1, 0, 0, 0});
}

/// Chain under I48(17 children).  Remove chain → I48 shrinks to I16.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CascadeChainUnderI48) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  // Chain keys first, then 16 short keys → root I48(17 children).
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x10; ++t)
    verifier.insert(make_short_key(enc, t), val);
  // root I48(17) + chain I4 + bottom I4 + 18 leaves
  verifier.assert_node_counts({TestFixture::LC(18), 1, 0, 1, 0});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.check_present_values();
  // Chain reclaimed.  I48(17) → remove chain slot → shrink to I16(16).
  verifier.assert_node_counts({TestFixture::LC(16), 0, 1, 0, 0});
}

/// Chain under I256(49 children).  Remove chain → I256 shrinks to I48.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CascadeChainUnderI256) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  // Chain keys first, then 48 short keys → root I256(49 children).
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x30; ++t)
    verifier.insert(make_short_key(enc, t), val);
  // root I256(49) + chain I4 + bottom I4 + 50 leaves
  verifier.assert_node_counts({TestFixture::LC(50), 1, 0, 0, 1});

  verifier.remove(make_key(enc, 0x42, 1));
  verifier.remove(make_key(enc, 0x42, 2));
  verifier.check_present_values();
  // Chain reclaimed.  I256(49) → remove chain slot → shrink to I48(48).
  verifier.assert_node_counts({TestFixture::LC(48), 0, 0, 1, 0});
}

// Chain under I48(18 children, above min_size).  Remove chain child
// via remove_child_entry (no shrink).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainRemoveFromI48AboveMin) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainRemoveFromI256AboveMin) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, MultiLevelChainRemoveAll) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  auto make17 = [&](std::uint8_t last) {
    enc.reset();
    for (unsigned i = 0; i < 16; ++i) enc.encode(std::uint8_t{0xAA});
    enc.encode(last);
    return enc.get_key_view();
  };
  verifier.insert(make17(0x01), val);
  verifier.insert(make17(0x02), val);
  verifier.assert_node_counts({TestFixture::LC(2), 3, 0, 0, 0});

  verifier.remove(make17(0x01));
  // Keyless leaf prevents collapse.  Three chain I4s remain.
  verifier.assert_node_counts({TestFixture::LC(1), 3, 0, 0, 0});

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainParentGrowthI4ToI16) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  // Chain subtree under tag=0x42.
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  // 4 short keys to grow parent past I4 capacity.
  for (std::uint8_t t = 0x01; t <= 0x04; ++t)
    verifier.insert(make_short_key(enc, t), val);
  verifier.check_present_values();
  // root I16(5: 0x42→chain, 0x01..0x04→bare leaves) + chain-I4 + bottom-I4
  // Short keys have no suffix → no chain wrappers → I4=1.
  verifier.assert_node_counts({TestFixture::LC(6), 1, 1, 0, 0});
}

/// Parent I16→I48 growth with a chain child.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainParentGrowthI16ToI48) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x10; ++t)
    verifier.insert(make_short_key(enc, t), val);
  verifier.check_present_values();
  // root I48(17) + chain-I4 + bottom-I4
  verifier.assert_node_counts({TestFixture::LC(18), 1, 0, 1, 0});
}

/// Parent I48→I256 growth with a chain child.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainParentGrowthI48ToI256) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  for (std::uint8_t t = 0x01; t <= 0x30; ++t)
    verifier.insert(make_short_key(enc, t), val);
  verifier.check_present_values();
  // root I256(49) + chain-I4 + bottom-I4
  verifier.assert_node_counts({TestFixture::LC(50), 1, 0, 0, 1});
}

// -------------------------------------------------------------------
// Group 8: Bottom inode growth with cumulative stats (GAP C)
// -------------------------------------------------------------------

/// Verify growing_inodes through I4→I16→I48→I256 under a chain.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainBottomGrowthStats) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  // Insert 4 keys: chain-I4 + bottom-I4(4).
  for (std::uint64_t i = 1; i <= 4; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({TestFixture::LC(4), 2, 0, 0, 0});
  // chain-I4 creation + bottom-I4 creation = 2 I4 grows.
  verifier.assert_growing_inodes({2, 0, 0, 0});

  // 5th key: bottom I4→I16.
  verifier.insert(make_key(enc, 0x42, 5), val);
  verifier.assert_node_counts({TestFixture::LC(5), 1, 1, 0, 0});
  verifier.assert_growing_inodes({2, 1, 0, 0});

  // 17th key: bottom I16→I48.
  for (std::uint64_t i = 6; i <= 17; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({TestFixture::LC(17), 1, 0, 1, 0});
  verifier.assert_growing_inodes({2, 1, 1, 0});

  // 49th key: bottom I48→I256.
  for (std::uint64_t i = 18; i <= 49; ++i)
    verifier.insert(make_key(enc, 0x42, i), val);
  verifier.assert_node_counts({TestFixture::LC(49), 1, 0, 0, 1});
  verifier.assert_growing_inodes({2, 1, 1, 1});

  verifier.check_present_values();
}

// -------------------------------------------------------------------
// Group 9: Multi-level chain with fat bottom inode (GAP D)
// -------------------------------------------------------------------

/// Two chain levels + I16 at bottom (5 keys, 17 bytes each).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, MultiLevelChainFatBottom) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  auto make17 = [&](std::uint8_t last) {
    enc.reset();
    for (unsigned i = 0; i < 16; ++i) enc.encode(std::uint8_t{0xAA});
    enc.encode(last);
    return enc.get_key_view();
  };

  for (std::uint8_t i = 1; i <= 5; ++i) verifier.insert(make17(i), val);
  verifier.check_present_values();
  // 2 chain-I4s + bottom I16(5) + 5 leaves
  verifier.assert_node_counts({TestFixture::LC(5), 2, 1, 0, 0});
  verifier.assert_growing_inodes({3, 1, 0, 0});

  // Remove 1 → I16 at min_size → shrink to I4.
  verifier.remove(make17(1));
  verifier.assert_node_counts({TestFixture::LC(4), 3, 0, 0, 0});
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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, MidLevelInodeGrowth) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
  verifier.assert_node_counts({TestFixture::LC(10), 11, 1, 0, 0});
}

/// Mid-level inode shrinkage with chains above and below.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, MidLevelInodeShrink) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
  verifier.assert_node_counts({TestFixture::LC(8), 10, 0, 0, 0});
  verifier.assert_shrinking_inodes({2, 1, 0, 0});
}

// -------------------------------------------------------------------
// Group 10a: Chain remove — key not found / mismatch coverage
// -------------------------------------------------------------------

// Remove non-existent key where chain I4's find_child returns nullptr.
// The chain has one child at dispatch byte 0x01; we try dispatch 0x03.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainRemoveKeyNotFound) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.remove(make_key(enc, 0x42, 2));
  // Chain I4 has 1 child at dispatch 0x01.  v=3 → dispatch 0x03.
  UNODB_ASSERT_FALSE(verifier.get_db().remove(make_key(enc, 0x42, 3)));
  verifier.check_present_values();
}

// Remove non-existent key when root is a single leaf.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, RemoveMissRootIsLeaf) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  verifier.insert(make_key(enc, 0x42, 1), val);
  UNODB_ASSERT_FALSE(verifier.get_db().remove(make_key(enc, 0x42, 2)));
  verifier.check_present_values();
}

// Remove key where chain I4's child is a leaf that doesn't match.
// Use a 10-byte key that shares the chain prefix and dispatch byte
// but differs at the leaf level.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainRemoveLeafMismatch) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainCutCD1CollapseToLeaf) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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

  // Remove A: chain cut (CD=1). Root I4(2→1).
  // Surviving child is chain-I4 with 7-byte prefix.  Merge would be
  // 0 + 1 + 7 = 8 > 7 → prefix overflow, no collapse.
  // Tree: root-I4(1) → chain-I4 → leaf(C).
  verifier.remove(make18(0x01, 0x00, 0x01));
  verifier.check_present_values();
  verifier.assert_node_counts({TestFixture::LC(1), 2, 0, 0, 0});

  // Remove C: tree empty.
  verifier.remove(make18(0x02, 0x00, 0x01));
  verifier.assert_empty();
}

// T7: I4(2) collapse with CD=0 chain, remaining child is inode.
// Remove A → chain cut, I4 collapses, remaining child promoted.
// Use 1-byte keys for B,C so remaining child has short prefix.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainCutCD0CollapseToInode) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainCutCD1CollapseToInode) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainCutCD1RemoveFromI4) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest,
                 ChainCutCD1CollapseToInodeShortPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainCutCD2CollapseToLeaf) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainCutCD0PrefixOverflow) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainCutCD1PrefixOverflow) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainCutCD1ShrinkI16) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
  verifier.assert_node_counts({TestFixture::LC(8), 9, 0, 0, 0});
  verifier.assert_shrinking_inodes({2, 1, 0, 0});
}

// T16: I48(min) shrink with CD=1 chain.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainCutCD1ShrinkI48) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
  verifier.assert_node_counts({TestFixture::LC(32), 32, 1, 0, 0});
  verifier.assert_shrinking_inodes({2, 0, 1, 0});
}

// T18: I256(min) shrink with CD=1 chain.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ChainCutCD1ShrinkI256) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
  verifier.assert_node_counts({TestFixture::LC(96), 96, 0, 1, 0});
  verifier.assert_shrinking_inodes({2, 0, 0, 1});
}

// T14: Keyless leaf no-collapse guard.
// I4(2) where surviving child is a keyless leaf → collapse blocked.
// Tree: root-I4(2: 0x01→chain→leaf(A), 0x02→chain→leaf(B))
// Remove A → root-I4(2→1).  Surviving child is chain-I4 (inode),
// but that chain's only child is a keyless leaf(B).  The chain-I4
// itself is an inode, so can_collapse checks prefix overflow, not
// the keyless guard.  However, the chain-I4 above leaf(B) is also
// I4(1), and if IT were collapsed, the leaf would become root —
// losing key bytes.  The no-collapse guard fires at the chain-I4
// level, not at root.
//
// To test the guard directly: need I4(2) where one child is removed
// and the other is directly a keyless leaf.  This requires a 1-byte
// key (dispatch byte only, no chain above the leaf).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, KeylessLeafNoCollapseGuard) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  // Two 1-byte keys with different dispatch bytes.
  std::array<std::byte, 1> buf_a{};
  std::array<std::byte, 1> buf_b{};
  auto kv = enc.reset().encode(std::uint8_t{0x01}).get_key_view();
  std::ignore = std::ranges::copy(kv, buf_a.begin());
  kv = enc.reset().encode(std::uint8_t{0x02}).get_key_view();
  std::ignore = std::ranges::copy(kv, buf_b.begin());
  const auto key_a = unodb::key_view{buf_a.data(), buf_a.size()};
  const auto key_b = unodb::key_view{buf_b.data(), buf_b.size()};

  verifier.insert(key_a, val);
  verifier.insert(key_b, val);
  verifier.check_present_values();

  // 2 leaves + root-I4.  1-byte keys have no prefix bytes, so no
  // chain I4s are created — leaves go directly into root.
  verifier.assert_node_counts({TestFixture::LC(2), 1, 0, 0, 0});

  // Remove A.  Root-I4(2→1).  Surviving child is leaf(B) (keyless).
  // can_collapse returns false (keyless leaf guard).  Root-I4(1) stays.
  verifier.remove(key_a);
  verifier.check_present_values();

  // Root-I4(1) + 1 leaf.  No collapse.
  verifier.assert_node_counts({TestFixture::LC(1), 1, 0, 0, 0});

  verifier.remove(key_b);
  verifier.assert_empty();
}

// T15: Merge allowed when surviving child is an inode (not keyless leaf).
// I4(2) where one child is a chain subtree, the other is an inode subtree.
// Remove the chain → I4(2→1), surviving child is inode → collapse allowed.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, CollapseToInodeAllowed) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  // A: 9-byte key tag=0x01.
  // B,C: 9-byte keys tag=0x02, different suffixes → I4(2) under 0x02.
  std::array<std::byte, 9> buf_a{};
  std::array<std::byte, 9> buf_b{};
  std::array<std::byte, 9> buf_c{};
  auto kv = enc.reset()
                .encode(std::uint8_t{0x01})
                .encode(std::uint64_t{0})
                .get_key_view();
  std::ignore = std::ranges::copy(kv, buf_a.begin());
  kv = enc.reset()
           .encode(std::uint8_t{0x02})
           .encode(std::uint64_t{0})
           .get_key_view();
  std::ignore = std::ranges::copy(kv, buf_b.begin());
  kv = enc.reset()
           .encode(std::uint8_t{0x02})
           .encode(std::uint64_t{1})
           .get_key_view();
  std::ignore = std::ranges::copy(kv, buf_c.begin());
  const auto key_a = unodb::key_view{buf_a.data(), buf_a.size()};
  const auto key_b = unodb::key_view{buf_b.data(), buf_b.size()};
  const auto key_c = unodb::key_view{buf_c.data(), buf_c.size()};

  verifier.insert(key_a, val);
  verifier.insert(key_b, val);
  verifier.insert(key_c, val);
  verifier.check_present_values();

  // Remove A.  Root-I4(2→1).  Surviving child under 0x02 is an I4(2)
  // (inode, not leaf).  Prefix merge fits → collapse allowed.
  verifier.remove(key_a);
  verifier.check_present_values();

  // After collapse: root-I4 merged into chain-I4 (prefix overflow
  // prevents further collapse into bottom-I4).
  // Root-chain-I4(1 child) + bottom-I4(2 children) + 2 leaves.
  verifier.assert_node_counts({TestFixture::LC(2), 2, 0, 0, 0});

  verifier.remove(key_b);
  verifier.remove(key_c);
  verifier.assert_empty();
}

// -------------------------------------------------------------------
// Verify tree structures used by concurrent chain cut tests (CT1-CT4).
// These confirm the chain depth and that insert/remove work correctly
// on these key patterns before we add concurrency.

// CT1/CT3 tree: 26-byte keys, 3 chain levels after removing B.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ConcurrentTestTree26Byte) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ConcurrentTestTree34Byte) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

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
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, ScanChainMixedLengths) {
  unodb::test::tree_verifier<TypeParam> verifier;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  // 9-byte and 10-byte keys in the same chain subtree.
  // make_key(0x42, v) = 9 bytes.  make_long_key(0x42, v, s) = 10 bytes.
  // Keys must not be in a prefix relationship — use different v values.
  verifier.insert(make_key(enc, 0x42, 1), val);
  verifier.insert(make_key(enc, 0x42, 2), val);
  verifier.insert(make_long_key(enc, 0x42, 3, 0x01), val);
  verifier.insert(make_long_key(enc, 0x42, 4, 0x01), val);
  // check_present_values does a full scan + per-key probe.
  verifier.check_present_values();
  // Chain I4 (prefix) + divergence I4 + chain I4s for each key's suffix
  verifier.assert_node_counts({TestFixture::LC(4), 4, 0, 0, 0});

  // Remove a 10-byte key, verify scan still works.
  verifier.remove(make_long_key(enc, 0x42, 3, 0x01));
  verifier.check_present_values();

  // Remove a 9-byte key.
  verifier.remove(make_key(enc, 0x42, 1));
  verifier.check_present_values();
}

#endif  // UNODB_DETAIL_WITH_STATS

#ifdef UNODB_DETAIL_WITH_STATS

// ===================================================================
// Stack structure validation tests (D5).
//
// Verify that the iterator stack encodes the full key in the inode
// path for can_eliminate_leaf trees (Mode 3).  At each leaf position,
// the concatenation of (prefix + dispatch_byte) across inode entries
// must equal the full encoded key.
// ===================================================================

template <class Db>
void verify_stack(typename Db::iterator& it, unodb::key_view expected_key) {
  UNODB_ASSERT_TRUE(it.valid());
  auto stk = it.test_only_stack();
  UNODB_ASSERT_TRUE(stk.size() >= 1U);

  // For can_eliminate_leaf types, the top is a packed value sentinel (0xFF).
  // For leaf types, the top is a LEAF node.
  if (stk.back().child_index == static_cast<std::uint8_t>(0xFFU)) {
    // Packed value — no type check possible.
  } else {
    UNODB_EXPECT_EQ(stk.back().node.type(), unodb::node_type::LEAF);
  }

  const auto inode_end = stk.size() - 1;
  for (std::size_t i = 0; i < inode_end; ++i) {
    UNODB_EXPECT_NE(stk[i].node.type(), unodb::node_type::LEAF);
  }

  // Reconstruct key from inode prefix+dispatch bytes.
  std::vector<std::byte> reconstructed;
  for (std::size_t i = 0; i < inode_end; ++i) {
    auto prefix = stk[i].prefix.get_key_view();
    for (std::size_t j = 0; j < prefix.size(); ++j)
      reconstructed.push_back(prefix[j]);
    reconstructed.push_back(stk[i].key_byte);
  }
  UNODB_ASSERT_EQ(reconstructed.size(), expected_key.size());
  for (std::size_t i = 0; i < reconstructed.size(); ++i) {
    UNODB_EXPECT_EQ(reconstructed[i], expected_key[i]);
  }
}

/// Two keys sharing a long prefix (chain structure).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, StackStructureTwoChainKeys) {
  TypeParam db;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  std::array<std::byte, 9> buf_a{};
  std::array<std::byte, 9> buf_b{};
  auto kv = enc.reset()
                .encode(std::uint8_t{0x01})
                .encode(std::uint64_t{100})
                .get_key_view();
  std::ignore = std::ranges::copy(kv, buf_a.begin());
  kv = enc.reset()
           .encode(std::uint8_t{0x01})
           .encode(std::uint64_t{200})
           .get_key_view();
  std::ignore = std::ranges::copy(kv, buf_b.begin());

  const auto key_a = unodb::key_view{buf_a.data(), buf_a.size()};
  const auto key_b = unodb::key_view{buf_b.data(), buf_b.size()};

  std::ignore = db.insert(key_a, val);
  std::ignore = db.insert(key_b, val);

  auto it = db.test_only_iterator();
  it.first();
  verify_stack<TypeParam>(it, it.get_key().view());
  it.next();
  if (it.valid()) verify_stack<TypeParam>(it, it.get_key().view());
}

/// Three keys with different first bytes (wide I4, no chain).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, StackStructureWideNode) {
  TypeParam db;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  std::array<std::byte, 1> ba{};
  std::array<std::byte, 1> bb{};
  std::array<std::byte, 1> bc{};
  auto kv = enc.reset().encode(std::uint8_t{0x10}).get_key_view();
  std::ignore = std::ranges::copy(kv, ba.begin());
  kv = enc.reset().encode(std::uint8_t{0x20}).get_key_view();
  std::ignore = std::ranges::copy(kv, bb.begin());
  kv = enc.reset().encode(std::uint8_t{0x30}).get_key_view();
  std::ignore = std::ranges::copy(kv, bc.begin());

  const auto ka = unodb::key_view{ba.data(), ba.size()};
  const auto kb = unodb::key_view{bb.data(), bb.size()};
  const auto kc = unodb::key_view{bc.data(), bc.size()};

  std::ignore = db.insert(ka, val);
  std::ignore = db.insert(kb, val);
  std::ignore = db.insert(kc, val);

  auto it = db.test_only_iterator();
  for (it.first(); it.valid(); it.next())
    verify_stack<TypeParam>(it, it.get_key().view());
}

/// Second insert with different tag must also create a full chain.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, StackStructureSecondInsertChain) {
  TypeParam db;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  std::array<std::byte, 9> buf_a{};
  std::array<std::byte, 9> buf_b{};
  auto kv = enc.reset()
                .encode(std::uint8_t{0x01})
                .encode(std::uint64_t{0})
                .get_key_view();
  std::ignore = std::ranges::copy(kv, buf_a.begin());
  kv = enc.reset()
           .encode(std::uint8_t{0x02})
           .encode(std::uint64_t{0})
           .get_key_view();
  std::ignore = std::ranges::copy(kv, buf_b.begin());

  const auto ka = unodb::key_view{buf_a.data(), buf_a.size()};
  const auto kb = unodb::key_view{buf_b.data(), buf_b.size()};

  std::ignore = db.insert(ka, val);
  std::ignore = db.insert(kb, val);

  auto it = db.test_only_iterator();
  for (it.first(); it.valid(); it.next())
    verify_stack<TypeParam>(it, it.get_key().view());
}

/// Forward and reverse scan, verify stack at every position.
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, StackStructureFullScan) {
  TypeParam db;
  unodb::key_encoder enc;
  const auto val = unodb::test::get_test_value<TypeParam>(0);

  struct kh {
    std::array<std::byte, 18> buf{};
    std::size_t len{};
    [[nodiscard]] unodb::key_view kv() const { return {buf.data(), len}; }
  };
  auto make = [&](std::uint8_t tag, std::uint64_t v) {
    kh h;
    auto k = enc.reset().encode(tag).encode(v).get_key_view();
    std::ignore = std::ranges::copy(k, h.buf.begin());
    h.len = k.size();
    return h;
  };

  auto k1 = make(0x01, 100);
  auto k2 = make(0x01, 200);
  auto k3 = make(0x02, 300);
  auto k4 = make(0x03, 0);

  std::ignore = db.insert(k1.kv(), val);
  std::ignore = db.insert(k2.kv(), val);
  std::ignore = db.insert(k3.kv(), val);
  std::ignore = db.insert(k4.kv(), val);

  // Forward scan.
  {
    auto it = db.test_only_iterator();
    int count = 0;
    for (it.first(); it.valid(); it.next()) {
      verify_stack<TypeParam>(it, it.get_key().view());
      ++count;
    }
    UNODB_EXPECT_EQ(count, 4);
  }

  // Reverse scan.
  {
    auto it = db.test_only_iterator();
    int count = 0;
    for (it.last(); it.valid(); it.prior()) {
      verify_stack<TypeParam>(it, it.get_key().view());
      ++count;
    }
    UNODB_EXPECT_EQ(count, 4);
  }
}

#endif  // UNODB_DETAIL_WITH_STATS

// Empty key_view must be rejected (not UB).
UNODB_TYPED_TEST(ARTKeyViewFullChainTest, EmptyKeyRejected) {
  TypeParam db;
  const std::byte empty_buf{};
  const unodb::key_view empty_key{&empty_buf, 0};
  UNODB_ASSERT_THROW(std::ignore = db.insert(
                         empty_key, unodb::test::get_test_value<TypeParam>(0)),
                     std::length_error);
  UNODB_ASSERT_TRUE(db.empty());
}

// Regression test: scan_range with 1000 compound float keys must return
// results in encoded key order.  A bug in keybuf_.pop() treated child_index
// 0xFF as a VIS sentinel even when can_eliminate_leaf was false, corrupting
// the reconstructed key during iterator ascent.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26426)
TEST(ARTKeyViewValueViewTest, ScanRangeFloatCompoundKeyOrder) {
  unodb::db<unodb::key_view, unodb::value_view> db;
  unodb::key_encoder enc;
  const std::uint8_t flag = 0x76;
  const float step = 100.0f / 1000.0f;
  const std::uint64_t dummy_val = 42;
  const unodb::value_view val{reinterpret_cast<const std::byte*>(&dummy_val),
                              sizeof(dummy_val)};

  for (int i = 0; i < 1000; i++) {
    enc.reset();
    enc.encode(step * static_cast<float>(i));
    enc.encode(flag);
    enc.encode(static_cast<std::uint64_t>(i));
    ASSERT_TRUE(db.insert(enc.get_key_view(), val));
  }

  unodb::key_encoder e1;
  unodb::key_encoder e2;
  e1.reset();
  e1.encode(0.0f);
  e1.encode(std::uint8_t{0});
  e1.encode(std::uint64_t{0});
  e2.reset();
  e2.encode(100.0f);
  e2.encode(std::uint8_t{0xFF});
  e2.encode(~std::uint64_t{0});

  float prev = -1.0f;
  int count = 0;
  db.scan_range(e1.get_key_view(), e2.get_key_view(), [&](auto& v) {
    unodb::key_decoder dec(v.get_key());
    float decoded{};
    dec.decode(decoded);
    EXPECT_GE(decoded, prev);
    prev = decoded;
    count++;
    return false;
  });
  EXPECT_EQ(count, 1000);
}

// Regression test: compound key insert must not crash under GCC -O2 strict
// aliasing.  After add_to_nonfull inserts a leaf, the full_key_in_inode_path
// code calls find_child to locate the slot for chain wrapping.  GCC 12+ at -O2
// can optimize away the re-read of inode data (strict aliasing / #700),
// causing find_child to return nullptr and crash.  This test exercises the
// exact pattern: 6 compound keys (float + uint8 + uint64) that diverge at
// different prefix depths, forcing inode splits.
TEST(KeyViewFullChainRegression, CompoundKeyInsertStrictAliasing) {
  unodb::db<unodb::key_view, unodb::value_view> db;
  unodb::key_encoder enc;
  const float values[] = {-100.0f, -1.0f, 0.0f, 1.0f, 100.0f, 1000.0f};
  constexpr std::uint8_t flag = 0x76;

  for (int i = 0; i < 6; ++i) {
    enc.reset();
    enc.encode(values[i]);
    enc.encode(flag);
    enc.encode(static_cast<std::uint64_t>(i));
    auto key = enc.get_key_view();
    auto val =
        unodb::value_view(reinterpret_cast<const std::byte*>(&i), sizeof(i));
    ASSERT_TRUE(db.insert(key, val)) << "insert failed at i=" << i;
  }

  for (int i = 0; i < 6; ++i) {
    enc.reset();
    enc.encode(values[i]);
    enc.encode(flag);
    enc.encode(static_cast<std::uint64_t>(i));
    ASSERT_TRUE(db.get(enc.get_key_view()).has_value())
        << "get failed at i=" << i;
  }
}

// Regression: scan key reconstruction with 0xFF child index in VIS trees.
// The iterator used child_index==0xFF as a sentinel for value-in-slot entries,
// but 0xFF is a valid child index.  With enough compound keys the c1 subtree
// inode fills all 256 child slots including 0xFF, causing pop() to skip keybuf
// truncation and corrupt every subsequent reconstructed key.
TEST(KeyViewFullChainRegression, ScanKeyReconstructionFF) {
  unodb::db<unodb::key_view, std::uint64_t> db;
  constexpr int N = 321;  // enough to span the c1->c2 encoded-float boundary
  const float step = 100.0f / 1000.0f;

  for (int i = 0; i < N; ++i) {
    unodb::key_encoder enc;
    enc.encode(step * static_cast<float>(i));
    enc.encode(static_cast<std::uint8_t>(0x76));
    enc.encode(static_cast<std::uint64_t>(i));
    ASSERT_TRUE(db.insert(enc.get_key_view(), static_cast<std::uint64_t>(i)));
  }

  int count = 0;
  float prev = -1.0f;
  db.scan([&](auto& v) {
    auto tkv = v.get_key();
    EXPECT_EQ(tkv.size(), 13u) << "wrong key size at index " << count;
    unodb::key_decoder dec(unodb::key_view(tkv.data(), tkv.size()));
    float val{};
    dec.decode(val);
    EXPECT_GE(val, prev) << "order violation at index " << count;
    prev = val;
    ++count;
    return false;
  });
  EXPECT_EQ(count, N);
}

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

}  // namespace
