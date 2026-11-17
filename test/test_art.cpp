// Copyright 2019-2026 UnoDB contributors

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

// IWYU pragma: no_include <array>
// IWYU pragma: no_include <span>
// IWYU pragma: no_include <string>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "art_common.hpp"
#include "art_test_data.hpp"
#include "db_test_utils.hpp"
#include "gtest_utils.hpp"
#include "test_utils.hpp"
#include "thread_sync.hpp"

namespace {
using unodb::detail::thread_syncs;
using unodb::test_data::test_values;
/// For key_view types, build_chain creates 1 chain I4 per 8-byte key.
/// These are counter adjustment values (0 or 1), not booleans.
template <class Db>
[[maybe_unused]] constexpr std::uint64_t chain_i4_per_key =
    std::is_same_v<typename Db::key_type, unodb::key_view> ? 1 : 0;

/// OLC remove collapses chain I4s during prefix merge; db/mutex_db don't.
/// Counter adjustment value (0 or 1), not a boolean.
template <class Db>
[[maybe_unused]] constexpr std::uint64_t olc_chain_collapse =
    (std::is_same_v<typename Db::key_type, unodb::key_view> &&
     std::is_same_v<
         Db, unodb::olc_db<typename Db::key_type, typename Db::value_type>>)
        ? 1
        : 0;

template <class Db>
class ARTCorrectnessTest : public ::testing::Test {
 public:
  using Test::Test;
};

using ARTTypes =
    ::testing::Types<unodb::test::u64_db, unodb::test::u64_mutex_db,
                     unodb::test::u64_olc_db, unodb::test::key_view_db,
                     unodb::test::key_view_mutex_db,
                     unodb::test::key_view_olc_db>;

UNODB_TYPED_TEST_SUITE(ARTCorrectnessTest, ARTTypes)

UNODB_TYPED_TEST(ARTCorrectnessTest, SingleNodeTreeEmptyValue) {
  unodb::test::tree_verifier<TypeParam> verifier;
  verifier.check_absent_keys({1});
  verifier.insert(1, {});

  verifier.check_present_values();
  verifier.check_absent_keys({0});

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({1, ci4, 0, 0, 0});
  verifier.assert_growing_inodes({ci4, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, SingleNodeTreeNonemptyValue) {
  unodb::test::tree_verifier<TypeParam> verifier;
  verifier.insert(1, test_values[2]);

  verifier.check_present_values();
  verifier.check_absent_keys({0, 2});

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({1, ci4, 0, 0, 0});
  verifier.assert_growing_inodes({ci4, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, TooLongValue) {
  unodb::test::tree_verifier<TypeParam> verifier;

  UNODB_ASSERT_THROW(std::ignore = verifier.try_insert(
                         1, unodb::test_data::too_long_value_view()),
                     std::length_error);

  verifier.check_absent_keys({1});
  verifier.assert_empty();

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_growing_inodes({0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, ExpandLeafToNode4) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(0, test_values[1]);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({1, ci4, 0, 0, 0});
  verifier.assert_growing_inodes({ci4, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.insert(1, test_values[2]);

  verifier.check_present_values();
  verifier.check_absent_keys({2});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({2, 1, 0, 0, 0});
  verifier.assert_growing_inodes({1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, DuplicateKey) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(0, test_values[0]);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({1, ci4, 0, 0, 0});

  const auto mem_use_before = verifier.get_db().get_current_memory_use();
#endif  // UNODB_DETAIL_WITH_STATS

  unodb::test::must_not_allocate([&verifier] {
    UNODB_ASSERT_FALSE(verifier.try_insert(0, test_values[3]));
  });

  verifier.check_present_values();

#ifdef UNODB_DETAIL_WITH_STATS
  UNODB_ASSERT_EQ(mem_use_before, verifier.get_db().get_current_memory_use());

  verifier.assert_node_counts({1, ci4, 0, 0, 0});
  verifier.assert_growing_inodes({ci4, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, InsertToFullNode4) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0, 4);

  verifier.check_present_values();
  verifier.check_absent_keys({5, 4});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({4, 1, 0, 0, 0});
  verifier.assert_growing_inodes({1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node4InsertFFByte) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0xFC, 4);

  verifier.check_present_values();
  verifier.check_absent_keys({0, 0xFB});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({4, 1, 0, 0, 0});
  verifier.assert_growing_inodes({1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, TwoNode4) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(1, test_values[0]);
  verifier.insert(3, test_values[2]);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_growing_inodes({1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  // Insert a value that does not share full prefix with the current Node4
  verifier.insert(0xFF01, test_values[3]);

  verifier.check_present_values();
  verifier.check_absent_keys({0xFF00, 2});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({3, 2 + ci4, 0, 0, 0});
  verifier.assert_growing_inodes({2 + ci4, 0, 0, 0});
  verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, DbInsertNodeRecursion) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(1, test_values[0]);
  verifier.insert(3, test_values[2]);
  // Insert a value that does not share full prefix with the current Node4
  verifier.insert(0xFF0001, test_values[3]);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_growing_inodes({2 + ci4, 0, 0, 0});
  verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS

  // Then insert a value that shares full prefix with the above node and will
  // ask for a recursive insertion there
  verifier.insert(0xFF0101, test_values[1]);

  verifier.check_present_values();
  verifier.check_absent_keys({0xFF0100, 0xFF0000, 2});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({4, 3 + (2 * ci4), 0, 0, 0});
  verifier.assert_growing_inodes({3 + (2 * ci4), 0, 0, 0});

  verifier.assert_key_prefix_splits((1 + ci4));
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node16) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0, 4);
  verifier.check_present_values();
  verifier.insert(5, test_values[0]);

  verifier.check_present_values();
  verifier.check_absent_keys({6ULL, 0x0100ULL, 0xFFFFFFFFFFFFFFFFULL});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({5, 0, 1, 0, 0});
  verifier.assert_growing_inodes({1, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, FullNode16) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0, 16);

  verifier.check_absent_keys({16});
  verifier.check_present_values();

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({16, 0, 1, 0, 0});
  verifier.assert_growing_inodes({1, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node16KeyPrefixSplit) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(10, 5);

  // Insert a value that does share full prefix with the current Node16
  verifier.insert(0x1020, test_values[0]);

  verifier.check_present_values();
  verifier.check_absent_keys({9, 0x10FF});

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({6, (1 + ci4), 1, 0, 0});
  verifier.assert_growing_inodes({2 + ci4, 1, 0, 0});
  verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node16KeyInsertOrderDescending) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(5, test_values[0]);
  verifier.insert(4, test_values[1]);
  verifier.insert(3, test_values[2]);
  verifier.insert(2, test_values[3]);
  verifier.insert(1, test_values[4]);
  verifier.insert(0, test_values[0]);

  verifier.check_present_values();
  verifier.check_absent_keys({6});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({6, 0, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node16ConstructWithFFKeyByte) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0xFB, 4);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({4, 1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.insert(0xFF, test_values[0]);

  verifier.check_present_values();
  verifier.check_absent_keys({0, 0xFA});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({5, 0, 1, 0, 0});
  verifier.assert_growing_inodes({1, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node48) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0, 17);

  verifier.check_present_values();
  verifier.check_absent_keys({17});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({17, 0, 0, 1, 0});
  verifier.assert_growing_inodes({1, 1, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, FullNode48) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0, 48);

  verifier.check_present_values();
  verifier.check_absent_keys({49});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({48, 0, 0, 1, 0});
  verifier.assert_growing_inodes({1, 1, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node48KeyPrefixSplit) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(10, 17);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({17, 0, 0, 1, 0});
  verifier.assert_growing_inodes({1, 1, 1, 0});
  verifier.assert_key_prefix_splits(0);
#endif  // UNODB_DETAIL_WITH_STATS

  // Insert a value that does share full prefix with the current Node48
  verifier.insert(0x100020, test_values[0]);

  verifier.check_present_values();
  verifier.check_absent_keys({9, 27, 0x100019, 0x100100, 0x110000});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({18, (1 + ci4), 0, 1, 0});
  verifier.assert_growing_inodes({2 + ci4, 1, 1, 0});
  verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node256) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 49);

  verifier.check_present_values();
  verifier.check_absent_keys({50});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({49, 0, 0, 0, 1});
  verifier.assert_growing_inodes({1, 1, 1, 1});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, FullNode256) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0, 256);

  verifier.check_present_values();
  verifier.check_absent_keys({256});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({256, 0, 0, 0, 1});
  verifier.assert_growing_inodes({1, 1, 1, 1});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node256KeyPrefixSplit) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(20, 49);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({49, 0, 0, 0, 1});
  verifier.assert_growing_inodes({1, 1, 1, 1});
  verifier.assert_key_prefix_splits(0);
#endif  // UNODB_DETAIL_WITH_STATS

  // Insert a value that does share full prefix with the current Node256
  verifier.insert(0x100020, test_values[0]);

  verifier.check_present_values();
  verifier.check_absent_keys({19, 69, 0x100019, 0x100100, 0x110000});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({50, (1 + ci4), 0, 0, 1});
  verifier.assert_growing_inodes({2 + ci4, 1, 1, 1});
  verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, TryDeleteFromEmpty) {
  unodb::test::tree_verifier<TypeParam> verifier;

  unodb::test::must_not_allocate(
      [&verifier] { verifier.attempt_remove_missing_keys({1}); });

  verifier.assert_empty();
  verifier.check_absent_keys({1});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, SingleNodeTreeDelete) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(1, test_values[0]);

  unodb::test::must_not_allocate([&verifier] { verifier.remove(1); });

  verifier.assert_empty();
  verifier.check_absent_keys({1});
  verifier.attempt_remove_missing_keys({1});
  verifier.check_absent_keys({1});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, SingleNodeTreeAttemptDeleteAbsent) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(2, test_values[1]);

  unodb::test::must_not_allocate(
      [&verifier] { verifier.attempt_remove_missing_keys({1, 3, 0xFF02}); });

  verifier.check_present_values();
  verifier.check_absent_keys({1, 3, 0xFF02});

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({1, ci4, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node4AttemptDeleteAbsent) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 4);

  unodb::test::must_not_allocate([&verifier] {
    verifier.attempt_remove_missing_keys({0ULL, 6ULL, 0xFF000001ULL});
  });

  verifier.check_absent_keys({0, 6, 0xFF00000});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({4, 1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node4FullDeleteMiddleAndBeginning) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 4);

  // Delete from Node4 middle
  unodb::test::must_not_allocate([&verifier] { verifier.remove(2); });

  verifier.check_present_values();
  verifier.check_absent_keys({0, 2, 5});

  // Delete from Node4 beginning
  unodb::test::must_not_allocate([&verifier] { verifier.remove(1); });

  verifier.check_present_values();
  verifier.check_absent_keys({1, 0, 2, 5});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({2, 1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node4FullDeleteEndAndMiddle) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 4);

  // Delete from Node4 end
  unodb::test::must_not_allocate([&verifier] { verifier.remove(4); });

  verifier.check_present_values();
  verifier.check_absent_keys({4, 0, 5});

  // Delete from Node4 middle
  unodb::test::must_not_allocate([&verifier] { verifier.remove(2); });

  verifier.check_present_values();
  verifier.check_absent_keys({2, 4, 0, 5});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({2, 1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node4ShrinkToSingleLeaf) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 2);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_shrinking_inodes({0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  unodb::test::must_not_allocate([&verifier] { verifier.remove(1); });

  verifier.check_present_values();
  verifier.check_absent_keys({1});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({1 - ci4, 0, 0, 0});
  verifier.assert_node_counts({1, ci4, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node4DeleteLowerNode) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0, 2);
  // Insert a value that does not share full prefix with the current Node4
  verifier.insert(0xFF00, test_values[3]);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_shrinking_inodes({0, 0, 0, 0});
  verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS

  // Make the lower Node4 shrink to a single value leaf
  unodb::test::must_not_allocate([&verifier] { verifier.remove(0); });

  verifier.check_present_values();
  verifier.check_absent_keys({0, 2, 0xFF01});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({1 - ci4, 0, 0, 0});
  verifier.assert_key_prefix_splits(1);
  verifier.assert_node_counts({2, 1 + (2 * ci4), 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node4DeleteKeyPrefixMerge) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0x8001, 2);
  // Insert a value that does not share full prefix with the current Node4
  verifier.insert(0x90AA, test_values[3]);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_key_prefix_splits(1);
  verifier.assert_node_counts({3, 2 + ci4, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  // And delete it
  unodb::test::must_not_allocate([&verifier] { verifier.remove(0x90AA); });

  verifier.check_present_values();
  verifier.check_absent_keys({0x90AA, 0x8003});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_key_prefix_splits(1);
  constexpr auto oc = olc_chain_collapse<TypeParam>;
  verifier.assert_node_counts({2, (1 + ci4) - oc, 0, 0, 0});
  verifier.assert_shrinking_inodes({1 + oc, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node4DeleteKeyPrefixMerge2) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(0x0000000003020102, test_values[0]);
  verifier.insert(0x0000000003030302, test_values[1]);
  verifier.insert(0x0000000100010102, test_values[2]);

  unodb::test::must_not_allocate([&verifier] {
    verifier.remove(0x0000000100010102);
    verifier.remove(0x0000000003020102);
  });

  verifier.check_present_values();
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node16DeleteBeginningMiddleEnd) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 16);

  unodb::test::must_not_allocate([&verifier] {
    verifier.remove(5);
    verifier.remove(1);
    verifier.remove(16);
  });

  verifier.check_present_values();
  verifier.check_absent_keys({0, 1, 5, 16, 17});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({13, 0, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node16ShrinkToNode4DeleteMiddle) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 5);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({5, 0, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.remove(2);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({0, 1, 0, 0});
  verifier.assert_node_counts({4, 1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.check_present_values();
  verifier.check_absent_keys({0, 2, 6});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node16ShrinkToNode4DeleteBeginning) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 5);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({5, 0, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.remove(1);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({0, 1, 0, 0});
  verifier.assert_node_counts({4, 1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.check_present_values();
  verifier.check_absent_keys({0, 1, 6});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node16ShrinkToNode4DeleteEnd) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 5);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({5, 0, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.remove(5);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({0, 1, 0, 0});
  verifier.assert_node_counts({4, 1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.check_present_values();
  verifier.check_absent_keys({0, 5, 6});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node16KeyPrefixMerge) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(10, 5);
  // Insert a value that does not share full prefix with the current Node16
  verifier.insert(0x1020, test_values[0]);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({6, (1 + ci4), 1, 0, 0});
  verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS

  // And delete it, so that upper level Node4 key prefix gets merged with
  // Node16 one
  unodb::test::must_not_allocate([&verifier] { verifier.remove(0x1020); });

  verifier.check_present_values();
  verifier.check_absent_keys({9, 16, 0x1020});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes(
      {1 + olc_chain_collapse<TypeParam>, 0, 0, 0});
  verifier.assert_node_counts(
      {5, ci4 - olc_chain_collapse<TypeParam>, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node48DeleteBeginningMiddleEnd) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 48);

  unodb::test::must_not_allocate([&verifier] {
    verifier.remove(30);
    verifier.remove(48);
    verifier.remove(1);
  });

  verifier.check_present_values();
  verifier.check_absent_keys({0, 1, 30, 48, 49});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({45, 0, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node48ShrinkToNode16DeleteMiddle) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(0x80, 17);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({17, 0, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.remove(0x85);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({0, 0, 1, 0});
  verifier.assert_node_counts({16, 0, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.check_present_values();
  verifier.check_absent_keys({0x7F, 0x85, 0x91});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node48ShrinkToNode16DeleteBeginning) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 17);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({17, 0, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.remove(1);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({0, 0, 1, 0});
  verifier.assert_node_counts({16, 0, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.check_present_values();
  verifier.check_absent_keys({0, 1, 18});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node48ShrinkToNode16DeleteEnd) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 17);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({17, 0, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.remove(17);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({0, 0, 1, 0});
  verifier.assert_node_counts({16, 0, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.check_present_values();
  verifier.check_absent_keys({0, 17, 18});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node48KeyPrefixMerge) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(10, 17);
  // Insert a value that does not share full prefix with the current Node48
  verifier.insert(0x2010, test_values[1]);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({18, (1 + ci4), 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  // And delete it, so that upper level Node4 key prefix gets merged with
  // Node48 one
  unodb::test::must_not_allocate([&verifier] { verifier.remove(0x2010); });

  verifier.check_present_values();
  verifier.check_absent_keys({9, 0x2010, 28});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes(
      {1 + olc_chain_collapse<TypeParam>, 0, 0, 0});
  verifier.assert_node_counts(
      {17, ci4 - olc_chain_collapse<TypeParam>, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node256DeleteBeginningMiddleEnd) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 256);

  unodb::test::must_not_allocate([&verifier] {
    verifier.remove(180);
    verifier.remove(1);
    verifier.remove(256);
  });

  verifier.check_present_values();
  verifier.check_absent_keys({0, 1, 180, 256});

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  constexpr auto oc = olc_chain_collapse<TypeParam>;
  verifier.assert_node_counts({253, ci4 - oc, 0, 0, 1});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node256ShrinkToNode48DeleteMiddle) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 49);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({49, 0, 0, 0, 1});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.remove(25);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({0, 0, 0, 1});
  verifier.assert_node_counts({48, 0, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.check_present_values();
  verifier.check_absent_keys({0, 25, 50});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node256ShrinkToNode48DeleteBeginning) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 49);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({49, 0, 0, 0, 1});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.remove(1);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({0, 0, 0, 1});
  verifier.assert_node_counts({48, 0, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.check_present_values();
  verifier.check_absent_keys({0, 1, 50});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node256ShrinkToNode48DeleteEnd) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(1, 49);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({49, 0, 0, 0, 1});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.remove(49);

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes({0, 0, 0, 1});
  verifier.assert_node_counts({48, 0, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS

  verifier.check_present_values();
  verifier.check_absent_keys({0, 49, 50});
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Node256KeyPrefixMerge) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert_key_range(10, 49);
  // Insert a value that does not share full prefix with the current Node256
  verifier.insert(0x2010, test_values[1]);

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({50, (1 + ci4), 0, 0, 1});
#endif  // UNODB_DETAIL_WITH_STATS

  // And delete it, so that upper level Node4 key prefix gets merged with
  // Node256 one
  unodb::test::must_not_allocate([&verifier] { verifier.remove(0x2010); });

  verifier.check_present_values();
  verifier.check_absent_keys({9, 0x2010, 60});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_shrinking_inodes(
      {1 + olc_chain_collapse<TypeParam>, 0, 0, 0});
  verifier.assert_node_counts(
      {49, ci4 - olc_chain_collapse<TypeParam>, 0, 0, 1});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, MissingKeyWithPresentPrefix) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(0x010000, test_values[0]);
  verifier.insert(0x000001, test_values[1]);
  verifier.insert(0x010001, test_values[2]);

  unodb::test::must_not_allocate([&verifier] {
    verifier.attempt_remove_missing_keys({0x000002, 0x010100, 0x010002});
  });
}

UNODB_TYPED_TEST(ARTCorrectnessTest, MissingKeyMatchingInodePath) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(0x0100, test_values[0]);
  verifier.insert(0x0200, test_values[1]);

  unodb::test::must_not_allocate(
      [&verifier] { verifier.attempt_remove_missing_keys({0x0101, 0x0202}); });
}

#ifdef UNODB_DETAIL_WITH_STATS

UNODB_TYPED_TEST(ARTCorrectnessTest, MemoryAccountingDuplicateKeyInsert) {
  unodb::test::tree_verifier<TypeParam> verifier;
  verifier.insert(0, test_values[0]);
  unodb::test::must_not_allocate([&verifier] {
    UNODB_ASSERT_FALSE(verifier.try_insert(0, test_values[1]));
  });
  verifier.remove(0);
  UNODB_ASSERT_EQ(verifier.get_db().get_current_memory_use(), 0);
}

#endif  // UNODB_DETAIL_WITH_STATS

UNODB_TYPED_TEST(ARTCorrectnessTest, Node48InsertIntoDeletedSlot) {
  unodb::test::tree_verifier<TypeParam> verifier;
  verifier.insert(16865361447928765957ULL, test_values[0]);
  verifier.insert(7551546784238320931ULL, test_values[1]);
  verifier.insert(10913915230368519832ULL, test_values[2]);
  verifier.insert(3754602112003529886ULL, test_values[3]);
  verifier.insert(15202487832924025715ULL, test_values[4]);
  verifier.insert(501264303707694295ULL, test_values[0]);
  verifier.insert(9228847637821057196ULL, test_values[1]);
  verifier.insert(4772373217231458680ULL, test_values[2]);
  verifier.insert(10396278540561456315ULL, test_values[3]);
  verifier.insert(16646085826334346534ULL, test_values[4]);
  verifier.insert(3854084731240466350ULL, test_values[0]);
  verifier.insert(12957550352669724359ULL, test_values[1]);
  verifier.insert(6583227679421302512ULL, test_values[2]);
  verifier.insert(6829398721825682578ULL, test_values[3]);
  verifier.insert(11455392605080430684ULL, test_values[4]);
  verifier.insert(10176313584012002900ULL, test_values[0]);
  verifier.insert(13700634388772836888ULL, test_values[1]);
  verifier.insert(17872125209760305988ULL, test_values[2]);

  unodb::test::must_not_allocate(
      [&verifier] { verifier.remove(6583227679421302512ULL); });
  verifier.insert(0, test_values[0]);

  verifier.check_present_values();

#ifdef UNODB_DETAIL_WITH_STATS
  constexpr auto ci4 = chain_i4_per_key<TypeParam>;
  verifier.assert_node_counts({18, 18 * ci4, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, ClearOnEmpty) {
  unodb::test::tree_verifier<TypeParam> verifier;

  unodb::test::must_not_allocate([&verifier] { verifier.clear(); });

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({0, 0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, Clear) {
  unodb::test::tree_verifier<TypeParam> verifier;

  verifier.insert(1, test_values[0]);

  unodb::test::must_not_allocate([&verifier] { verifier.clear(); });

  verifier.check_absent_keys({1});

#ifdef UNODB_DETAIL_WITH_STATS
  verifier.assert_node_counts({0, 0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
}

UNODB_TYPED_TEST(ARTCorrectnessTest, TwoInstances) {
  unodb::test::tree_verifier<TypeParam> v1;
  unodb::test::tree_verifier<TypeParam> v2;

  unodb::test::thread<TypeParam> second_thread([&v2] {
    thread_syncs[0].notify();
    thread_syncs[1].wait();

    v2.insert(0, test_values[0]);
    v2.check_present_values();
  });

  thread_syncs[0].wait();
  thread_syncs[1].notify();

  v1.insert(0, test_values[1]);
  v1.check_present_values();

  second_thread.join();
}

/// Check that \a db maps key 1 to \a expected, then set \a ran_on, which only
/// a passed check reaches.
// Not noexcept: under --gtest_throw_on_failure a failed check throws.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
void check_key_1(const unodb::test::u64_db& db, unodb::value_view expected,
                 bool& ran_on) {
  ASSERT_VALUE_FOR_KEY(unodb::test::u64_db, db, 1, expected);
  ran_on = true;
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// A passed ASSERT_VALUE_FOR_KEY lets its caller run on.
UNODB_TEST(AssertValueForKey, PassedCheckLetsCallerRunOn) {
  unodb::test::u64_db db;
  UNODB_ASSERT_TRUE(db.insert(1, test_values[0]));
  bool ran_on = false;
  check_key_1(db, test_values[0], ran_on);
  UNODB_ASSERT_TRUE(ran_on);
}

/// Check that check_key_1(\a db, \a expected), whose check fails, ends without
/// setting its flag and reports two fatal failures, which it intercepts into
/// \a failures.
void assert_check_key_1_ends_caller(const unodb::test::u64_db& db,
                                    unodb::value_view expected,
                                    testing::TestPartResultArray& failures) {
  bool ran_on = false;
  {
    const unodb::test::intercepted_failure_guard guard;
    const testing::ScopedFakeTestPartResultReporter reporter{
        testing::ScopedFakeTestPartResultReporter::
            INTERCEPT_ONLY_CURRENT_THREAD,
        &failures};
    check_key_1(db, expected, ran_on);
  }
  UNODB_ASSERT_FALSE(ran_on);
  // The failed check, then ASSERT_NO_FATAL_FAILURE's own failure.
  UNODB_ASSERT_EQ(failures.size(), 2);
  UNODB_ASSERT_TRUE(failures.GetTestPartResult(0).fatally_failed());
  UNODB_ASSERT_TRUE(failures.GetTestPartResult(1).fatally_failed());
}

/// An ASSERT_VALUE_FOR_KEY that finds the wrong value ends its caller.
UNODB_TEST(AssertValueForKey, WrongValueEndsCaller) {
  unodb::test::u64_db db;
  UNODB_ASSERT_TRUE(db.insert(1, test_values[0]));
  testing::TestPartResultArray failures;
  assert_check_key_1_ends_caller(db, test_values[1], failures);
}

/// An ASSERT_VALUE_FOR_KEY that does not find the key ends its caller, and its
/// failure message carries the tree dump.
UNODB_TEST(AssertValueForKey, MissEndsCaller) {
  unodb::test::u64_db db;
  UNODB_ASSERT_TRUE(db.insert(2, test_values[0]));
  testing::TestPartResultArray failures;
  UNODB_ASSERT_NO_FATAL_FAILURE(
      assert_check_key_1_ends_caller(db, test_values[0], failures));
  const auto* const message = failures.GetTestPartResult(0).message();
  UNODB_ASSERT_THAT(message, ::testing::HasSubstr("db.get did not find"));
  UNODB_ASSERT_THAT(message, ::testing::HasSubstr("db dump"));
}

/// Google Test's `break_on_failure` and `throw_on_failure` flags, in that
/// order.
using gtest_failure_flags = std::pair<bool, bool>;

/// Read both flags.
[[nodiscard]] gtest_failure_flags get_gtest_failure_flags() noexcept {
  return {GTEST_FLAG_GET(break_on_failure), GTEST_FLAG_GET(throw_on_failure)};
}

/// Set both flags to \a flags.
void set_gtest_failure_flags(gtest_failure_flags flags) noexcept {
  GTEST_FLAG_SET(break_on_failure, flags.first);
  GTEST_FLAG_SET(throw_on_failure, flags.second);
}

/// Set the flags to \a flags and return them as read inside an
/// intercepted_failure_guard's scope and then after it.
[[nodiscard]] std::pair<gtest_failure_flags, gtest_failure_flags>
flags_inside_and_after_guard(gtest_failure_flags flags) noexcept {
  set_gtest_failure_flags(flags);
  gtest_failure_flags inside{};
  {
    const unodb::test::intercepted_failure_guard guard;
    inside = get_gtest_failure_flags();
  }
  return {inside, get_gtest_failure_flags()};
}

/// intercepted_failure_guard clears both flags for its lifetime and then
/// restores each from its own saved value.  Each flag is set alone, as the
/// test_art_intercepted_failure_* reruns set them, so that a guard that
/// restores a flag from the other's value fails too.  The flags are restored
/// before anything is asserted, so that a broken guard fails this test instead
/// of trapping or throwing in it.
UNODB_TEST(InterceptedFailureGuard, ClearsAndRestoresFlags) {
  const auto saved = get_gtest_failure_flags();
  const auto break_alone = flags_inside_and_after_guard({true, false});
  const auto throw_alone = flags_inside_and_after_guard({false, true});
  set_gtest_failure_flags(saved);
  constexpr gtest_failure_flags cleared{false, false};
  UNODB_ASSERT_EQ(break_alone.first, cleared);
  UNODB_ASSERT_EQ(break_alone.second, (gtest_failure_flags{true, false}));
  UNODB_ASSERT_EQ(throw_alone.first, cleared);
  UNODB_ASSERT_EQ(throw_alone.second, (gtest_failure_flags{false, true}));
}

}  // namespace
