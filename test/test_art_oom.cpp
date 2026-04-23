// Copyright 2022-2025 UnoDB contributors

#ifndef NDEBUG

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

// IWYU pragma: no_include <__new/exceptions.h>
// IWYU pragma: no_include <array>
// IWYU pragma: no_include <string>

#include <cstdint>
#include <new>  // IWYU pragma: keep

#include <gtest/gtest.h>

#include "art_common.hpp"
#include "db_test_utils.hpp"
#include "gtest_utils.hpp"
#include "test_heap.hpp"

// The OOM tests are dependent on the number of heap allocations in the test,
// that's brittle and hardcoded. Suppose some op takes 5 heap allocations. The
// tests is written in that it knows that the test should fail on OOMs injected
// on the 1st-5th allocation and pass on the 6th one. The allocations done by
// libstdc++ are included.
//
// Changing the data structure in the main code or the test suite might perturb
// this, causing tests to fail. If this happens you need to decide whether the
// change in behavior was for a valid reason or not. If tests fail in that
// "expected exception was not thrown", try incrementing the allocation counter
// in the test. If they fail in that "exception was thrown but we weren't
// expecting it", try decrementing it.
//
// TODO(laurynas) OOM tests for the scan API.
namespace {

template <class TypeParam, typename Init, typename Test, typename CheckAfterOOM,
          typename CheckAfterSuccess>
void oom_test(unsigned fail_limit, Init init, Test test,
              CheckAfterOOM check_after_oom,
              CheckAfterSuccess check_after_success) {
  unsigned fail_n;
  for (fail_n = 1; fail_n < fail_limit; ++fail_n) {
    unodb::test::tree_verifier<TypeParam> verifier;
    init(verifier);

    unodb::test::allocation_failure_injector::fail_on_nth_allocation(fail_n);
    UNODB_ASSERT_THROW(test(verifier), std::bad_alloc);
    unodb::test::allocation_failure_injector::reset();

    verifier.check_present_values();
    check_after_oom(verifier);
  }

  unodb::test::tree_verifier<TypeParam> verifier;
  init(verifier);

  unodb::test::allocation_failure_injector::fail_on_nth_allocation(fail_n);
  test(verifier);
  unodb::test::allocation_failure_injector::reset();

  verifier.check_present_values();
  check_after_success(verifier);
}

template <class TypeParam, typename Init, typename CheckAfterSuccess>
void oom_insert_test(unsigned fail_limit, Init init, std::uint64_t k,
                     unodb::value_view v,
                     CheckAfterSuccess check_after_success) {
  oom_test<TypeParam>(
      fail_limit, init,
      [k, v](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(k, v);
      },
      [k](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.check_absent_keys({k});
      },
      check_after_success);
}

template <class TypeParam, typename Init, typename CheckAfterSuccess>
void oom_remove_test(unsigned fail_limit, Init init, std::uint64_t k,
                     CheckAfterSuccess check_after_success) {
  oom_test<TypeParam>(
      fail_limit, init,
      [k](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.remove(k);
      },
      [](unodb::test::tree_verifier<TypeParam>&) {},
      [k,
       check_after_success](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.check_absent_keys({k});
        check_after_success(verifier);
      });
}

template <class Db>
class ARTOOMTest : public ::testing::Test {
 public:
  using Test::Test;
};

using ARTTypes =
    ::testing::Types<unodb::test::u64_db, unodb::test::u64_mutex_db,
                     unodb::test::u64_olc_db>;

UNODB_TYPED_TEST_SUITE(ARTOOMTest, ARTTypes)

UNODB_TYPED_TEST(ARTOOMTest, CtorDoesNotAllocate) {
  unodb::test::allocation_failure_injector::fail_on_nth_allocation(1);
  const TypeParam tree;
  unodb::test::allocation_failure_injector::reset();
}

UNODB_TYPED_TEST(ARTOOMTest, SingleNodeTreeEmptyValue) {
  oom_insert_test<TypeParam>(
      2,
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({0, 0, 0, 0, 0});
        verifier.assert_growing_inodes({0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      },
      1, {},
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({1, 0, 0, 0, 0});
        verifier.assert_growing_inodes({0, 0, 0, 0});
#endif
      });
}

UNODB_TYPED_TEST(ARTOOMTest, SingleNodeTreeNonemptyValue) {
  oom_insert_test<TypeParam>(
      2,
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({0, 0, 0, 0, 0});
        verifier.assert_growing_inodes({0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      },
      1, unodb::test::test_values[2],
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({1, 0, 0, 0, 0});
        verifier.assert_growing_inodes({0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, ExpandLeafToNode4) {
  oom_insert_test<TypeParam>(
      3,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(0, unodb::test::test_values[1]);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({1, 0, 0, 0, 0});
        verifier.assert_growing_inodes({0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      },
      1, unodb::test::test_values[2],
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({2, 1, 0, 0, 0});
        verifier.assert_growing_inodes({1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, TwoNode4) {
  oom_insert_test<TypeParam>(
      3,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(1, unodb::test::test_values[0]);
        verifier.insert(3, unodb::test::test_values[2]);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_growing_inodes({1, 0, 0, 0});
        verifier.assert_node_counts({2, 1, 0, 0, 0});
        verifier.assert_key_prefix_splits(0);
#endif  // UNODB_DETAIL_WITH_STATS
      },
      // Insert a value that does not share full prefix with the current Node4
      0xFF01, unodb::test::test_values[3],
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({3, 2, 0, 0, 0});
        verifier.assert_growing_inodes({2, 0, 0, 0});
        verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, DbInsertNodeRecursion) {
  oom_insert_test<TypeParam>(
      3,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(1, unodb::test::test_values[0]);
        verifier.insert(3, unodb::test::test_values[2]);
        // Insert a value that does not share full prefix with the current Node4
        verifier.insert(0xFF0001, unodb::test::test_values[3]);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({3, 2, 0, 0, 0});
        verifier.assert_growing_inodes({2, 0, 0, 0});
        verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS
      },
      // Then insert a value that shares full prefix with the above node and
      // will ask for a recursive insertion there
      0xFF0101, unodb::test::test_values[1],
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({4, 3, 0, 0, 0});
        verifier.assert_growing_inodes({3, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, Node16) {
  oom_insert_test<TypeParam>(
      3,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert_key_range(0, 4);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({4, 1, 0, 0, 0});
        verifier.assert_growing_inodes({1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      },
      5, unodb::test::test_values[0],
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({5, 0, 1, 0, 0});
        verifier.assert_growing_inodes({1, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, Node16KeyPrefixSplit) {
  oom_insert_test<TypeParam>(
      3,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert_key_range(10, 5);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({5, 0, 1, 0, 0});
        verifier.assert_growing_inodes({1, 1, 0, 0});
        verifier.assert_key_prefix_splits(0);
#endif  // UNODB_DETAIL_WITH_STATS
      },
      // Insert a value that does share full prefix with the current Node16
      0x1020, unodb::test::test_values[0],
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({6, 1, 1, 0, 0});
        verifier.assert_growing_inodes({2, 1, 0, 0});
        verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, Node48) {
  oom_insert_test<TypeParam>(
      3,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert_key_range(0, 16);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({16, 0, 1, 0, 0});
        verifier.assert_growing_inodes({1, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      },
      16, unodb::test::test_values[0],
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({17, 0, 0, 1, 0});
        verifier.assert_growing_inodes({1, 1, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, Node48KeyPrefixSplit) {
  oom_insert_test<TypeParam>(
      3,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert_key_range(10, 17);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({17, 0, 0, 1, 0});
        verifier.assert_growing_inodes({1, 1, 1, 0});
        verifier.assert_key_prefix_splits(0);
#endif  // UNODB_DETAIL_WITH_STATS
      },
      // Insert a value that does share full prefix with the current Node48
      0x100020, unodb::test::test_values[0],
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({18, 1, 0, 1, 0});
        verifier.assert_growing_inodes({2, 1, 1, 0});
        verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, Node256) {
  oom_insert_test<TypeParam>(
      3,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert_key_range(0, 48);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({48, 0, 0, 1, 0});
        verifier.assert_growing_inodes({1, 1, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      },
      49, unodb::test::test_values[0],
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({49, 0, 0, 0, 1});
        verifier.assert_growing_inodes({1, 1, 1, 1});
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, Node256KeyPrefixSplit) {
  oom_insert_test<TypeParam>(
      3,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert_key_range(20, 49);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({49, 0, 0, 0, 1});
        verifier.assert_growing_inodes({1, 1, 1, 1});
        verifier.assert_key_prefix_splits(0);
#endif  // UNODB_DETAIL_WITH_STATS
      },
      // Insert a value that does share full prefix with the current Node48
      0x100020, unodb::test::test_values[0],
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({50, 1, 0, 0, 1});
        verifier.assert_growing_inodes({2, 1, 1, 1});
        verifier.assert_key_prefix_splits(1);
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, Node16ShrinkToNode4) {
  oom_remove_test<TypeParam>(
      2,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert_key_range(1, 5);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({5, 0, 1, 0, 0});
        verifier.assert_shrinking_inodes({0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      },
      2,
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_shrinking_inodes({0, 1, 0, 0});
        verifier.assert_node_counts({4, 1, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, Node48ShrinkToNode16) {
  oom_remove_test<TypeParam>(
      2,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert_key_range(0x80, 17);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({17, 0, 0, 1, 0});
        verifier.assert_shrinking_inodes({0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      },
      0x85,
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_shrinking_inodes({0, 0, 1, 0});
        verifier.assert_node_counts({16, 0, 1, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

UNODB_TYPED_TEST(ARTOOMTest, Node256ShrinkToNode48) {
  oom_remove_test<TypeParam>(
      2,
      [](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert_key_range(1, 49);
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_node_counts({49, 0, 0, 0, 1});
        verifier.assert_shrinking_inodes({0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      },
      25,
      [](unodb::test::tree_verifier<TypeParam>&
#ifdef UNODB_DETAIL_WITH_STATS
             verifier
#endif  // UNODB_DETAIL_WITH_STATS
      ) {
#ifdef UNODB_DETAIL_WITH_STATS
        verifier.assert_shrinking_inodes({0, 0, 0, 1});
        verifier.assert_node_counts({48, 0, 0, 1, 0});
#endif  // UNODB_DETAIL_WITH_STATS
      });
}

// ===================================================================
// key_view OOM tests: exercise build_chain allocation failure paths.
// build_chain is only invoked when full_key_in_inode_path is true
// (i.e., Key = key_view) and the key is long enough to need chain I4
// nodes beyond the dispatch byte.
//
// Allocation counts differ between VIS (no leaf allocation) and leaf-based
// paths.  VIS packs the value into the child slot; leaf-based allocates a
// leaf node.  The fail_limit must be exactly (allocations needed + 1).
//
// Nonfull: VIS = chain I4 only (1 alloc, limit 2)
//          Leaf = leaf + chain I4 (2 allocs, limit 3)
// Grow:    VIS = I16 create + chain I4 (2 allocs, limit 3)
//          Leaf = leaf + I16 create + chain I4 (3 allocs, limit 4)
// Prefix split: VIS = I4 create + chain I4 (2 allocs, limit 3)
//               Leaf = leaf + I4 create + chain I4 (3 allocs, limit 4)
template <class Db>
constexpr unsigned chain_oom_limit(unsigned vis_allocs) {
  if constexpr (std::is_same_v<typename Db::value_type, unodb::value_view>)
    return vis_allocs + 2;  // +1 for leaf, +1 for success iteration
  else
    return vis_allocs + 1;  // +1 for success iteration
}
// ===================================================================

template <class Db>
class ARTKeyViewOOMTest : public ::testing::Test {
 public:
  using Test::Test;
};

using ARTKeyViewTypes =
    ::testing::Types<unodb::test::key_view_u64val_db, unodb::test::key_view_db,
                     unodb::test::key_view_u64val_olc_db,
                     unodb::test::key_view_olc_db>;

UNODB_TYPED_TEST_SUITE(ARTKeyViewOOMTest, ARTKeyViewTypes)

// Insert 3 short (1-byte) keys into I4, then insert a long (9-byte) key.
// The I4 has room (nonfull path). build_chain allocates chain I4 node(s).
// OOM during build_chain must leave tree consistent.
UNODB_TYPED_TEST(ARTKeyViewOOMTest, BuildChainNonfull) {
  const auto v = unodb::test::get_test_value<TypeParam>(0);
  const auto v_long = unodb::test::get_test_value<TypeParam>(1);

  unodb::key_encoder enc1;
  unodb::key_encoder enc2;
  unodb::key_encoder enc3;
  unodb::key_encoder enc_long;
  const auto short1 = enc1.encode(std::uint8_t{1}).get_key_view();
  const auto short2 = enc2.encode(std::uint8_t{2}).get_key_view();
  const auto short3 = enc3.encode(std::uint8_t{3}).get_key_view();
  const auto long_key = enc_long.encode(std::uint8_t{0x10})
                            .encode(std::uint64_t{1})
                            .get_key_view();

  oom_test<TypeParam>(
      chain_oom_limit<TypeParam>(1),
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(short1, v);
        verifier.insert(short2, v);
        verifier.insert(short3, v);
      },
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(long_key, v_long);
      },
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        // After OOM, the tree should be unchanged — the long key should
        // not be present, and the tree should be fully consistent.
        // With the current bug, the bare child is left in the tree,
        // corrupting memory accounting.
        UNODB_ASSERT_FALSE(verifier.get_db().get(long_key).has_value());
      },
      [](unodb::test::tree_verifier<TypeParam>&) {});
}

// Insert 4 short (1-byte) keys filling I4, then insert a long (9-byte) key.
// Triggers I4→I16 grow, then build_chain on the new child slot.
// OOM during build_chain must leave tree consistent.
UNODB_TYPED_TEST(ARTKeyViewOOMTest, BuildChainGrow) {
  const auto v = unodb::test::get_test_value<TypeParam>(0);
  const auto v_long = unodb::test::get_test_value<TypeParam>(1);

  unodb::key_encoder enc1;
  unodb::key_encoder enc2;
  unodb::key_encoder enc3;
  unodb::key_encoder enc4;
  unodb::key_encoder enc_long;
  const auto short1 = enc1.encode(std::uint8_t{1}).get_key_view();
  const auto short2 = enc2.encode(std::uint8_t{2}).get_key_view();
  const auto short3 = enc3.encode(std::uint8_t{3}).get_key_view();
  const auto short4 = enc4.encode(std::uint8_t{4}).get_key_view();
  const auto long_key = enc_long.encode(std::uint8_t{0x10})
                            .encode(std::uint64_t{1})
                            .get_key_view();

  oom_test<TypeParam>(
      chain_oom_limit<TypeParam>(2),
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(short1, v);
        verifier.insert(short2, v);
        verifier.insert(short3, v);
        verifier.insert(short4, v);
      },
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(long_key, v_long);
      },
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        UNODB_ASSERT_FALSE(verifier.get_db().get(long_key).has_value());
      },
      [](unodb::test::tree_verifier<TypeParam>&) {});
}

// Insert one 9-byte key, then insert another 9-byte key that diverges within
// the chain prefix. Triggers prefix split → new I4 → build_chain.
// OOM during build_chain must leave tree consistent.
//
// key1: 0x42 0x00 ... 0x01 (tag + uint64{1})
// key2: 0x42 0x80 ... 0x01 (tag + uint64 with high bit set in first byte)
// They share only the tag byte; diverge at byte 1 (within the chain prefix).
UNODB_TYPED_TEST(ARTKeyViewOOMTest, BuildChainPrefixSplit) {
  const auto v1 = unodb::test::get_test_value<TypeParam>(0);
  const auto v2 = unodb::test::get_test_value<TypeParam>(1);

  unodb::key_encoder enc1;
  unodb::key_encoder enc2;
  const auto key1 =
      enc1.encode(std::uint8_t{0x42}).encode(std::uint64_t{1}).get_key_view();
  // uint64 value with high bit set → first encoded byte is 0x80, diverges
  // at byte 1 from key1's 0x00.
  const auto key2 = enc2.encode(std::uint8_t{0x42})
                        .encode(std::uint64_t{0x8000000000000001ULL})
                        .get_key_view();

  oom_test<TypeParam>(
      chain_oom_limit<TypeParam>(2),
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(key1, v1);
      },
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(key2, v2);
      },
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        UNODB_ASSERT_FALSE(verifier.get_db().get(key2).has_value());
      },
      [](unodb::test::tree_verifier<TypeParam>&) {});
}

// Multi-node chain: key long enough to produce 2 chain I4 nodes.
// Exercises the build_chain loop (owns_current=true) cleanup path.
// Encoded key: uint8{0x10} + uint64{1} + uint64{2} = 17 bytes.
// Chain starts at depth 1 → 16 bytes → 2 chain I4 nodes.
UNODB_TYPED_TEST(ARTKeyViewOOMTest, BuildChainMultiNode) {
  const auto v = unodb::test::get_test_value<TypeParam>(0);
  const auto v_long = unodb::test::get_test_value<TypeParam>(1);

  unodb::key_encoder enc1;
  unodb::key_encoder enc2;
  unodb::key_encoder enc3;
  unodb::key_encoder enc_long;
  const auto short1 = enc1.encode(std::uint8_t{1}).get_key_view();
  const auto short2 = enc2.encode(std::uint8_t{2}).get_key_view();
  const auto short3 = enc3.encode(std::uint8_t{3}).get_key_view();
  const auto long_key = enc_long.encode(std::uint8_t{0x10})
                            .encode(std::uint64_t{1})
                            .encode(std::uint64_t{2})
                            .get_key_view();

  oom_test<TypeParam>(
      chain_oom_limit<TypeParam>(2),
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(short1, v);
        verifier.insert(short2, v);
        verifier.insert(short3, v);
      },
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        verifier.insert(long_key, v_long);
      },
      [&](unodb::test::tree_verifier<TypeParam>& verifier) {
        UNODB_ASSERT_FALSE(verifier.get_db().get(long_key).has_value());
      },
      [](unodb::test::tree_verifier<TypeParam>&) {});
}

}  // namespace

#endif  // #ifndef NDEBUG
