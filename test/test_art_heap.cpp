// Copyright 2026 UnoDB contributors
/// \file test_art_heap.cpp
/// \brief Tests for the TupleHeap secondary index variant of olc_db.
///
/// Uses a trivial TestHeap that stores pre-encoded keys in a flat vector.
/// This exercises the full heap-mode code path: insert with divergence,
/// get with key verification, remove with key verification, and scan.

#include "global.hpp"  // NOLINT(misc-include-cleaner)

UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26409)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26426)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26432)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26436)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26447)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26818)

#include <tuple>

#include "gtest/gtest.h"
#include "olc_art.hpp"
#include "qsbr.hpp"
#include "qsbr_test_utils.hpp"
#include "test_tuple_heap.hpp"

namespace {

using unodb::test::encode_u64;
using unodb::test::TestHeap;

/// Alias for the heap-backed tree type.
using heap_db = unodb::olc_db<unodb::key_view, std::uint64_t, TestHeap>;

/// Test fixture providing QSBR context.
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class HeapArtTest : public ::testing::Test {
 protected:
  HeapArtTest() noexcept { unodb::test::expect_idle_qsbr(); }

  ~HeapArtTest() noexcept override {
    unodb::this_thread().quiescent();
    unodb::test::expect_idle_qsbr();
  }
};

TEST_F(HeapArtTest, ConstructDestruct) {
  const TestHeap heap;
  const heap_db db{heap};
  EXPECT_TRUE(db.empty());
}

TEST_F(HeapArtTest, InsertGetSingle) {
  TestHeap heap;
  const auto key_bytes = encode_u64(42);
  const unodb::key_view key{key_bytes.data(), key_bytes.size()};
  heap.add_tuple(1, key);

  heap_db db{heap};
  ASSERT_TRUE(db.insert(key, 1));
  EXPECT_FALSE(db.empty());

  const auto result = db.get(key);
  ASSERT_TRUE(result.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(result.value(), 1U);
}

TEST_F(HeapArtTest, InsertDuplicate) {
  TestHeap heap;
  const auto key_bytes = encode_u64(100);
  const unodb::key_view key{key_bytes.data(), key_bytes.size()};
  heap.add_tuple(1, key);
  heap.add_tuple(2, key);

  heap_db db{heap};
  ASSERT_TRUE(db.insert(key, 1));
  EXPECT_FALSE(db.insert(key, 2));
}

TEST_F(HeapArtTest, InsertRemove) {
  TestHeap heap;
  const auto key_bytes = encode_u64(7);
  const unodb::key_view key{key_bytes.data(), key_bytes.size()};
  heap.add_tuple(1, key);

  heap_db db{heap};
  ASSERT_TRUE(db.insert(key, 1));
  ASSERT_TRUE(db.remove(key));
  EXPECT_TRUE(db.empty());
  EXPECT_FALSE(db.get(key).has_value());
}

TEST_F(HeapArtTest, InsertDivergence) {
  // Two keys that share a 6-byte prefix but diverge at byte 7.
  TestHeap heap;
  std::array<std::byte, 8> key_a{};
  std::array<std::byte, 8> key_b{};
  // prefix: 0x01 0x02 0x03 0x04 0x05 0x06, then diverge
  for (unsigned i = 0; i < 6; ++i) {
    key_a[i] = static_cast<std::byte>(i + 1U);
    key_b[i] = static_cast<std::byte>(i + 1U);
  }
  key_a[6] = std::byte{0xAA};
  key_a[7] = std::byte{0x01};
  key_b[6] = std::byte{0xBB};
  key_b[7] = std::byte{0x02};

  heap.add_tuple(10, key_a);
  heap.add_tuple(20, key_b);

  heap_db db{heap};
  const unodb::key_view kv_a{key_a.data(), key_a.size()};
  const unodb::key_view kv_b{key_b.data(), key_b.size()};

  ASSERT_TRUE(db.insert(kv_a, 10));
  ASSERT_TRUE(db.insert(kv_b, 20));

  const auto r_a = db.get(kv_a);
  ASSERT_TRUE(r_a.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(r_a.value(), 10U);

  const auto r_b = db.get(kv_b);
  ASSERT_TRUE(r_b.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(r_b.value(), 20U);
}

TEST_F(HeapArtTest, ScanOrder) {
  TestHeap heap;
  // Insert keys in random order, scan should yield them in sorted order.
  const std::vector<std::uint64_t> values{5, 2, 8, 1, 9, 3, 7, 4, 6, 0};
  std::vector<std::array<std::byte, 8>> key_store;
  key_store.reserve(values.size());

  heap_db db{heap};
  for (auto v : values) {
    key_store.push_back(encode_u64(v));
    const unodb::key_view kv{key_store.back().data(), key_store.back().size()};
    heap.add_tuple(v, kv);
    ASSERT_TRUE(db.insert(kv, v));
  }

  // Scan and collect results.
  std::vector<std::uint64_t> scan_results;
  db.scan([&scan_results](auto visitor) {
    scan_results.push_back(visitor.get_value());
    return false;  // continue
  });

  // Should be in ascending key order (big-endian encoding preserves order).
  const std::vector<std::uint64_t> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_EQ(scan_results, expected);
}

TEST_F(HeapArtTest, ManyInserts) {
  TestHeap heap;
  heap_db db{heap};

  constexpr std::uint64_t count = 1000;
  std::vector<std::array<std::byte, 8>> key_store;
  key_store.reserve(count);

  for (std::uint64_t i = 0; i < count; ++i) {
    key_store.push_back(encode_u64(i));
    const unodb::key_view kv{key_store.back().data(), key_store.back().size()};
    heap.add_tuple(i, kv);
    ASSERT_TRUE(db.insert(kv, i));
  }

  // Verify all retrievable.
  for (std::uint64_t i = 0; i < count; ++i) {
    const unodb::key_view kv{key_store[i].data(), key_store[i].size()};
    const auto result = db.get(kv);
    ASSERT_TRUE(result.has_value()) << "Missing key for i=" << i;
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(result.value(), i);
  }

  // Remove all.
  for (std::uint64_t i = 0; i < count; ++i) {
    const unodb::key_view kv{key_store[i].data(), key_store[i].size()};
    ASSERT_TRUE(db.remove(kv)) << "Failed to remove i=" << i;
  }
  EXPECT_TRUE(db.empty());
}

TEST_F(HeapArtTest, ConcurrentInsertGet) {
  // Multiple threads insert and get concurrently on a heap-backed tree.
  TestHeap heap;
  heap_db db{heap};

  constexpr std::uint64_t keys_per_thread = 200;
  constexpr unsigned num_threads = 4;
  constexpr std::uint64_t total_keys = keys_per_thread * num_threads;

  // Pre-populate the heap with all keys (heap must be thread-safe for reads).
  std::vector<std::array<std::byte, 8>> key_store;
  key_store.reserve(total_keys);
  for (std::uint64_t i = 0; i < total_keys; ++i) {
    key_store.push_back(encode_u64(i));
    heap.add_tuple(i, key_store.back());
  }

  // Each thread inserts its range, then verifies all inserted keys.
  auto worker = [&](unsigned thread_id) {
    const auto start = thread_id * keys_per_thread;
    const auto end = start + keys_per_thread;

    // Insert phase.
    for (std::uint64_t i = start; i < end; ++i) {
      const unodb::key_view kv{key_store[i].data(), key_store[i].size()};
      std::ignore = db.insert(kv, i);  // May race; ignore result.
      unodb::this_thread().quiescent();
    }

    // Get phase — verify own keys are present.
    for (std::uint64_t i = start; i < end; ++i) {
      const unodb::key_view kv{key_store[i].data(), key_store[i].size()};
      const auto result = db.get(kv);
      if (result.has_value()) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        EXPECT_EQ(result.value(), i);
      }
      // Key might be missing if another thread hasn't finished; that's ok
      // for this stress test. We just verify no crashes.
      unodb::this_thread().quiescent();
    }
  };

  // Launch threads.
  std::vector<unodb::qsbr_thread> threads;
  threads.reserve(num_threads);
  for (unsigned t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& t : threads) {
    t.join();
  }

  // Final verification: all keys should be present.
  for (std::uint64_t i = 0; i < total_keys; ++i) {
    const unodb::key_view kv{key_store[i].data(), key_store[i].size()};
    const auto result = db.get(kv);
    ASSERT_TRUE(result.has_value())
        << "Missing key after concurrent test i=" << i;
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(result.value(), i);
  }
}

}  // namespace

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
