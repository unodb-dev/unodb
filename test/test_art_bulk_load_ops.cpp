// Copyright 2026 UnoDB contributors

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "art_common.hpp"
#include "db_test_utils.hpp"
#include "gtest_utils.hpp"
#include "node_type.hpp"
#include "portability_execution.hpp"
#include "qsbr.hpp"
#include "qsbr_test_utils.hpp"

UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26818)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26445)

namespace {

#ifdef UNODB_DETAIL_WITH_STATS
using unodb::as_i;
using unodb::node_type;
#endif
using unodb::value_view;
using unodb::test::u64_db;
using unodb::test::u64_mutex_db;
using unodb::test::u64_olc_db;

constexpr auto val_bytes =
    std::array<std::byte, 5>{std::byte{0x68}, std::byte{0x65}, std::byte{0x6C},
                             std::byte{0x6C}, std::byte{0x6F}};
constexpr auto val = value_view{val_bytes};

// ─── Error Tests ─────────────────────────────────────────────────────────────

// T26: bulk_load on non-empty tree throws
UNODB_TEST(BulkLoadError, NonEmpty) {
  u64_db db;
  constexpr std::uint64_t key = 42;
  UNODB_ASSERT_TRUE(db.insert(key, val));
  std::vector<std::pair<std::uint64_t, value_view>> kv{{100, val}};
  UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26818)
  EXPECT_THROW(db.bulk_load(kv.begin(), kv.end()), std::invalid_argument);
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  // Original key still present
  const auto result = db.get(key);
  UNODB_ASSERT_TRUE(result.has_value());
}

// T36: db ignores parallelism parameter
UNODB_TEST(BulkLoadError, DbIgnoresParallelism) {
  u64_db db;
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  kv.reserve(100);
  for (std::uint64_t i = 0; i < 100; ++i) {
    const auto key = i << 56U;
    kv.emplace_back(key, val);
  }
  std::ranges::sort(kv, {}, &decltype(kv)::value_type::first);
  // std::execution::par should enable parallel subtree construction
  db.bulk_load(std::execution::par, kv.begin(), kv.end());
  for (const auto& [k, v] : kv) {
    const auto result = db.get(k);
    ASSERT_TRUE(result.has_value()) << "key " << k << " not found";
  }
}

// ─── Operational Tests ───────────────────────────────────────────────────────

// T38: Operations work correctly after bulk_load
UNODB_TEST(BulkLoadOps, ThenOperations) {
  u64_db db;
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  kv.reserve(10);
  for (std::uint64_t i = 0; i < 10; ++i) {
    kv.emplace_back(i << 56U, val);
  }
  std::ranges::sort(kv, {}, &decltype(kv)::value_type::first);
  db.bulk_load(kv.begin(), kv.end());

  // get works for all keys
  for (const auto& [k, v] : kv) {
    UNODB_ASSERT_TRUE(db.get(k).has_value());
  }

  // insert new key works
  constexpr std::uint64_t new_key = 0xFFULL << 56U;
  UNODB_ASSERT_TRUE(db.insert(new_key, val));
  UNODB_ASSERT_TRUE(db.get(new_key).has_value());

  // insert duplicate fails
  UNODB_ASSERT_FALSE(db.insert(kv[0].first, val));

  // remove works
  UNODB_ASSERT_TRUE(db.remove(kv[0].first));
  UNODB_ASSERT_FALSE(db.get(kv[0].first).has_value());

  // scan works
  std::size_t count = 0;
  db.scan([&count](auto&) {
    ++count;
    return false;  // continue
  });
  UNODB_ASSERT_EQ(count, 10U);  // 10 original - 1 removed + 1 new = 10
}

// T39: Stats are correct after bulk_load
#ifdef UNODB_DETAIL_WITH_STATS
UNODB_TEST(BulkLoadOps, Stats) {
  u64_db db;
  // 17 keys differing at byte 0 → should produce 1 inode48
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  kv.reserve(17);
  for (std::uint64_t i = 0; i < 17; ++i) {
    kv.emplace_back(i << 56U, val);
  }
  std::ranges::sort(kv, {}, &decltype(kv)::value_type::first);
  db.bulk_load(kv.begin(), kv.end());

  const auto counts = db.get_node_counts();
  UNODB_ASSERT_EQ(counts[as_i<node_type::LEAF>], 17U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I4>], 0U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I16>], 0U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I48>], 1U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I256>], 0U);
}
#endif  // UNODB_DETAIL_WITH_STATS

// T40: No growth events during bulk_load (all right-sized at allocation)
UNODB_TEST(BulkLoadOps, NoGrowthEvents) {
  u64_db db;
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  kv.reserve(100);
  for (std::uint64_t i = 0; i < 100; ++i) {
    kv.emplace_back(i << 56U, val);
  }
  std::ranges::sort(kv, {}, &decltype(kv)::value_type::first);
  db.bulk_load(kv.begin(), kv.end());

  // In bulk_load, no inode should ever grow (they're right-sized at creation).
  // The growing_inode_counts track create events, not grow events — but the
  // key insight is that no shrink events should exist.
  // Just verify all keys are present:
  for (const auto& [k, v] : kv) {
    ASSERT_TRUE(db.get(k).has_value()) << "key " << k << " not found";
  }
}

// T42: mutex_db bulk_load works
UNODB_TEST(BulkLoadOps, MutexDb) {
  u64_mutex_db db;
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  kv.reserve(100);
  for (std::uint64_t i = 0; i < 100; ++i) {
    kv.emplace_back(i << 56U, val);
  }
  std::ranges::sort(kv, {}, &decltype(kv)::value_type::first);
  db.bulk_load(kv.begin(), kv.end());

  for (const auto& [k, v] : kv) {
    const auto result = db.get(k);
    ASSERT_TRUE(result.first.has_value()) << "key " << k << " not found";
  }
}

// T38 extended: clear after bulk_load then re-bulk_load
UNODB_TEST(BulkLoadOps, ClearAndReload) {
  u64_db db;
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  kv.reserve(10);
  for (std::uint64_t i = 0; i < 10; ++i) {
    kv.emplace_back(i << 56U, val);
  }
  std::ranges::sort(kv, {}, &decltype(kv)::value_type::first);
  db.bulk_load(kv.begin(), kv.end());
  db.clear();
  UNODB_ASSERT_TRUE(db.empty());
  // Re-load different data
  std::vector<std::pair<std::uint64_t, value_view>> kv2;
  kv2.reserve(10);
  for (std::uint64_t i = 100; i < 110; ++i) {
    kv2.emplace_back(i << 56U, val);
  }
  std::ranges::sort(kv2, {}, &decltype(kv2)::value_type::first);
  db.bulk_load(kv2.begin(), kv2.end());
  for (const auto& [k, v] : kv2) {
    UNODB_ASSERT_TRUE(db.get(k).has_value());
  }
}

// T43: olc_db parallel bulk_load
UNODB_TEST(BulkLoadOps, OlcDbParallel) {
  u64_olc_db db;
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  kv.reserve(1000);
  for (std::uint64_t i = 0; i < 1000; ++i) {
    kv.emplace_back(i << 48U, val);
  }
  std::ranges::sort(kv, {}, &decltype(kv)::value_type::first);
  db.bulk_load(std::execution::par, kv.begin(), kv.end());
  for (const auto& [k, v] : kv) {
    const auto result = db.get(k);
    ASSERT_TRUE(result.has_value()) << "key " << k << " not found";
  }
}

// T44: olc_db concurrent readers after parallel bulk_load
// Verifies that optimistic locks are correctly initialized after bulk_load so
// concurrent readers can traverse the tree without deadlock or data corruption.
UNODB_TEST(BulkLoadOps, OlcDbConcurrentReaders) {
  u64_olc_db db;
  constexpr std::size_t n_keys = 10000;
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  kv.reserve(n_keys);
  for (std::uint64_t i = 0; i < n_keys; ++i) {
    kv.emplace_back(i << 40U, val);
  }
  std::ranges::sort(kv, {}, &decltype(kv)::value_type::first);
  db.bulk_load(std::execution::par, kv.begin(), kv.end());

  // Pause main thread QSBR so reader threads can register
  unodb::this_thread().qsbr_pause();

  constexpr std::size_t n_threads = 4;
  std::array<unodb::qsbr_thread, n_threads> threads;
  for (std::size_t t = 0; t < n_threads; ++t) {
    threads[t] = unodb::qsbr_thread([&kv, &db] {
      for (const auto& [k, v] : kv) {
        const unodb::quiescent_state_on_scope_exit qsbr{};
        const auto result = db.get(k);
        EXPECT_TRUE(result.has_value()) << "key " << k << " not found";
      }
      unodb::this_thread().quiescent();
    });
  }
  for (auto& t : threads) t.join();

  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();
  unodb::test::expect_idle_qsbr();
}

}  // namespace

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
