// Copyright 2026 UnoDB contributors

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "art_common.hpp"
#include "db_test_utils.hpp"
#include "gtest_utils.hpp"
#include "node_type.hpp"

#ifdef UNODB_DETAIL_WITH_STATS

UNODB_DETAIL_DISABLE_MSVC_WARNING(26445)

namespace {

using unodb::as_i;
using unodb::node_type;
using unodb::value_view;
using unodb::test::u64_db;

constexpr auto val_bytes =
    std::array<std::byte, 5>{std::byte{0x68}, std::byte{0x65}, std::byte{0x6C},
                             std::byte{0x6C}, std::byte{0x6F}};
constexpr auto val = value_view{val_bytes};

/// Generate N keys that differ at byte 0 (big-endian uint64_t).
/// For N <= 256, all keys differ only at byte 0 (single root node).
/// For N > 256, extra keys share byte 0 with key 0 but differ at byte 1.
[[nodiscard]] std::vector<std::pair<std::uint64_t, value_view>> make_keys(
    std::size_t n) {
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  kv.reserve(n);
  for (std::uint64_t i = 0; i < n && i < 256; ++i) {
    const auto key = i << 56U;
    kv.emplace_back(key, val);
  }
  // Overflow keys: share byte 0 == 0x00, differ at byte 1
  for (std::uint64_t i = 256; i < n; ++i) {
    const auto key = (i - 255) << 48U;
    kv.emplace_back(key, val);
  }
  std::ranges::sort(kv, {}, &decltype(kv)::value_type::first);
  return kv;
}

// T01
UNODB_TEST(BulkLoad, Empty) {
  u64_db db;
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  db.bulk_load(kv.begin(), kv.end());
  UNODB_ASSERT_TRUE(db.empty());
  const auto counts = db.get_node_counts();
  for (const auto c : counts) {
    UNODB_ASSERT_EQ(c, 0U);
  }
}

// T02
UNODB_TEST(BulkLoad, Single) {
  u64_db db;
  constexpr std::uint64_t key = 1;
  std::vector<std::pair<std::uint64_t, value_view>> kv{{key, val}};
  db.bulk_load(kv.begin(), kv.end());
  const auto result = db.get(key);
  UNODB_ASSERT_TRUE(result.has_value());
  UNODB_ASSERT_EQ(  // NOLINT(bugprone-unchecked-optional-access)
      result.value().size(), val.size());
  const auto counts = db.get_node_counts();
  UNODB_ASSERT_EQ(counts[as_i<node_type::LEAF>], 1U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I4>], 0U);
}

// T03
UNODB_TEST(BulkLoad, Small4Keys) {
  u64_db db;
  auto kv = make_keys(4);
  db.bulk_load(kv.begin(), kv.end());
  for (const auto& [k, v] : kv) {
    UNODB_ASSERT_TRUE(db.get(k).has_value());
  }
  const auto counts = db.get_node_counts();
  UNODB_ASSERT_EQ(counts[as_i<node_type::LEAF>], 4U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I4>], 1U);
}

// T04
UNODB_TEST(BulkLoad, Boundary4) {
  u64_db db;
  auto kv = make_keys(4);
  db.bulk_load(kv.begin(), kv.end());
  const auto counts = db.get_node_counts();
  UNODB_ASSERT_EQ(counts[as_i<node_type::I4>], 1U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I16>], 0U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I48>], 0U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I256>], 0U);
}

// T05
UNODB_TEST(BulkLoad, Boundary5) {
  u64_db db;
  auto kv = make_keys(5);
  db.bulk_load(kv.begin(), kv.end());
  const auto counts = db.get_node_counts();
  UNODB_ASSERT_EQ(counts[as_i<node_type::I16>], 1U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I4>], 0U);
}

// T06
UNODB_TEST(BulkLoad, Boundary16) {
  u64_db db;
  auto kv = make_keys(16);
  db.bulk_load(kv.begin(), kv.end());
  const auto counts = db.get_node_counts();
  UNODB_ASSERT_EQ(counts[as_i<node_type::I16>], 1U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I48>], 0U);
}

// T07
UNODB_TEST(BulkLoad, Boundary17) {
  u64_db db;
  auto kv = make_keys(17);
  db.bulk_load(kv.begin(), kv.end());
  const auto counts = db.get_node_counts();
  UNODB_ASSERT_EQ(counts[as_i<node_type::I48>], 1U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I16>], 0U);
}

// T08
UNODB_TEST(BulkLoad, Boundary48) {
  u64_db db;
  auto kv = make_keys(48);
  db.bulk_load(kv.begin(), kv.end());
  const auto counts = db.get_node_counts();
  UNODB_ASSERT_EQ(counts[as_i<node_type::I48>], 1U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I256>], 0U);
}

// T09
UNODB_TEST(BulkLoad, Boundary49) {
  u64_db db;
  auto kv = make_keys(49);
  db.bulk_load(kv.begin(), kv.end());
  const auto counts = db.get_node_counts();
  UNODB_ASSERT_EQ(counts[as_i<node_type::I256>], 1U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::I48>], 0U);
}

// T10
UNODB_TEST(BulkLoad, Growth260) {
  u64_db db;
  auto kv = make_keys(260);
  db.bulk_load(kv.begin(), kv.end());
  for (const auto& [k, v] : kv) {
    UNODB_ASSERT_TRUE(db.get(k).has_value());
  }
  const auto counts = db.get_node_counts();
  UNODB_ASSERT_EQ(counts[as_i<node_type::I256>], 1U);
  UNODB_ASSERT_EQ(counts[as_i<node_type::LEAF>], 260U);
}

}  // namespace

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

#endif  // UNODB_DETAIL_WITH_STATS
