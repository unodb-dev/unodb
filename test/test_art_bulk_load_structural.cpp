// Copyright 2026 UnoDB contributors

#include "global.hpp"  // IWYU pragma: keep

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
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
using unodb::key_view;
using unodb::node_type;
using unodb::value_view;
using unodb::test::key_view_db;
using unodb::test::key_view_u64val_db;
using unodb::test::u64_db;

constexpr auto val16 = std::array<std::byte, 16>{};
constexpr auto large_val = value_view{val16};
constexpr auto sval_bytes =
    std::array<std::byte, 5>{std::byte{0x68}, std::byte{0x65}, std::byte{0x6C},
                             std::byte{0x6C}, std::byte{0x6F}};
constexpr auto sval = value_view{sval_bytes};

// T11
UNODB_TEST(BulkLoadStructural, BulkLoadPrefix7) {
  u64_db db;
  std::vector<std::pair<std::uint64_t, value_view>> kv{
      {0x0102030405060700ULL, sval}, {0x0102030405060701ULL, sval}};
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 1U);
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 2U);
}
// T12
UNODB_TEST(BulkLoadStructural, BulkLoadPrefix8) {
  std::vector<std::vector<std::byte>> s{
      {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5},
       std::byte{6}, std::byte{7}, std::byte{8}, std::byte{0}},
      {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5},
       std::byte{6}, std::byte{7}, std::byte{8}, std::byte{1}}};
  std::vector<std::pair<key_view, value_view>> kv;
  for (auto& v : s)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        key_view{v}, large_val);
  key_view_db db;
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 2U);
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 2U);
}
// T13
UNODB_TEST(BulkLoadStructural, BulkLoadPrefix15) {
  std::vector<std::vector<std::byte>> s(
      2, std::vector<std::byte>(16, std::byte{0xAA}));
  s[0][15] = std::byte{0};
  s[1][15] = std::byte{1};
  std::vector<std::pair<key_view, value_view>> kv;
  for (auto& v : s)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        key_view{v}, large_val);
  key_view_db db;
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 2U);
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 2U);
}
// T14
UNODB_TEST(BulkLoadStructural, BulkLoadPrefix16) {
  std::vector<std::vector<std::byte>> s(
      2, std::vector<std::byte>(17, std::byte{0xBB}));
  s[0][16] = std::byte{0};
  s[1][16] = std::byte{1};
  std::vector<std::pair<key_view, value_view>> kv;
  for (auto& v : s)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        key_view{v}, large_val);
  key_view_db db;
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 3U);
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 2U);
}
// T15
UNODB_TEST(BulkLoadStructural, BulkLoadVIS) {
  std::vector<std::vector<std::byte>> s{{std::byte{1}}, {std::byte{2}}};
  std::vector<std::pair<key_view, std::uint64_t>> kv;
  for (auto& v : s)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        key_view{v}, 42ULL);
  key_view_u64val_db db;
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 1U);
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 0U);
}
// T16
UNODB_TEST(BulkLoadStructural, BulkLoadVISWithChain) {
  std::vector<std::vector<std::byte>> s{{std::byte{1}, std::byte{0xAA}},
                                        {std::byte{1}, std::byte{0xBB}}};
  std::vector<std::pair<key_view, std::uint64_t>> kv;
  for (auto& v : s)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        key_view{v}, 99ULL);
  key_view_u64val_db db;
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 1U);
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 0U);
}
// T17
UNODB_TEST(BulkLoadStructural, BulkLoadVISLongPrefix) {
  std::vector<std::vector<std::byte>> s(
      2, std::vector<std::byte>(9, std::byte{0xCC}));
  s[0][8] = std::byte{0};
  s[1][8] = std::byte{1};
  std::vector<std::pair<key_view, std::uint64_t>> kv;
  for (auto& v : s)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        key_view{v}, 77ULL);
  key_view_u64val_db db;
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 2U);
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 0U);
}
// T18
UNODB_TEST(BulkLoadStructural, BulkLoadKeylessLeaf) {
  std::vector<std::vector<std::byte>> s{
      {std::byte{1}, std::byte{2}, std::byte{3}},
      {std::byte{4}, std::byte{5}, std::byte{6}}};
  std::vector<std::pair<key_view, value_view>> kv;
  for (auto& v : s)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        key_view{v}, large_val);
  key_view_db db;
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  // Root I4 (dispatch byte 0) + 2 chain I4s wrapping keyless leaves
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 3U);
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 2U);
}
// T19
UNODB_TEST(BulkLoadStructural, BulkLoadKeylessLeafWithChain) {
  std::vector<std::vector<std::byte>> s{
      {std::byte{1}, std::byte{0xAA}, std::byte{0xFF}},
      {std::byte{1}, std::byte{0xBB}, std::byte{0xFF}}};
  std::vector<std::pair<key_view, value_view>> kv;
  for (auto& v : s)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        key_view{v}, large_val);
  key_view_db db;
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 3U);
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 2U);
}
// T20
UNODB_TEST(BulkLoadStructural, BulkLoadLarge) {
  std::mt19937 rng(42);  // NOLINT(cert-msc32-c,cert-msc51-cpp)
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  kv.reserve(100000);
  UNODB_DETAIL_DISABLE_GCC_WARNING("-Wuseless-cast")
  for (int i = 0; i < 100000; ++i)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        (static_cast<std::uint64_t>(rng()) << 32U) | rng(), sval);
  UNODB_DETAIL_RESTORE_GCC_WARNINGS()
  std::ranges::sort(kv, {}, &decltype(kv)::value_type::first);
  kv.erase(std::unique(  // NOLINT(modernize-use-ranges)
               kv.begin(), kv.end(),
               [](const auto& a, const auto& b) { return a.first == b.first; }),
           kv.end());
  u64_db db;
  db.bulk_load(kv.begin(), kv.end());
  for (const auto& [k, v] : kv) UNODB_ASSERT_TRUE(db.get(k).has_value());
  std::uint64_t prev = 0;
  bool first = true;
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
  db.scan([&](auto& visitor) {
    unodb::key_decoder dec{visitor.get_key()};
    std::uint64_t k{};
    dec.decode(k);
    if (!first) {
      UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)
      UNODB_DETAIL_DISABLE_MSVC_WARNING(26818)
      EXPECT_GE(k, prev);
      UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
      UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
    }
    prev = k;
    first = false;
    return false;
  });
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
}
// T21 — key_view bulk_load with random fixed-length keys (prefix-free).
UNODB_TEST(BulkLoadStructural, BulkLoadKeyView) {
  constexpr std::size_t key_len =
      10;                 // Fixed length → no key is prefix of another
  std::mt19937 rng(123);  // NOLINT(cert-msc32-c,cert-msc51-cpp)
  std::vector<std::vector<std::byte>> storage;
  for (int i = 0; i < 1000; ++i) {
    std::vector<std::byte> k(key_len);
    for (auto& b : k) b = static_cast<std::byte>(rng() & 0xFFU);
    storage.push_back(std::move(k));
  }
  std::ranges::sort(storage);
  // NOLINTNEXTLINE(modernize-use-ranges)
  storage.erase(std::unique(storage.begin(), storage.end()), storage.end());
  std::vector<std::pair<key_view, value_view>> kv;
  kv.reserve(storage.size());
  for (auto& v : storage)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        key_view{v}, large_val);
  key_view_db db;
  db.bulk_load(kv.begin(), kv.end());
  for (const auto& [k, v] : kv) UNODB_ASSERT_TRUE(db.get(k).has_value());
}
// T22
UNODB_TEST(BulkLoadStructural, BulkLoadVISRootSingle) {
  std::vector<std::vector<std::byte>> s{{std::byte{1}, std::byte{2}}};
  std::vector<std::pair<key_view, std::uint64_t>> kv;
  kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
      key_view{s[0]}, 55ULL);
  key_view_u64val_db db;
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 1U);
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 0U);
  UNODB_ASSERT_TRUE(db.get(key_view{s[0]}).has_value());
}
// T23
UNODB_TEST(BulkLoadStructural, BulkLoadKeylessLeafRootSingle) {
  std::vector<std::vector<std::byte>> s{
      {std::byte{1}, std::byte{2}, std::byte{3}}};
  std::vector<std::pair<key_view, value_view>> kv;
  kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
      key_view{s[0]}, large_val);
  key_view_db db;
  db.bulk_load(kv.begin(), kv.end());
  UNODB_ASSERT_TRUE(db.get(key_view{s[0]}).has_value());
}
// T24
UNODB_TEST(BulkLoadStructural, BulkLoadFullLeafRootSingle) {
  u64_db db;
  std::vector<std::pair<std::uint64_t, value_view>> kv{{0xDEADBEEFULL, sval}};
  db.bulk_load(kv.begin(), kv.end());
  const auto c = db.get_node_counts();
  UNODB_ASSERT_EQ(c[as_i<node_type::LEAF>], 1U);
  UNODB_ASSERT_EQ(c[as_i<node_type::I4>], 0U);
}
// T25
UNODB_TEST(BulkLoadStructural, BulkLoadScanOrder) {
  std::vector<std::pair<std::uint64_t, value_view>> kv;
  for (std::uint64_t i = 0; i < 256; ++i)
    kv.emplace_back(  // NOLINT(performance-inefficient-vector-operation)
        i << 56U, sval);
  u64_db db;
  db.bulk_load(kv.begin(), kv.end());
  std::uint64_t prev = 0;
  std::size_t count = 0;
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
  db.scan([&](auto& visitor) {
    unodb::key_decoder dec{visitor.get_key()};
    std::uint64_t k{};
    dec.decode(k);
    if (count > 0) {
      UNODB_EXPECT_GT(k, prev);
    }
    prev = k;
    ++count;
    return false;
  });
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  UNODB_ASSERT_EQ(count, 256U);
}

}  // namespace

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

#endif  // UNODB_DETAIL_WITH_STATS
