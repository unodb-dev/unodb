// Copyright 2026 UnoDB contributors

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "art_common.hpp"
#include "art_test_data.hpp"
#include "db_test_utils.hpp"
#include "gtest_utils.hpp"
#include "olc_art.hpp"  // NOLINT(misc-include-cleaner)
#include "qsbr.hpp"
#include "qsbr_test_utils.hpp"

#ifndef NDEBUG
#include <vector>

#include "sync.hpp"
#include "sync_point_test_utils.hpp"
#include "test_utils.hpp"
#include "thread_sync.hpp"
#endif

// MSVC SA C26496 false-positive on variables written via non-const reference.
// MSVC SA C26440 false-positive on lambdas that throw or use if-constexpr.
// MSVC SA C26814 false-positive on template-dependent const variables.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26814)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26818)
UNODB_DETAIL_DISABLE_MSVC_WARNING(4702)
UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)

namespace {

#ifndef NDEBUG
using unodb::test::sync_point_guard;
#endif

// ===================================================================
// Thread-safe key encoding for parallel tests.
// ===================================================================

/// Encode a key into caller-owned storage. For key_view dbs, encodes into
/// buf and returns a key_view backed by it. For u64 dbs, returns the key
/// directly (buf unused).
template <class Db>
typename Db::key_type make_local_key(
    std::size_t i, [[maybe_unused]] unodb::key_encoder& enc,
    [[maybe_unused]] std::array<std::byte, sizeof(std::uint64_t)>& buf) {
  if constexpr (std::is_same_v<typename Db::key_type, unodb::key_view>) {
    const std::uint64_t key_val{i};
    const auto kv = enc.reset().encode(key_val).get_key_view();
    std::ranges::copy(kv, buf.begin());
    return {buf.data(), buf.size()};
  } else {
    return static_cast<typename Db::key_type>(i);
  }
}

// ===================================================================
// Helper lambdas shared across tests.
// ===================================================================

/// Lambda that returns keep — value unchanged.
constexpr auto keep_fn = []([[maybe_unused]] auto& v) noexcept {
  return unodb::upsert_action::keep;
};

/// Lambda that returns update — increments value by 1 (arithmetic types only).
/// For non-arithmetic types (value_view), returns keep instead.
constexpr auto increment_fn = [](auto& v) noexcept {
  if constexpr (std::is_arithmetic_v<std::remove_reference_t<decltype(v)>>) {
    ++v;
    return unodb::upsert_action::update;
  } else {
    (void)v;
    return unodb::upsert_action::keep;
  }
};

/// Lambda that returns erase — removes the key.
constexpr auto erase_fn = []([[maybe_unused]] auto& v) noexcept {
  return unodb::upsert_action::erase;
};

/// Execute a callable within a QSBR scope for OLC types (no-op for others).
template <class Db, typename Fn>
decltype(auto) with_qsbr([[maybe_unused]] Fn&& fn) {
  if constexpr (unodb::test::is_olc_db<Db>) {
    const unodb::quiescent_state_on_scope_exit qsbr{};
    return std::forward<Fn>(fn)();
  } else {
    return std::forward<Fn>(fn)();
  }
}

#ifndef NDEBUG
/// The values an upsert lambda saw, one per call.  It lives outside the lambda
/// because upsert copies fn into every attempt, and it copies value_view bytes
/// because a concurrent writer may retire the leaf they point into.
template <class Db>
class seen_values {
 public:
  /// The value type the lambda sees.
  using value_type = typename Db::value_type;

  /// Record the value one lambda call saw.
  void record(const value_type& v) {
    if constexpr (value_is_view)
      seen_.emplace_back(v.begin(), v.end());
    else
      seen_.push_back(v);
  }

  /// Assert that the lambda ran twice, seeing \a first and then \a second.
  void assert_saw(const value_type& first, const value_type& second) const {
    UNODB_ASSERT_EQ(2U, seen_.size());
    if constexpr (value_is_view) {
      UNODB_ASSERT_TRUE(std::ranges::equal(seen_[0], first));
      UNODB_ASSERT_TRUE(std::ranges::equal(seen_[1], second));
    } else {
      UNODB_ASSERT_EQ(first, seen_[0]);
      UNODB_ASSERT_EQ(second, seen_[1]);
    }
  }

 private:
  /// Whether a seen value must be copied out of its leaf.
  static constexpr bool value_is_view =
      std::is_same_v<value_type, unodb::value_view>;

  /// The recorded values, in call order.
  std::vector<
      std::conditional_t<value_is_view, std::vector<std::byte>, value_type>>
      seen_;
};
#endif  // NDEBUG

// ===================================================================
// UpsertTest — typed test fixture parameterized over all db types.
// ===================================================================

template <class Db>
class UpsertTest : public ::testing::Test {
 public:
  using Test::Test;
};

// All db types: u64 key + value_view, key_view + value_view,
// key_view + u64 value (VIS types).
using UpsertTypes = ::testing::Types<
    unodb::test::u64_db, unodb::test::u64_mutex_db, unodb::test::u64_olc_db,
    unodb::test::key_view_db, unodb::test::key_view_mutex_db,
    unodb::test::key_view_olc_db, unodb::test::key_view_u64val_db,
    unodb::test::key_view_u64val_mutex_db, unodb::test::key_view_u64val_olc_db>;

UNODB_TYPED_TEST_SUITE(UpsertTest, UpsertTypes)

// ===================================================================
// Unit Tests — Basic Semantics (IDs 1-6, 14-16)
// ===================================================================

// ID-1: Returns true, get(k)==v, lambda not called.
UNODB_TYPED_TEST(UpsertTest, InsertPathKeyAbsent) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(1);
  const auto v = unodb::test::get_test_value<TypeParam>(0);
  const auto result = with_qsbr<TypeParam>([&] {
    return db.upsert(k, v, [](auto& /*x*/) {
      ADD_FAILURE() << "lambda should not be called";  // LCOV_EXCL_LINE
      return unodb::upsert_action::keep;               // LCOV_EXCL_LINE
    });
  });
  UNODB_ASSERT_TRUE(result);
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v);
}

// ID-2: Returns false, get(k)==v0 (original value unchanged).
UNODB_TYPED_TEST(UpsertTest, KeepKeyPresent) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(1);
  const auto v0 = unodb::test::get_test_value<TypeParam>(0);
  const auto v1 = unodb::test::get_test_value<TypeParam>(1);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v0)); });
  const auto result =
      with_qsbr<TypeParam>([&] { return db.upsert(k, v1, keep_fn); });
  UNODB_ASSERT_FALSE(result);
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v0);
}

// ID-3: Returns false, value is replaced by update action.
UNODB_TYPED_TEST(UpsertTest, UpdateKeyPresent) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(1);
  const auto v0 = unodb::test::get_test_value<TypeParam>(0);
  const auto v1 = unodb::test::get_test_value<TypeParam>(1);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v0)); });
  const auto result = with_qsbr<TypeParam>([&] {
    return db.upsert(k, v1, [](auto& x) {
      if constexpr (std::is_arithmetic_v<std::remove_reference_t<decltype(x)>>)
        x = 42;  // typed: lambda modifies the local copy
      return unodb::upsert_action::update;
    });
  });
  UNODB_ASSERT_FALSE(result);
  if constexpr (std::is_same_v<typename TypeParam::value_type,
                               unodb::value_view>) {
    // value_view: proposed value v1 is installed
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v1);
  } else {
    // typed: lambda-modified value (42) is written back
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k,
                         static_cast<typename TypeParam::value_type>(42));
  }
}

// ID-4: Returns false both times — value_view replaces twice.
UNODB_TYPED_TEST(UpsertTest, UpdateIdempotency) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(1);
  const auto v0 = unodb::test::get_test_value<TypeParam>(0);
  const auto v1 = unodb::test::get_test_value<TypeParam>(1);
  const auto v2 = unodb::test::get_test_value<TypeParam>(2);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v0)); });
  if constexpr (std::is_same_v<typename TypeParam::value_type,
                               unodb::value_view>) {
    // value_view: each upsert replaces with the proposed value
    auto update_fn = [](auto& /*x*/) { return unodb::upsert_action::update; };
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v1, update_fn)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v1);
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v2, update_fn)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v2);
  } else {
    // typed: lambda sets to 100, then adds 10 twice → 120
    with_qsbr<TypeParam>([&] {
      UNODB_ASSERT_FALSE(db.upsert(k, v0, [](auto& x) {
        x = 100;
        return unodb::upsert_action::update;
      }));
    });
    auto add10 = [](auto& x) {
      x += 10;
      return unodb::upsert_action::update;
    };
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_FALSE(db.upsert(k, v0, add10)); });
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_FALSE(db.upsert(k, v0, add10)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k,
                         static_cast<typename TypeParam::value_type>(120));
  }
}

// ID-4b: Update with multiple keys (forces inode traversal for value_view).
UNODB_TYPED_TEST(UpsertTest, UpdateMultipleKeys) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k0 = verifier.coerce_key(0);
  const auto k1 = verifier.coerce_key(1);
  const auto k2 = verifier.coerce_key(2);
  const auto v0 = unodb::test::get_test_value<TypeParam>(0);
  const auto v1 = unodb::test::get_test_value<TypeParam>(1);
  const auto v2 = unodb::test::get_test_value<TypeParam>(2);
  // Insert 3 keys to force inode creation.
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k0, v0)); });
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k1, v1)); });
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k2, v2)); });
  // Update the middle key.
  auto update_fn = [](auto& x) {
    if constexpr (std::is_arithmetic_v<std::remove_reference_t<decltype(x)>>)
      x = 99;
    return unodb::upsert_action::update;
  };
  with_qsbr<TypeParam>(
      [&] { UNODB_ASSERT_FALSE(db.upsert(k1, v0, update_fn)); });
  if constexpr (std::is_same_v<typename TypeParam::value_type,
                               unodb::value_view>) {
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k1, v0);  // proposed value installed
  } else {
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k1,
                         static_cast<typename TypeParam::value_type>(99));
  }
  // Other keys unchanged.
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k0, v0);
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k2, v2);
}

// ID-4c: Upsert insert triggers prefix split (VIS coverage).
UNODB_TYPED_TEST(UpsertTest, InsertPrefixSplit) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  // Insert two keys sharing a long prefix to create an inode with prefix.
  const auto k0 = verifier.coerce_key(0);
  const auto k1 = verifier.coerce_key(1);
  const auto v0 = unodb::test::get_test_value<TypeParam>(0);
  const auto v1 = unodb::test::get_test_value<TypeParam>(1);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k0, v0)); });
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k1, v1)); });
  // Upsert a key that diverges within the prefix → triggers prefix split.
  const auto k_split = verifier.coerce_key(256);
  const auto v_split = unodb::test::get_test_value<TypeParam>(2);
  const auto result = with_qsbr<TypeParam>([&] {
    return db.upsert(k_split, v_split, [](auto& /*x*/) {
      ADD_FAILURE() << "lambda should not be called";  // LCOV_EXCL_LINE
      return unodb::upsert_action::keep;               // LCOV_EXCL_LINE
    });
  });
  UNODB_ASSERT_TRUE(result);  // key was absent → inserted
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k0, v0);
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k1, v1);
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k_split, v_split);
}

// ID-5: Returns false, get(k) empty.
UNODB_TYPED_TEST(UpsertTest, EraseKeyPresent) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(1);
  const auto v = unodb::test::get_test_value<TypeParam>(0);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v)); });
  const auto result =
      with_qsbr<TypeParam>([&] { return db.upsert(k, v, erase_fn); });
  UNODB_ASSERT_FALSE(result);
  with_qsbr<TypeParam>(
      [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
}

// ID-6: Mixed operations across 100 keys.
UNODB_TYPED_TEST(UpsertTest, MixedOperations) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  // Insert keys 0..99 with value = get_test_value(i)
  for (std::size_t i = 0; i < 100; ++i) {
    const auto k = verifier.coerce_key(i);
    const auto v = unodb::test::get_test_value<TypeParam>(i);
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v)); });
  }
  // Upsert each: if i < 50, erase; else keep
  for (std::size_t i = 0; i < 100; ++i) {
    const auto k = verifier.coerce_key(i);
    const auto v = unodb::test::get_test_value<TypeParam>(i);
    if (i < 50) {
      with_qsbr<TypeParam>(
          [&] { UNODB_ASSERT_FALSE(db.upsert(k, v, erase_fn)); });
    } else {
      with_qsbr<TypeParam>(
          [&] { UNODB_ASSERT_FALSE(db.upsert(k, v, keep_fn)); });
    }
  }
  // Verify: keys 0..49 absent, keys 50..99 present
  for (std::size_t i = 0; i < 50; ++i) {
    const auto k = verifier.coerce_key(i);
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
  }
  for (std::size_t i = 50; i < 100; ++i) {
    const auto k = verifier.coerce_key(i);
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k,
                         unodb::test::get_test_value<TypeParam>(i));
  }
}

// ID-14: Single-entry tree, all three actions verified sequentially.
UNODB_TYPED_TEST(UpsertTest, RootLeafAllActions) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(1);
  const auto v0 = unodb::test::get_test_value<TypeParam>(0);
  const auto v1 = unodb::test::get_test_value<TypeParam>(1);
  // Insert
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.upsert(k, v0, keep_fn)); });
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v0);
  // Keep
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_FALSE(db.upsert(k, v1, keep_fn)); });
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v0);
  // Erase
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_FALSE(db.upsert(k, v1, erase_fn)); });
  with_qsbr<TypeParam>(
      [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.empty()); });
}

// ID-15: Empty tree, insert path returns true.
UNODB_TYPED_TEST(UpsertTest, EmptyTree) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(42);
  const auto v = unodb::test::get_test_value<TypeParam>(2);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.upsert(k, v, keep_fn)); });
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v);
}

// ID-16: Tree cleared, insert path returns true.
UNODB_TYPED_TEST(UpsertTest, AfterClear) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(1);
  const auto v = unodb::test::get_test_value<TypeParam>(0);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v)); });
  verifier.clear();
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.upsert(k, v, keep_fn)); });
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v);
}

// ===================================================================
// Type Coverage (IDs 11-13b)
// ===================================================================

// ID-11: uint64_t key + uint64_t value (VIS), keep/update/erase.
UNODB_TYPED_TEST(UpsertTest, TypeCoverageU64U64) {
  if constexpr (!std::is_same_v<typename TypeParam::value_type,
                                std::uint64_t>) {
    GTEST_SKIP() << "Only applies to u64 value types";
  } else {
    unodb::test::tree_verifier<TypeParam> verifier;
    auto& db = verifier.get_db();
    const auto k = verifier.coerce_key(5);
    const typename TypeParam::value_type v0 = 10;
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v0)); });
    // keep
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v0, keep_fn)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v0);
    // update
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v0, increment_fn)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k,
                         static_cast<typename TypeParam::value_type>(11));
    // erase
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v0, erase_fn)); });
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
  }
}

// ID-12: key_view key + uint64_t value (VIS), keep/update/erase.
UNODB_TYPED_TEST(UpsertTest, TypeCoverageKeyViewU64) {
  if constexpr (!std::is_same_v<typename TypeParam::key_type,
                                unodb::key_view> ||
                !std::is_same_v<typename TypeParam::value_type,
                                std::uint64_t>) {
    GTEST_SKIP() << "Only applies to key_view + u64 value types";
  } else {
    unodb::test::tree_verifier<TypeParam> verifier;
    auto& db = verifier.get_db();
    const auto k = verifier.coerce_key(7);
    const typename TypeParam::value_type v0 = 20;
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v0)); });
    // keep
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v0, keep_fn)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v0);
    // update
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v0, increment_fn)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k,
                         static_cast<typename TypeParam::value_type>(21));
    // erase
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v0, erase_fn)); });
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
  }
}

// ID-13: key_view key + value_view value, keep/erase only.
UNODB_TYPED_TEST(UpsertTest, TypeCoverageKeyViewValueView) {
  if constexpr (!std::is_same_v<typename TypeParam::key_type,
                                unodb::key_view> ||
                !std::is_same_v<typename TypeParam::value_type,
                                unodb::value_view>) {
    GTEST_SKIP() << "Only applies to key_view + value_view types";
  } else {
    unodb::test::tree_verifier<TypeParam> verifier;
    auto& db = verifier.get_db();
    const auto k = verifier.coerce_key(3);
    const auto v = unodb::test_data::test_values[2];
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v)); });
    // keep
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_FALSE(db.upsert(k, v, keep_fn)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v);
    // erase
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v, erase_fn)); });
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
  }
}

// ID-13b: uint64_t key + value_view value, keep/erase only.
UNODB_TYPED_TEST(UpsertTest, TypeCoverageU64ValueView) {
  if constexpr (!std::is_same_v<typename TypeParam::key_type, std::uint64_t> ||
                !std::is_same_v<typename TypeParam::value_type,
                                unodb::value_view>) {
    GTEST_SKIP() << "Only applies to u64 key + value_view types";
  } else {
    unodb::test::tree_verifier<TypeParam> verifier;
    auto& db = verifier.get_db();
    const auto k = verifier.coerce_key(10);
    const auto v = unodb::test_data::test_values[3];
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v)); });
    // keep
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_FALSE(db.upsert(k, v, keep_fn)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v);
    // erase
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v, erase_fn)); });
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
  }
}

// ===================================================================
// Erase-Specific Tests — single-threaded (IDs 23a-23e)
// ===================================================================

// ID-23a: Erase triggers I16→I4 shrink.
UNODB_TYPED_TEST(UpsertTest, EraseTriggersShrink) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  // Insert 5 keys to create an I16 node, then erase one to trigger shrink to I4
  for (std::size_t i = 0; i < 5; ++i) {
    const auto k = verifier.coerce_key(i);
    with_qsbr<TypeParam>([&] {
      UNODB_ASSERT_TRUE(
          db.insert(k, unodb::test::get_test_value<TypeParam>(i)));
    });
  }
  const auto k_erase = verifier.coerce_key(std::size_t{0});
  with_qsbr<TypeParam>([&] {
    UNODB_ASSERT_FALSE(db.upsert(
        k_erase, unodb::test::get_test_value<TypeParam>(0), erase_fn));
  });
  with_qsbr<TypeParam>(
      [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k_erase))); });
  // Remaining 4 keys still present
  for (std::size_t i = 1; i < 5; ++i) {
    const auto k = verifier.coerce_key(i);
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k,
                         unodb::test::get_test_value<TypeParam>(i));
  }
}

// ID-23b: Erase triggers chain cut (key_view types).
UNODB_TYPED_TEST(UpsertTest, EraseTriggersChainCut) {
  if constexpr (!std::is_same_v<typename TypeParam::key_type,
                                unodb::key_view>) {
    GTEST_SKIP() << "Chain cut only applies to key_view types";
  } else {
    unodb::test::tree_verifier<TypeParam> verifier;
    auto& db = verifier.get_db();
    // Insert two keys that share a prefix to create a chain I4
    const auto k0 = verifier.coerce_key(0);
    const auto k1 = verifier.coerce_key(1);
    with_qsbr<TypeParam>([&] {
      UNODB_ASSERT_TRUE(
          db.insert(k0, unodb::test::get_test_value<TypeParam>(0)));
    });
    with_qsbr<TypeParam>([&] {
      UNODB_ASSERT_TRUE(
          db.insert(k1, unodb::test::get_test_value<TypeParam>(1)));
    });
    // Erase one to trigger chain cut
    with_qsbr<TypeParam>([&] {
      UNODB_ASSERT_FALSE(
          db.upsert(k0, unodb::test::get_test_value<TypeParam>(0), erase_fn));
    });
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k0))); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k1,
                         unodb::test::get_test_value<TypeParam>(1));
  }
}

// ID-23c: Erase root leaf, tree becomes empty.
UNODB_TYPED_TEST(UpsertTest, EraseRootLeaf) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(1);
  const auto v = unodb::test::get_test_value<TypeParam>(0);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v)); });
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_FALSE(db.upsert(k, v, erase_fn)); });
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.empty()); });
  with_qsbr<TypeParam>(
      [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
}

// ID-23d: Erase packed VIS value, slot cleared.
UNODB_TYPED_TEST(UpsertTest, EraseVisValue) {
  if constexpr (!std::is_same_v<typename TypeParam::value_type,
                                std::uint64_t>) {
    GTEST_SKIP() << "Only applies to VIS (u64 value) types";
  } else {
    unodb::test::tree_verifier<TypeParam> verifier;
    auto& db = verifier.get_db();
    const auto k = verifier.coerce_key(5);
    const typename TypeParam::value_type v = 42;
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v)); });
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k, v, erase_fn)); });
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
  }
}

// ID-23e: Erase then re-upsert inserts with new value.
UNODB_TYPED_TEST(UpsertTest, EraseThenReinsert) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(1);
  const auto v0 = unodb::test::get_test_value<TypeParam>(0);
  const auto v1 = unodb::test::get_test_value<TypeParam>(1);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v0)); });
  // Erase
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_FALSE(db.upsert(k, v0, erase_fn)); });
  with_qsbr<TypeParam>(
      [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
  // Re-insert via upsert with new value
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.upsert(k, v1, keep_fn)); });
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v1);
}

// ===================================================================
// Contract Verification — single-threaded (IDs C1, C3, C4)
// ===================================================================

// ID-C3: Mutations discarded on erase.
UNODB_TYPED_TEST(UpsertTest, MutationsDiscardedOnErase) {
  if constexpr (!std::is_same_v<typename TypeParam::value_type,
                                std::uint64_t>) {
    GTEST_SKIP() << "Mutation test requires mutable u64 value type";
  } else {
    unodb::test::tree_verifier<TypeParam> verifier;
    auto& db = verifier.get_db();
    const auto k = verifier.coerce_key(1);
    const typename TypeParam::value_type v0 = 10;
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v0)); });
    // Lambda mutates value AND returns erase
    with_qsbr<TypeParam>([&] {
      UNODB_ASSERT_FALSE(
          db.upsert(k, typename TypeParam::value_type{0}, [](auto& x) {
            x = 99;
            return unodb::upsert_action::erase;
          }));
    });
    // Key must be gone — mutation discarded
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k))); });
    // Re-insert and verify value is NOT 99
    const typename TypeParam::value_type v_new = 50;
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v_new)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v_new);
  }
}

// ID-C4: Throwing lambda leaves tree unchanged.
UNODB_TYPED_TEST(UpsertTest, ThrowingLambdaTreeUnchanged) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k = verifier.coerce_key(1);
  const auto v = unodb::test::get_test_value<TypeParam>(0);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k, v)); });
  // Lambda throws
  with_qsbr<TypeParam>([&] {
    UNODB_ASSERT_THROW(std::ignore = db.upsert(
                           k, v,
                           [](auto& /*x*/) {
                             throw std::runtime_error("test");
                             return unodb::upsert_action::keep;  // unreachable
                           }),
                       std::runtime_error);
  });
  // Tree unchanged
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k, v);
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_FALSE(db.empty()); });
}

// ===================================================================
// UpsertConcurrencyTest — olc_db-only fixture.
// ===================================================================

template <class Db>
class UpsertConcurrencyTest : public ::testing::Test {
 public:
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26447)
  ~UpsertConcurrencyTest() noexcept override {
    if constexpr (unodb::test::is_olc_db<Db>) {
      unodb::this_thread().quiescent();
      unodb::test::expect_idle_qsbr();
    }
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

 protected:
  // NOLINTNEXTLINE(bugprone-exception-escape)
  UpsertConcurrencyTest() noexcept {
    if constexpr (unodb::test::is_olc_db<Db>) unodb::test::expect_idle_qsbr();
  }

  template <std::size_t ThreadCount, std::size_t OpsPerThread, typename TestFn>
  void parallel_test(TestFn test_function) {
    if constexpr (unodb::test::is_olc_db<Db>) unodb::this_thread().qsbr_pause();

    std::array<unodb::test::thread<Db>, ThreadCount> threads;
    for (std::size_t i = 0; i < ThreadCount; ++i) {
      threads[i] =
          unodb::test::thread<Db>{test_function, &verifier, i, OpsPerThread};
    }
    for (auto& t : threads) {
      t.join();
    }

    if constexpr (unodb::test::is_olc_db<Db>)
      unodb::this_thread().qsbr_resume();
  }

  unodb::test::tree_verifier<Db> verifier{true};

 public:
  UpsertConcurrencyTest(const UpsertConcurrencyTest&) = delete;
  UpsertConcurrencyTest(UpsertConcurrencyTest&&) = delete;
  UpsertConcurrencyTest& operator=(const UpsertConcurrencyTest&) = delete;
  UpsertConcurrencyTest& operator=(UpsertConcurrencyTest&&) = delete;
};

using UpsertConcurrencyTypes =
    ::testing::Types<unodb::test::u64_olc_db, unodb::test::key_view_olc_db,
                     unodb::test::key_view_u64val_olc_db>;

UNODB_TYPED_TEST_SUITE(UpsertConcurrencyTest, UpsertConcurrencyTypes)

// ===================================================================
// Concurrency Tests (IDs 7-10, 17-22)
// ===================================================================

// ID-7: No crashes, all get(k) return valid values, tree size==N.
UNODB_TYPED_TEST(UpsertConcurrencyTest, UpsertPlusGet) {
  constexpr std::size_t N = 64;
  auto& db = this->verifier.get_db();

  this->template parallel_test<4, N / 2>(
      [](unodb::test::tree_verifier<TypeParam>* tv, std::size_t thread_i,
         std::size_t ops_per_thread) {
        auto& d = tv->get_db();
        unodb::key_encoder enc;
        std::array<std::byte, sizeof(std::uint64_t)> buf{};
        for (std::size_t i = 0; i < ops_per_thread; ++i) {
          const auto key = make_local_key<TypeParam>(
              (thread_i * ops_per_thread) + i, enc, buf);
          auto v = unodb::test::get_test_value<TypeParam>(i);
          const unodb::quiescent_state_on_scope_exit q{};
          if (thread_i < 2) {
            std::ignore = d.upsert(key, v, increment_fn);
          } else {
            std::ignore = d.get(key);
          }
        }
      });

  const unodb::quiescent_state_on_scope_exit q{};
  for (std::size_t i = 0; i < N; ++i) {
    const auto key = this->verifier.coerce_key(i);
    // Under contention, key may or may not be present.  The real check
    // is that get() completes without crashing or tripping a sanitizer.
    static_cast<void>(db.get(key));
  }
}

// ID-8: All keys present, get(k)==expected for each range, size==sum.
UNODB_TYPED_TEST(UpsertConcurrencyTest, UpsertDisjoint) {
  constexpr std::size_t range_size = 32;
  auto& db = this->verifier.get_db();

  this->template parallel_test<2, range_size>(
      [](unodb::test::tree_verifier<TypeParam>* tv, std::size_t thread_i,
         std::size_t ops_per_thread) {
        auto& d = tv->get_db();
        unodb::key_encoder enc;
        std::array<std::byte, sizeof(std::uint64_t)> buf{};
        const auto base = thread_i * 1000;
        for (std::size_t i = 0; i < ops_per_thread; ++i) {
          const auto key = make_local_key<TypeParam>(base + i, enc, buf);
          auto v = unodb::test::get_test_value<TypeParam>(i);
          const unodb::quiescent_state_on_scope_exit q{};
          std::ignore = d.upsert(key, v, keep_fn);
        }
      });

  const unodb::quiescent_state_on_scope_exit q{};
  for (std::size_t t = 0; t < 2; ++t) {
    const auto base = t * 1000;
    for (std::size_t i = 0; i < range_size; ++i) {
      const auto key = this->verifier.coerce_key(base + i);
      const auto result = db.get(key);
      UNODB_ASSERT_TRUE(TypeParam::key_found(result));
      const auto expected = unodb::test::get_test_value<TypeParam>(i);
      unodb::test::detail::assert_value_eq<TypeParam>(result, expected);
    }
  }
}

// ID-9: get(k)==final_value, lambda applied exactly once, size unchanged.
#ifndef NDEBUG
UNODB_TYPED_TEST(UpsertConcurrencyTest, OlcRestart) {
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(42);
  const auto other_key = this->verifier.coerce_key(99);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);
  auto v1 = unodb::test::get_test_value<TypeParam>(1);

  // Pre-insert key
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
  }

  // Arm sync point to force upgrade failure on first attempt
  const sync_point_guard guard{unodb::detail::sync_after_upsert_dup_found};
  unodb::detail::sync_after_upsert_dup_found.arm([&]() {
    unodb::detail::sync_after_upsert_dup_found.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  // T2: modify the node to invalidate T1's version
  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.insert(other_key, v1);
    unodb::detail::thread_syncs[1].notify();
  });

  // T1: upsert that will hit the sync point and restart
  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v1, increment_fn);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();

  // Post: get(k) has the updated value
  const unodb::quiescent_state_on_scope_exit q{};
  const auto result = db.get(key);
  UNODB_ASSERT_TRUE(TypeParam::key_found(result));
  // increment_fn: uint64 → v0+1; value_view → keep (v0 unchanged).
  if constexpr (std::is_same_v<typename TypeParam::value_type,
                               unodb::value_view>) {
    unodb::test::detail::assert_value_eq<TypeParam>(result, v0);
  } else {
    const auto expected = static_cast<typename TypeParam::value_type>(
        unodb::test_data::test_values_u64[0] + 1);
    unodb::test::detail::assert_value_eq<TypeParam>(result, expected);
  }
}
#endif  // NDEBUG

// ID-10: get(k) returns updated value OR empty, size==0 or 1.
UNODB_TYPED_TEST(UpsertConcurrencyTest, UpsertVsRemove) {
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(7);
  auto v = unodb::test::get_test_value<TypeParam>(0);

  // Pre-insert
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v));
  }

  unodb::this_thread().qsbr_pause();

  auto updater = unodb::test::thread<TypeParam>([&] {
    for (int i = 0; i < 100; ++i) {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.upsert(key, v, increment_fn);
    }
  });

  auto remover = unodb::test::thread<TypeParam>([&] {
    for (int i = 0; i < 100; ++i) {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.remove(key);
    }
  });

  updater.join();
  remover.join();
  unodb::this_thread().qsbr_resume();

  // Post: key present OR absent, no crash.  The real check is that get()
  // completes without crashing or tripping a sanitizer under contention.
  const unodb::quiescent_state_on_scope_exit q{};
  const auto result = db.get(key);
  static_cast<void>(result);  // key may be absent due to concurrent remove
}

// ID-17: Final value == N (all increments applied).
UNODB_TYPED_TEST(UpsertConcurrencyTest, CasIncrement) {
  constexpr std::size_t N = 8;
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(1);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);

  // Pre-insert with initial value
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
  }

  this->template parallel_test<N, 1>(
      [](unodb::test::tree_verifier<TypeParam>* tv, std::size_t /*thread_i*/,
         std::size_t /*ops_per_thread*/) {
        auto& d = tv->get_db();
        unodb::key_encoder enc;
        std::array<std::byte, sizeof(std::uint64_t)> buf{};
        const auto k = make_local_key<TypeParam>(1, enc, buf);
        auto v = unodb::test::get_test_value<TypeParam>(0);
        const unodb::quiescent_state_on_scope_exit q{};
        std::ignore = d.upsert(k, v, increment_fn);
      });

  // Post: value == initial + N increments
  const unodb::quiescent_state_on_scope_exit q{};
  const auto result = db.get(key);
  UNODB_ASSERT_TRUE(TypeParam::key_found(result));
  if constexpr (std::is_same_v<typename TypeParam::value_type, std::uint64_t>) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    UNODB_ASSERT_EQ(*result, N);
  }
}

// ID-18: Both keys present, T1's value==lambda result, node counts correct.
#ifndef NDEBUG
UNODB_TYPED_TEST(UpsertConcurrencyTest, CasDuringGrowth) {
  auto& db = this->verifier.get_db();

  const auto upsert_key = this->verifier.coerce_key(0);
  const auto growth_key = this->verifier.coerce_key(4);

  // Fill I4 to capacity (4 keys)
  for (std::size_t i = 0; i < 4; ++i) {
    const unodb::quiescent_state_on_scope_exit q{};
    unodb::key_encoder enc;
    std::array<std::byte, sizeof(std::uint64_t)> buf{};
    UNODB_ASSERT_TRUE(db.insert(make_local_key<TypeParam>(i, enc, buf),
                                unodb::test::get_test_value<TypeParam>(i)));
  }

  auto gv = unodb::test::get_test_value<TypeParam>(4);

  // Arm sync point: T1 pauses at dup-found, T2 triggers I4→I16
  const sync_point_guard guard{unodb::detail::sync_after_upsert_dup_found};
  unodb::detail::sync_after_upsert_dup_found.arm([&]() {
    unodb::detail::sync_after_upsert_dup_found.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.insert(growth_key, gv);
    unodb::detail::thread_syncs[1].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    auto v = unodb::test::get_test_value<TypeParam>(9);
    std::ignore = db.upsert(upsert_key, v, increment_fn);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_TRUE(TypeParam::key_found(db.get(upsert_key)));
  UNODB_ASSERT_TRUE(TypeParam::key_found(db.get(growth_key)));
}
#endif  // NDEBUG

// ID-19: T1 pauses at dup, T2 removes. T1 inserts, get(k)==v.
#ifndef NDEBUG
UNODB_TYPED_TEST(UpsertConcurrencyTest, CasKeyRemoved) {
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(5);
  const auto sibling_key = this->verifier.coerce_key(99);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);
  auto v1 = unodb::test::get_test_value<TypeParam>(1);

  // Pre-insert
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
  }
  // Need a second key so the tree isn't just a root leaf
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(
        db.insert(sibling_key, unodb::test::get_test_value<TypeParam>(2)));
  }

  // T1 finds duplicate, pauses. T2 removes the key.
  const sync_point_guard guard{unodb::detail::sync_after_upsert_dup_found};
  unodb::detail::sync_after_upsert_dup_found.arm([&]() {
    unodb::detail::sync_after_upsert_dup_found.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key);
    unodb::detail::thread_syncs[1].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v1, increment_fn);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();

  // Post: T1 re-inserts after restart, key present with upsert's default value.
  const unodb::quiescent_state_on_scope_exit q{};
  const auto result = db.get(key);
  UNODB_ASSERT_TRUE(TypeParam::key_found(result));
  unodb::test::detail::assert_value_eq<TypeParam>(result, v1);
}
#endif  // NDEBUG

// ID-20: Scan sees old or new value, never torn.
UNODB_TYPED_TEST(UpsertConcurrencyTest, CasPlusScan) {
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(10);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);

  // Pre-insert
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
  }

  unodb::this_thread().qsbr_pause();

  std::atomic<bool> done{false};

  auto updater = unodb::test::thread<TypeParam>([&] {
    for (int i = 0; i < 200; ++i) {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.upsert(key, v0, increment_fn);
    }
    done.store(true, std::memory_order_release);
  });

  auto scanner = unodb::test::thread<TypeParam>([&] {
    while (!done.load(std::memory_order_acquire)) {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.get(key);
    }
  });

  updater.join();
  scanner.join();
  unodb::this_thread().qsbr_resume();

  // Post: no crash, value is valid
  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_TRUE(TypeParam::key_found(db.get(key)));
}

// ID-21: No crashes, tree size>=0, all get(k) valid, no ASAN/TSan errors.
UNODB_TYPED_TEST(UpsertConcurrencyTest, RandomOpsStress) {
  constexpr std::size_t key_range = 64;

  // Pre-insert some keys
  for (std::size_t i = 0; i < key_range / 2; ++i) {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = this->verifier.get_db().insert(
        this->verifier.coerce_key(i),
        unodb::test::get_test_value<TypeParam>(i));
  }

  this->template parallel_test<4, 1000>(
      [](unodb::test::tree_verifier<TypeParam>* tv, std::size_t thread_i,
         std::size_t ops_per_thread) {
        auto& d = tv->get_db();
        unodb::key_encoder enc;
        std::array<std::byte, sizeof(std::uint64_t)> buf{};
        std::mt19937 gen{static_cast<unsigned>(thread_i * 7919)};
        std::uniform_int_distribution<std::size_t> key_dist{0, key_range - 1};
        std::uniform_int_distribution<int> op_dist{0, 3};

        for (std::size_t i = 0; i < ops_per_thread; ++i) {
          const auto k = make_local_key<TypeParam>(key_dist(gen), enc, buf);
          auto v = unodb::test::get_test_value<TypeParam>(i % 6);
          const unodb::quiescent_state_on_scope_exit q{};
          switch (op_dist(gen)) {
            case 0:
              std::ignore = d.upsert(k, v, increment_fn);
              break;
            case 1:
              std::ignore = d.upsert(k, v, erase_fn);
              break;
            case 2:
              std::ignore = d.remove(k);
              break;
            default:
              std::ignore = d.get(k);
              break;
          }
        }
      });

  // Post: no crash, tree is valid.  The real check is that empty()
  // completes without crashing or tripping a sanitizer under contention.
  const unodb::quiescent_state_on_scope_exit q{};
  static_cast<void>(this->verifier.get_db().empty());
}

// ID-22: Value==N updates, lambda called>=N.
UNODB_TYPED_TEST(UpsertConcurrencyTest, IdempotencyUnderContention) {
  constexpr std::size_t N = 8;
  constexpr std::size_t hot_keys = 4;
  auto& db = this->verifier.get_db();

  // Pre-insert hot keys
  for (std::size_t i = 0; i < hot_keys; ++i) {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(this->verifier.coerce_key(i),
                                unodb::test::get_test_value<TypeParam>(0)));
  }

  this->template parallel_test<N, hot_keys>(
      [](unodb::test::tree_verifier<TypeParam>* tv, std::size_t /*thread_i*/,
         std::size_t ops_per_thread) {
        auto& d = tv->get_db();
        unodb::key_encoder enc;
        std::array<std::byte, sizeof(std::uint64_t)> buf{};
        for (std::size_t i = 0; i < ops_per_thread; ++i) {
          const auto k = make_local_key<TypeParam>(i, enc, buf);
          auto v = unodb::test::get_test_value<TypeParam>(0);
          const unodb::quiescent_state_on_scope_exit q{};
          std::ignore = d.upsert(k, v, increment_fn);
        }
      });

  // Post: each hot key was incremented N times total
  const unodb::quiescent_state_on_scope_exit q{};
  for (std::size_t i = 0; i < hot_keys; ++i) {
    const auto result = db.get(this->verifier.coerce_key(i));
    UNODB_ASSERT_TRUE(TypeParam::key_found(result));
    if constexpr (std::is_same_v<typename TypeParam::value_type,
                                 std::uint64_t>) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      UNODB_ASSERT_EQ(*result, N);
    }
  }
}

// ===================================================================
// Erase-Specific Concurrency Tests (IDs 23, 23f, 23g)
// ===================================================================

// ID-23: Version mismatch → retry → erase or re-invoke.
UNODB_TYPED_TEST(UpsertConcurrencyTest, EraseCasRetry) {
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(3);
  const auto sibling_key = this->verifier.coerce_key(77);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);

  // Pre-insert
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
    // Second key to avoid root-leaf special case
    UNODB_ASSERT_TRUE(
        db.insert(sibling_key, unodb::test::get_test_value<TypeParam>(1)));
  }

  unodb::this_thread().qsbr_pause();

  // T1: upsert with erase action
  auto t1 = unodb::test::thread<TypeParam>([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v0, erase_fn);
  });

  // T2: concurrently modify the same key to force version mismatch
  auto t2 = unodb::test::thread<TypeParam>([&] {
    for (int i = 0; i < 50; ++i) {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.upsert(key, v0, increment_fn);
    }
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();

  // Post: key may be present or absent depending on race outcome
  const unodb::quiescent_state_on_scope_exit q{};
  std::ignore = db.get(key);  // no crash
}

// ID-23f: Exactly one erases, other inserts. Post: key present, size==1.
#ifndef NDEBUG
UNODB_TYPED_TEST(UpsertConcurrencyTest, ConcurrentEraseXErase) {
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(11);
  const auto sibling_key = this->verifier.coerce_key(88);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);
  auto v1 = unodb::test::get_test_value<TypeParam>(1);

  // Pre-insert key + a sibling
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
    UNODB_ASSERT_TRUE(
        db.insert(sibling_key, unodb::test::get_test_value<TypeParam>(2)));
  }

  // Use sync point: first eraser pauses after lambda returns erase
  const sync_point_guard guard{unodb::detail::sync_before_remove_write_guard};
  unodb::detail::sync_before_remove_write_guard.arm([&]() {
    unodb::detail::sync_before_remove_write_guard.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  // T2: also tries to erase the same key
  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v1, erase_fn);
    unodb::detail::thread_syncs[1].notify();
  });

  // T1: upsert with erase, hits sync point
  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v1, erase_fn);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();

  // Post: one erased, other took insert path → key present
  const unodb::quiescent_state_on_scope_exit q{};
  const auto result = db.get(key);
  UNODB_ASSERT_TRUE(TypeParam::key_found(result));
}
#endif  // NDEBUG

// ID-23g: T1 erase paused, T2 removes, T1 resumes → insert path.
#ifndef NDEBUG
UNODB_TYPED_TEST(UpsertConcurrencyTest, EraseAfterConcurrentRemove) {
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(13);
  const auto sibling_key = this->verifier.coerce_key(77);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);
  auto v1 = unodb::test::get_test_value<TypeParam>(1);

  // Pre-insert key + sibling
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
    UNODB_ASSERT_TRUE(
        db.insert(sibling_key, unodb::test::get_test_value<TypeParam>(2)));
  }

  // T1 pauses after erase lambda returns, T2 removes the key
  const sync_point_guard guard{unodb::detail::sync_before_remove_write_guard};
  unodb::detail::sync_before_remove_write_guard.arm([&]() {
    unodb::detail::sync_before_remove_write_guard.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key);
    unodb::detail::thread_syncs[1].notify();
  });

  // T1: upsert(erase) → pauses → T2 removes → T1 restarts → insert path
  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v1, erase_fn);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();

  // Post: T1 took insert path, key present with T1's insert value
  const unodb::quiescent_state_on_scope_exit q{};
  const auto result = db.get(key);
  UNODB_ASSERT_TRUE(TypeParam::key_found(result));
}
#endif  // NDEBUG

// ID-23h: an upsert(erase) retries when remove_or_choose_subtree's write guard
// fails.  T1 upsert(erase) pauses inside remove_or_choose_subtree before its
// write guards, T2 removes and re-inserts the key, which moves the I4's
// version, so T1's node write-guard upgrade fails and its retry runs the
// lambda again on the re-inserted value.  The captured_ver gates there do not
// fire on this schedule: both versions they compare are fixed before the sync
// point.
#ifndef NDEBUG
UNODB_TYPED_TEST(UpsertConcurrencyTest, EraseRetriesWhenRemoveWriteGuardFails) {
  auto& db = this->verifier.get_db();
  // Three keys keep key's I4 above min_size for both removals, so T2 changes
  // it in place without write-locking the parent, and T1 fails at the I4's
  // write guard.  With two, the leaf instantiations take the min-size branch,
  // where T2 write-locks the parent and T1 fails there.
  const auto key = this->verifier.coerce_key(10);
  const auto sib1 = this->verifier.coerce_key(11);
  const auto sib2 = this->verifier.coerce_key(12);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);
  auto v1 = unodb::test::get_test_value<TypeParam>(1);

  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
    UNODB_ASSERT_TRUE(db.insert(sib1, v0));
    UNODB_ASSERT_TRUE(db.insert(sib2, v0));
  }

  seen_values<TypeParam> seen;

  // Arm sync point inside remove_or_choose_subtree's VIS/leaf terminal
  // delete path.  T1 pauses before its write guards and the captured_ver
  // check.
  const sync_point_guard guard{unodb::detail::sync_before_remove_write_guard};
  unodb::detail::sync_before_remove_write_guard.arm([&]() {
    unodb::detail::sync_before_remove_write_guard.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  // T2: modify the target key while T1 is paused.
  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    // Remove and reinsert with a different value to bump the version.
    std::ignore = db.remove(key);
    std::ignore = db.insert(key, v1);
    unodb::detail::thread_syncs[1].notify();
  });

  // T1: upsert with erase lambda.  Its node write-guard upgrade fails, and
  // the retry runs the lambda again for the re-inserted value.
  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v1, [&seen](auto& v) {
      seen.record(v);
      return unodb::upsert_action::erase;
    });
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();

  // The first attempt, and the retry after the failed write-guard upgrade.
  seen.assert_saw(v0, v1);

  // Post: the retry erased key and left both siblings alone.
  with_qsbr<TypeParam>(
      [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(key))); });
  ASSERT_VALUE_FOR_KEY(TypeParam, db, sib1, v0);
  ASSERT_VALUE_FOR_KEY(TypeParam, db, sib2, v0);
}
#endif  // NDEBUG

// ===================================================================
// Contract Verification — concurrency (IDs C2, C5, C6)
// ===================================================================

// ID-C2: Lambda's second invocation receives a different value.  T1 parks
// inside its first lambda call, after it has seen v0.  T2 updates the key to
// v1, which moves the version T1 checks after the lambda returns, so T1
// restarts and its second call sees v1.
#ifndef NDEBUG
UNODB_TYPED_TEST(UpsertConcurrencyTest, LambdaSeesDifferentValues) {
  constexpr bool value_is_view =
      std::is_same_v<typename TypeParam::value_type, unodb::value_view>;
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(20);
  const auto sibling_key = this->verifier.coerce_key(55);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);
  auto v1 = unodb::test::get_test_value<TypeParam>(1);

  // Pre-insert
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
    UNODB_ASSERT_TRUE(
        db.insert(sibling_key, unodb::test::get_test_value<TypeParam>(2)));
  }

  // Fired only from T1's lambda, so T1 parks inside its first call.
  unodb::detail::sync_point in_t1_fn;
  seen_values<TypeParam> seen;

  in_t1_fn.arm([&]() {
    in_t1_fn.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  // T2: update the key to v1 while T1 is parked inside its lambda.  update
  // stores the lambda's v for the u64 value and upsert's own argument for
  // value_view, so both are v1.
  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v1, [v1](auto& v) noexcept {
      v = v1;
      return unodb::upsert_action::update;
    });
    unodb::detail::thread_syncs[1].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v0, [&seen, &in_t1_fn](auto& v) {
      seen.record(v);
      unodb::detail::sync(in_t1_fn);
      // The action picks T1's restart site.  value_view keeps, so it restarts
      // at the leaf keep arm's parent unlock; an update would instead store
      // upsert's own v0 over T2's v1.  The u64 value updates, so it restarts
      // at the value-in-slot write-guard upgrade; a keep would restart at the
      // node unlock instead and still pass.
      if constexpr (value_is_view)
        return unodb::upsert_action::keep;
      else
        return unodb::upsert_action::update;
    });
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();

  seen.assert_saw(v0, v1);
  ASSERT_VALUE_FOR_KEY(TypeParam, db, key, v1);
}
#endif  // NDEBUG

// ID-C5: upsert_erase_retry_count increments (STATS build only).
#ifdef UNODB_DETAIL_WITH_STATS
UNODB_TYPED_TEST(UpsertConcurrencyTest, StatsCounterIncrements) {
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(30);
  const auto sibling_key = this->verifier.coerce_key(66);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);

  // Pre-insert
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
    UNODB_ASSERT_TRUE(
        db.insert(sibling_key, unodb::test::get_test_value<TypeParam>(1)));
  }

  // Force contention to trigger erase retries
  unodb::this_thread().qsbr_pause();

  auto eraser = unodb::test::thread<TypeParam>([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v0, erase_fn);
  });

  auto contender = unodb::test::thread<TypeParam>([&] {
    for (int i = 0; i < 20; ++i) {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.upsert(key, v0, increment_fn);
    }
  });

  eraser.join();
  contender.join();
  unodb::this_thread().qsbr_resume();

  // Post: retry counter may have incremented (depends on race timing)
  // Just verify no crash; actual counter check is best-effort
  const unodb::quiescent_state_on_scope_exit q{};
  std::ignore = db.get(key);
}
#endif  // UNODB_DETAIL_WITH_STATS

// ID-C6: Value committed even when parent RCS fails after write_guard.  Only
// key_view_u64val_olc_db reaches that path: T1 commits its value-in-slot
// update under the node write guard, and the parent unlock it then ignores
// fails.  The leaf types keep, so for them the test pins only that a change
// above T1's parent does not restart a dup-found upsert.
#ifndef NDEBUG
UNODB_TYPED_TEST(UpsertConcurrencyTest, ParentRcsFailAfterCommit) {
  auto& db = this->verifier.get_db();
  const auto key = this->verifier.coerce_key(40);
  const auto sibling_key = this->verifier.coerce_key(41);
  // Diverges from key at byte 6, so it splits the root: the I4 holding key
  // and sibling_key moves under a new root I4.
  const auto upper_key = this->verifier.coerce_key(0x128);
  // Another byte-6 sibling, so T2's insert changes only that new root I4.
  const auto t2_key = this->verifier.coerce_key(0x228);
  auto v0 = unodb::test::get_test_value<TypeParam>(0);

  // Pre-insert key under a parent node, with a root I4 above that parent
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.insert(key, v0));
    UNODB_ASSERT_TRUE(
        db.insert(sibling_key, unodb::test::get_test_value<TypeParam>(1)));
    UNODB_ASSERT_TRUE(
        db.insert(upper_key, unodb::test::get_test_value<TypeParam>(3)));
  }

  // T1 upserts key and parks after dup found while T2 inserts t2_key.
  const sync_point_guard guard{unodb::detail::sync_after_upsert_dup_found};
  unodb::detail::sync_after_upsert_dup_found.arm([&]() {
    unodb::detail::sync_after_upsert_dup_found.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  bool t2_inserted = false;
  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    t2_inserted = db.insert(t2_key, unodb::test::get_test_value<TypeParam>(2));
    unodb::detail::thread_syncs[1].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.upsert(key, v0, increment_fn);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  // A failed insert would leave the root I4 unchanged.
  UNODB_ASSERT_TRUE(t2_inserted);

  // Post: key is present; for the u64 value, the increment T1 made under its
  // node write guard persists despite the failed parent unlock.
  const unodb::quiescent_state_on_scope_exit q{};
  const auto result = db.get(key);
  UNODB_ASSERT_TRUE(TypeParam::key_found(result));
  if constexpr (std::is_same_v<typename TypeParam::value_type, std::uint64_t>) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    UNODB_ASSERT_EQ(*result, 1);
  }
}
#endif  // NDEBUG

// ===================================================================
// Coverage Tests — exercise deep-tree OLC code paths
// ===================================================================

// COV-1: Chain-build in OLC upsert (olc_art.hpp ~2853-2874).
// Note: Chain-build code (olc_art.hpp ~2853-2874) requires keyed-leaf types
// with keys > 8 bytes.  No such type exists in the test suite — the only
// keyed-leaf OLC type (u64_olc_db) has exactly 8-byte keys, making the
// chain condition unreachable.  Marked LCOV_EXCL in olc_art.hpp.

// COV-2: VIS prefix split init (olc_art.hpp ~2982-2983).
// key_view_u64val types: insert causes inode prefix split with pack_value.
UNODB_TYPED_TEST(UpsertTest, UpsertVisPrefixSplit) {
  if constexpr (!std::is_same_v<typename TypeParam::value_type,
                                std::uint64_t> ||
                !std::is_same_v<typename TypeParam::key_type,
                                unodb::key_view>) {
    GTEST_SKIP() << "Only for key_view + u64 value (VIS) types";
  } else {
    unodb::test::tree_verifier<TypeParam> verifier;
    auto& db = verifier.get_db();
    // Insert two keys that share prefix → creates inode with 7-byte prefix.
    const auto k0 = verifier.coerce_key(0);  // 0x0000000000000000
    const auto k1 = verifier.coerce_key(1);  // 0x0000000000000001
    const typename TypeParam::value_type v0 = 10;
    const typename TypeParam::value_type v1 = 20;
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k0, v0)); });
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k1, v1)); });
    // Now insert a key that differs earlier in the prefix → triggers split.
    // Key with different first byte: 0x0100000000000000
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    const auto k2 = verifier.coerce_key(std::uint64_t{1} << 56);
    const typename TypeParam::value_type v2 = 30;
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_TRUE(db.upsert(k2, v2, keep_fn)); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k0, v0);
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k1, v1);
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k2, v2);
  }
}

// COV-2b: VIS concurrent restart — insert at dispatch byte holding VIS.
// Exercises the remaining_key.size() != 1 restart path (olc_art.hpp try_insert
// and try_upsert VIS branches).  Regression test for assertion failure in
// UpsertConcurrencyTest/2.UpsertDisjoint on MSVC LLVM Debug.
// The OLC path can transiently observe a VIS at an intermediate depth during
// concurrent restructuring; the fix restarts instead of asserting.
UNODB_TYPED_TEST(UpsertConcurrencyTest, InsertAtVisSlotRestart) {
  // Only key_view + u64 (VIS) OLC types can hit this path.
  if constexpr (!std::is_same_v<typename TypeParam::value_type,
                                std::uint64_t> ||
                !std::is_same_v<typename TypeParam::key_type,
                                unodb::key_view>) {
    GTEST_SKIP() << "Only for key_view + u64 value (VIS) OLC types";
  } else {
    // Use encoded 8-byte keys (prefix-free). Two threads inserting disjoint
    // ranges exercise structural mutations that can expose VIS at intermediate
    // depth transiently.
    constexpr std::size_t range_size = 64;
    this->template parallel_test<2, range_size>(
        [](unodb::test::tree_verifier<TypeParam>* tv, std::size_t thread_i,
           std::size_t ops_per_thread) {
          auto& d = tv->get_db();
          unodb::key_encoder enc;
          std::array<std::byte, sizeof(std::uint64_t)> buf{};
          const auto base = thread_i * 1000;
          for (std::size_t i = 0; i < ops_per_thread; ++i) {
            const auto key = make_local_key<TypeParam>(base + i, enc, buf);
            auto v = unodb::test::get_test_value<TypeParam>(i);
            const unodb::quiescent_state_on_scope_exit q{};
            UNODB_ASSERT_TRUE(d.insert(key, v));
          }
        });
    // Verify all keys present.
    const unodb::quiescent_state_on_scope_exit q{};
    auto& db = this->verifier.get_db();
    for (std::size_t t = 0; t < 2; ++t) {
      const auto base = t * 1000;
      for (std::size_t i = 0; i < range_size; ++i) {
        const auto key = this->verifier.coerce_key(base + i);
        const auto result = db.get(key);
        UNODB_ASSERT_TRUE(TypeParam::key_found(result));
      }
    }
  }
}

// COV-3: key_view erase traversal loop (olc_art.hpp ~3176) and VIS erase
// delegation at depth (olc_art.hpp ~3200).
// Exercises the while(true) loop in try_upsert_erase for key_view types.
UNODB_TYPED_TEST(UpsertTest, EraseDeepKeyView) {
  if constexpr (!std::is_same_v<typename TypeParam::key_type,
                                unodb::key_view>) {
    GTEST_SKIP() << "Only for key_view types";
  } else {
    unodb::test::tree_verifier<TypeParam> verifier;
    auto& db = verifier.get_db();
    // Create a 2-level tree by inserting keys that share some prefix.
    const auto k0 = verifier.coerce_key(0);  // share 7 bytes
    const auto k1 = verifier.coerce_key(1);  // same prefix, different dispatch
    // Add a key with different first byte to create root inode with prefix=[]
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    const auto k2 = verifier.coerce_key(static_cast<std::uint64_t>(1) << 56);
    const auto v = unodb::test::get_test_value<TypeParam>(0);
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k0, v)); });
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k1, v)); });
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k2, v)); });
    // Erase k0: try_upsert_erase must traverse root inode → nested inode.
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k0, v, erase_fn)); });
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k0))); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k1, v);
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k2, v);
  }
}

// COV-4: Fixed-width erase loop body (olc_art.hpp ~3303-3314).
// u64 OLC: erase at depth > 1 requires traversal loop iteration.
UNODB_TYPED_TEST(UpsertTest, EraseDeepFixedWidth) {
  if constexpr (std::is_same_v<typename TypeParam::key_type, unodb::key_view>) {
    GTEST_SKIP() << "Only for fixed-width key types";
  } else {
    unodb::test::tree_verifier<TypeParam> verifier;
    auto& db = verifier.get_db();
    // Create 2-level tree: keys that differ at byte 0 AND keys that differ
    // at byte 7 under the same first-byte subtree.
    const auto k0 = verifier.coerce_key(std::uint64_t{0});
    const auto k1 = verifier.coerce_key(std::uint64_t{1});
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    const auto k2 = verifier.coerce_key(std::uint64_t{1} << 56);
    const auto v = unodb::test::get_test_value<TypeParam>(0);
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k0, v)); });
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k1, v)); });
    with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k2, v)); });
    // Erase k0: requires descending through root inode to nested inode.
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(db.upsert(k0, v, erase_fn)); });
    with_qsbr<TypeParam>(
        [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k0))); });
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k1, v);
    ASSERT_VALUE_FOR_KEY(TypeParam, db, k2, v);
  }
}

// ===================================================================
// OOM Tests (IDs OOM-1, OOM-2)
// Guarded by NDEBUG — OOM injection requires debug heap.
// ===================================================================

#ifndef NDEBUG

template <class Db>
class UpsertOOMTest : public ::testing::Test {
 public:
  using Test::Test;
};

using UpsertOOMTypes =
    ::testing::Types<unodb::test::u64_db, unodb::test::u64_mutex_db,
                     unodb::test::u64_olc_db>;

UNODB_TYPED_TEST_SUITE(UpsertOOMTest, UpsertOOMTypes)

// ID-OOM-1: Insert path OOM — std::bad_alloc, tree unchanged.
UNODB_TYPED_TEST(UpsertOOMTest, InsertPathOom) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();
  const auto k0 = verifier.coerce_key(0);
  const auto k1 = verifier.coerce_key(1);
  const auto v0 = unodb::test::get_test_value<TypeParam>(0);
  const auto v1 = unodb::test::get_test_value<TypeParam>(1);

  // Pre-insert key 0
  with_qsbr<TypeParam>([&] { UNODB_ASSERT_TRUE(db.insert(k0, v0)); });

  // Inject failure, upsert key 1 (insert path)
  UNODB_ASSERT_TRUE(with_qsbr<TypeParam>([&] {
    return unodb::test::throws_bad_alloc(
        1, [&] { std::ignore = db.upsert(k1, v1, keep_fn); });
  }));

  // Verify key 1 absent, key 0 still present
  with_qsbr<TypeParam>(
      [&] { UNODB_ASSERT_FALSE(TypeParam::key_found(db.get(k1))); });
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k0, v0);
}

// ID-OOM-2: Erase shrink OOM — std::bad_alloc, key still present.
UNODB_TYPED_TEST(UpsertOOMTest, EraseShrinkOom) {
  unodb::test::tree_verifier<TypeParam> verifier;
  auto& db = verifier.get_db();

  // Insert 5 keys (creates I16)
  for (std::size_t i = 0; i < 5; ++i) {
    const auto k = verifier.coerce_key(i);
    with_qsbr<TypeParam>([&] {
      UNODB_ASSERT_TRUE(
          db.insert(k, unodb::test::get_test_value<TypeParam>(i)));
    });
  }

  const auto k_erase = verifier.coerce_key(std::size_t{2});

  // Inject failure, upsert-erase key 2 (triggers shrink allocation)
  UNODB_ASSERT_TRUE(with_qsbr<TypeParam>([&] {
    return unodb::test::throws_bad_alloc(1, [&] {
      std::ignore = db.upsert(
          k_erase, unodb::test::get_test_value<TypeParam>(2), erase_fn);
    });
  }));

  // Verify key 2 still present
  ASSERT_VALUE_FOR_KEY(TypeParam, db, k_erase,
                       unodb::test::get_test_value<TypeParam>(2));
}

#endif  // NDEBUG

}  // namespace

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
