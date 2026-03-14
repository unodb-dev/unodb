// Copyright 2019-2025 UnoDB contributors
#ifndef UNODB_DETAIL_DB_TEST_UTILS_HPP
#define UNODB_DETAIL_DB_TEST_UTILS_HPP

// Should be the first include
#include "global.hpp"

// IWYU pragma: no_include <__cstddef/byte.h>
// IWYU pragma: no_include <__ostream/basic_ostream.h>
// IWYU pragma: no_include <__vector/vector.h>
// IWYU pragma: no_include <iomanip>
// IWYU pragma: no_include <string>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>  // IWYU pragma: keep

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "art.hpp"
#include "art_common.hpp"
#include "art_internal.hpp"
#include "assert.hpp"
#include "gtest_utils.hpp"
#include "mutex_art.hpp"
#include "node_type.hpp"
#include "olc_art.hpp"
#include "qsbr.hpp"
#include "test_heap.hpp"

extern template class unodb::db<std::uint64_t, unodb::value_view>;
extern template class unodb::mutex_db<std::uint64_t, unodb::value_view>;
extern template class unodb::olc_db<std::uint64_t, unodb::value_view>;

extern template class unodb::db<unodb::key_view, unodb::value_view>;
extern template class unodb::mutex_db<unodb::key_view, unodb::value_view>;
extern template class unodb::olc_db<unodb::key_view, unodb::value_view>;

extern template class unodb::db<unodb::key_view, std::uint64_t>;
extern template class unodb::mutex_db<unodb::key_view, std::uint64_t>;
extern template class unodb::olc_db<unodb::key_view, std::uint64_t>;

namespace unodb::test {

template <class TestDb>
constexpr bool is_olc_db =
    std::is_same_v<TestDb, unodb::olc_db<typename TestDb::key_type,
                                         typename TestDb::value_type>>;

template <class TestDb>
constexpr bool is_mutex_db =
    std::is_same_v<TestDb, unodb::mutex_db<typename TestDb::key_type,
                                           typename TestDb::value_type>>;

template <class TestDb>
using thread = typename std::conditional_t<is_olc_db<TestDb>,
                                           unodb::qsbr_thread, std::thread>;

constexpr auto test_value_1 = std::array<std::byte, 1>{std::byte{0x00}};
constexpr auto test_value_2 =
    std::array<std::byte, 2>{std::byte{0x00}, std::byte{0x02}};
constexpr auto test_value_3 =
    std::array<std::byte, 3>{std::byte{0x03}, std::byte{0x00}, std::byte{0x01}};
constexpr auto test_value_4 = std::array<std::byte, 4>{
    std::byte{0x04}, std::byte{0x01}, std::byte{0x00}, std::byte{0x02}};
constexpr auto test_value_5 =
    std::array<std::byte, 5>{std::byte{0x05}, std::byte{0xF4}, std::byte{0xFF},
                             std::byte{0x00}, std::byte{0x01}};
constexpr auto empty_test_value = std::array<std::byte, 0>{};

constexpr std::array<unodb::value_view, 6> test_values = {
    unodb::value_view{test_value_1},     // [0] { 00              }
    unodb::value_view{test_value_2},     // [1] { 00 02           }
    unodb::value_view{test_value_3},     // [2] { 03 00 01        }
    unodb::value_view{test_value_4},     // [3] { 04 01 00 02     }
    unodb::value_view{test_value_5},     // [4] { 05 F4 FF 00 01  }
    unodb::value_view{empty_test_value}  // [5] {                 }
};

constexpr std::array<std::uint64_t, 6> test_values_u64 = {0, 1, 2, 3, 4, 5};

/// Return a test value appropriate for the db's value type.
template <class Db>
constexpr typename Db::value_type get_test_value(std::size_t i) {
  if constexpr (std::is_same_v<typename Db::value_type, unodb::value_view>)
    return test_values[i % test_values.size()];
  else
    return test_values_u64[i % test_values_u64.size()];
}

namespace detail {

UNODB_DETAIL_DISABLE_CLANG_WARNING("-Wused-but-marked-unused")
UNODB_DETAIL_DISABLE_CLANG_WARNING("-Wunused-parameter")

template <class Db>
void assert_value_eq(const typename Db::get_result& result,
                     typename Db::value_type expected) noexcept {
  if constexpr (is_mutex_db<Db>) {
    UNODB_DETAIL_ASSERT(result.second.owns_lock());
    UNODB_DETAIL_ASSERT(result.first.has_value());
    if constexpr (std::is_same_v<typename Db::value_type, unodb::value_view>)
      UNODB_ASSERT_TRUE(std::ranges::equal(*result.first, expected));
    else
      UNODB_ASSERT_EQ(*result.first, expected);
  } else {
    UNODB_DETAIL_ASSERT(result.has_value());
    if constexpr (std::is_same_v<typename Db::value_type, unodb::value_view>)
      UNODB_ASSERT_TRUE(std::ranges::equal(*result, expected));
    else
      UNODB_ASSERT_EQ(*result, expected);
  }
}

template <class Db>
void assert_not_found(const typename Db::get_result& result) noexcept {
  if constexpr (is_mutex_db<Db>) {
    UNODB_DETAIL_ASSERT(!result.second.owns_lock());
    UNODB_DETAIL_ASSERT(!result.first.has_value());
  } else {
    UNODB_DETAIL_ASSERT(!result.has_value());
  }
}

template <class Db>
void do_assert_result_eq(const Db& db, typename Db::key_type key,
                         typename Db::value_type expected, const char* file,
                         int line) {
  std::ostringstream msg;
  unodb::detail::dump_key(msg, key);
  const testing::ScopedTrace trace(file, line, msg.str());
  const auto result = db.get(key);
  if (!Db::key_found(result)) {
    // LCOV_EXCL_START
    std::cerr << "db.get did not find ";
    unodb::detail::dump_key(std::cerr, key);
    std::cerr << '\n';
    db.dump(std::cerr);
    FAIL();
    // LCOV_EXCL_STOP
  }
  assert_value_eq<Db>(result, expected);
}

UNODB_DETAIL_RESTORE_CLANG_WARNINGS()

template <class Db>
void assert_result_eq(const Db& db, typename Db::key_type key,
                      typename Db::value_type expected, const char* file,
                      int line) {
  if constexpr (is_olc_db<Db>) {
    const quiescent_state_on_scope_exit qsbr_after_get{};
    do_assert_result_eq<Db>(db, key, expected, file, line);
  } else {
    do_assert_result_eq<Db>(db, key, expected, file, line);
  }
}

}  // namespace detail

#define ASSERT_VALUE_FOR_KEY(DbType, test_db, key, expected) \
  detail::assert_result_eq<DbType>(test_db, key, expected, __FILE__, __LINE__)

/// Utility class supporting verification of the system under test.
///
/// \note For the ::key_view cases, the verifier assumes that we are storing u64
/// keys encoded into a ::key_view.  The caller's ::key_view is decoded to
/// obtain the u64 key.  We then encode each u64 value in the specified range
/// into a ::key_view.
template <class Db>
class [[nodiscard]] tree_verifier final {
  using key_type = typename Db::key_type;
  using value_type = typename Db::value_type;

  // replaces the use of try_get(0) with a parameterized key type.
  key_type unused_key{};

  /// \note This is the historical type accepted for unodb::db keys.  It is used
  /// in the unodb::test::tree_verifier to perform explicit type promotion where
  /// unit tests where using implicit type conversion from `int` to
  /// `std::uint64_t`.
  using u64 = std::uint64_t;

  /// Used to wrap the non-owned keys with an owned, comparable, etc. type iff
  /// the the keys of the Db are unodb::key_view.
  using key_wrapper = std::shared_ptr<std::vector<std::byte>>;

  /// Declares the type for the keys of the tree_verifier::map. Since ::key_view
  /// is a non-owned type, we need to copy the data into an owned type that can
  /// be used with the map.
  template <typename Db2 = Db>
  using ikey_type = typename std::conditional<
      (std::is_same_v<typename Db2::key_type, typename unodb::key_view>),
      key_wrapper, typename Db2::key_type>::type;

  /// Convert an external key (Db::key_type) to an internal key (one the
  /// unodb::test::tree_verifier stores in its ground truth key/value
  /// collection).
  [[nodiscard]] ikey_type<Db> to_ikey(typename Db::key_type key) const
      noexcept(!std::is_same_v<key_type, unodb::key_view>) {
    if constexpr (std::is_same_v<key_type, unodb::key_view>) {
      // Allocate a vector, make a copy of the key into the vector,
      // and return a shared_ptr to that vector.
      UNODB_DETAIL_PAUSE_HEAP_TRACKING_GUARD();
      return std::make_shared<std::vector<std::byte>>(key.begin(), key.end());
    } else {
      return key;  // NOP
    }
  }

  template <typename T>
  key_type coerce_key_internal(T key) noexcept(
      !std::is_same_v<key_type, unodb::key_view> ||
      std::is_same_v<T, unodb::key_view>) {
    if constexpr (std::is_same_v<key_type, unodb::key_view>) {  // Db<key_view>?
      if constexpr (std::is_same_v<T, unodb::key_view>) {  // Given key_view?
        // key_view pass through
        return key;
      } else {
        // type promote and then encode the key into a key_view.
        return make_key(static_cast<u64>(key));
      }
    } else {  // type promotion.
      return static_cast<u64>(key);
    }
  }

 public:
  /// Coerce an external key into the Db::key_type.
  ///
  /// \note Historically, the unit tests were written to some mixture of `int`,
  /// `unsigned`, and `std::uint64_t` keys and relied on implicit type
  /// promotion.  This method takes the place of that implicit type promotion
  /// and also handles conversion from such simple keys to unodb::key_view.
  ///
  /// \note Type promotion is explicitly to std::uint64_t since that is the
  /// historical external type against which the unit tests were written.
  ///
  /// \param key An external key.
  ///
  /// \return A key of a type suitable for the db under test.
  //
  // Note: This method is used mostly internally within the tree_verifier.
  // However, it is also used by test_art_iter to from the type specific keys
  // for the db::iterator::seek() API.
  template <typename T>
  key_type coerce_key(T key) noexcept(
      !std::is_same_v<key_type, unodb::key_view> ||
      std::is_same_v<T, unodb::key_view>) {
    return coerce_key_internal(key);
  }

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)

  /// Return an unodb::key_view backed by a `std::array` in an internal
  /// collection whose existence is scoped to the life cycle of the
  /// tree_verifier.
  unodb::key_view make_key(std::uint64_t k) {
    constexpr auto sz{sizeof(k)};
    UNODB_DETAIL_PAUSE_HEAP_TRACKING_GUARD();
    // Encode the key, emplace an array into the list of encoded keys
    // that we are tracking, and copy the encoded key into that
    // emplaced array.
    unodb::key_encoder enc;
    const auto kv{enc.encode(k).get_key_view()};
    key_views.emplace_back(std::array<std::byte, sz>{});
    auto& a = key_views.back();        // a *reference* to data emplaced_back.
    std::ranges::copy(kv, a.begin());  // copy data into array.
    // Return a key_view backed by the array that we just put on that
    // list.
    return unodb::key_view(a.data(), sz);  // view of array's data.
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

 private:
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
  // Replace std::enable_if_t in the next two methods with
  // requires(std::is_same_v) when LLVM 15 is the oldest supported LLVM version.
  // Earlier versions give errors of different declarations having identical
  // mangled names.
  // NOLINTBEGIN(modernize-use-constraints)
  template <class Db2 = Db>
  std::enable_if_t<!is_olc_db<Db2>, void> do_insert(key_type k, value_type v) {
    UNODB_ASSERT_TRUE(test_db.insert(k, v));
  }

  template <class Db2 = Db>
  std::enable_if_t<is_olc_db<Db2>, void> do_insert(key_type k, value_type v) {
    const quiescent_state_on_scope_exit qsbr_after_get{};
    UNODB_ASSERT_TRUE(test_db.insert(k, v));
  }
  // NOLINTEND(modernize-use-constraints)
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  void do_remove(key_type k, bool bypass_verifier) {
    if (!bypass_verifier) {
      const auto remove_result = values.erase(to_ikey(k));
      UNODB_ASSERT_EQ(remove_result, 1);
    }

#ifdef UNODB_DETAIL_WITH_STATS
    const auto node_counts_before = test_db.get_node_counts();
    const auto mem_use_before = test_db.get_current_memory_use();
    UNODB_ASSERT_GT(mem_use_before, 0);
    if constexpr (std::is_same_v<value_type, unodb::value_view>) {
      UNODB_ASSERT_GT(node_counts_before[as_i<unodb::node_type::LEAF>], 0);
    }
    const auto growing_inodes_before = test_db.get_growing_inode_counts();
    const auto shrinking_inodes_before = test_db.get_shrinking_inode_counts();
    const auto key_prefix_splits_before = test_db.get_key_prefix_splits();
#endif  // UNODB_DETAIL_WITH_STATS

    try {
      if (!test_db.remove(k)) {
        // LCOV_EXCL_START
        std::cerr << "test_db.remove failed for ";
        unodb::detail::dump_key(std::cerr, k);
        std::cerr << '\n';
        test_db.dump(std::cerr);
        FAIL();
        // LCOV_EXCL_STOP
      }
    } catch (...) {
#ifdef UNODB_DETAIL_WITH_STATS
      if (!parallel_test) {
        UNODB_ASSERT_EQ(mem_use_before, test_db.get_current_memory_use());
        UNODB_ASSERT_THAT(test_db.get_node_counts(),
                          ::testing::ElementsAreArray(node_counts_before));
        UNODB_ASSERT_THAT(test_db.get_growing_inode_counts(),
                          ::testing::ElementsAreArray(growing_inodes_before));
        UNODB_ASSERT_THAT(test_db.get_shrinking_inode_counts(),
                          ::testing::ElementsAreArray(shrinking_inodes_before));
        UNODB_ASSERT_EQ(test_db.get_key_prefix_splits(),
                        key_prefix_splits_before);
      }
#endif  // UNODB_DETAIL_WITH_STATS
      throw;
    }

#ifdef UNODB_DETAIL_WITH_STATS
    if (!parallel_test) {
      const auto mem_use_after = test_db.get_current_memory_use();
      UNODB_ASSERT_LT(mem_use_after, mem_use_before);

      const auto leaf_count_after =
          test_db.template get_node_count<::unodb::node_type::LEAF>();
      if constexpr (std::is_same_v<value_type, unodb::value_view>) {
        UNODB_ASSERT_EQ(leaf_count_after,
                        node_counts_before[as_i<unodb::node_type::LEAF>] - 1);
      }
    }
#endif  // UNODB_DETAIL_WITH_STATS
  }

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
  // Replace std::enable_if_t in the next two methods with
  // requires(std::is_same_v) when LLVM 15 is the oldest supported LLVM version.
  // Earlier versions give errors of different declarations having identical
  // mangled names.
  // NOLINTBEGIN(modernize-use-constraints)
  template <class Db2 = Db, typename T>
  std::enable_if_t<!is_olc_db<Db2>, void> do_try_remove_missing_key(
      T absent_key) {
    UNODB_ASSERT_FALSE(test_db.remove(coerce_key(absent_key)));
  }

  template <class Db2 = Db, typename T>
  std::enable_if_t<is_olc_db<Db2>, void> do_try_remove_missing_key(
      T absent_key) {
    const quiescent_state_on_scope_exit qsbr_after_get{};
    UNODB_ASSERT_FALSE(test_db.remove(coerce_key(absent_key)));
  }
  // NOLINTEND(modernize-use-constraints)
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  // cppcheck-suppress passedByValue
  void insert_internal(key_type k, value_type v, bool bypass_verifier = false) {
    const auto empty_before = test_db.empty();
#ifdef UNODB_DETAIL_WITH_STATS
    const auto mem_use_before =
        parallel_test ? 0 : test_db.get_current_memory_use();
    const auto node_counts_before = test_db.get_node_counts();
    const auto growing_inodes_before = test_db.get_growing_inode_counts();
    const auto shrinking_inodes_before = test_db.get_shrinking_inode_counts();
    const auto key_prefix_splits_before = test_db.get_key_prefix_splits();
#endif  // UNODB_DETAIL_WITH_STATS

    try {
      do_insert(k, v);
    } catch (...) {
      if (!parallel_test) {
        UNODB_ASSERT_EQ(empty_before, test_db.empty());
#ifdef UNODB_DETAIL_WITH_STATS
        UNODB_ASSERT_EQ(mem_use_before, test_db.get_current_memory_use());
        UNODB_ASSERT_THAT(test_db.get_node_counts(),
                          ::testing::ElementsAreArray(node_counts_before));
        UNODB_ASSERT_THAT(test_db.get_growing_inode_counts(),
                          ::testing::ElementsAreArray(growing_inodes_before));
        UNODB_ASSERT_THAT(test_db.get_shrinking_inode_counts(),
                          ::testing::ElementsAreArray(shrinking_inodes_before));
        UNODB_ASSERT_EQ(test_db.get_key_prefix_splits(),
                        key_prefix_splits_before);
#endif  // UNODB_DETAIL_WITH_STATS
      }
      throw;
    }

    UNODB_ASSERT_FALSE(test_db.empty());

#ifdef UNODB_DETAIL_WITH_STATS
    const auto mem_use_after = test_db.get_current_memory_use();
    if (parallel_test)
      UNODB_ASSERT_GT(mem_use_after, 0);
    else if constexpr (std::is_same_v<value_type, unodb::value_view>)
      UNODB_ASSERT_LT(mem_use_before, mem_use_after);

    const auto leaf_count_after =
        test_db.template get_node_count<unodb::node_type::LEAF>();
    if constexpr (std::is_same_v<value_type, unodb::value_view>) {
      if (parallel_test)
        UNODB_ASSERT_GT(leaf_count_after, 0);
      else
        UNODB_ASSERT_EQ(leaf_count_after,
                        node_counts_before[as_i<unodb::node_type::LEAF>] + 1);
    }
#endif  // UNODB_DETAIL_WITH_STATS

    if (!bypass_verifier) {
#ifndef NDEBUG
      allocation_failure_injector::reset();
#endif
      UNODB_DETAIL_PAUSE_HEAP_TRACKING_GUARD();
      const auto [pos, insert_succeeded] = values.try_emplace(to_ikey(k), v);
      (void)pos;
      UNODB_ASSERT_TRUE(insert_succeeded);
    }
  }

  // cppcheck-suppress passedByValue
  void insert_key_range_internal(key_type start_key, std::size_t count,
                                 bool bypass_verifier = false) {
    if constexpr (std::is_same_v<key_type, key_view>) {
      // decode to figure out the start key for the loop.
      unodb::key_decoder dec(start_key);
      std::uint64_t start_key_dec;
      dec.decode(start_key_dec);
      unodb::key_encoder enc;
      for (auto key = start_key_dec; key < start_key_dec + count; ++key) {
        insert(enc.reset().encode(key).get_key_view(), get_test_value<Db>(key),
               bypass_verifier);
      }
    } else {
      for (auto key = start_key; key < start_key + count; ++key) {
        insert(key, get_test_value<Db>(key), bypass_verifier);
      }
    }
  }

 public:
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26455)
  explicit tree_verifier(bool parallel_test_ = false)
      : parallel_test{parallel_test_} {
    assert_empty();
#ifdef UNODB_DETAIL_WITH_STATS
    assert_growing_inodes({0, 0, 0, 0});
    assert_shrinking_inodes({0, 0, 0, 0});
    assert_key_prefix_splits(0);
#endif  // UNODB_DETAIL_WITH_STATS
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  template <typename T>
  void insert(T k, value_type v, bool bypass_verifier = false) {
    insert_internal(coerce_key(k), v, bypass_verifier);
  }

  template <typename T>
  void insert_key_range(T start_key, std::size_t count,
                        bool bypass_verifier = false) {
    insert_key_range_internal(coerce_key(start_key), count, bypass_verifier);
  }

  template <typename T>
  bool try_insert(T k, value_type v) {
    return test_db.insert(coerce_key(k), v);
  }

  template <typename T>
  void preinsert_key_range_to_verifier_only(T start_key1, std::size_t count) {
    const auto start_key{coerce_key(start_key1)};
    if constexpr (std::is_same_v<key_type, key_view>) {
      unodb::key_decoder dec(start_key);
      std::uint64_t start_key_dec;
      dec.decode(start_key_dec);
      unodb::key_encoder enc;
      for (auto key = start_key_dec; key < start_key_dec + count; ++key) {
        auto tmp = to_ikey(enc.reset().encode(key).get_key_view());
        const auto [pos, insert_succeeded] =
            values.try_emplace(tmp, get_test_value<Db>(key));
        (void)pos;
        UNODB_ASSERT_TRUE(insert_succeeded);
      }
    } else {
      for (auto key = start_key; key < start_key + count; ++key) {
        const auto [pos, insert_succeeded] =
            values.try_emplace(to_ikey(key), get_test_value<Db>(key));
        (void)pos;
        UNODB_ASSERT_TRUE(insert_succeeded);
      }
    }
  }

  template <typename T>
  void insert_preinserted_key_range(T start_key1, std::size_t count) {
    const auto start_key{coerce_key(start_key1)};
    if constexpr (std::is_same_v<key_type, key_view>) {
      unodb::key_decoder dec(start_key);
      std::uint64_t start_key_dec;
      dec.decode(start_key_dec);
      unodb::key_encoder enc;
      for (auto key = start_key_dec; key < start_key_dec + count; ++key) {
        do_insert(enc.reset().encode(key).get_key_view(),
                  get_test_value<Db>(key));
      }
    } else {
      for (auto key = start_key; key < start_key + count; ++key) {
        do_insert(key, get_test_value<Db>(key));
      }
    }
  }

  // Replace std::enable_if_t in the next four methods with
  // requires(std::is_same_v) when LLVM 15 is the oldest supported LLVM version.
  // Earlier versions give errors of different declarations having identical
  // mangled names.
  // NOLINTBEGIN(modernize-use-constraints)
  template <class Db2 = Db, typename T>
  std::enable_if_t<!is_olc_db<Db2>, void> remove(T k,
                                                 bool bypass_verifier = false) {
    do_remove(coerce_key(k), bypass_verifier);
  }

  template <class Db2 = Db, typename T>
  std::enable_if_t<is_olc_db<Db2>, void> remove(T k,
                                                bool bypass_verifier = false) {
    const quiescent_state_on_scope_exit qsbr_after_get{};
    do_remove(coerce_key(k), bypass_verifier);
  }

  template <class Db2 = Db, typename T>
  std::enable_if_t<!is_olc_db<Db2>, void> try_remove(T k) {
    std::ignore = test_db.remove(coerce_key(k));
  }

  template <class Db2 = Db, typename T>
  std::enable_if_t<is_olc_db<Db2>, void> try_remove(T k) {
    const quiescent_state_on_scope_exit qsbr_after_get{};
    std::ignore = test_db.remove(coerce_key(k));
  }
  // NOLINTEND(modernize-use-constraints)

  template <typename T>
  void attempt_remove_missing_keys(std::initializer_list<T> absent_keys) {
#ifdef UNODB_DETAIL_WITH_STATS
    const auto mem_use_before =
        parallel_test ? 0 : test_db.get_current_memory_use();
#endif  // UNODB_DETAIL_WITH_STATS

    for (const auto absent_key : absent_keys) {
      const auto k{coerce_key(absent_key)};

      const auto remove_result = values.erase(to_ikey(k));
      UNODB_ASSERT_EQ(remove_result, 0);
      do_try_remove_missing_key(k);
#ifdef UNODB_DETAIL_WITH_STATS
      if (!parallel_test) {
        UNODB_ASSERT_EQ(mem_use_before, test_db.get_current_memory_use());
      }
#endif  // UNODB_DETAIL_WITH_STATS
    }
  }

  // Replace std::enable_if_t in the next two methods with
  // requires(std::is_same_v) when LLVM 15 is the oldest supported LLVM version.
  // Earlier versions give errors of different declarations having identical
  // mangled names.
  // NOLINTBEGIN(modernize-use-constraints)
  template <class Db2 = Db, typename T>
  std::enable_if_t<!is_olc_db<Db2>, void> try_get(T k) noexcept(
      noexcept(coerce_key(k)) && noexcept(this->test_db.get(coerce_key(k)))) {
    std::ignore = test_db.get(coerce_key(k));
  }

  template <class Db2 = Db, typename T>
  std::enable_if_t<is_olc_db<Db2>, void> try_get(T k) {
    const quiescent_state_on_scope_exit qsbr_after_get{};
    std::ignore = test_db.get(coerce_key(k));
  }
  // NOLINTEND(modernize-use-constraints)

  /// Verify that each key and value in the internal ground truth collection can
  /// be found in the test db. This also performs a full scan of the test db and
  /// verify that each (key,val) visited in lexicographic order (we can't probe
  /// the ground truth with the encoded keys so the number of keys visited by
  /// the scan can be checked, but not whether each key is in the ground truth -
  /// doing that requires knowledge about how the keys were encoded and the
  /// encoding needs to be reversible, which it is not in the general case).
  void check_present_values() const {
    // Probe the test_db for each key, verifying the expected value is found
    // under that key.
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26445)
    for (const auto& [key, value] : values) {
      if constexpr (std::is_same_v<typename Db::key_type, unodb::key_view>) {
        ASSERT_VALUE_FOR_KEY(Db, test_db, *key, value);
      } else {
        ASSERT_VALUE_FOR_KEY(Db, test_db, key, value);
      }
    }
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
    // Scan the test_db.  For each (key,val) visited, verify (a) that each key
    // is visited in lexicographic order; and (b) that each (key,val) pair also
    // appears in the ground truth collection.
    //
    // Note: This depends on the ability to decode the key. Therefore, the
    // caller SHOULD disable this scan when the keys do not support 100%
    // faithful round-trip encoding and decoding.
    UNODB_DETAIL_PAUSE_HEAP_TRACKING_GUARD();
    std::size_t n{0};
    bool first = true;
    std::vector<std::byte> prev_copy{};
    auto fn = [&n, &first, &prev_copy](
                  const unodb::visitor<typename Db::iterator>& visitor) {
      UNODB_DETAIL_DISABLE_MSVC_WARNING(26445)
      const auto kv = static_cast<unodb::key_view>(visitor.get_key());
      UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

      if (UNODB_DETAIL_UNLIKELY(first)) {
        prev_copy.assign(kv.begin(), kv.end());
        first = false;
      } else {
        const unodb::key_view prev{prev_copy};
        UNODB_EXPECT_LT(unodb::detail::compare(prev, kv), 0);
        prev_copy.assign(kv.begin(), kv.end());
      }
      n++;
      return false;
    };
    const_cast<Db&>(test_db).scan(fn);
    // FIXME(thompsonbry) variable length keys - enable this assert.
    // 3 OOM tests are failing (for each Db type) when this is enabled
    // (off by one).  What is going on there?
    //
    // const auto sz = values.size();  // #of (key,val) pairs expected.
    // UNODB_EXPECT_EQ(sz, n);
  }

  template <typename T>
  void check_absent_keys(std::initializer_list<T> absent_keys) noexcept(
      noexcept(this->try_get(coerce_key(*absent_keys.begin())))) {
    for (const auto absent_key : absent_keys) {
      const auto k{coerce_key(absent_key)};
      UNODB_ASSERT_EQ(values.find(to_ikey(k)), values.cend());
      try_get(k);
    }
  }

  //
  // scan API
  //

  // TODO(thompsonbry) Add verification against ground truth into the
  // scan() API calls?  Maybe by having the caller NOT specify the
  // lambda FN if they want built-in verification?  Some of the scan
  // UTs are explicit about how the ground truth is populated, but
  // their validation could still be replaced by validating against
  // the tree_verifier::values map.

  template <typename FN, typename T>
  void scan(FN fn, bool fwd = true) {
    test_db.scan_from(fn, fwd);
  }

  template <typename FN, typename T>
  void scan_from(T from_key, FN fn, bool fwd = true) {
    const auto fk{coerce_key(from_key)};
    test_db.scan_from(fk, fn, fwd);
  }

  template <typename FN, typename T>
  void scan_range(T from_key, T to_key, FN fn) {
    const auto fk{coerce_key(from_key)};
    const auto tk{coerce_key(to_key)};
    test_db.scan_range(fk, tk, fn);
  }

  void assert_empty() const {
    UNODB_ASSERT_TRUE(test_db.empty());

#ifdef UNODB_DETAIL_WITH_STATS
    UNODB_ASSERT_EQ(test_db.get_current_memory_use(), 0);

    assert_node_counts({0, 0, 0, 0, 0});
#endif  // UNODB_DETAIL_WITH_STATS
  }

#ifdef UNODB_DETAIL_WITH_STATS

  void assert_node_counts(
      const node_type_counter_array& expected_node_counts) const {
    // Dump the tree to a string. Do not attempt to check the dump format, only
    // that dumping does not crash
    std::stringstream dump_sink;
    test_db.dump(dump_sink);

    const auto actual_node_counts = test_db.get_node_counts();
    UNODB_ASSERT_THAT(actual_node_counts,
                      ::testing::ElementsAreArray(expected_node_counts));
  }

  constexpr void assert_growing_inodes(
      const inode_type_counter_array& expected_growing_inode_counts)
      const noexcept {
    const auto actual_growing_inode_counts = test_db.get_growing_inode_counts();
    UNODB_ASSERT_THAT(
        actual_growing_inode_counts,
        ::testing::ElementsAreArray(expected_growing_inode_counts));
  }

  constexpr void assert_shrinking_inodes(
      const inode_type_counter_array& expected_shrinking_inode_counts) {
    const auto actual_shrinking_inode_counts =
        test_db.get_shrinking_inode_counts();
    UNODB_ASSERT_THAT(
        actual_shrinking_inode_counts,
        ::testing::ElementsAreArray(expected_shrinking_inode_counts));
  }

  constexpr void assert_key_prefix_splits(std::uint64_t splits) const noexcept {
    UNODB_ASSERT_EQ(test_db.get_key_prefix_splits(), splits);
  }

#endif  // UNODB_DETAIL_WITH_STATS

  void clear() {
    test_db.clear();
#ifndef NDEBUG
    allocation_failure_injector::reset();
#endif
    assert_empty();

    values.clear();
  }

  [[nodiscard, gnu::pure]] constexpr Db& get_db() noexcept { return test_db; }

  tree_verifier(const tree_verifier&) = delete;
  tree_verifier& operator=(const tree_verifier&) = delete;

 private:
  // Custom comparator is required for key_view.
  struct comparator {
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26415)
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26418)
    bool operator()(const ikey_type<Db>& lhs,
                    const ikey_type<Db>& rhs) const noexcept {
      if constexpr (std::is_same_v<typename Db::key_type, unodb::key_view>) {
        // Handle wrapped keys.
        return unodb::detail::compare(lhs->data(), lhs->size(), rhs->data(),
                                      rhs->size()) < 0;
      } else {
        return lhs < rhs;
      }
    }
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  };

  /// The tree under test.
  Db test_db{};

  /// Ground truth (key,val) pairs.
  ///
  /// \note The `std::map` and `std::unordered_map` do not support non-owned
  /// unodb::key_view objects as keys.  To handle this, unodb::key_view keys are
  /// wrapped as an owning type.
  std::map<ikey_type<Db>, value_type, comparator> values;

  const bool parallel_test;

  /// Arrays backing unodb::key_view objects.
  std::vector<std::array<std::byte, sizeof(std::uint64_t)>> key_views{};
};

using u64_db = unodb::db<std::uint64_t, unodb::value_view>;
using u64_mutex_db = unodb::mutex_db<std::uint64_t, unodb::value_view>;
using u64_olc_db = unodb::olc_db<std::uint64_t, unodb::value_view>;

using key_view_db = unodb::db<unodb::key_view, unodb::value_view>;
using key_view_mutex_db = unodb::mutex_db<unodb::key_view, unodb::value_view>;
using key_view_olc_db = unodb::olc_db<unodb::key_view, unodb::value_view>;

using key_view_u64val_db = unodb::db<unodb::key_view, std::uint64_t>;
using key_view_u64val_mutex_db =
    unodb::mutex_db<unodb::key_view, std::uint64_t>;
using key_view_u64val_olc_db = unodb::olc_db<unodb::key_view, std::uint64_t>;

extern template class tree_verifier<u64_db>;
extern template class tree_verifier<u64_mutex_db>;
extern template class tree_verifier<u64_olc_db>;

extern template class tree_verifier<key_view_db>;
extern template class tree_verifier<key_view_mutex_db>;
extern template class tree_verifier<key_view_olc_db>;

extern template class tree_verifier<key_view_u64val_db>;
extern template class tree_verifier<key_view_u64val_mutex_db>;
extern template class tree_verifier<key_view_u64val_olc_db>;

}  // namespace unodb::test

#endif  // UNODB_DETAIL_DB_TEST_UTILS_HPP
