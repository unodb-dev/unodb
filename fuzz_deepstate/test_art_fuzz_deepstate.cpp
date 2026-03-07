// Copyright 2019-2025 Laurynas Biveinis

#include "global.hpp"  // IWYU pragma: keep

#include <algorithm>  // IWYU pragma: keep
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <limits>
#include <new>
#include <optional>  // IWYU pragma: keep
#include <sstream>
#include <tuple>
#include <unordered_map>  // IWYU pragma: keep
#include <utility>
#include <vector>

#include <deepstate/DeepState.h>
#include <deepstate/DeepState.hpp>

#include "art.hpp"         // IWYU pragma: keep
#include "art_common.hpp"  // IWYU pragma: keep
#include "deepstate_utils.hpp"
#include "node_type.hpp"
#include "test_heap.hpp"

namespace {

constexpr auto maximum_value_len = 1024 * 1024;  // 1MB
// Close to the longest test run that fits into 8192 random bytes provided by
// DeepState API.
constexpr auto test_length = 480;

using dynamic_value = std::vector<std::byte>;

using values_type = std::vector<dynamic_value>;

using oracle_type = std::unordered_map<std::uint64_t, unodb::value_view>;
[[nodiscard]] dynamic_value make_random_value(dynamic_value::size_type length) {
  dynamic_value result{length};
  for (dynamic_value::size_type i = 0; i < length; i++) {
    // Ideally we would take random bytes from DeepState, but we'd end up
    // exhausting their default source len too soon. Do something deterministic
    // that has embedded zero bytes to shake out any C string API use
    result[i] = static_cast<std::byte>(i % 256);
  }
  return result;
}

[[nodiscard]] unodb::value_view get_value(dynamic_value::size_type max_length,
                                          values_type& values) {
  const auto make_new_value = values.empty() || DeepState_Bool();
  ASSERT(max_length <= std::numeric_limits<std::uint32_t>::max());
  if (make_new_value) {
    const dynamic_value::size_type new_value_len =
        DeepState_SizeTInRange(0, max_length);
    auto new_value = make_random_value(new_value_len);
    LOG(TRACE) << "Making a new value of length " << new_value_len;
    const auto& inserted_value = values.emplace_back(std::move(new_value));
    return unodb::value_view{inserted_value};
  }
  LOG(TRACE) << "Reusing an existing value";
  ASSERT(values.size() <= std::numeric_limits<std::uint32_t>::max());
  const auto existing_value_i = DeepState_ContainerIndex(values);
  const auto& existing_value = values[existing_value_i];
  return unodb::value_view{existing_value};
}

[[nodiscard]] std::uint64_t get_key(std::uint64_t max_key_value,
                                    const std::vector<std::uint64_t>& keys) {
  const auto use_existing_key = !keys.empty() && DeepState_Bool();
  if (use_existing_key) {
    ASSERT(!keys.empty());
    ASSERT(keys.size() <= std::numeric_limits<std::uint32_t>::max());
    const auto existing_key_i = DeepState_ContainerIndex(keys);
    return keys[existing_key_i];
  }
  return DeepState_UInt64InRange(0, max_key_value);
}

void dump_tree(const unodb::db<std::uint64_t, unodb::value_view>& tree) {
  // Dump the tree to a string. Do not attempt to check the dump format, only
  // that dumping does not crash
  std::stringstream dump_sink;
  tree.dump(dump_sink);
}
#ifdef UNODB_DETAIL_WITH_STATS

void assert_unchanged_tree_after_failed_op(
    const unodb::db<std::uint64_t, unodb::value_view>& test_db,
    std::size_t mem_use_before,
    const unodb::node_type_counter_array& node_counts_before,
    const unodb::inode_type_counter_array& growing_inode_counts_before,
    const unodb::inode_type_counter_array& shrinking_inode_counts_before,
    std::size_t key_prefix_splits_before) {
  const auto mem_use_after = test_db.get_current_memory_use();
  ASSERT(mem_use_after == mem_use_before);
  const auto node_counts_after = test_db.get_node_counts();
  ASSERT(node_counts_before == node_counts_after);
  const auto growing_inode_counts_after = test_db.get_growing_inode_counts();
  ASSERT(growing_inode_counts_before == growing_inode_counts_after);
  const auto shrinking_inode_counts_after =
      test_db.get_shrinking_inode_counts();
  ASSERT(shrinking_inode_counts_before == shrinking_inode_counts_after);
  const auto key_prefix_splits_after = test_db.get_key_prefix_splits();
  ASSERT(key_prefix_splits_before == key_prefix_splits_after);
}

#endif  // UNODB_DETAIL_WITH_STATS

void op_with_oom_test(oracle_type& oracle, std::vector<std::uint64_t>& keys,
                      unodb::db<std::uint64_t, unodb::value_view>& test_db,
                      std::uint64_t key,
                      std::optional<unodb::value_view> value) {
  const auto do_insert = value.has_value();

#ifdef UNODB_DETAIL_WITH_STATS
  const auto mem_use_before = test_db.get_current_memory_use();
  const auto node_counts_before = test_db.get_node_counts();
  const auto growing_inode_counts_before = test_db.get_growing_inode_counts();
  const auto shrinking_inode_counts_before =
      test_db.get_shrinking_inode_counts();
  const auto key_prefix_splits_before = test_db.get_key_prefix_splits();
#endif  // UNODB_DETAIL_WITH_STATS

  bool op_result;
#ifndef NDEBUG
  unsigned fail_n = 1;
#endif

  while (true) {
    UNODB_DETAIL_FAIL_ON_NTH_ALLOCATION(fail_n);
    bool op_completed;
    try {
      op_result = do_insert ? test_db.insert(key, *value) : test_db.remove(key);
      op_completed = true;
    } catch (const std::bad_alloc&) {
      const auto search_result = test_db.get(key).has_value();
      ASSERT(search_result == (oracle.find(key) != oracle.cend()));
#ifdef UNODB_DETAIL_WITH_STATS
      assert_unchanged_tree_after_failed_op(
          test_db, mem_use_before, node_counts_before,
          growing_inode_counts_before, shrinking_inode_counts_before,
          key_prefix_splits_before);
#endif  // UNODB_DETAIL_WITH_STATS
      UNODB_DETAIL_RESET_ALLOCATION_FAILURE_INJECTOR();
#ifndef NDEBUG
      ++fail_n;
#endif
      op_completed = false;
    }
    if (op_completed) break;
  }

  if (op_result) {
#ifdef UNODB_DETAIL_WITH_STATS
    const auto mem_use_after = test_db.get_current_memory_use();
#endif  // UNODB_DETAIL_WITH_STATS
    if (do_insert) {
#ifdef UNODB_DETAIL_WITH_STATS
      ASSERT(mem_use_after > mem_use_before);
#endif  // UNODB_DETAIL_WITH_STATS
      UNODB_DETAIL_RESET_ALLOCATION_FAILURE_INJECTOR();
      LOG(TRACE) << "Inserted key" << key;
      const auto [insert_itr, insert_ok] = oracle.try_emplace(key, *value);
      std::ignore = insert_itr;
      ASSERT(insert_ok);
      keys.emplace_back(key);
    } else {
#ifdef UNODB_DETAIL_WITH_STATS
      ASSERT(mem_use_after < mem_use_before);
#endif  // UNODB_DETAIL_WITH_STATS
      UNODB_DETAIL_RESET_ALLOCATION_FAILURE_INJECTOR();
      LOG(TRACE) << "Deleted key " << key;
      const auto oracle_delete_result = oracle.erase(key);
      ASSERT(oracle_delete_result == 1);
    }
  } else {
#ifdef UNODB_DETAIL_WITH_STATS
    assert_unchanged_tree_after_failed_op(
        test_db, mem_use_before, node_counts_before,
        growing_inode_counts_before, shrinking_inode_counts_before,
        key_prefix_splits_before);
#endif  // UNODB_DETAIL_WITH_STATS
    ASSERT((oracle.find(key) == oracle.cend()) == !do_insert);
    UNODB_DETAIL_RESET_ALLOCATION_FAILURE_INJECTOR();
    LOG(TRACE) << (do_insert ? "Tried inserting duplicated key "
                             : "Tried deleting missing key ")
               << key;
  }

  dump_tree(test_db);
#ifdef UNODB_DETAIL_WITH_STATS
  LOG(TRACE) << "Current mem use: " << test_db.get_current_memory_use();
#endif  // UNODB_DETAIL_WITH_STATS
}

}  // namespace

UNODB_START_DEEPSTATE_TESTS()

// NOLINTBEGIN(clang-analyzer-security.ArrayBound): false positive in
// DeepState OneOf() template (DeepState.hpp:360), clang-21+.
TEST(ART, DeepStateFuzz) {
  const auto limit_max_key = DeepState_Bool();
  const auto max_key_value =
      limit_max_key ? DeepState_UInt64InRange(
                          0, std::numeric_limits<std::uint64_t>::max())
                    : std::numeric_limits<std::uint64_t>::max();
  if (limit_max_key)
    LOG(TRACE) << "Limiting maximum key value to " << max_key_value;
  else
    LOG(TRACE) << "Not limiting maximum key value (" << max_key_value << ")";

  static_assert(maximum_value_len <= std::numeric_limits<std::uint32_t>::max());
  const auto limit_value_length = DeepState_Bool();
  const auto max_value_length =
      limit_value_length ? DeepState_UIntInRange(0, maximum_value_len)
                         : maximum_value_len;
  if (limit_value_length)
    LOG(TRACE) << "Limiting maximum value length to " << max_value_length;
  else
    LOG(TRACE) << "Not limiting value length (" << max_value_length << ")";

  unodb::db<std::uint64_t, unodb::value_view> test_db;
  ASSERT(test_db.empty());

  std::vector<std::uint64_t> keys;
  values_type values;
  oracle_type oracle;

  const auto start_tm = std::time(nullptr);

  for (auto i = 0; i < test_length; i++) {
    LOG(TRACE) << "Iteration " << i;
    deepstate::OneOf(
        // Insert
        [&] {
          const auto key = DeepState_UInt64InRange(0, max_key_value);
          const auto value = get_value(max_value_length, values);
          LOG(TRACE) << "Inserting key " << key;
          op_with_oom_test(oracle, keys, test_db, key, value);
        },
        // Query
        [&] {
          UNODB_DETAIL_FAIL_ON_NTH_ALLOCATION(1);
          const auto key = get_key(max_key_value, keys);
          LOG(TRACE) << "Searching for key " << key;
          const auto search_result = test_db.get(key);
          const auto oracle_search_result = oracle.find(key);
          if (search_result) {
            ASSERT(!test_db.empty());
            ASSERT(oracle_search_result != oracle.cend())
                << "If search found a key, oracle must contain that key";
            ASSERT(std::equal(std::cbegin(*search_result),
                              std::cend(*search_result),
                              std::cbegin(oracle_search_result->second),
                              std::cend(oracle_search_result->second)))
                << "Values stored in ART and in oracle must match";
          } else {
            ASSERT(oracle_search_result == oracle.cend())
                << "If search did not find a key, oracle must not find it too ";
          }
          UNODB_DETAIL_RESET_ALLOCATION_FAILURE_INJECTOR();
        },
        // Delete
        [&] {
          // Delete everything with 0.1% probability
          const auto clear = (DeepState_UIntInRange(0, 999) == 0);
          if (clear) {
            LOG(TRACE) << "Clearing the tree";
            UNODB_DETAIL_FAIL_ON_NTH_ALLOCATION(1);
            test_db.clear();
            oracle.clear();
            // Once stats are compiled unconditionally again, restore:
            // ASSERT(test_db.get_current_memory_use() == 0);
            ASSERT(test_db.empty());
            UNODB_DETAIL_RESET_ALLOCATION_FAILURE_INJECTOR();
            return;
          }

          const auto key = get_key(max_key_value, keys);
          LOG(TRACE) << "Deleting key " << key;
          op_with_oom_test(oracle, keys, test_db, key, {});
        });
    if (unodb::test::timeout_reached(start_tm)) break;
  }

#ifdef UNODB_DETAIL_WITH_STATS
  auto prev_mem_use = test_db.get_current_memory_use();
#endif  // UNODB_DETAIL_WITH_STATS
  while (!oracle.empty()) {
    const auto [key, value] = *oracle.cbegin();
    LOG(TRACE) << "Shutdown: deleting key " << key;
    const auto oracle_remove_result = oracle.erase(key);
    ASSERT(oracle_remove_result == 1);
    const auto db_remove_result = test_db.remove(key);
    ASSERT(db_remove_result);
#ifdef UNODB_DETAIL_WITH_STATS
    const auto current_mem_use = test_db.get_current_memory_use();
    LOG(TRACE) << "Current mem use: " << current_mem_use;
    ASSERT(current_mem_use < prev_mem_use);
    prev_mem_use = current_mem_use;
#endif  // UNODB_DETAIL_WITH_STATS
  }
#ifdef UNODB_DETAIL_WITH_STATS
  ASSERT(prev_mem_use == 0);
#endif  // UNODB_DETAIL_WITH_STATS
  ASSERT(test_db.empty());
}
// NOLINTEND(clang-analyzer-security.ArrayBound)
