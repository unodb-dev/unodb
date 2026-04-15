// Copyright 2019-2025 UnoDB contributors
#ifndef UNODB_DETAIL_MUTEX_ART_HPP
#define UNODB_DETAIL_MUTEX_ART_HPP

// Should be the first include
#include "global.hpp"

#include <cassert>
#include <mutex>
#include <type_traits>
#include <utility>

#include "art.hpp"

namespace unodb {

/// A thread-safe implementation of the Adaptive Radix Tree (ART) using an
/// explicit global lock.  All get, insert, remove and scan operations take the
/// global lock and hold it for the duration of the operation.
///
/// \sa unodb::olc_art for a highly concurrent thread-safe ART implementation.
template <typename Key, typename Value>
class mutex_db final {
 public:
  /// The type of the keys in the index.
  using key_type = Key;
  /// The type of the value associated with the keys in the index.
  using value_type = Value;
  using value_view = unodb::value_view;

  /// If the search key was found, that is, the first pair member has a value,
  /// then the second member is a locked tree mutex which must be released ASAP
  /// after reading the first pair member. Otherwise, the second member is
  /// undefined.
  using get_result = std::pair<typename db<Key, Value>::get_result,
                               std::unique_lock<std::mutex>>;

 private:
  using art_key_type = detail::basic_art_key<Key>;

  /// Querying with an encoded key.
  [[nodiscard]] get_result get_internal(art_key_type k) const noexcept {
    std::unique_lock guard{mutex};
    const auto db_get_result{db_.get_internal(k)};
    if (!db_get_result) {
      guard.unlock();
      return std::make_pair(db_get_result, std::unique_lock<std::mutex>{});
    }
    return std::make_pair(db_get_result, std::move(guard));
  }

  /// Modifying with an encoded key.
  ///
  /// Cannot be called during stack unwinding with `std::uncaught_exceptions() >
  /// 0`.
  [[nodiscard]] bool insert_internal(art_key_type k, value_type v) {
    const std::lock_guard guard{mutex};
    return db_.insert_internal(k, v);
  }

  /// Removing with an encoded key.
  [[nodiscard]] bool remove_internal(art_key_type k) {
    const std::lock_guard guard{mutex};
    return db_.remove_internal(k);
  }

 public:
  // Creation and destruction
  mutex_db() noexcept = default;

  /// Query for a value associated with a key.
  ///
  /// \param search_key If Key is a simple primitive type, then it is converted
  /// into a binary comparable key.  If Key is unodb::key_view, then it is
  /// assumed to already be a binary comparable key, e.g., as produced by
  /// unodb::key_encoder.
  [[nodiscard, gnu::pure]] get_result get(Key search_key) const noexcept {
    const art_key_type k{search_key};
    return get_internal(k);
  }

  [[nodiscard]] bool empty() const {
    const std::lock_guard guard{mutex};
    return db_.empty();
  }

  /// Insert a value under a key iff there is no entry for that key.
  ///
  /// \note Cannot be called during stack unwinding with
  /// `std::uncaught_exceptions() > 0`.
  ///
  /// \param insert_key If Key is a simple primitive type, then it is converted
  /// into a binary comparable key.  If Key is unodb::key_view, then it is
  /// assumed to already be a binary comparable key, e.g., as produced by
  /// unodb::key_encoder.
  ///
  /// \param v The value of type `value_type` to be inserted under
  /// that key.
  ///
  /// \return true iff the key value pair was inserted.
  ///
  /// \sa key_encoder, which provides for encoding text and multi-field records
  /// when Key is unodb::key_view.
  [[nodiscard]] bool insert(Key insert_key, value_type v) {
    const art_key_type k{insert_key};
    return insert_internal(k, v);
  }

  /// Remove the entry associated with the key.
  ///
  /// \param search_key If Key is a simple primitive type, then it is converted
  /// into a binary comparable key.  If Key is unodb::key_view, then it is
  /// assumed to already be a binary comparable key, e.g., as produced by
  /// unodb::key_encoder.
  [[nodiscard]] bool remove(Key search_key) {
    const auto k = art_key_type{search_key};
    return remove_internal(k);
  }

  /// Removes all entries in the index.
  void clear() {
    const std::lock_guard guard{mutex};
    db_.clear();
  }

  //
  // scan API.
  //

  using iterator = typename unodb::db<Key, Value>::iterator;

  /// Scan the tree, applying the caller's lambda to each visited leaf.  The
  /// tree remains locked for the duration of the scan.
  ///
  /// \param fn A function `f(unodb::visitor<unodb::mutex_db::iterator>&)`
  /// returning `bool`.  The traversal will halt if the function returns \c
  /// true.
  ///
  /// \param fwd When \c true perform a forward scan, otherwise perform a
  /// reverse scan.
  template <typename FN>
  void scan(FN fn, bool fwd = true) noexcept {
    const std::lock_guard guard{mutex};
    db_.scan(fn, fwd);
  }

  /// Scan in the indicated direction, applying the caller's lambda to each
  /// visited leaf.  The tree remains locked for the duration of the scan.
  ///
  /// \param from_key is an inclusive lower bound for the starting point of the
  /// scan.
  ///
  /// \param fn A function `f(unodb::visitor<unodb::mutex_db::iterator>&)`
  /// returning `bool`.  The traversal will halt if the function returns \c
  /// true.
  ///
  /// \param fwd When \c true perform a forward scan, otherwise perform a
  /// reverse scan.
  template <typename FN>
  void scan_from(Key from_key, FN fn, bool fwd = true) noexcept {
    const std::lock_guard guard{mutex};
    db_.scan_from(from_key, fn, fwd);
  }

  /// Scan a half-open key range, applying the caller's lambda to each visited
  /// leaf.  The scan will proceed in lexicographic order iff \a from_key is
  /// less than \a to_key and in reverse lexicographic order iff \a to_key is
  /// less than \a from_key.  When `from_key < to_key`, the scan will visit all
  /// index entries in the half-open range `[from_key,to_key)` in forward order.
  /// Otherwise the scan will visit all index entries in the half-open range
  /// `(from_key,to_key]` in reverse order.
  ///
  /// \param from_key is an inclusive bound for the starting point of the scan.
  ///
  /// \param to_key is an exclusive bound for the ending point of the scan.
  ///
  /// \param fn A function `f(unodb::visitor<unodb::mutex_db::iterator>&)`
  /// returning `bool`.  The traversal will halt if the function returns \c
  /// true.
  template <typename FN>
  void scan_range(Key from_key, Key to_key, FN fn) noexcept {
    const std::lock_guard guard{mutex};
    db_.scan_range(from_key, to_key, fn);
  }

  //
  // TEST ONLY METHODS
  //

  // Used to write the iterator tests.
  iterator test_only_iterator() noexcept { return db_.test_only_iterator(); }

  // Stats
#ifdef UNODB_DETAIL_WITH_STATS

  [[nodiscard]] std::size_t get_current_memory_use() const {
    const std::lock_guard guard{mutex};
    return db_.get_current_memory_use();
  }

  template <node_type NodeType>
  [[nodiscard]] std::uint64_t get_node_count() const {
    const std::lock_guard guard{mutex};
    return db_.template get_node_count<NodeType>();
  }

  [[nodiscard]] node_type_counter_array get_node_counts() const {
    const std::lock_guard guard{mutex};
    return db_.get_node_counts();
  }

  template <node_type NodeType>
  [[nodiscard]] std::uint64_t get_growing_inode_count() const {
    const std::lock_guard guard{mutex};
    return db_.template get_growing_inode_count<NodeType>();
  }

  [[nodiscard]] inode_type_counter_array get_growing_inode_counts() const {
    const std::lock_guard guard{mutex};
    return db_.get_growing_inode_counts();
  }

  template <node_type NodeType>
  [[nodiscard]] std::uint64_t get_shrinking_inode_count() const {
    const std::lock_guard guard{mutex};
    return db_.template get_shrinking_inode_count<NodeType>();
  }

  [[nodiscard]] inode_type_counter_array get_shrinking_inode_counts() const {
    const std::lock_guard guard{mutex};
    return db_.get_shrinking_inode_counts();
  }

  [[nodiscard]] std::uint64_t get_key_prefix_splits() const {
    const std::lock_guard guard{mutex};
    return db_.get_key_prefix_splits();
  }

#endif  // UNODB_DETAIL_WITH_STATS

  // Public utils

  // Releases the mutex in the case key was not found, keeps it locked
  // otherwise.
  [[nodiscard]] static bool key_found(const get_result& result) noexcept {
#ifndef NDEBUG
    const auto& lock{result.second};
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    assert(!result.first || lock.owns_lock());
#endif

    return static_cast<bool>(result.first);
  }

  // Debugging
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os) const {
    const std::lock_guard guard{mutex};
    db_.dump(os);
  }

 private:
  db<Key, Value> db_;
  mutable std::mutex mutex;
};

}  // namespace unodb

#endif  // UNODB_DETAIL_MUTEX_ART_HPP
