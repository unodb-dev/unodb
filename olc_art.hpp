// Copyright 2019-2025 UnoDB contributors
#ifndef UNODB_DETAIL_OLC_ART_HPP
#define UNODB_DETAIL_OLC_ART_HPP

/// \file
/// Concurrent Adaptive Radix Tree based on Optimistic Lock Coupling

// Should be the first include
#include "global.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stack>
#include <tuple>
#include <type_traits>

#include <boost/container/small_vector.hpp>

#include "art_common.hpp"
#include "art_internal.hpp"
#include "art_internal_impl.hpp"
#include "assert.hpp"
#include "node_type.hpp"
#include "optimistic_lock.hpp"
#include "portability_arch.hpp"
#include "qsbr.hpp"
#include "qsbr_ptr.hpp"
#include "sync.hpp"

namespace unodb {

template <typename Key, typename Value>
class olc_db;

namespace detail {

/// Sync point: fires after chain locked, before acquiring cut_point_parent.
inline sync_point sync_after_chain_locked;

/// Sync point: fires inside Step 2 loop between chain node locks.
inline sync_point sync_between_chain_locks;

/// OLC ART node header contains an unodb::optimistic_lock object for this node.
///
/// The node type is constant throughout the node lifetime, is stored outside of
/// the node (in the pointing-to pointer tag), and should be accessed without
/// following the \ref olc-read-protocol "OLC read protocol."
struct [[nodiscard]] olc_node_header {
  // Return a reference to the [optimistic_lock].
  [[nodiscard]] constexpr optimistic_lock& lock() const noexcept {
    return m_lock;
  }

#ifndef NDEBUG
  // This is passed as a debug callback to QSBR deallocation to be
  // checked at the physical deallocation time. This checks that the
  // node being deallocated has no open RCS (read_critical_section).
  static void check_on_dealloc(const void* ptr) noexcept {
    static_cast<const olc_node_header*>(ptr)->m_lock.check_on_dealloc();
  }
#endif

 private:
  mutable optimistic_lock m_lock;  // The lock.
};
static_assert(std::is_standard_layout_v<olc_node_header>);

template <typename Key, typename Value>
class olc_inode;

template <typename Key, typename Value>
class olc_inode_4;

template <typename Key, typename Value>
class olc_inode_16;

template <typename Key, typename Value>
class olc_inode_48;

template <typename Key, typename Value>
class olc_inode_256;

template <typename Key, typename Value>
using olc_inode_defs =
    basic_inode_def<olc_inode<Key, Value>, olc_inode_4<Key, Value>,
                    olc_inode_16<Key, Value>, olc_inode_48<Key, Value>,
                    olc_inode_256<Key, Value>>;

using olc_node_ptr = basic_node_ptr<olc_node_header>;

template <typename, typename, class>
class db_inode_qsbr_deleter;  // IWYU pragma: keep

template <class>
class db_leaf_qsbr_deleter;  // IWYU pragma: keep

struct olc_impl_helpers;

template <typename Key, typename Value>
using olc_art_policy = basic_art_policy<
    Key, Value, unodb::olc_db, unodb::in_critical_section,
    unodb::optimistic_lock, unodb::optimistic_lock::read_critical_section,
    olc_node_ptr, olc_inode_defs, db_inode_qsbr_deleter, db_leaf_qsbr_deleter>;

template <typename Key, typename Value>
using olc_db_leaf_unique_ptr =
    typename olc_art_policy<Key, Value>::db_leaf_unique_ptr;

template <typename Key, typename Value>
using olc_inode_base = basic_inode_impl<olc_art_policy<Key, Value>>;

template <typename Key, typename Value>
class olc_inode : public olc_inode_base<Key, Value> {};

template <typename Key, typename Value>
using olc_leaf_type = typename olc_art_policy<Key, Value>::leaf_type;

//
//
//

template <class AtomicArray>
using non_atomic_array =
    std::array<typename AtomicArray::value_type::value_type,
               std::tuple_size<AtomicArray>::value>;

template <class T>
inline non_atomic_array<T> copy_atomic_to_nonatomic(T& atomic_array) noexcept {
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26494)
  non_atomic_array<T> result;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  for (typename decltype(result)::size_type i = 0; i < result.size(); ++i) {
    result[i] = atomic_array[i].load(std::memory_order_relaxed);
  }
  return result;
}

template <typename Key, typename Value>
using olc_leaf_unique_ptr =
    basic_db_leaf_unique_ptr<Key, Value, olc_node_header, olc_db>;

}  // namespace detail

using qsbr_value_view = qsbr_ptr_span<const std::byte>;

/// A thread-safe Adaptive Radix Tree that is synchronized using optimistic lock
/// coupling. At any time, at most two directly-related tree nodes can be
/// write-locked by the insert algorithm and three by the delete algorithm. The
/// lock used is optimistic lock (see optimistic_lock.hpp), where only writers
/// lock and readers access nodes optimistically with node version checks. For
/// deleted node reclamation, Quiescent State-Based Reclamation is used.
template <typename Key, typename Value>
class olc_db final {
 public:
  /// The type of the keys in the index.
  using key_type = Key;
  /// The type of the value associated with the key in the index.
  using value_type = Value;
  using value_view = unodb::qsbr_value_view;
  using get_result = std::optional<value_view>;
  using inode_base = detail::olc_inode_base<Key, Value>;
  using leaf_type = detail::olc_leaf_type<Key, Value>;
  using db_type = olc_db<Key, Value>;

 private:
  using art_key_type = detail::basic_art_key<Key>;

  /// Query for a value associated with an encoded key.
  [[nodiscard, gnu::pure]] get_result get_internal(
      art_key_type search_key) const noexcept;

  /// Insert a value under an encoded key iff there is no entry for
  /// that key.
  ///
  /// \note Cannot be called during stack unwinding with
  /// `std::uncaught_exceptions() > 0`.
  ///
  /// \return true iff the key value pair was inserted.
  [[nodiscard]] bool insert_internal(art_key_type insert_key, value_type v);

  /// Remove the entry associated with the encoded key.
  ///
  /// \return true if the delete was successful (i.e. the key was found in the
  /// tree and the associated index entry was removed).
  [[nodiscard]] bool remove_internal(art_key_type remove_key);

 public:
  // Creation and destruction
  olc_db() noexcept = default;

  ~olc_db() noexcept;

  /// Query for a value associated with a key.
  ///
  /// \param search_key If Key is a simple primitive type, then it is converted
  /// into a binary comparable key.  If Key is unodb::key_view, then it is
  /// assumed to already be a binary comparable key, e.g., as produced by
  /// unodb::key_encoder.
  [[nodiscard, gnu::pure]] get_result get(Key search_key) const noexcept {
    const auto k = art_key_type{search_key};
    return get_internal(k);
  }

  /// Return true iff the tree is empty (no root leaf).
  [[nodiscard]] auto empty() const noexcept { return root == nullptr; }

  /// Insert a value under a binary comparable key iff there is no entry for
  /// that key.
  ///
  /// \note Cannot be called during stack unwinding with
  /// `std::uncaught_exceptions() > 0`.
  ///
  /// \param insert_key If Key is a simple primitive type, then it is converted
  /// into a binary comparable key.  If Key is unodb::key_view, then it is
  /// assumed to already be a binary comparable key, e.g., as produced by
  /// unodb::key_encoder.
  ///
  /// \param v The value of type `value_type` to be inserted under that key.
  ///
  /// \return true iff the key value pair was inserted.
  ///
  /// \sa key_encoder, which provides for encoding text and multi-field records
  /// when Key is unodb::key_view.
  [[nodiscard]] bool insert(Key insert_key, value_type v) {
    // TODO(thompsonbry) There should be a lambda variant of this to
    // handle conflicts and support upsert or delete-upsert
    // semantics. This would call the caller's lambda once the method
    // was positioned on the leaf.  The caller could then update the
    // value or perhaps delete the entry under the key.
    const auto k = art_key_type{insert_key};
    return insert_internal(k, v);
  }

  /// Remove the entry associated with the key.
  ///
  /// \param search_key If Key is a simple primitive type, then it is converted
  /// into a binary comparable key.  If Key is unodb::key_view, then it is
  /// assumed to already be a binary comparable key, e.g., as produced by
  /// unodb::key_encoder.
  ///
  /// \return true if the delete was successful (i.e. the key was found in the
  /// tree and the associated index entry was removed).
  [[nodiscard]] bool remove(Key search_key) {
    const auto k = art_key_type{search_key};
    return remove_internal(k);
  }

  /// Removes all entries in the index.
  ///
  /// \note Only legal in single-threaded context, as destructor
  void clear() noexcept;

  //
  // iterator (the iterator is an internal API, the public API is scan()).
  //

  /// The OLC scan() logic tracks the version tag (a read_critical_section) for
  /// each node in the stack.  This information is required because the
  /// iter_result tuples already contain physical information read from nodes
  /// which may have been invalidated by subsequent mutations.  The scan is
  /// built on iterator methods for seek(), next(), prior(), etc.  These methods
  /// must restart (rebuilding the stack and redoing the work) if they encounter
  /// a version tag for an element on the stack which is no longer valid.
  /// Restart works by performing a seek() to the key for the leaf on the bottom
  /// of the stack.  Restarts can be full (from the root) or partial (from the
  /// first element in the stack which was not invalidated by the structural
  /// modification).
  ///
  /// During scan(), the iterator seek()s to some key and then invokes the
  /// caller's lambda passing a reference to a visitor object.  That visitor
  /// allows the caller to access the key and/or value associated with the leaf.
  /// If the leaf is concurrently deleted from the structure, the visitor relies
  /// on epoch protection to return the data from the now invalidated leaf.
  /// This is still the state that the caller would have observed without the
  /// concurrent structural modification.  When next() is called, it will
  /// discover that the leaf on the bottom of the stack is not valid (it is
  /// marked as obsolete) and it will have to restart by seek() to the key for
  /// that leaf and then invoking next() if the key still exists and otherwise
  /// we should already be on the successor of that leaf.
  ///
  /// \note The OLC thread safety mechanisms permit concurrent non-atomic
  /// (multi-work) mutations to be applied to nodes.  This means that a thread
  /// can read junk in the middle of some reorganization of a node (e.g., the
  /// keys or children are being reordered to maintain an invariant for \c I16).
  /// To protect against reading such junk, the thread reads the version tag
  /// before and after it accesses data in the node and restarts if the version
  /// information has changed.  The thread must not act on information that it
  /// had read until it verifies that the version tag remained unchanged across
  /// the read operation.
  ///
  /// \note The iterator is backed by a std::stack. This means that the iterator
  /// methods accessing the stack can not be declared as \c noexcept.
  class iterator {
    friend class olc_db<Key, Value>;
    template <class>
    friend class visitor;

    /// Alias for the elements on the stack.
    struct stack_entry : public inode_base::iter_result {
      /// The version tag invariant for the node.  This contains the version
      /// information that must be valid to use data read from the node.  The
      /// version tag is cached when when those data are read from the node.
      ///
      /// \note This is just the data for the version tag and not the
      /// unodb::read_critical_section (RCS).  Moving the RCS onto the stack
      /// creates problems in the `while(...)` loops that use parent and node
      /// lock chaining since the RCS in the loop is invalid as soon as it is
      /// moved onto the stack.  Hence, this is just the data and the \c while
      /// loops continue to use the normal OLC pattern for lock chaining.
      version_tag_type version;
    };

   protected:
    /// Construct an empty iterator (one that is logically not
    /// positioned on anything and which will report !valid()).
    explicit iterator(olc_db& tree UNODB_DETAIL_LIFETIMEBOUND) noexcept
        : db_(tree) {}

    // iterator is not flyweight. disallow copy and move.
    iterator(const iterator&) = delete;
    iterator(iterator&&) = delete;
    iterator& operator=(const iterator&) = delete;
    // iterator& operator=(iterator&&) = delete; // test_only_iterator()

   public:
    using key_type = Key;
    using value_type = Value;

    // EXPOSED TO THE TESTS

    /// Position the iterator on the first entry in the index.
    iterator& first();

    /// Advance the iterator to next entry in the index.
    iterator& next();

    /// Position the iterator on the last entry in the index, which
    /// can be used to initiate a reverse traversal.
    iterator& last();

    /// Position the iterator on the previous entry in the index.
    iterator& prior();

    /// Position the iterator on, before, or after the caller's key.  If the
    /// iterator can not be positioned, it will be invalidated.  For example, if
    /// `fwd:=true` and the \a search_key is GT any key in the index then the
    /// iterator will be invalidated since there is no index entry greater than
    /// the search key.  Likewise, if `fwd:=false` and the \a search_key is LT
    /// any key in the index, then the iterator will be invalidated since there
    /// is no index entry LT the \c search_key.
    ///
    /// \param search_key The internal key used to position the iterator.
    ///
    /// \param match Will be set to true iff the search key is an exact match in
    /// the index data.  Otherwise, the match is not exact and the iterator is
    /// positioned either before or after the search_key.
    ///
    /// \param fwd When true, the iterator will be positioned first entry which
    /// orders GTE the search_key and invalidated if there is no such entry.
    /// Otherwise, the iterator will be positioned on the last key which orders
    /// LTE the search_key and invalidated if there is no such entry.
    iterator& seek(art_key_type search_key, bool& match, bool fwd = true);

    /// Return the key_view associated with the current position of
    /// the iterator.
    ///
    /// \pre The iterator MUST be valid().
    [[nodiscard]] key_view get_key() noexcept;

    /// Return the value_view associated with the current position of
    /// the iterator.
    ///
    /// \pre The iterator MUST be valid().
    [[nodiscard, gnu::pure]] qsbr_value_view get_val() const noexcept;

    /// Debugging
    // LCOV_EXCL_START
    [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os) const {
      if (empty()) {
        os << "iter::stack:: empty\n";
        return;
      }
      // Dump the key buffer maintained by the iterator.
      os << "keybuf=";
      detail::dump_key(os, keybuf_.get_key_view());
      os << "\n";
      // Create a new stack and copy everything there.  Using the new
      // stack, print out the stack in top-bottom order.  This avoids
      // modifications to the existing stack for the iterator.
      auto tmp = stack_;
      auto level = tmp.size() - 1;
      while (!tmp.empty()) {
        const auto& e = tmp.top();
        const auto& np = e.node;
        os << "iter::stack:: level = " << level << ", key_byte=0x" << std::hex
           << std::setfill('0') << std::setw(2)
           << static_cast<uint64_t>(e.key_byte) << std::dec
           << ", child_index=0x" << std::hex << std::setfill('0')
           << std::setw(2) << static_cast<std::uint64_t>(e.child_index)
           << std::dec << ", prefix(" << e.prefix.length() << ")=";
        detail::dump_key(os, e.prefix.get_key_view());
        os << ", version=";
        optimistic_lock::version_type(e.version).dump(os);  // version tag.
        os << ", ";
        art_policy::dump_node(os, np, false /*recursive*/);  // node or leaf.
        if (np.type() != node_type::LEAF) os << '\n';
        tmp.pop();
        level--;
      }
    }
    // LCOV_EXCL_STOP

    /// Debugging
    // LCOV_EXCL_START
    [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump() const { dump(std::cerr); }
    // LCOV_EXCL_STOP

    /// Return true unless the stack is empty (exposed to tests)
    [[nodiscard]] bool valid() const noexcept { return !stack_.empty(); }

   protected:
    /// Compare the given key (e.g., the to_key) to the current key in the
    /// internal buffer.
    ///
    /// \return -1, 0, or 1 if this key is LT, EQ, or GT the other key.
    [[nodiscard, gnu::pure]] int cmp(const art_key_type& akey) const noexcept;

    //
    // stack access methods.
    //

    /// Return true iff the stack is empty.
    [[nodiscard]] bool empty() const noexcept { return stack_.empty(); }

    /// Push an entry onto the stack.
    bool try_push(detail::olc_node_ptr node, std::byte key_byte,
                  std::uint8_t child_index, detail::key_prefix_snapshot prefix,
                  const optimistic_lock::read_critical_section& rcs) {
      // For variable length keys we need to know the number of bytes
      // associated with the node's key_prefix.  In addition there is
      // one byte for the descent to the child node along the
      // child_index.  That information needs to be stored on the
      // stack so we can pop off the right number of bytes even for
      // OLC where the node might be concurrently modified.
      UNODB_DETAIL_ASSERT(node.type() != node_type::LEAF);
      stack_.push({{node, key_byte, child_index, prefix}, rcs.get()});
      keybuf_.push(prefix.get_key_view());
      keybuf_.push(key_byte);
      return true;
    }

    /// Push a leaf onto the stack.
    bool try_push_leaf(detail::olc_node_ptr aleaf,
                       const optimistic_lock::read_critical_section& rcs) {
      // The [key], [child_index] and [prefix] are ignored for a leaf.
      stack_.push({{aleaf,
                    static_cast<std::byte>(0xFFU),     // key_byte
                    static_cast<std::uint8_t>(0xFFU),  // child_index
                    detail::key_prefix_snapshot(0)},   // empty key_prefix
                   rcs.get()});
      return true;
    }

    /// Push an entry onto the stack.
    bool try_push(const typename inode_base::iter_result& e,
                  const optimistic_lock::read_critical_section& rcs) {
      const auto node_type = e.node.type();
      if (UNODB_DETAIL_UNLIKELY(node_type == node_type::LEAF)) {
        return try_push_leaf(e.node, rcs);
      }
      return try_push(e.node, e.key_byte, e.child_index, e.prefix, rcs);
    }

    /// Pop an entry from the stack and the corresponding bytes from
    /// the key_buffer.
    void pop() noexcept {
      UNODB_DETAIL_ASSERT(!empty());

      // Note: We DO NOT need to check the RCS here. The prefix_len
      // on the stack is known to be valid at the time that the entry
      // was pushed onto the stack and the stack and the keybuf are in
      // sync with one another.  So we can just do a simple POP for
      // each of them.
      const auto prefix_len = top().prefix.length();
      keybuf_.pop(prefix_len);
      stack_.pop();
    }

    /// Return the entry on the top of the stack.
    [[nodiscard]] const stack_entry& top() const noexcept {
      UNODB_DETAIL_ASSERT(!stack_.empty());
      return stack_.top();
    }

    /// Return the node on the top of the stack and nullptr if the
    /// stack is empty (similar to top(), but handles an empty stack).
    [[nodiscard]] detail::olc_node_ptr current_node() const noexcept {
      return stack_.empty() ? detail::olc_node_ptr(nullptr) : stack_.top().node;
    }

   private:
    /// Invalidate the iterator (pops everything off of the stack).
    ///
    /// post-condition: The iterator is !valid().
    iterator& invalidate() noexcept {
      while (!stack_.empty()) stack_.pop();  // clear the stack
      return *this;
    }

    //
    // Core logic invoked from retry loops.
    //

    [[nodiscard]] bool try_first();
    [[nodiscard]] bool try_last();
    [[nodiscard]] bool try_next();
    [[nodiscard]] bool try_prior();

    /// Push the given node onto the stack and traverse from the
    /// caller's node to the left-most leaf under that node, pushing
    /// nodes onto the stack as they are visited.
    [[nodiscard]] bool try_left_most_traversal(
        detail::olc_node_ptr node,
        optimistic_lock::read_critical_section& parent_critical_section);

    /// Descend from the current state of the stack to the right most
    /// child leaf, updating the state of the iterator during the
    /// descent.
    [[nodiscard]] bool try_right_most_traversal(
        detail::olc_node_ptr node,
        optimistic_lock::read_critical_section& parent_critical_section);

    /// Core logic invoked from retry loop.
    [[nodiscard]] bool try_seek(art_key_type search_key, bool& match, bool fwd);

    /// The outer db instance.
    olc_db& db_;

    /// A stack reflecting the parent path from the root of the tree
    /// to the current leaf.  An empty stack corresponds to a
    /// logically empty iterator and can be detected using !valid().
    /// The iterator for an empty tree is an empty stack.
    std::stack<stack_entry> stack_{};

    /// A buffer into which visited encoded (binary comparable) keys
    /// are materialized by during the iterator traversal.  Bytes are
    /// pushed onto this buffer when we push something onto the
    /// iterator stack and popped off of this buffer when we pop
    /// something off of the iterator stack.
    detail::key_buffer keybuf_{};
  };  // class iterator

  //
  // end of the iterator API, which is an internal API.
  //

 public:
  ///
  /// public scan API
  ///

  // Note: The scan() interface is public.  The iterator and the
  // methods to obtain an iterator are protected (except for tests).
  // This encapsulation makes it easier to define methods which
  // operate on external keys (scan()) and those which operate on
  // internal keys (seek() and the iterator). It also makes life
  // easier for mutex_db since scan() can take the lock.

  /// Scan the tree, applying the caller's lambda to each visited leaf.
  ///
  /// \param fn A function `f(unodb::visitor<unodb::olc_db::iterator>&)`
  /// returning `bool`.  The traversal will halt if the function returns \c
  /// true.
  ///
  /// \param fwd When \c true perform a forward scan, otherwise perform a
  /// reverse scan.
  template <typename FN>
  void scan(FN fn, bool fwd = true) {
    if (fwd) {
      iterator it(*this);
      it.first();
      const visitor_type v{it};
      while (it.valid()) {
        if (UNODB_DETAIL_UNLIKELY(fn(v))) break;
        it.next();
      }
    } else {
      iterator it(*this);
      it.last();
      const visitor_type v{it};
      while (it.valid()) {
        if (UNODB_DETAIL_UNLIKELY(fn(v))) break;
        it.prior();
      }
    }
  }

  /// Scan in the indicated direction, applying the caller's lambda to each
  /// visited leaf.
  ///
  /// \param from_key is an inclusive lower bound for the starting point of the
  /// scan.
  ///
  /// \param fn A function `f(unodb::visitor<unodb::olc_db::iterator>&)`
  /// returning `bool`.  The traversal will halt if the function returns \c
  /// true.
  ///
  /// \param fwd When \c true perform a forward scan, otherwise perform a
  /// reverse scan.
  template <typename FN>
  void scan_from(Key from_key, FN fn, bool fwd = true) {
    const auto from_key_ = art_key_type{from_key};  // convert to internal key
    bool match{};
    if (fwd) {
      iterator it(*this);
      it.seek(from_key_, match, true /*fwd*/);
      const visitor_type v{it};
      while (it.valid()) {
        if (UNODB_DETAIL_UNLIKELY(fn(v))) break;
        it.next();
      }
    } else {
      iterator it(*this);
      it.seek(from_key_, match, false /*fwd*/);
      const visitor_type v{it};
      while (it.valid()) {
        if (UNODB_DETAIL_UNLIKELY(fn(v))) break;
        it.prior();
      }
    }
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
  /// \param fn A function `f(unodb::visitor<unodb::olc_db::iterator>&)`
  /// returning `bool`.  The traversal will halt if the function returns \c
  /// true.
  template <typename FN>
  void scan_range(Key from_key, Key to_key, FN fn) {
    // TODO(thompsonbry) : variable length keys. Explore a cheaper way
    // to handle the exclusive bound case when developing variable
    // length key support based on the maintained key buffer.
    constexpr bool debug = false;                   // set true to debug scan.
    const auto from_key_ = art_key_type{from_key};  // convert to internal key
    const auto to_key_ = art_key_type{to_key};      // convert to internal key
    const auto ret = from_key_.cmp(to_key_);  // compare the internal keys.
    const bool fwd{ret < 0};                  // from key is less than to key
    if (ret == 0) return;                     // NOP
    bool match{};
    if (fwd) {
      iterator it(*this);
      it.seek(from_key_, match, true /*fwd*/);
      if constexpr (debug) {
        std::cerr << "scan_range:: fwd"
                  << ", from_key=" << from_key_ << ", to_key=" << to_key_
                  << "\n";
        it.dump(std::cerr);
      }
      const visitor_type v{it};
      while (it.valid() && it.cmp(to_key_) < 0) {
        if (UNODB_DETAIL_UNLIKELY(fn(v))) break;
        it.next();
        if constexpr (debug) {
          std::cerr << "scan: next()\n";
          it.dump(std::cerr);
        }
      }
    } else {  // reverse traversal.
      iterator it(*this);
      it.seek(from_key_, match, false /*fwd*/);
      if constexpr (debug) {
        std::cerr << "scan_range:: rev"
                  << ", from_key=" << from_key_ << ", to_key=" << to_key_
                  << "\n";
        it.dump(std::cerr);
      }
      const visitor_type v{it};
      while (it.valid() && it.cmp(to_key_) > 0) {
        if (UNODB_DETAIL_UNLIKELY(fn(v))) break;
        it.prior();
        if constexpr (debug) {
          std::cerr << "scan: prior()\n";
          it.dump(std::cerr);
        }
      }
    }
  }

  //
  // TEST ONLY METHODS
  //

  // Used to write the iterator tests.
  iterator test_only_iterator() noexcept { return iterator(*this); }

  // Stats

#ifdef UNODB_DETAIL_WITH_STATS

  // Return current memory use by tree nodes in bytes
  [[nodiscard]] std::size_t get_current_memory_use() const noexcept {
    return current_memory_use.load(std::memory_order_relaxed);
  }

  template <node_type NodeType>
  [[nodiscard]] std::uint64_t get_node_count() const noexcept {
    return node_counts[as_i<NodeType>].load(std::memory_order_relaxed);
  }

  [[nodiscard]] node_type_counter_array get_node_counts() const noexcept {
    return detail::copy_atomic_to_nonatomic(node_counts);
  }

  template <node_type NodeType>
  [[nodiscard]] std::uint64_t get_growing_inode_count() const noexcept {
    return growing_inode_counts[internal_as_i<NodeType>].load(
        std::memory_order_relaxed);
  }

  [[nodiscard]] inode_type_counter_array get_growing_inode_counts()
      const noexcept {
    return detail::copy_atomic_to_nonatomic(growing_inode_counts);
  }

  template <node_type NodeType>
  [[nodiscard]] std::uint64_t get_shrinking_inode_count() const noexcept {
    return shrinking_inode_counts[internal_as_i<NodeType>].load(
        std::memory_order_relaxed);
  }

  [[nodiscard]] inode_type_counter_array get_shrinking_inode_counts()
      const noexcept {
    return detail::copy_atomic_to_nonatomic(shrinking_inode_counts);
  }

  [[nodiscard]] std::uint64_t get_key_prefix_splits() const noexcept {
    return key_prefix_splits.load(std::memory_order_relaxed);
  }

#endif  // UNODB_DETAIL_WITH_STATS

  // Public utils
  [[nodiscard]] static constexpr bool key_found(
      const get_result& result) noexcept {
    return static_cast<bool>(result);
  }

  // Debugging
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os) const;
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump() const;

  olc_db(const olc_db&) noexcept = delete;
  olc_db(olc_db&&) noexcept = delete;
  olc_db& operator=(const olc_db&) noexcept = delete;
  olc_db& operator=(olc_db&&) noexcept = delete;

 private:
  using art_policy = detail::olc_art_policy<Key, Value>;
  using header_type = typename art_policy::header_type;
  using inode_type = detail::olc_inode<Key, Value>;
  using inode_4 = detail::olc_inode_4<Key, Value>;
  using inode_16 = detail::olc_inode_16<Key, Value>;
  using inode_48 = detail::olc_inode_48<Key, Value>;
  using inode_256 = detail::olc_inode_256<Key, Value>;
  using tree_depth_type = detail::tree_depth<art_key_type>;
  using visitor_type = visitor<db_type::iterator>;
  using olc_db_leaf_unique_ptr_type =
      detail::olc_db_leaf_unique_ptr<Key, Value>;
  // If get_result is not present, the search was interrupted. Yes, this
  // resolves to std::optional<std::optional<value_view>>, but IMHO both
  // levels of std::optional are clear here
  using try_get_result_type = std::optional<get_result>;

  using try_update_result_type = std::optional<bool>;

  [[nodiscard]] try_get_result_type try_get(art_key_type k) const noexcept;

  /// Single-attempt insert.
  ///
  /// For variable-length keys (key_view) only, two keys may share more than
  /// key_prefix_capacity (7) bytes at a given depth, causing their dispatch
  /// bytes to collide.  When this happens, a single-child "chain" I4 node is
  /// created that consumes 7 prefix bytes + 1 dispatch byte, advancing depth
  /// by 8.  The insert then restarts from the root.  Subsequent attempts
  /// descend through the chain and either extend it (if keys still collide)
  /// or complete a normal two-child split.
  [[nodiscard]] try_update_result_type try_insert(
      art_key_type k, value_type v, olc_db_leaf_unique_ptr_type& cached_leaf);

  [[nodiscard]] try_update_result_type try_remove(art_key_type k);

  /// Stack entry for key_view remove traversal.
  struct remove_entry {
    detail::olc_node_ptr node;
    std::uint8_t child_i;  // child index within this node we descended through
    std::byte key_byte;    // dispatch byte used to descend FROM this node
    version_tag_type version;
    in_critical_section<detail::olc_node_ptr>* slot;
  };
  using remove_stack_type = boost::container::small_vector<remove_entry, 32>;

  /// Two-pass removal with chain cleanup for variable-length keys.
  ///
  /// Pass 1 (downward): traverses from root to the target leaf, building
  /// a stack of inode entries with cached versions.  Single-child I4
  /// nodes are identified as chain members.
  ///
  /// Pass 2 (upward / chain cut): if the leaf is under a chain, delegates
  /// to try_chain_cut for atomic disconnection.  Otherwise, uses the
  /// standard remove_or_choose_subtree path.
  [[nodiscard]] try_update_result_type try_remove_key_view(art_key_type k);

  /// Atomic chain cut: remove a leaf under a chain of single-child I4 nodes.
  ///
  /// Locks the chain bottom-up, then atomically disconnects the entire chain
  /// from its parent (the "cut point") in a single child-pointer update.
  /// If a concurrent modification invalidates a chain node during locking,
  /// the cut point is adjusted downward.  If the cut-point parent's version
  /// has changed, the operation restarts without modifying the tree.
  /// After the cut, the cut-point parent is shrunk or collapsed as needed
  /// to maintain well-formedness (no single-child I4 except during chain
  /// insert).  Disconnected nodes are reclaimed via QSBR.
  [[nodiscard]] try_update_result_type try_chain_cut(
      detail::olc_node_ptr chain_bottom, detail::olc_node_ptr leaf_ptr,
      optimistic_lock::write_guard parent_guard,
      optimistic_lock::write_guard chain_bottom_guard,
      optimistic_lock::write_guard leaf_guard, remove_stack_type& stk,
      std::size_t stk_n);

  /// Collapse an I4 after removing one of its two children.
  /// Removes del_ci, checks prefix overflow, and if the merge fits,
  /// promotes the remaining child into *slot (or root if slot==nullptr).
  /// Returns true if the I4 was collapsed and reclaimed.
  [[nodiscard]] bool try_collapse_i4(
      detail::olc_node_ptr i4_node, std::uint8_t del_ci,
      in_critical_section<detail::olc_node_ptr>* slot,
      optimistic_lock::write_guard& guard);

  /// Single-pass removal for fixed-width keys (no chain nodes).
  [[nodiscard]] try_update_result_type try_remove_fixed_width_key(
      art_key_type k);

  void delete_root_subtree() noexcept;

#ifdef UNODB_DETAIL_WITH_STATS
  void increase_memory_use(std::size_t delta) noexcept;
  void decrease_memory_use(std::size_t delta) noexcept;

  void increment_leaf_count(std::size_t leaf_size) noexcept {
    increase_memory_use(leaf_size);
    node_counts[as_i<node_type::LEAF>].fetch_add(1, std::memory_order_relaxed);
  }

  void decrement_leaf_count(std::size_t leaf_size) noexcept {
    decrease_memory_use(leaf_size);

    const auto old_leaf_count UNODB_DETAIL_USED_IN_DEBUG =
        node_counts[as_i<node_type::LEAF>].fetch_sub(1,
                                                     std::memory_order_relaxed);
    UNODB_DETAIL_ASSERT(old_leaf_count > 0);
  }

  template <class INode>
  constexpr void increment_inode_count() noexcept;

  template <class INode>
  constexpr void decrement_inode_count() noexcept;

  template <node_type NodeType>
  constexpr void account_growing_inode() noexcept;

  template <node_type NodeType>
  constexpr void account_shrinking_inode() noexcept;

#endif  // UNODB_DETAIL_WITH_STATS

  // optimistic lock guarding the [root].
  alignas(
      detail::hardware_destructive_interference_size) mutable optimistic_lock
      root_pointer_lock;

  // The root of the tree, guarded by the [root_pointer_lock].
  in_critical_section<detail::olc_node_ptr> root{detail::olc_node_ptr{nullptr}};

  static_assert(sizeof(root_pointer_lock) + sizeof(root) <=
                detail::hardware_constructive_interference_size);

#ifdef UNODB_DETAIL_WITH_STATS

  // Current logically allocated memory that is not scheduled to be reclaimed.
  // The total memory currently allocated is this plus the QSBR deallocation
  // backlog (qsbr::previous_interval_total_dealloc_size +
  // qsbr::current_interval_total_dealloc_size).
  alignas(detail::hardware_destructive_interference_size)
      std::atomic<std::size_t> current_memory_use{0};

  alignas(detail::hardware_destructive_interference_size)
      std::atomic<std::uint64_t> key_prefix_splits{0};

  template <class T>
  using atomic_array = std::array<std::atomic<typename T::value_type>,
                                  std::tuple_size<T>::value>;

  alignas(detail::hardware_destructive_interference_size)
      atomic_array<node_type_counter_array> node_counts{};
  alignas(detail::hardware_destructive_interference_size)
      atomic_array<inode_type_counter_array> growing_inode_counts{};
  alignas(detail::hardware_destructive_interference_size)
      atomic_array<inode_type_counter_array> shrinking_inode_counts{};

#endif  // UNODB_DETAIL_WITH_STATS

  friend auto detail::make_db_leaf_ptr<Key, Value, olc_db>(art_key_type,
                                                           unodb::value_view,
                                                           olc_db&);

  template <class>
  friend class detail::basic_db_leaf_deleter;

  template <class>
  friend class detail::db_leaf_qsbr_deleter;

  template <typename, typename, class>
  friend class detail::db_inode_qsbr_deleter;

  template <typename,                             // Key
            typename,                             // Value
            template <typename, typename> class,  // Db
            template <class> class,               // CriticalSectionPolicy
            class,                                // LockPolicy
            class,                                // ReadCriticalSection
            class,                                // NodePtr
            template <typename, typename> class,  // INodeDefs
            template <typename, typename, class> class,  // INodeReclamator
            template <class> class>                      // LeafReclamator
  friend struct detail::basic_art_policy;

  template <class, class>
  friend class detail::basic_db_inode_deleter;

  friend struct detail::olc_impl_helpers;
};

namespace detail {

template <typename Key, typename Value, class INode>
using db_inode_qsbr_deleter_parent =
    unodb::detail::basic_db_inode_deleter<INode, unodb::olc_db<Key, Value>>;

template <typename Key, typename Value, class INode>
class db_inode_qsbr_deleter
    : public db_inode_qsbr_deleter_parent<Key, Value, INode> {
 public:
  using db_inode_qsbr_deleter_parent<Key, Value,
                                     INode>::db_inode_qsbr_deleter_parent;

  void operator()(INode* inode_ptr) {
    static_assert(std::is_trivially_destructible_v<INode>);

    this_thread().on_next_epoch_deallocate(inode_ptr
#ifdef UNODB_DETAIL_WITH_STATS
                                           ,
                                           sizeof(INode)
#endif
#ifndef NDEBUG
                                               ,
                                           olc_node_header::check_on_dealloc
#endif
    );

#ifdef UNODB_DETAIL_WITH_STATS
    this->get_db().template decrement_inode_count<INode>();
#endif  // UNODB_DETAIL_WITH_STATS
  }
};

template <class Db>
class db_leaf_qsbr_deleter {
 public:
  using key_type = typename Db::key_type;
  using leaf_type = basic_leaf<key_type, typename Db::header_type>;

  static_assert(std::is_trivially_destructible_v<leaf_type>);

  constexpr explicit db_leaf_qsbr_deleter(Db& db_
                                          UNODB_DETAIL_LIFETIMEBOUND) noexcept
      : db_instance{db_} {}

  void operator()(leaf_type* to_delete) const {
#ifdef UNODB_DETAIL_WITH_STATS
    const auto leaf_size = to_delete->get_size();
#endif  // UNODB_DETAIL_WITH_STATS

    this_thread().on_next_epoch_deallocate(to_delete
#ifdef UNODB_DETAIL_WITH_STATS
                                           ,
                                           leaf_size
#endif  // UNODB_DETAIL_WITH_STATS
#ifndef NDEBUG
                                           ,
                                           olc_node_header::check_on_dealloc
#endif
    );

#ifdef UNODB_DETAIL_WITH_STATS
    db_instance.decrement_leaf_count(leaf_size);
#endif  // UNODB_DETAIL_WITH_STATS
  }

  ~db_leaf_qsbr_deleter() = default;
  db_leaf_qsbr_deleter(const db_leaf_qsbr_deleter&) = default;
  db_leaf_qsbr_deleter& operator=(const db_leaf_qsbr_deleter&) = delete;
  db_leaf_qsbr_deleter(db_leaf_qsbr_deleter&&) = delete;
  db_leaf_qsbr_deleter& operator=(db_leaf_qsbr_deleter&&) = delete;

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  Db& db_instance;
};

/// Return a reference to the unodb::optimistic_lock from the node header
/// associated with the unodb::detail::olc_node_ptr..
///
/// \note This returns the lock rather than trying to acquire the lock.
[[nodiscard]] inline auto& node_ptr_lock(const unodb::detail::olc_node_ptr& node
                                         UNODB_DETAIL_LIFETIMEBOUND) noexcept {
  return node.ptr<unodb::detail::olc_node_header*>()->lock();
}

#ifndef NDEBUG

template <typename Key, typename Value>
[[nodiscard]] auto& node_ptr_lock(
    const unodb::detail::olc_leaf_type<Key, Value>* const node
    UNODB_DETAIL_LIFETIMEBOUND) noexcept {
  return node->lock();
}

#endif

template <class INode>
[[nodiscard]] constexpr auto& lock(const INode& inode
                                   UNODB_DETAIL_LIFETIMEBOUND) noexcept {
  return inode.lock();
}

template <class T>
[[nodiscard]] T& obsolete(T& t UNODB_DETAIL_LIFETIMEBOUND,
                          unodb::optimistic_lock::write_guard& guard) noexcept {
  UNODB_DETAIL_ASSERT(guard.guards(lock(t)));

  // My first attempt was to pass guard by value and let it destruct at the end
  // of this scope, but due to copy elision (?) the destructor got called way
  // too late, after the owner node was destructed.
  guard.unlock_and_obsolete();

  return t;
}

[[nodiscard]] inline auto obsolete_child_by_index(
    std::uint8_t child UNODB_DETAIL_LIFETIMEBOUND,
    unodb::optimistic_lock::write_guard& guard) noexcept {
  guard.unlock_and_obsolete();

  return child;
}

// Wrap olc_inode_add in a struct so that the latter and not the former could be
// declared as friend of olc_db, avoiding the need to forward declare the likes
// of olc_db_leaf_unique_ptr.
struct olc_impl_helpers {
  // GCC 10 diagnoses parameters that are present only in uninstantiated if
  // constexpr branch, such as node_in_parent for olc_inode_256.
  UNODB_DETAIL_DISABLE_GCC_10_WARNING("-Wunused-parameter")

  template <typename Key, typename Value, class INode>
  [[nodiscard]] static std::optional<in_critical_section<olc_node_ptr>*>
  add_or_choose_subtree(
      INode& inode, std::byte key_byte, basic_art_key<Key> k, value_view v,
      olc_db<Key, Value>& db_instance, tree_depth<basic_art_key<Key>> depth,
      optimistic_lock::read_critical_section& node_critical_section,
      in_critical_section<olc_node_ptr>* node_in_parent,
      optimistic_lock::read_critical_section& parent_critical_section,
      olc_db_leaf_unique_ptr<Key, Value>& cached_leaf);

  UNODB_DETAIL_RESTORE_GCC_10_WARNINGS()

  template <typename Key, typename Value, class INode>
  [[nodiscard]] static std::optional<bool> remove_or_choose_subtree(
      INode& inode, std::byte key_byte, basic_art_key<Key> k,
      olc_db<Key, Value>& db_instance,
      optimistic_lock::read_critical_section& parent_critical_section,
      optimistic_lock::read_critical_section& node_critical_section,
      in_critical_section<olc_node_ptr>* node_in_parent,
      in_critical_section<olc_node_ptr>** child_in_parent,
      optimistic_lock::read_critical_section* child_critical_section,
      node_type* child_type, olc_node_ptr* child);

  olc_impl_helpers() = delete;
};

//
// OLC inode classes extend the basic inode classes and wrap them with
// additional policy stuff.
//
// Note: These classes may assert that appropriate optimistic locks are held,
// but they do not take those locks.  That happens above the inode abstraction
// in the various algorithms which must follow the OLC patterns to ensure that
// they do not take action on data before they have verified that the optimistic
// condition remained true while data was read from the inode.
//

template <typename Key, typename Value>
using olc_inode_4_parent = basic_inode_4<olc_art_policy<Key, Value>>;

template <typename Key, typename Value>
class [[nodiscard]] olc_inode_4 final : public olc_inode_4_parent<Key, Value> {
  using parent_class = olc_inode_4_parent<Key, Value>;

 public:
  using db_type = olc_db<Key, Value>;
  using inode_16_type = olc_inode_16<Key, Value>;
  using art_key_type = basic_art_key<Key>;
  using tree_depth_type = tree_depth<art_key_type>;
  using leaf_type = olc_leaf_type<Key, Value>;
  using olc_db_leaf_unique_ptr_type = olc_db_leaf_unique_ptr<Key, Value>;

  using parent_class::parent_class;

  void init(db_type& db_instance, inode_16_type& source_node,
            unodb::optimistic_lock::write_guard& source_node_guard,
            std::uint8_t child_to_delete,
            unodb::optimistic_lock::write_guard& child_guard);

  void init(key_view k1, art_key_type shifted_k2, tree_depth_type depth,
            const leaf_type* child1,
            olc_db_leaf_unique_ptr_type&& child2) noexcept {
    UNODB_DETAIL_ASSERT((node_ptr_lock<Key, Value>(child1).is_write_locked()));

    parent_class::init(k1, shifted_k2, depth, child1, std::move(child2));
  }

  void init(olc_node_ptr source_node, unsigned len, tree_depth_type depth,
            olc_db_leaf_unique_ptr_type&& child1) {
    UNODB_DETAIL_ASSERT(node_ptr_lock(source_node).is_write_locked());

    parent_class::init(source_node, len, depth, std::move(child1));
  }

  template <typename... Args>
  [[nodiscard]] auto add_or_choose_subtree(Args&&... args) {
    return olc_impl_helpers::add_or_choose_subtree(*this,
                                                   std::forward<Args>(args)...);
  }

  template <typename... Args>
  [[nodiscard]] auto remove_or_choose_subtree(Args&&... args) {
    return olc_impl_helpers::remove_or_choose_subtree(
        *this, std::forward<Args>(args)...);
  }

  void remove(std::uint8_t child_index, db_type& db_instance) noexcept {
    UNODB_DETAIL_ASSERT(lock(*this).is_write_locked());

    parent_class::remove(child_index, db_instance);
  }

  [[nodiscard]] auto leave_last_child(std::uint8_t child_to_delete,
                                      db_type& db_instance) noexcept {
    UNODB_DETAIL_ASSERT(lock(*this).is_obsoleted_by_this_thread());
    UNODB_DETAIL_ASSERT(node_ptr_lock(this->children[child_to_delete].load())
                            .is_obsoleted_by_this_thread());

    return parent_class::leave_last_child(child_to_delete, db_instance);
  }

  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os,
                                                bool recursive) const {
    os << ", ";
    lock(*this).dump(os);
    parent_class::dump(os, recursive);
  }
};  // basic_inode_4

using olc_inode_4_test_type = olc_inode_4<std::uint64_t, unodb::value_view>;
// 48 (or 56) == sizeof(inode_4)
#ifndef _MSC_VER
#ifdef NDEBUG
static_assert(sizeof(olc_inode_4_test_type) == 48 + 8);
#else
static_assert(sizeof(olc_inode_4_test_type) == 48 + 24);
#endif
#else  // #ifndef _MSC_VER
#ifdef NDEBUG
static_assert(sizeof(olc_inode_4_test_type) == 56 + 8);
#else
static_assert(sizeof(olc_inode_4_test_type) == 56 + 24);
#endif
#endif  // #ifndef _MSC_VER

template <typename Key, typename Value>
using olc_inode_16_parent = basic_inode_16<olc_art_policy<Key, Value>>;

template <typename Key, typename Value>
class [[nodiscard]] olc_inode_16 final
    : public olc_inode_16_parent<Key, Value> {
  using parent_class = olc_inode_16_parent<Key, Value>;

 public:
  using typename parent_class::find_result;
  using db_type = olc_db<Key, Value>;
  using inode_4_type = olc_inode_4<Key, Value>;
  using inode_48_type = olc_inode_48<Key, Value>;
  using art_key_type = basic_art_key<Key>;
  using tree_depth_type = tree_depth<art_key_type>;
  using olc_db_leaf_unique_ptr_type = olc_db_leaf_unique_ptr<Key, Value>;

  using parent_class::parent_class;

  void init(db_type& db_instance, inode_4_type& source_node,
            unodb::optimistic_lock::write_guard& source_node_guard,
            olc_db_leaf_unique_ptr_type&& child,
            tree_depth_type depth) noexcept {
    UNODB_DETAIL_ASSERT(source_node_guard.guards(lock(source_node)));
    parent_class::init(db_instance, obsolete(source_node, source_node_guard),
                       std::move(child), depth);
    UNODB_DETAIL_ASSERT(!source_node_guard.active());
  }

  void init(db_type& db_instance, inode_48_type& source_node,
            unodb::optimistic_lock::write_guard& source_node_guard,
            std::uint8_t child_to_delete,
            unodb::optimistic_lock::write_guard& child_guard) noexcept;

  template <typename... Args>
  [[nodiscard]] auto add_or_choose_subtree(Args&&... args) {
    return olc_impl_helpers::add_or_choose_subtree(*this,
                                                   std::forward<Args>(args)...);
  }

  template <typename... Args>
  [[nodiscard]] auto remove_or_choose_subtree(Args&&... args) {
    return olc_impl_helpers::remove_or_choose_subtree(
        *this, std::forward<Args>(args)...);
  }

  void remove(std::uint8_t child_index, db_type& db_instance) noexcept {
    UNODB_DETAIL_ASSERT(lock(*this).is_write_locked());

    parent_class::remove(child_index, db_instance);
  }

  [[nodiscard]] find_result find_child(std::byte key_byte) noexcept {
#ifdef UNODB_DETAIL_THREAD_SANITIZER
    const auto children_count_ = this->get_children_count();
    for (unsigned i = 0; i < children_count_; ++i)
      if (parent_class::keys.byte_array[i] == key_byte)
        return std::make_pair(i, &parent_class::children[i]);
    return parent_class::child_not_found;
#else
    return parent_class::find_child(key_byte);
#endif
  }

  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os,
                                                bool recursive) const {
    os << ", ";
    lock(*this).dump(os);
    parent_class::dump(os, recursive);
  }
};

using olc_inode_16_test_type = olc_inode_16<std::uint64_t, unodb::value_view>;
// 160 == sizeof(inode_16)
#ifdef NDEBUG
static_assert(sizeof(olc_inode_16_test_type) == 160 + 16);
#else   // #ifdef NDEBUG
static_assert(sizeof(olc_inode_16_test_type) == 160 + 32);
#endif  // #ifdef NDEBUG

template <typename Key, typename Value>
void olc_inode_4<Key, Value>::init(
    db_type& db_instance, inode_16_type& source_node,
    unodb::optimistic_lock::write_guard& source_node_guard,
    std::uint8_t child_to_delete,
    unodb::optimistic_lock::write_guard& child_guard) {
  UNODB_DETAIL_ASSERT(source_node_guard.guards(lock(source_node)));
  UNODB_DETAIL_ASSERT(child_guard.active());

  parent_class::init(db_instance, obsolete(source_node, source_node_guard),
                     obsolete_child_by_index(child_to_delete, child_guard));

  UNODB_DETAIL_ASSERT(!source_node_guard.active());
  UNODB_DETAIL_ASSERT(!child_guard.active());
}

template <typename Key, typename Value>
using olc_inode_48_parent = basic_inode_48<olc_art_policy<Key, Value>>;

template <typename Key, typename Value>
class [[nodiscard]] olc_inode_48 final
    : public olc_inode_48_parent<Key, Value> {
  using parent_class = olc_inode_48_parent<Key, Value>;

 public:
  using db_type = olc_db<Key, Value>;
  using inode_16_type = olc_inode_16<Key, Value>;
  using inode_256_type = olc_inode_256<Key, Value>;
  using art_key_type = basic_art_key<Key>;
  using tree_depth_type = tree_depth<art_key_type>;
  using olc_db_leaf_unique_ptr_type = olc_db_leaf_unique_ptr<Key, Value>;

  using parent_class::parent_class;

  void init(db_type& db_instance, inode_16_type& source_node,
            unodb::optimistic_lock::write_guard& source_node_guard,
            olc_db_leaf_unique_ptr_type&& child,
            tree_depth_type depth) noexcept {
    UNODB_DETAIL_ASSERT(source_node_guard.guards(lock(source_node)));
    parent_class::init(db_instance, obsolete(source_node, source_node_guard),
                       std::move(child), depth);
    UNODB_DETAIL_ASSERT(!source_node_guard.active());
  }

  void init(db_type& db_instance, inode_256_type& source_node,
            unodb::optimistic_lock::write_guard& source_node_guard,
            std::uint8_t child_to_delete,
            unodb::optimistic_lock::write_guard& child_guard) noexcept;

  template <typename... Args>
  [[nodiscard]] auto add_or_choose_subtree(Args&&... args) {
    return olc_impl_helpers::add_or_choose_subtree(*this,
                                                   std::forward<Args>(args)...);
  }

  template <typename... Args>
  [[nodiscard]] auto remove_or_choose_subtree(Args&&... args) {
    return olc_impl_helpers::remove_or_choose_subtree(
        *this, std::forward<Args>(args)...);
  }

  void remove(std::uint8_t child_index, db_type& db_instance) noexcept {
    UNODB_DETAIL_ASSERT(lock(*this).is_write_locked());

    parent_class::remove(child_index, db_instance);
  }

  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os,
                                                bool recursive) const {
    os << ", ";
    lock(*this).dump(os);
    parent_class::dump(os, recursive);
  }
};

using olc_inode_48_test_type = olc_inode_48<std::uint64_t, unodb::value_view>;
// sizeof(inode_48) == 672 on AVX2, 656 otherwise
#ifdef NDEBUG
// AVX2 too. Padding?
static_assert(sizeof(olc_inode_48_test_type) == 656 + 16);
#else  // #ifdef NDEBUG
#if defined(UNODB_DETAIL_AVX2)
static_assert(sizeof(olc_inode_48_test_type) == 672 + 32);
#else
static_assert(sizeof(olc_inode_48_test_type) == 656 + 32);
#endif
#endif  // #ifdef NDEBUG

template <typename Key, typename Value>
void olc_inode_16<Key, Value>::init(
    db_type& db_instance, inode_48_type& source_node,
    unodb::optimistic_lock::write_guard& source_node_guard,
    std::uint8_t child_to_delete,
    unodb::optimistic_lock::write_guard& child_guard) noexcept {
  UNODB_DETAIL_ASSERT(source_node_guard.guards(lock(source_node)));
  UNODB_DETAIL_ASSERT(child_guard.active());

  parent_class::init(db_instance, obsolete(source_node, source_node_guard),
                     obsolete_child_by_index(child_to_delete, child_guard));

  UNODB_DETAIL_ASSERT(!source_node_guard.active());
  UNODB_DETAIL_ASSERT(!child_guard.active());
}

template <typename Key, typename Value>
using olc_inode_256_parent = basic_inode_256<olc_art_policy<Key, Value>>;

template <typename Key, typename Value>
class [[nodiscard]] olc_inode_256 final
    : public olc_inode_256_parent<Key, Value> {
  using parent_class = olc_inode_256_parent<Key, Value>;

 public:
  using db_type = olc_db<Key, Value>;
  using inode_48_type = olc_inode_48<Key, Value>;
  using art_key_type = basic_art_key<Key>;
  using tree_depth_type = tree_depth<art_key_type>;
  using olc_db_leaf_unique_ptr_type = olc_db_leaf_unique_ptr<Key, Value>;

  using parent_class::parent_class;

  void init(db_type& db_instance, inode_48_type& source_node,
            unodb::optimistic_lock::write_guard& source_node_guard,
            olc_db_leaf_unique_ptr_type&& child,
            tree_depth_type depth) noexcept {
    UNODB_DETAIL_ASSERT(source_node_guard.guards(lock(source_node)));
    parent_class::init(db_instance, obsolete(source_node, source_node_guard),
                       std::move(child), depth);
    UNODB_DETAIL_ASSERT(!source_node_guard.active());
  }

  template <typename... Args>
  [[nodiscard]] auto add_or_choose_subtree(Args&&... args) {
    return olc_impl_helpers::add_or_choose_subtree(*this,
                                                   std::forward<Args>(args)...);
  }

  template <typename... Args>
  [[nodiscard]] auto remove_or_choose_subtree(Args&&... args) {
    return olc_impl_helpers::remove_or_choose_subtree(
        *this, std::forward<Args>(args)...);
  }

  void remove(std::uint8_t child_index, db_type& db_instance) noexcept {
    UNODB_DETAIL_ASSERT(lock(*this).is_write_locked());

    parent_class::remove(child_index, db_instance);
  }

  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os,
                                                bool recursive) const {
    os << ", ";
    lock(*this).dump(os);
    parent_class::dump(os, recursive);
  }
};

using olc_inode_256_test_type = olc_inode_256<std::uint64_t, unodb::value_view>;
// 2064 == sizeof(inode_256)
#ifdef NDEBUG
static_assert(sizeof(olc_inode_256_test_type) == 2064 + 8);
#else
static_assert(sizeof(olc_inode_256_test_type) == 2064 + 24);
#endif

template <typename Key, typename Value>
void olc_inode_48<Key, Value>::init(
    db_type& db_instance, inode_256_type& source_node,
    unodb::optimistic_lock::write_guard& source_node_guard,
    std::uint8_t child_to_delete,
    unodb::optimistic_lock::write_guard& child_guard) noexcept {
  UNODB_DETAIL_ASSERT(source_node_guard.guards(lock(source_node)));
  UNODB_DETAIL_ASSERT(child_guard.active());

  parent_class::init(db_instance, obsolete(source_node, source_node_guard),
                     obsolete_child_by_index(child_to_delete, child_guard));

  UNODB_DETAIL_ASSERT(!source_node_guard.active());
  UNODB_DETAIL_ASSERT(!child_guard.active());
}

template <typename Key, typename Value>
void create_leaf_if_needed(olc_db_leaf_unique_ptr<Key, Value>& cached_leaf,
                           basic_art_key<Key> k, unodb::value_view v,
                           unodb::olc_db<Key, Value>& db_instance) {
  if (UNODB_DETAIL_LIKELY(cached_leaf == nullptr)) {
    UNODB_DETAIL_ASSERT(&cached_leaf.get_deleter().get_db() == &db_instance);
    // Do not assign because we do not need to assign the deleter
    // NOLINTNEXTLINE(misc-uniqueptr-reset-release)
    cached_leaf.reset(
        olc_art_policy<Key, Value>::make_db_leaf_ptr(k, v, db_instance)
            .release());
  }
}

UNODB_DETAIL_DISABLE_MSVC_WARNING(26460)
template <typename Key, typename Value, class INode>
[[nodiscard]] std::optional<in_critical_section<olc_node_ptr>*>
olc_impl_helpers::add_or_choose_subtree(
    INode& inode, std::byte key_byte, basic_art_key<Key> k, value_view v,
    olc_db<Key, Value>& db_instance, tree_depth<basic_art_key<Key>> depth,
    optimistic_lock::read_critical_section& node_critical_section,
    in_critical_section<olc_node_ptr>* node_in_parent,
    optimistic_lock::read_critical_section& parent_critical_section,
    olc_db_leaf_unique_ptr<Key, Value>& cached_leaf) {
  auto* const child_in_parent = inode.find_child(key_byte).second;

  if (child_in_parent == nullptr) {
    create_leaf_if_needed(cached_leaf, k, v, db_instance);

    const auto children_count = inode.get_children_count();

    if constexpr (!std::is_same_v<INode, olc_inode_256<Key, Value>>) {
      if (UNODB_DETAIL_UNLIKELY(children_count == INode::capacity)) {
        auto larger_node{
            INode::larger_derived_type::create(db_instance, inode)};
        {
          const optimistic_lock::write_guard write_unlock_on_exit{
              std::move(parent_critical_section)};
          if (UNODB_DETAIL_UNLIKELY(write_unlock_on_exit.must_restart()))
            return {};  // LCOV_EXCL_LINE

          optimistic_lock::write_guard node_write_guard{
              std::move(node_critical_section)};
          if (UNODB_DETAIL_UNLIKELY(node_write_guard.must_restart())) return {};

          larger_node->init(db_instance, inode, node_write_guard,
                            std::move(cached_leaf), depth);
          *node_in_parent = detail::olc_node_ptr{
              larger_node.release(), INode::larger_derived_type::type};

          UNODB_DETAIL_ASSERT(!node_write_guard.active());
        }

#ifdef UNODB_DETAIL_WITH_STATS
        db_instance
            .template account_growing_inode<INode::larger_derived_type::type>();
#endif  // UNODB_DETAIL_WITH_STATS

        return child_in_parent;
      }
    }

    const optimistic_lock::write_guard write_unlock_on_exit{
        std::move(node_critical_section)};
    if (UNODB_DETAIL_UNLIKELY(write_unlock_on_exit.must_restart())) return {};

    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE

    inode.add_to_nonfull(std::move(cached_leaf), depth, children_count);
  }

  return child_in_parent;
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

template <typename Key, typename Value, class INode>
[[nodiscard]] std::optional<bool> olc_impl_helpers::remove_or_choose_subtree(
    INode& inode, std::byte key_byte, basic_art_key<Key> k,
    olc_db<Key, Value>& db_instance,
    optimistic_lock::read_critical_section& parent_critical_section,
    optimistic_lock::read_critical_section& node_critical_section,
    in_critical_section<olc_node_ptr>* node_in_parent,
    in_critical_section<olc_node_ptr>** child_in_parent,
    optimistic_lock::read_critical_section* child_critical_section,
    node_type* child_type, olc_node_ptr* child) {
  const auto [child_i, found_child]{inode.find_child(key_byte)};

  if (found_child == nullptr) {
    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE

    return false;
  }

  *child = found_child->load();

  if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check())) return {};

  auto& child_lock{node_ptr_lock(*child)};
  *child_critical_section = child_lock.try_read_lock();
  if (UNODB_DETAIL_UNLIKELY(child_critical_section->must_restart())) return {};

  *child_type = child->type();

  if (*child_type != node_type::LEAF) {
    *child_in_parent = found_child;
    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE
    return true;
  }

  const auto* const leaf{child->ptr<olc_leaf_type<Key, Value>*>()};
  if (!leaf->matches(k)) {
    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE
    if (UNODB_DETAIL_UNLIKELY(!child_critical_section->try_read_unlock()))
      return {};  // LCOV_EXCL_LINE

    return false;
  }

  const auto is_node_min_size{inode.is_min_size()};

  if (UNODB_DETAIL_LIKELY(!is_node_min_size)) {
    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE

    const optimistic_lock::write_guard node_guard{
        std::move(node_critical_section)};
    if (UNODB_DETAIL_UNLIKELY(node_guard.must_restart())) return {};

    optimistic_lock::write_guard child_guard{
        std::move(*child_critical_section)};
    if (UNODB_DETAIL_UNLIKELY(child_guard.must_restart())) return {};

    child_guard.unlock_and_obsolete();

    inode.remove(child_i, db_instance);

    *child_in_parent = nullptr;
    return true;
  }

  UNODB_DETAIL_ASSERT(is_node_min_size);

  if constexpr (std::is_same_v<INode, olc_inode_4<Key, Value>>) {
    const optimistic_lock::write_guard parent_guard{
        std::move(parent_critical_section)};
    if (UNODB_DETAIL_UNLIKELY(parent_guard.must_restart())) return {};

    optimistic_lock::write_guard node_guard{std::move(node_critical_section)};
    if (UNODB_DETAIL_UNLIKELY(node_guard.must_restart())) return {};

    optimistic_lock::write_guard child_guard{
        std::move(*child_critical_section)};
    if (UNODB_DETAIL_UNLIKELY(child_guard.must_restart())) return {};

    // For variable-length keys, check leave_last_child precondition:
    // prefix merge must not overflow key_prefix_capacity.
    if constexpr (std::is_same_v<Key, key_view>) {
      const std::uint8_t child_to_leave = (child_i == 0) ? 1U : 0U;
      const auto remaining = inode.get_child(child_to_leave);
      if (remaining.type() != node_type::LEAF) {
        const auto* const ri{
            remaining
                .template ptr<detail::olc_inode_base<Key, Value> const*>()};
        if (ri->get_key_prefix().length() + inode.get_key_prefix().length() +
                1 >
            detail::key_prefix_capacity) {
          child_guard.unlock_and_obsolete();
          inode.remove(child_i, db_instance);
          *child_in_parent = nullptr;
          return true;
        }
      }
    }
    auto current_node{olc_art_policy<Key, Value>::make_db_inode_reclaimable_ptr(
        &inode, db_instance)};
    node_guard.unlock_and_obsolete();
    child_guard.unlock_and_obsolete();
    *node_in_parent = current_node->leave_last_child(child_i, db_instance);

    UNODB_DETAIL_ASSERT(!node_guard.active());
    UNODB_DETAIL_ASSERT(!child_guard.active());

    *child_in_parent = nullptr;
  } else {
    auto smaller_node{INode::smaller_derived_type::create(db_instance, inode)};

    const optimistic_lock::write_guard parent_guard{
        std::move(parent_critical_section)};
    if (UNODB_DETAIL_UNLIKELY(parent_guard.must_restart())) return {};

    optimistic_lock::write_guard node_guard{std::move(node_critical_section)};
    if (UNODB_DETAIL_UNLIKELY(node_guard.must_restart())) return {};

    optimistic_lock::write_guard child_guard{
        std::move(*child_critical_section)};
    if (UNODB_DETAIL_UNLIKELY(child_guard.must_restart())) return {};

    smaller_node->init(db_instance, inode, node_guard, child_i, child_guard);
    *node_in_parent = detail::olc_node_ptr{smaller_node.release(),
                                           INode::smaller_derived_type::type};

    UNODB_DETAIL_ASSERT(!node_guard.active());
    UNODB_DETAIL_ASSERT(!child_guard.active());

    *child_in_parent = nullptr;
  }

#ifdef UNODB_DETAIL_WITH_STATS
  db_instance.template account_shrinking_inode<INode::type>();
#endif  // UNODB_DETAIL_WITH_STATS

  return true;
}

}  // namespace detail

//
// olc_db implementation
//

template <typename Key, typename Value>
olc_db<Key, Value>::~olc_db() noexcept {
  UNODB_DETAIL_ASSERT(
      qsbr_state::single_thread_mode(qsbr::instance().get_state()));

  delete_root_subtree();
}  // namespace >::~

template <typename Key, typename Value>
void olc_db<Key, Value>::delete_root_subtree() noexcept {
  UNODB_DETAIL_ASSERT(
      qsbr_state::single_thread_mode(qsbr::instance().get_state()));

  if (root != nullptr) art_policy::delete_subtree(root, *this);

#ifdef UNODB_DETAIL_WITH_STATS
  // It is possible to reset the counter to zero instead of decrementing it for
  // each leaf, but not sure the savings will be significant.
  UNODB_DETAIL_ASSERT(
      node_counts[as_i<node_type::LEAF>].load(std::memory_order_relaxed) == 0);
#endif  // UNODB_DETAIL_WITH_STATS
}

template <typename Key, typename Value>
void olc_db<Key, Value>::clear() noexcept {
  UNODB_DETAIL_ASSERT(
      qsbr_state::single_thread_mode(qsbr::instance().get_state()));

  delete_root_subtree();

  root = detail::olc_node_ptr{nullptr};

#ifdef UNODB_DETAIL_WITH_STATS
  current_memory_use.store(0, std::memory_order_relaxed);

  node_counts[as_i<node_type::I4>].store(0, std::memory_order_relaxed);
  node_counts[as_i<node_type::I16>].store(0, std::memory_order_relaxed);
  node_counts[as_i<node_type::I48>].store(0, std::memory_order_relaxed);
  node_counts[as_i<node_type::I256>].store(0, std::memory_order_relaxed);
#endif  // UNODB_DETAIL_WITH_STATS
}

template <typename Key, typename Value>
typename olc_db<Key, Value>::get_result olc_db<Key, Value>::get_internal(
    art_key_type k) const noexcept {
  try_get_result_type result;

  while (true) {
    result = try_get(k);
    if (result) break;
    // TODO(laurynas): upgrade to write locks to prevent starving after a
    // certain number of failures?
  }

  return *result;
}

template <typename Key, typename Value>
typename olc_db<Key, Value>::try_get_result_type olc_db<Key, Value>::try_get(
    art_key_type k) const noexcept {
  auto parent_critical_section = root_pointer_lock.try_read_lock();
  if (UNODB_DETAIL_UNLIKELY(parent_critical_section.must_restart())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }

  detail::olc_node_ptr node{root.load()};  // load root into [node].

  if (UNODB_DETAIL_UNLIKELY(node == nullptr)) {  // special path if empty tree.
    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock())) {
      // LCOV_EXCL_START
      spin_wait_loop_body();
      return {};
      // LCOV_EXCL_STOP
    }
    // return an empty result (breaks out of caller's while(true) loop)
    return std::make_optional<get_result>(std::nullopt);
  }

  // A check() is required before acting on [node] by taking the lock.
  if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }

  auto remaining_key{k};

  while (true) {
    // Lock version chaining (node and parent)
    auto node_critical_section = node_ptr_lock(node).try_read_lock();
    if (UNODB_DETAIL_UNLIKELY(node_critical_section.must_restart())) return {};
    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE

    const auto node_type = node.type();

    if (node_type == node_type::LEAF) {
      const auto* const leaf{node.ptr<leaf_type*>()};
      if (leaf->matches(k)) {
        const auto val_view{leaf->get_value_view()};
        if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
          return {};  // LCOV_EXCL_LINE
        return qsbr_ptr_span<const std::byte>{val_view};
      }
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
        return {};  // LCOV_EXCL_LINE
      return std::make_optional<get_result>(std::nullopt);
    }

    auto* const inode{node.ptr<inode_type*>()};
    const auto& key_prefix{inode->get_key_prefix()};
    const auto key_prefix_length{key_prefix.length()};
    const auto shared_key_prefix_length{
        key_prefix.get_shared_length(remaining_key)};

    if (shared_key_prefix_length < key_prefix_length) {
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
        return {};  // LCOV_EXCL_LINE
      return std::make_optional<get_result>(std::nullopt);
    }

    UNODB_DETAIL_ASSERT(shared_key_prefix_length == key_prefix_length);

    remaining_key.shift_right(key_prefix_length);

    const auto* const child_in_parent{
        inode->find_child(node_type, remaining_key[0]).second};

    if (child_in_parent == nullptr) {
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
        return {};  // LCOV_EXCL_LINE
      return std::make_optional<get_result>(std::nullopt);
    }

    const auto child = child_in_parent->load();

    parent_critical_section = std::move(node_critical_section);
    node = child;
    remaining_key.shift_right(1);

    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) return {};
  }
}

template <typename Key, typename Value>
bool olc_db<Key, Value>::insert_internal(art_key_type insert_key,
                                         value_type v) {
  try_update_result_type result;
  olc_db_leaf_unique_ptr_type cached_leaf{
      nullptr, detail::basic_db_leaf_deleter<olc_db<Key, Value>>{*this}};

  while (true) {
    result = try_insert(insert_key, v, cached_leaf);
    if (result) break;
  }

  return *result;
}

template <typename Key, typename Value>
typename olc_db<Key, Value>::try_update_result_type
olc_db<Key, Value>::try_insert(art_key_type k, value_type v,
                               olc_db_leaf_unique_ptr_type& cached_leaf) {
  auto parent_critical_section = root_pointer_lock.try_read_lock();
  if (UNODB_DETAIL_UNLIKELY(parent_critical_section.must_restart())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }

  auto node{root.load()};

  if (UNODB_DETAIL_UNLIKELY(node == nullptr)) {
    create_leaf_if_needed(cached_leaf, k, v, *this);

    const optimistic_lock::write_guard write_unlock_on_exit{
        std::move(parent_critical_section)};
    if (UNODB_DETAIL_UNLIKELY(write_unlock_on_exit.must_restart())) {
      // Do not call spin_wait_loop_body here - creating the leaf took some time
      return {};  // LCOV_EXCL_LINE
    }

    root = detail::olc_node_ptr{cached_leaf.release(), node_type::LEAF};
    return true;
  }

  auto* node_in_parent{&root};
  tree_depth_type depth{};
  auto remaining_key{k};

  if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }

  while (true) {
    auto node_critical_section = node_ptr_lock(node).try_read_lock();
    if (UNODB_DETAIL_UNLIKELY(node_critical_section.must_restart())) return {};

    const auto node_type = node.type();

    if (node_type == node_type::LEAF) {
      const auto* const leaf{node.template ptr<leaf_type*>()};
      const auto existing_key{leaf->get_key_view()};
      if (UNODB_DETAIL_UNLIKELY(k.cmp(existing_key) == 0)) {
        if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
          return {};  // LCOV_EXCL_LINE
        if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
          return {};  // LCOV_EXCL_LINE

        if (UNODB_DETAIL_UNLIKELY(cached_leaf != nullptr)) {
          cached_leaf.reset();  // LCOV_EXCL_LINE
        }
        return false;  // exists
      }

      // When the keys share more than key_prefix_capacity bytes at
      // this depth, the dispatch bytes collide and we cannot create a
      // two-child inode_4 directly.  Instead, create a single-child
      // chain inode_4 and restart — the next attempt will descend
      // through the chain and repeat until the keys diverge.
      constexpr auto cap = detail::key_prefix_capacity;
      const auto remaining_existing = existing_key.subspan(depth);
      const auto shared = detail::key_prefix_snapshot::shared_len(
          detail::get_u64(remaining_existing), remaining_key.get_u64(), cap);
      if (shared >= cap && remaining_existing.size() > cap &&
          remaining_key.size() > cap &&
          remaining_existing[cap] == remaining_key[cap]) {
        const auto dispatch = remaining_existing[cap];
        auto chain{inode_4::create(*this, existing_key, remaining_key, depth,
                                   dispatch, node)};
        {
          const optimistic_lock::write_guard parent_guard{
              std::move(parent_critical_section)};
          if (UNODB_DETAIL_UNLIKELY(parent_guard.must_restart())) return {};

          const optimistic_lock::write_guard node_guard{
              std::move(node_critical_section)};
          if (UNODB_DETAIL_UNLIKELY(node_guard.must_restart())) return {};

          *node_in_parent =
              detail::olc_node_ptr{chain.release(), node_type::I4};
        }
#ifdef UNODB_DETAIL_WITH_STATS
        account_growing_inode<node_type::I4>();
#endif              // UNODB_DETAIL_WITH_STATS
        return {};  // restart to descend through the new chain node
      }

      create_leaf_if_needed(cached_leaf, k, v, *this);
      auto new_node{inode_4::create(*this, existing_key, remaining_key, depth)};

      {
        const optimistic_lock::write_guard parent_guard{
            std::move(parent_critical_section)};
        if (UNODB_DETAIL_UNLIKELY(parent_guard.must_restart())) return {};

        const optimistic_lock::write_guard node_guard{
            std::move(node_critical_section)};
        if (UNODB_DETAIL_UNLIKELY(node_guard.must_restart())) return {};

        new_node->init(existing_key, remaining_key, depth, leaf,
                       std::move(cached_leaf));
        *node_in_parent =
            detail::olc_node_ptr{new_node.release(), node_type::I4};
      }
#ifdef UNODB_DETAIL_WITH_STATS
      account_growing_inode<node_type::I4>();
#endif  // UNODB_DETAIL_WITH_STATS
      return true;
    }

    UNODB_DETAIL_ASSERT(node_type != node_type::LEAF);

    auto* const inode{node.template ptr<inode_type*>()};
    const auto& key_prefix{inode->get_key_prefix()};
    const auto key_prefix_length{key_prefix.length()};
    const auto shared_prefix_length{
        key_prefix.get_shared_length(remaining_key)};

    if (shared_prefix_length < key_prefix_length) {
      create_leaf_if_needed(cached_leaf, k, v, *this);
      auto new_node{inode_4::create(*this, node, shared_prefix_length)};

      {
        const optimistic_lock::write_guard parent_guard{
            std::move(parent_critical_section)};
        if (UNODB_DETAIL_UNLIKELY(parent_guard.must_restart())) return {};

        const optimistic_lock::write_guard node_guard{
            std::move(node_critical_section)};
        if (UNODB_DETAIL_UNLIKELY(node_guard.must_restart())) return {};

        new_node->init(node, shared_prefix_length, depth,
                       std::move(cached_leaf));
        *node_in_parent =
            detail::olc_node_ptr{new_node.release(), node_type::I4};
      }

#ifdef UNODB_DETAIL_WITH_STATS
      account_growing_inode<node_type::I4>();
      key_prefix_splits.fetch_add(1, std::memory_order_relaxed);
#endif  // UNODB_DETAIL_WITH_STATS

      return true;
    }

    UNODB_DETAIL_ASSERT(shared_prefix_length == key_prefix_length);

    depth += key_prefix_length;
    remaining_key.shift_right(key_prefix_length);

    const auto add_result{inode->template add_or_choose_subtree<
        std::optional<in_critical_section<detail::olc_node_ptr>*>>(
        node_type, remaining_key[0], k, v, *this, depth, node_critical_section,
        node_in_parent, parent_critical_section, cached_leaf)};

    if (UNODB_DETAIL_UNLIKELY(!add_result)) return {};

    auto* const child_in_parent = *add_result;
    if (child_in_parent == nullptr) return true;

    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE

    const auto child = child_in_parent->load();

    parent_critical_section = std::move(node_critical_section);
    node = child;
    node_in_parent = child_in_parent;
    ++depth;
    remaining_key.shift_right(1);

    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) return {};
  }
}

template <typename Key, typename Value>
bool olc_db<Key, Value>::remove_internal(art_key_type remove_key) {
  try_update_result_type result;
  while (true) {
    result = try_remove(remove_key);
    if (result) break;
  }

  return *result;
}

template <typename Key, typename Value>
typename olc_db<Key, Value>::try_update_result_type
olc_db<Key, Value>::try_remove(art_key_type k) {
  if constexpr (std::is_same_v<Key, key_view>) {
    return try_remove_key_view(k);
  } else {
    return try_remove_fixed_width_key(k);
  }
}

template <typename Key, typename Value>
typename olc_db<Key, Value>::try_update_result_type
olc_db<Key, Value>::try_remove_key_view(art_key_type k) {
  auto parent_critical_section = root_pointer_lock.try_read_lock();
  if (UNODB_DETAIL_UNLIKELY(parent_critical_section.must_restart())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }

  auto node{root.load()};

  if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }

  if (UNODB_DETAIL_UNLIKELY(node == nullptr)) return false;

  auto node_critical_section = node_ptr_lock(node).try_read_lock();
  if (UNODB_DETAIL_UNLIKELY(node_critical_section.must_restart())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }

  auto node_type = node.type();

  if (node_type == node_type::LEAF) {
    auto* const leaf{node.template ptr<leaf_type*>()};
    if (leaf->matches(k)) {
      const optimistic_lock::write_guard parent_guard{
          std::move(parent_critical_section)};
      // Do not call spin_wait_loop_body from this point on - assume
      // the above took enough time.
      if (UNODB_DETAIL_UNLIKELY(parent_guard.must_restart())) return {};

      optimistic_lock::write_guard node_guard{std::move(node_critical_section)};
      if (UNODB_DETAIL_UNLIKELY(node_guard.must_restart())) return {};

      node_guard.unlock_and_obsolete();

      const auto r{art_policy::reclaim_leaf_on_scope_exit(leaf, *this)};
      root = detail::olc_node_ptr{nullptr};
      return true;
    }

    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE

    return false;
  }

  auto* node_in_parent{&root};
  tree_depth_type depth{};
  auto remaining_key{k};

  remove_stack_type stk;
  std::size_t stk_n = 0;

  while (true) {
    UNODB_DETAIL_ASSERT(node_type != node_type::LEAF);

    auto* const inode{node.template ptr<inode_type*>()};
    const auto& key_prefix{inode->get_key_prefix()};
    const auto key_prefix_length{key_prefix.length()};
    const auto shared_prefix_length{
        key_prefix.get_shared_length(remaining_key)};

    if (shared_prefix_length < key_prefix_length) {
      if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
        return {};  // LCOV_EXCL_LINE
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
        return {};  // LCOV_EXCL_LINE
      return false;
    }

    UNODB_DETAIL_ASSERT(shared_prefix_length == key_prefix_length);
    depth += key_prefix_length;
    remaining_key.shift_right(key_prefix_length);

    // --- Chain I4 detection ---
    if (node_type == node_type::I4 && inode->get_children_count() == 1) {
      const auto [ci, cp]{inode->find_child(node_type, remaining_key[0])};
      if (cp == nullptr) {
        // LCOV_EXCL_START — single-child I4: if the prefix matched,
        // the dispatch byte is determined by the shared key bytes, so
        // find_child always succeeds.  nullptr only via concurrent race.
        if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
          return {};
        if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
          return {};
        return false;
        // LCOV_EXCL_STOP
      }
      const auto cv{cp->load()};
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check())) return {};

      if (cv.type() == node_type::LEAF) {
        // Leaf under chain I4.  Verify match.
        UNODB_DETAIL_DISABLE_MSVC_WARNING(26462)
        const auto* const leaf{cv.template ptr<leaf_type*>()};
        UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
        if (!leaf->matches(k)) {
          if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
            return {};  // LCOV_EXCL_LINE
          if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
            return {};  // LCOV_EXCL_LINE
          return false;
        }

        // Write-lock parent, chain, and leaf.
        optimistic_lock::write_guard pg{std::move(parent_critical_section)};
        if (UNODB_DETAIL_UNLIKELY(pg.must_restart())) return {};
        optimistic_lock::write_guard ng{std::move(node_critical_section)};
        if (UNODB_DETAIL_UNLIKELY(ng.must_restart())) return {};
        auto lcs = node_ptr_lock(cv).try_read_lock();
        if (UNODB_DETAIL_UNLIKELY(lcs.must_restart())) return {};
        optimistic_lock::write_guard lg{std::move(lcs)};
        if (UNODB_DETAIL_UNLIKELY(lg.must_restart())) return {};

        // ============================================================
        return try_chain_cut(node, cv, std::move(pg), std::move(ng),
                             std::move(lg), stk, stk_n);
      }

      // Chain child is not a leaf — push and descend.
      stk.push_back({node, ci, remaining_key[0], node_critical_section.get(),
                     node_in_parent});
      ++stk_n;

      auto ccs = node_ptr_lock(cv).try_read_lock();
      if (UNODB_DETAIL_UNLIKELY(ccs.must_restart())) return {};

      parent_critical_section = std::move(node_critical_section);
      node = cv;
      node_in_parent = cp;
      node_critical_section = std::move(ccs);
      node_type = cv.type();
      ++depth;
      remaining_key.shift_right(1);
      if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) return {};
      continue;
    }

    // --- Non-chain: delegate to existing remove_or_choose_subtree ---
    // Capture stack entry BEFORE remove_or_choose_subtree consumes
    // the critical sections (it may move them into write guards).
    const auto pre_version = node_critical_section.get();
    const auto pre_key_byte = remaining_key[0];
    const auto [pre_ci, pre_cp]{inode->find_child(node_type, remaining_key[0])};

    UNODB_DETAIL_DISABLE_MSVC_WARNING(26494)
    in_critical_section<detail::olc_node_ptr>* child_in_parent;
    enum node_type child_type;
    detail::olc_node_ptr child;
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

    optimistic_lock::read_critical_section child_critical_section;

    const auto opt_remove_result{
        inode->template remove_or_choose_subtree<std::optional<bool>>(
            node_type, remaining_key[0], k, *this, parent_critical_section,
            node_critical_section, node_in_parent, &child_in_parent,
            &child_critical_section, &child_type, &child)};

    if (UNODB_DETAIL_UNLIKELY(!opt_remove_result)) return {};

    if (const auto remove_result{*opt_remove_result}; !remove_result)
      return false;

    if (child_in_parent == nullptr) return true;

    // Push stack entry for the non-chain node we just traversed.
    if (pre_cp != nullptr) {
      stk.push_back({node, pre_ci, pre_key_byte, pre_version, node_in_parent});
      ++stk_n;
    }

    parent_critical_section = std::move(node_critical_section);
    node = child;
    node_in_parent = child_in_parent;
    node_critical_section = std::move(child_critical_section);
    node_type = child_type;

    ++depth;
    remaining_key.shift_right(1);
  }
}

template <typename Key, typename Value>
typename olc_db<Key, Value>::try_update_result_type
olc_db<Key, Value>::try_remove_fixed_width_key(art_key_type k) {
  auto parent_critical_section = root_pointer_lock.try_read_lock();
  if (UNODB_DETAIL_UNLIKELY(parent_critical_section.must_restart())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }

  auto node{root.load()};

  if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }

  if (UNODB_DETAIL_UNLIKELY(node == nullptr)) return false;

  auto node_critical_section = node_ptr_lock(node).try_read_lock();
  if (UNODB_DETAIL_UNLIKELY(node_critical_section.must_restart())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }

  auto node_type = node.type();

  if (node_type == node_type::LEAF) {
    auto* const leaf{node.template ptr<leaf_type*>()};
    if (leaf->matches(k)) {
      const optimistic_lock::write_guard parent_guard{
          std::move(parent_critical_section)};
      // Do not call spin_wait_loop_body from this point on - assume
      // the above took enough time
      if (UNODB_DETAIL_UNLIKELY(parent_guard.must_restart())) return {};

      optimistic_lock::write_guard node_guard{std::move(node_critical_section)};
      if (UNODB_DETAIL_UNLIKELY(node_guard.must_restart())) return {};

      node_guard.unlock_and_obsolete();

      const auto r{art_policy::reclaim_leaf_on_scope_exit(leaf, *this)};
      root = detail::olc_node_ptr{nullptr};
      return true;
    }

    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
      return {};  // LCOV_EXCL_LINE

    return false;
  }

  auto* node_in_parent{&root};
  tree_depth_type depth{};
  auto remaining_key{k};

  while (true) {
    UNODB_DETAIL_ASSERT(node_type != node_type::LEAF);

    auto* const inode{node.template ptr<inode_type*>()};
    const auto& key_prefix{inode->get_key_prefix()};
    const auto key_prefix_length{key_prefix.length()};
    const auto shared_prefix_length{
        key_prefix.get_shared_length(remaining_key)};

    if (shared_prefix_length < key_prefix_length) {
      if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
        return {};  // LCOV_EXCL_LINE
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
        return {};  // LCOV_EXCL_LINE

      return false;
    }

    UNODB_DETAIL_ASSERT(shared_prefix_length == key_prefix_length);
    depth += key_prefix_length;
    remaining_key.shift_right(key_prefix_length);

    in_critical_section<detail::olc_node_ptr>* child_in_parent{nullptr};
    enum node_type child_type {};
    detail::olc_node_ptr child{nullptr};
    optimistic_lock::read_critical_section child_critical_section;

    const auto opt_remove_result{
        inode->template remove_or_choose_subtree<std::optional<bool>>(
            node_type, remaining_key[0], k, *this, parent_critical_section,
            node_critical_section, node_in_parent, &child_in_parent,
            &child_critical_section, &child_type, &child)};

    if (UNODB_DETAIL_UNLIKELY(!opt_remove_result)) return {};

    if (const auto remove_result{*opt_remove_result}; !remove_result)
      return false;

    if (child_in_parent == nullptr) return true;

    parent_critical_section = std::move(node_critical_section);
    node = child;
    node_in_parent = child_in_parent;
    node_critical_section = std::move(child_critical_section);
    node_type = child_type;

    ++depth;
    remaining_key.shift_right(1);
  }
}

///
/// ART iterator implementation.
///

template <typename Key, typename Value>
typename olc_db<Key, Value>::iterator& olc_db<Key, Value>::iterator::first() {
  while (!try_first()) {
    unodb::spin_wait_loop_body();  // LCOV_EXCL_LINE
  }
  return *this;
}

// Traverse to the left-most leaf. The stack is cleared first and then
// re-populated as we step down along the path to the left-most leaf.
// If the tree is empty, then the result is the same as end().
template <typename Key, typename Value>
bool olc_db<Key, Value>::iterator::try_first() {
  invalidate();  // clear the stack
  auto parent_critical_section = db_.root_pointer_lock.try_read_lock();
  if (UNODB_DETAIL_UNLIKELY(parent_critical_section.must_restart()))
    return false;  // LCOV_EXCL_LINE
  auto node{db_.root.load()};
  if (UNODB_DETAIL_UNLIKELY(node == nullptr)) {
    return UNODB_DETAIL_LIKELY(parent_critical_section.try_read_unlock());
  }
  return try_left_most_traversal(node, parent_critical_section);
}

template <typename Key, typename Value>
typename olc_db<Key, Value>::iterator& olc_db<Key, Value>::iterator::last() {
  while (!try_last()) {
    unodb::spin_wait_loop_body();  // LCOV_EXCL_LINE
  }
  return *this;
}

// Traverse to the right-most leaf. The stack is cleared first and then
// re-populated as we step down along the path to the right-most leaf.
// If the tree is empty, then the result is the same as end().
template <typename Key, typename Value>
bool olc_db<Key, Value>::iterator::try_last() {
  invalidate();  // clear the stack
  auto parent_critical_section = db_.root_pointer_lock.try_read_lock();
  if (UNODB_DETAIL_UNLIKELY(parent_critical_section.must_restart()))
    return false;  // LCOV_EXCL_LINE
  auto node{db_.root.load()};
  if (UNODB_DETAIL_UNLIKELY(node == nullptr)) {
    return UNODB_DETAIL_LIKELY(parent_critical_section.try_read_unlock());
  }
  return try_right_most_traversal(node, parent_critical_section);
}

template <typename Key, typename Value>
typename olc_db<Key, Value>::iterator& olc_db<Key, Value>::iterator::next() {
  const auto node = current_node();
  if (node != nullptr) {
    UNODB_DETAIL_ASSERT(node.type() == node_type::LEAF);      // On a leaf.
    const auto* const leaf{node.template ptr<leaf_type*>()};  // current leaf
    // TODO(thompsonbry) : variable length keys: We need a temporary
    // copy of the key since actions on the stack will make it
    // impossible to reconstruct the key.  So maybe we have two
    // internal buffers on the iterator to support this?
    const auto& akey = leaf->get_key();  // access the key on the leaf.
    if (UNODB_DETAIL_LIKELY(try_next())) return *this;
    while (true) {
      bool match{};
      // seek to the current key (or its successor).
      if (!try_seek(akey, match, true /*fwd*/)) continue;
      if (!match) {
        // The key no longer exists, so its successor is the next leaf
        // and we are done.
        return *this;
      }
      if (!try_next()) continue;  // seek to the successor
      return *this;               // done.
    }
  }
  return *this;  // LCOV_EXCL_LINE
}

template <typename Key, typename Value>
bool olc_db<Key, Value>::iterator::try_next() {
  while (!empty()) {
    const auto& e = top();
    const auto node{e.node};  // the node on the top of the stack.
    UNODB_DETAIL_ASSERT(node != nullptr);
    auto node_critical_section(
        node_ptr_lock(node).rehydrate_read_lock(e.version));
    // Restart check (fails if node was modified after it was pushed
    // onto the stack).
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check())) return false;
    const auto node_type = node.type();
    if (node_type == node_type::LEAF) {
      pop();  // pop off the leaf
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
        return false;  // LCOV_EXCL_LINE
      continue;        // falls through loop if just a root leaf since stack now
                       // empty.
    }
    auto* inode{node.template ptr<inode_type*>()};
    auto nxt = inode->next(node_type,
                           e.child_index);  // next child of that parent.
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check())) {
      // restart check
      return false;  // LCOV_EXCL_LINE
    }
    if (!nxt.has_value()) {
      pop();  // Nothing more for that inode.
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
        return false;  // LCOV_EXCL_LINE
      continue;  // We will look for the right sibling of the parent inode.
    }
    // Fix up stack for new parent node state and left-most descent.
    const auto& e2 = nxt.value();
    pop();
    if (UNODB_DETAIL_UNLIKELY(!try_push(e2, node_critical_section)))
      return false;                                            // LCOV_EXCL_LINE
    auto child = inode->get_child(node_type, e2.child_index);  // descend
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check()))  // before using
      return false;  // LCOV_EXCL_LINE
    return try_left_most_traversal(child, node_critical_section);
  }
  return true;  // stack is empty, so iterator == end().
}

template <typename Key, typename Value>
typename olc_db<Key, Value>::iterator& olc_db<Key, Value>::iterator::prior() {
  const auto node = current_node();
  if (node != nullptr) {
    UNODB_DETAIL_ASSERT(node.type() == node_type::LEAF);      // On a leaf.
    const auto* const leaf{node.template ptr<leaf_type*>()};  // current leaf
    // TODO(thompsonbry) : variable length keys: We need a temporary
    // copy of the key since actions on the stack will make it
    // impossible to reconstruct the key.  So maybe we have two
    // internal buffers on the iterator to support this?
    const auto& akey = leaf->get_key();  // access the key on the leaf.
    if (UNODB_DETAIL_LIKELY(try_prior())) return *this;
    while (true) {
      bool match{};
      // seek to the current key (or its predecessor)
      if (!try_seek(akey, match, false /*fwd*/)) continue;
      if (!match) {
        // The key no longer exists, so its predecessor is the prior
        // leaf and we are done.
        return *this;
      }
      if (!try_prior()) continue;  // seek to the predecessor
      return *this;                // done.
    }
  }
  return *this;  // LCOV_EXCL_LINE
}

// Position the iterator on the prior leaf in the index.
template <typename Key, typename Value>
bool olc_db<Key, Value>::iterator::try_prior() {
  while (!empty()) {
    const auto& e = top();
    const auto node{e.node};  // the node on the top of the stack.
    UNODB_DETAIL_ASSERT(node != nullptr);
    auto node_critical_section(
        node_ptr_lock(node).rehydrate_read_lock(e.version));
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check()))
      return false;  // LCOV_EXCL_LINE
    const auto node_type = node.type();
    if (node_type == node_type::LEAF) {
      pop();  // pop off the leaf
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
        return false;  // LCOV_EXCL_LINE
      continue;  // falls through loop if just a root leaf since stack now empty
    }
    auto* inode{node.template ptr<inode_type*>()};
    auto nxt = inode->prior(node_type, e.child_index);  // prev child of parent
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check()))
      return false;  // LCOV_EXCL_LINE
    if (!nxt) {
      pop();  // Nothing more for that inode.
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
        return false;  // LCOV_EXCL_LINE
      continue;        // We will look for the left sibling of the parent inode.
    }
    // Fix up stack for new parent node state and right-most descent.
    UNODB_DETAIL_ASSERT(nxt);  // value exists for std::optional.
    const auto& e2 = nxt.value();
    pop();
    if (UNODB_DETAIL_UNLIKELY(!try_push(e2, node_critical_section)))
      return false;                                            // LCOV_EXCL_LINE
    auto child = inode->get_child(node_type, e2.child_index);  // get child
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check()))  // before using
      return false;  // LCOV_EXCL_LINE
    return try_right_most_traversal(child, node_critical_section);
  }
  return true;  // stack is empty, so iterator == end().
}

template <typename Key, typename Value>
typename olc_db<Key, Value>::iterator& olc_db<Key, Value>::iterator::seek(
    art_key_type search_key, bool& match, bool fwd) {
  while (!try_seek(search_key, match, fwd)) {
    unodb::spin_wait_loop_body();  // LCOV_EXCL_LINE
  }
  return *this;
}

// Ensure that the read_critical_section is unlocked regardless of the
// outcome of some computation.
//
// \return the outcome of that computation and false if the
// read_critical_section could not be unlocked.
[[nodiscard]] inline bool unlock_and_return(
    const unodb::optimistic_lock::read_critical_section& cs,
    bool ret) noexcept {
  return UNODB_DETAIL_LIKELY(cs.try_read_unlock()) ? ret : false;
}

// Note: The basic seek() logic is similar to ::get() as long as the
// search_key exists in the data.  However, the iterator is positioned
// instead of returning the value for the key.  Life gets a lot more
// complicated when the search_key is not in the data and we have to
// consider the cases for both forward traversal and reverse traversal
// from a key that is not in the data.  See seek() method declaration
// for details.
//
// TODO(thompsonbry) We could do partial invalidation, in which case
// caller's might need to explicitly unwind the stack to the first
// valid node.  This is deferred for now as it would make the logic
// more complicated and there is no data as yet about the importance
// of this (which just optimizes part of the seek away) while the code
// complexity would be definitely increased.
template <typename Key, typename Value>
bool olc_db<Key, Value>::iterator::try_seek(art_key_type search_key,
                                            bool& match, bool fwd) {
  invalidate();   // invalidate the iterator (clear the stack).
  match = false;  // unless we wind up with an exact match.
  auto parent_critical_section = db_.root_pointer_lock.try_read_lock();
  if (UNODB_DETAIL_UNLIKELY(parent_critical_section.must_restart())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return false;
    // LCOV_EXCL_STOP
  }
  auto node{db_.root.load()};
  if (UNODB_DETAIL_UNLIKELY(node == nullptr)) {
    return UNODB_DETAIL_LIKELY(parent_critical_section.try_read_unlock());
  }
  // A check() is required before acting on [node] by taking the lock.
  if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return false;
    // LCOV_EXCL_STOP
  }
  const auto k = search_key;
  auto remaining_key{k};
  while (true) {
    UNODB_DETAIL_ASSERT(node != nullptr);
    // Lock version chaining (node and parent)
    auto node_critical_section = node_ptr_lock(node).try_read_lock();
    if (UNODB_DETAIL_UNLIKELY(node_critical_section.must_restart()))
      return false;  // LCOV_EXCL_LINE
    // TODO(thompsonbry) Should be redundant.  Checked before entering
    // the while() loop and at the bottom of the while() loop.
    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) return false;
    // Note: We DO NOT unlock the parent_critical_section here.  It is
    // done below along all code paths.
    const auto node_type = node.type();
    if (node_type == node_type::LEAF) {
      if (UNODB_DETAIL_UNLIKELY(
              !parent_critical_section.try_read_unlock()))  // unlock parent
        return false;                                       // LCOV_EXCL_LINE
      const auto* const leaf{node.template ptr<leaf_type*>()};
      if (UNODB_DETAIL_UNLIKELY(!try_push_leaf(node, node_critical_section)))
        return false;  // LCOV_EXCL_LINE
      const auto cmp_ = leaf->cmp(k);
      if (UNODB_DETAIL_UNLIKELY(!node_critical_section.try_read_unlock()))
        return false;  // LCOV_EXCL_LINE
      if (cmp_ == 0) {
        match = true;
        return true;  // done
      }
      if (fwd) {  // GTE semantics
        // if search_key < leaf, use leaf, else next().
        return (cmp_ < 0) ? true : try_next();
      }
      // LTE semantics: if search_key > leaf, use leaf, else prior().
      return (cmp_ > 0) ? true : try_prior();
    }
    UNODB_DETAIL_ASSERT(node_type != node_type::LEAF);
    auto* const inode{node.template ptr<inode_type*>()};  // some internal node.
    const auto key_prefix{inode->get_key_prefix().get_snapshot()};  // prefix
    const auto key_prefix_length{key_prefix.length()};  // length of that prefix
    const auto shared_length = key_prefix.get_shared_length(
        remaining_key.get_u64());  // #of prefix bytes matched.
    if (shared_length < key_prefix_length) {
      // We have visited an internal node whose prefix is longer than
      // the bytes in the key that we need to match.  To figure out
      // whether the search key would be located before or after the
      // current internal node, we need to compare the respective key
      // spans lexicographically.  Since we have [shared_length] bytes
      // in common, we know that the next byte will tell us the
      // relative ordering of the key vs the prefix. So now we compare
      // prefix and key and the first byte where they differ.
      const auto cmp_ = static_cast<int>(remaining_key[shared_length]) -
                        static_cast<int>(key_prefix[shared_length]);
      UNODB_DETAIL_ASSERT(cmp_ != 0);
      if (fwd) {
        // Note: parent_critical_section is unlocked along all paths
        // by try_(left|right)_most_traversal
        if (cmp_ < 0) {
          // FWD and the search key is ordered before this node.  We
          // want the left-most leaf under the node.
          return unlock_and_return(
              node_critical_section,
              try_left_most_traversal(node, parent_critical_section));
        }
        // FWD and the search key is ordered after this node.  Right
        // most descent and then next().
        return unlock_and_return(
                   node_critical_section,
                   try_right_most_traversal(node, parent_critical_section)) &&
               try_next();
      }
      // reverse traversal
      if (cmp_ < 0) {
        // REV and the search key is ordered before this node.  We
        // want the preceding key.
        return unlock_and_return(
                   node_critical_section,
                   try_left_most_traversal(node, parent_critical_section)) &&
               try_prior();
      }
      // REV and the search key is ordered after this node.
      return unlock_and_return(
          node_critical_section,
          try_right_most_traversal(node, parent_critical_section));
    }
    remaining_key.shift_right(key_prefix_length);
    const auto res = inode->find_child(node_type, remaining_key[0]);
    if (res.second == nullptr) {
      // We are on a key byte during the descent that is not mapped by
      // the current node.  Where we go next depends on whether we are
      // doing forward or reverse traversal.
      if (fwd) {
        // FWD: Take the next child_index that is mapped in the data
        // and then do a left-most descent to land on the key that is
        // the immediate successor of the desired key in the data.
        //
        // Note: We are probing with a key byte which does not appear
        // in our list of keys (this was verified above) so this will
        // always be the index the first entry whose key byte is
        // greater-than the probe value and [false] if there is no
        // such entry.
        //
        // Note: [node] has not been pushed onto the stack yet!
        auto nxt = inode->gte_key_byte(node_type, remaining_key[0]);
        if (!nxt) {
          // Pop entries off the stack until we find one with a
          // right-sibling of the path we took to this node and then
          // do a left-most descent under that right-sibling. If there
          // is no such parent, we will wind up with an empty stack
          // (aka the end() iterator) and return that state.
          if (UNODB_DETAIL_UNLIKELY(
                  !parent_critical_section.try_read_unlock()))  // unlock parent
            return false;  // LCOV_EXCL_LINE
          if (UNODB_DETAIL_UNLIKELY(
                  !node_critical_section.try_read_unlock()))  // unlock node
            return false;                                     // LCOV_EXCL_LINE
          if (!empty()) pop();
          while (!empty()) {
            const auto& centry = top();
            const auto cnode{centry.node};  // a possible parent from the stack.
            auto c_critical_section(
                node_ptr_lock(cnode).rehydrate_read_lock(centry.version));
            if (UNODB_DETAIL_UNLIKELY(!c_critical_section.check()))
              return false;  // LCOV_EXCL_LINE
            auto* const icnode{cnode.template ptr<inode_type*>()};
            const auto cnxt = icnode->next(
                cnode.type(), centry.child_index);  // right-sibling.
            if (cnxt) {
              auto nchild = icnode->get_child(
                  cnode.type(), centry.child_index);  // get the child
              if (UNODB_DETAIL_UNLIKELY(
                      !c_critical_section.check()))  // before using [nchild]
                return false;                        // LCOV_EXCL_LINE
              return try_left_most_traversal(nchild, c_critical_section);
            }
            pop();
            if (UNODB_DETAIL_UNLIKELY(!c_critical_section.try_read_unlock()))
              return false;  // LCOV_EXCL_LINE
          }
          return true;  // stack is empty (aka end()).
        }
        const auto& tmp = nxt.value();  // unwrap.
        const auto child_index = tmp.child_index;
        const auto child =
            inode->get_child(node_type, child_index);  // get child
        if (UNODB_DETAIL_UNLIKELY(
                !node_critical_section.check()))  // before using [child]
          return false;                           // LCOV_EXCL_LINE
        if (UNODB_DETAIL_UNLIKELY(
                !parent_critical_section.try_read_unlock()))  // unlock parent
          return false;                                       // LCOV_EXCL_LINE
        // push the path we took
        if (UNODB_DETAIL_UNLIKELY(!try_push(node, tmp.key_byte, child_index,
                                            tmp.prefix, node_critical_section)))
          return false;  // LCOV_EXCL_LINE
        return try_left_most_traversal(child, node_critical_section);
      }
      // REV: Take the prior child_index that is mapped and then do
      // a right-most descent to land on the key that is the
      // immediate predecessor of the desired key in the data.
      auto nxt = inode->lte_key_byte(node_type, remaining_key[0]);
      if (!nxt) {
        // Pop off the current entry until we find one with a
        // left-sibling and then do a right-most descent under that
        // left-sibling.  In the extreme case there is no such
        // previous entry and we will wind up with an empty stack.
        if (UNODB_DETAIL_UNLIKELY(
                !parent_critical_section.try_read_unlock()))  // unlock parent
          return false;                                       // LCOV_EXCL_LINE
        if (UNODB_DETAIL_UNLIKELY(
                !node_critical_section.try_read_unlock()))  // unlock node
          return false;                                     // LCOV_EXCL_LINE
        if (!empty()) pop();
        while (!empty()) {
          const auto& centry = top();
          const auto cnode{centry.node};  // a possible parent from stack
          auto c_critical_section(
              node_ptr_lock(cnode).rehydrate_read_lock(centry.version));
          if (UNODB_DETAIL_UNLIKELY(!c_critical_section.check()))
            return false;  // LCOV_EXCL_LINE
          auto* const icnode{cnode.template ptr<inode_type*>()};
          const auto cnxt =
              icnode->prior(cnode.type(), centry.child_index);  // left-sibling.
          if (cnxt) {
            auto nchild = icnode->get_child(
                cnode.type(), centry.child_index);  // get the child
            if (UNODB_DETAIL_UNLIKELY(
                    !c_critical_section.check()))  // before using [nchild]
              return false;                        // LCOV_EXCL_LINE
            return try_right_most_traversal(nchild, c_critical_section);
          }
          pop();
          if (UNODB_DETAIL_UNLIKELY(!c_critical_section.try_read_unlock()))
            return false;  // LCOV_EXCL_LINE
        }
        return true;  // stack is empty (aka end()).
      }
      const auto& tmp = nxt.value();  // unwrap.
      const auto child_index = tmp.child_index;
      const auto child =
          inode->get_child(node_type, child_index);  // get the child
      if (UNODB_DETAIL_UNLIKELY(
              !node_critical_section.check()))  // before using [child]
        return false;                           // LCOV_EXCL_LINE
      if (UNODB_DETAIL_UNLIKELY(
              !parent_critical_section.try_read_unlock()))  // unlock parent
        return false;                                       // LCOV_EXCL_LINE
      // push the path we took
      if (UNODB_DETAIL_UNLIKELY(!try_push(node, tmp.key_byte, child_index,
                                          tmp.prefix, node_critical_section)))
        return false;  // LCOV_EXCL_LINE
      return try_right_most_traversal(child, node_critical_section);
    }
    // Simple case. There is a child for the current key byte.
    const auto child_index{res.first};
    const auto* const child{res.second};
    if (UNODB_DETAIL_UNLIKELY(!try_push(node, remaining_key[0], child_index,
                                        key_prefix, node_critical_section)))
      return false;  // LCOV_EXCL_LINE
    node = *child;
    remaining_key.shift_right(1);
    // check node before using [child] and before we std::move() the RCS.
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check()))
      return false;  // LCOV_EXCL_LINE
    // Move RCS (will check invariant at top of loop)
    parent_critical_section = std::move(node_critical_section);
  }  // while ( true )
  UNODB_DETAIL_CANNOT_HAPPEN();
}

// Push the given node onto the stack and traverse from the caller's
// node to the left-most leaf under that node, pushing nodes onto the
// stack as they are visited.  An optimistic lock is obtained for the
// caller's node and the parent critical section is then released
// (lock chaining).  No optimistic locks are held on exit.
template <typename Key, typename Value>
bool olc_db<Key, Value>::iterator::try_left_most_traversal(
    detail::olc_node_ptr node,
    optimistic_lock::read_critical_section& parent_critical_section) {
  // A check() is required before acting on [node] by taking the lock.
  if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }
  while (true) {
    UNODB_DETAIL_ASSERT(node != nullptr);
    // Lock version chaining (node and parent)
    auto node_critical_section = node_ptr_lock(node).try_read_lock();
    if (UNODB_DETAIL_UNLIKELY(node_critical_section.must_restart()))
      return false;  // LCOV_EXCL_LINE
    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
      return false;  // LCOV_EXCL_LINE
    const auto node_type = node.type();
    if (node_type == node_type::LEAF) {
      if (UNODB_DETAIL_UNLIKELY(!try_push_leaf(node, node_critical_section)))
        return false;  // LCOV_EXCL_LINE
      return UNODB_DETAIL_LIKELY(node_critical_section.try_read_unlock());
    }
    // recursive descent.
    auto* const inode{node.ptr<inode_type*>()};
    const auto t =
        inode->begin(node_type);  // first chold of current internal node
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check()))
      return false;  // LCOV_EXCL_LINE
    if (UNODB_DETAIL_UNLIKELY(!try_push(t, node_critical_section)))
      return false;                                     // LCOV_EXCL_LINE
    node = inode->get_child(node_type, t.child_index);  // get child
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check()))  // before using
      return false;  // LCOV_EXCL_LINE
    // Move RCS (will check invariant at top of loop)
    parent_critical_section = std::move(node_critical_section);
  }
  UNODB_DETAIL_CANNOT_HAPPEN();
}

// Push the given node onto the stack and traverse from the caller's
// node to the right-most leaf under that node, pushing nodes onto the
// stack as they are visited.  An optimistic lock is obtained for the
// caller's node and the parent critical section is then released
// (lock chaining). No optimistic locks are held on exit.
template <typename Key, typename Value>
bool olc_db<Key, Value>::iterator::try_right_most_traversal(
    detail::olc_node_ptr node,
    optimistic_lock::read_critical_section& parent_critical_section) {
  // A check() is required before acting on [node] by taking the lock.
  if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) {
    // LCOV_EXCL_START
    spin_wait_loop_body();
    return {};
    // LCOV_EXCL_STOP
  }
  while (true) {
    UNODB_DETAIL_ASSERT(node != nullptr);
    // Lock version chaining (node and parent)
    auto node_critical_section = node_ptr_lock(node).try_read_lock();
    if (UNODB_DETAIL_UNLIKELY(node_critical_section.must_restart()))
      return false;  // LCOV_EXCL_LINE
    if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.try_read_unlock()))
      return false;  // LCOV_EXCL_LINE
    const auto node_type = node.type();
    if (node_type == node_type::LEAF) {
      if (UNODB_DETAIL_UNLIKELY(!try_push_leaf(node, node_critical_section)))
        return false;  // LCOV_EXCL_LINE
      return UNODB_DETAIL_LIKELY(node_critical_section.try_read_unlock());
    }
    // recursive descent.
    auto* const inode{node.ptr<inode_type*>()};
    const auto t =
        inode->last(node_type);  // last child of current internal node
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check()))
      return false;  // LCOV_EXCL_LINE
    if (UNODB_DETAIL_UNLIKELY(!try_push(t, node_critical_section)))
      return false;                                     // LCOV_EXCL_LINE
    node = inode->get_child(node_type, t.child_index);  // get child
    if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check()))  // before using
      return false;  // LCOV_EXCL_LINE
    // Move RCS (will check invariant at top of loop)
    parent_critical_section = std::move(node_critical_section);
  }
  UNODB_DETAIL_CANNOT_HAPPEN();
}

UNODB_DETAIL_DISABLE_GCC_WARNING("-Wsuggest-attribute=pure")
template <typename Key, typename Value>
key_view olc_db<Key, Value>::iterator::get_key() noexcept {
  UNODB_DETAIL_ASSERT(valid());  // by contract
  // Note: If the iterator is on a leaf, we return the key for that
  // leaf regardless of whether the leaf has been deleted.  This is
  // part of the design semantics for the OLC ART scan.
  //
  // TODO(thompsonbry) : variable length keys. The simplest case
  // where this does not work today is a single root leaf.  In that
  // case, there is no inode path and we can not properly track the
  // key in the key_buffer.
  //
  // return keybuf_.get_key_view();
  const auto& e = stack_.top();
  const auto& node = e.node;
  UNODB_DETAIL_ASSERT(node.type() == node_type::LEAF);      // On a leaf.
  const auto* const leaf{node.template ptr<leaf_type*>()};  // current leaf.
  return leaf->get_key_view();
}
UNODB_DETAIL_RESTORE_GCC_WARNINGS()

template <typename Key, typename Value>
qsbr_value_view olc_db<Key, Value>::iterator::get_val() const noexcept {
  // Note: If the iterator is on a leaf, we return the value for
  // that leaf regardless of whether the leaf has been deleted.
  // This is part of the design semantics for the OLC ART scan.
  UNODB_DETAIL_ASSERT(valid());  // by contract
  const auto& e = stack_.top();
  const auto& node = e.node;
  UNODB_DETAIL_ASSERT(node.type() == node_type::LEAF);      // On a leaf.
  const auto* const leaf{node.template ptr<leaf_type*>()};  // current leaf.
  return qsbr_ptr_span{leaf->get_value_view()};
}

template <typename Key, typename Value>
int olc_db<Key, Value>::iterator::cmp(const art_key_type& akey) const noexcept {
  // TODO(thompsonbry) : variable length keys. Explore a cheaper way
  // to handle the exclusive bound case when developing variable
  // length key support based on the maintained key buffer.
  UNODB_DETAIL_ASSERT(!stack_.empty());
  auto& node = stack_.top().node;
  UNODB_DETAIL_ASSERT(node.type() == node_type::LEAF);
  const auto* const leaf{node.template ptr<leaf_type*>()};
  return unodb::detail::compare(leaf->get_key_view(), akey.get_key_view());
}

///
/// OLC scan implementation
///

#ifdef UNODB_DETAIL_WITH_STATS

UNODB_DETAIL_DISABLE_GCC_WARNING("-Wsuggest-attribute=cold")

template <typename Key, typename Value>
void olc_db<Key, Value>::increase_memory_use(std::size_t delta) noexcept {
  UNODB_DETAIL_ASSERT(delta > 0);

  current_memory_use.fetch_add(delta, std::memory_order_relaxed);
}

UNODB_DETAIL_RESTORE_GCC_WARNINGS()

template <typename Key, typename Value>
void olc_db<Key, Value>::decrease_memory_use(std::size_t delta) noexcept {
  UNODB_DETAIL_ASSERT(delta > 0);
  UNODB_DETAIL_ASSERT(delta <=
                      current_memory_use.load(std::memory_order_relaxed));

  current_memory_use.fetch_sub(delta, std::memory_order_relaxed);
}

template <typename Key, typename Value>
template <class INode>
constexpr void olc_db<Key, Value>::increment_inode_count() noexcept {
  static_assert(detail::olc_inode_defs<Key, Value>::template is_inode<INode>());

  node_counts[as_i<INode::type>].fetch_add(1, std::memory_order_relaxed);
  increase_memory_use(sizeof(INode));
}

template <typename Key, typename Value>
template <class INode>
constexpr void olc_db<Key, Value>::decrement_inode_count() noexcept {
  static_assert(detail::olc_inode_defs<Key, Value>::template is_inode<INode>());

  const auto old_inode_count UNODB_DETAIL_USED_IN_DEBUG =
      node_counts[as_i<INode::type>].fetch_sub(1, std::memory_order_relaxed);
  UNODB_DETAIL_ASSERT(old_inode_count > 0);

  decrease_memory_use(sizeof(INode));
}

template <typename Key, typename Value>
template <node_type NodeType>
constexpr void olc_db<Key, Value>::account_growing_inode() noexcept {
  static_assert(NodeType != node_type::LEAF);

  // NOLINTNEXTLINE(google-readability-casting)
  growing_inode_counts[internal_as_i<NodeType>].fetch_add(
      1, std::memory_order_relaxed);
}

template <typename Key, typename Value>
template <node_type NodeType>
constexpr void olc_db<Key, Value>::account_shrinking_inode() noexcept {
  static_assert(NodeType != node_type::LEAF);

  shrinking_inode_counts[internal_as_i<NodeType>].fetch_add(
      1, std::memory_order_relaxed);
}

#endif  // UNODB_DETAIL_WITH_STATS

template <typename Key, typename Value>
void olc_db<Key, Value>::dump(std::ostream& os) const {
#ifdef UNODB_DETAIL_WITH_STATS
  os << "olc_db dump, current memory use = " << get_current_memory_use()
     << '\n';
#else
  os << "olc_db dump\n";
#endif  // UNODB_DETAIL_WITH_STATS
  art_policy::dump_node(os, root.load());
}

// LCOV_EXCL_START
template <typename Key, typename Value>
void olc_db<Key, Value>::dump() const {
  dump(std::cerr);
}
// LCOV_EXCL_STOP

template <typename Key, typename Value>
UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
bool olc_db<Key, Value>::try_collapse_i4(
    detail::olc_node_ptr i4_node, std::uint8_t del_ci,
    in_critical_section<detail::olc_node_ptr>* slot,
    optimistic_lock::write_guard& guard) {
  auto* const i4{i4_node.template ptr<inode_4*>()};
  i4->remove_child_entry(del_ci);
  const auto rem_iter = i4->begin();
  const auto rem = i4->get_child(0);
  if (rem.type() != node_type::LEAF) {
    auto* const ri{rem.template ptr<inode_type*>()};
    if (ri->get_key_prefix().length() + i4->get_key_prefix().length() + 1 >
        detail::key_prefix_capacity) {
      return false;  // prefix overflow — leave as single-child I4
    }
    ri->get_key_prefix().prepend(i4->get_key_prefix(), rem_iter.key_byte);
  }
  if (slot != nullptr)
    *slot = rem;
  else
    root = rem;
  guard.unlock_and_obsolete();
  {
    const auto r{
        art_policy::template make_db_inode_reclaimable_ptr<inode_4>(i4, *this)};
  }
#ifdef UNODB_DETAIL_WITH_STATS
  account_shrinking_inode<node_type::I4>();
#endif
  return true;
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

template <typename Key, typename Value>
typename olc_db<Key, Value>::try_update_result_type
olc_db<Key, Value>::try_chain_cut(
    detail::olc_node_ptr chain_bottom, detail::olc_node_ptr leaf_ptr,
    optimistic_lock::write_guard parent_guard,
    optimistic_lock::write_guard chain_bottom_guard,
    optimistic_lock::write_guard leaf_guard, remove_stack_type& stk,
    std::size_t stk_n) {
  auto* const leaf{leaf_ptr.template ptr<detail::olc_leaf_type<Key, Value>*>()};
  // Atomic Chain Cut — remove leaf under a single-child chain I4.
  // Parameters are self-documenting; chain_bottom is NOT on the stack.
  //
  // Stack entry semantics (same as iterator):
  //   stk[i].child_i  = child index within stk[i] for descent
  //   stk[i].key_byte = dispatch byte used FROM stk[i] to stk[i+1]
  // ============================================================

  // --- Identify chain_top ---
  std::size_t chain_top = stk_n;  // = no chain on stack
  if (stk_n > 0) {
    const auto& top_e = stk[stk_n - 1];
    if (top_e.node.type() == node_type::I4 &&
        top_e.node.template ptr<inode_type*>()->get_children_count() == 1) {
      chain_top = stk_n - 1;
      while (chain_top > 0) {
        const auto& e = stk[chain_top - 1];
        if (e.node.type() != node_type::I4) break;
        if (e.node.template ptr<inode_type*>()->get_children_count() != 1)
          break;
        --chain_top;
      }
    }
  }

  // --- Step 2: Lock the chain bottom-up ---
  boost::container::small_vector<optimistic_lock::write_guard, 16> chain_guards;
  std::size_t cut_level = stk_n;

  if (stk_n > 0 && chain_top < stk_n) {
    chain_guards.push_back(std::move(parent_guard));
    cut_level = stk_n - 1;
    detail::sync(detail::sync_between_chain_locks);

    for (std::size_t i = stk_n - 2; i + 1 > chain_top && i < stk_n; --i) {
      const auto& entry = stk[i];
      auto rcs = node_ptr_lock(entry.node).rehydrate_read_lock(entry.version);
      if (UNODB_DETAIL_UNLIKELY(!rcs.check())) {
        rcs = node_ptr_lock(entry.node).try_read_lock();
        if (UNODB_DETAIL_UNLIKELY(rcs.must_restart())) break;
        const auto etype = entry.node.type();
        const auto ecount =
            entry.node.template ptr<inode_type*>()->get_children_count();
        if (UNODB_DETAIL_UNLIKELY(!rcs.check())) break;
        if (!(etype == node_type::I4 && ecount == 1)) break;
      }
      optimistic_lock::write_guard wg{std::move(rcs)};
      if (UNODB_DETAIL_UNLIKELY(wg.must_restart())) break;
      chain_guards.push_back(std::move(wg));
      cut_level = i;
    }
  }

  detail::sync(detail::sync_after_chain_locked);

  // --- Step 3: Acquire the cut_point_parent ---
  optimistic_lock::write_guard cpp_guard;

  if (chain_top >= stk_n) {
    cpp_guard = std::move(parent_guard);
  } else {
    for (;;) {
      if (cut_level == 0) {
        auto rcs = root_pointer_lock.try_read_lock();
        if (UNODB_DETAIL_UNLIKELY(rcs.must_restart())) return {};
        cpp_guard = optimistic_lock::write_guard{std::move(rcs)};
        if (UNODB_DETAIL_UNLIKELY(cpp_guard.must_restart())) return {};
        break;
      }
      const auto& pe = stk[cut_level - 1];
      auto rcs = node_ptr_lock(pe.node).rehydrate_read_lock(pe.version);
      if (!rcs.check()) {
        rcs = node_ptr_lock(pe.node).try_read_lock();
        if (UNODB_DETAIL_UNLIKELY(rcs.must_restart())) return {};
      }
      auto* const pi{pe.node.template ptr<inode_type*>()};
      const auto ptype = pe.node.type();
      const auto pcount = pi->get_children_count();
      const auto [pci, pcp]{pi->find_child(ptype, stk[cut_level - 1].key_byte)};
      if (UNODB_DETAIL_UNLIKELY(!rcs.check())) continue;
      if (pcp == nullptr || pcp->load() != stk[cut_level].node)
        return {};  // Child gone, restart.

      // LCOV_EXCL_START — Case B: a chain_bottom that was multi-child during
      // the downward pass became single-child due to a concurrent
      // remove.  Extend the chain by locking it.
      if (ptype == node_type::I4 && pcount == 1) {
        optimistic_lock::write_guard wg{std::move(rcs)};
        if (UNODB_DETAIL_UNLIKELY(wg.must_restart())) continue;
        chain_guards.push_back(std::move(wg));
        --cut_level;
        continue;
      }
      // LCOV_EXCL_STOP
      cpp_guard = optimistic_lock::write_guard{std::move(rcs)};
      if (UNODB_DETAIL_UNLIKELY(cpp_guard.must_restart())) continue;
      break;
    }
  }

  // --- Step 4.1: Pre-acquire grandparent if shrinkage needed ---
  const bool cpp_is_root =
      (cut_level == 0) || (chain_top >= stk_n && stk_n == 0);
  const std::size_t cpp_idx = cpp_is_root ? 0 : (cut_level - 1);

  bool needs_shrink = false;
  optimistic_lock::write_guard gp_guard;

  if (!cpp_is_root) {
    const auto& cpe = stk[cpp_idx];
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26462)
    auto* const cpi{cpe.node.template ptr<inode_type*>()};
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
    const auto cpt = cpe.node.type();
    const auto cpc = cpi->get_children_count();

    needs_shrink = (cpt == node_type::I4 && cpc <= 2) ||
                   (cpt == node_type::I16 &&
                    cpc == detail::basic_inode_16<art_policy>::min_size) ||
                   (cpt == node_type::I48 &&
                    cpc == detail::basic_inode_48<art_policy>::min_size) ||
                   (cpt == node_type::I256 &&
                    cpc == detail::basic_inode_256<art_policy>::min_size);

    if (needs_shrink) {
      if (cpp_idx >= 1) {
        const auto& gpe = stk[cpp_idx - 1];
        auto gp_rcs = node_ptr_lock(gpe.node).rehydrate_read_lock(gpe.version);
        if (UNODB_DETAIL_UNLIKELY(!gp_rcs.check())) return {};
        gp_guard = optimistic_lock::write_guard{std::move(gp_rcs)};
        if (UNODB_DETAIL_UNLIKELY(gp_guard.must_restart())) return {};
      } else {
        auto rcs = root_pointer_lock.try_read_lock();
        if (UNODB_DETAIL_UNLIKELY(rcs.must_restart())) return {};
        gp_guard = optimistic_lock::write_guard{std::move(rcs)};
        if (UNODB_DETAIL_UNLIKELY(gp_guard.must_restart())) return {};
      }
    }
  }

  // --- Step 4.2: Point of no return — cut and shrink ---
  if (cpp_is_root) {
    root = detail::olc_node_ptr{nullptr};
  } else {
    const auto& cpe = stk[cpp_idx];
    auto* const cpi{cpe.node.template ptr<inode_type*>()};
    const auto cpt = cpe.node.type();

    const std::uint8_t del_ci = (chain_top >= stk_n)
                                    ? stk[cpp_idx].child_i
                                    : stk[cut_level - 1].child_i;

    if (!needs_shrink) {
      cpi->remove_child_entry(cpt, del_ci);
    } else if (cpt == node_type::I4) {
      // I4 collapse: remove entry, promote remaining child if
      // prefix merge fits.
      auto* slot = (cpp_idx >= 1) ? cpe.slot : nullptr;
      std::ignore = try_collapse_i4(cpe.node, del_ci, slot, cpp_guard);
    } else {
      // I16/I48/I256 shrink via shrink constructors.
      UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
      detail::olc_node_ptr replacement{};
      if (cpt == node_type::I16) {
        auto* const src = cpe.node.template ptr<inode_16*>();
        auto s{inode_4::create(*this, *src)};
        s->init(*this, *src, cpp_guard, del_ci, chain_bottom_guard);
        replacement = detail::olc_node_ptr{s.release(), node_type::I4};
      } else if (cpt == node_type::I48) {
        auto* const src = cpe.node.template ptr<inode_48*>();
        auto s{inode_16::create(*this, *src)};
        s->init(*this, *src, cpp_guard, del_ci, chain_bottom_guard);
        replacement = detail::olc_node_ptr{s.release(), node_type::I16};
      } else {
        UNODB_DETAIL_ASSERT(cpt == node_type::I256);
        auto* const src = cpe.node.template ptr<inode_256*>();
        auto s{inode_48::create(*this, *src)};
        s->init(*this, *src, cpp_guard, del_ci, chain_bottom_guard);
        replacement = detail::olc_node_ptr{s.release(), node_type::I48};
      }
      UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
      if (cpp_idx >= 1)
        *cpe.slot = replacement;
      else
        root = replacement;
#ifdef UNODB_DETAIL_WITH_STATS
      if (cpt == node_type::I16)
        account_shrinking_inode<node_type::I16>();
      else if (cpt == node_type::I48)
        account_shrinking_inode<node_type::I48>();
      else
        account_shrinking_inode<node_type::I256>();
#endif
    }
  }

  // --- Step 4.4: Reclaim leaf and chain nodes ---
  leaf_guard.unlock_and_obsolete();
  {
    const auto r{art_policy::reclaim_leaf_on_scope_exit(leaf, *this)};
  }

  // chain_bottom_guard may have been consumed by init() in the shrink path.
  // must_restart() returns true when the guard is inactive (no lock).
  if (!chain_bottom_guard.must_restart()) {
    chain_bottom_guard.unlock_and_obsolete();
  }
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26815) {
    const auto r{art_policy::template make_db_inode_reclaimable_ptr<inode_4>(
        chain_bottom.template ptr<inode_4*>(), *this)};
  }
#ifdef UNODB_DETAIL_WITH_STATS
  account_shrinking_inode<node_type::I4>();
#endif

  for (std::size_t gi = 0; gi < chain_guards.size(); ++gi) {
    const std::size_t si = (stk_n - 1) - gi;
    chain_guards[gi].unlock_and_obsolete();
    {
      const auto r{art_policy::template make_db_inode_reclaimable_ptr<inode_4>(
          stk[si].node.template ptr<inode_4*>(), *this)};
    }
#ifdef UNODB_DETAIL_WITH_STATS
    account_shrinking_inode<node_type::I4>();
#endif
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  return true;
}
}  // namespace unodb

#endif  // UNODB_DETAIL_OLC_ART_HPP
