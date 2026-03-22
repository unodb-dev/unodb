// Copyright 2019-2026 UnoDB contributors

/// \file
/// Non-thread-safe Adaptive Radix Tree (ART) implementation.
///
/// Provides unodb::db, a single-threaded ART index supporting get, insert,
/// remove, and scan operations. For thread-safe alternatives, see
/// unodb::olc_db.

#ifndef UNODB_DETAIL_ART_HPP
#define UNODB_DETAIL_ART_HPP

// Should be the first include
#include "global.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stack>
#include <type_traits>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "art_common.hpp"
#include "art_internal.hpp"
#include "art_internal_impl.hpp"
#include "assert.hpp"
#include "in_fake_critical_section.hpp"
#include "node_type.hpp"

namespace unodb {

namespace detail {

template <typename Key, typename Value>
class inode;

template <typename Key, typename Value>
class inode_4;

template <typename Key, typename Value>
class inode_16;

template <typename Key, typename Value>
class inode_48;

template <typename Key, typename Value>
class inode_256;

/// Node header type for non-thread-safe ART implementation.
///
/// Has no extra fields. Used as the header type parameter for basic_node_ptr
/// and basic_leaf.
struct [[nodiscard]] node_header {};

static_assert(std::is_empty_v<node_header>);

/// Node pointer type for non-thread-safe ART.
using node_ptr = basic_node_ptr<node_header>;

struct impl_helpers;

/// Type definitions bundle for all internal node types.
///
/// Packages inode, inode_4, inode_16, inode_48, and inode_256 for use by the
/// ART policy configuration.
template <typename Key, typename Value>
using inode_defs = basic_inode_def<inode<Key, Value>, inode_4<Key, Value>,
                                   inode_16<Key, Value>, inode_48<Key, Value>,
                                   inode_256<Key, Value>>;

/// Custom deleter for internal nodes in non-thread-safe ART.
///
/// Manages cleanup of internal nodes when they are removed from the tree.
template <typename Key, typename Value, class INode>
using db_inode_deleter = basic_db_inode_deleter<INode, unodb::db<Key, Value>>;

/// Policy configuration for non-thread-safe ART implementation.
///
/// Bundles type definitions and synchronization primitives (fake locks for
/// single-threaded access) used throughout the ART implementation.
template <typename Key, typename Value>
using art_policy =
    basic_art_policy<Key, Value, unodb::db, unodb::in_fake_critical_section,
                     unodb::fake_lock, unodb::fake_read_critical_section,
                     node_ptr, inode_defs, db_inode_deleter,
                     basic_db_leaf_deleter>;

/// Base class for all internal nodes in non-thread-safe ART.
///
/// Provides common functionality for inode_4, inode_16, inode_48, and
/// inode_256.
template <typename Key, typename Value>
using inode_base = basic_inode_impl<art_policy<Key, Value>>;

/// Leaf node type for non-thread-safe ART.
///
/// Stores a key-value pair (or value only for keyless leaves).
template <typename Key, typename Value>
using leaf_type = basic_leaf<leaf_key_type<Key, Value>, node_header>;

/// Internal node base class for non-thread-safe ART.
///
/// Serves as a common base for all internal node types, allowing polymorphic
/// access to inode_4, inode_16, inode_48, and inode_256.
template <typename Key, typename Value>
class inode : public inode_base<Key, Value> {};

}  // namespace detail

template <typename Key, typename Value>
class mutex_db;

/// A non-thread-safe implementation of the Adaptive Radix Tree (ART).
///
/// \sa unodb::olc_db for a highly concurrent thread-safe ART implementation.
template <typename Key, typename Value>
class db final {
  /// Allow mutex_db to access private members for thread-safe wrapper.
  friend class mutex_db<Key, Value>;

 public:
  /// The type of the keys in the index.
  using key_type = Key;

  /// The type of the value associated with the keys in the index.
  using value_type = Value;

  /// View type for values stored in the index.
  using value_view = unodb::value_view;

  /// Result type for get operations.
  ///
  /// Contains value_view if key was found, otherwise empty.
  using get_result = std::optional<value_type>;

  /// Base class type for internal nodes.
  using inode_base = detail::inode_base<Key, Value>;

 private:
  /// Internal encoded key type used for tree operations.
  using art_key_type = detail::basic_art_key<Key>;

  /// Leaf node type (keyless when can_eliminate_key_in_leaf).
  using leaf_type = detail::leaf_type<Key, Value>;

  /// Database type (self-reference for template instantiation).
  using db_type = db<Key, Value>;

  /// Query for a value associated with an encoded \a search_key.
  [[nodiscard, gnu::pure]] get_result get_internal(
      art_key_type search_key) const noexcept;

  /// Insert a value under an encoded key iff there is no entry for that key.
  ///
  /// \param insert_key Encoded key to insert
  /// \param v Value to associate with the key
  ///
  /// \note Cannot be called during stack unwinding with
  /// `std::uncaught_exceptions() > 0`.
  ///
  /// \return true iff the key value pair was inserted.
  [[nodiscard]] bool insert_internal(art_key_type insert_key, value_type v);

  /// Insert for fixed-width keys (no chain nodes needed).
  [[nodiscard]] bool insert_internal_fixed(art_key_type insert_key,
                                           value_type v);

  /// Insert for variable-length keys (may create chain I4 nodes).
  [[nodiscard]] bool insert_internal_key_view(art_key_type insert_key,
                                              value_type v);

  /// Build an inode chain for the first key_view insert into an empty tree.
  ///
  /// For key_view keys, the tree must always have at least one inode above
  /// every leaf so that the iterator's key buffer (keybuf_) is populated
  /// during traversal.  This method wraps \a child in a chain of single-child
  /// I4 nodes, each consuming up to key_prefix_capacity prefix bytes + 1
  /// dispatch byte from the key.  Built bottom-up: the last I4 created
  /// (with prefix from the start of the key) becomes the root.
  ///
  /// \param k The full encoded key
  /// \param child The node to place at the bottom of the chain
  /// \param start_depth Depth at which the chain starts (default 0)
  /// \return The chain top node, or \a child if no bytes to encode
  [[nodiscard]] detail::node_ptr build_chain(
      art_key_type k, detail::node_ptr child,
      detail::tree_depth<art_key_type> start_depth);

  /// Remove the entry associated with the encoded key \a remove_key.
  ///
  /// \return true if the delete was successful (i.e. the key was found in the
  /// tree and the associated index entry was removed).
  [[nodiscard]] bool remove_internal(art_key_type remove_key);

  /// Two-pass removal with chain cleanup for variable-length keys.
  [[nodiscard]] bool remove_internal_key_view(art_key_type remove_key);

  /// Single-pass removal for fixed-width keys (no chain nodes).
  [[nodiscard]] bool remove_internal_fixed(art_key_type remove_key);

 public:
  // Creation and destruction

  /// Construct empty ART index.
  db() noexcept = default;

  /// Destroy ART index, freeing all tree nodes.
  ~db() noexcept;

  // TODO(laurynas): implement copy and move operations

  /// Copy constructor (deleted).
  db(const db&) = delete;

  /// Move constructor (deleted).
  db(db&&) = delete;

  /// Copy assignment operator (deleted).
  db& operator=(const db&) = delete;

  /// Move assignment operator (deleted).
  db& operator=(db&&) = delete;

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

  /// Return true iff the index is empty.
  [[nodiscard, gnu::pure]] bool empty() const noexcept {
    return root == nullptr;
  }

  /// Insert a value under a key iff there is no entry for that key.
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
  /// \note Cannot be called during stack unwinding with
  /// `std::uncaught_exceptions() > 0`.
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
  ///
  /// \return true if the delete was successful (i.e. the key was found in the
  /// tree and the associated index entry was removed).
  [[nodiscard]] bool remove(Key search_key) {
    const art_key_type k{search_key};
    return remove_internal(k);
  }

  /// Removes all entries in the index.
  ///
  /// After this operation, empty() returns true and all memory used by tree
  /// nodes is freed. Node growth/shrink counters are preserved.
  void clear() noexcept;

  /// Internal iterator for tree traversal.
  ///
  /// Iterator is an internal API. Use scan() for the public API.
  class iterator {
    // Note: The iterator is backed by a std::stack. This means that
    // the iterator methods accessing the stack can not be declared as
    // [[noexcept]].

    /// unodb::db
    friend class db;

    /// Allow visitor to access iterator for tree traversal operations.
    template <class>
    friend class visitor;

    /// Alias used for the elements of the stack.
    using stack_entry = typename inode_base::iter_result;

   protected:
    /// Construct an empty iterator (one that is logically not
    /// positioned on anything and which will report !valid()).
    explicit iterator(db& tree UNODB_DETAIL_LIFETIMEBOUND) noexcept
        : db_(tree) {}

    // iterator is not flyweight. disallow copy and move.

    /// Copy constructor (deleted).
    iterator(const iterator&) = delete;

    /// Move constructor (deleted).
    iterator(iterator&&) = delete;

    /// Copy assignment operator (deleted).
    iterator& operator=(const iterator&) = delete;
    // iterator& operator=(iterator&&) = delete; // test_only_iterator()

   public:
    /// Key type of the index entries.
    using key_type = Key;

    /// Value type of the index entries.
    using value_type = Value;

    // EXPOSED TO THE TESTS

    /// Position the iterator on the first entry in the index.
    ///
    /// Traverse to the left-most leaf. The stack is cleared first and then
    /// re-populated as we step down along the path to the left-most leaf.
    ///
    /// \return Reference to this iterator
    iterator& first();

    /// Advance the iterator to next entry in the index.
    iterator& next();

    /// Position the iterator on the last entry in the index, which
    /// can be used to initiate a reverse traversal.
    ///
    /// Traverse to the right-most leaf. The stack is cleared first and then
    /// re-populated as we step down along the path to the right-most leaf.
    ///
    /// \return Reference to this iterator
    iterator& last();

    /// Position the iterator on the previous entry in the index.
    iterator& prior();

    /// Position the iterator on, before, or after the caller's key. If the
    /// iterator can not be positioned, it will be invalidated. For example, if
    /// \a fwd is true and the \a search_key is greater than any key in the
    /// index then the iterator will be invalidated since there is no index
    /// entry greater than the search key. Likewise, if \a fwd is false and the
    /// \a search_key is less than any key in the index, then the iterator will
    /// be invalidated since there is no index entry less than the \a
    /// search_key.
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

    /// Return type for get_key(): key_view when the leaf stores the key,
    /// transient_key_view when the key is reconstructed from the inode path.
    using get_key_result = std::conditional_t<
        detail::art_policy<Key, Value>::full_key_in_inode_path,
        transient_key_view, key_view>;

    /// Return the key associated with the current position of the iterator.
    ///
    /// For full_key_in_inode_path trees, returns a transient_key_view that is
    /// valid only until the next iterator movement.  For other trees,
    /// returns a key_view into the leaf (stable for the leaf's lifetime).
    ///
    /// \pre The iterator MUST be valid().
    [[nodiscard]] get_key_result get_key() noexcept;

    /// Return the value_view associated with the current position of
    /// the iterator.
    ///
    /// \pre The iterator MUST be valid().
    [[nodiscard, gnu::pure]] value_type get_val() const noexcept;

    /// Output iterator state to stream \a os for debugging.
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
      // Create a new stack and copy everything there. Using the new stack,
      // print out the stack in top-bottom order. This avoids modifications to
      // the existing stack for the iterator.
      auto tmp = stack_;
      auto level = tmp.size() - 1;
      while (!tmp.empty()) {
        const auto& e = tmp.top();
        const auto& np = e.node;
        os << "iter::stack:: level = " << level << ", key_byte=0x" << std::hex
           << std::setfill('0') << std::setw(2)
           << static_cast<std::uint64_t>(e.key_byte) << std::dec
           << ", child_index=0x" << std::hex << std::setfill('0')
           << std::setw(2) << static_cast<std::uint64_t>(e.child_index)
           << std::dec << ", prefix(" << e.prefix.length() << ")=";
        detail::dump_key(os, e.prefix.get_key_view());
        os << ", ";
        art_policy::dump_node(os, np, false /*recursive*/);
        if (np.type() != node_type::LEAF) os << '\n';
        tmp.pop();
        level--;
      }
    }
    // LCOV_EXCL_STOP

    /// Output iterator state to stderr for debugging.
    ///
    /// Convenience wrapper for `dump(std::cerr)`.
    /// \note For debugging purposes only, not part of stable API
    // LCOV_EXCL_START
    [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump() const { dump(std::cerr); }
    // LCOV_EXCL_STOP

    /// Return true unless the stack is empty (exposed to tests).
    [[nodiscard]] bool valid() const noexcept { return !stack_.empty(); }

#ifdef UNODB_DETAIL_WITH_STATS
    /// Return stack entries bottom-to-top (test only).
    [[nodiscard]] std::vector<stack_entry> test_only_stack() const {
      auto tmp = stack_;
      std::vector<stack_entry> result;
      while (!tmp.empty()) {
        result.push_back(tmp.top());
        tmp.pop();
      }
      std::reverse(result.begin(), result.end());
      return result;
    }
#endif  // UNODB_DETAIL_WITH_STATS

   protected:
    /// Descend to left-most leaf from given \a node.
    ///
    /// Pushes visited nodes onto the stack during descent, updating the key
    /// buffer to track the path taken. Used by first(), seek() with forward
    /// traversal, and other iterator positioning operations.
    ///
    /// \return Reference to this iterator
    iterator& left_most_traversal(detail::node_ptr node);

    /// Descend to right-most leaf from given \a node.
    ///
    /// Pushes visited nodes onto the stack during descent, updating the key
    /// buffer to track the path taken. Used by last(), seek() with reverse
    /// traversal, and other iterator positioning operations.
    ///
    /// \return Reference to this iterator
    iterator& right_most_traversal(detail::node_ptr node);

    /// Descend left if child is an inode, or push as leaf if value-in-slot.
    iterator& descend_left([[maybe_unused]] const auto* inode,
                           [[maybe_unused]] node_type ntype,
                           [[maybe_unused]] std::uint8_t child_i,
                           detail::node_ptr child) {
      if constexpr (art_policy::can_eliminate_leaf) {
        if (inode->is_value_in_slot(ntype, child_i)) {
          push_leaf(child);
          return *this;
        }
      }
      return left_most_traversal(child);
    }

    /// Descend right if child is an inode, or push as leaf if value-in-slot.
    iterator& descend_right([[maybe_unused]] const auto* inode,
                            [[maybe_unused]] node_type ntype,
                            [[maybe_unused]] std::uint8_t child_i,
                            detail::node_ptr child) {
      if constexpr (art_policy::can_eliminate_leaf) {
        if (inode->is_value_in_slot(ntype, child_i)) {
          push_leaf(child);
          return *this;
        }
      }
      return right_most_traversal(child);
    }

    /// Compare the given key \a akey to the current key in the internal buffer.
    ///
    /// \return -1, 0, or 1 if this key is LT, EQ, or GT the other
    /// key.
    [[nodiscard]] int cmp(art_key_type akey) const noexcept {
      UNODB_DETAIL_ASSERT(!stack_.empty());
      if constexpr (art_policy::full_key_in_inode_path) {
        return unodb::detail::compare(keybuf_.get_key_view(),
                                      akey.get_key_view());
      } else {
        auto& node = stack_.top().node;
        UNODB_DETAIL_ASSERT(node.type() == node_type::LEAF);
        const auto* const leaf{node.template ptr<leaf_type*>()};
        return unodb::detail::compare(leaf->get_key_view(),
                                      akey.get_key_view());
      }
    }

    /// \name Stack access methods
    /// \{

    /// Return true iff the iterator stack is empty.
    [[nodiscard]] bool empty() const noexcept { return stack_.empty(); }

    /// Push internal node entry onto iterator stack.
    ///
    /// Updates both the stack and the key buffer to reflect descent through
    /// the given node.
    ///
    /// \param node Internal node pointer (must not be a leaf)
    /// \param key_byte Byte value along which descent occurs
    /// \param child_index Index of child in node's child array
    /// \param prefix Snapshot of node's key prefix
    void push(detail::node_ptr node, std::byte key_byte,
              std::uint8_t child_index, detail::key_prefix_snapshot prefix) {
      // For variable length keys we need to know the number of bytes associated
      // with the node's key_prefix. In addition there is one byte for the
      // descent to the child node along the child_index. That information needs
      // to be stored on the stack so we can pop off the right number of bytes
      // even for OLC where the node might be concurrently modified.
      UNODB_DETAIL_ASSERT(node.type() != node_type::LEAF);
      stack_.push({node, key_byte, child_index, prefix});
      keybuf_.push(prefix.get_key_view());
      keybuf_.push(key_byte);
    }

    /// Push leaf entry \a aleaf onto iterator stack.
    ///
    /// \param aleaf Leaf node pointer
    void push_leaf(detail::node_ptr aleaf) {
      stack_.push({
          aleaf,
          static_cast<std::byte>(0xFFU),     // ignored for leaf
          static_cast<std::uint8_t>(0xFFU),  // ignored for leaf
          detail::key_prefix_snapshot(0),    // ignored for leaf
          true                               // packed_leaf
      });
      // No change in the key_buffer.
    }

    /// Push an entry \a e onto the stack.
    void push(const typename inode_base::iter_result& e) {
      const auto node_type = e.node.type();
      if (UNODB_DETAIL_UNLIKELY(node_type == node_type::LEAF)) {
        push_leaf(e.node);
        return;
      }
      push(e.node, e.key_byte, e.child_index, e.prefix);
    }

    /// Pop entry from stack and truncate key buffer accordingly.
    void pop() noexcept {
      UNODB_DETAIL_ASSERT(!empty());

      const auto& e = top();
      const auto n = static_cast<std::size_t>(
          (e.node.type() != node_type::LEAF &&
           !(art_policy::can_eliminate_leaf && e.packed_leaf))
              ? e.prefix.length() + 1
              : 0);
      keybuf_.pop(n);
      stack_.pop();
    }

    /// Return entry at top of stack.
    ///
    /// \pre Stack must not be empty
    [[nodiscard]] const stack_entry& top() const noexcept {
      UNODB_DETAIL_ASSERT(!stack_.empty());
      return stack_.top();
    }

    /// Return node at top of stack, or nullptr if stack is empty.
    [[nodiscard]] detail::node_ptr current_node() const noexcept {
      return stack_.empty() ? detail::node_ptr(nullptr) : stack_.top().node;
    }

    /// \}

   private:
    /// Invalidate iterator by clearing the stack and key buffer.
    ///
    /// After this operation, valid() returns false.
    /// \return Reference to this iterator
    iterator& invalidate() noexcept {
      while (!stack_.empty()) stack_.pop();  // clear the stack
      keybuf_.reset();                       // clear the key buffer
      return *this;
    }

    /// The outer db instance.
    db& db_;

    /// A stack reflecting the parent path from the root of the tree to the
    /// current leaf. An empty stack corresponds to a logically empty iterator
    /// and the iterator will report ! valid(). The iterator for an empty tree
    /// is an empty stack.
    ///
    /// The stack is made up of `(node_ptr, key, child_index)` entries.
    ///
    /// The `node_ptr` is never `nullptr` and points to the internal node or
    /// leaf for that step in the path from the root to some leaf. For the
    /// bottom of the stack, `node_ptr` is the root. For the top of the stack,
    /// `node_ptr` is the current leaf. In the degenerate case where the tree is
    /// a single root leaf, then the stack contains just that leaf.
    ///
    /// The `key` is the `std::byte` along which the path descends from that
    /// `node_ptr`. The `key` has no meaning for a leaf. The key byte may be
    /// used to reconstruct the full key (along with any prefix bytes in the
    /// nodes along the path). The key byte is tracked to avoid having to search
    /// the keys of some node types (detail::inode_48) when the `child_index`
    /// does not directly imply the key byte.
    ///
    /// The `child_index` is the `std::uint8_t` index position in the parent at
    /// which the `child_ptr` was found. The `child_index` has no meaning for a
    /// leaf. In the special case of detail::inode_48, the `child_index` is the
    /// index into the `child_indexes[]`. For all other internal node types, the
    /// `child_index` is a direct index into the `children[]`. When finding the
    /// successor (or predecessor) the `child_index` needs to be interpreted
    /// according to the node type. For detail::inode_4 and detail::inode_16,
    /// you just look at the next slot in the `children[]` to find the
    /// successor. For detail::inode_256, you look at the next non-null slot in
    /// the `children[]`. detail::inode_48 is the oddest of the node types. For
    /// it, you have to look at the `child_indexes[]`, find the next mapped key
    /// value greater than the current one, and then look at its entry in the
    /// `children[]`.
    std::stack<stack_entry> stack_{};

    /// A buffer into which visited encoded (binary comparable) keys are
    /// materialized during the iterator traversal. Bytes are pushed onto this
    /// buffer when we push something onto the iterator stack and popped off of
    /// this buffer when we pop something off of the iterator stack.
    detail::key_buffer keybuf_{};
  };  // class iterator

  /// \name Public scan API
  /// \{

  // Note: The scan() interface is public. The iterator and the methods to
  // obtain an iterator are protected (except for tests). This encapsulation
  // makes it easier to define methods which operate on external keys (scan())
  // and those which operate on internal keys (seek() and the iterator). It also
  // makes life easier for mutex_db since scan() can take the lock.

  /// Scan the tree, applying the caller's lambda to each visited leaf.
  ///
  /// \param fn A function `f(unodb::visitor<unodb::db::iterator>&)` returning
  /// `bool`.  The traversal will halt if the function returns \c true.
  ///
  /// \param fwd When \c true perform a forward scan, otherwise perform a
  /// reverse scan.
  template <typename FN>
  void scan(FN fn, bool fwd = true);

  /// Scan in the indicated direction, applying the caller's lambda to each
  /// visited leaf.
  ///
  /// \param from_key is an inclusive lower bound for the starting point of the
  /// scan.
  ///
  /// \param fn A function `f(unodb::visitor<unodb::db::iterator>&)` returning
  /// `bool`.  The traversal will halt if the function returns \c true.
  ///
  /// \param fwd When \c true perform a forward scan, otherwise perform a
  /// reverse scan.
  template <typename FN>
  void scan_from(Key from_key, FN fn, bool fwd = true);

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
  /// \param fn A function `f(unodb::visitor<unodb::db::iterator>&)` returning
  /// `bool`.  The traversal will halt if the function returns \c true.
  template <typename FN>
  void scan_range(Key from_key, Key to_key, FN fn);

  /// \}

  // Used to write the iterator tests. Use only in tests.
  iterator test_only_iterator() noexcept { return iterator(*this); }

#ifdef UNODB_DETAIL_WITH_STATS

  /// \name Statistics
  /// \{

  /// Return current memory use by tree nodes in bytes.
  ///
  /// \note Only available when compiled with UNODB_DETAIL_WITH_STATS defined.
  [[nodiscard, gnu::pure]] constexpr std::size_t get_current_memory_use()
      const noexcept {
    return current_memory_use;
  }

  /// Return count of nodes of given type.
  ///
  /// \tparam NodeType Node type to count (node_type::LEAF, node_type::I4,
  /// node_type::I16, node_type::I48, or node_type::I256)
  /// \note Only available when compiled with UNODB_DETAIL_WITH_STATS defined.
  template <node_type NodeType>
  [[nodiscard, gnu::pure]] constexpr std::uint64_t get_node_count()
      const noexcept {
    return node_counts[as_i<NodeType>];
  }

  /// Return counts of all node types.
  ///
  /// \return Array indexed by node_type containing counts
  /// \note Only available when compiled with UNODB_DETAIL_WITH_STATS defined.
  [[nodiscard, gnu::pure]] constexpr node_type_counter_array get_node_counts()
      const noexcept {
    return node_counts;
  }

  /// Return count of growing operations for given inode type.
  ///
  /// Growing operations occur when an internal node reaches capacity and is
  /// replaced with a larger node type (e.g., detail::inode_4 to
  /// detail::inode_16).
  ///
  /// \tparam NodeType Internal node type (node_type::I4, node_type::I16,
  /// node_type::I48, or node_type::I256)
  /// \return Number of growth operations that created a node of this type
  /// \note Only available when compiled with UNODB_DETAIL_WITH_STATS defined.
  template <node_type NodeType>
  [[nodiscard, gnu::pure]] constexpr std::uint64_t get_growing_inode_count()
      const noexcept {
    return growing_inode_counts[internal_as_i<NodeType>];
  }

  /// Return counts of all growing inode operations.
  ///
  /// \return Array indexed by internal node type containing growth counts
  /// \note Only available when compiled with UNODB_DETAIL_WITH_STATS defined.
  [[nodiscard, gnu::pure]] constexpr inode_type_counter_array
  get_growing_inode_counts() const noexcept {
    return growing_inode_counts;
  }

  /// Return count of shrinking operations for given inode type.
  ///
  /// Shrinking operations occur when an internal node falls below minimum
  /// occupancy and is replaced with a smaller node type (e.g., detail::inode_16
  /// to detail::inode_4).
  ///
  /// \tparam NodeType Internal node type (node_type::I4, node_type::I16,
  /// node_type::I48, or node_type::I256)
  /// \return Number of times this node type was shrunk to a smaller type
  /// \note Only available when compiled with UNODB_DETAIL_WITH_STATS defined.
  template <node_type NodeType>
  [[nodiscard, gnu::pure]] constexpr std::uint64_t get_shrinking_inode_count()
      const noexcept {
    return shrinking_inode_counts[internal_as_i<NodeType>];
  }

  /// Return counts of all shrinking inode operations.
  ///
  /// \return Array indexed by internal node type containing shrink counts
  /// \note Only available when compiled with UNODB_DETAIL_WITH_STATS defined.
  [[nodiscard, gnu::pure]] constexpr inode_type_counter_array
  get_shrinking_inode_counts() const noexcept {
    return shrinking_inode_counts;
  }

  /// Return count of key prefix split operations.
  ///
  /// Key prefix splits occur when inserting a key that differs from an
  /// existing node's key prefix, requiring the node to be split into a
  /// new internal node with two children.
  ///
  /// \note Only available when compiled with UNODB_DETAIL_WITH_STATS defined.
  [[nodiscard, gnu::pure]] constexpr std::uint64_t get_key_prefix_splits()
      const noexcept {
    return key_prefix_splits;
  }

  /// \}

#endif  // UNODB_DETAIL_WITH_STATS

  /// Check if get operation found a key.
  ///
  /// \param result Result from get() operation
  /// \return true if the key was found, false otherwise
  [[nodiscard, gnu::const]] static constexpr bool key_found(
      const get_result& result) noexcept {
    return static_cast<bool>(result);
  }

  /// \name Debugging
  /// \{

  /// Output tree structure to stream for debugging.
  ///
  /// \param os Output stream to write tree representation
  /// \note For debugging purposes only, not part of stable API
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os) const;

  /// Output tree structure to stderr for debugging.
  ///
  /// Convenience wrapper for `dump(std::cerr)`.
  /// \note For debugging purposes only, not part of stable API
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump() const;

  /// \}

 private:
  /// Policy type for ART implementation.
  using art_policy = detail::art_policy<Key, Value>;

  /// Node header type.
  using header_type = typename art_policy::header_type;

  /// Internal node base type.
  using inode_type = detail::inode<Key, Value>;

  /// Node type with 4 children.
  using inode_4 = detail::inode_4<Key, Value>;

  /// Tree depth tracking type.
  using tree_depth_type = detail::tree_depth<art_key_type>;

  /// Visitor type for scan operations.
  using visitor_type = visitor<db_type::iterator>;

  /// Internal node definitions bundle.
  using inode_defs_type = detail::inode_defs<Key, Value>;

  /// Delete entire tree starting from root.
  void delete_root_subtree() noexcept;

#ifdef UNODB_DETAIL_WITH_STATS

  /// Increase tracked memory usage by \a delta bytes.
  constexpr void increase_memory_use(std::size_t delta) noexcept {
    UNODB_DETAIL_ASSERT(delta > 0);
    UNODB_DETAIL_ASSERT(
        std::numeric_limits<decltype(current_memory_use)>::max() - delta >=
        current_memory_use);

    current_memory_use += delta;
  }

  /// Decrease tracked memory usage by \a delta bytes.
  constexpr void decrease_memory_use(std::size_t delta) noexcept {
    UNODB_DETAIL_ASSERT(delta > 0);
    UNODB_DETAIL_ASSERT(delta <= current_memory_use);

    current_memory_use -= delta;
  }

  /// Increment leaf node count and bump memory usage by \a leaf_size bytes.
  constexpr void increment_leaf_count(std::size_t leaf_size) noexcept {
    increase_memory_use(leaf_size);
    ++node_counts[as_i<node_type::LEAF>];
  }

  /// Decrement leaf node count and decrease memory usage by \a leaf_size bytes.
  constexpr void decrement_leaf_count(std::size_t leaf_size) noexcept {
    decrease_memory_use(leaf_size);

    UNODB_DETAIL_ASSERT(node_counts[as_i<node_type::LEAF>] > 0);
    --node_counts[as_i<node_type::LEAF>];
  }

  /// Increment internal node count for given type.
  ///
  /// \tparam INode Internal node class
  template <class INode>
  constexpr void increment_inode_count() noexcept;

  /// Decrement internal node count for given type.
  ///
  /// \tparam INode Internal node class
  template <class INode>
  constexpr void decrement_inode_count() noexcept;

  /// Record node growth operation.
  ///
  /// \tparam NodeType Node type that was grown
  template <node_type NodeType>
  constexpr void account_growing_inode() noexcept;

  /// Record node shrink operation.
  ///
  /// \tparam NodeType Node type that was shrunk
  template <node_type NodeType>
  constexpr void account_shrinking_inode() noexcept;

#endif  // UNODB_DETAIL_WITH_STATS

  /// Root of the tree (nullptr if empty).
  detail::node_ptr root{nullptr};

#ifdef UNODB_DETAIL_WITH_STATS

  /// Current memory use by all tree nodes in bytes.
  std::size_t current_memory_use{0};

  /// Count of nodes by type.
  node_type_counter_array node_counts{};

  /// Count of node growth operations by internal node type.
  inode_type_counter_array growing_inode_counts{};

  /// Count of node shrink operations by internal node type.
  inode_type_counter_array shrinking_inode_counts{};

  /// Count of key prefix split operations.
  std::uint64_t key_prefix_splits{0};

#endif  // UNODB_DETAIL_WITH_STATS

  // Type names in the Doxygen comments below make them clickable in the output

  /// detail::make_db_leaf_ptr
  friend auto detail::make_db_leaf_ptr<Key, Value, db>(art_key_type, value_type,
                                                       db&);

  /// detail::basic_db_leaf_deleter
  template <class>
  friend class detail::basic_db_leaf_deleter;

  /// detail::basic_art_policy
  template <typename,                             // Key
            typename,                             // Value
            template <typename, typename> class,  // Db
            template <class> class,               // CriticalSectionPolicy
            class,                                // Fake lock implementation
            class,  // Fake read_critical_section implementation
            class,  // NodePtr
            template <typename, typename> class,         // INodeDefs
            template <typename, typename, class> class,  // INodeReclamator
            template <class> class>                      // LeafReclamator
  friend struct detail::basic_art_policy;

  /// detail::basic_db_inode_deleter
  template <typename, class>
  friend class detail::basic_db_inode_deleter;

  /// detail::impl_helpers
  friend struct detail::impl_helpers;
};

namespace detail {

/// Helper functions for node insertion and removal operations.
///
/// Provides static methods used by inode_4, inode_16, inode_48, and inode_256
/// to handle child addition and removal, including node growth and shrinkage.
struct impl_helpers {
  // GCC 10 diagnoses parameters that are present only in uninstantiated if
  // constexpr branch, such as node_in_parent for inode_256.
  UNODB_DETAIL_DISABLE_GCC_10_WARNING("-Wunused-parameter")

  /// Add child to internal node or choose subtree for insertion.
  ///
  /// Handles node growth when capacity is reached.
  ///
  /// \tparam Key Key type
  /// \tparam Value Value type
  /// \tparam INode Internal node type
  ///
  /// \param inode Internal node to modify
  /// \param key_byte Byte value for child lookup
  /// \param k Complete encoded key being inserted
  /// \param v Value to insert
  /// \param db_instance Database instance
  /// \param depth Current tree depth
  /// \param node_in_parent Pointer to this node in parent's children array
  ///
  /// \return Pointer to location for further descent or insertion
  template <typename Key, typename Value, class INode>
  [[nodiscard]] static detail::node_ptr* add_or_choose_subtree(
      INode& inode, std::byte key_byte, basic_art_key<Key> k, Value v,
      db<Key, Value>& db_instance, tree_depth<basic_art_key<Key>> depth,
      detail::node_ptr* node_in_parent);

  UNODB_DETAIL_RESTORE_GCC_10_WARNINGS()

  /// Remove child from internal node or choose subtree for removal.
  ///
  /// Handles node shrinkage when occupancy falls below minimum.
  ///
  /// \tparam Key Key type
  /// \tparam Value Value type
  /// \tparam INode Internal node type
  ///
  /// \param inode Internal node to modify
  /// \param key_byte Byte value for child lookup
  /// \param k Complete encoded key being removed
  /// \param db_instance Database instance
  /// \param node_in_parent Pointer to this node in parent's children array
  ///
  /// \return Optional pointer to location for further descent, or empty if key
  /// not found
  template <typename Key, typename Value, class INode>
  [[nodiscard]] static std::optional<detail::node_ptr*>
  remove_or_choose_subtree(INode& inode, std::byte key_byte,
                           basic_art_key<Key> k, db<Key, Value>& db_instance,
                           detail::node_ptr* node_in_parent);

  /// Deleted constructor (static-only helper class).
  impl_helpers() = delete;
};

/// Base class for inode_4
template <typename Key, typename Value>
using inode_4_parent = basic_inode_4<art_policy<Key, Value>>;

/// Internal node with 4 children for non-thread-safe ART.
///
/// Smallest internal node type, used when a leaf needs to be split or when
/// a larger node shrinks below minimum occupancy.
template <typename Key, typename Value>
class [[nodiscard]] inode_4 final : public inode_4_parent<Key, Value> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  using inode_4_parent<Key, Value>::inode_4_parent;

  /// Add child or choose subtree for insertion.
  ///
  /// Forwards to impl_helpers::add_or_choose_subtree.
  ///
  /// \param args Arguments forwarded to impl_helpers
  template <typename... Args>
  [[nodiscard]] auto add_or_choose_subtree(Args&&... args) {
    return impl_helpers::add_or_choose_subtree(*this,
                                               std::forward<Args>(args)...);
  }

  /// Remove child or choose subtree for removal.
  ///
  /// Forwards to impl_helpers::remove_or_choose_subtree.
  ///
  /// \param args Arguments forwarded to impl_helpers
  template <typename... Args>
  [[nodiscard]] auto remove_or_choose_subtree(Args&&... args) {
    return impl_helpers::remove_or_choose_subtree(*this,
                                                  std::forward<Args>(args)...);
  }
};

/// Test instantiation of inode_4 for size verification.
using inode_4_test_type = inode_4<std::uint64_t, unodb::value_view>;
#ifndef _MSC_VER
static_assert(sizeof(inode_4_test_type) == 48);
#else
// MSVC pads the first field to 8 byte boundary even though its natural
// alignment is 4 bytes, maybe due to parent class sizeof
static_assert(sizeof(inode_4_test_type) == 56);
#endif

/// Base class for inode_16
template <typename Key, typename Value>
using inode_16_parent = basic_inode_16<art_policy<Key, Value>>;

/// Internal node with 16 children for non-thread-safe ART.
///
/// Used when inode_4 grows beyond capacity or when inode_48 shrinks below
/// minimum occupancy.
template <typename Key, typename Value>
class [[nodiscard]] inode_16 final : public inode_16_parent<Key, Value> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  using inode_16_parent<Key, Value>::inode_16_parent;

  /// Add child or choose subtree for insertion.
  ///
  /// Forwards to impl_helpers::add_or_choose_subtree.
  ///
  /// \param args Arguments forwarded to impl_helpers
  template <typename... Args>
  [[nodiscard]] auto add_or_choose_subtree(Args&&... args) {
    return impl_helpers::add_or_choose_subtree(*this,
                                               std::forward<Args>(args)...);
  }

  /// Remove child or choose subtree for removal.
  ///
  /// Forwards to impl_helpers::remove_or_choose_subtree.
  ///
  /// \param args Arguments forwarded to impl_helpers
  template <typename... Args>
  [[nodiscard]] auto remove_or_choose_subtree(Args&&... args) {
    return impl_helpers::remove_or_choose_subtree(*this,
                                                  std::forward<Args>(args)...);
  }
};

static_assert(sizeof(inode_16<std::uint64_t, unodb ::value_view>) == 160);

/// Base class for inode_48
template <typename Key, typename Value>
using inode_48_parent = basic_inode_48<art_policy<Key, Value>>;

/// Internal node with 48 children for non-thread-safe ART.
///
/// Used when inode_16 grows beyond capacity or when inode_256 shrinks below
/// minimum occupancy.
template <typename Key, typename Value>
class [[nodiscard]] inode_48 final : public inode_48_parent<Key, Value> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  using inode_48_parent<Key, Value>::inode_48_parent;

  /// Add child or choose subtree for insertion.
  ///
  /// Forwards to impl_helpers::add_or_choose_subtree.
  ///
  /// \param args Arguments forwarded to impl_helpers
  template <typename... Args>
  [[nodiscard]] auto add_or_choose_subtree(Args&&... args) {
    return impl_helpers::add_or_choose_subtree(*this,
                                               std::forward<Args>(args)...);
  }

  /// Remove child or choose subtree for removal.
  ///
  /// Forwards to impl_helpers::remove_or_choose_subtree.
  ///
  /// \param args Arguments forwarded to impl_helpers
  template <typename... Args>
  [[nodiscard]] auto remove_or_choose_subtree(Args&&... args) {
    return impl_helpers::remove_or_choose_subtree(*this,
                                                  std::forward<Args>(args)...);
  }
};

/// Test instantiation of inode_48 for size verification.
using inode_48_test_type = inode_48<std::uint64_t, unodb::value_view>;
#ifdef UNODB_DETAIL_AVX2
static_assert(sizeof(inode_48_test_type) == 672);
#else
static_assert(sizeof(inode_48_test_type) == 656);
#endif

/// Base class for inode_256
template <typename Key, typename Value>
using inode_256_parent = basic_inode_256<art_policy<Key, Value>>;

/// Internal node with 256 children for non-thread-safe ART.
///
/// Largest internal node type, used when inode_48 grows beyond capacity.
/// Has direct mapping from byte values to children.
template <typename Key, typename Value>
class [[nodiscard]] inode_256 final : public inode_256_parent<Key, Value> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  using inode_256_parent<Key, Value>::inode_256_parent;

  /// Add child or choose subtree for insertion.
  ///
  /// Forwards to impl_helpers::add_or_choose_subtree.
  ///
  /// \param args Arguments forwarded to impl_helpers
  template <typename... Args>
  [[nodiscard]] auto add_or_choose_subtree(Args&&... args) {
    return impl_helpers::add_or_choose_subtree(*this,
                                               std::forward<Args>(args)...);
  }

  /// Remove child or choose subtree for removal.
  ///
  /// Forwards to impl_helpers::remove_or_choose_subtree.
  ///
  /// \param args Arguments forwarded to impl_helpers
  template <typename... Args>
  [[nodiscard]] auto remove_or_choose_subtree(Args&&... args) {
    return impl_helpers::remove_or_choose_subtree(*this,
                                                  std::forward<Args>(args)...);
  }
};

static_assert(sizeof(inode_256<std::uint64_t, unodb::value_view>) == 2064);

/// Unwrap fake critical section wrapper to access underlying node pointer.
///
/// Converts from in_fake_critical_section wrapper to raw node_ptr for
/// non-thread-safe ART where critical sections are no-ops. Uses
/// reinterpret_cast because we cannot dereference, load(), or take the address
/// of the wrapper directly (it is a temporary by then).
///
/// \param ptr Pointer to wrapped node pointer
/// \return Pointer to unwrapped node pointer
UNODB_DETAIL_DISABLE_MSVC_WARNING(26490)
inline auto* unwrap_fake_critical_section(
    unodb::in_fake_critical_section<unodb::detail::node_ptr>* ptr) noexcept {
  return reinterpret_cast<unodb::detail::node_ptr*>(ptr);
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// Return the correct child node for insertion operation (i.e. having the
/// matching next key byte value). If there is no child node for the next key
/// byte, create the leaf and insert it in the current node.
template <typename Key, typename Value, class INode>
detail::node_ptr* impl_helpers::add_or_choose_subtree(
    INode& inode, std::byte key_byte, basic_art_key<Key> k, Value v,
    db<Key, Value>& db_instance, tree_depth<basic_art_key<Key>> depth,
    detail::node_ptr* node_in_parent) {
  auto* const child =
      unwrap_fake_critical_section(inode.find_child(key_byte).second);

  if (child != nullptr) return child;

  if constexpr (art_policy<Key, Value>::can_eliminate_leaf) {
    // Value-in-slot: pack value directly, no leaf allocation.
    const auto packed = art_policy<Key, Value>::pack_value(v);
    const auto children_count = inode.get_children_count();

    if constexpr (!std::is_same_v<INode, inode_256<Key, Value>>) {
      if (UNODB_DETAIL_UNLIKELY(children_count == INode::capacity)) {
        auto larger_node{INode::larger_derived_type::create(
            db_instance, inode, packed, depth, key_byte)};
        *node_in_parent =
            node_ptr{larger_node.release(), INode::larger_derived_type::type};
#ifdef UNODB_DETAIL_WITH_STATS
        db_instance
            .template account_growing_inode<INode::larger_derived_type::type>();
#endif
        if constexpr (art_policy<Key, Value>::full_key_in_inode_path) {
          const auto chain_start =
              static_cast<tree_depth<basic_art_key<Key>>>(depth + 1);
          if (chain_start < k.size()) {
            const volatile auto* vp = node_in_parent;
            auto& new_inode =
                *const_cast<detail::node_ptr&>(*vp)
                     .template ptr<typename INode::larger_derived_type*>();
            auto [ci, slotraw] = new_inode.find_child(key_byte);
            auto* const slot = unwrap_fake_critical_section(slotraw);
            UNODB_DETAIL_ASSERT(slot != nullptr);
            *slot = db_instance.build_chain(k, *slot, chain_start);
            new_inode.clear_value_bit(ci);
          }
        }
        return child;
      }
    }
    inode.add_to_nonfull(packed, depth, key_byte, children_count);

    if constexpr (art_policy<Key, Value>::full_key_in_inode_path) {
      const auto chain_start =
          static_cast<tree_depth<basic_art_key<Key>>>(depth + 1);
      if (chain_start < k.size()) {
        std::atomic_signal_fence(std::memory_order_acq_rel);
        auto [ci2, slotraw2] = inode.find_child(key_byte);
        auto* const slot = unwrap_fake_critical_section(slotraw2);
        UNODB_DETAIL_ASSERT(slot != nullptr);
        *slot = db_instance.build_chain(k, *slot, chain_start);
        if constexpr (art_policy<Key, Value>::can_eliminate_leaf) {
          inode.clear_value_bit(ci2);
        }
      }
    }
    return child;
  } else {
    // Leaf-based: allocate leaf as before.
    auto leaf = art_policy<Key, Value>::make_db_leaf_ptr(k, v, db_instance);
    const auto children_count = inode.get_children_count();

    if constexpr (!std::is_same_v<INode, inode_256<Key, Value>>) {
      if (UNODB_DETAIL_UNLIKELY(children_count == INode::capacity)) {
        auto larger_node{INode::larger_derived_type::create(
            db_instance, inode, std::move(leaf), depth, key_byte)};
        *node_in_parent =
            node_ptr{larger_node.release(), INode::larger_derived_type::type};
#ifdef UNODB_DETAIL_WITH_STATS
        db_instance
            .template account_growing_inode<INode::larger_derived_type::type>();
#endif  // UNODB_DETAIL_WITH_STATS

        // For full_key_in_inode_path: wrap the bare leaf in a chain.
        if constexpr (art_policy<Key, Value>::full_key_in_inode_path) {
          const auto chain_start =
              static_cast<tree_depth<basic_art_key<Key>>>(depth + 1);
          if (chain_start < k.size()) {
            // TODO(#700): volatile works around type punning through unions.
            const volatile auto* vp = node_in_parent;
            auto& new_inode =
                *const_cast<detail::node_ptr&>(*vp)
                     .template ptr<typename INode::larger_derived_type*>();
            auto* const slot = unwrap_fake_critical_section(
                new_inode.find_child(key_byte).second);
            UNODB_DETAIL_ASSERT(slot != nullptr);
            *slot = db_instance.build_chain(k, *slot, chain_start);
          }
        }

        return child;
      }
    }
    inode.add_to_nonfull(std::move(leaf), depth, key_byte, children_count);

    // For full_key_in_inode_path: wrap the bare leaf in a chain encoding
    // the remaining key suffix.  The leaf was just inserted into the slot
    // for key_byte — find it and replace with the chain top.
    // Compiler barrier required: GCC 12+ at -O2 can elide the re-read of
    // inode data after add_to_nonfull due to strict aliasing (#700).
    // atomic_signal_fence is a zero-cost compiler-only barrier.
    if constexpr (art_policy<Key, Value>::full_key_in_inode_path) {
      const auto chain_start =
          static_cast<tree_depth<basic_art_key<Key>>>(depth + 1);
      if (chain_start < k.size()) {
        std::atomic_signal_fence(std::memory_order_acq_rel);
        auto [ci, slotraw] = inode.find_child(key_byte);
        auto* const slot = unwrap_fake_critical_section(slotraw);
        UNODB_DETAIL_ASSERT(slot != nullptr);
        *slot = db_instance.build_chain(k, *slot, chain_start);
        if constexpr (art_policy<Key, Value>::can_eliminate_leaf) {
          inode.clear_value_bit(ci);
        }
      }
    }

    return child;
  }  // else (leaf-based)
}

// MSVC C26815 false positive: create() returns smart pointer with LIFETIMEBOUND
// on db param, but release() transfers ownership and the raw pointer's validity
// is independent of the temporary unique_ptr.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
template <typename Key, typename Value, class INode>
std::optional<detail::node_ptr*> impl_helpers::remove_or_choose_subtree(
    INode& inode, std::byte key_byte, basic_art_key<Key> k,
    db<Key, Value>& db_instance, detail::node_ptr* node_in_parent) {
  const auto [child_i, child_ptr]{inode.find_child(key_byte)};

  if (child_ptr == nullptr) return {};

  const auto child_ptr_val{child_ptr->load()};

  if constexpr (art_policy<Key, Value>::can_eliminate_leaf) {
    if (!inode.is_value_in_slot(child_i)) {
      return unwrap_fake_critical_section(child_ptr);
    }
  } else {
    if (child_ptr_val.type() != node_type::LEAF)
      return unwrap_fake_critical_section(child_ptr);

    const auto* const leaf{
        child_ptr_val.template ptr<typename db<Key, Value>::leaf_type*>()};
    if (!leaf->matches(k)) return {};
  }

  if (UNODB_DETAIL_UNLIKELY(inode.is_min_size())) {
    if constexpr (std::is_same_v<INode, inode_4<Key, Value>>) {
      if (UNODB_DETAIL_LIKELY(inode.can_collapse(child_i))) {
        auto current_node{art_policy<Key, Value>::make_db_inode_unique_ptr(
            &inode, db_instance)};
        *node_in_parent = current_node->leave_last_child(child_i, db_instance);
#ifdef UNODB_DETAIL_WITH_STATS
        db_instance.template account_shrinking_inode<INode::type>();
#endif
      } else {
        // Prefix overflow — cannot collapse.  Just remove the child entry.
        inode.remove(child_i, db_instance);
      }
    } else {
      auto new_node{
          INode::smaller_derived_type::create(db_instance, inode, child_i)};
      *node_in_parent =
          node_ptr{new_node.release(), INode::smaller_derived_type::type};
#ifdef UNODB_DETAIL_WITH_STATS
      db_instance.template account_shrinking_inode<INode::type>();
#endif
    }
    return nullptr;
  }

  inode.remove(child_i, db_instance);
  return nullptr;
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

}  // namespace detail

template <typename Key, typename Value>
db<Key, Value>::~db() noexcept {
  delete_root_subtree();
}

template <typename Key, typename Value>
typename db<Key, Value>::get_result db<Key, Value>::get_internal(
    art_key_type k) const noexcept {
  if (UNODB_DETAIL_UNLIKELY(root == nullptr)) return {};
  if constexpr (std::is_same_v<Key, key_view>) {
    if (UNODB_DETAIL_UNLIKELY(k.size() == 0)) return {};
  }

  auto node{root};
  auto remaining_key{k};

  while (true) {
    const auto node_type = node.type();
    if (node_type == node_type::LEAF) {
      const auto* const leaf{node.template ptr<leaf_type*>()};
      if constexpr (art_policy::can_eliminate_key_in_leaf) {
        if (remaining_key.size() == 0)
          return leaf->template get_value<value_type>();
      } else {
        if (leaf->matches(k)) return leaf->template get_value<value_type>();
      }
      return {};
    }

    UNODB_DETAIL_ASSERT(node_type != node_type::LEAF);

    auto* const inode{node.template ptr<inode_type*>()};
    const auto& key_prefix{inode->get_key_prefix()};
    const auto key_prefix_length{key_prefix.length()};
    if (key_prefix.get_shared_length(remaining_key) < key_prefix_length)
      return {};
    remaining_key.shift_right(key_prefix_length);
    const auto [child_i,
                child_ptr]{inode->find_child(node_type, remaining_key[0])};
    if (child_ptr == nullptr) return {};

    if constexpr (art_policy::can_eliminate_leaf) {
      if (inode->is_value_in_slot(node_type, child_i)) {
        return art_policy::unpack_value(child_ptr->load());
      }
    }

    node = *child_ptr;
    remaining_key.shift_right(1);
  }
}

UNODB_DETAIL_DISABLE_MSVC_WARNING(26430)
// MSVC C26815 false positive: make_db_leaf_ptr/inode::create return smart
// pointers with LIFETIMEBOUND on db param, but release() transfers ownership
// and the raw pointer's validity is independent of the temporary unique_ptr.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
template <typename Key, typename Value>
bool db<Key, Value>::insert_internal(art_key_type insert_key, value_type v) {
  if constexpr (std::is_same_v<Key, key_view>) {
    if (UNODB_DETAIL_UNLIKELY(insert_key.size() == 0)) {
      throw std::length_error("Key must not be empty");
    }
    if (UNODB_DETAIL_UNLIKELY(
            insert_key.size() >
            std::numeric_limits<unodb::key_size_type>::max())) {
      throw std::length_error("Key length must fit in std::uint32_t");
    }
  }

  if (UNODB_DETAIL_UNLIKELY(root == nullptr)) {
    if constexpr (art_policy::can_eliminate_leaf) {
      root = build_chain(insert_key, art_policy::pack_value(v),
                         tree_depth_type{0});
    } else {
      auto leaf = art_policy::make_db_leaf_ptr(insert_key, v, *this);
      if constexpr (art_policy::can_eliminate_key_in_leaf) {
        root = build_chain(insert_key,
                           detail::node_ptr{leaf.release(), node_type::LEAF},
                           tree_depth_type{0});
      } else {
        root = detail::node_ptr{leaf.release(), node_type::LEAF};
      }
    }
    return true;
  }

  if constexpr (std::is_same_v<Key, key_view>) {
    return insert_internal_key_view(insert_key, v);
  } else {
    return insert_internal_fixed(insert_key, v);
  }
}

template <typename Key, typename Value>
UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
bool db<Key, Value>::insert_internal_fixed(art_key_type insert_key,
                                           value_type v) {
  if constexpr (std::is_same_v<Key, key_view>) {
    // Unreachable: caller dispatches key_view to insert_internal_key_view.
    std::ignore = insert_key;
    std::ignore = v;
    UNODB_DETAIL_CANNOT_HAPPEN();
  } else {
    auto* node = &root;
    tree_depth_type depth{};
    auto remaining_key{insert_key};

    while (true) {
      const auto node_type = node->type();
      if (node_type == node_type::LEAF) {
        auto* const leaf{node->template ptr<leaf_type*>()};
        const auto existing_key{leaf->get_key_view()};
        const auto cmp = insert_key.cmp(existing_key);
        if (UNODB_DETAIL_UNLIKELY(cmp == 0)) {
          return false;  // exists
        }
        auto new_leaf = art_policy::make_db_leaf_ptr(insert_key, v, *this);
        auto new_node{inode_4::create(*this, existing_key, remaining_key, depth,
                                      leaf, std::move(new_leaf))};
        *node = detail::node_ptr{new_node.release(), node_type::I4};
#ifdef UNODB_DETAIL_WITH_STATS
        account_growing_inode<node_type::I4>();
#endif  // UNODB_DETAIL_WITH_STATS
        return true;
      }

      UNODB_DETAIL_ASSERT(node_type != node_type::LEAF);

      auto* const inode{node->template ptr<inode_type*>()};
      const auto& key_prefix{inode->get_key_prefix()};
      const auto key_prefix_length{key_prefix.length()};
      const auto shared_prefix_len{key_prefix.get_shared_length(remaining_key)};
      if (shared_prefix_len < key_prefix_length) {
        auto leaf = art_policy::make_db_leaf_ptr(insert_key, v, *this);
        auto new_node =
            inode_4::create(*this, *node, shared_prefix_len, depth,
                            std::move(leaf), remaining_key[shared_prefix_len]);
        *node = detail::node_ptr{new_node.release(), node_type::I4};
#ifdef UNODB_DETAIL_WITH_STATS
        account_growing_inode<node_type::I4>();
        ++key_prefix_splits;
        UNODB_DETAIL_ASSERT(growing_inode_counts[internal_as_i<node_type::I4>] >
                            key_prefix_splits);
#endif  // UNODB_DETAIL_WITH_STATS
        return true;
      }
      UNODB_DETAIL_ASSERT(shared_prefix_len == key_prefix_length);
      depth += key_prefix_length;
      remaining_key.shift_right(key_prefix_length);

      node = inode->template add_or_choose_subtree<detail::node_ptr*>(
          node_type, remaining_key[0], insert_key, v, *this, depth, node);

      if (node == nullptr) return true;

      if constexpr (art_policy::can_eliminate_leaf) {
        const auto [ci, _] = inode->find_child(node_type, remaining_key[0]);
        if (inode->is_value_in_slot(node_type, ci)) {
          // TODO(#707): chain split not implemented. Two keys share
          // a chain prefix but diverge deeper. Need to replace the
          // packed value with a new I4 holding both values.
          UNODB_DETAIL_CANNOT_HAPPEN();
        }
      }

      ++depth;
      remaining_key.shift_right(1);
    }
  }  // else (non-key_view)
}

template <typename Key, typename Value>
detail::node_ptr db<Key, Value>::build_chain(art_key_type k,
                                             detail::node_ptr child,
                                             tree_depth_type start_depth) {
  constexpr std::size_t cap = detail::key_prefix_capacity;
  const auto full_key = k.get_key_view();
  const auto key_len = k.size();
  const auto start = static_cast<std::size_t>(start_depth);
  auto current = child;
  bool child_is_value = art_policy::can_eliminate_leaf;
  bool owns_current = false;  // true once we've built at least one chain node
  // Build bottom-up: start from end of key, work toward start_depth.
  // Each chain I4 consumes up to cap prefix bytes + 1 dispatch byte.
  try {
    std::size_t pos = key_len;
    while (pos > start + cap) {
      const auto depth = pos - cap - 1;
      const auto dispatch = full_key[pos - 1];
      auto remaining = k;
      remaining.shift_right(depth);
      auto chain{
          inode_4::create(*this, full_key, remaining,
                          tree_depth_type{static_cast<std::uint32_t>(depth)},
                          dispatch, current)};
      if (child_is_value) {
        chain->set_value_bit(0);
        child_is_value = false;
      }
      current = detail::node_ptr{chain.release(), node_type::I4};
      owns_current = true;
#ifdef UNODB_DETAIL_WITH_STATS
      account_growing_inode<node_type::I4>();
#endif
      pos = depth;
    }
    // Tail: remaining bytes from start_depth to pos.
    if (pos > start) {
      const auto dispatch = full_key[pos - 1];
      auto chain{inode_4::create(
          *this, full_key, tree_depth_type{static_cast<std::uint32_t>(start)},
          static_cast<detail::key_prefix_size>(pos - start - 1), dispatch,
          current)};
      if (child_is_value) {
        chain->set_value_bit(0);
      }
      current = detail::node_ptr{chain.release(), node_type::I4};
#ifdef UNODB_DETAIL_WITH_STATS
      account_growing_inode<node_type::I4>();
#endif
    }
  } catch (...) {
    if (owns_current) art_policy::delete_subtree(current, *this);
    throw;
  }
  return current;
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

template <typename Key, typename Value>
bool db<Key, Value>::insert_internal_key_view(art_key_type insert_key,
                                              value_type v) {
  auto* node = &root;
  tree_depth_type depth{};
  auto remaining_key{insert_key};

  while (true) {
    if constexpr (art_policy::can_eliminate_leaf) {
      // Value-in-slot: no LEAF nodes in the tree. The insert loop
      // only descends through inodes. Packed values are detected
      // by is_value_in_slot after add_or_choose_subtree above.
    } else {
      const auto node_type = node->type();
      if (node_type == node_type::LEAF) {
        if constexpr (art_policy::can_eliminate_key_in_leaf) {
          // Keyless leaf: the inode path consumed all bytes of the existing
          // key.  The ART prefix restriction guarantees no key is a prefix
          // of another, so remaining_key must be empty → duplicate.
          UNODB_DETAIL_ASSERT(remaining_key.size() == 0);
          return false;
        } else {
          auto* const leaf{node->template ptr<leaf_type*>()};
          const auto existing_key{leaf->get_key_view()};
          const auto cmp = insert_key.cmp(existing_key);
          if (UNODB_DETAIL_UNLIKELY(cmp == 0)) {
            return false;  // exists
          }
          constexpr auto cap = detail::key_prefix_capacity;
          const auto remaining_existing = existing_key.subspan(depth);
          const auto shared = detail::key_prefix_snapshot::shared_len(
              detail::get_u64(remaining_existing), remaining_key.get_u64(),
              cap);
          if (shared >= cap && remaining_existing.size() > cap &&
              remaining_key.size() > cap &&
              remaining_existing[cap] == remaining_key[cap]) {
            const auto dispatch = remaining_existing[cap];
            auto chain{inode_4::create(*this, existing_key, remaining_key,
                                       depth, dispatch, *node)};
            *node = detail::node_ptr{chain.release(), node_type::I4};
#ifdef UNODB_DETAIL_WITH_STATS
            account_growing_inode<node_type::I4>();
#endif  // UNODB_DETAIL_WITH_STATS
            continue;
          }
          auto new_leaf = art_policy::make_db_leaf_ptr(insert_key, v, *this);
          auto new_node{inode_4::create(*this, existing_key, remaining_key,
                                        depth, leaf, std::move(new_leaf))};
          *node = detail::node_ptr{new_node.release(), node_type::I4};
#ifdef UNODB_DETAIL_WITH_STATS
          account_growing_inode<node_type::I4>();
#endif  // UNODB_DETAIL_WITH_STATS

          if constexpr (art_policy::full_key_in_inode_path) {
            const auto chain_start =
                static_cast<tree_depth_type>(depth + shared + 1);
            if (chain_start < insert_key.size()) {
              auto* const new_inode = node->template ptr<inode_type*>();
              auto [ci3, slotraw3] =
                  new_inode->find_child(node_type::I4, remaining_key[shared]);
              auto* const slot = unwrap_fake_critical_section(slotraw3);
              UNODB_DETAIL_ASSERT(slot != nullptr);
              *slot = build_chain(insert_key, *slot, chain_start);
              if constexpr (art_policy::can_eliminate_leaf) {
                new_inode->clear_value_bit(node_type::I4, ci3);
              }
            }
          }

          return true;
        }  // else (keyed leaf)
      }
    }  // else (!can_eliminate_leaf)

    const auto node_type = node->type();
    UNODB_DETAIL_ASSERT(node_type != node_type::LEAF);

    auto* const inode{node->template ptr<inode_type*>()};
    const auto& key_prefix{inode->get_key_prefix()};
    const auto key_prefix_length{key_prefix.length()};
    const auto shared_prefix_len{key_prefix.get_shared_length(remaining_key)};
    if (shared_prefix_len < key_prefix_length) {
      if constexpr (art_policy::can_eliminate_leaf) {
        auto new_node = inode_4::create(*this, *node, shared_prefix_len, depth,
                                        art_policy::pack_value(v),
                                        remaining_key[shared_prefix_len]);
        *node = detail::node_ptr{new_node.release(), node_type::I4};
      } else {
        auto leaf = art_policy::make_db_leaf_ptr(insert_key, v, *this);
        auto new_node =
            inode_4::create(*this, *node, shared_prefix_len, depth,
                            std::move(leaf), remaining_key[shared_prefix_len]);
        *node = detail::node_ptr{new_node.release(), node_type::I4};
      }
#ifdef UNODB_DETAIL_WITH_STATS
      account_growing_inode<node_type::I4>();
      ++key_prefix_splits;
      UNODB_DETAIL_ASSERT(growing_inode_counts[internal_as_i<node_type::I4>] >
                          key_prefix_splits);
#endif  // UNODB_DETAIL_WITH_STATS

      if constexpr (art_policy::full_key_in_inode_path) {
        const auto chain_start =
            static_cast<tree_depth_type>(depth + shared_prefix_len + 1);
        if (chain_start < insert_key.size()) {
          auto* const new_inode = node->template ptr<inode_type*>();
          auto [ci4, slotraw4] = new_inode->find_child(
              node_type::I4, remaining_key[shared_prefix_len]);
          auto* const slot = unwrap_fake_critical_section(slotraw4);
          UNODB_DETAIL_ASSERT(slot != nullptr);
          *slot = build_chain(insert_key, *slot, chain_start);
          if constexpr (art_policy::can_eliminate_leaf) {
            new_inode->clear_value_bit(node_type::I4, ci4);
          }
        }
      }
      return true;
    }
    UNODB_DETAIL_ASSERT(shared_prefix_len == key_prefix_length);
    depth += key_prefix_length;
    remaining_key.shift_right(key_prefix_length);

    node = inode->template add_or_choose_subtree<detail::node_ptr*>(
        node_type, remaining_key[0], insert_key, v, *this, depth, node);

    if (node == nullptr) return true;

    if constexpr (art_policy::can_eliminate_leaf) {
      const auto [ci, _] = inode->find_child(node_type, remaining_key[0]);
      if (inode->is_value_in_slot(node_type, ci)) {
        // The chain encoded the full key. Reaching a packed value
        // means the key already exists (duplicate).
        return false;
      }
    }

    ++depth;
    remaining_key.shift_right(1);
  }
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

template <typename Key, typename Value>
bool db<Key, Value>::remove_internal(art_key_type remove_key) {
  if (UNODB_DETAIL_UNLIKELY(root == nullptr)) return false;
  if constexpr (std::is_same_v<Key, key_view>) {
    if (UNODB_DETAIL_UNLIKELY(remove_key.size() == 0)) return false;
  }

  if constexpr (!art_policy::can_eliminate_leaf) {
    if (root.type() == node_type::LEAF) {
      auto* const root_leaf{root.ptr<leaf_type*>()};
      if (root_leaf->matches(remove_key)) {
        const auto r{art_policy::reclaim_leaf_on_scope_exit(root_leaf, *this)};
        root = nullptr;
        return true;
      }
      return false;
    }
  }

  if constexpr (std::is_same_v<Key, key_view>) {
    return remove_internal_key_view(remove_key);
  } else {
    return remove_internal_fixed(remove_key);
  }
}

template <typename Key, typename Value>
bool db<Key, Value>::remove_internal_fixed(art_key_type remove_key) {
  auto* node = &root;
  auto remaining_key{remove_key};

  while (true) {
    const auto node_type = node->type();
    UNODB_DETAIL_ASSERT(node_type != node_type::LEAF);

    auto* const inode{node->template ptr<inode_type*>()};
    const auto& key_prefix{inode->get_key_prefix()};
    const auto key_prefix_length{key_prefix.length()};
    const auto shared_prefix_len{key_prefix.get_shared_length(remaining_key)};
    if (shared_prefix_len < key_prefix_length) return false;

    UNODB_DETAIL_ASSERT(shared_prefix_len == key_prefix_length);
    remaining_key.shift_right(key_prefix_length);

    const auto remove_result{inode->template remove_or_choose_subtree<
        std::optional<detail::node_ptr*>>(node_type, remaining_key[0],
                                          remove_key, *this, node)};
    if (UNODB_DETAIL_UNLIKELY(!remove_result)) return false;

    auto* const child_ptr{*remove_result};
    if (child_ptr == nullptr) return true;

    node = child_ptr;
    remaining_key.shift_right(1);
  }
}

template <typename Key, typename Value>
// MSVC C26815 false positive: ptr<>() on local node_ptr values
UNODB_DETAIL_DISABLE_MSVC_WARNING(26815) bool db<
    Key, Value>::remove_internal_key_view(art_key_type remove_key) {
  struct stack_entry {
    detail::node_ptr* slot;
    std::uint8_t child_i;
  };

  boost::container::small_vector<stack_entry, 32> stack;
  auto* slot = &root;
  auto remaining_key{remove_key};

  // --- Downward pass: find the leaf ---
  while (true) {
    const auto node_val = *slot;
    const auto ntype = node_val.type();
    UNODB_DETAIL_ASSERT(ntype != node_type::LEAF);

    auto* const inode{node_val.template ptr<inode_type*>()};
    const auto& kp{inode->get_key_prefix()};
    const auto kp_len{kp.length()};
    if (kp.get_shared_length(remaining_key) < kp_len) return false;
    remaining_key.shift_right(kp_len);

    const auto [child_i, child_ptr]{inode->find_child(ntype, remaining_key[0])};
    if (child_ptr == nullptr) return false;

    const auto child_val{child_ptr->load()};
    if constexpr (art_policy::can_eliminate_leaf) {
      if (!inode->is_value_in_slot(ntype, child_i)) {
        stack.push_back({slot, child_i});
        slot = detail::unwrap_fake_critical_section(child_ptr);
        remaining_key.shift_right(1);
        continue;
      }
      // Found a packed value — verify key bytes consumed.
      if (remaining_key.size() != 1) return false;
    } else {
      if (child_val.type() != node_type::LEAF) {
        stack.push_back({slot, child_i});
        slot = detail::unwrap_fake_critical_section(child_ptr);
        remaining_key.shift_right(1);
        continue;
      }

      // Found a leaf — verify it matches.
      const auto* const leaf{child_val.template ptr<const leaf_type*>()};
      if constexpr (art_policy::can_eliminate_key_in_leaf) {
        if (remaining_key.size() != 1) return false;
      } else {
        if (!leaf->matches(remove_key)) return false;
      }
    }

    // --- Upward pass ---
    const auto count = inode->get_children_count();

    if (count > 1 || ntype != node_type::I4) {
      const auto remove_result UNODB_DETAIL_USED_IN_DEBUG{
          inode->template remove_or_choose_subtree<
              std::optional<detail::node_ptr*>>(ntype, remaining_key[0],
                                                remove_key, *this, slot)};
      UNODB_DETAIL_ASSERT(remove_result.has_value());
      UNODB_DETAIL_ASSERT(*remove_result == nullptr);
      return true;
    }

    // Single-child inode (chain node).  Reclaim leaf and chain,
    // then walk up cleaning any further empty chains.
    if constexpr (!art_policy::can_eliminate_leaf) {
      auto* const leaf{child_val.template ptr<leaf_type*>()};
      const auto rl{art_policy::reclaim_leaf_on_scope_exit(leaf, *this)};
    }
    {
      const auto ri{art_policy::make_db_inode_unique_ptr(
          node_val.template ptr<inode_4*>(), *this)};
    }
#ifdef UNODB_DETAIL_WITH_STATS
    account_shrinking_inode<node_type::I4>();
#endif

    while (!stack.empty()) {
      const auto entry = stack.back();
      stack.pop_back();
      const auto parent_val = *entry.slot;
      const auto ptype = parent_val.type();
      auto* const pinode{parent_val.template ptr<inode_type*>()};
      const auto pcount = pinode->get_children_count();

      if (pcount == 1 && ptype == node_type::I4) {
        {
          const auto ri{art_policy::make_db_inode_unique_ptr(
              parent_val.template ptr<inode_4*>(), *this)};
        }
#ifdef UNODB_DETAIL_WITH_STATS
        account_shrinking_inode<node_type::I4>();
#endif
        continue;
      }

      if (ptype == node_type::I4 &&
          pcount == detail::basic_inode_4<art_policy>::min_size) {
        pinode->remove_child_entry(ptype, entry.child_i);
        auto* const pi4{parent_val.template ptr<inode_4*>()};
        const auto remaining_iter = pi4->begin();
        const auto remaining = pi4->get_child(0);
        UNODB_DETAIL_DISABLE_MSVC_WARNING(26814)
        const bool remaining_is_value =
            art_policy::can_eliminate_leaf && pi4->is_value_in_slot(0);
        UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
        if (!remaining_is_value && remaining.type() != node_type::LEAF) {
          auto* const remaining_inode{remaining.template ptr<inode_type*>()};
          const auto child_prefix_len =
              remaining_inode->get_key_prefix().length();
          const auto parent_prefix_len = pi4->get_key_prefix().length();
          if (child_prefix_len + parent_prefix_len + 1 >=
              detail::key_prefix_capacity) {
            return true;
          }
          remaining_inode->get_key_prefix().prepend(pi4->get_key_prefix(),
                                                    remaining_iter.key_byte);
        } else if constexpr (art_policy::can_eliminate_key_in_leaf) {
          // Keyless leaf: don't collapse — keep the chain intact.
          return true;
        }
        *entry.slot = remaining;
        { const auto ri{art_policy::make_db_inode_unique_ptr(pi4, *this)}; }
#ifdef UNODB_DETAIL_WITH_STATS
        account_shrinking_inode<node_type::I4>();
#endif
      } else if (ptype == node_type::I16 &&
                 pcount == detail::basic_inode_16<art_policy>::min_size) {
        auto new_node{inode_4::create(
            *this, *parent_val.template ptr<detail::inode_16<Key, Value>*>(),
            entry.child_i)};
        *entry.slot = detail::node_ptr{new_node.release(), node_type::I4};
#ifdef UNODB_DETAIL_WITH_STATS
        account_shrinking_inode<node_type::I16>();
#endif
      } else if (ptype == node_type::I48 &&
                 pcount == detail::basic_inode_48<art_policy>::min_size) {
        auto new_node{detail::inode_16<Key, Value>::create(
            *this, *parent_val.template ptr<detail::inode_48<Key, Value>*>(),
            entry.child_i)};
        *entry.slot = detail::node_ptr{new_node.release(), node_type::I16};
#ifdef UNODB_DETAIL_WITH_STATS
        account_shrinking_inode<node_type::I48>();
#endif
      } else if (ptype == node_type::I256 &&
                 pcount == detail::basic_inode_256<art_policy>::min_size) {
        auto new_node{detail::inode_48<Key, Value>::create(
            *this, *parent_val.template ptr<detail::inode_256<Key, Value>*>(),
            entry.child_i)};
        *entry.slot = detail::node_ptr{new_node.release(), node_type::I48};
#ifdef UNODB_DETAIL_WITH_STATS
        account_shrinking_inode<node_type::I256>();
#endif
      } else {
        pinode->remove_child_entry(ptype, entry.child_i);
      }
      return true;
    }

    // Stack empty — chain was the root.
    root = nullptr;
    return true;
  }
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

//
// ART Iterator Implementation
//

// TODO(laurynas): the method pairs first, last; next, prior;
// left_most_traversal, right_most_traversal are identical except for a couple
// lines. Extract helper methods templatized on the differences.
template <typename Key, typename Value>
typename db<Key, Value>::iterator& db<Key, Value>::iterator::first() {
  invalidate();  // clear the stack
  if (UNODB_DETAIL_UNLIKELY(db_.root == nullptr)) return *this;  // empty tree.
  const auto node{db_.root};
  return left_most_traversal(node);
}

template <typename Key, typename Value>
typename db<Key, Value>::iterator& db<Key, Value>::iterator::last() {
  invalidate();  // clear the stack
  if (UNODB_DETAIL_UNLIKELY(db_.root == nullptr)) return *this;  // empty tree.
  const auto node{db_.root};
  return right_most_traversal(node);
}

template <typename Key, typename Value>
typename db<Key, Value>::iterator& db<Key, Value>::iterator::next() {
  while (!empty()) {
    const auto& e = top();
    const auto node{e.node};
    UNODB_DETAIL_ASSERT(node != nullptr);
    const auto node_type = node.type();
    if (node_type == node_type::LEAF ||
        (art_policy::can_eliminate_leaf && e.packed_leaf)) {
      pop();     // pop off the leaf
      continue;  // falls through loop if just a root leaf since stack now
                 // empty.
    }
    auto* inode{node.template ptr<inode_type*>()};
    const auto nxt = inode->next(node_type,
                                 e.child_index);  // next child of that parent.
    if (!nxt.has_value()) {
      pop();     // Nothing more for that inode.
      continue;  // We will look for the right sibling of the parent inode.
    }
    // Fix up stack for new parent node state and left-most descent.
    const auto& e2 = nxt.value();
    pop();
    push(e2);
    const auto child = inode->get_child(node_type, e2.child_index);  // descend
    if constexpr (art_policy::can_eliminate_leaf) {
      if (inode->is_value_in_slot(node_type, e2.child_index)) {
        push_leaf(child);
        return *this;
      }
    }
    return left_most_traversal(child);
  }
  return *this;  // stack is empty, so iterator is at the end
}

template <typename Key, typename Value>
typename db<Key, Value>::iterator& db<Key, Value>::iterator::prior() {
  while (!empty()) {
    const auto& e = top();
    const auto node{e.node};
    UNODB_DETAIL_ASSERT(node != nullptr);
    const auto node_type = node.type();
    if (node_type == node_type::LEAF ||
        (art_policy::can_eliminate_leaf && e.packed_leaf)) {
      pop();     // pop off the leaf
      continue;  // falls through loop if just a root leaf since stack now
                 // empty.
    }
    auto* inode{node.template ptr<inode_type*>()};
    auto nxt = inode->prior(node_type, e.child_index);  // parent's prev child
    if (!nxt) {
      pop();     // Nothing more for that inode.
      continue;  // We will look for the left sibling of the parent inode.
    }
    // Fix up stack for new parent node state and right-most descent.
    UNODB_DETAIL_ASSERT(nxt.has_value());  // value exists for std::optional
    const auto& e2 = nxt.value();
    pop();
    push(e2);
    auto child = inode->get_child(node_type, e2.child_index);  // descend
    if constexpr (art_policy::can_eliminate_leaf) {
      if (inode->is_value_in_slot(node_type, e2.child_index)) {
        push_leaf(child);
        return *this;
      }
    }
    return right_most_traversal(child);
  }
  return *this;  // stack is empty, so iterator is at the end.
}

template <typename Key, typename Value>
typename db<Key, Value>::iterator&
db<Key, Value>::iterator::left_most_traversal(detail::node_ptr node) {
  while (true) {
    UNODB_DETAIL_ASSERT(node != nullptr);
    const auto node_type = node.type();
    if (node_type == node_type::LEAF) {
      push_leaf(node);
      return *this;  // done
    }
    // recursive descent.
    auto* const inode{node.ptr<inode_type*>()};
    const auto e =
        inode->begin(node_type);  // first child of current internal node
    push(e);                      // push the entry on the stack.
    const auto child = inode->get_child(node_type, e.child_index);
    if constexpr (art_policy::can_eliminate_leaf) {
      if (inode->is_value_in_slot(node_type, e.child_index)) {
        push_leaf(child);
        return *this;
      }
    }
    node = child;
  }
  UNODB_DETAIL_CANNOT_HAPPEN();
}

template <typename Key, typename Value>
typename db<Key, Value>::iterator&
db<Key, Value>::iterator::right_most_traversal(detail::node_ptr node) {
  while (true) {
    UNODB_DETAIL_ASSERT(node != nullptr);
    const auto node_type = node.type();
    if (node_type == node_type::LEAF) {
      push_leaf(node);
      return *this;  // done
    }
    // recursive descent.
    auto* const inode{node.ptr<inode_type*>()};
    const auto e =
        inode->last(node_type);  // last child of current internal node
    push(e);                     // push the entry on the stack.
    const auto child = inode->get_child(node_type, e.child_index);
    if constexpr (art_policy::can_eliminate_leaf) {
      if (inode->is_value_in_slot(node_type, e.child_index)) {
        push_leaf(child);
        return *this;
      }
    }
    node = child;
  }
  UNODB_DETAIL_CANNOT_HAPPEN();
}

template <typename Key, typename Value>
typename db<Key, Value>::iterator& db<Key, Value>::iterator::seek(
    art_key_type search_key, bool& match, bool fwd) {
  invalidate();   // invalidate the iterator (clear the stack).
  match = false;  // unless we wind up with an exact match.
  if (UNODB_DETAIL_UNLIKELY(db_.root == nullptr)) return *this;  // aka end

  auto node{db_.root};
  const auto k = search_key;
  auto remaining_key{k};

  while (true) {
    const auto node_type = node.type();
    if constexpr (!art_policy::can_eliminate_leaf) {
      if (node_type == node_type::LEAF) {
        const auto* const leaf{node.template ptr<leaf_type*>()};
        push_leaf(node);
        int cmp_{0};
        if constexpr (art_policy::full_key_in_inode_path) {
          cmp_ =
              unodb::detail::compare(keybuf_.get_key_view(), k.get_key_view());
        } else {
          cmp_ = leaf->cmp(k);
        }
        if (cmp_ == 0) {
          match = true;
          return *this;
        }
        if (fwd) {  // GTE semantics
          // if search_key < leaf, use the leaf, else next().
          return (cmp_ < 0) ? *this : next();
        }
        // LTE semantics: if search_key > leaf, use the leaf, else prior().
        return (cmp_ > 0) ? *this : prior();
      }
    }  // if constexpr (!can_eliminate_leaf)
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
        if (cmp_ < 0) {
          // FWD and the search key is ordered before this node.  We
          // want the left-most leaf under the node.
          return left_most_traversal(node);
        }
        // FWD and the search key is ordered after this node.  Right
        // most descent and then next().
        return right_most_traversal(node).next();
      }
      // reverse traversal
      if (cmp_ < 0) {
        // REV and the search key is ordered before this node.  We
        // want the preceding key.
        return left_most_traversal(node).prior();
      }
      // REV and the search key is ordered after this node.
      return right_most_traversal(node);
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
          // (iterator at the end) and return that state.
          if (!empty()) pop();
          while (!empty()) {
            const auto& centry = top();
            const auto cnode{centry.node};  // possible parent from the stack
            auto* const icnode{cnode.template ptr<inode_type*>()};
            const auto cnxt = icnode->next(
                cnode.type(), centry.child_index);  // right-sibling.
            if (cnxt) {
              auto nchild = icnode->get_child(cnode.type(), centry.child_index);
              return descend_left(icnode, cnode.type(), centry.child_index,
                                  nchild);
            }
            pop();
          }
          return *this;  // stack is empty (aka end iterator).
        }
        const auto& tmp = nxt.value();  // unwrap.
        const auto child_index = tmp.child_index;
        const auto child = inode->get_child(node_type, child_index);
        push(node, tmp.key_byte, child_index, tmp.prefix);  // the path we took
        return descend_left(inode, node_type, child_index, child);
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
        if (!empty()) pop();
        while (!empty()) {
          const auto& centry = top();
          const auto cnode{centry.node};  // possible parent from stack
          auto* const icnode{cnode.template ptr<inode_type*>()};
          const auto cnxt =
              icnode->prior(cnode.type(), centry.child_index);  // left-sibling.
          if (cnxt) {
            auto nchild = icnode->get_child(cnode.type(), centry.child_index);
            return descend_right(icnode, cnode.type(), centry.child_index,
                                 nchild);
          }
          pop();
        }
        return *this;  // stack is empty (aka end iterator).
      }
      const auto& tmp = nxt.value();  // unwrap.
      const auto child_index{tmp.child_index};
      const auto child = inode->get_child(node_type, child_index);
      push(node, tmp.key_byte, child_index, tmp.prefix);  // the path we took
      return descend_right(inode, node_type, child_index, child);
    }
    // Simple case. There is a child for the current key byte.
    const auto child_index{res.first};
    const auto* const child{res.second};
    push(node, remaining_key[0], child_index, key_prefix);
    if constexpr (art_policy::can_eliminate_leaf) {
      if (inode->is_value_in_slot(node_type, child_index)) {
        push_leaf(*child);
        // Exact match — remaining key consumed by prefix + dispatch bytes.
        match = (remaining_key.size() <= 1);
        return *this;
      }
    }
    node = *child;
    remaining_key.shift_right(1);
  }  // while ( true )
  UNODB_DETAIL_CANNOT_HAPPEN();
}

UNODB_DETAIL_DISABLE_GCC_WARNING("-Wsuggest-attribute=pure")
template <typename Key, typename Value>
typename db<Key, Value>::iterator::get_key_result
db<Key, Value>::iterator::get_key() noexcept {
  UNODB_DETAIL_ASSERT(valid());  // by contract
  if constexpr (art_policy::full_key_in_inode_path) {
    return transient_key_view{keybuf_.get_key_view()};
  } else {
    const auto& e = stack_.top();
    const auto& node = e.node;
    UNODB_DETAIL_ASSERT(node.type() == node_type::LEAF);
    const auto* const leaf{node.template ptr<leaf_type*>()};
    return leaf->get_key_view();
  }
}
UNODB_DETAIL_RESTORE_GCC_WARNINGS()

template <typename Key, typename Value>
typename db<Key, Value>::value_type db<Key, Value>::iterator::get_val()
    const noexcept {
  UNODB_DETAIL_ASSERT(valid());  // by contract
  const auto& e = stack_.top();
  const auto& node = e.node;
  if constexpr (art_policy::can_eliminate_leaf) {
    return art_policy::unpack_value(node);
  } else {
    UNODB_DETAIL_ASSERT(node.type() == node_type::LEAF);
    const auto* const leaf{node.template ptr<leaf_type*>()};
    return leaf->template get_value<value_type>();
  }
}

//
// ART scan implementations.
//

template <typename Key, typename Value>
template <typename FN>
void db<Key, Value>::scan(FN fn, bool fwd) {
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

template <typename Key, typename Value>
template <typename FN>
void db<Key, Value>::scan_from(Key from_key, FN fn, bool fwd) {
  const art_key_type from_key_{from_key};  // convert to internal key
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

template <typename Key, typename Value>
template <typename FN>
void db<Key, Value>::scan_range(Key from_key, Key to_key, FN fn) {
  // TODO(thompsonbry) : variable length keys. Explore a cheaper way
  // to handle the exclusive bound case when developing variable
  // length key support based on the maintained key buffer.
  constexpr bool debug = false;             // set true to debug scan.
  const art_key_type from_key_{from_key};   // convert to internal key
  const art_key_type to_key_{to_key};       // convert to internal key
  const auto ret = from_key_.cmp(to_key_);  // compare the internal keys
  const bool fwd{ret < 0};                  // from_key is less than to_key
  if (ret == 0) return;                     // NOP
  bool match{};
  if (fwd) {
    iterator it(*this);
    it.seek(from_key_, match, true /*fwd*/);
    if constexpr (debug) {
      std::cerr << "scan_range:: fwd"
                << ", from_key=" << from_key_ << ", to_key=" << to_key_ << "\n";
      it.dump(std::cerr);
    }
    const visitor_type v{it};
    while (it.valid() && it.cmp(to_key_) < 0) {
      if (UNODB_DETAIL_UNLIKELY(fn(v))) break;
      it.next();
      if constexpr (debug) {
        std::cerr << "scan_range:: next()\n";
        it.dump(std::cerr);
      }
    }
  } else {  // reverse traversal.
    iterator it(*this);
    it.seek(from_key_, match, false /*fwd*/);
    if constexpr (debug) {
      std::cerr << "scan_range:: rev"
                << ", from_key=" << from_key_ << ", to_key=" << to_key_ << "\n";
      it.dump(std::cerr);
    }
    const visitor_type v{it};
    while (it.valid() && it.cmp(to_key_) > 0) {
      if (UNODB_DETAIL_UNLIKELY(fn(v))) break;
      it.prior();
      if constexpr (debug) {
        std::cerr << "scan_range:: prior()\n";
        it.dump(std::cerr);
      }
    }
  }
}

template <typename Key, typename Value>
void db<Key, Value>::delete_root_subtree() noexcept {
  if (root != nullptr) art_policy::delete_subtree(root, *this);

#ifdef UNODB_DETAIL_WITH_STATS
  // It is possible to reset the counter to zero instead of decrementing it for
  // each leaf, but not sure the savings will be significant.
  UNODB_DETAIL_ASSERT(node_counts[as_i<node_type::LEAF>] == 0);
#endif  // UNODB_DETAIL_WITH_STATS
}

template <typename Key, typename Value>
void db<Key, Value>::clear() noexcept {
  delete_root_subtree();

  root = nullptr;
#ifdef UNODB_DETAIL_WITH_STATS
  current_memory_use = 0;
  node_counts[as_i<node_type::I4>] = 0;
  node_counts[as_i<node_type::I16>] = 0;
  node_counts[as_i<node_type::I48>] = 0;
  node_counts[as_i<node_type::I256>] = 0;
#endif  // UNODB_DETAIL_WITH_STATS
}

#ifdef UNODB_DETAIL_WITH_STATS

template <typename Key, typename Value>
template <class INode>
constexpr void db<Key, Value>::increment_inode_count() noexcept {
  static_assert(inode_defs_type::template is_inode<INode>());

  ++node_counts[as_i<INode::type>];
  increase_memory_use(sizeof(INode));
}

template <typename Key, typename Value>
template <class INode>
constexpr void db<Key, Value>::decrement_inode_count() noexcept {
  static_assert(inode_defs_type::template is_inode<INode>());
  UNODB_DETAIL_ASSERT(node_counts[as_i<INode::type>] > 0);

  --node_counts[as_i<INode::type>];
  decrease_memory_use(sizeof(INode));
}

template <typename Key, typename Value>
template <node_type NodeType>
constexpr void db<Key, Value>::account_growing_inode() noexcept {
  static_assert(NodeType != node_type::LEAF);

  // NOLINTNEXTLINE(google-readability-casting)
  ++growing_inode_counts[internal_as_i<NodeType>];
  UNODB_DETAIL_ASSERT(growing_inode_counts[internal_as_i<NodeType>] >=
                      node_counts[as_i<NodeType>]);
}

template <typename Key, typename Value>
template <node_type NodeType>
constexpr void db<Key, Value>::account_shrinking_inode() noexcept {
  static_assert(NodeType != node_type::LEAF);

  ++shrinking_inode_counts[internal_as_i<NodeType>];
  UNODB_DETAIL_ASSERT(shrinking_inode_counts[internal_as_i<NodeType>] <=
                      growing_inode_counts[internal_as_i<NodeType>]);
}

#endif  // UNODB_DETAIL_WITH_STATS

template <typename Key, typename Value>
void db<Key, Value>::dump(std::ostream& os) const {
#ifdef UNODB_DETAIL_WITH_STATS
  os << "db dump, current memory use = " << get_current_memory_use() << '\n';
#else
  os << "db dump\n";
#endif  // UNODB_DETAIL_WITH_STATS
  art_policy::dump_node(os, root);
}

// LCOV_EXCL_START
template <typename Key, typename Value>
void db<Key, Value>::dump() const {
  dump(std::cerr);
}
// LCOV_EXCL_STOP

}  // namespace unodb

#endif  // UNODB_DETAIL_ART_HPP
