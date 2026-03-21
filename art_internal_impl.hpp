// Copyright 2019-2026 UnoDB contributors

/// \file
/// Implementation details for Adaptive Radix Tree (ART) internal nodes.
///
/// Provides node implementations (basic_inode_4, basic_inode_16,
/// basic_inode_48, basic_inode_256), leaf storage, key prefix compression, and
/// supporting infrastructure for both single-threaded and optimistic lock
/// coupling (OLC) variants. This header is not part of the public API.

#ifndef UNODB_DETAIL_ART_INTERNAL_IMPL_HPP
#define UNODB_DETAIL_ART_INTERNAL_IMPL_HPP

// Should be the first include
#include "global.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef UNODB_DETAIL_X86_64
#include <emmintrin.h>
#ifdef UNODB_DETAIL_AVX2
#include <immintrin.h>
#elif defined(UNODB_DETAIL_SSE4_2)
#include <smmintrin.h>
#endif
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "art_common.hpp"
#include "art_internal.hpp"
#include "assert.hpp"
#include "heap.hpp"
#include "node_type.hpp"
#include "portability_builtins.hpp"

namespace unodb {

template <typename Key, typename Value>
class db;

template <typename Key, typename Value>
class olc_db;

}  // namespace unodb

namespace unodb::detail {

/// A bitmask that tracks which child slots hold packed values rather than
/// pointers.  When \p Enabled is false the struct is empty and all operations
/// compile away, so inodes pay zero overhead for db types that do not use
/// value-in-slot.
template <bool Enabled, class Storage>
struct value_bitmask_field {
  Storage bits{};

  [[nodiscard]] constexpr bool test(std::uint8_t i) const noexcept {
    return (bits >> i) & 1U;
  }
  constexpr void set(std::uint8_t i) noexcept {
    bits |= static_cast<Storage>(Storage{1} << i);
  }
  constexpr void clear(std::uint8_t i) noexcept {
    bits &= static_cast<Storage>(~(Storage{1} << i));
  }
  /// Remove bit at position \p i, shifting higher bits down.
  constexpr void remove_at(std::uint8_t i) noexcept {
    const auto above = static_cast<Storage>(bits >> (i + 1));
    const auto below = static_cast<Storage>(bits & ((Storage{1} << i) - 1));
    bits = static_cast<Storage>(below | (above << i));
  }
};

/// Specialization for array-based bitmasks (I48 uses 6 bytes, I256 uses 32).
template <bool Enabled, class T, std::size_t N>
struct value_bitmask_field<Enabled, std::array<T, N>> {
  std::array<T, N> bits{};

  [[nodiscard]] constexpr bool test(std::uint8_t i) const noexcept {
    return (bits[static_cast<std::size_t>(i) / 8] >> (i % 8)) & 1U;
  }
  constexpr void set(std::uint8_t i) noexcept {
    bits[static_cast<std::size_t>(i) / 8] |= static_cast<T>(T{1} << (i % 8));
  }
  constexpr void clear(std::uint8_t i) noexcept {
    bits[static_cast<std::size_t>(i) / 8] &= static_cast<T>(~(T{1} << (i % 8)));
  }
};

/// Disabled specialization — empty struct, all ops are no-ops.
template <class Storage>
struct value_bitmask_field<false, Storage> {
  [[nodiscard]] static constexpr bool test(std::uint8_t) noexcept {
    return false;
  }
  static constexpr void set(std::uint8_t) noexcept {}
  static constexpr void clear(std::uint8_t) noexcept {}
  static constexpr void remove_at(std::uint8_t) noexcept {}
};

/// Disabled specialization for array-based bitmasks.
template <class T, std::size_t N>
struct value_bitmask_field<false, std::array<T, N>> {
  [[nodiscard]] static constexpr bool test(std::uint8_t) noexcept {
    return false;
  }
  static constexpr void set(std::uint8_t) noexcept {}
  static constexpr void clear(std::uint8_t) noexcept {}
  static constexpr void remove_at(std::uint8_t) noexcept {}
};

#ifdef UNODB_DETAIL_X86_64

/// Compare packed unsigned 8-bit integers for less-than-or-equal.
///
/// SSE helper implementing unsigned byte comparison (SSE only provides signed).
///
/// \param x First operand
/// \param y Second operand
/// \return Comparison mask where each byte is 0xFF if x <= y, else 0x00
///
/// \sa https://stackoverflow.com/a/32945715/80458
[[nodiscard, gnu::const]] inline auto _mm_cmple_epu8(__m128i x,
                                                     __m128i y) noexcept {
  return _mm_cmpeq_epi8(_mm_max_epu8(y, x), y);
}

#elif !defined(__aarch64__)

/// Check if 32-bit word \a v contains a zero byte.
///
/// \return Non-zero if \a v contains a zero byte
///
/// \sa https://graphics.stanford.edu/~seander/bithacks.html
[[nodiscard, gnu::const]] constexpr std::uint32_t has_zero_byte(
    std::uint32_t v) noexcept {
  return ((v - 0x01010101UL) & ~v & 0x80808080UL);
}

/// Check if 32-bit word \a v contains byte value \a b.
///
/// \return Non-zero if \a v contains byte \a b
[[nodiscard, gnu::const]] constexpr std::uint32_t contains_byte(
    std::uint32_t v, std::byte b) noexcept {
  return has_zero_byte(v ^ (~0U / 255 * static_cast<std::uint8_t>(b)));
}

#endif  // #ifdef UNODB_DETAIL_X86_64

/// Leaf node storing key-value pair.
///
/// Handles most leaf behavior for the index. Specialized for OLC which includes
/// additional information in the header. The leaf contains a copy of the key
/// and a copy of the value.
///
/// \tparam Key Key type (fixed-width integral or `key_view`)
/// \tparam Header Node header type (varies between plain and OLC variants)
//
// TODO(thompsonbry) Partial or no key in leaf.  Once we template for the Value
// type, we can optimize for u64 or smaller values, the leaf should be replaced
// by the use of a variant {Value,node_ptr} entry in the inode along with a bit
// mask to indicate for each position whether it is a leaf or a node.  This
// provides a significant optimization for secondary index use cases (tree
// height is reduced by one, no small allocations for leaves, the key is no
// longer explicitly stored, the key size is no longer stored, the value size is
// no longer stored, etc.).  However, we would still need to allocate a leaf for
// the case where Value is a std::span.  But this becomes a simple immutable
// data structure which exists solely to wrap the Value.
template <class Key, class Header>
class [[nodiscard]] basic_leaf final : public Header {
 public:
  /// A type alias determining the maximum size of a key that may be
  /// stored in the index.
  using key_size_type = unodb::key_size_type;

  /// A type alias determining the maximum size of a value that may be
  /// stored in the index.
  using value_size_type = unodb::value_size_type;

  /// The maximum size of any key in bytes.
  static constexpr std::size_t max_key_size =
      std::numeric_limits<key_size_type>::max();

  /// The maximum size of any value in bytes.
  static constexpr std::size_t max_value_size =
      std::numeric_limits<value_size_type>::max();

  /// Internal ART key type for this leaf.
  using art_key_type = basic_art_key<Key>;

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26485)

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)
  /// Construct leaf with key and value.
  ///
  /// \param k Key to store (will be copied)
  /// \param v Value to store (will be copied)
  constexpr basic_leaf(art_key_type k, value_view v) noexcept
      : key_size{static_cast<key_size_type>(k.size())},
        value_size{static_cast<value_size_type>(v.size())} {
    // Note: Runtime checks are handled upstream of this by
    // make_db_leaf_ptr().
    UNODB_DETAIL_ASSERT(k.size() <= max_key_size);
    UNODB_DETAIL_ASSERT(v.size() <= max_value_size);

    const auto tmp{k.get_key_view()};
    std::memcpy(data, tmp.data(), key_size);  // store encoded key
    if (!v.empty()) std::memcpy(data + key_size, v.data(), value_size);
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Return binary comparable key stored in leaf.
  ///
  /// \return ART key wrapping stored key bytes
  //
  // TODO(thompsonbry) : Partial or no key in leaf?
  [[nodiscard, gnu::pure]] constexpr auto get_key() const noexcept {
    if constexpr (std::is_same_v<Key, key_view>) {
      return art_key_type{get_key_view()};
    } else {
      // Use memcpy since alignment is not guaranteed because the
      // [key] is not an explicit part of the leaf data structure.
      //
      // TODO(thompsonbry) memory align leaf::data[0] to 8 bytes?
      Key u{};
      std::memcpy(&u, data, sizeof(u));
      // Note: The encoded key is stored in the leaf.  Since the
      // art_key constructor encodes the key, we need to decode the
      // key before calling the constructor.
      return art_key_type{bswap(u)};
    }
  }

  /// Return view onto key stored in leaf.
  ///
  /// \return Key view of stored key bytes
  //
  // TODO(thompsonbry) : Partial or no key in leaf?
  [[nodiscard, gnu::pure]] constexpr auto get_key_view() const noexcept {
    return key_view{data, key_size};
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Check if leaf key matches given key \a k.
  ///
  /// \return True if keys are equal
  //
  // TODO(thompsonbry) : Partial or no key in leaf?
  [[nodiscard, gnu::pure]] constexpr auto matches(
      // cppcheck-suppress passedByValue
      art_key_type k) const noexcept {
    return cmp(k) == 0;
  }

  /// Compare leaf key with given key \a k.
  ///
  /// \return Negative if this < \a k, positive if this > \a k, zero if equal
  //
  // TODO(thompsonbry) : Partial or no key in leaf?
  [[nodiscard, gnu::pure]] constexpr auto cmp(art_key_type k) const noexcept {
    return k.cmp(get_key_view());
  }

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26485)

  /// Return view onto value stored in leaf.
  ///
  /// \return Value view of stored value bytes
  [[nodiscard, gnu::pure]] constexpr auto get_value_view() const noexcept {
    return value_view{data + key_size, value_size};
  }

  /// Return value stored in leaf, converted to the given type.
  ///
  /// For value_view, returns a span over the stored bytes.
  /// For fixed-width types (e.g., uint64_t), deserializes from stored bytes.
  template <typename Value>
  [[nodiscard, gnu::pure]] constexpr auto get_value() const noexcept {
    if constexpr (std::is_same_v<Value, value_view>) {
      return get_value_view();
    } else {
      static_assert(std::is_trivially_copyable_v<Value>);
      Value v{};
      // cppcheck-suppress memsetClass
      std::memcpy(&v, data + key_size, sizeof(v));
      return v;
    }
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

#ifdef UNODB_DETAIL_WITH_STATS

  /// Return byte size of leaf data structure.
  ///
  /// \return Size in bytes
  [[nodiscard, gnu::pure]] constexpr auto get_size() const noexcept {
    return compute_size(key_size, value_size);
  }

#endif  // UNODB_DETAIL_WITH_STATS

  /// Dump leaf contents to stream for debugging.
  ///
  /// \param os Output stream
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os,
                                                bool /*recursive*/) const {
    os << ", ";
    ::unodb::detail::dump_key(os, get_key_view());
    os << ", ";
    ::unodb::detail::dump_val(os, get_value_view());
    os << '\n';
  }

  /// Compute required byte size of leaf for given key and value.
  ///
  /// \param key_size Size in bytes of key portion stored in leaf
  /// \param val_size Size in bytes of value stored in leaf
  /// \return Total allocation size in bytes
  [[nodiscard, gnu::const]] static constexpr auto compute_size(
      // cppcheck-suppress passedByValue
      key_size_type key_size, value_size_type val_size) noexcept {
    return sizeof(basic_leaf<Key, Header>) + key_size + val_size -
           1  // because of the [1] byte on the end of the struct.
        ;
  }

 private:
  /// The byte length of the key.
  const key_size_type key_size;
  /// The byte length of the value.
  const value_size_type value_size;
  /// The leaf's key and value data starts at data[0].  The key comes first
  /// followed by the data.
  //
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  std::byte data[1];
};  // class basic_leaf

/// Keyless leaf specialization.  Stores only the value — no key data,
/// no key access methods.  Used when can_eliminate_key_in_leaf is true
/// (full key is encoded in the inode path).
///
/// get_key(), get_key_view(), cmp(), and matches() are intentionally
/// absent.  Any call site that attempts to access the key from a keyless
/// leaf will produce a compile error.
template <class Header>
class [[nodiscard]] basic_leaf<no_key_tag, Header> final : public Header {
 public:
  using value_size_type = unodb::value_size_type;

  static constexpr std::size_t max_value_size =
      std::numeric_limits<value_size_type>::max();

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26485)
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)

  /// Construct keyless leaf with value only.
  constexpr explicit basic_leaf(value_view v) noexcept
      : value_size{static_cast<value_size_type>(v.size())} {
    UNODB_DETAIL_ASSERT(v.size() <= max_value_size);
    if (!v.empty()) std::memcpy(data, v.data(), value_size);
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Keyless leaf always matches — the key was verified by the inode path.
  template <typename ArtKey>
  [[nodiscard, gnu::pure]] constexpr auto matches(ArtKey /*k*/) const noexcept {
    return true;
  }

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26485)

  /// Return view onto value stored in leaf.
  [[nodiscard, gnu::pure]] constexpr auto get_value_view() const noexcept {
    return value_view{data, value_size};
  }

  /// Return value stored in leaf, converted to the given type.
  template <typename Value>
  [[nodiscard, gnu::pure]] constexpr auto get_value() const noexcept {
    if constexpr (std::is_same_v<Value, value_view>) {
      return get_value_view();
    } else {
      static_assert(std::is_trivially_copyable_v<Value>);
      Value v{};
      std::memcpy(&v, data, sizeof(v));
      return v;
    }
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

#ifdef UNODB_DETAIL_WITH_STATS
  [[nodiscard, gnu::pure]] constexpr auto get_size() const noexcept {
    return compute_size(value_size);
  }
#endif

  /// Dump keyless leaf contents to stream for debugging.
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os,
                                                bool /*recursive*/) const {
    os << ", (keyless), ";
    ::unodb::detail::dump_val(os, get_value_view());
    os << '\n';
  }

  /// Compute required byte size for a keyless leaf.
  [[nodiscard, gnu::const]] static constexpr auto compute_size(
      value_size_type val_size) noexcept {
    return sizeof(basic_leaf<no_key_tag, Header>) + val_size - 1;
  }

  /// Two-arg overload for compatibility with generic code.
  [[nodiscard, gnu::const]] static constexpr auto compute_size(
      // cppcheck-suppress passedByValue
      key_size_type /*key_size*/, value_size_type val_size) noexcept {
    return compute_size(val_size);
  }

 private:
  const value_size_type value_size;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  std::byte data[1];
};  // class basic_leaf<no_key_tag, Header>

/// Create unique pointer to new leaf with given key and value.
///
/// Allocates memory for leaf and constructs it with placement new.
///
/// \tparam Key Key type
/// \tparam Value Value type
/// \tparam Db Database template
///
/// \param k Key to store
/// \param v Value to store
/// \param db Database instance
///
/// \return Unique pointer to newly created leaf
///
/// \throws std::length_error if key or value exceeds maximum size
template <typename Key, typename Value, template <typename, typename> class Db>
[[nodiscard]] auto make_db_leaf_ptr(
    basic_art_key<Key> k, Value v,
    Db<Key, Value>& db UNODB_DETAIL_LIFETIMEBOUND) {
  using db_type = Db<Key, Value>;
  using header_type = typename db_type::header_type;
  using leaf_type = basic_leaf<leaf_key_type<Key, Value>, header_type>;

  if constexpr (!can_eliminate_key_in_leaf_v<Key, Value> &&
                std::is_same_v<Key, key_view>) {
    if (UNODB_DETAIL_UNLIKELY(k.size() > leaf_type::max_key_size)) {
      throw std::length_error("Key length must fit in std::uint32_t");
    }
  }

  // Serialize value to bytes for leaf storage.
  value_view leaf_val_bytes;
  [[maybe_unused]] std::byte val_buf[sizeof(Value)];
  if constexpr (std::is_same_v<Value, value_view>) {
    leaf_val_bytes = v;
  } else {
    static_assert(std::is_trivially_copyable_v<Value>);
    std::memcpy(val_buf, &v, sizeof(v));
    leaf_val_bytes = value_view{val_buf, sizeof(v)};
  }

  if (UNODB_DETAIL_UNLIKELY(leaf_val_bytes.size_bytes() >
                            leaf_type::max_value_size)) {
    throw std::length_error("Value length must fit in std::uint32_t");
  }

  std::size_t size;
  if constexpr (can_eliminate_key_in_leaf_v<Key, Value>) {
    size = leaf_type::compute_size(
        static_cast<typename leaf_type::value_size_type>(
            leaf_val_bytes.size_bytes()));
  } else {
    size = leaf_type::compute_size(
        static_cast<typename leaf_type::key_size_type>(k.size()),
        static_cast<typename leaf_type::value_size_type>(
            leaf_val_bytes.size_bytes()));
  }

  auto* const leaf_mem = static_cast<std::byte*>(
      allocate_aligned(size, alignment_for_new<leaf_type>()));

#ifdef UNODB_DETAIL_WITH_STATS
  db.increment_leaf_count(size);
#endif  // UNODB_DETAIL_WITH_STATS

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26402)
  return basic_db_leaf_unique_ptr<Key, Value, header_type, Db>{
      [&]() {
        if constexpr (can_eliminate_key_in_leaf_v<Key, Value>) {
          return new (leaf_mem) leaf_type{leaf_val_bytes};
        } else {
          return new (leaf_mem) leaf_type{k, leaf_val_bytes};
        }
      }(),
      basic_db_leaf_deleter<db_type>{db}};
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
}

/// Metaprogramming struct listing all concrete internal node types.
///
/// \tparam INode Base internal node type
/// \tparam Node4 Node type for 2-4 children
/// \tparam Node16 Node type for 5-16 children
/// \tparam Node48 Node type for 17-48 children
/// \tparam Node256 Node type for 49-256 children
template <class INode, class Node4, class Node16, class Node48, class Node256>
struct basic_inode_def final {
  /// Base internal node type.
  using inode = INode;
  /// Node type for 2-4 children.
  using n4 = Node4;
  /// Node type for 5-16 children.
  using n16 = Node16;
  /// Node type for 17-48 children.
  using n48 = Node48;
  /// Node type for 49-256 children.
  using n256 = Node256;

  /// Check if type is one of the internal node types.
  ///
  /// \tparam Node Type to check
  /// \return True if \a Node is an internal node type
  template <class Node>
  [[nodiscard]] static constexpr bool is_inode() noexcept {
    return std::is_same_v<Node, n4> || std::is_same_v<Node, n16> ||
           std::is_same_v<Node, n48> || std::is_same_v<Node, n256>;
  }

  /// Instances cannot be created
  basic_inode_def() = delete;
};

// Implementation of deleters declared in art_internal.hpp

template <class Db>
inline void basic_db_leaf_deleter<Db>::operator()(
    leaf_type* to_delete) const noexcept {
#ifdef UNODB_DETAIL_WITH_STATS
  const auto leaf_size = to_delete->get_size();
#endif  // UNODB_DETAIL_WITH_STATS

  free_aligned(to_delete);

#ifdef UNODB_DETAIL_WITH_STATS
  db.decrement_leaf_count(leaf_size);
#endif  // UNODB_DETAIL_WITH_STATS
}

template <class INode, class Db>
inline void basic_db_inode_deleter<INode, Db>::operator()(
    INode* inode_ptr) noexcept {
  static_assert(std::is_trivially_destructible_v<INode>);

  free_aligned(inode_ptr);

#ifdef UNODB_DETAIL_WITH_STATS
  db.template decrement_inode_count<INode>();
#endif  // UNODB_DETAIL_WITH_STATS
}

/// Policy class encapsulating ART implementation differences.
///
/// Encapsulates differences between plain and OLC ART, such as extra header
/// field and node access critical section type.
///
/// \tparam Key Key type
/// \tparam Value Value type
/// \tparam Db Database template
/// \tparam CriticalSectionPolicy Critical section wrapper policy
/// \tparam LockPolicy Lock acquisition policy
/// \tparam ReadCriticalSection Read critical section type
/// \tparam NodePtr Tagged node pointer type
/// \tparam INodeDefs Internal node definitions template
/// \tparam INodeReclamator Internal node reclamation policy
/// \tparam LeafReclamator Leaf reclamation policy
template <typename Key, typename Value, template <typename, typename> class Db,
          template <class> class CriticalSectionPolicy, class LockPolicy,
          class ReadCriticalSection, class NodePtr,
          template <typename, typename> class INodeDefs,
          template <typename, typename, class> class INodeReclamator,
          template <class> class LeafReclamator>
struct basic_art_policy final {
  /// \name Type aliases
  /// \{

  /// Key type.
  using key_type = Key;

  /// Value type.
  using value_type = Value;

  /// Internal ART key type.
  using art_key_type = basic_art_key<Key>;

  /// Tagged node pointer type.
  using node_ptr = NodePtr;

  /// Node header type.
  using header_type = typename NodePtr::header_type;

  /// Lock acquisition policy.
  using lock_policy = LockPolicy;

  /// Read critical section.
  using read_critical_section = ReadCriticalSection;

  /// Internal node definitions.
  using inode_defs = INodeDefs<Key, Value>;

  /// Base internal node type.
  using inode = typename inode_defs::inode;

  /// Node type for 2-4 children.
  using inode4_type = typename inode_defs::n4;

  /// Node type for 5-16 children.
  using inode16_type = typename inode_defs::n16;

  /// Node type for 17-48 children.
  using inode48_type = typename inode_defs::n48;

  /// Node type for 49-256 children.
  using inode256_type = typename inode_defs::n256;

  /// Tree depth wrapper.
  using tree_depth_type = tree_depth<art_key_type>;

  /// Whether values are stored directly in inode child slots rather than
  /// in separate leaf nodes.  True when the value fits in a uint64_t.
  static constexpr bool value_in_slot =
      (sizeof(Value) <= sizeof(std::uint64_t));
  static_assert(sizeof(std::uintptr_t) <= sizeof(std::uint64_t),
                "node_ptr must fit in a uint64_t slot");

  /// Whether the full key is encoded in the inode path (prefix + dispatch
  /// bytes at every level).  True for key_view keys with small values.
  static constexpr bool full_key_in_inode_path = std::is_same_v<Key, key_view>;

  /// Whether the key can be omitted from the leaf.
  static constexpr bool can_eliminate_key_in_leaf =
      can_eliminate_key_in_leaf_v<Key, Value>;

  /// Whether leaf allocation can be eliminated entirely.  Requires the
  /// full key in the inode path AND the value in the inode child slot.
  static constexpr bool can_eliminate_leaf =
      full_key_in_inode_path && value_in_slot;

  /// Sentinel XOR'd into packed values to ensure they are never nullptr.
  /// Any non-zero constant works; using a pattern unlikely to collide
  /// with valid pointer alignment.
  static constexpr std::uintptr_t pack_xor_sentinel = 0x8000000000000001ULL;

  /// Pack a value into a node_ptr slot (value-in-slot mode).
  /// The parent inode's value_bitmask distinguishes this from a pointer.
  [[nodiscard]] static node_ptr pack_value(Value v) noexcept {
    static_assert(can_eliminate_leaf);
    std::uint64_t raw{};
    static_assert(sizeof(v) <= sizeof(raw));
    std::memcpy(&raw, &v, sizeof(v));
    raw ^= pack_xor_sentinel;
    node_ptr result{nullptr};
    std::memcpy(static_cast<void*>(&result), &raw, sizeof(raw));
    return result;
  }

  /// Extract a value from a node_ptr slot (value-in-slot mode).
  [[nodiscard]] static Value unpack_value(node_ptr n) noexcept {
    static_assert(can_eliminate_leaf);
    auto raw = n.raw_val() ^ pack_xor_sentinel;
    Value v{};
    std::memcpy(&v, &raw, sizeof(v));
    return v;
  }

  /// Leaf type — no_leaf_tag when can_eliminate_leaf (no leaf nodes in tree).
  using leaf_type =
      std::conditional_t<can_eliminate_leaf, no_leaf_tag,
                         basic_leaf<leaf_key_type<Key, Value>, header_type>>;

  /// Database type.
  using db_type = Db<Key, Value>;

 private:
  /// Internal node deleter type.
  template <class INode>
  using db_inode_deleter = basic_db_inode_deleter<INode, db_type>;

  /// Leaf pointer with reclamation policy.
  using leaf_reclaimable_ptr =
      std::unique_ptr<leaf_type, LeafReclamator<db_type>>;

 public:
  /// Critical section policy wrapper.
  template <typename T>
  using critical_section_policy = CriticalSectionPolicy<T>;

  /// \}

  /// \name Smart pointer aliases
  /// \{

  /// Unique pointer to internal node with deleter.
  template <class INode>
  using db_inode_unique_ptr = std::unique_ptr<INode, db_inode_deleter<INode>>;

  /// Unique pointer to basic_inode_4.
  using db_inode4_unique_ptr = db_inode_unique_ptr<inode4_type>;

  /// Unique pointer to basic_inode_16.
  using db_inode16_unique_ptr = db_inode_unique_ptr<inode16_type>;

  /// Unique pointer to basic_inode_48.
  using db_inode48_unique_ptr = db_inode_unique_ptr<inode48_type>;

  /// Unique pointer to basic_inode_256.
  using db_inode256_unique_ptr = db_inode_unique_ptr<inode256_type>;

  /// Unique pointer to internal node for deferred reclamation.
  template <class INode>
  using db_inode_reclaimable_ptr =
      std::unique_ptr<INode, INodeReclamator<Key, Value, INode>>;

  /// Unique pointer to leaf.
  using db_leaf_unique_ptr =
      basic_db_leaf_unique_ptr<key_type, value_type, header_type, Db>;

  /// \}

  /// \name Factory methods
  /// \{

  /// Create new leaf with given key and value.
  ///
  /// \param k Key to store in leaf
  /// \param v Value to store in leaf
  /// \param db_instance Database for memory tracking
  ///
  /// \return Unique pointer to newly allocated leaf
  [[nodiscard]] static auto make_db_leaf_ptr(
      art_key_type k, value_type v,
      db_type& db_instance UNODB_DETAIL_LIFETIMEBOUND) {
    static_assert(
        !can_eliminate_leaf,
        "make_db_leaf_ptr must not be called when leaf is eliminated");
    return ::unodb::detail::make_db_leaf_ptr<Key, Value, Db>(k, v, db_instance);
  }

  /// Create reclaimable pointer to leaf for deferred deletion.
  ///
  /// \param leaf Leaf pointer to wrap
  /// \param db_instance Database for reclamation
  ///
  /// \return Reclaimable pointer that defers leaf deletion
  [[nodiscard]] static auto reclaim_leaf_on_scope_exit(
      leaf_type* leaf UNODB_DETAIL_LIFETIMEBOUND,
      db_type& db_instance UNODB_DETAIL_LIFETIMEBOUND) noexcept {
    static_assert(!can_eliminate_leaf,
                  "reclaim_leaf_on_scope_exit must not be called when leaf is "
                  "eliminated");
    return leaf_reclaimable_ptr{leaf, LeafReclamator<db_type>{db_instance}};
  }

  /// Reclaim a child node if it is a leaf; ignore if it is an inode.
  ///
  /// Used by shrink constructors where the deleted child may be a
  /// dead chain inode (already freed) rather than a leaf.  The type
  /// tag is embedded in the pointer value, so checking type() is safe
  /// even if the pointed-to memory has been freed.
  ///
  /// \param child Tagged node pointer to conditionally reclaim
  /// \param db_instance Database for memory reclamation
  /// \return Reclaimable pointer (no-op if child is not a leaf)
  [[nodiscard]] static auto reclaim_if_leaf(
      node_ptr child,
      db_type& db_instance UNODB_DETAIL_LIFETIMEBOUND) noexcept {
    if constexpr (can_eliminate_leaf) {
      struct noop_guard {};
      return noop_guard{};
    } else {
      if (child.type() == node_type::LEAF) {
        return leaf_reclaimable_ptr{child.template ptr<leaf_type*>(),
                                    LeafReclamator<db_type>{db_instance}};
      }
      return leaf_reclaimable_ptr{nullptr,
                                  LeafReclamator<db_type>{db_instance}};
    }
  }

  /// Create new internal node.
  ///
  /// \tparam INode Internal node type
  /// \tparam Args Constructor argument types
  ///
  /// \param db_instance Database instance
  /// \param args Constructor arguments forwarded to node
  ///
  /// \return Unique pointer to newly constructed node
  UNODB_DETAIL_DISABLE_GCC_11_WARNING("-Wmismatched-new-delete")
  template <class INode, class... Args>
  [[nodiscard]] static auto make_db_inode_unique_ptr(
      db_type& db_instance UNODB_DETAIL_LIFETIMEBOUND, Args&&... args) {
    auto* const inode_mem = static_cast<std::byte*>(
        allocate_aligned(sizeof(INode), alignment_for_new<INode>()));

#ifdef UNODB_DETAIL_WITH_STATS
    db_instance.template increment_inode_count<INode>();
#endif  // UNODB_DETAIL_WITH_STATS

    return db_inode_unique_ptr<INode>{
        new (inode_mem) INode{db_instance, std::forward<Args>(args)...},
        db_inode_deleter<INode>{db_instance}};
  }
  UNODB_DETAIL_RESTORE_GCC_11_WARNINGS()

  /// Wrap existing internal node pointer in unique pointer.
  ///
  /// \tparam INode Internal node type
  ///
  /// \param inode_ptr Existing node pointer to wrap
  /// \param db_instance Database for deleter
  ///
  /// \return Unique pointer owning the node
  template <class INode>
  [[nodiscard]] static auto make_db_inode_unique_ptr(
      INode* inode_ptr UNODB_DETAIL_LIFETIMEBOUND,
      db_type& db_instance UNODB_DETAIL_LIFETIMEBOUND) noexcept {
    return db_inode_unique_ptr<INode>{inode_ptr,
                                      db_inode_deleter<INode>{db_instance}};
  }

  /// Create reclaimable pointer to internal node for deferred deletion.
  ///
  /// \tparam INode Internal node type
  ///
  /// \param inode_ptr Node pointer to wrap
  /// \param db_instance Database for reclamation
  ///
  /// \return Reclaimable pointer that defers node deletion
  template <class INode>
  [[nodiscard]] static auto make_db_inode_reclaimable_ptr(
      INode* inode_ptr UNODB_DETAIL_LIFETIMEBOUND,
      db_type& db_instance UNODB_DETAIL_LIFETIMEBOUND) noexcept {
    return db_inode_reclaimable_ptr<INode>{
        inode_ptr, INodeReclamator<Key, Value, INode>{db_instance}};
  }

 private:
  /// Wrap raw leaf pointer in unique_ptr with deleter.
  ///
  /// \param leaf Existing leaf pointer to wrap
  /// \param db_instance Database for deleter
  ///
  /// \return Unique pointer owning the leaf
  [[nodiscard]] static auto make_db_leaf_ptr(
      leaf_type* leaf UNODB_DETAIL_LIFETIMEBOUND,
      db_type& db_instance UNODB_DETAIL_LIFETIMEBOUND) noexcept
    requires(!can_eliminate_leaf)
  {
    return basic_db_leaf_unique_ptr<key_type, value_type, header_type, Db>{
        leaf, basic_db_leaf_deleter<db_type>{db_instance}};
  }

  /// \}

  /// RAII helper deleting node on scope exit.
  struct delete_db_node_ptr_at_scope_exit final {
    /// Construct guard for node \a node_ptr_ deletion from \a db_.
    constexpr explicit delete_db_node_ptr_at_scope_exit(
        NodePtr node_ptr_ UNODB_DETAIL_LIFETIMEBOUND,
        db_type& db_ UNODB_DETAIL_LIFETIMEBOUND) noexcept
        : node_ptr{node_ptr_}, db{db_} {}

    /// Delete the node based on its type.
    // MSVC C26815 false positive: the destructor creates unique_ptrs whose
    // deleters hold references to db. These are immediately destroyed while db
    // (a member reference) is still valid - the constructor's LIFETIMEBOUND
    // guarantees the referent outlives this object.
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
    ~delete_db_node_ptr_at_scope_exit() noexcept {
      switch (node_ptr.type()) {
        case node_type::LEAF: {
          if constexpr (!can_eliminate_leaf) {
            const auto r{
                make_db_leaf_ptr(node_ptr.template ptr<leaf_type*>(), db)};
          }
          return;
        }
        case node_type::I4: {
          // cppcheck-suppress throwInNoexceptFunction
          const auto r{make_db_inode_unique_ptr(
              node_ptr.template ptr<inode4_type*>(), db)};
          return;
        }
        case node_type::I16: {
          const auto r{make_db_inode_unique_ptr(
              node_ptr.template ptr<inode16_type*>(), db)};
          return;
        }
        case node_type::I48: {
          const auto r{make_db_inode_unique_ptr(
              node_ptr.template ptr<inode48_type*>(), db)};
          return;
        }
        case node_type::I256: {
          const auto r{make_db_inode_unique_ptr(
              node_ptr.template ptr<inode256_type*>(), db)};
          return;
        }
      }
      UNODB_DETAIL_CANNOT_HAPPEN();  // LCOV_EXCL_LINE
    }
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

    /// Non-copyable.
    delete_db_node_ptr_at_scope_exit(const delete_db_node_ptr_at_scope_exit&) =
        delete;

    /// Non-movable.
    delete_db_node_ptr_at_scope_exit(delete_db_node_ptr_at_scope_exit&&) =
        delete;

    /// Non-copy-assignable.
    auto& operator=(const delete_db_node_ptr_at_scope_exit&) = delete;

    /// Non-move-assignable.
    auto& operator=(delete_db_node_ptr_at_scope_exit&&) = delete;

   private:
    /// Node to delete on destruction.
    const NodePtr node_ptr;

    /// Database reference for memory tracking.
    db_type& db;
  };

 public:
  /// \name Tree operations
  /// \{

  /// Recursively delete node and all its descendants.
  ///
  /// \param node Root of subtree to delete
  /// \param db_instance Database instance
  static void delete_subtree(NodePtr node, db_type& db_instance) noexcept {
    delete_db_node_ptr_at_scope_exit delete_on_scope_exit{node, db_instance};

    switch (node.type()) {
      case node_type::LEAF:
        return;
      case node_type::I4: {
        auto* const subtree_ptr{node.template ptr<inode4_type*>()};
        subtree_ptr->delete_subtree(db_instance);
        return;
      }
      case node_type::I16: {
        auto* const subtree_ptr{node.template ptr<inode16_type*>()};
        subtree_ptr->delete_subtree(db_instance);
        return;
      }
      case node_type::I48: {
        auto* const subtree_ptr{node.template ptr<inode48_type*>()};
        subtree_ptr->delete_subtree(db_instance);
        return;
      }
      case node_type::I256: {
        auto* const subtree_ptr{node.template ptr<inode256_type*>()};
        subtree_ptr->delete_subtree(db_instance);
        return;
      }
    }
  }

  /// Dump node contents to stream for debugging.
  ///
  /// \param os Output stream
  /// \param node Node to dump
  /// \param recursive If true, also dump child nodes
  [[gnu::cold]] UNODB_DETAIL_NOINLINE static void dump_node(
      std::ostream& os, const NodePtr& node, bool recursive = true) {
    os << "node at: " << node.template ptr<void*>() << ", tagged ptr = 0x"
       << std::hex << node.raw_val() << std::dec;
    if (node == nullptr) {
      os << '\n';
      return;
    }
    os << ", type = ";
    switch (node.type()) {
      case node_type::LEAF:
        os << "LEAF";
        if constexpr (!can_eliminate_leaf) {
          node.template ptr<leaf_type*>()->dump(os, recursive);
        }
        break;
      case node_type::I4:
        os << "I4";
        node.template ptr<inode4_type*>()->dump(os, recursive);
        break;
      case node_type::I16:
        os << "I16";
        node.template ptr<inode16_type*>()->dump(os, recursive);
        break;
      case node_type::I48:
        os << "I48";
        node.template ptr<inode48_type*>()->dump(os, recursive);
        break;
      case node_type::I256:
        os << "I256";
        node.template ptr<inode256_type*>()->dump(os, recursive);
        break;
    }
  }

  /// \}

  /// Not instantiable.
  basic_art_policy() = delete;
};  // class basic_art_policy

/// Size type for key prefix length.
using key_prefix_size = std::uint8_t;

/// Maximum number of bytes in key prefix.
static constexpr key_prefix_size key_prefix_capacity = 7;

/// Key prefix snapshot for iterator use.
///
/// Exposes a `key_view` over internal data that was atomically copied from
/// a `key_prefix`. Direct use of `key_prefix` is not possible because:
/// (a) thread safety concerns; and (b) `key_view` is a non-owned view.
union [[nodiscard]] key_prefix_snapshot {
  // TODO(thompsonbry) Can this be replaced by [using
  // key_prefix_snapshot = key_prefix<std::uint64_t,
  // in_fake_critical_section>]?  We need the snapshot to be
  // atomic. Does that work with this using decl? [it does not work
  // trivially.]
 private:
  /// Storage type for prefix bytes.
  using key_prefix_data = std::array<std::byte, key_prefix_capacity>;

  /// Structured view of prefix data.
  struct [[nodiscard]] inode_fields {
    /// The prefix bytes.
    key_prefix_data key_prefix;
    /// Number of bytes in prefix.
    key_prefix_size key_prefix_length;
  };

  /// Structured access to prefix.
  inode_fields f;
  /// Raw 64-bit access for atomic operations.
  std::uint64_t u64;

 public:
  /// Construct from raw 64-bit value.
  ///
  /// \param v Raw value containing prefix bytes and length
  constexpr explicit key_prefix_snapshot(std::uint64_t v) noexcept : u64(v) {}

  /// Return view onto snapshot of key prefix.
  ///
  /// \return Key view of stored prefix bytes
  [[nodiscard]] constexpr key_view get_key_view() const noexcept {
    return key_view(f.key_prefix.data(), f.key_prefix_length);
  }

  /// Return number of prefix bytes.
  ///
  /// \return Prefix length
  [[nodiscard]] constexpr key_prefix_size length() const noexcept {
    return f.key_prefix_length;
  }

  /// Return shared prefix length with shifted key.
  ///
  /// Computes bytes in common between this prefix and the next 64 bits (max)
  /// of the shifted key from which leading matched bytes have been discarded.
  ///
  /// \param shifted_key_u64 Shifted key value to compare against
  /// \return Number of common bytes
  [[nodiscard]] constexpr auto get_shared_length(
      std::uint64_t shifted_key_u64) const noexcept {
    return shared_len(shifted_key_u64, u64, length());
  }

  /// Return byte at specified index.
  ///
  /// \param i Index (must be less than length())
  /// \return Byte at position \a i
  [[nodiscard]] constexpr auto operator[](std::size_t i) const noexcept {
    UNODB_DETAIL_ASSERT(i < length());
    return f.key_prefix[i];
  }

  /// Compute shared prefix length between two 64-bit key values.
  ///
  /// \param k1 First key as u64 (from get_u64)
  /// \param k2 Second key as u64 (from get_u64)
  /// \param clamp_byte_pos Maximum prefix length to consider
  /// \return Number of leading bytes in common (at most \a clamp_byte_pos)
  [[nodiscard, gnu::const]] static constexpr unsigned shared_len(
      std::uint64_t k1, std::uint64_t k2, unsigned clamp_byte_pos) noexcept {
    UNODB_DETAIL_ASSERT(clamp_byte_pos < 8);

    const auto diff = k1 ^ k2;
    const auto clamped = diff | (1ULL << (clamp_byte_pos * 8U));
    return static_cast<unsigned>(std::countr_zero(clamped) >> 3U);
  }
};  // class key_prefix_snapshot
static_assert(sizeof(key_prefix_snapshot) == sizeof(std::uint64_t));

/// Key prefix for prefix compression in internal nodes.
///
/// Stores zero or more bytes that are common prefix shared by all children
/// of a node, supporting prefix compression in the index.
///
/// \tparam ArtKey Internal ART key type
/// \tparam CriticalSectionPolicy Policy for thread-safe access
template <typename ArtKey, template <class> class CriticalSectionPolicy>
union [[nodiscard]] key_prefix {
 private:
  /// Critical section wrapper for type T.
  template <typename T>
  using critical_section_policy = CriticalSectionPolicy<T>;

  /// Storage type for prefix bytes with thread-safety wrapper.
  using key_prefix_data =
      std::array<critical_section_policy<std::byte>, key_prefix_capacity>;

  /// Structured view of prefix data.
  struct [[nodiscard]] inode_fields {
    /// Prefix bytes.
    key_prefix_data key_prefix;
    /// Length.
    critical_section_policy<key_prefix_size> key_prefix_length;
  };

  /// Structured access to prefix.
  inode_fields f;
  /// Raw 64-bit access.
  critical_section_policy<std::uint64_t> u64;

 public:
  /// Construct from two keys sharing common prefix.
  ///
  /// \param k1 First key view
  /// \param shifted_k2 Second key already shifted
  /// \param depth Current tree depth
  key_prefix(key_view k1, ArtKey shifted_k2, tree_depth<ArtKey> depth) noexcept
      : u64{make_u64(k1, shifted_k2, depth)} {}

  /// Construct with explicit prefix length, copying bytes from k1 at depth.
  key_prefix(detail::key_prefix_size prefix_len, key_view k1,
             tree_depth<ArtKey> depth) noexcept
      : u64{make_u64_explicit(prefix_len, k1, depth)} {}

  /// Construct with truncated length from source.
  ///
  /// \param key_prefix_len New prefix length (must not exceed capacity)
  /// \param source_key_prefix Source to copy bytes from
  key_prefix(unsigned key_prefix_len,
             const key_prefix& source_key_prefix) noexcept
      : u64{(source_key_prefix.u64 & key_bytes_mask) |
            length_to_word(key_prefix_len)} {
    UNODB_DETAIL_ASSERT(key_prefix_len <= key_prefix_capacity);
  }

  /// Copy constructor.
  key_prefix(const key_prefix& other) noexcept : u64{other.u64.load()} {}

  /// Destructor.
  ~key_prefix() noexcept = default;

  /// Return shared prefix length with shifted key.
  ///
  /// \param shifted_key Shifted key to compare
  /// \return Number of common bytes
  [[nodiscard]] constexpr auto get_shared_length(
      ArtKey shifted_key) const noexcept {
    return get_shared_length(shifted_key.get_u64());
  }

  /// Return shared prefix length with raw key value.
  ///
  /// \param shifted_key_u64 Shifted key as 64-bit integer
  /// \return Number of common bytes
  [[nodiscard]] constexpr auto get_shared_length(
      std::uint64_t shifted_key_u64) const noexcept {
    return shared_len(shifted_key_u64, u64, length());
  }

  /// Return atomic snapshot of prefix data.
  ///
  /// \return Snapshot for consistent access in OLC iterators
  ///
  /// \note Required when consistent view of prefix state is needed, e.g., for
  /// OLC iterator to verify RCS validity after obtaining data.
  [[nodiscard]] constexpr key_prefix_snapshot get_snapshot() const noexcept {
    return key_prefix_snapshot(u64);
  }

  /// Return number of prefix bytes.
  ///
  /// \return Prefix length
  [[nodiscard]] constexpr key_prefix_size length() const noexcept {
    const auto result = f.key_prefix_length.load();
    UNODB_DETAIL_ASSERT(result <= key_prefix_capacity);
    return result;
  }

  /// Remove leading bytes from prefix.
  ///
  /// \param cut_len Number of bytes to remove (must be positive and <= length)
  constexpr void cut(key_prefix_size cut_len) noexcept {
    UNODB_DETAIL_ASSERT(cut_len > 0);
    UNODB_DETAIL_ASSERT(cut_len <= length());

    u64 = ((u64 >> (cut_len * 8)) & key_bytes_mask) |
          length_to_word(static_cast<key_prefix_size>(length() - cut_len));

    UNODB_DETAIL_ASSERT(f.key_prefix_length.load() <= key_prefix_capacity);
  }

  /// Prepend prefix and single byte to current prefix.
  ///
  /// Result is: \a prefix1 + \a prefix2 + current_prefix.
  ///
  /// \param prefix1 Prefix to prepend at start
  /// \param prefix2 Single byte between \a prefix1 and current prefix
  constexpr void prepend(const key_prefix& prefix1,
                         std::byte prefix2) noexcept {
    UNODB_DETAIL_ASSERT(length() + prefix1.length() < key_prefix_capacity);

    const auto prefix1_bit_length = prefix1.length() * 8U;
    const auto prefix1_mask = (1ULL << prefix1_bit_length) - 1;
    const auto prefix3_bit_length = length() * 8U;
    const auto prefix3_mask = (1ULL << prefix3_bit_length) - 1;
    const auto prefix3 = u64 & prefix3_mask;
    const auto shifted_prefix3 = prefix3 << (prefix1_bit_length + 8U);
    const auto shifted_prefix2 = static_cast<std::uint64_t>(prefix2)
                                 << prefix1_bit_length;
    const auto masked_prefix1 = prefix1.u64 & prefix1_mask;

    u64 = shifted_prefix3 | shifted_prefix2 | masked_prefix1 |
          length_to_word(length() + prefix1.length() + 1U);

    UNODB_DETAIL_ASSERT(f.key_prefix_length.load() <= key_prefix_capacity);
  }

  /// Return byte at specified index.
  ///
  /// \param i Index (must be less than length())
  /// \return Byte at position \a i
  [[nodiscard]] constexpr auto operator[](std::size_t i) const noexcept {
    UNODB_DETAIL_ASSERT(i < length());
    return f.key_prefix[i].load();
  }

  /// Dump prefix contents to stream for debugging.
  ///
  /// \param os Output stream
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os) const {
    const auto len = length();
    os << ", prefix(" << len << ")";
    if (len > 0) {
      os << ": 0x";
      for (std::size_t i = 0; i < len; ++i) dump_byte(os, f.key_prefix[i]);
    }
  }

  /// Non-movable.
  key_prefix(key_prefix&&) = delete;

  /// Non-copy-assignable.
  key_prefix& operator=(const key_prefix&) = delete;

  /// Non-move-assignable.
  key_prefix& operator=(key_prefix&&) = delete;

 private:
  /// Mask for extracting key bytes from 64-bit word.
  static constexpr auto key_bytes_mask = 0x00FF'FFFF'FFFF'FFFFULL;

  /// Convert length to 64-bit word with length in high byte.
  [[nodiscard, gnu::const]] static constexpr std::uint64_t length_to_word(
      unsigned length) {
    UNODB_DETAIL_ASSERT(length <= key_prefix_capacity);
    return static_cast<std::uint64_t>(length) << 56U;
  }

  /// Compute shared prefix length between two 64-bit values.
  [[nodiscard, gnu::const]] static constexpr unsigned shared_len(
      std::uint64_t k1, std::uint64_t k2, unsigned clamp_byte_pos) noexcept {
    UNODB_DETAIL_ASSERT(clamp_byte_pos < 8);

    const auto diff = k1 ^ k2;
    const auto clamped = diff | (1ULL << (clamp_byte_pos * 8U));
    return static_cast<unsigned>(std::countr_zero(clamped) >> 3U);
  }

  /// Create packed u64 representation from two keys at given depth.
  ///
  /// \param k1 First key view
  /// \param shifted_k2 Second key (already shifted)
  /// \param depth Current tree depth
  ///
  /// \return Packed u64 containing shared prefix bytes and length
  [[nodiscard, gnu::const]] static constexpr std::uint64_t make_u64(
      key_view k1, ArtKey shifted_k2, tree_depth<ArtKey> depth) noexcept {
    k1 = k1.subspan(depth);  // shift_right(depth)

    const auto k1_u64 = get_u64(k1) & key_bytes_mask;

    return k1_u64 | length_to_word(shared_len(k1_u64, shifted_k2.get_u64(),
                                              key_prefix_capacity));
  }

  /// Build u64 with explicit prefix length, copying bytes from k1 at depth.
  [[nodiscard, gnu::const]] static constexpr std::uint64_t make_u64_explicit(
      detail::key_prefix_size prefix_len, key_view k1,
      tree_depth<ArtKey> depth) noexcept {
    k1 = k1.subspan(depth);
    const auto k1_u64 = get_u64(k1) & key_bytes_mask;
    return k1_u64 | length_to_word(prefix_len);
  }
};  // class key_prefix

/// Iterator traversal result representing tree path element.
///
/// Returned by iterator visitation pattern to represent a position in the tree.
///
/// \tparam NodeHeader Node header type
template <class NodeHeader>
struct iter_result {
  /// Node pointer type.
  using node_ptr = basic_node_ptr<NodeHeader>;

  /// Node pointer (internal or leaf).
  node_ptr node;

  /// Key byte consumed at this level when stepping down to the child node. For
  /// basic_inode_48 and basic_inode_256, this equals the child index; for
  /// basic_inode_4 and basic_inode_16 it differs due to sparse key encoding.
  /// Explicit representation avoids searching for key byte in basic_inode_48
  /// and basic_inode_256 cases.
  std::byte key_byte;

  /// Child index within parent node (except for basic_inode_48, where it
  /// indexes into basic_inode_48::child_indexes and equals key_byte). Overflow
  /// for child_index can occur for basic_inode_48 and basic_inode_256. When
  /// overflow happens, the iter_result is undefined and the wrapping
  /// std::optional returns false.
  std::uint8_t child_index;

  /// Snapshot of key prefix for node.
  key_prefix_snapshot prefix;

  /// True when this entry represents a packed value (value-in-slot) rather
  /// than an inode or leaf pointer.  Used by the iterator to distinguish
  /// value-in-slot entries from regular children whose child_index happens
  /// to equal 0xFF.
  bool packed_leaf{false};
};

/// Optional wrapper for iter_result.
///
/// \tparam NodeHeader Node header type
template <class NodeHeader>
using iter_result_opt = std::optional<iter_result<NodeHeader>>;

/// Sentinel type for template arguments in basic_inode.
///
/// Used as the larger node type for basic_inode_256 and smaller node type for
/// basic_inode_4.
class fake_inode final {
 public:
  /// Not instantiable.
  fake_inode() = delete;
};

/// Base class for all internal node types.
///
/// Extends common header and defines methods shared by all internal nodes.
/// The header type is specific to thread-safety policy: for OLC, it includes
/// lock and version tag metadata.
///
/// Contains generic internal node code: key prefix, children count, and
/// dispatch for add/remove/find operations to specific node types.
///
/// \tparam ArtPolicy Policy class defining types and operations
template <class ArtPolicy>
class basic_inode_impl : public ArtPolicy::header_type {
 public:
  /// \name Type aliases
  /// \{

  /// Key type.
  using key_type = typename ArtPolicy::key_type;
  /// Value type.
  using value_type = typename ArtPolicy::value_type;
  /// Internal key type.
  using art_key_type = typename ArtPolicy::art_key_type;
  /// Node pointer type.
  using node_ptr = typename ArtPolicy::node_ptr;

  /// Critical section policy wrapper.
  template <typename T>
  using critical_section_policy =
      typename ArtPolicy::template critical_section_policy<T>;

  /// Lock policy type.
  using lock_policy = typename ArtPolicy::lock_policy;
  /// Read critical section type.
  using read_critical_section = typename ArtPolicy::read_critical_section;

  /// Leaf pointer.
  using db_leaf_unique_ptr = typename ArtPolicy::db_leaf_unique_ptr;

  /// Database type.
  using db_type = typename ArtPolicy::db_type;

  /// Result of find_child operation.
  ///
  /// First element is child index in node. Second element is pointer to child.
  /// If no child exists, pointer is nullptr and index is undefined.
  using find_result =
      std::pair<std::uint8_t, critical_section_policy<node_ptr>*>;

  /// Header type.
  using header_type = typename ArtPolicy::header_type;
  /// Iterator result.
  using iter_result = detail::iter_result<header_type>;
  /// Optional iterator result.
  using iter_result_opt = detail::iter_result_opt<header_type>;

 protected:
  /// Base internal node type.
  using inode_type = typename ArtPolicy::inode;
  /// Unique pointer to basic_inode_4.
  using db_inode4_unique_ptr = typename ArtPolicy::db_inode4_unique_ptr;
  /// Unique pointer to basic_inode_16.
  using db_inode16_unique_ptr = typename ArtPolicy::db_inode16_unique_ptr;
  /// Unique pointer to basic_inode_48.
  using db_inode48_unique_ptr = typename ArtPolicy::db_inode48_unique_ptr;
  /// Tree depth type.
  using tree_depth_type = typename ArtPolicy::tree_depth_type;

 private:
  /// basic_inode_4 node type.
  using inode4_type = typename ArtPolicy::inode4_type;
  /// basic_inode_16 node type.
  using inode16_type = typename ArtPolicy::inode16_type;
  /// basic_inode_48 node type.
  using inode48_type = typename ArtPolicy::inode48_type;
  /// basic_inode_256 node type.
  using inode256_type = typename ArtPolicy::inode256_type;

  /// Leaf type.
  using leaf_type = typename ArtPolicy::leaf_type;

  /// Read the dispatch byte from a leaf at the given depth.
  /// For keyless leaves this is unreachable — the caller must not
  /// invoke this path.
  [[nodiscard]] static constexpr std::uint8_t leaf_key_byte_at(
      const leaf_type* leaf, tree_depth_type depth) noexcept {
    if constexpr (ArtPolicy::can_eliminate_key_in_leaf) {
      UNODB_DETAIL_CANNOT_HAPPEN();
      return 0;
    } else {
      return static_cast<std::uint8_t>(leaf->get_key_view()[depth]);
    }
  }

  /// \}

 public:
  /// \name Accessors
  /// \{

  /// Return const reference to key prefix.
  [[nodiscard]] constexpr const auto& get_key_prefix() const noexcept {
    return k_prefix;
  }

  /// Return mutable reference to key prefix.
  [[nodiscard]] constexpr auto& get_key_prefix() noexcept { return k_prefix; }

  /// Return number of children.
  ///
  /// \note For internal use only.
  [[nodiscard]] constexpr auto get_children_count() const noexcept {
    return children_count.load();
  }

  /// \}

  /// \name Type dispatch methods
  /// Methods that dispatch to specific node type implementations.
  /// \{

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26491)

  /// Dispatch add or choose subtree operation to appropriate node type.
  ///
  /// \tparam ReturnType Return type of the operation
  /// \tparam Args Argument types forwarded to concrete node
  ///
  /// \param type Current node type for dispatch
  /// \param args Arguments forwarded to concrete implementation
  ///
  /// \return Result from concrete node's add_or_choose_subtree
  template <typename ReturnType, typename... Args>
  [[nodiscard]] ReturnType add_or_choose_subtree(node_type type,
                                                 Args&&... args) {
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->add_or_choose_subtree(
            std::forward<Args>(args)...);
      case node_type::I16:
        return static_cast<inode16_type*>(this)->add_or_choose_subtree(
            std::forward<Args>(args)...);
      case node_type::I48:
        return static_cast<inode48_type*>(this)->add_or_choose_subtree(
            std::forward<Args>(args)...);
      case node_type::I256:
        return static_cast<inode256_type*>(this)->add_or_choose_subtree(
            std::forward<Args>(args)...);
        // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Dispatch remove or choose subtree operation to appropriate node type.
  ///
  /// \tparam ReturnType Return type of the operation
  /// \tparam Args Argument types forwarded to concrete node
  ///
  /// \param type Current node type for dispatch
  /// \param args Arguments forwarded to concrete implementation
  ///
  /// \return Result from concrete node's remove_or_choose_subtree
  template <typename ReturnType, typename... Args>
  [[nodiscard]] ReturnType remove_or_choose_subtree(node_type type,
                                                    Args&&... args) {
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->remove_or_choose_subtree(
            std::forward<Args>(args)...);
      case node_type::I16:
        return static_cast<inode16_type*>(this)->remove_or_choose_subtree(
            std::forward<Args>(args)...);
      case node_type::I48:
        return static_cast<inode48_type*>(this)->remove_or_choose_subtree(
            std::forward<Args>(args)...);
      case node_type::I256:
        return static_cast<inode256_type*>(this)->remove_or_choose_subtree(
            std::forward<Args>(args)...);
        // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// \}

  /// \name Child access methods
  /// These methods may operate on inconsistent nodes during parallel updates
  /// with the OLC flavor. They must not assert, and may produce incorrect
  /// results that will be checked before acting on them.
  /// \{

  /// Find child matching key byte.
  ///
  /// \param type Current node type
  /// \param key_byte Key byte to search for
  /// \return find_result with child index and pointer
  [[nodiscard, gnu::pure]] constexpr find_result find_child(
      node_type type, std::byte key_byte) noexcept {
    UNODB_DETAIL_ASSERT(type != node_type::LEAF);
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->find_child(key_byte);
      case node_type::I16:
        return static_cast<inode16_type*>(this)->find_child(key_byte);
      case node_type::I48:
        return static_cast<inode48_type*>(this)->find_child(key_byte);
      case node_type::I256:
        return static_cast<inode256_type*>(this)->find_child(key_byte);
        // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Check if child at index holds a packed value, dispatching by type.
  [[nodiscard]] constexpr bool is_value_in_slot(
      node_type type, std::uint8_t child_i) const noexcept {
    switch (type) {
      case node_type::I4:
        return static_cast<const inode4_type*>(this)->is_value_in_slot(child_i);
      case node_type::I16:
        return static_cast<const inode16_type*>(this)->is_value_in_slot(
            child_i);
      case node_type::I48:
        return static_cast<const inode48_type*>(this)->is_value_in_slot(
            child_i);
      case node_type::I256:
        return static_cast<const inode256_type*>(this)->is_value_in_slot(
            child_i);
      // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Clear value bitmask bit, dispatching by type.
  constexpr void clear_value_bit(node_type type,
                                 std::uint8_t child_i) noexcept {
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->clear_value_bit(child_i);
      case node_type::I16:
        return static_cast<inode16_type*>(this)->clear_value_bit(child_i);
      case node_type::I48:
        return static_cast<inode48_type*>(this)->clear_value_bit(child_i);
      case node_type::I256:
        return static_cast<inode256_type*>(this)->clear_value_bit(child_i);
      // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Remove child entry without reclaiming the child, dispatching by type.
  ///
  /// \param type Current node type
  /// \param child_index Child index to remove
  constexpr void remove_child_entry(node_type type,
                                    std::uint8_t child_index) noexcept {
    switch (type) {
      case node_type::I4:
        static_cast<inode4_type*>(this)->remove_child_entry(child_index);
        return;
      case node_type::I16:
        static_cast<inode16_type*>(this)->remove_child_entry(child_index);
        return;
      case node_type::I48:
        static_cast<inode48_type*>(this)->remove_child_entry(child_index);
        return;
      case node_type::I256:
        static_cast<inode256_type*>(this)->remove_child_entry(child_index);
        return;
        // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Return child node at specified child index.
  ///
  /// \param type Current node type
  /// \param child_index Child index
  /// \return Child node pointer
  ///
  /// \note For basic_inode_48, child_index is index into child_indices[]. For
  /// other types, it is direct index into children[]. This method hides this
  /// distinction.
  [[nodiscard, gnu::pure]] constexpr node_ptr get_child(
      node_type type, std::uint8_t child_index) noexcept {
    UNODB_DETAIL_ASSERT(type != node_type::LEAF);
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->get_child(child_index);
      case node_type::I16:
        return static_cast<inode16_type*>(this)->get_child(child_index);
      case node_type::I48:
        return static_cast<inode48_type*>(this)->get_child(child_index);
      case node_type::I256:
        return static_cast<inode256_type*>(this)->get_child(child_index);
        // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Return iterator result for first valid child.
  ///
  /// \param type Current node type
  /// \return iter_result for first child
  [[nodiscard, gnu::pure]] constexpr iter_result begin(
      node_type type) noexcept {
    UNODB_DETAIL_ASSERT(type != node_type::LEAF);
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->begin();
      case node_type::I16:
        return static_cast<inode16_type*>(this)->begin();
      case node_type::I48:
        return static_cast<inode48_type*>(this)->begin();
      case node_type::I256:
        return static_cast<inode256_type*>(this)->begin();
      // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Return iterator result for last valid child.
  ///
  /// \param type Current node type
  /// \return iter_result for last child
  [[nodiscard, gnu::pure]] constexpr iter_result last(node_type type) noexcept {
    UNODB_DETAIL_ASSERT(type != node_type::LEAF);
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->last();
      case node_type::I16:
        return static_cast<inode16_type*>(this)->last();
      case node_type::I48:
        return static_cast<inode48_type*>(this)->last();
      case node_type::I256:
        return static_cast<inode256_type*>(this)->last();
      // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Return iterator result for next child after given index.
  ///
  /// \param type Current node type
  /// \param child_index Current position within node
  /// \return Optional iter_result for next child if one exists
  [[nodiscard, gnu::pure]] constexpr iter_result_opt next(
      node_type type, std::uint8_t child_index) noexcept {
    UNODB_DETAIL_ASSERT(type != node_type::LEAF);
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->next(child_index);
      case node_type::I16:
        return static_cast<inode16_type*>(this)->next(child_index);
      case node_type::I48:
        return static_cast<inode48_type*>(this)->next(child_index);
      case node_type::I256:
        return static_cast<inode256_type*>(this)->next(child_index);
      // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Return iterator result for previous child before given index.
  ///
  /// \param type Current node type
  /// \param child_index Current position within node
  /// \return Optional iter_result for previous child if one exists
  [[nodiscard, gnu::pure]] constexpr iter_result_opt prior(
      node_type type, std::uint8_t child_index) noexcept {
    UNODB_DETAIL_ASSERT(type != node_type::LEAF);
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->prior(child_index);
      case node_type::I16:
        return static_cast<inode16_type*>(this)->prior(child_index);
      case node_type::I48:
        return static_cast<inode48_type*>(this)->prior(child_index);
      case node_type::I256:
        return static_cast<inode256_type*>(this)->prior(child_index);
      // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Return iterator result for greatest key byte <= given key byte.
  ///
  /// Used by seek() to find path before key when key is not mapped.
  ///
  /// \param type Current node type
  /// \param key_byte Key byte to compare
  /// \return Optional iter_result for matching or lesser child
  [[nodiscard, gnu::pure]] constexpr iter_result_opt lte_key_byte(
      node_type type, std::byte key_byte) noexcept {
    UNODB_DETAIL_ASSERT(type != node_type::LEAF);
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->lte_key_byte(key_byte);
      case node_type::I16:
        return static_cast<inode16_type*>(this)->lte_key_byte(key_byte);
      case node_type::I48:
        return static_cast<inode48_type*>(this)->lte_key_byte(key_byte);
      case node_type::I256:
        return static_cast<inode256_type*>(this)->lte_key_byte(key_byte);
        // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  /// Return iterator result for smallest key byte >= given key byte.
  ///
  /// Used by seek() to find path after key when key is not mapped.
  ///
  /// \param type Node type
  /// \param key_byte Key byte to compare
  /// \return Optional iter_result for matching or greater child
  [[nodiscard, gnu::pure]] constexpr iter_result_opt gte_key_byte(
      node_type type, std::byte key_byte) noexcept {
    UNODB_DETAIL_ASSERT(type != node_type::LEAF);
    switch (type) {
      case node_type::I4:
        return static_cast<inode4_type*>(this)->gte_key_byte(key_byte);
      case node_type::I16:
        return static_cast<inode16_type*>(this)->gte_key_byte(key_byte);
      case node_type::I48:
        return static_cast<inode48_type*>(this)->gte_key_byte(key_byte);
      case node_type::I256:
        return static_cast<inode256_type*>(this)->gte_key_byte(key_byte);
        // LCOV_EXCL_START
      case node_type::LEAF:
        UNODB_DETAIL_CANNOT_HAPPEN();
    }
    UNODB_DETAIL_CANNOT_HAPPEN();
    // LCOV_EXCL_STOP
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// \}

  /// \name Constructors
  /// \{

  /// Construct with shared prefix from two keys.
  ///
  /// \param children_count_ Initial children count
  /// \param k1 First key for prefix computation
  /// \param shifted_k2 Second key shifted to current depth
  /// \param depth Current tree depth
  constexpr basic_inode_impl(unsigned children_count_, key_view k1,
                             art_key_type shifted_k2,
                             tree_depth<art_key_type> depth) noexcept
      : k_prefix{k1, shifted_k2, depth},
        children_count{static_cast<std::uint8_t>(children_count_)} {}

  /// Construct with explicit prefix length from key at depth.
  constexpr basic_inode_impl(unsigned children_count_,
                             detail::key_prefix_size prefix_len, key_view k1,
                             tree_depth<art_key_type> depth) noexcept
      : k_prefix{prefix_len, k1, depth},
        children_count{static_cast<std::uint8_t>(children_count_)} {}

  /// Construct with truncated prefix from source node.
  ///
  /// \param children_count_ Initial children count
  /// \param key_prefix_len Length of prefix to copy
  /// \param key_prefix_source_node Node to copy prefix from
  constexpr basic_inode_impl(unsigned children_count_, unsigned key_prefix_len,
                             const inode_type& key_prefix_source_node) noexcept
      : k_prefix{key_prefix_len, key_prefix_source_node.get_key_prefix()},
        children_count{static_cast<std::uint8_t>(children_count_)} {}

  /// Copy constructor with specified children count.
  ///
  /// \param children_count_ Initial children count
  /// \param other Node to copy prefix from
  constexpr basic_inode_impl(unsigned children_count_,
                             const basic_inode_impl& other) noexcept
      : k_prefix{other.k_prefix},
        children_count{static_cast<std::uint8_t>(children_count_)} {}

  /// \}

 protected:
  /// Dump node contents to stream \a os for debugging.
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os, bool) const {
    k_prefix.dump(os);
    const auto children_count_ = this->children_count.load();
    os << ", # children = "
       << (children_count_ == 0 ? 256 : static_cast<unsigned>(children_count_));
  }

 private:
  /// Key prefix for this node.
  key_prefix<art_key_type, critical_section_policy> k_prefix;
  static_assert(sizeof(k_prefix) == 8);

  /// Number of children in this node.
  critical_section_policy<std::uint8_t> children_count;

  /// Sentinel value for child not found.
  static constexpr std::uint8_t child_not_found_i = 0xFFU;

 protected:
  /// Represents the find_result when no such child was found.
  static constexpr find_result child_not_found{child_not_found_i, nullptr};

  /// Iterator result value at the end of iteration.
  static constexpr iter_result_opt end_result{};

  friend class unodb::db<key_type, value_type>;
  friend class unodb::olc_db<key_type, value_type>;
  friend struct olc_inode_immediate_deleter;

  template <class, unsigned, unsigned, node_type, class, class, class>
  friend class basic_inode;

  template <class>
  friend class basic_inode_4;

  template <class>
  friend class basic_inode_16;

  template <class>
  friend class basic_inode_48;

  template <class>
  friend class basic_inode_256;
};  // class basic_inode_impl

/// CRTP base class for concrete internal node types.
///
/// Common ancestor for all internal node types (basic_inode_4, basic_inode_16,
/// basic_inode_48, basic_inode_256) in both OLC and regular variants. Provides
/// size constraints, factory method, and constructors for node transitions.
///
/// \tparam ArtPolicy Policy class defining types and operations
/// \tparam MinSize Minimum number of children
/// \tparam Capacity Maximum number of children
/// \tparam NodeType Node type enum value
/// \tparam SmallerDerived Smaller node type (for shrinking) or fake_inode
/// \tparam LargerDerived Larger node type (for growing) or fake_inode
/// \tparam Derived Concrete derived class (CRTP)
template <class ArtPolicy, unsigned MinSize, unsigned Capacity,
          node_type NodeType, class SmallerDerived, class LargerDerived,
          class Derived>
class [[nodiscard]] basic_inode : public basic_inode_impl<ArtPolicy> {
  static_assert(NodeType != node_type::LEAF);
  static_assert(!std::is_same_v<Derived, LargerDerived>);
  static_assert(!std::is_same_v<SmallerDerived, Derived>);
  static_assert(!std::is_same_v<SmallerDerived, LargerDerived>);
  static_assert(MinSize < Capacity);

  /// Base class type.
  using parent = basic_inode_impl<ArtPolicy>;

 public:
  using typename parent::db_leaf_unique_ptr;
  using typename parent::db_type;
  using typename parent::node_ptr;

  /// Create new internal node.
  ///
  /// \param db_instance Database instance
  /// \param args Constructor arguments
  /// \return Unique pointer to new node
  template <typename... Args>
  [[nodiscard]] static constexpr auto create(db_type& db_instance,
                                             Args&&... args) {
    return ArtPolicy::template make_db_inode_unique_ptr<Derived>(
        db_instance, std::forward<Args>(args)...);
  }

#ifndef NDEBUG

  /// Check if node is at capacity (debug only).
  [[nodiscard]] constexpr bool is_full_for_add() const noexcept {
    return this->children_count == capacity;
  }

#endif

  /// Check if node is at minimum size.
  [[nodiscard]] constexpr bool is_min_size() const noexcept {
    return this->children_count == min_size;
  }

  /// Minimum children count.
  static constexpr auto min_size = MinSize;
  /// Maximum children count.
  static constexpr auto capacity = Capacity;
  /// Node type enum value.
  static constexpr auto type = NodeType;

  /// Larger node type.
  using larger_derived_type = LargerDerived;
  /// Smaller node type.
  using smaller_derived_type = SmallerDerived;
  /// Base internal node type.
  using inode_type = typename parent::inode_type;

 protected:
  using typename parent::art_key_type;
  using typename parent::tree_depth_type;

  /// Construct new node with prefix from two diverging keys.
  ///
  /// \param k1 First key for prefix computation
  /// \param shifted_k2 Second key shifted to current depth
  /// \param depth Current tree depth
  constexpr basic_inode(unodb::key_view k1, art_key_type shifted_k2,
                        tree_depth<art_key_type> depth) noexcept
      : parent{MinSize, k1, shifted_k2, depth} {
    UNODB_DETAIL_ASSERT(is_min_size());
  }

  /// Construct with truncated prefix from source node.
  ///
  /// \param key_prefix_len Length of prefix to copy
  /// \param key_prefix_source_node Node to copy prefix from
  constexpr basic_inode(unsigned key_prefix_len,
                        const inode_type& key_prefix_source_node) noexcept
      : parent{MinSize, key_prefix_len, key_prefix_source_node} {
    UNODB_DETAIL_ASSERT(is_min_size());
  }

  /// Construct by growing from \a source_node of smaller type.
  explicit constexpr basic_inode(const SmallerDerived& source_node) noexcept
      : parent{MinSize, source_node} {
    // Cannot assert that source_node.is_full_for_add because we are creating
    // this node optimistically in the case of OLC.
    UNODB_DETAIL_ASSERT(is_min_size());
  }

  /// Construct by shrinking from \a source_node of larger type.
  explicit constexpr basic_inode(const LargerDerived& source_node) noexcept
      : parent{Capacity, source_node} {
    // Cannot assert that source_node.is_min_size because we are creating this
    // node optimistically in the case of OLC.
    UNODB_DETAIL_ASSERT(is_full_for_add());
  }

  /// Tag type to select the single-child chain constructor.
  struct single_child_tag {};

  /// Construct single-child node with prefix from two identical keys.
  ///
  /// Used to create chain nodes when keys share more than
  /// key_prefix_capacity bytes. The prefix is computed from the keys
  /// at the given depth, and children_count is set to 1.
  ///
  /// \param k1 First key for prefix computation
  /// \param shifted_k2 Second key shifted to current depth
  /// \param depth Current tree depth
  constexpr basic_inode(unodb::key_view k1, art_key_type shifted_k2,
                        tree_depth<art_key_type> depth,
                        single_child_tag) noexcept
      : parent{1, k1, shifted_k2, depth} {}

  /// Construct single-child node with explicit prefix length.
  /// Used by build_chain when remaining key < key_prefix_capacity.
  constexpr basic_inode(detail::key_prefix_size prefix_len, unodb::key_view k1,
                        tree_depth<art_key_type> depth,
                        single_child_tag) noexcept
      : parent{1, prefix_len, k1, depth} {}
};

/// Type alias for basic_inode_4 parent class.
template <class ArtPolicy>
using basic_inode_4_parent =
    basic_inode<ArtPolicy, 2, 4, node_type::I4, fake_inode,
                typename ArtPolicy::inode16_type,
                typename ArtPolicy::inode4_type>;

/// Internal node with 2-4 children (N4).
///
/// Keys are stored in sorted order in a 4-byte array. Linear search is used
/// since the array is small. A corresponding array of 4 child pointers exists
/// where each key position indexes the corresponding child pointer.
/// This is the smallest internal node type.
///
/// \tparam ArtPolicy Policy class defining types and operations
template <class ArtPolicy>
class basic_inode_4
    : public basic_inode_4_parent<ArtPolicy>,
      private value_bitmask_field<ArtPolicy::can_eliminate_leaf, std::uint8_t> {
  /// Parent class type.
  using parent_class = basic_inode_4_parent<ArtPolicy>;
  /// Bitmask base (empty via EBO when can_eliminate_leaf is false).
  using bitmask_base =
      value_bitmask_field<ArtPolicy::can_eliminate_leaf, std::uint8_t>;

  using typename parent_class::inode16_type;
  using typename parent_class::inode4_type;
  using typename parent_class::inode_type;

  /// Thread-safety policy for member access.
  template <typename T>
  using critical_section_policy =
      typename ArtPolicy::template critical_section_policy<T>;

 public:
  using typename parent_class::art_key_type;
  using typename parent_class::db_inode4_unique_ptr;
  using typename parent_class::db_leaf_unique_ptr;
  using typename parent_class::db_type;
  using typename parent_class::find_result;
  using typename parent_class::larger_derived_type;
  using typename parent_class::leaf_type;
  using typename parent_class::node_ptr;

  /// Tree depth type.
  using tree_depth_type = tree_depth<art_key_type>;

  /// \name Constructors
  /// \{

  /// Construct empty node with prefix from two diverging keys.
  ///
  /// \param k1 First key for prefix computation
  /// \param shifted_k2 Second key shifted to current depth
  /// \param depth Current tree depth
  constexpr basic_inode_4(db_type&, unodb::key_view k1, art_key_type shifted_k2,
                          // cppcheck-suppress passedByValue
                          tree_depth_type depth) noexcept
      : parent_class{k1, shifted_k2, depth} {}

  /// Construct with prefix and two initial children.
  ///
  /// \param k1 First key for prefix computation
  /// \param shifted_k2 Second key shifted to current depth
  /// \param depth Current tree depth
  /// \param child1 First child leaf
  /// \param child2 Second child leaf (ownership transferred)
  constexpr basic_inode_4(db_type&, key_view k1, art_key_type shifted_k2,
                          // cppcheck-suppress passedByValue
                          tree_depth_type depth, leaf_type* child1,
                          db_leaf_unique_ptr&& child2) noexcept
      : parent_class{k1, shifted_k2, depth} {
    init(k1, shifted_k2, depth, child1, std::move(child2));
  }

  /// Construct with truncated prefix from source node.
  ///
  /// \param source_node Node to copy prefix from
  /// \param len Length of prefix to copy
  // cppcheck-suppress passedByValue
  constexpr basic_inode_4(db_type&, node_ptr source_node, unsigned len) noexcept
      : parent_class{len, *source_node.template ptr<inode_type*>()} {}

  /// Construct by splitting prefix from source node.
  ///
  /// \param source_node Node to split prefix from
  /// \param len Length of prefix to copy
  /// \param depth Current tree depth
  /// \param child1 Child leaf to add (ownership transferred)
  constexpr basic_inode_4(db_type&, node_ptr source_node, unsigned len,
                          // cppcheck-suppress passedByValue
                          [[maybe_unused]] tree_depth_type depth,
                          db_leaf_unique_ptr&& child1) noexcept
      : parent_class{len, *source_node.template ptr<inode_type*>()} {
    init(source_node, len, depth, std::move(child1));
  }

  /// Construct by splitting prefix, with explicit dispatch byte for child1.
  constexpr basic_inode_4(db_type&, node_ptr source_node, unsigned len,
                          // cppcheck-suppress passedByValue
                          tree_depth_type depth, db_leaf_unique_ptr&& child1,
                          std::byte child1_key_byte) noexcept
      : parent_class{len, *source_node.template ptr<inode_type*>()} {
    init(source_node, len, depth, std::move(child1), child1_key_byte);
  }

  /// Construct by splitting prefix, value-in-slot child with dispatch byte.
  constexpr basic_inode_4(db_type&, node_ptr source_node, unsigned len,
                          // cppcheck-suppress passedByValue
                          tree_depth_type depth, node_ptr child1_value,
                          std::byte child1_key_byte) noexcept
      : parent_class{len, *source_node.template ptr<inode_type*>()} {
    init(source_node, len, depth, child1_value, child1_key_byte);
  }

  /// Construct by shrinking from basic_inode_16 \a source_node.
  constexpr basic_inode_4(db_type&, const inode16_type& source_node) noexcept
      : parent_class{source_node} {}

  /// Construct by shrinking from basic_inode_16 with deletion.
  ///
  /// \param db_instance Database instance
  /// \param source_node N16 node to shrink from
  /// \param child_to_delete Index of child to delete
  constexpr basic_inode_4(db_type& db_instance, inode16_type& source_node,
                          std::uint8_t child_to_delete)
      : parent_class{source_node} {
    init(db_instance, source_node, child_to_delete);
  }

  /// Construct single-child chain node for long shared prefixes.
  ///
  /// Creates an inode_4 with one child, used when two keys share more
  /// than key_prefix_capacity bytes. The prefix is the first 7 bytes
  /// of the key at \a depth, and the single child is placed under
  /// \a key_byte.
  ///
  /// \param k1 Key for prefix computation
  /// \param remaining_key Same key shifted to current depth
  /// \param depth Current tree depth
  /// \param key_byte Dispatch byte for the single child
  /// \param child The single child node
  constexpr basic_inode_4(db_type&, key_view k1, art_key_type remaining_key,
                          // cppcheck-suppress passedByValue
                          [[maybe_unused]] tree_depth_type depth,
                          std::byte key_byte, node_ptr child) noexcept
      : parent_class{k1, remaining_key, depth,
                     typename parent_class::single_child_tag{}} {
    init(key_byte, child);
  }

  /// Check if collapsing this min-size I4 would overflow the prefix.
  [[nodiscard]] constexpr bool can_collapse(
      std::uint8_t child_to_delete) const noexcept {
    UNODB_DETAIL_ASSERT(this->is_min_size());
    const std::uint8_t child_to_leave = (child_to_delete == 0) ? 1U : 0U;
    const auto child_ptr = children[child_to_leave].load();
    if constexpr (ArtPolicy::can_eliminate_leaf) {
      if (is_value_in_slot(child_to_leave)) return false;
    }
    if (child_ptr.type() == node_type::LEAF) {
      // For keyless leaves, collapsing would lose key bytes encoded
      // in this inode's prefix+dispatch.  Keep the chain intact.
      return !ArtPolicy::can_eliminate_key_in_leaf;
    }
    const auto* const child_inode{child_ptr.template ptr<inode_type*>()};
    return this->get_key_prefix().length() +
               child_inode->get_key_prefix().length() <
           detail::key_prefix_capacity;
  }

  /// Construct single-child chain node with explicit prefix length.
  /// Used by build_chain when remaining key <= key_prefix_capacity.
  constexpr basic_inode_4(db_type&, key_view k1,
                          // cppcheck-suppress passedByValue
                          [[maybe_unused]] tree_depth_type depth,
                          detail::key_prefix_size prefix_len,
                          std::byte key_byte, node_ptr child) noexcept
      : parent_class{prefix_len, k1, depth,
                     typename parent_class::single_child_tag{}} {
    init(key_byte, child);
  }

  /// \}

  /// \name Initialization methods
  /// \{

  /// Initialize by splitting prefix from source node.
  ///
  /// \param source_node Node whose prefix is being split
  /// \param shared_prefix_len Length of shared prefix
  /// \param depth Current tree depth
  /// \param child1 New leaf child to insert
  /// \param child1_key_byte Dispatch byte for child1 (used for keyless
  ///   leaves where the byte cannot be read from the leaf)
  template <typename LeafPtr>
  constexpr void init(node_ptr source_node, unsigned shared_prefix_len,
                      [[maybe_unused]] tree_depth_type depth, LeafPtr&& child1,
                      std::byte child1_key_byte) {
    auto* const source_inode{source_node.template ptr<inode_type*>()};
    auto& source_key_prefix = source_inode->get_key_prefix();
    UNODB_DETAIL_ASSERT(shared_prefix_len < source_key_prefix.length());

    const auto source_node_key_byte = source_key_prefix[shared_prefix_len];
    source_key_prefix.cut(static_cast<key_prefix_size>(shared_prefix_len) + 1U);
    add_two_to_empty(source_node_key_byte, source_node, child1_key_byte,
                     std::move(child1));
  }

  /// Initialize by splitting prefix from source node (keyed leaf variant).
  /// Reads the dispatch byte from the leaf's key.
  template <typename LeafPtr>
  constexpr void init(node_ptr source_node, unsigned shared_prefix_len,
                      [[maybe_unused]] tree_depth_type depth,
                      LeafPtr&& child1) {
    const auto diff_key_byte_i = depth + shared_prefix_len;
    init(source_node, shared_prefix_len, depth, std::forward<LeafPtr>(child1),
         static_cast<std::byte>(this->leaf_key_byte_at(
             child1.get(), tree_depth_type{diff_key_byte_i})));
  }

  /// Initialize by shrinking from basic_inode_16 node.
  ///
  /// \param db_instance Database for memory tracking
  /// \param source_node N16 node to shrink from
  /// \param child_to_delete Index of child to remove
  // MSVC C26815 false positive: reclaim objects intentionally destroyed at
  // scope exit while db_instance (passed by caller) remains valid.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
  constexpr void init(db_type& db_instance, inode16_type& source_node,
                      std::uint8_t child_to_delete) {
    const auto reclaim_source_node{
        ArtPolicy::template make_db_inode_reclaimable_ptr<inode16_type>(
            &source_node, db_instance)};
    auto source_keys_itr = source_node.keys.byte_array.cbegin();
    auto keys_itr = keys.byte_array.begin();
    auto source_children_itr = source_node.children.cbegin();
    auto children_itr = children.begin();

    while (source_keys_itr !=
           source_node.keys.byte_array.cbegin() + child_to_delete) {
      *keys_itr++ = *source_keys_itr++;
      *children_itr++ = *source_children_itr++;
    }

    [[maybe_unused]] const auto r{
        ArtPolicy::reclaim_if_leaf(source_children_itr->load(), db_instance)};

    ++source_keys_itr;
    ++source_children_itr;

    while (source_keys_itr !=
           source_node.keys.byte_array.cbegin() + inode16_type::min_size) {
      *keys_itr++ = *source_keys_itr++;
      *children_itr++ = *source_children_itr++;
    }

    UNODB_DETAIL_ASSERT(this->children_count == basic_inode_4::capacity);
    UNODB_DETAIL_ASSERT(
        std::is_sorted(keys.byte_array.cbegin(),
                       keys.byte_array.cbegin() + basic_inode_4::capacity));

    if constexpr (ArtPolicy::can_eliminate_leaf) {
      std::uint8_t dst = 0;
      for (std::uint8_t src = 0; src < inode16_type::min_size; ++src) {
        if (src == child_to_delete) continue;
        if (source_node.is_value_in_slot(src)) set_value_bit(dst);
        ++dst;
      }
    }
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Initialize with two leaf children from diverging keys.
  ///
  /// \param k1 First key
  /// \param shifted_k2 Second key shifted to current depth
  /// \param depth Current tree depth
  /// \param child1 First leaf child
  /// \param child2 Second leaf child to insert
  constexpr void init(key_view k1, art_key_type shifted_k2,
                      // cppcheck-suppress passedByValue
                      tree_depth_type depth, const leaf_type* child1,
                      db_leaf_unique_ptr&& child2) noexcept {
    const auto k2_next_byte_depth = this->get_key_prefix().length();
    const auto k1_next_byte_depth = k2_next_byte_depth + depth;
    add_two_to_empty(k1[k1_next_byte_depth], node_ptr{child1, node_type::LEAF},
                     shifted_k2[k2_next_byte_depth], std::move(child2));
  }

  /// Initialize single-child node.
  ///
  /// \param key_byte Dispatch byte for the child
  /// \param child The single child node
  constexpr void init(std::byte key_byte, node_ptr child) noexcept {
    UNODB_DETAIL_ASSERT(this->children_count == 1);

    keys.byte_array[0] = key_byte;
    children[0] = child;
#ifndef UNODB_DETAIL_X86_64
    keys.byte_array[1] = unused_key_byte;
    keys.byte_array[2] = unused_key_byte;
    keys.byte_array[3] = unused_key_byte;
#endif
  }

  /// \}

  /// Add child to node that has available capacity.
  ///
  /// \param child Leaf child to add
  /// \param depth Current tree depth
  /// \param children_count_ Current children count
  ///
  /// \note The node already keeps its current children count, but all callers
  /// have already loaded it.
  constexpr void add_to_nonfull(db_leaf_unique_ptr&& child,
                                // cppcheck-suppress passedByValue
                                [[maybe_unused]] tree_depth_type depth,
                                std::byte key_byte,
                                std::uint8_t children_count_) noexcept {
    UNODB_DETAIL_ASSERT(children_count_ == this->children_count);
    UNODB_DETAIL_ASSERT(children_count_ < parent_class::capacity);
    UNODB_DETAIL_ASSERT(std::is_sorted(
        keys.byte_array.cbegin(), keys.byte_array.cbegin() + children_count_));

    const auto kb = static_cast<std::uint8_t>(key_byte);
#ifdef UNODB_DETAIL_X86_64
    const auto mask = (1U << children_count_) - 1;
    const auto insert_pos_index = get_insert_pos(kb, mask);
#else
    // This is also currently the best ARM implementation.
    const auto first_lt = ((keys.integer & 0xFFU) < kb) ? 1 : 0;
    const auto second_lt = (((keys.integer >> 8U) & 0xFFU) < kb) ? 1 : 0;
    const auto third_lt = ((keys.integer >> 16U) & 0xFFU) < kb ? 1 : 0;
    const auto insert_pos_index =
        static_cast<unsigned>(first_lt + second_lt + third_lt);
#endif

    for (typename decltype(keys.byte_array)::size_type i = children_count_;
         i > insert_pos_index; --i) {
      keys.byte_array[i] = keys.byte_array[i - 1];
      children[i] = children[i - 1];
    }
    keys.byte_array[insert_pos_index] = key_byte;
    children[insert_pos_index] = node_ptr{child.release(), node_type::LEAF};

    ++children_count_;
    this->children_count = children_count_;

    UNODB_DETAIL_ASSERT(std::is_sorted(
        keys.byte_array.cbegin(), keys.byte_array.cbegin() + children_count_));
  }

  /// Add a packed value to a non-full node (value-in-slot variant).
  constexpr void add_to_nonfull(node_ptr packed_value,
                                // cppcheck-suppress passedByValue
                                [[maybe_unused]] tree_depth_type depth,
                                std::byte key_byte,
                                std::uint8_t children_count_) noexcept {
    UNODB_DETAIL_ASSERT(children_count_ == this->children_count);
    UNODB_DETAIL_ASSERT(children_count_ < parent_class::capacity);

    const auto kb = static_cast<std::uint8_t>(key_byte);
#ifdef UNODB_DETAIL_X86_64
    const auto mask = (1U << children_count_) - 1;
    const auto insert_pos_index = get_insert_pos(kb, mask);
#else
    const auto first_lt = ((keys.integer & 0xFFU) < kb) ? 1 : 0;
    const auto second_lt = (((keys.integer >> 8U) & 0xFFU) < kb) ? 1 : 0;
    const auto third_lt = ((keys.integer >> 16U) & 0xFFU) < kb ? 1 : 0;
    const auto insert_pos_index =
        static_cast<unsigned>(first_lt + second_lt + third_lt);
#endif

    for (typename decltype(keys.byte_array)::size_type i = children_count_;
         i > insert_pos_index; --i) {
      keys.byte_array[i] = keys.byte_array[i - 1];
      children[i] = children[i - 1];
    }
    keys.byte_array[insert_pos_index] = key_byte;
    children[insert_pos_index] = packed_value;
    set_value_bit(static_cast<std::uint8_t>(insert_pos_index));

    ++children_count_;
    this->children_count = children_count_;
  }

  /// Remove child at given index.
  ///
  /// \param child_index Index of child to remove
  /// \param db_instance Database instance
  // MSVC C26815 false positive: reclaim object intentionally destroyed at
  // scope exit while db_instance (passed by caller) remains valid.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
  constexpr void remove(std::uint8_t child_index,
                        db_type& db_instance) noexcept {
    UNODB_DETAIL_ASSERT(child_index < this->children_count.load());

    if constexpr (!ArtPolicy::can_eliminate_leaf) {
      const auto r{ArtPolicy::reclaim_leaf_on_scope_exit(
          children[child_index].load().template ptr<leaf_type*>(),
          db_instance)};
    } else {
      (void)db_instance;
    }

    remove_child_entry(child_index);
  }

  /// Remove child entry without reclaiming the child.
  ///
  /// Used during upward walk to remove empty chain inodes from their
  /// parent.  The caller is responsible for reclaiming the child.
  ///
  /// \param child_index Index of child to remove
  constexpr void remove_child_entry(std::uint8_t child_index) noexcept {
    auto children_count_ = this->children_count.load();

    UNODB_DETAIL_ASSERT(child_index < children_count_);
    UNODB_DETAIL_ASSERT(std::is_sorted(
        keys.byte_array.cbegin(), keys.byte_array.cbegin() + children_count_));

    typename decltype(keys.byte_array)::size_type i = child_index;
    for (; i < static_cast<unsigned>(children_count_ - 1); ++i) {
      keys.byte_array[i] = keys.byte_array[i + 1];
      children[i] = children[i + 1];
    }
#ifndef UNODB_DETAIL_X86_64
    keys.byte_array[i] = unused_key_byte;
#endif

    --children_count_;
    this->children_count = children_count_;

    // Shift bitmask bits to match shifted children array.
    if constexpr (ArtPolicy::can_eliminate_leaf) {
      bitmask_base::remove_at(child_index);
    }

    UNODB_DETAIL_ASSERT(std::is_sorted(
        keys.byte_array.cbegin(), keys.byte_array.cbegin() + children_count_));
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// For a node with two children, remove one child and return the other one.
  ///
  /// \param child_to_delete Index of child to remove (0 or 1)
  /// \param db_instance Database instance
  ///
  /// \return Pointer to the remaining child
  UNODB_DETAIL_DISABLE_CLANG_21_WARNING("-Wnrvo")
  [[nodiscard]] constexpr auto leave_last_child(std::uint8_t child_to_delete,
                                                db_type& db_instance) noexcept {
    UNODB_DETAIL_ASSERT(this->is_min_size());
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    UNODB_DETAIL_ASSERT(child_to_delete == 0 || child_to_delete == 1);

    if constexpr (!ArtPolicy::can_eliminate_leaf) {
      auto* const child_to_delete_ptr{
          children[child_to_delete].load().template ptr<leaf_type*>()};
      const auto r{ArtPolicy::reclaim_leaf_on_scope_exit(child_to_delete_ptr,
                                                         db_instance)};
    }

    const std::uint8_t child_to_leave = (child_to_delete == 0) ? 1U : 0U;
    const auto child_to_leave_ptr = children[child_to_leave].load();
    const bool remaining_is_value =
        ArtPolicy::can_eliminate_leaf && is_value_in_slot(child_to_leave);
    if (!remaining_is_value && child_to_leave_ptr.type() != node_type::LEAF) {
      auto* const inode_to_leave_ptr{
          child_to_leave_ptr.template ptr<inode_type*>()};
      inode_to_leave_ptr->get_key_prefix().prepend(
          this->get_key_prefix(), keys.byte_array[child_to_leave]);
    }
    return child_to_leave_ptr;
  }
  UNODB_DETAIL_RESTORE_CLANG_21_WARNINGS()

  /// Find child by key byte.
  ///
  /// \param key_byte Key byte to search for
  ///
  /// \return Result with child index and pointer, or not found
  [[nodiscard, gnu::pure]] find_result find_child(std::byte key_byte) noexcept {
#ifdef UNODB_DETAIL_X86_64
    const auto replicated_search_key =
        _mm_set1_epi8(static_cast<char>(key_byte));
    const auto keys_in_sse_reg =
        _mm_cvtsi32_si128(static_cast<std::int32_t>(keys.integer.load()));
    const auto matching_key_positions =
        _mm_cmpeq_epi8(replicated_search_key, keys_in_sse_reg);
    const auto mask = (1U << this->children_count.load()) - 1;
    const auto bit_field =
        static_cast<unsigned>(_mm_movemask_epi8(matching_key_positions)) & mask;
    if (bit_field != 0) {
      const auto i = static_cast<std::uint8_t>(std::countr_zero(bit_field));
      return std::make_pair(
          i, static_cast<critical_section_policy<node_ptr>*>(&children[i]));
    }
    return parent_class::child_not_found;
#elif defined(__aarch64__)
    const auto replicated_search_key =
        vdupq_n_u8(static_cast<std::uint8_t>(key_byte));
    const auto keys_in_neon_reg = vreinterpretq_u8_u32(
        // NOLINTNEXTLINE(misc-const-correctness)
        vsetq_lane_u32(keys.integer.load(), vdupq_n_u32(0), 0));
    const auto mask = (1ULL << (this->children_count.load() << 3U)) - 1;
    const auto matching_key_positions =
        vceqq_u8(replicated_search_key, keys_in_neon_reg);
    const auto u64_pos_in_vec =
        vget_low_u64(vreinterpretq_u64_u8(matching_key_positions));
    // NOLINTNEXTLINE(misc-const-correctness)
    const auto pos_in_scalar = vget_lane_u64(u64_pos_in_vec, 0);
    const auto masked_pos = pos_in_scalar & mask;

    if (masked_pos == 0) return parent_class::child_not_found;

    const auto i = static_cast<unsigned>(std::countr_zero(masked_pos) >> 3U);
    return std::make_pair(
        i, static_cast<critical_section_policy<node_ptr>*>(&children[i]));
#else   // #ifdef UNODB_DETAIL_X86_64
    // Bit twiddling:
    // contains_byte:    std::countr_zero:   for key index:
    //    0x80000000               0x1F                3
    //      0x800000               0x17                2
    //        0x8000               0x0F                1
    //          0x80               0x07                0
    //           0x0          UB (check!)       not found
    const auto match = contains_byte(keys.integer, key_byte);
    if (match == 0) return parent_class::child_not_found;

    const auto result =
        static_cast<typename decltype(keys.byte_array)::size_type>(
            std::countr_zero(match) >> 3U);

    // The condition could be replaced with masking, but this seems to result
    // in a benchmark regression
    if (result >= this->children_count.load())
      return parent_class::child_not_found;

    return std::make_pair(
        result,
        static_cast<critical_section_policy<node_ptr>*>(&children[result]));
#endif  // #ifdef UNODB_DETAIL_X86_64
  }

  /// Get child pointer at given index.
  ///
  /// \param child_index Index of child
  ///
  /// \return Child node pointer
  [[nodiscard, gnu::pure]] constexpr node_ptr get_child(
      std::uint8_t child_index) noexcept {
    return children[child_index].load();
  }

  /// Get iterator result for first child.
  ///
  /// \return Iterator result pointing to first child
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_4::iter_result
  begin() noexcept {
    const auto key = keys.byte_array[0].load();
    return {node_ptr{this, node_type::I4}, key, static_cast<uint8_t>(0),
            this->get_key_prefix().get_snapshot()};
  }

  /// Get iterator result for last child.
  ///
  /// \return Iterator result pointing to last child
  // TODO(laurynas) The iter_result-returning sequences follow the
  // same pattern once child_index is known. Look into extracting a
  // small helper method. This might apply to other inode types
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_4::iter_result
  last() noexcept {
    const auto child_index{
        static_cast<std::uint8_t>(this->children_count.load() - 1)};
    const auto key = keys.byte_array[child_index].load();
    return {node_ptr{this, node_type::I4}, key, child_index,
            this->get_key_prefix().get_snapshot()};
  }

  /// Get iterator result for next child after given index.
  ///
  /// \param child_index Current child index
  ///
  /// \return Iterator result for next child, or empty if at end
  // TODO(laurynas) explore 1) branchless 2) SIMD implementations for
  // begin(), last(), next(), prior(), get_key_byte(), and
  // lte_key_byte().  next() and begin() will be the most frequently
  // invoked methods (assuming forward traversal), so that would be
  // the place to start.  The GTE and LTE methods are only used by
  // seek() so they are relatively rarely invoked. Look at each of the
  // inode types when doing this.
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_4::iter_result_opt
  next(std::uint8_t child_index) noexcept {
    const auto nchildren{this->children_count.load()};
    const auto next_index{static_cast<uint8_t>(child_index + 1)};
    if (next_index >= nchildren) return parent_class::end_result;
    const auto key = keys.byte_array[next_index].load();
    return {{node_ptr{this, node_type::I4}, key, next_index,
             this->get_key_prefix().get_snapshot()}};
  }

  /// Get iterator result for previous child before given index.
  ///
  /// \param child_index Current child index
  ///
  /// \return Iterator result for previous child, or empty if at start
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_4::iter_result_opt
  prior(std::uint8_t child_index) noexcept {
    if (child_index == 0) return parent_class::end_result;
    const auto next_index{static_cast<std::uint8_t>(child_index - 1)};
    const auto key = keys.byte_array[next_index].load();
    return {{node_ptr{this, node_type::I4}, key, next_index,
             this->get_key_prefix().get_snapshot()}};
  }

  /// Find first child with key byte greater than or equal to given value.
  ///
  /// \param key_byte Key byte to compare against
  ///
  /// \return Iterator result for matching child, or empty if none found
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_4::iter_result_opt
  gte_key_byte(std::byte key_byte) noexcept {
    const auto nchildren{this->children_count.load()};
    for (std::uint8_t i = 0; i < nchildren; ++i) {
      const auto key = keys.byte_array[i].load();
      if (key >= key_byte) {
        return {{node_ptr{this, node_type::I4}, key, i,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    // This should only occur if there is no entry in the keys[] which
    // is greater-than the given [key_byte].
    return parent_class::end_result;
  }

  /// Find last child with key byte less than or equal to given value.
  ///
  /// \param key_byte Key byte to compare against
  ///
  /// \return Iterator result for matching child, or empty if none found
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_4::iter_result_opt
  lte_key_byte(std::byte key_byte) noexcept {
    const auto children_count_ = this->children_count.load();
    for (std::int64_t i = children_count_ - 1; i >= 0; i--) {
      const auto child_index = static_cast<std::uint8_t>(i);
      const auto key = keys.byte_array[child_index].load();
      if (key <= key_byte) {
        return {{node_ptr{this, node_type::I4}, key, child_index,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    // The first key in the node is GT the given key_byte.
    return parent_class::end_result;
  }

  /// Recursively delete all children.
  ///
  /// \param db_instance Database instance
  constexpr void delete_subtree(db_type& db_instance) noexcept {
    const std::uint8_t children_count_ = this->children_count.load();
    for (std::uint8_t i = 0; i < children_count_; ++i) {
      if constexpr (ArtPolicy::can_eliminate_leaf) {
        if (is_value_in_slot(i)) continue;  // packed value, nothing to free
      }
      ArtPolicy::delete_subtree(children[i], db_instance);
    }
  }

  /// Dump node contents to stream for debugging.
  ///
  /// \param os Output stream
  /// \param recursive Whether to dump children recursively
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os,
                                                bool recursive) const {
    parent_class::dump(os, recursive);
    const auto children_count_ = this->children_count.load();
    os << ", key_bytes:";
    for (std::uint8_t i = 0; i < children_count_; i++)
      dump_byte(os, keys.byte_array[i]);
    if (recursive) {
      os << ", children:  \n";
      for (std::uint8_t i = 0; i < children_count_; i++) {
        if (is_value_in_slot(i)) {
          os << "  [" << static_cast<unsigned>(i) << "] packed value\n";
        } else {
          ArtPolicy::dump_node(os, children[i].load());
        }
      }
    }
  }

 private:
  /// Initialize node with two children in sorted order.
  ///
  /// \param key1 Key byte for first child
  /// \param child1 First child pointer
  /// \param key2 Key byte for second child
  /// \param child2 Second child as unique pointer
  constexpr void add_two_to_empty(std::byte key1, node_ptr child1,
                                  std::byte key2,
                                  db_leaf_unique_ptr child2) noexcept {
    UNODB_DETAIL_ASSERT(key1 != key2);
    UNODB_DETAIL_ASSERT(this->children_count == 2);

    const std::uint8_t key1_i = key1 < key2 ? 0U : 1U;
    const std::uint8_t key2_i = 1U - key1_i;
    keys.byte_array[key1_i] = key1;
    children[key1_i] = child1;
    keys.byte_array[key2_i] = key2;
    children[key2_i] = node_ptr{child2.release(), node_type::LEAF};
#ifndef UNODB_DETAIL_X86_64
    keys.byte_array[2] = unused_key_byte;
    keys.byte_array[3] = unused_key_byte;
#endif

    UNODB_DETAIL_ASSERT(
        std::is_sorted(keys.byte_array.cbegin(),
                       keys.byte_array.cbegin() + this->children_count));
  }

  /// Add two children to an empty node (value-in-slot variant).
  constexpr void add_two_to_empty(std::byte key1, node_ptr child1,
                                  std::byte key2, node_ptr child2) noexcept {
    UNODB_DETAIL_ASSERT(key1 != key2);
    UNODB_DETAIL_ASSERT(this->children_count == 2);

    const std::uint8_t key1_i = key1 < key2 ? 0U : 1U;
    const std::uint8_t key2_i = 1U - key1_i;
    keys.byte_array[key1_i] = key1;
    children[key1_i] = child1;
    keys.byte_array[key2_i] = key2;
    children[key2_i] = child2;
    // child2 is always a packed value in the value-in-slot path.
    // child1 may be an inode (existing subtree) or a packed value.
    set_value_bit(key2_i);
#ifndef UNODB_DETAIL_X86_64
    keys.byte_array[2] = unused_key_byte;
    keys.byte_array[3] = unused_key_byte;
#endif

    UNODB_DETAIL_ASSERT(
        std::is_sorted(keys.byte_array.cbegin(),
                       keys.byte_array.cbegin() + this->children_count));
  }

  /// Union for key byte storage with integer access for SIMD.
  union key_union {
    /// Array access to individual key bytes.
    std::array<critical_section_policy<std::byte>, basic_inode_4::capacity>
        byte_array;
    /// Integer access for SIMD operations.
    critical_section_policy<std::uint32_t> integer;

    /// Default constructor.
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26495)
    constexpr key_union() noexcept {}
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  };

  /// Check if child at index holds a packed value (not a pointer).
 public:
  [[nodiscard]] constexpr bool is_value_in_slot(std::uint8_t i) const noexcept {
    return bitmask_base::test(i);
  }
  /// Mark child at index as holding a packed value.
  constexpr void set_value_bit(std::uint8_t i) noexcept {
    bitmask_base::set(i);
  }
  /// Mark child at index as holding a pointer (clear value bit).
  constexpr void clear_value_bit(std::uint8_t i) noexcept {
    bitmask_base::clear(i);
  }

  /// Key bytes for child lookup.
  key_union keys;

  static_assert(std::alignment_of_v<decltype(keys)> == 4);
  static_assert(sizeof(keys) == 4);

 protected:
  /// Child pointers array.
  std::array<critical_section_policy<node_ptr>, basic_inode_4::capacity>
      children;

  static_assert(sizeof(children) == 32);

 private:
#ifdef UNODB_DETAIL_X86_64
  /// Find insertion position using x86_64 SIMD comparison.
  ///
  /// \param insert_key_byte Key byte to insert
  /// \param node_key_mask Bitmask of valid key positions
  ///
  /// \return Index where new key should be inserted
  [[nodiscard]] auto get_insert_pos(std::uint8_t insert_key_byte,
                                    unsigned node_key_mask) const noexcept {
    UNODB_DETAIL_ASSERT(node_key_mask ==
                        (1U << this->children_count.load()) - 1);

    const auto replicated_insert_key_byte =
        _mm_set1_epi8(static_cast<char>(insert_key_byte));
    const auto node_keys_in_sse_reg =
        _mm_cvtsi32_si128(static_cast<std::int32_t>(keys.integer.load()));
    // Since the existing and insert key values cannot be equal, it's OK to use
    // "<=" comparison as "<".
    const auto le_node_key_positions =
        _mm_cmple_epu8(node_keys_in_sse_reg, replicated_insert_key_byte);
    const auto bit_field =
        static_cast<unsigned>(_mm_movemask_epi8(le_node_key_positions)) &
        node_key_mask;
    return static_cast<unsigned>(std::popcount(bit_field));
  }
#else
  /// Sentinel value for unused key slots in non-x86_64-SIMD implementations.
  static constexpr std::byte unused_key_byte{0xFF};
#endif

  template <class>
  friend class basic_inode_16;
  friend class basic_inode_impl<ArtPolicy>;
};  // class basic_inode_4

/// Type alias for basic_inode_16 parent class.
template <class ArtPolicy>
using basic_inode_16_parent = basic_inode<
    ArtPolicy, 5, 16, node_type::I16, typename ArtPolicy::inode4_type,
    typename ArtPolicy::inode48_type, typename ArtPolicy::inode16_type>;

/// Internal node with 5-16 children (N16).
///
/// Like in basic_inode_4, keys are maintained in sorted order and child
/// pointers correspond 1:1 with key positions. Uses SIMD (SSE2/AVX2/NEON) for
/// optimized key search when available.
///
/// \tparam ArtPolicy Policy class defining types and operations
template <class ArtPolicy>
class basic_inode_16
    : public basic_inode_16_parent<ArtPolicy>,
      private value_bitmask_field<ArtPolicy::can_eliminate_leaf,
                                  std::uint16_t> {
  /// Parent class type.
  using parent_class = basic_inode_16_parent<ArtPolicy>;
  /// Bitmask base (empty via EBO when can_eliminate_leaf is false).
  using bitmask_base =
      value_bitmask_field<ArtPolicy::can_eliminate_leaf, std::uint16_t>;

  using typename parent_class::inode16_type;
  using typename parent_class::inode48_type;
  using typename parent_class::inode4_type;
  using typename parent_class::leaf_type;
  using typename parent_class::node_ptr;

  /// Thread-safety policy for member access.
  template <typename T>
  using critical_section_policy =
      typename ArtPolicy::template critical_section_policy<T>;

 public:
  using typename parent_class::db_leaf_unique_ptr;
  using typename parent_class::db_type;
  using typename parent_class::find_result;
  using typename parent_class::tree_depth_type;

  /// Construct by growing from \a source_node of basic_inode_4 type.
  constexpr basic_inode_16(db_type&, const inode4_type& source_node) noexcept
      : parent_class{source_node} {}

  /// Construct by shrinking from \a source_node of basic_inode_48 type.
  constexpr basic_inode_16(db_type&, const inode48_type& source_node) noexcept
      : parent_class{source_node} {}

  /// Construct by growing from basic_inode_4 and adding a child.
  ///
  /// \param db_instance Database for memory tracking
  /// \param source_node N4 node to grow from
  /// \param child New child to add
  /// \param depth Current tree depth
  constexpr basic_inode_16(db_type& db_instance, inode4_type& source_node,
                           db_leaf_unique_ptr&& child,
                           [[maybe_unused]] tree_depth_type depth,
                           std::byte key_byte) noexcept
      : parent_class{source_node} {
    init(db_instance, source_node, std::move(child), depth, key_byte);
  }

  /// Construct by growing from basic_inode_4 (value-in-slot variant).
  constexpr basic_inode_16(db_type& db_instance, inode4_type& source_node,
                           node_ptr packed_value,
                           [[maybe_unused]] tree_depth_type depth,
                           std::byte key_byte) noexcept
      : parent_class{source_node} {
    init(db_instance, source_node, packed_value, depth, key_byte);
  }

  /// Construct by shrinking from basic_inode_48 and removing a child.
  ///
  /// \param db_instance Database instance
  /// \param source_node basic_inode_48 node to shrink from
  /// \param child_to_delete Index of child to remove
  constexpr basic_inode_16(db_type& db_instance, inode48_type& source_node,
                           std::uint8_t child_to_delete) noexcept
      : parent_class{source_node} {
    init(db_instance, source_node, child_to_delete);
  }

  /// Initialize by growing from basic_inode_4 and adding a child.
  ///
  /// \param db_instance Database instance
  /// \param source_node N4 node to grow from
  /// \param child New child to add
  /// \param depth Current tree depth
  constexpr void init(db_type& db_instance, inode4_type& source_node,
                      db_leaf_unique_ptr child,
                      [[maybe_unused]] tree_depth_type depth,
                      std::byte key_byte) noexcept {
    // This overload is only for trees with actual leaves.  When
    // can_eliminate_leaf is true, init_grow unconditionally sets the
    // value bit — which is correct because only the packed-value
    // overload below can be called in that mode.
    static_assert(!ArtPolicy::can_eliminate_leaf,
                  "leaf init must not be called when leaves are eliminated");
    init_grow(db_instance, source_node,
              node_ptr{child.release(), node_type::LEAF}, key_byte);
  }

  /// Initialize by growing from basic_inode_4 (value-in-slot variant).
  constexpr void init(db_type& db_instance, inode4_type& source_node,
                      node_ptr packed_value,
                      [[maybe_unused]] tree_depth_type depth,
                      std::byte key_byte) noexcept {
    init_grow(db_instance, source_node, packed_value, key_byte);
  }

  /// Common grow logic: insert child_val at sorted position.
  constexpr void init_grow(db_type& db_instance, inode4_type& source_node,
                           node_ptr child_val, std::byte key_byte) noexcept {
    const auto reclaim_source_node{
        ArtPolicy::template make_db_inode_reclaimable_ptr<inode4_type>(
            &source_node, db_instance)};
    const auto kb = static_cast<std::uint8_t>(key_byte);

#ifdef UNODB_DETAIL_X86_64
    const auto insert_pos_index = source_node.get_insert_pos(kb, 0xFU);
#else
    const auto keys_integer = source_node.keys.integer.load();
    const auto first_lt = ((keys_integer & 0xFFU) < kb) ? 1 : 0;
    const auto second_lt = (((keys_integer >> 8U) & 0xFFU) < kb) ? 1 : 0;
    const auto third_lt = (((keys_integer >> 16U) & 0xFFU) < kb) ? 1 : 0;
    const auto fourth_lt = (((keys_integer >> 24U) & 0xFFU) < kb) ? 1 : 0;
    const auto insert_pos_index =
        static_cast<unsigned>(first_lt + second_lt + third_lt + fourth_lt);
#endif

    unsigned i = 0;
    for (; i < insert_pos_index; ++i) {
      keys.byte_array[i] = source_node.keys.byte_array[i];
      children[i] = source_node.children[i];
    }

    UNODB_DETAIL_ASSUME(i < parent_class::capacity);

    keys.byte_array[i] = key_byte;
    children[i] = child_val;
    ++i;

    for (; i <= inode4_type::capacity; ++i) {
      keys.byte_array[i] = source_node.keys.byte_array[i - 1];
      children[i] = source_node.children[i - 1];
    }

    // Copy value bitmask from source, inserting the new child's bit.
    if constexpr (ArtPolicy::can_eliminate_leaf) {
      for (unsigned j = 0; j < insert_pos_index; ++j) {
        if (source_node.is_value_in_slot(static_cast<std::uint8_t>(j)))
          set_value_bit(static_cast<std::uint8_t>(j));
      }
      // The new child at insert_pos_index is a packed value
      // (this overload is only called for value-in-slot).
      set_value_bit(static_cast<std::uint8_t>(insert_pos_index));
      for (unsigned j = insert_pos_index; j < inode4_type::capacity; ++j) {
        if (source_node.is_value_in_slot(static_cast<std::uint8_t>(j)))
          set_value_bit(static_cast<std::uint8_t>(j + 1));
      }
    }
  }

  /// Initialize by shrinking from basic_inode_48 and removing a child.
  ///
  /// \param db_instance Database instance
  /// \param source_node basic_inode_48 node to shrink from
  /// \param child_to_delete Index of child to remove
  constexpr void init(db_type& db_instance, inode48_type& source_node,
                      std::uint8_t child_to_delete) noexcept {
    const auto reclaim_source_node{
        ArtPolicy::template make_db_inode_reclaimable_ptr<inode48_type>(
            &source_node, db_instance)};
    source_node.remove_child_pointer(child_to_delete, db_instance);
    source_node.child_indexes[child_to_delete] = inode48_type::empty_child;

    // TODO(laurynas): consider AVX2 gather?
    unsigned next_child = 0;
    unsigned i = 0;
    while (true) {
      const auto source_child_i = source_node.child_indexes[i].load();
      if (source_child_i != inode48_type::empty_child) {
        keys.byte_array[next_child] = static_cast<std::byte>(i);
        const auto source_child_ptr =
            source_node.children.pointer_array[source_child_i].load();
        UNODB_DETAIL_ASSERT(source_child_ptr != nullptr);
        children[next_child] = source_child_ptr;
        ++next_child;
        if (next_child == basic_inode_16::capacity) break;
      }
      UNODB_DETAIL_ASSERT(i < 255);
      ++i;
    }

    UNODB_DETAIL_ASSERT(this->children_count == basic_inode_16::capacity);
    UNODB_DETAIL_ASSERT(
        std::is_sorted(keys.byte_array.cbegin(),
                       keys.byte_array.cbegin() + basic_inode_16::capacity));

    if constexpr (ArtPolicy::can_eliminate_leaf) {
      // Copy bitmask from I48. I48 uses slot indices; we mapped them
      // to sequential positions in the loop above. Re-scan to set bits.
      next_child = 0;
      for (unsigned j = 0; j <= i; ++j) {
        const auto sci = source_node.child_indexes[j].load();
        if (sci != inode48_type::empty_child) {
          if (source_node.is_value_in_slot_by_ci(sci))
            set_value_bit(static_cast<std::uint8_t>(next_child));
          ++next_child;
          if (next_child == basic_inode_16::capacity) break;
        }
      }
    }
  }

  /// Add child to node that has available capacity.
  ///
  /// \param child Leaf child to add
  /// \param depth Current tree depth
  /// \param children_count_ Current children count
  ///
  /// \note The node already keeps its current children count, but all callers
  /// have already loaded it.
  constexpr void add_to_nonfull(db_leaf_unique_ptr&& child,
                                [[maybe_unused]] tree_depth_type depth,
                                std::byte key_byte,
                                std::uint8_t children_count_) noexcept {
    UNODB_DETAIL_ASSERT(children_count_ == this->children_count);
    UNODB_DETAIL_ASSERT(children_count_ < parent_class::capacity);
    UNODB_DETAIL_ASSERT(std::is_sorted(
        keys.byte_array.cbegin(), keys.byte_array.cbegin() + children_count_));

    const auto insert_pos_index =
        get_sorted_key_array_insert_position(key_byte);

    if (insert_pos_index != children_count_) {
      UNODB_DETAIL_ASSERT(insert_pos_index < children_count_);
      UNODB_DETAIL_ASSERT(keys.byte_array[insert_pos_index] != key_byte);

      std::copy_backward(keys.byte_array.cbegin() + insert_pos_index,
                         keys.byte_array.cbegin() + children_count_,
                         keys.byte_array.begin() + children_count_ + 1);
      std::copy_backward(children.begin() + insert_pos_index,
                         children.begin() + children_count_,
                         children.begin() + children_count_ + 1);
    }

    keys.byte_array[insert_pos_index] = key_byte;
    children[insert_pos_index] = node_ptr{child.release(), node_type::LEAF};
    ++children_count_;
    this->children_count = children_count_;

    UNODB_DETAIL_ASSERT(std::is_sorted(
        keys.byte_array.cbegin(), keys.byte_array.cbegin() + children_count_));
  }

  /// Add a packed value to a non-full node (value-in-slot variant).
  constexpr void add_to_nonfull(node_ptr packed_value,
                                [[maybe_unused]] tree_depth_type depth,
                                std::byte key_byte,
                                std::uint8_t children_count_) noexcept {
    UNODB_DETAIL_ASSERT(children_count_ == this->children_count);
    UNODB_DETAIL_ASSERT(children_count_ < parent_class::capacity);
    const auto insert_pos_index =
        get_sorted_key_array_insert_position(key_byte);
    if (insert_pos_index != children_count_) {
      std::copy_backward(keys.byte_array.cbegin() + insert_pos_index,
                         keys.byte_array.cbegin() + children_count_,
                         keys.byte_array.begin() + children_count_ + 1);
      std::copy_backward(children.begin() + insert_pos_index,
                         children.begin() + children_count_,
                         children.begin() + children_count_ + 1);
    }
    keys.byte_array[insert_pos_index] = key_byte;
    children[insert_pos_index] = packed_value;
    set_value_bit(static_cast<std::uint8_t>(insert_pos_index));
    ++children_count_;
    this->children_count = children_count_;
  }

  /// Remove child at given index.
  ///
  /// \param child_index Index of child to remove
  /// \param db_instance Database instance
  // MSVC C26815 false positive: reclaim object intentionally destroyed at
  // scope exit while db_instance (passed by caller) remains valid.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
  constexpr void remove(std::uint8_t child_index,
                        db_type& db_instance) noexcept {
    UNODB_DETAIL_ASSERT(child_index < this->children_count.load());

    if constexpr (!ArtPolicy::can_eliminate_leaf) {
      const auto r{ArtPolicy::reclaim_leaf_on_scope_exit(
          children[child_index].load().template ptr<leaf_type*>(),
          db_instance)};
    } else {
      (void)db_instance;
    }

    remove_child_entry(child_index);
  }

  /// Remove child entry without reclaiming the child.
  ///
  /// \param child_index Index of child to remove
  constexpr void remove_child_entry(std::uint8_t child_index) noexcept {
    auto children_count_ = this->children_count.load();
    UNODB_DETAIL_ASSERT(child_index < children_count_);
    UNODB_DETAIL_ASSERT(std::is_sorted(
        keys.byte_array.cbegin(), keys.byte_array.cbegin() + children_count_));

    for (unsigned i = child_index + 1U; i < children_count_; ++i) {
      keys.byte_array[i - 1] = keys.byte_array[i];
      children[i - 1] = children[i];
    }

    --children_count_;
    this->children_count = children_count_;

    // Shift bitmask bits to match shifted children array.
    if constexpr (ArtPolicy::can_eliminate_leaf) {
      bitmask_base::remove_at(child_index);
    }

    UNODB_DETAIL_ASSERT(std::is_sorted(
        keys.byte_array.cbegin(), keys.byte_array.cbegin() + children_count_));
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Find child by key byte.
  ///
  /// \param key_byte Key byte to search for
  ///
  /// \return Result with child index and pointer, or not found
  [[nodiscard, gnu::pure]] constexpr find_result find_child(
      std::byte key_byte) noexcept {
#ifdef UNODB_DETAIL_X86_64
    const auto replicated_search_key =
        _mm_set1_epi8(static_cast<char>(key_byte));
    const auto matching_key_positions =
        _mm_cmpeq_epi8(replicated_search_key, keys.byte_vector);
    const auto mask = (1U << this->children_count) - 1;
    const auto bit_field =
        static_cast<unsigned>(_mm_movemask_epi8(matching_key_positions)) & mask;
    if (bit_field != 0) {
      const auto i = static_cast<std::uint8_t>(std::countr_zero(bit_field));
      return std::make_pair(
          i, static_cast<critical_section_policy<node_ptr>*>(&children[i]));
    }
    return parent_class::child_not_found;
#elif defined(__aarch64__)
    const auto replicated_search_key =
        vdupq_n_u8(static_cast<std::uint8_t>(key_byte));
    const auto matching_key_positions =
        vceqq_u8(replicated_search_key, keys.byte_vector);
    const auto narrowed_positions =
        vshrn_n_u16(vreinterpretq_u16_u8(matching_key_positions), 4);
    const auto scalar_pos =
        // NOLINTNEXTLINE(misc-const-correctness)
        vget_lane_u64(vreinterpret_u64_u8(narrowed_positions), 0);
    const auto child_count = this->children_count.load();
    const auto mask = (child_count == 16) ? 0xFFFFFFFF'FFFFFFFFULL
                                          : (1ULL << (child_count << 2U)) - 1;
    const auto masked_pos = scalar_pos & mask;

    if (masked_pos == 0) return parent_class::child_not_found;

    const auto i = static_cast<unsigned>(std::countr_zero(masked_pos) >> 2U);
    return std::make_pair(
        i, static_cast<critical_section_policy<node_ptr>*>(&children[i]));
#else
    for (size_t i = 0; i < this->children_count.load(); ++i)
      if (key_byte == keys.byte_array[i])
        return std::make_pair(
            i, static_cast<critical_section_policy<node_ptr>*>(&children[i]));
    return parent_class::child_not_found;
#endif
  }

  /// Get child pointer at given index.
  ///
  /// \param child_index Index of child
  ///
  /// \return Child node pointer
  [[nodiscard, gnu::pure]] constexpr node_ptr get_child(
      std::uint8_t child_index) noexcept {
    return children[child_index].load();
  }

  /// Get iterator result for first child.
  ///
  /// \return Iterator result pointing to first child
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_16::iter_result
  begin() noexcept {
    const auto key = keys.byte_array[0].load();
    return {node_ptr{this, node_type::I16}, key, 0,
            this->get_key_prefix().get_snapshot()};
  }

  /// Get iterator result for last child.
  ///
  /// \return Iterator result pointing to last child
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_16::iter_result
  last() noexcept {
    const auto child_index{
        static_cast<std::uint8_t>(this->children_count.load() - 1)};
    const auto key = keys.byte_array[child_index].load();
    return {node_ptr{this, node_type::I16}, key, child_index,
            this->get_key_prefix().get_snapshot()};
  }

  /// Get iterator result for next child after given index.
  ///
  /// \param child_index Current child index
  ///
  /// \return Iterator result for next child, or empty if at end
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_16::iter_result_opt
  next(std::uint8_t child_index) noexcept {
    const auto nchildren{this->children_count.load()};
    const auto next_index{static_cast<std::uint8_t>(child_index + 1)};
    if (next_index >= nchildren) return parent_class::end_result;
    const auto key = keys.byte_array[next_index].load();
    return {{node_ptr{this, node_type::I16}, key, next_index,
             this->get_key_prefix().get_snapshot()}};
  }

  /// Get iterator result for previous child before given index.
  ///
  /// \param child_index Current child index
  ///
  /// \return Iterator result for previous child, or empty if at start
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_16::iter_result_opt
  prior(std::uint8_t child_index) noexcept {
    if (child_index == 0) return parent_class::end_result;
    const auto next_index{static_cast<std::uint8_t>(child_index - 1)};
    const auto key = keys.byte_array[next_index].load();
    return {{node_ptr{this, node_type::I16}, key, next_index,
             this->get_key_prefix().get_snapshot()}};
  }

  /// Find last child with key byte less than or equal to given value.
  ///
  /// \param key_byte Key byte to compare against
  ///
  /// \return Iterator result for matching child, or empty if none found
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_16::iter_result_opt
  lte_key_byte(std::byte key_byte) noexcept {
    const auto children_count_ = this->children_count.load();
    for (std::int64_t i = children_count_ - 1; i >= 0; i--) {
      const auto child_index = static_cast<std::uint8_t>(i);
      const auto key = keys.byte_array[child_index].load();
      if (key <= key_byte) {
        return {{node_ptr{this, node_type::I16}, key, child_index,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    // The first key in the node is GT the given key_byte.
    return parent_class::end_result;
  }

  /// Find first child with key byte greater than or equal to given value.
  ///
  /// \param key_byte Key byte to compare against
  ///
  /// \return Iterator result for matching child, or empty if none found
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_16::iter_result_opt
  gte_key_byte(std::byte key_byte) noexcept {
    const auto children_count_ = this->children_count.load();
    for (std::uint8_t i = 0; i < children_count_; ++i) {
      const auto key = keys.byte_array[i].load();
      if (key >= key_byte) {
        return {{node_ptr{this, node_type::I16}, key, i,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    // This should only occur if there is no entry in the keys[] which
    // is greater-than the given [key_byte].
    return parent_class::end_result;
  }

  /// Recursively delete all children.
  ///
  /// \param db_instance Database instance
  constexpr void delete_subtree(db_type& db_instance) noexcept {
    const uint8_t children_count_ = this->children_count.load();
    for (std::uint8_t i = 0; i < children_count_; ++i) {
      if constexpr (ArtPolicy::can_eliminate_leaf) {
        if (is_value_in_slot(i)) continue;
      }
      ArtPolicy::delete_subtree(children[i], db_instance);
    }
  }

  /// Dump node contents to stream for debugging.
  ///
  /// \param os Output stream
  /// \param recursive Whether to dump children recursively
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os,
                                                bool recursive) const {
    parent_class::dump(os, recursive);
    const auto children_count_ = this->children_count.load();
    os << ", key bytes =";
    for (std::uint8_t i = 0; i < children_count_; ++i)
      dump_byte(os, keys.byte_array[i]);
    if (recursive) {
      os << ", children:  \n";
      for (std::uint8_t i = 0; i < children_count_; ++i) {
        if (is_value_in_slot(i)) {
          os << "  [" << static_cast<unsigned>(i) << "] packed value\n";
        } else {
          ArtPolicy::dump_node(os, children[i].load());
        }
      }
    }
  }

 private:
  /// Find position in sorted key array for insertion.
  ///
  /// \param key_byte Key byte to find insertion position for
  ///
  /// \return Index where key should be inserted
  [[nodiscard, gnu::pure]] constexpr auto get_sorted_key_array_insert_position(
      std::byte key_byte) noexcept {
    const auto children_count_ = this->children_count.load();

    UNODB_DETAIL_ASSERT(children_count_ < basic_inode_16::capacity);
    UNODB_DETAIL_ASSERT(std::is_sorted(
        keys.byte_array.cbegin(), keys.byte_array.cbegin() + children_count_));
    UNODB_DETAIL_ASSERT(
        std::adjacent_find(keys.byte_array.cbegin(),
                           keys.byte_array.cbegin() + children_count_) >=
        keys.byte_array.cbegin() + children_count_);

#ifdef UNODB_DETAIL_X86_64
    const auto replicated_insert_key =
        _mm_set1_epi8(static_cast<char>(key_byte));
    const auto lesser_key_positions =
        _mm_cmple_epu8(replicated_insert_key, keys.byte_vector);
    const auto mask = (1U << children_count_) - 1;
    const auto bit_field =
        static_cast<unsigned>(_mm_movemask_epi8(lesser_key_positions)) & mask;
    const auto result =
        (bit_field != 0)
            ? static_cast<std::uint8_t>(std::countr_zero(bit_field))
            : static_cast<std::uint8_t>(children_count_);
#else
    // This is also the best current ARM implementation
    const auto result = static_cast<std::uint8_t>(
        std::lower_bound(keys.byte_array.cbegin(),
                         keys.byte_array.cbegin() + children_count_, key_byte) -
        keys.byte_array.cbegin());
#endif

    UNODB_DETAIL_ASSERT(
        result == children_count_ ||
        (result < children_count_ && keys.byte_array[result] != key_byte));
    return result;
  }

 protected:
  /// Union for key byte storage with SIMD vector access.
  union key_union {
    /// Array access to individual key bytes.
    std::array<critical_section_policy<std::byte>, basic_inode_16::capacity>
        byte_array;
#ifdef UNODB_DETAIL_X86_64
    /// SSE vector for SIMD operations.
    __m128i byte_vector;
#elif defined(__aarch64__)
    /// NEON vector for SIMD operations.
    uint8x16_t byte_vector;
#endif
    /// Default constructor.
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26495)
    key_union() noexcept {}
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  };

 public:
  [[nodiscard]] constexpr bool is_value_in_slot(std::uint8_t i) const noexcept {
    return bitmask_base::test(i);
  }
  constexpr void set_value_bit(std::uint8_t i) noexcept {
    bitmask_base::set(i);
  }
  constexpr void clear_value_bit(std::uint8_t i) noexcept {
    bitmask_base::clear(i);
  }

  /// Key bytes for child lookup.
  key_union keys;

  /// Child pointers array.
  std::array<critical_section_policy<node_ptr>, basic_inode_16::capacity>
      children;

 private:
  /// Sentinel value for empty child slot.
  static constexpr std::uint8_t empty_child = 0xFF;

  template <class>
  friend class basic_inode_4;
  friend class basic_inode_impl<ArtPolicy>;
  template <class>
  friend class basic_inode_48;
};  // class basic_inode_16

/// Type alias for basic_inode_48 parent class.
template <class ArtPolicy>
using basic_inode_48_parent = basic_inode<
    ArtPolicy, 17, 48, node_type::I48, typename ArtPolicy::inode16_type,
    typename ArtPolicy::inode256_type, typename ArtPolicy::inode48_type>;

/// Internal node with 17-48 children (N48).
///
/// Uses 256-byte child_indexes array directly indexed by key byte, storing
/// indices into 48-element child pointer array. Neither child_indexes nor
/// child pointers are dense. Uses SIMD for finding first empty slot.
///
/// \tparam ArtPolicy Policy class defining types and operations
template <class ArtPolicy>
class basic_inode_48
    : public basic_inode_48_parent<ArtPolicy>,
      private value_bitmask_field<ArtPolicy::can_eliminate_leaf,
                                  std::array<std::uint8_t, 6>> {
  /// Base class type alias.
  using parent_class = basic_inode_48_parent<ArtPolicy>;
  /// Bitmask base (empty via EBO when can_eliminate_leaf is false).
  using bitmask_base = value_bitmask_field<ArtPolicy::can_eliminate_leaf,
                                           std::array<std::uint8_t, 6>>;

  using typename parent_class::inode16_type;
  using typename parent_class::inode256_type;
  using typename parent_class::inode48_type;
  using typename parent_class::leaf_type;
  using typename parent_class::node_ptr;

  /// Thread-safety policy for member access.
  template <typename T>
  using critical_section_policy =
      typename ArtPolicy::template critical_section_policy<T>;

 public:
  using typename parent_class::db_leaf_unique_ptr;
  using typename parent_class::db_type;
  using typename parent_class::tree_depth_type;

  /// Construct by growing from \a source_node of basic_inode_16 type.
  constexpr basic_inode_48(db_type&, const inode16_type& source_node) noexcept
      : parent_class{source_node} {}

  /// Construct by shrinking from \a source_node of basic_inode_256 type.
  constexpr basic_inode_48(db_type&, const inode256_type& source_node) noexcept
      : parent_class{source_node} {}

  /// Construct by growing from basic_inode_16 and adding a child.
  ///
  /// \param db_instance Database instance
  /// \param source_node N16 node to grow from
  /// \param child New child to add
  /// \param depth Current tree depth
  constexpr basic_inode_48(db_type& db_instance,
                           inode16_type& __restrict source_node,
                           db_leaf_unique_ptr&& child,
                           [[maybe_unused]] tree_depth_type depth,
                           std::byte key_byte) noexcept
      : parent_class{source_node} {
    init(db_instance, source_node, std::move(child), depth, key_byte);
  }

  /// Construct by growing from basic_inode_16 (value-in-slot variant).
  constexpr basic_inode_48(db_type& db_instance,
                           inode16_type& __restrict source_node,
                           node_ptr packed_value,
                           [[maybe_unused]] tree_depth_type depth,
                           std::byte key_byte) noexcept
      : parent_class{source_node} {
    init(db_instance, source_node, packed_value, depth, key_byte);
  }

  /// Construct by shrinking from basic_inode_256 and removing a child.
  ///
  /// \param db_instance Database instance
  /// \param source_node N256 node to shrink from
  /// \param child_to_delete Key byte of child to remove
  constexpr basic_inode_48(db_type& db_instance,
                           inode256_type& __restrict source_node,
                           std::uint8_t child_to_delete) noexcept
      : parent_class{source_node} {
    init(db_instance, source_node, child_to_delete);
  }

  /// Initialize by growing from basic_inode_16 and adding a child.
  ///
  /// \param db_instance Database instance
  /// \param source_node N16 node to grow from
  /// \param child New child to add
  /// \param depth Current tree depth
  constexpr void init(db_type& db_instance,
                      inode16_type& __restrict source_node,
                      db_leaf_unique_ptr child,
                      [[maybe_unused]] tree_depth_type depth,
                      std::byte key_byte) noexcept {
    static_assert(!ArtPolicy::can_eliminate_leaf,
                  "leaf init must not be called when leaves are eliminated");
    init_grow(db_instance, source_node,
              node_ptr{child.release(), node_type::LEAF}, key_byte);
  }

  /// Initialize by growing from basic_inode_16 (value-in-slot variant).
  constexpr void init(db_type& db_instance,
                      inode16_type& __restrict source_node,
                      node_ptr packed_value,
                      [[maybe_unused]] tree_depth_type depth,
                      std::byte key_byte) noexcept {
    init_grow(db_instance, source_node, packed_value, key_byte);
  }

  /// Common grow logic from I16.
  constexpr void init_grow(db_type& db_instance,
                           inode16_type& __restrict source_node,
                           node_ptr child_val, std::byte key_byte) noexcept {
    const auto reclaim_source_node{
        ArtPolicy::template make_db_inode_reclaimable_ptr<inode16_type>(
            &source_node, db_instance)};

    std::uint8_t i = 0;
    for (; i < inode16_type::capacity; ++i) {
      const auto existing_key_byte = source_node.keys.byte_array[i].load();
      child_indexes[static_cast<std::uint8_t>(existing_key_byte)] = i;
    }
    for (i = 0; i < inode16_type::capacity; ++i) {
      children.pointer_array[i] = source_node.children[i];
    }

    UNODB_DETAIL_ASSERT(child_indexes[static_cast<std::uint8_t>(key_byte)] ==
                        empty_child);
    UNODB_DETAIL_ASSUME(i == inode16_type::capacity);

    child_indexes[static_cast<std::uint8_t>(key_byte)] = i;
    children.pointer_array[i] = child_val;
    if constexpr (ArtPolicy::can_eliminate_leaf) {
      // Copy bitmask from source I16 (indexed by position).
      for (std::uint8_t j = 0; j < inode16_type::capacity; ++j) {
        if (source_node.is_value_in_slot(j)) set_value_bit_by_ci(j);
      }
      // The new child is a packed value.
      set_value_bit_by_ci(i);
    }
    for (i = this->children_count; i < basic_inode_48::capacity; i++) {
      children.pointer_array[i] = node_ptr{nullptr};
    }
  }

  /// Initialize by shrinking from basic_inode_256 and removing a child.
  ///
  /// \param db_instance Database instance
  /// \param source_node N256 node to shrink from
  /// \param child_to_delete Key byte of child to remove
  // MSVC C26815 false positive: reclaim objects intentionally destroyed at
  // scope exit while db_instance (passed by caller) remains valid.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
  constexpr void init(db_type& db_instance,
                      inode256_type& __restrict source_node,
                      std::uint8_t child_to_delete) noexcept {
    const auto reclaim_source_node{
        ArtPolicy::template make_db_inode_reclaimable_ptr<inode256_type>(
            &source_node, db_instance)};
    [[maybe_unused]] const auto r{ArtPolicy::reclaim_if_leaf(
        source_node.children[child_to_delete].load(), db_instance)};

    source_node.children[child_to_delete] = node_ptr{nullptr};

    std::uint8_t next_child = 0;
    for (unsigned child_i = 0; child_i < 256; child_i++) {
      const auto child_ptr = source_node.children[child_i].load();
      if (child_ptr == nullptr) continue;

      child_indexes[child_i] = next_child;
      children.pointer_array[next_child] = source_node.children[child_i].load();
      if constexpr (ArtPolicy::can_eliminate_leaf) {
        if (source_node.is_value_in_slot(static_cast<std::uint8_t>(child_i)))
          set_value_bit_by_ci(next_child);
      }
      ++next_child;

      if (next_child == basic_inode_48::capacity) return;
    }
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Add child to non-full node.
  ///
  /// \param child Child leaf to add
  /// \param depth Current tree depth
  /// \param children_count_ Current child count
  ///
  /// \note The node already keeps its current children count, but all callers
  /// have already loaded it.
  constexpr void add_to_nonfull(db_leaf_unique_ptr&& child,
                                [[maybe_unused]] tree_depth_type depth,
                                std::byte key_byte,
                                std::uint8_t children_count_) noexcept {
    UNODB_DETAIL_ASSERT(this->children_count == children_count_);
    UNODB_DETAIL_ASSERT(children_count_ >= parent_class::min_size);
    UNODB_DETAIL_ASSERT(children_count_ < parent_class::capacity);

    UNODB_DETAIL_ASSERT(child_indexes[static_cast<std::uint8_t>(key_byte)] ==
                        empty_child);
    unsigned i{0};
#ifdef UNODB_DETAIL_SSE4_2
    const auto nullptr_vector = _mm_setzero_si128();
    while (true) {
      const auto ptr_vec0 = _mm_load_si128(&children.pointer_vector[i]);
      const auto ptr_vec1 = _mm_load_si128(&children.pointer_vector[i + 1]);
      const auto ptr_vec2 = _mm_load_si128(&children.pointer_vector[i + 2]);
      const auto ptr_vec3 = _mm_load_si128(&children.pointer_vector[i + 3]);
      const auto vec0_cmp = _mm_cmpeq_epi64(ptr_vec0, nullptr_vector);
      const auto vec1_cmp = _mm_cmpeq_epi64(ptr_vec1, nullptr_vector);
      const auto vec2_cmp = _mm_cmpeq_epi64(ptr_vec2, nullptr_vector);
      const auto vec3_cmp = _mm_cmpeq_epi64(ptr_vec3, nullptr_vector);
      // OK to treat 64-bit comparison result as 32-bit vector: we need to find
      // the first 0xFF only.
      const auto vec01_cmp = _mm_packs_epi32(vec0_cmp, vec1_cmp);
      const auto vec23_cmp = _mm_packs_epi32(vec2_cmp, vec3_cmp);
      const auto vec_cmp = _mm_packs_epi32(vec01_cmp, vec23_cmp);
      const auto cmp_mask =
          static_cast<std::uint64_t>(_mm_movemask_epi8(vec_cmp));
      if (cmp_mask != 0) {
        i = (i << 1U) +
            ((static_cast<unsigned>(std::countr_zero(cmp_mask)) + 1U) >> 1U);
        break;
      }
      i += 4;
    }
#elif defined(UNODB_DETAIL_AVX2)
    const auto nullptr_vector = _mm256_setzero_si256();
    while (true) {
      const auto ptr_vec0 = _mm256_load_si256(&children.pointer_vector[i]);
      const auto ptr_vec1 = _mm256_load_si256(&children.pointer_vector[i + 1]);
      const auto ptr_vec2 = _mm256_load_si256(&children.pointer_vector[i + 2]);
      const auto ptr_vec3 = _mm256_load_si256(&children.pointer_vector[i + 3]);
      const auto vec0_cmp = _mm256_cmpeq_epi64(ptr_vec0, nullptr_vector);
      const auto vec1_cmp = _mm256_cmpeq_epi64(ptr_vec1, nullptr_vector);
      const auto vec2_cmp = _mm256_cmpeq_epi64(ptr_vec2, nullptr_vector);
      const auto vec3_cmp = _mm256_cmpeq_epi64(ptr_vec3, nullptr_vector);
      const auto interleaved_vec01_cmp = _mm256_packs_epi32(vec0_cmp, vec1_cmp);
      const auto interleaved_vec23_cmp = _mm256_packs_epi32(vec2_cmp, vec3_cmp);
      const auto doubly_interleaved_vec_cmp =
          _mm256_packs_epi32(interleaved_vec01_cmp, interleaved_vec23_cmp);
      if (!_mm256_testz_si256(doubly_interleaved_vec_cmp,
                              doubly_interleaved_vec_cmp)) {
        const auto vec01_cmp =
            _mm256_permute4x64_epi64(interleaved_vec01_cmp, 0b11'01'10'00);
        const auto vec23_cmp =
            _mm256_permute4x64_epi64(interleaved_vec23_cmp, 0b11'01'10'00);
        const auto interleaved_vec_cmp =
            _mm256_packs_epi32(vec01_cmp, vec23_cmp);
        const auto vec_cmp =
            _mm256_permute4x64_epi64(interleaved_vec_cmp, 0b11'01'10'00);
        const auto cmp_mask =
            static_cast<std::uint64_t>(_mm256_movemask_epi8(vec_cmp));
        i = (i << 2U) +
            (static_cast<unsigned>(std::countr_zero(cmp_mask)) >> 1U);
        break;
      }
      i += 4;
    }
#elif defined(__aarch64__)
    const auto nullptr_vector = vdupq_n_u64(0);
    while (true) {
      const auto ptr_vec0 = children.pointer_vector[i];
      const auto ptr_vec1 = children.pointer_vector[i + 1];
      const auto ptr_vec2 = children.pointer_vector[i + 2];
      const auto ptr_vec3 = children.pointer_vector[i + 3];
      const auto vec0_cmp = vceqq_u64(nullptr_vector, ptr_vec0);
      const auto vec1_cmp = vceqq_u64(nullptr_vector, ptr_vec1);
      const auto vec2_cmp = vceqq_u64(nullptr_vector, ptr_vec2);
      const auto vec3_cmp = vceqq_u64(nullptr_vector, ptr_vec3);
      const auto narrowed_cmp0 = vshrn_n_u64(vec0_cmp, 4);
      const auto narrowed_cmp1 = vshrn_n_u64(vec1_cmp, 4);
      const auto narrowed_cmp2 = vshrn_n_u64(vec2_cmp, 4);
      const auto narrowed_cmp3 = vshrn_n_u64(vec3_cmp, 4);
      const auto cmp01 = vcombine_u32(narrowed_cmp0, narrowed_cmp1);
      const auto cmp23 = vcombine_u32(narrowed_cmp2, narrowed_cmp3);
      // NOLINTNEXTLINE(misc-const-correctness)
      const auto narrowed_cmp01 = vshrn_n_u32(cmp01, 4);
      // NOLINTNEXTLINE(misc-const-correctness)
      const auto narrowed_cmp23 = vshrn_n_u32(cmp23, 4);
      const auto cmp = vcombine_u16(narrowed_cmp01, narrowed_cmp23);
      // NOLINTNEXTLINE(misc-const-correctness)
      const auto narrowed_cmp = vshrn_n_u16(cmp, 4);
      const auto scalar_pos =
          // NOLINTNEXTLINE(misc-const-correctness)
          vget_lane_u64(vreinterpret_u64_u8(narrowed_cmp), 0);
      if (scalar_pos != 0) {
        i = (i << 1U) +
            static_cast<unsigned>(std::countr_zero(scalar_pos) >> 3U);
        break;
      }
      i += 4;
    }
#else   // #ifdef UNODB_DETAIL_X86_64
    node_ptr child_ptr;
    while (true) {
      child_ptr = children.pointer_array[i];
      if (child_ptr == nullptr) break;
      UNODB_DETAIL_ASSERT(i < 255);
      ++i;
    }
#endif  // #ifdef UNODB_DETAIL_X86_64

    UNODB_DETAIL_ASSUME(i < parent_class::capacity);

#ifndef NDEBUG
    UNODB_DETAIL_ASSERT(children.pointer_array[i] == nullptr);
    for (unsigned j = 0; j < i; ++j)
      UNODB_DETAIL_ASSERT(children.pointer_array[j] != nullptr);
#endif

    child_indexes[static_cast<std::uint8_t>(key_byte)] =
        static_cast<std::uint8_t>(i);
    children.pointer_array[i] = node_ptr{child.release(), node_type::LEAF};
    this->children_count = children_count_ + 1U;
  }

  /// Add a packed value to a non-full node (value-in-slot variant).
  constexpr void add_to_nonfull(node_ptr packed_value,
                                [[maybe_unused]] tree_depth_type depth,
                                std::byte key_byte,
                                std::uint8_t children_count_) noexcept {
    // Reuse the leaf overload by wrapping the packed value in a
    // temporary unique_ptr that immediately releases.  This avoids
    // duplicating the SIMD slot-finding logic.
    // TODO(#707): refactor to share slot-finding code without this hack.
    (void)children_count_;
    UNODB_DETAIL_ASSERT(child_indexes[static_cast<std::uint8_t>(key_byte)] ==
                        empty_child);
    // Find first empty slot (same logic as leaf version, simplified).
    unsigned slot = 0;
    while (children.pointer_array[slot] != nullptr) ++slot;
    child_indexes[static_cast<std::uint8_t>(key_byte)] =
        static_cast<std::uint8_t>(slot);
    children.pointer_array[slot] = packed_value;
    set_value_bit(static_cast<std::uint8_t>(key_byte));
    this->children_count = this->children_count + 1U;
  }

  /// Remove child at given key byte index.
  ///
  /// \param child_index Key byte of child to remove
  /// \param db_instance Database for memory reclamation
  constexpr void remove(std::uint8_t child_index,
                        db_type& db_instance) noexcept {
    remove_child_pointer(child_index, db_instance);
    remove_child_entry(child_index);
  }

  /// Remove child entry without reclaiming the child.
  ///
  /// \param child_index Key byte of child to remove
  constexpr void remove_child_entry(std::uint8_t child_index) noexcept {
    if constexpr (ArtPolicy::can_eliminate_leaf) {
      clear_value_bit(child_index);
    }
    children.pointer_array[child_indexes[child_index]] = node_ptr{nullptr};
    child_indexes[child_index] = empty_child;
    --this->children_count;
  }

  /// Find child by key byte.
  ///
  /// \param key_byte Key byte to search for
  ///
  /// \return Result with child index and pointer, or not found
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_48::find_result
  find_child(std::byte key_byte) noexcept {
    const auto child_i =
        child_indexes[static_cast<std::uint8_t>(key_byte)].load();
    if (child_i != empty_child) {
      return std::make_pair(static_cast<std::uint8_t>(key_byte),
                            &children.pointer_array[child_i]);
    }
    return parent_class::child_not_found;
  }

  /// Get child pointer at given key byte index.
  ///
  /// Indirects through child_indexes array to find actual child pointer.
  /// Returns `nullptr` if index is empty.
  ///
  /// \param child_index Key byte index of child
  ///
  /// \return Child node pointer, or nullptr if empty slot
  // N48: This is the case where we need to indirect through child_indices.
  [[nodiscard, gnu::pure]] constexpr node_ptr get_child(
      std::uint8_t child_index) noexcept {
    const auto child_i = child_indexes[child_index].load();
    // In a data race, the child_indices[] can be concurrently
    // modified, which will cause the OLC version tag to get
    // bumped. However, we are in the middle of reading and acting on
    // the data while that happens.  This can cause the value stored
    // in child_indices[] at our desired child_index to be empty_child
    // (0xFF).  In this circumstance, the caller will correctly detect
    // a problem when they do read_critical_section::check(), but we
    // will have still indirected beyond the end of the allocation and
    // ASAN can fail us.  To prevent that and read only the data that
    // is legally allocated to the node, we return nullptr in this
    // case and rely on the caller to detect a problem when they call
    // read_critical_section::check().
    return UNODB_DETAIL_UNLIKELY(child_i == empty_child)
               ? node_ptr()  // aka nullptr
               : children.pointer_array[child_i].load();
  }

  /// Get iterator result for first child.
  ///
  /// Scans child_indexes[256] for first mapped entry (smallest key).
  ///
  /// \return Iterator result pointing to first child
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_48::iter_result
  begin() noexcept {
    for (std::uint64_t i = 0; i < 256; i++) {
      // cppcheck-suppress useStlAlgorithm
      if (child_indexes[i] != empty_child) {
        const auto key = static_cast<std::byte>(i);
        const auto child_index = static_cast<std::uint8_t>(i);
        return {node_ptr{this, node_type::I48}, key, child_index,
                this->get_key_prefix().get_snapshot()};
      }
    }
    // because we always have at least 17 keys.
    UNODB_DETAIL_CANNOT_HAPPEN();  // LCOV_EXCL_LINE
  }

  /// Get iterator result for last child.
  ///
  /// Scans child_indexes[256] in reverse for last mapped entry (greatest key).
  ///
  /// \return Iterator result pointing to last child
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_48::iter_result
  last() noexcept {
    for (std::int64_t i = 255; i >= 0; i--) {
      if (child_indexes[static_cast<std::uint8_t>(i)] != empty_child) {
        const auto key = static_cast<std::byte>(i);
        const auto child_index = static_cast<std::uint8_t>(i);
        return {node_ptr{this, node_type::I48}, key, child_index,
                this->get_key_prefix().get_snapshot()};
      }
    }
    // because we always have at least 17 keys.
    UNODB_DETAIL_CANNOT_HAPPEN();  // LCOV_EXCL_LINE
  }

  /// Get iterator result for next child after given index.
  ///
  /// \param child_index Current key byte index
  ///
  /// \return Iterator result for next child, or empty if at end
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_48::iter_result_opt
  next(std::uint8_t child_index) noexcept {
    // loop over the remaining byte values in lexical order.
    for (auto i = static_cast<std::uint64_t>(child_index) + 1; i < 256; i++) {
      // cppcheck-suppress useStlAlgorithm
      if (child_indexes[i] != empty_child) {
        const auto key = static_cast<std::byte>(i);
        const auto next_index = static_cast<std::uint8_t>(i);
        return {{node_ptr{this, node_type::I48}, key, next_index,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    return parent_class::end_result;
  }

  /// Get iterator result for previous child before given index.
  ///
  /// \param child_index Current key byte index
  ///
  /// \return Iterator result for previous child, or empty if at start
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_48::iter_result_opt
  prior(std::uint8_t child_index) noexcept {
    // loop over the prior byte values in lexical order.
    for (auto i = static_cast<std::int64_t>(child_index) - 1; i >= 0; i--) {
      if (child_indexes[static_cast<std::uint8_t>(i)] != empty_child) {
        const auto key = static_cast<std::byte>(i);
        const auto next_index = static_cast<std::uint8_t>(i);
        return {{node_ptr{this, node_type::I48}, key, next_index,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    return parent_class::end_result;
  }

  /// Find last child with key byte less than or equal to given value.
  ///
  /// \param key_byte Key byte to compare against
  ///
  /// \return Iterator result for matching child, or empty if none found
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_48::iter_result_opt
  lte_key_byte(std::byte key_byte) noexcept {
    // loop over the prior byte values in lexical order.
    for (auto i = static_cast<std::int64_t>(key_byte); i >= 0; i--) {
      const auto child_index = static_cast<std::uint8_t>(i);
      if (child_indexes[child_index] != empty_child) {
        const auto key = static_cast<std::byte>(i);
        return {{node_ptr{this, node_type::I48}, key, child_index,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    return parent_class::end_result;
  }

  /// Find first child with key byte greater than or equal to given value.
  ///
  /// \param key_byte Key byte to compare against
  ///
  /// \return Iterator result for matching child, or empty if none found
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_48::iter_result_opt
  gte_key_byte(std::byte key_byte) noexcept {
    // loop over the remaining byte values in lexical order.
    for (auto i = static_cast<std::uint64_t>(key_byte); i < 256; i++) {
      const auto child_index = static_cast<std::uint8_t>(i);
      if (child_indexes[child_index] != empty_child) {
        const auto key = static_cast<std::byte>(i);
        return {{node_ptr{this, node_type::I48}, key, child_index,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    return parent_class::end_result;
  }

  /// Recursively delete all children.
  ///
  /// \param db_instance Database instance
  constexpr void delete_subtree(db_type& db_instance) noexcept {
#ifndef NDEBUG
    const auto children_count_ = this->children_count.load();
    unsigned actual_children_count = 0;
#endif

    for (unsigned i = 0; i < this->capacity; ++i) {
      const auto child = children.pointer_array[i].load();
      if (child != nullptr) {
        if constexpr (ArtPolicy::can_eliminate_leaf) {
          if (!is_value_in_slot_by_ci(static_cast<std::uint8_t>(i))) {
            ArtPolicy::delete_subtree(child, db_instance);
          }
        } else {
          ArtPolicy::delete_subtree(child, db_instance);
        }
#ifndef NDEBUG
        ++actual_children_count;
        UNODB_DETAIL_ASSERT(actual_children_count <= children_count_);
#endif
      }
    }
    UNODB_DETAIL_ASSERT(actual_children_count == children_count_);
  }

  /// Dump node contents to stream for debugging.
  ///
  /// \param os Output stream
  /// \param recursive Whether to dump children recursively
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os,
                                                bool recursive) const {
    parent_class::dump(os, recursive);
#ifndef NDEBUG
    const auto children_count_ = this->children_count.load();
    unsigned actual_children_count = 0;
#endif

    os << ", key bytes & child indexes\n";
    for (unsigned i = 0; i < 256; i++)
      if (child_indexes[i] != empty_child) {
        os << " ";
        dump_byte(os, static_cast<std::byte>(i));
        os << ", child index = " << static_cast<unsigned>(child_indexes[i])
           << ": ";
        UNODB_DETAIL_ASSERT(children.pointer_array[child_indexes[i]] !=
                            nullptr);
        if (recursive) {
          const auto ci = child_indexes[i].load();
          if (is_value_in_slot_by_ci(ci)) {
            os << "packed value\n";
          } else {
            ArtPolicy::dump_node(os, children.pointer_array[ci].load());
          }
        }
#ifndef NDEBUG
        ++actual_children_count;
        UNODB_DETAIL_ASSERT(actual_children_count <= children_count_);
#endif
      }

    UNODB_DETAIL_ASSERT(actual_children_count == children_count_);
  }

 private:
  /// Remove child pointer by key byte index.
  ///
  /// \param child_index Key byte of child to remove
  /// \param db_instance Database for memory reclamation
  constexpr void remove_child_pointer(std::uint8_t child_index,
                                      db_type& db_instance) noexcept {
    direct_remove_child_pointer(child_indexes[child_index], db_instance);
  }

  /// Remove child pointer by direct children array index.
  ///
  /// \param children_i Index in children.pointer_array
  /// \param db_instance Database for memory reclamation
  // MSVC C26815 false positive: reclaim object intentionally destroyed at
  // scope exit while db_instance (passed by caller) remains valid.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
  constexpr void direct_remove_child_pointer(std::uint8_t children_i,
                                             db_type& db_instance) noexcept {
    UNODB_DETAIL_ASSERT(children_i != empty_child);

    [[maybe_unused]] const auto r{ArtPolicy::reclaim_if_leaf(
        children.pointer_array[children_i].load(), db_instance)};
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Sentinel value for empty child slot.
  static constexpr std::uint8_t empty_child = 0xFF;

 public:
  /// For I48, the child_index from find_child is the key byte.
  /// Resolve to children array index before accessing the bitmask.
  [[nodiscard]] constexpr bool is_value_in_slot(
      std::uint8_t key_byte_i) const noexcept {
    if constexpr (!ArtPolicy::can_eliminate_leaf) return false;
    const auto ci = child_indexes[key_byte_i].load();
    if (ci == empty_child) return false;
    return is_value_in_slot_by_ci(ci);
  }
  constexpr void set_value_bit(std::uint8_t key_byte_i) noexcept {
    if constexpr (!ArtPolicy::can_eliminate_leaf) return;
    const auto ci = child_indexes[key_byte_i].load();
    UNODB_DETAIL_ASSERT(ci != empty_child);
    bitmask_base::set(ci);
  }
  constexpr void clear_value_bit(std::uint8_t key_byte_i) noexcept {
    if constexpr (!ArtPolicy::can_eliminate_leaf) return;
    const auto ci = child_indexes[key_byte_i].load();
    UNODB_DETAIL_ASSERT(ci != empty_child);
    bitmask_base::clear(ci);
  }
  /// Check by children array index (for internal iteration).
  [[nodiscard]] constexpr bool is_value_in_slot_by_ci(
      std::uint8_t ci) const noexcept {
    return bitmask_base::test(ci);
  }
  constexpr void set_value_bit_by_ci(std::uint8_t ci) noexcept {
    bitmask_base::set(ci);
  }
  // The only way I found to initialize this array so that everyone is happy and
  // efficient. In the case of OLC, a std::fill compiles to a loop doing a
  // single byte per iteration. memset is likely an UB, and atomic_ref is not
  // available in C++17, and I don't like using it anyway, because this variable
  // *is* atomic.
  std::array<critical_section_policy<std::uint8_t>, 256> child_indexes{
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child, empty_child, empty_child, empty_child, empty_child,
      empty_child};

  /// Union for child pointer storage with SIMD vector access.
  union children_union {
    /// Array access to child pointers.
    std::array<critical_section_policy<node_ptr>, basic_inode_48::capacity>
        pointer_array;
#ifdef UNODB_DETAIL_SSE4_2
    static_assert(basic_inode_48::capacity % 8 == 0);
    // No std::array below because it would ignore the alignment attribute
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    /// SSE vector for SIMD operations.
    __m128i
        pointer_vector[basic_inode_48::capacity / 2];  // NOLINT(runtime/arrays)
#elif defined(UNODB_DETAIL_AVX2)
    static_assert(basic_inode_48::capacity % 16 == 0);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    /// AVX vector for SIMD operations.
    __m256i
        pointer_vector[basic_inode_48::capacity / 4];  // NOLINT(runtime/arrays)
#elif defined(__aarch64__)
    static_assert(basic_inode_48::capacity % 8 == 0);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    /// NEON vector for SIMD operations.
    uint64x2_t
        pointer_vector[basic_inode_48::capacity / 2];  // NOLINT(runtime/arrays)
#endif

    /// Default constructor.
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26495)
    children_union() noexcept {}
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  };

  /// Child pointers array.
  children_union children;

  template <class>
  friend class basic_inode_16;
  friend class basic_inode_impl<ArtPolicy>;
  template <class>
  friend class basic_inode_256;
};  // class basic_inode_48

/// Type alias for basic_inode_256 parent class.
template <class ArtPolicy>
using basic_inode_256_parent =
    basic_inode<ArtPolicy, 49, 256, node_type::I256,
                typename ArtPolicy::inode48_type, fake_inode,
                typename ArtPolicy::inode256_type>;

/// Internal node with 49-256 children (N256).
///
/// Uses key byte as direct index into 256-element child pointer array.
/// No separate keys array is needed. This is the largest internal node
/// type and cannot grow further.
///
/// \tparam ArtPolicy Policy class defining types and operations
/// \sa basic_inode for inherited template parameters
template <class ArtPolicy>
class basic_inode_256
    : public basic_inode_256_parent<ArtPolicy>,
      private value_bitmask_field<ArtPolicy::can_eliminate_leaf,
                                  std::array<std::uint8_t, 32>> {
  /// Base class type alias.
  using parent_class = basic_inode_256_parent<ArtPolicy>;
  /// Bitmask base (empty via EBO when can_eliminate_leaf is false).
  using bitmask_base = value_bitmask_field<ArtPolicy::can_eliminate_leaf,
                                           std::array<std::uint8_t, 32>>;

  using typename parent_class::inode48_type;
  using typename parent_class::leaf_type;
  using typename parent_class::node_ptr;

  /// Thread-safety policy for member access.
  template <typename T>
  using critical_section_policy =
      typename ArtPolicy::template critical_section_policy<T>;

 public:
  using typename parent_class::db_leaf_unique_ptr;
  using typename parent_class::db_type;
  using typename parent_class::tree_depth_type;

  /// Construct by growing from \a source_node of basic_inode_48 type.
  constexpr basic_inode_256(db_type&, const inode48_type& source_node) noexcept
      : parent_class{source_node} {}

  /// Construct by growing from basic_inode_48 and adding a child.
  ///
  /// \param db_instance Database for memory tracking
  /// \param source_node N48 node to grow from
  /// \param child New child to add
  /// \param depth Current tree depth
  constexpr basic_inode_256(db_type& db_instance, inode48_type& source_node,
                            db_leaf_unique_ptr&& child,
                            [[maybe_unused]] tree_depth_type depth,
                            std::byte key_byte) noexcept
      : parent_class{source_node} {
    init(db_instance, source_node, std::move(child), depth, key_byte);
  }

  /// Construct by growing from basic_inode_48 (value-in-slot variant).
  constexpr basic_inode_256(db_type& db_instance, inode48_type& source_node,
                            node_ptr packed_value,
                            [[maybe_unused]] tree_depth_type depth,
                            std::byte key_byte) noexcept
      : parent_class{source_node} {
    init(db_instance, source_node, packed_value, depth, key_byte);
  }

  /// Initialize by growing from basic_inode_48 and adding a child.
  ///
  /// \param db_instance Database for memory tracking
  /// \param source_node N48 node to grow from
  /// \param child New child to add
  /// \param depth Current tree depth
  constexpr void init(db_type& db_instance,
                      inode48_type& __restrict source_node,
                      db_leaf_unique_ptr child,
                      [[maybe_unused]] tree_depth_type depth,
                      std::byte key_byte) noexcept {
    static_assert(!ArtPolicy::can_eliminate_leaf,
                  "leaf init must not be called when leaves are eliminated");
    init_grow(db_instance, source_node,
              node_ptr{child.release(), node_type::LEAF}, key_byte);
  }

  /// Initialize by growing from basic_inode_48 (value-in-slot variant).
  constexpr void init(db_type& db_instance,
                      inode48_type& __restrict source_node,
                      node_ptr packed_value,
                      [[maybe_unused]] tree_depth_type depth,
                      std::byte key_byte) noexcept {
    init_grow(db_instance, source_node, packed_value, key_byte);
  }

  /// Common grow logic from I48.
  constexpr void init_grow(db_type& db_instance,
                           inode48_type& __restrict source_node,
                           node_ptr child_val, std::byte key_byte) noexcept {
    const auto reclaim_source_node{
        ArtPolicy::template make_db_inode_reclaimable_ptr<inode48_type>(
            &source_node, db_instance)};
    unsigned children_copied = 0;
    unsigned i = 0;
    while (true) {
      const auto children_i = source_node.child_indexes[i].load();
      if (children_i == inode48_type::empty_child) {
        children[i] = node_ptr{nullptr};
      } else {
        children[i] = source_node.children.pointer_array[children_i].load();
        ++children_copied;
        if (children_copied == inode48_type::capacity) break;
      }
      ++i;
    }

    ++i;
    for (; i < basic_inode_256::capacity; ++i) children[i] = node_ptr{nullptr};

    UNODB_DETAIL_ASSERT(children[static_cast<std::uint8_t>(key_byte)] ==
                        nullptr);
    children[static_cast<std::uint8_t>(key_byte)] = child_val;
    if constexpr (ArtPolicy::can_eliminate_leaf) {
      // Copy bitmask from source I48. I48 indexes by slot position,
      // I256 indexes by key byte. Map through child_indexes.
      for (unsigned j = 0; j < 256; ++j) {
        const auto ci = source_node.child_indexes[j].load();
        if (ci != inode48_type::empty_child &&
            source_node.is_value_in_slot_by_ci(ci)) {
          set_value_bit(static_cast<std::uint8_t>(j));
        }
      }
      set_value_bit(static_cast<std::uint8_t>(key_byte));
    }
  }

  /// Add child to non-full node.
  ///
  /// \param child Child leaf to add
  /// \param depth Current tree depth
  /// \param children_count_ Current child count
  ///
  /// \note The node already keeps its current children count, but all callers
  /// have already loaded it.
  constexpr void add_to_nonfull(db_leaf_unique_ptr&& child,
                                [[maybe_unused]] tree_depth_type depth,
                                std::byte key_byte,
                                std::uint8_t children_count_) noexcept {
    UNODB_DETAIL_ASSERT(this->children_count == children_count_);
    UNODB_DETAIL_ASSERT(children_count_ < parent_class::capacity);

    UNODB_DETAIL_ASSERT(children[static_cast<std::uint8_t>(key_byte)] ==
                        nullptr);
    children[static_cast<std::uint8_t>(key_byte)] =
        node_ptr{child.release(), node_type::LEAF};
    this->children_count = static_cast<std::uint8_t>(children_count_ + 1U);
  }

  /// Add a packed value to a non-full node (value-in-slot variant).
  constexpr void add_to_nonfull(node_ptr packed_value,
                                [[maybe_unused]] tree_depth_type depth,
                                std::byte key_byte,
                                std::uint8_t children_count_) noexcept {
    UNODB_DETAIL_ASSERT(this->children_count == children_count_);
    UNODB_DETAIL_ASSERT(children[static_cast<std::uint8_t>(key_byte)] ==
                        nullptr);
    children[static_cast<std::uint8_t>(key_byte)] = packed_value;
    set_value_bit(static_cast<std::uint8_t>(key_byte));
    this->children_count = static_cast<std::uint8_t>(children_count_ + 1U);
  }

  /// Remove child at given key byte index.
  ///
  /// \param child_index Key byte of child to remove
  /// \param db_instance Database for memory reclamation
  // MSVC C26815 false positive: reclaim object intentionally destroyed at
  // scope exit while db_instance (passed by caller) remains valid.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
  constexpr void remove(std::uint8_t child_index,
                        db_type& db_instance) noexcept {
    if constexpr (!ArtPolicy::can_eliminate_leaf) {
      const auto r{ArtPolicy::reclaim_leaf_on_scope_exit(
          children[child_index].load().template ptr<leaf_type*>(),
          db_instance)};
    } else {
      (void)db_instance;
    }

    remove_child_entry(child_index);
  }

  /// Remove child entry without reclaiming the child.
  ///
  /// \param child_index Key byte of child to remove
  constexpr void remove_child_entry(std::uint8_t child_index) noexcept {
    if constexpr (ArtPolicy::can_eliminate_leaf) {
      clear_value_bit(child_index);
    }
    children[child_index] = node_ptr{nullptr};
    --this->children_count;
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Find child by key byte.
  ///
  /// \param key_byte Key byte to search for
  ///
  /// \return Result with child index and pointer, or not found
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_256::find_result
  find_child(std::byte key_byte) noexcept {
    const auto key_int_byte = static_cast<std::uint8_t>(key_byte);
    if (children[key_int_byte] != nullptr)
      return std::make_pair(key_int_byte, &children[key_int_byte]);
    return parent_class::child_not_found;
  }

  /// Get child pointer at given key byte index.
  ///
  /// \param child_index Key byte index of child
  ///
  /// \return Child node pointer
  [[nodiscard, gnu::pure]] constexpr node_ptr get_child(
      std::uint8_t child_index) noexcept {
    return children[child_index].load();
  }

  /// Get iterator result for first child.
  ///
  /// Scans children array for first non-null entry (smallest key).
  ///
  /// \return Iterator result pointing to first child
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_256::iter_result
  begin() noexcept {
    for (std::uint64_t i = 0; i < basic_inode_256::capacity; i++) {
      // cppcheck-suppress useStlAlgorithm
      if (children[i] != nullptr) {
        const auto key = static_cast<std::byte>(i);  // child_index is key byte
        const auto child_index = static_cast<std::uint8_t>(i);
        return {node_ptr{this, node_type::I256}, key, child_index,
                this->get_key_prefix().get_snapshot()};
      }
    }
    // because we always have at least 49 keys.
    UNODB_DETAIL_CANNOT_HAPPEN();  // LCOV_EXCL_LINE
  }

  /// Get iterator result for last child.
  ///
  /// Scans children array in reverse for last non-null entry (greatest key).
  ///
  /// \return Iterator result pointing to last child
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_256::iter_result
  last() noexcept {
    for (std::int64_t i = basic_inode_256::capacity - 1; i >= 0; i--) {
      if (children[static_cast<std::uint8_t>(i)] != nullptr) {
        const auto key = static_cast<std::byte>(i);  // child_index is key byte
        const auto child_index = static_cast<std::uint8_t>(i);
        return {node_ptr{this, node_type::I256}, key, child_index,
                this->get_key_prefix().get_snapshot()};
      }
    }
    // because we always have at least 49 keys.
    UNODB_DETAIL_CANNOT_HAPPEN();  // LCOV_EXCL_LINE
  }

  /// Get iterator result for next child after given index.
  ///
  /// \param child_index Current key byte index
  ///
  /// \return Iterator result for next child, or empty if at end
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_256::iter_result_opt
  next(const std::uint8_t child_index) noexcept {
    // loop over the remaining byte values in lexical order.
    for (auto i = static_cast<std::uint64_t>(child_index) + 1;
         i < basic_inode_256::capacity; i++) {
      // cppcheck-suppress useStlAlgorithm
      if (children[i] != nullptr) {
        const auto key = static_cast<std::byte>(i);
        const auto next_index = static_cast<std::uint8_t>(i);
        return {{node_ptr{this, node_type::I256}, key, next_index,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    return parent_class::end_result;
  }

  /// Get iterator result for previous child before given index.
  ///
  /// \param child_index Current key byte index
  ///
  /// \return Iterator result for previous child, or empty if at start
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_256::iter_result_opt
  prior(const std::uint8_t child_index) noexcept {
    // loop over the remaining byte values in lexical order.
    for (auto i = static_cast<std::int64_t>(child_index) - 1; i >= 0; i--) {
      const auto next_index = static_cast<std::uint8_t>(i);
      if (children[next_index] != nullptr) {
        const auto key = static_cast<std::byte>(i);
        return {{node_ptr{this, node_type::I256}, key, next_index,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    return parent_class::end_result;
  }

  /// Find last child with key byte less than or equal to given value.
  ///
  /// \param key_byte Key byte to compare against
  ///
  /// \return Iterator result for matching child, or empty if none found
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_256::iter_result_opt
  lte_key_byte(std::byte key_byte) noexcept {
    // loop over the prior byte values in lexical order.
    for (auto i = static_cast<std::int64_t>(key_byte); i >= 0; i--) {
      const auto child_index = static_cast<std::uint8_t>(i);
      if (children[child_index] != nullptr) {
        const auto key = static_cast<std::byte>(i);
        return {{node_ptr{this, node_type::I256}, key, child_index,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    return parent_class::end_result;
  }

  /// Find first child with key byte greater than or equal to given value.
  ///
  /// \param key_byte Key byte to compare against
  ///
  /// \return Iterator result for matching child, or empty if none found
  [[nodiscard, gnu::pure]] constexpr typename basic_inode_256::iter_result_opt
  gte_key_byte(std::byte key_byte) noexcept {
    // loop over the remaining byte values in lexical order.
    for (auto i = static_cast<std::uint64_t>(key_byte);
         i < basic_inode_256::capacity; i++) {
      const auto child_index = static_cast<std::uint8_t>(i);
      if (children[child_index] != nullptr) {
        const auto key = static_cast<std::byte>(i);
        return {{node_ptr{this, node_type::I256}, key, child_index,
                 this->get_key_prefix().get_snapshot()}};
      }
    }
    return parent_class::end_result;
  }

  /// Iterate over all children with callback function.
  ///
  /// \tparam Function Callable type accepting (unsigned index, node_ptr child)
  /// \param func Callback to invoke for each non-null child
  // TODO(laurynas) Lifting this out might help with iterator and
  // lambda patterns.
  template <typename Function>
  constexpr void for_each_child(Function func) const
      noexcept(noexcept(func(0, node_ptr{nullptr}))) {
#ifndef NDEBUG
    const auto children_count_ = this->children_count.load();
    std::uint8_t actual_children_count = 0;
#endif

    for (unsigned i = 0; i < 256; ++i) {
      const auto child_ptr = children[i].load();
      if (child_ptr != nullptr) {
        func(i, child_ptr);
#ifndef NDEBUG
        ++actual_children_count;
        UNODB_DETAIL_ASSERT(actual_children_count <= children_count_ ||
                            children_count_ == 0);
#endif
      }
    }
    UNODB_DETAIL_ASSERT(actual_children_count == children_count_);
  }

  /// Recursively delete all children.
  ///
  /// \param db_instance Database instance
  constexpr void delete_subtree(db_type& db_instance) noexcept {
    if constexpr (ArtPolicy::can_eliminate_leaf) {
      for_each_child([this, &db_instance](unsigned i, node_ptr child) noexcept {
        if (this->is_value_in_slot(static_cast<std::uint8_t>(i))) return;
        ArtPolicy::delete_subtree(child, db_instance);
      });
    } else {
      for_each_child(
          [&db_instance]([[maybe_unused]] unsigned i, node_ptr child) noexcept {
            ArtPolicy::delete_subtree(child, db_instance);
          });
    }
  }

  /// Dump node contents to stream for debugging.
  ///
  /// \param os Output stream
  /// \param recursive Whether to dump children recursively
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os,
                                                bool recursive) const {
    parent_class::dump(os, recursive);
    os << ", key bytes & children:\n";
    for_each_child([&os, recursive, this](unsigned i, node_ptr child) {
      os << ' ';
      dump_byte(os, static_cast<std::byte>(i));
      os << ' ';
      if (recursive) {
        if (is_value_in_slot(static_cast<std::uint8_t>(i))) {
          os << "packed value\n";
        } else {
          ArtPolicy::dump_node(os, child);
        }
      }
    });
  }

 public:
  [[nodiscard]] constexpr bool is_value_in_slot(std::uint8_t i) const noexcept {
    return bitmask_base::test(i);
  }
  constexpr void set_value_bit(std::uint8_t i) noexcept {
    bitmask_base::set(i);
  }
  constexpr void clear_value_bit(std::uint8_t i) noexcept {
    bitmask_base::clear(i);
  }

  /// Child pointers indexed directly by key byte.
  std::array<critical_section_policy<node_ptr>, basic_inode_256::capacity>
      children;

  template <class>
  friend class basic_inode_48;
  friend class basic_inode_impl<ArtPolicy>;
};  // class basic_inode_256

}  // namespace unodb::detail

#endif  // UNODB_DETAIL_ART_INTERNAL_IMPL_HPP
