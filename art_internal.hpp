// Copyright 2019-2025 UnoDB contributors
#ifndef UNODB_DETAIL_ART_INTERNAL_HPP
#define UNODB_DETAIL_ART_INTERNAL_HPP

/// \file
/// Internal implementation details for Adaptive Radix Tree (ART) index.
///
/// Provides key types, tree traversal utilities, and deleter types for the ART
/// implementation. This header is not part of the public API.

// Should be the first include
#include "global.hpp"

// IWYU pragma: no_include <__cstddef/byte.h>
// IWYU pragma: no_include <_string.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <type_traits>

#include "art_common.hpp"
#include "assert.hpp"
#include "heap.hpp"
#include "node_type.hpp"
#include "portability_builtins.hpp"

namespace unodb::detail {

template <class, class>
class [[nodiscard]] basic_leaf;

template <class>
class [[nodiscard]] basic_db_leaf_deleter;

/// Compare byte spans lexicographically.
///
/// \param a First byte sequence
/// \param alen Length of first sequence
/// \param b Second byte sequence
/// \param blen Length of second sequence
/// \return -1, 0, or 1 if first key is LT, EQ, or GT second key
[[nodiscard, gnu::pure]] inline int compare(const void* a, const size_t alen,
                                            const void* b,
                                            const size_t blen) noexcept {
  // TODO(thompsonbry) consider changing this over to std::span
  // arguments and do not let the (ptr,len) pattern propagate
  // outwards.
  const auto shared_length = std::min(alen, blen);
  auto ret = std::memcmp(a, b, shared_length);
  ret = (ret != 0) ? ret : ((alen == blen) ? 0 : (alen < blen ? -1 : 1));
  return ret;
}

/// Compare key views lexicographically.
///
/// \param a First key view
/// \param b Second key view
/// \return -1, 0, or 1 if first key is LT, EQ, or GT second key
[[nodiscard, gnu::pure]] inline int compare(const unodb::key_view a,
                                            const unodb::key_view b) noexcept {
  return compare(a.data(), a.size_bytes(), b.data(), b.size_bytes());
}

/// Return first 64 bits of encoded \a key view.
///
/// Used by prefix compression logic to identify bytes in common between an
/// art_key and an inode having some key_prefix.
///
/// \return First 64 bits of key, zero-padded if key is shorter
[[nodiscard, gnu::pure]] inline std::uint64_t get_u64(key_view key) noexcept {
  std::uint64_t u{};  // will hold the first 64-bits.
  std::memcpy(&u, key.data(), std::min(key.size_bytes(), sizeof(u)));
  return u;
}

/// Internal ART key in lexicographically ordered binary-comparable format.
///
/// Application keys may be simple fixed width types (such as `std::uint64_t`)
/// or variable length keys. For the former, there are convenience methods on
/// unodb::db, unodb::olc_db, etc. to convert external keys into the binary
/// comparable format. For the latter, the application is responsible for
/// converting the data (e.g., certain columns in some ordering for a row of
/// some relation) into the internal binary comparable key format.
///
/// A convenience class (unodb::key_encoder) is offered to encode data. The
/// encoding is always well defined and decoding (unodb::key_decoder) exists
/// for all simple fixed width data types.
///
/// Unicode encoding is complex and out of scope - use a quality library such
/// as ICU to produce appropriate Unicode sort keys for your application.
/// Unicode decoding is NOT well defined. Applications involving database
/// records and Unicode data will typically store the record identifier in a
/// secondary index (ART) as the value associated with the key. Using the
/// record identifier, the original tuple can be discovered and the original
/// Unicode data recovered from that tuple.
///
/// \tparam KeyType Fixed-width integral type or `key_view` for variable length
template <typename KeyType>
struct [[nodiscard]] basic_art_key final {
 private:
  /// Convert simple external key into internal key for lexicographic
  /// comparison.
  ///
  /// \param k External key to convert
  /// \return Binary comparable key
  [[nodiscard, gnu::const]] static UNODB_DETAIL_CONSTEXPR_NOT_MSVC KeyType
  make_binary_comparable(KeyType k) noexcept {
    static_assert(std::endian::native == std::endian::little,
                  "Big-endian support needs implementing");
    return bswap(k);
  }

 public:
  /// Construct from fixed width primitive type \a key_.
  ///
  /// \note Use a unodb::key_encoder for complex keys, including multiple key
  /// components or Unicode data.
  UNODB_DETAIL_CONSTEXPR_NOT_MSVC explicit basic_art_key(KeyType key_) noexcept
    requires std::is_integral_v<KeyType>
      : key{make_binary_comparable(key_)} {}

  /// Construct from \a key_ view which must already be lexicographically
  /// ordered.
  UNODB_DETAIL_CONSTEXPR_NOT_MSVC explicit basic_art_key(key_view key_) noexcept
      : key{key_} {}

  /// Compare with another \a key2.
  ///
  /// \return -1, 0, or 1 if this key is LT, EQ, or GT the other key
  [[nodiscard, gnu::pure]] constexpr int cmp(
      basic_art_key<KeyType> key2) const noexcept {
    if constexpr (std::is_same_v<KeyType, key_view>) {
      return compare(key, key2.key);
    } else {
      return std::memcmp(&key, &key2.key, sizeof(KeyType));
    }
  }

  /// Compare with a \a key2 view.
  ///
  /// \return -1, 0, or 1 if this key is LT, EQ, or GT the other key
  [[nodiscard, gnu::pure]] constexpr int cmp(key_view key2) const noexcept {
    if constexpr (std::is_same_v<KeyType, unodb::key_view>) {
      // variable length keys
      return compare(key, key2);
    } else {
      // fixed width keys
      return compare(&key, sizeof(KeyType), key2.data(), key2.size_bytes());
    }
  }

  /// Return byte at specified \a index position in binary comparable key.
  ///
  /// \return Byte at position
  [[nodiscard, gnu::pure]] constexpr auto operator[](
      std::size_t index) const noexcept {
    if constexpr (std::is_same_v<KeyType, key_view>) {
      return key[index];
    } else {
      // The key_bytes[] provides access the different byte positions
      // in the primitive type [Key key].
      UNODB_DETAIL_ASSERT(index < sizeof(KeyType));
      return key_bytes[index];
    }
  }

  /// Return backing key_view.
  ///
  /// \return Key view of this key's bytes
  ///
  /// \note For integral keys, this is a non-owned view of the data in
  /// the basic_art_key and will be invalid if that object goes out of
  /// scope.
  ///
  /// \note For key_view keys, this is the key_view backing this art_key and
  /// its validity depends on the scope of the backing byte array.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26490)
  [[nodiscard, gnu::pure]] constexpr key_view get_key_view() const noexcept {
    if constexpr (std::is_same_v<KeyType, key_view>) {
      return key;
    } else {
      return key_view(reinterpret_cast<const std::byte*>(&key), sizeof(key));
    }
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Return first 64 bits (max) of encoded key.
  ///
  /// Used by prefix compression logic to identify bytes in common between an
  /// art_key and an inode having some key_prefix.
  ///
  /// \return First 64 bits of key
  [[nodiscard, gnu::pure]] constexpr std::uint64_t get_u64() const noexcept {
    if constexpr (std::is_same_v<KeyType, key_view>) {
      return unodb::detail::get_u64(key);
    } else {
      return static_cast<std::uint64_t>(key);
    }
  }

  /// Shift internal key by \a num_bytes number of bytes to the right.
  ///
  /// Leading bytes are discarded, causing key to be shorter by that many bytes.
  /// It is a lexicographic key. The first byte is the most significant. The
  /// last byte is the least significant.
  ///
  /// When backed by a u64, we get trailing bytes which are zeros. Thus, for a
  /// fixed width type, this causes the key to be logically zero filled as it
  /// becomes shorter. E.g.
  ///
  /// `0x0011223344556677 shift_right(2) => 0x2233445566770000`
  constexpr void shift_right(const std::size_t num_bytes) noexcept {
    if constexpr (std::is_same_v<KeyType, key_view>) {
      UNODB_DETAIL_ASSERT(num_bytes <= key.size_bytes());
      key = key.subspan(num_bytes);
    } else {
      UNODB_DETAIL_ASSERT(num_bytes <= sizeof(KeyType));
      key >>= (num_bytes * 8);
    }
  }

  /// Return number of bytes required to represent key.
  ///
  /// \return Key size in bytes
  constexpr size_t size() const noexcept {
    if constexpr (std::is_same_v<KeyType, unodb::key_view>) {
      return key.size_bytes();
    } else {
      return sizeof(KeyType);
    }
  }

  union {
    /// Lexicographic byte-wise comparable binary key.
    ///
    /// \note When `KeyType == key_view`, this is all you need.
    KeyType key;

    /// Byte array overlay for indexing into fixed-width keys.
    ///
    /// Provides a mechanism to index into the bytes in the key for operator[].
    /// Ignored if `KeyType == key_view` as key_view provides a byte-wise index
    /// operator already.
    std::array<std::byte, sizeof(KeyType)> key_bytes;
  };

  /// Compile-time assertions for type invariants.
  static void static_asserts() {
    static_assert(std::is_trivially_copyable_v<basic_art_key<KeyType>>);
    static_assert(sizeof(basic_art_key<KeyType>) == sizeof(KeyType));
  }

  /// Dump key in lexicographic byte-wise order to stream \a os.
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os) const {
    dump_key(os, key);
  }

  /// Dump key to `std::cerr` for debugging.
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump() const;

  /// Dump key \a k to stream \a os.
  ///
  /// \return Reference to stream
  friend std::ostream& operator<<(std::ostream& os, const basic_art_key& k) {
    k.dump(os);
    return os;
  }
};  // class basic_art_key

/// Number of key bytes consumed along some path in the tree.
///
/// In general, the unodb::detail::key_prefix consumes up to some fixed number
/// of bytes and one byte is consumed based on the key_byte for each node as we
/// descend along that byte value to some child index.
///
/// \tparam ArtKey ART key type
template <typename ArtKey>
class [[nodiscard]] tree_depth final {
 public:
  /// Underlying storage type for depth value.
  using value_type = std::uint32_t;

  /// Construct with initial depth \a value_ (default 0).
  // cppcheck-suppress passedByValue
  explicit constexpr tree_depth(value_type value_ = 0) noexcept
      : value{value_} {}

  /// Convert implicitly to value_type.
  ///
  /// \return Current depth value
  // NOLINTNEXTLINE(google-explicit-constructor)
  [[nodiscard, gnu::pure]] constexpr operator value_type() const noexcept {
    return value;
  }

  /// Pre-increment operator.
  ///
  /// \return Reference to this after incrementing
  constexpr tree_depth& operator++() noexcept {
    ++value;
    return *this;
  }

  /// Add \a delta to depth.
  // cppcheck-suppress passedByValue
  constexpr void operator+=(value_type delta) noexcept { value += delta; }

 private:
  /// Current depth value.
  value_type value;
};

/// Functor to delete leaf nodes through database.
///
/// Custom deleter for use with `std::unique_ptr` that deletes leaf nodes by
/// notifying the owning database.
///
/// \tparam Db Database type
template <class Db>
class basic_db_leaf_deleter {
 public:
  /// Leaf type managed by this deleter.
  using leaf_type = basic_leaf<typename Db::key_type, typename Db::header_type>;

  static_assert(std::is_trivially_destructible_v<leaf_type>);

  /// Construct leaf deleter for database \a db_
  constexpr explicit basic_db_leaf_deleter(
      Db& db_ UNODB_DETAIL_LIFETIMEBOUND) noexcept
      : db{db_} {}

  /// Delete a leaf node \a to_delete through the database.
  void operator()(leaf_type* to_delete) const noexcept;

  /// Get reference to owning database.
  ///
  /// \return Reference to database
  [[nodiscard, gnu::pure]] Db& get_db() const noexcept { return db; }

 private:
  /// Reference to owning database.
  Db& db;
};

/// Unique pointer to leaf with database-aware deleter.
///
/// Not taken from Db to break a template dependency cycle.
///
/// \tparam Key Key type
/// \tparam Value Value type
/// \tparam Header Node header type
/// \tparam Db Database template
template <typename Key, typename Value, class Header,
          template <typename, typename> class Db>
using basic_db_leaf_unique_ptr =
    std::unique_ptr<basic_leaf<Key, Header>,
                    basic_db_leaf_deleter<Db<Key, Value>>>;

// TODO(laurynas): extract a base class db_ref?

/// Functor to delete internal nodes through database.
///
/// Custom deleter for use with `std::unique_ptr` that deletes internal nodes
/// by notifying the owning database.
///
/// \tparam INode Internal node type
/// \tparam Db Database type
template <class INode, class Db>
class basic_db_inode_deleter {
 public:
  /// Construct internal node deleter for database \a db_.
  constexpr explicit basic_db_inode_deleter(
      Db& db_ UNODB_DETAIL_LIFETIMEBOUND) noexcept
      : db{db_} {}

  /// Delete an internal node \a inode_ptr through the database.
  void operator()(INode* inode_ptr) noexcept;

  /// Get reference to owning database.
  ///
  /// \return Reference to database
  [[nodiscard, gnu::pure]] Db& get_db() noexcept { return db; }

 private:
  /// Reference to owning database.
  Db& db;
};

/// Tagged pointer with node type stored in low bits.
///
/// You have to know statically the target type, then call
/// `node_ptr_var.ptr<target_type*>()` to get `target_type*`.
///
/// \tparam Header Node header type
template <class Header>
class [[nodiscard]] basic_node_ptr {
 public:
  /// Node header type.
  using header_type = Header;

  /// Default constructor, does not initialize fields.
  ///
  /// Used by `std::array` and we don't want to initialize to zero or any other
  /// value there at construction time.
  // cppcheck-suppress uninitMemberVar
  constexpr basic_node_ptr() noexcept = default;

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26490)

  /// Construct null pointer.
  explicit basic_node_ptr(std::nullptr_t) noexcept
      : tagged_ptr{reinterpret_cast<std::uintptr_t>(nullptr)} {}

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Construct from raw pointer and node type.
  ///
  /// \param ptr Pointer to node
  /// \param type Node type tag
  basic_node_ptr(const header_type* ptr UNODB_DETAIL_LIFETIMEBOUND,
                 unodb::node_type type) noexcept
      : tagged_ptr{tag_ptr(ptr, type)} {}

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26490)

  /// Assign null pointer.
  ///
  /// \return Reference to this
  basic_node_ptr<Header>& operator=(std::nullptr_t) noexcept {
    tagged_ptr = reinterpret_cast<std::uintptr_t>(nullptr);
    return *this;
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Get node type from tag bits.
  ///
  /// \return Node type
  [[nodiscard, gnu::pure]] constexpr auto type() const noexcept {
    return static_cast<unodb::node_type>(tagged_ptr & tag_bit_mask);
  }

  /// Get raw tagged pointer value.
  ///
  /// \return Raw value including tag bits
  [[nodiscard, gnu::pure]] constexpr auto raw_val() const noexcept {
    return tagged_ptr;
  }

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26490)

  /// Get typed pointer with tag bits masked off.
  ///
  /// \tparam T Target pointer type
  /// \return Pointer to node as type T
  template <class T>
  [[nodiscard, gnu::pure]] auto* ptr() const noexcept {
    return reinterpret_cast<T>(tagged_ptr & ptr_bit_mask);
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Compare for equality (same raw value means same type and pointer).
  ///
  /// \param other Pointer to compare against
  /// \return True if equal
  [[nodiscard, gnu::pure]] constexpr bool operator==(
      basic_node_ptr other) const noexcept {
    return tagged_ptr == other.tagged_ptr;
  }

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26490)

  /// Compare with nullptr.
  ///
  /// \return True if this is a null pointer
  [[nodiscard, gnu::pure]] bool operator==(std::nullptr_t) const noexcept {
    return tagged_ptr == reinterpret_cast<std::uintptr_t>(nullptr);
  }

 private:
  /// Tagged pointer value with node type in low bits.
  std::uintptr_t tagged_ptr;

  /// Create tagged pointer from raw pointer \a ptr_ and node type \a tag.
  ///
  /// \return Tagged pointer value
  [[nodiscard, gnu::const]] static std::uintptr_t tag_ptr(
      const Header* ptr_, unodb::node_type tag) noexcept {
    const auto uintptr = reinterpret_cast<std::uintptr_t>(ptr_);
    const auto result =
        uintptr | static_cast<std::underlying_type_t<decltype(tag)>>(tag);
    UNODB_DETAIL_ASSERT((result & ptr_bit_mask) == uintptr);
    return result;
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Calculate bits needed to represent \a count values.
  ///
  /// \return Number of bits needed
  [[nodiscard, gnu::const]] static constexpr unsigned mask_bits_needed(
      unsigned count) noexcept {
    return count < 2 ? 1 : 1 + mask_bits_needed(count >> 1U);
  }

  /// Lowest bit not used for tag.
  static constexpr auto lowest_non_tag_bit =
      1ULL << mask_bits_needed(node_type_count);
  /// Mask for tag bits.
  static constexpr auto tag_bit_mask = lowest_non_tag_bit - 1;
  /// Mask for pointer bits.
  static constexpr auto ptr_bit_mask = ~tag_bit_mask;

  /// Compile-time assertions for type invariants.
  static void static_asserts() {
    static_assert(sizeof(basic_node_ptr<Header>) == sizeof(void*));
    static_assert(alignof(header_type) - 1 > lowest_non_tag_bit);
  }
};

/// Expandable buffer for binary comparable keys.
///
/// Used to track key by iterator as things are pushed and popped on the stack.
// TODO(thompsonbry) This data structure (std::vector but with unstable
// &data[0] pointer and initial stack allocation) is repeated and could be
// refactored out into a base class.
class key_buffer {
 protected:
  /// Get capacity of backing buffer.
  ///
  /// \return Buffer capacity in bytes
  [[nodiscard]] size_t capacity() const noexcept { return cap; }

  /// Get number of bytes of valid data in buffer.
  ///
  /// \return Number of valid bytes
  [[nodiscard]] size_t size_bytes() const noexcept { return off; }

  /// Ensure buffer can hold at least req additional bytes.
  ///
  /// \param req Number of additional bytes needed
  void ensure_available(size_t req) {
    if (UNODB_DETAIL_UNLIKELY(off + req > cap)) {
      ensure_capacity(off + req);  // resize
    }
  }

 public:
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26495)
  /// Construct new key_buffer.
  ///
  /// Backed by an internal buffer of configured size, extended if required for
  /// longer keys.
  key_buffer() noexcept = default;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Destructor, frees allocated buffer if any.
  ~key_buffer() {
    if (cap > sizeof(ibuf)) {  // free old buffer iff allocated
      detail::free_aligned(buf);
    }
  }

  /// Reset buffer to empty state.
  void reset() noexcept { off = 0; }

  /// Get read-only view of valid buffer contents.
  ///
  /// \return Key view of valid bytes
  [[nodiscard]] key_view get_key_view() const noexcept {
    return key_view(buf, off);
  }

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)

  /// Append a byte \a v to buffer.
  void push(std::byte v) {
    ensure_available(sizeof(v));
    buf[off++] = v;
  }

  /// Append bytes from key view \a v to buffer.
  void push(key_view v) {
    const auto n = v.size_bytes();
    ensure_available(n);
    std::memcpy(buf + off, v.data(), n);
    off += n;
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Pop \a n bytes from end of buffer.
  void pop(size_t n) noexcept {
    UNODB_DETAIL_ASSERT(off >= n);
    off -= n;
  }

  /// Non-copyable.
  key_buffer(const key_buffer&) = delete;
  /// Non-movable.
  key_buffer(key_buffer&&) = delete;
  /// Non-copy-assignable.
  key_buffer& operator=(const key_buffer&) = delete;
  /// Non-move-assignable.
  key_buffer& operator=(key_buffer&&) = delete;

 private:
  /// Ensure buffer has at least \a min_capacity capacity.
  void ensure_capacity(size_t min_capacity) {
    unodb::detail::ensure_capacity(buf, cap, off, min_capacity);
  }

  /// Initial internal buffer avoiding heap allocation.
  std::byte ibuf[detail::INITIAL_BUFFER_CAPACITY];

  /// Buffer accumulating key bytes.
  ///
  /// Initially points to ibuf, may be reallocated for longer keys.
  std::byte* buf{&ibuf[0]};
  /// Current buffer capacity in bytes.
  size_t cap{sizeof(ibuf)};
  /// Number of valid bytes in buffer.
  size_t off{0};
};  // class key_buffer

}  // namespace unodb::detail

#endif  // UNODB_DETAIL_ART_INTERNAL_HPP
