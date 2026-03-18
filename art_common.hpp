// Copyright 2019-2025 UnoDB contributors
#ifndef UNODB_DETAIL_ART_COMMON_HPP
#define UNODB_DETAIL_ART_COMMON_HPP

// Should be the first include
#include "global.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <span>
#include <string_view>

#include "duckdb_encode_decode.hpp"
#include "heap.hpp"
#include "portability_builtins.hpp"

/// \file
/// Common declarations for Adaptive Radix Tree (ART) index.
///
/// Provides key/value type aliases, visitor for scan API, and key encoding/
/// decoding utilities for generating binary comparable keys from primitive data
/// types.

namespace unodb {

template <typename Key, typename Value>
class db;

template <typename Key, typename Value>
class olc_db;

/// Type alias determining the maximum size in bytes of a key that may be stored
/// in the index.
using key_size_type = std::uint32_t;

/// Non-owning view of key bytes, copied into index upon insertion.
using key_view = std::span<const std::byte>;

namespace detail {

/// Tag type used as the Key parameter for keyless leaves.  When the full
/// key is encoded in the inode path, the leaf stores only the value.
struct no_key_tag {};

/// Tag type indicating that no leaf nodes exist in the tree.
/// Used when can_eliminate_leaf is true (values packed into inode slots).
struct no_leaf_tag {
  no_leaf_tag() = delete;
};

/// Whether the key can be omitted from the leaf.  True when Key is
/// key_view and Value fits in a pointer-sized slot.
template <typename Key, typename Value>
inline constexpr bool can_eliminate_key_in_leaf_v =
    std::is_same_v<Key, key_view>;

/// The Key type used for the leaf template.  Maps to no_key_tag when
/// the key can be eliminated, otherwise passes Key through unchanged.
template <typename Key, typename Value>
using leaf_key_type =
    std::conditional_t<can_eliminate_key_in_leaf_v<Key, Value>,
                       no_key_tag, Key>;

}  // namespace detail

/// Non-owning view of key bytes with scoped lifetime.  Returned by
/// iterator get_key() when the key is reconstructed from the inode
/// path (full_key_in_inode_path trees).  The view is valid only until
/// the next iterator movement (next/prior/seek/invalidate).
///
/// Does NOT allow implicit construction from key_view, preventing accidental
/// storage of a key_view where a transient_key_view is expected.  Does NOT
/// implicitly convert to key_view — use .view() for explicit access.
class transient_key_view {
  key_view kv_;

 public:
  explicit constexpr transient_key_view(key_view kv) noexcept : kv_{kv} {}

  [[nodiscard]] constexpr key_view view() const noexcept { return kv_; }

  [[nodiscard]] constexpr auto data() const noexcept { return kv_.data(); }
  [[nodiscard]] constexpr auto size() const noexcept { return kv_.size(); }
  [[nodiscard]] constexpr auto size_bytes() const noexcept {
    return kv_.size_bytes();
  }
  [[nodiscard]] constexpr auto operator[](std::size_t i) const noexcept {
    return kv_[i];
  }
  [[nodiscard]] constexpr auto begin() const noexcept { return kv_.begin(); }
  [[nodiscard]] constexpr auto end() const noexcept { return kv_.end(); }
  [[nodiscard]] constexpr bool empty() const noexcept { return kv_.empty(); }
};

/// Type alias determining the maximum size of a value that may be stored in the
/// index.
using value_size_type = std::uint32_t;

/// Non-owning view of value bytes, copied into index upon insertion.
using value_view = std::span<const std::byte>;

/// Wrapper providing access to key and value during index scan.
///
/// Passed to the caller's lambda by the scan API for each index entry. Provides
/// read-only access to the current key and value. References obtained from this
/// visitor are valid only within the scope of a single lambda invocation.
///
/// \tparam Iterator Internal iterator type (implementation detail).
///
/// \sa unodb::db::scan()
/// \sa unodb::olc_db::scan()
template <typename Iterator>
class visitor {
 protected:
  /// Reference to underlying iterator.
  Iterator& it;

  /// Construct visitor wrapping given iterator.
  explicit visitor(Iterator& it_ UNODB_DETAIL_LIFETIMEBOUND) noexcept
      : it(it_) {}

 public:
  /// Key type from underlying iterator.
  using key_type = typename Iterator::key_type;

  /// Value type from underlying iterator.
  using value_type = typename Iterator::value_type;

  /// Visit the encoded key.
  ///
  /// \return Encoded key view
  ///
  /// \note The lambda MUST NOT export a reference to the visited key. If you
  /// want to access the visited key outside of the scope of a single lambda
  /// invocation, then you MUST make a copy of the data.
  ///
  /// \note The application MAY use the unodb::key_decoder to decode any key
  /// corresponding to a sequence of one or more primitive data types. However,
  /// key decoding is not well defined for Unicode sort keys and all floating
  /// point \c NaN values are mapped to a canonical \c NaN by the
  /// unodb::key_encoder. The recommended pattern when the key contains Unicode
  /// data is to convert it to a sort key using some collation order. The
  /// Unicode data may then be recovered by associating the key with a record
  /// identifier, looking up the record and reading off the Unicode value there.
  /// This is a common secondary index scenario.
  [[nodiscard]] auto get_key() const noexcept(noexcept(it.get_key())) {
    // TODO(laurynas) The auto return appears required for get_key().
    // E.g., example_art.cpp fails with return type == key_type.
    return it.get_key();
  }

  /// Visit the value.
  ///
  /// \return Value view (type depends on database implementation)
  ///
  /// \note The lambda MUST NOT export a reference to the visited value. If you
  /// want to access the value outside of the scope of a single lambda
  /// invocation, then you must make a copy of the data.
  [[nodiscard]] auto get_value() const noexcept(noexcept(it.get_val())) {
    // Note: auto return required for olc qsbr wrapper.
    return it.get_val();
  }

 private:
  friend class olc_db<key_type, value_type>;
  friend class db<key_type, value_type>;
};  // class visitor

namespace detail {

/// Initial capacity for the unodb::key_encoder and other similar internal
/// buffers. It should be high enough that such objects DO NOT allocate for
/// commonly used key lengths. These objects use an internal buffer of this
/// capacity and then switch over to an explicitly allocated buffer if the
/// capacity would be exceeded.
///
/// \note: If you are only using fixed width keys, then this can be sizeof(T).
/// In typical scenarios these objects are on the stack and there is little if
/// any penalty to having a larger initial capacity for these buffers.
static constexpr size_t INITIAL_BUFFER_CAPACITY = 256;

/// Dump \a byte to \a os stream as a hexadecimal number.
[[gnu::cold]] void dump_byte(std::ostream& os, std::byte byte);

/// Dump \a v to \a os as a sequence of bytes.
[[gnu::cold]] void dump_val(std::ostream& os, unodb::value_view v);

/// Dump variable-length \a key to \a os as a sequence of bytes.
[[gnu::cold]] inline void dump_key(std::ostream& os, key_view key) {
  const auto sz = key.size_bytes();
  os << "key(" << sz << "): 0x";
  for (std::size_t i = 0; i < sz; ++i) unodb::detail::dump_byte(os, key[i]);
}

/// Dump \a key to \a os as a sequence of bytes.
/// \tparam T key type
template <typename T>
[[gnu::cold]] void dump_key(std::ostream& os, T key) {
  if constexpr (std::is_same_v<T, key_view>) {
    // TODO(laurynas): call dump_key?
    const auto sz = key.size_bytes();
    os << "key(" << sz << "): 0x";
    for (std::size_t i = 0; i < sz; ++i) unodb::detail::dump_byte(os, key[i]);
  } else {
    os << "key: 0x" << std::hex << std::setfill('0') << std::setw(sizeof(key))
       << key << std::dec;
  }
}

/// Utility method for power of two expansion of buffers.
///
/// \param buf The buffer to resize.
/// \param cap The current buffer capacity.
/// \param off The current number of used bytes.
/// \param min_capacity The desired new minimum capacity.
inline void ensure_capacity(std::byte*& buf, size_t& cap, size_t off,
                            size_t min_capacity) {
  // Find the smallest power of two >= min_capacity.
  const auto asize = std::bit_ceil(min_capacity);
  auto* tmp = detail::allocate_aligned(asize);  // new allocation.
  std::memcpy(tmp, buf, off);                   // copy over the data.
  if (cap > INITIAL_BUFFER_CAPACITY) {          // free old buffer iff allocated
    detail::free_aligned(buf);
  }
  buf = static_cast<std::byte*>(tmp);
  cap = asize;
}

}  // namespace detail

//
// Key encoder and key decoder
//

/// A utility class to generate binary comparable keys from a sequence of key
/// components. This class supports the various kinds of primitive data types
/// and provides support for the caller to pass through Unicode sort keys. The
/// encoded keys can be decoded with unodb::key_decoder.
///
/// \note This class is NOT final so people can extend or override the
/// unodb::key_encoder (and unodb::key_decoder) for language specific handling
/// of order within floating point values, handling of database \c NULLs, etc.
class key_encoder {
 public:
  /// This indirectly determines the unodb::key_encoder::maxlen and is used as
  /// the byte width for the run-length encoding of the padding.
  ///
  /// \note The choice of `std::uint16` here has implications for both the
  /// maximum allowed key length and the overhead for each encoded text field
  /// (since we must use the same stride to encode the pad run length). If this
  /// is changed to `std::uint32`, then you can encode longer text fields, but
  /// the padding overhead will be \c 5 bytes (vs \c 3 bytes today).
  using size_type = std::uint16_t;

  /// The pad byte used when encoding variable length text into a key to
  /// logically extend the text field to unodb::key_encoder::maxlen bytes. The
  /// pad byte (which is added to the buffer as an unsigned value) is followed
  /// by a run length count such that the key is logically padded out to the
  /// maximum length of a text field, which is unodb::key_encoder::maxlen. The
  /// run length count is expressed in the unodb::key_encoder::size_type.
  static constexpr auto pad{static_cast<std::byte>(0x00)};

  /// The maximum length of a text component of the key. Keys are truncated to
  /// at most this many bytes and then logically extended using the \c pad byte
  /// and a trailing run length until the field is logically \c maxlen bytes
  /// wide. This field is computed such that the total byte width of the encoded
  /// text can be indexed by `sizeof(size_type)`.
  static constexpr auto maxlen{static_cast<size_type>(
      std::numeric_limits<size_type>::max() - sizeof(pad) - sizeof(size_type))};
  static_assert(sizeof(maxlen) == sizeof(size_type));

 protected:
  /// \name Sign bit constants for signed integer encoding
  /// \{

  /// MSB for 8-bit integers.
  static constexpr std::uint8_t msb8 = 1U << 7;

  /// MSB for 16-bit integers.
  static constexpr std::uint16_t msb16 = 1U << 15;

  /// MSB for 32-bit integers.
  static constexpr std::uint32_t msb32 = 1U << 31;

  /// MSB for 64-bit integers.
  static constexpr std::uint64_t msb64 = 1ULL << 63;

  /// \}

  /// Convert integer \a v to big-endian (binary comparable) form.
  /// \tparam T integer type
  template <typename T>
  [[nodiscard]] static constexpr T make_binary_comparable_integral(
      T v) noexcept {
    static_assert(std::endian::native == std::endian::little,
                  "Big-endian support needs implementing");
    return unodb::detail::bswap(v);
  }

 public:
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26495)

  /// Construct empty key encoder with initial internal buffer.
  key_encoder() noexcept = default;

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Destroy encoder, freeing allocated buffer if any.
  ~key_encoder() {
    if (cap > sizeof(ibuf)) {  // free old buffer iff allocated
      detail::free_aligned(buf);
    }
  }

  /// Return number of bytes of data in internal buffer.
  [[nodiscard]] size_t size_bytes() const noexcept { return off; }

  /// Return current capacity of buffer.
  [[nodiscard]] size_t capacity() const noexcept { return cap; }

  /// Ensure that buffer can hold at least \a req additional bytes.
  void ensure_available(size_t req) {
    UNODB_DETAIL_ASSERT(req <= std::numeric_limits<std::size_t>::max() - off);
    if (UNODB_DETAIL_UNLIKELY(off + req > cap)) {
      ensure_capacity(off + req);  // resize
    }
  }

  /// Reset the encoder to encode another key.
  /// \return Self
  key_encoder& reset() noexcept {
    off = 0;
    return *this;
  }

  /// Return read-only view of internal buffer showing only those bytes that
  /// were encoded since the last unodb::key_encoder::reset call.
  [[nodiscard]] key_view get_key_view() const noexcept {
    return key_view(buf, off);
  }

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)

  /// Append a sequence of bytes \a data to the key.
  ///
  /// \return Self
  ///
  /// \note The caller is responsible for not violating the ART contract (no key
  /// may be a prefix of another key).
  key_encoder& append_bytes(std::span<const std::byte> data) {
    const auto sz = data.size_bytes();
    ensure_available(sz);
    std::memcpy(buf + off, data.data(), sz);
    off += sz;
    return *this;
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// \name Signed integer encoding
  /// Encode signed integers to binary comparable form by flipping the sign bit.
  /// \{

  /// Encode signed 8-bit integer \a v to binary comparable form.
  /// \return Self
  key_encoder& encode(std::int8_t v) {
    // TODO(laurynas): look into inlining the 1 constants.
    constexpr auto i_one = static_cast<int8_t>(1);
    constexpr auto u_one = static_cast<uint8_t>(1);
    const auto u = static_cast<uint8_t>(
        (v >= 0) ? msb8 + static_cast<uint8_t>(v)
                 : msb8 - static_cast<uint8_t>(-(v + i_one)) - u_one);
    return encode(u);
  }

  /// Encode signed 16-bit integer \a v to binary comparable form.
  /// \return Self
  key_encoder& encode(std::int16_t v) {
    constexpr auto i_one = static_cast<int16_t>(1);
    constexpr auto u_one = static_cast<uint16_t>(1);
    const auto u = static_cast<uint16_t>(
        (v >= 0) ? msb16 + static_cast<uint16_t>(v)
                 : msb16 - static_cast<uint16_t>(-(v + i_one)) - u_one);
    return encode(u);
  }

  /// Encode signed 32-bit integer \a v to binary comparable form.
  /// \return Self
  key_encoder& encode(std::int32_t v) {
    constexpr int32_t i_one = 1;
    constexpr uint32_t u_one = static_cast<uint32_t>(1);
    const uint32_t u =
        (v >= 0) ? msb32 + static_cast<uint32_t>(v)
                 : msb32 - static_cast<uint32_t>(-(v + i_one)) - u_one;
    return encode(u);
  }

  /// Encode signed 64-bit integer \a v to binary comparable form.
  /// \return Self
  key_encoder& encode(std::int64_t v) {
    constexpr int64_t i_one = static_cast<int64_t>(1);
    constexpr uint64_t u_one = static_cast<uint64_t>(1);
    const uint64_t u =
        (v >= 0) ? msb64 + static_cast<uint64_t>(v)
                 : msb64 - static_cast<uint64_t>(-(v + i_one)) - u_one;
    return encode(u);
  }

  /// \}

  /// \name Unsigned integer encoding
  /// Encode unsigned integers to binary comparable (big-endian) form.
  /// \{

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26490)

  /// Encode unsigned 8-bit integer \a v to binary comparable form.
  /// \return Self
  key_encoder& encode(std::uint8_t v) {
    ensure_available(sizeof(v));
    buf[off++] = reinterpret_cast<const std::byte&>(v);
    return *this;
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Encode unsigned 16-bit integer \a v to binary comparable form.
  /// \return Self
  key_encoder& encode(std::uint16_t v) {
    ensure_available(sizeof(v));
    const auto u = make_binary_comparable_integral(v);
    std::memcpy(buf + off, &u, sizeof(v));
    off += sizeof(v);
    return *this;
  }

  /// Encode unsigned 32-bit integer \a v to binary comparable form.
  /// \return Self
  key_encoder& encode(std::uint32_t v) {
    ensure_available(sizeof(v));
    const auto u = make_binary_comparable_integral(v);
    std::memcpy(buf + off, &u, sizeof(v));
    off += sizeof(v);
    return *this;
  }

  /// Encode unsigned 64-bit integer \a v to binary comparable form.
  /// \return Self
  key_encoder& encode(std::uint64_t v) {
    ensure_available(sizeof(v));
    const auto u = make_binary_comparable_integral(v);
    std::memcpy(buf + off, &u, sizeof(v));
    off += sizeof(v);
    return *this;
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// \}

  /// \name Floating point encoding
  /// \{

  /// Encode single-precision floating-point value \a v.
  ///
  /// \return Self
  ///
  /// \note Encoding maps all \c NaN values to a single canonical \c NaN. This
  /// means that decoding is not perfect and various kinds of \c NaN all decode
  /// as a single canonical \c NaN.
  ///
  /// \sa unodb::detail::encode_floating_point
  key_encoder& encode(float v) {
    return encode(unodb::detail::encode_floating_point<std::uint32_t>(v));
  }

  /// Encode double-precision floating-point value \a v.
  ///
  /// \return Self
  ///
  /// \note Encoding maps all \c NaN values to a single canonical \c NaN. This
  /// means that decoding is not perfect and various kinds of \c NaN all decode
  /// as a single canonical \c NaN.
  ///
  /// \sa unodb::detail::encode_floating_point
  key_encoder& encode(double v) {
    return encode(unodb::detail::encode_floating_point<std::uint64_t>(v));
  }

  /// \}

  /// \name Text encoding
  /// \{

  /// Encode ASCII text or Unicode sort key.
  ///
  /// Keys are logically padded out to unodb::key_encoder::maxlen bytes and will
  /// be truncated if they would exceed unodb::key_encoder::maxlen bytes.
  ///
  /// A unodb::key_encoder::pad byte and a run count are added to make all text
  /// fields logically unodb::key_encoder::maxlen bytes long. The truncation and
  /// padding (a) ensures that no key is a prefix of another key; and (b) keeps
  /// multi-field keys with embedded variable length text fields aligned such
  /// that the field following a variable length text field does not bleed into
  /// the lexicographic ordering of the variable length text field.
  ///
  /// When handling Unicode, the caller is responsible for using a quality
  /// library (e.g., ICU) to (a) normalize their Unicode data; and (b) generate
  /// a Unicode sort key from their Unicode data. The sort key will impose
  /// specific collation ordering semantics as configured by the application
  /// (locale, collation strength, decomposition mode).
  ///
  /// \param text A view onto some sequence of bytes. At most
  /// unodb::key_encoder::maxlen bytes will be read from it.
  ///
  /// \return Self
  ///
  /// \note The ART index disallows keys which are prefixes of other keys. The
  /// logical padding addresses this and other issues while preserving
  /// lexicographic ordering.
  key_encoder& encode_text(std::span<const std::byte> text) {
    // truncate view to at most maxlen bytes.
    text = (text.size_bytes() > maxlen) ? text.subspan(0, maxlen) : text;
    // normalize padding by stripping off any trailing [pad] bytes.
    auto sz = text.size_bytes();
    for (; sz > 0; sz--) {
      if (text[sz - 1] != pad) break;
    }
    text = text.subspan(0, sz);  // adjust span in case truncated.
    // Ensure enough room for the text, the pad byte, and the
    // run-length encoding of the pad byte.
    ensure_available(sz + 1 + sizeof(size_type));
    const auto padlen{static_cast<size_type>(maxlen - sz)};
    append_bytes(text);                      // append bytes to the buffer.
    encode(static_cast<std::uint8_t>(pad));  // encode as unsigned byte
    encode(padlen);  // logical run-length of the pad byte.
    return *this;
  }

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26490)

  /// Encode text from `std::string_view` \a sv.
  /// \return Self
  /// \overload
  key_encoder& encode_text(std::string_view sv) {
    return encode_text(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(sv.data()), sv.size()));
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// \}

  /// Non-copyable.
  key_encoder(const key_encoder&) = delete;
  /// Non-movable.
  key_encoder(key_encoder&&) = delete;
  /// Non-copy-assignable.
  key_encoder& operator=(const key_encoder&) = delete;
  /// Non-move-assignable.
  key_encoder& operator=(key_encoder&&) = delete;

 private:
  /// Grow buffer to at least \a min_capacity.
  void ensure_capacity(size_t min_capacity) {
    unodb::detail::ensure_capacity(buf, cap, off, min_capacity);
  }

  /// Initial internal buffer avoiding heap allocation for small keys.
  std::byte ibuf[detail::INITIAL_BUFFER_CAPACITY];

  /// Buffer accumulating the encoded key. Points to `ibuf` initially; replaced
  /// with heap allocation if capacity exceeded.
  std::byte* buf{&ibuf[0]};

  /// Current buffer capacity in bytes.
  size_t cap{sizeof(ibuf)};

  /// Current offset (number of encoded bytes) in buffer.
  size_t off{0};
};  // class key_encoder

/// A utility class for decoding binary comparable keys produced by
/// unodb::key_encoder, except for Unicode sort keys which are not reversible.
/// To use this class, you need to know how a given key was encoded as a
/// sequence of key components.
///
/// \note This class is NOT final so people can extend or override the
/// unodb::key_decoder (and unodb::key_encoder) for language-specific handling
/// of order within floating point values, handling of \c NULL values in
/// database query languages, etc.
class key_decoder {
 private:
  const std::byte* buf;  ///< Data buffer to decode.
  const size_t cap;      ///< Buffer size in bytes.
  size_t off{0};         ///< Current decode offset.

  // TODO(laurynas): share with the encoder

  /// \name Sign bit constants for signed integer decoding
  /// \{

  /// MSB for 8-bit integers.
  static constexpr std::uint8_t msb8 = 1U << 7U;

  /// MSB for 16-bit integers.
  static constexpr std::uint16_t msb16 = 1U << 15U;

  /// MSB for 32-bit integers.
  static constexpr std::uint32_t msb32 = 1U << 31U;

  /// MSB for 64-bit integers.
  static constexpr std::uint64_t msb64 = 1ULL << 63U;

  /// \}

  /// Convert big-endian integer \a u back to native form.
  /// \tparam T type of the integer
  template <typename T>
  [[nodiscard]] static constexpr T decode_binary_comparable_integral(
      T u) noexcept {
    static_assert(std::endian::native == std::endian::little,
                  "Big-endian support needs implementing");
    return unodb::detail::bswap(u);
  }

 public:
  /// Construct decoder for given key view \a kv.
  ///
  /// The key view must remain valid for the lifetime of this decoder. This is
  /// ensured trivially when used within a scan lambda.
  explicit key_decoder(key_view kv) noexcept
      : buf(kv.data()), cap(kv.size_bytes()) {}

  /// \name Signed integer decoding
  /// Decode signed integers from binary comparable form.
  /// \{

  /// Decode signed 8-bit integer \a v from binary comparable form.
  /// \return Self
  key_decoder& decode(std::int8_t& v) noexcept {
    std::uint8_t u;
    decode(u);
    v = static_cast<std::int8_t>(
        (u >= msb8) ? u - msb8 : ~static_cast<std::uint8_t>(msb8 - 1U - u));
    return *this;
  }

  /// Decode signed 16-bit integer \a v from binary comparable form.
  /// \return Self
  key_decoder& decode(std::int16_t& v) noexcept {
    std::uint16_t u;
    decode(u);
    v = static_cast<std::int16_t>(
        (u >= msb16) ? u - msb16 : ~static_cast<std::uint16_t>(msb16 - 1U - u));
    return *this;
  }

  /// Decode signed 32-bit integer \a v from binary comparable form.
  /// \return Self
  key_decoder& decode(std::int32_t& v) noexcept {
    constexpr auto one = static_cast<std::uint32_t>(1);
    std::uint32_t u;
    decode(u);
    v = (u >= msb32) ? static_cast<int32_t>(u - msb32)
                     : -static_cast<int32_t>(msb32 - one - u) - 1;
    return *this;
  }

  /// Decode signed 64-bit integer \a v from binary comparable form.
  /// \return Self
  key_decoder& decode(std::int64_t& v) noexcept {
    constexpr auto one = static_cast<std::uint64_t>(1);
    std::uint64_t u;
    decode(u);
    v = (u >= msb64) ? static_cast<int64_t>(u - msb64)
                     : -static_cast<int64_t>(msb64 - one - u) - 1LL;
    return *this;
  }

  /// \}

  /// \name Unsigned integer decoding
  /// Decode unsigned integers from big-endian form.
  /// \{

  UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26490)

  /// Decode unsigned 8-bit integer \a v.
  /// \return Self
  key_decoder& decode(std::uint8_t& v) noexcept {
    v = reinterpret_cast<const std::uint8_t&>(buf[off++]);
    return *this;
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Decode unsigned 16-bit integer \a v from big-endian form.
  /// \return Self
  key_decoder& decode(std::uint16_t& v) noexcept {
    std::uint16_t u;
    std::memcpy(&u, buf + off, sizeof(u));
    v = decode_binary_comparable_integral(u);
    off += sizeof(u);
    return *this;
  }

  /// Decode unsigned 32-bit integer \a v from big-endian form.
  /// \return Self
  key_decoder& decode(std::uint32_t& v) noexcept {
    std::uint32_t u;
    std::memcpy(&u, buf + off, sizeof(u));
    v = decode_binary_comparable_integral(u);
    off += sizeof(u);
    return *this;
  }

  /// Decode unsigned 64-bit integer \a v from big-endian form.
  /// \return Self
  key_decoder& decode(std::uint64_t& v) noexcept {
    std::uint64_t u;
    std::memcpy(&u, buf + off, sizeof(u));
    v = decode_binary_comparable_integral(u);
    off += sizeof(u);
    return *this;
  }

  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// \}

  /// \name Floating point decoding
  /// \{

  /// Decode single-precision floating-point value \a v.
  ///
  /// \return Self
  ///
  /// \note Encoding maps all \c NaN values to a single canonical \c NaN. This
  /// means that decoding is not perfect and various kinds of \c NaN all decode
  /// as a single canonical \c NaN.
  ///
  /// \sa unodb::detail::decode_floating_point
  key_decoder& decode(float& v) noexcept {
    std::uint32_t u;
    decode(u);
    v = unodb::detail::decode_floating_point<float>(u);
    return *this;
  }

  /// Decode double-precision floating-point value \a v.
  ///
  /// \return Self
  ///
  /// \note Encoding maps all \c NaN values to a single canonical \c NaN. This
  /// means that decoding is not perfect and various kinds of \c NaN all decode
  /// as a single canonical \c NaN.
  ///
  /// \sa unodb::detail::decode_floating_point
  key_decoder& decode(double& v) noexcept {
    std::uint64_t u;
    decode(u);
    v = unodb::detail::decode_floating_point<double>(u);
    return *this;
  }

  /// \}
};  // class key_decoder

}  // namespace unodb

#endif  // UNODB_DETAIL_ART_COMMON_HPP
