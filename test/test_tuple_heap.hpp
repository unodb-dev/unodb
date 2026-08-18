// Copyright 2026 UnoDB contributors
/// \file test_tuple_heap.hpp
/// \brief Shared TestHeap class and type traits for parameterized heap tests.
///
/// Provides a trivial TupleHeap implementation and type aliases for
/// heap-backed db/mutex_db/olc_db variants.  Used by tree_verifier and
/// parameterized test suites.

#ifndef UNODB_DETAIL_TEST_TUPLE_HEAP_HPP
#define UNODB_DETAIL_TEST_TUPLE_HEAP_HPP

#include "global.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "art_common.hpp"

namespace unodb::test {

/// A trivial tuple heap for testing.  Stores pre-encoded keys in a flat
/// vector indexed by tuple_id.
class TestHeap {
 public:
  /// Register a key for a tuple_id.  Must be called before the tree uses it.
  void add_tuple(std::uint64_t id, std::span<const std::byte> key) {
    if (id >= keys_.size()) keys_.resize(id + 1);
    keys_[id].assign(key.begin(), key.end());
  }

  /// Satisfy the TupleHeap concept: extract_key(id, buf) -> key_view.
  [[nodiscard]] unodb::key_view extract_key(
      std::uint64_t id, unodb::key_encoder& /*buf*/) const noexcept {
    const auto& k = keys_[id];
    return unodb::key_view{k.data(), k.size()};
  }

 private:
  std::vector<std::vector<std::byte>> keys_;
};

static_assert(unodb::TupleHeap<TestHeap, std::uint64_t>);

/// Helper: encode a uint64 key into a byte array (big-endian for ordering).
[[nodiscard]] constexpr std::array<std::byte, 8> encode_u64(
    std::uint64_t v) noexcept {
  std::array<std::byte, 8> buf{};
  for (unsigned i = 0; i < 8; ++i) {
    buf[7U - i] = static_cast<std::byte>(v & 0xFFU);
    v >>= 8U;
  }
  return buf;
}

/// Type trait: is Db a heap-backed database type?
template <typename Db>
inline constexpr bool is_heap_db_v = false;

}  // namespace unodb::test

#endif  // UNODB_DETAIL_TEST_TUPLE_HEAP_HPP
