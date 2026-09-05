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
#include "assert.hpp"

namespace unodb::test {

/// A trivial tuple heap for testing.  Stores pre-encoded keys in a flat
/// vector indexed by tuple_id.
///
/// \par Registration contract
/// `add_tuple(id, key)` **must** be called for every (id, key) pair
/// before the corresponding `tree.insert(key, id)`.  `extract_key` will
/// assert in debug builds if the id has not been registered.  This
/// precondition exists because the ART insert path may call extract_key
/// during node splits to recover keys from existing leaves.
class TestHeap {
 public:
  /// Register a key for a tuple_id.  Must be called before the tree uses it.
  void add_tuple(std::uint64_t id, std::span<const std::byte> key) {
    if (id >= keys_.size()) keys_.resize(id + 1);
    keys_[id].assign(key.begin(), key.end());
    if (id >= registered_.size()) registered_.resize(id + 1, false);
    registered_[id] = true;
  }

  /// Satisfy the TupleHeap concept: extract_key(id, buf) -> key_view.
  [[nodiscard]] unodb::key_view extract_key(
      std::uint64_t id, unodb::key_encoder& /*buf*/) const noexcept {
    UNODB_DETAIL_ASSERT(id < registered_.size() && registered_[id]);
    const auto& k = keys_[id];
    return unodb::key_view{k.data(), k.size()};
  }

 private:
  std::vector<std::vector<std::byte>> keys_;
  std::vector<bool> registered_;
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

}  // namespace unodb::test

#endif  // UNODB_DETAIL_TEST_TUPLE_HEAP_HPP
