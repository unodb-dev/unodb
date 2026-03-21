// Copyright 2019-2026 UnoDB contributors
#ifndef UNODB_DETAIL_MICRO_BENCHMARK_UTILS_HPP
#define UNODB_DETAIL_MICRO_BENCHMARK_UTILS_HPP

// Should be the first include
#include "global.hpp"

// IWYU pragma: no_include <__ostream/basic_ostream.h>
// IWYU pragma: no_include <__cstddef/byte.h>

#include <array>
#include <cstddef>  // IWYU pragma: keep
#include <cstdint>
#ifndef NDEBUG
#include <iostream>
#endif
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "art.hpp"
#include "art_common.hpp"
#ifndef NDEBUG
#include "assert.hpp"
#endif
#include "mutex_art.hpp"
#include "olc_art.hpp"
#include "qsbr.hpp"

// TODO(laurynas): std::uint64_t-specific

#define UNODB_START_BENCHMARKS()           \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26409) \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26426)

#define UNODB_BENCHMARK_MAIN()         \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS() \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS() \
  BENCHMARK_MAIN()

namespace unodb::benchmark {

// Benchmarked tree types (u64 keys)

using db = unodb::db<std::uint64_t, unodb::value_view>;
using mutex_db = unodb ::mutex_db<std::uint64_t, unodb::value_view>;
using olc_db = unodb::olc_db<std::uint64_t, unodb::value_view>;

// Benchmarked tree types (key_view keys)

using kv_db = unodb::db<unodb::key_view, unodb::value_view>;
using kv_olc_db = unodb::olc_db<unodb::key_view, unodb::value_view>;
using kv_u64_db = unodb::db<unodb::key_view, std::uint64_t>;
using kv_u64_olc_db = unodb::olc_db<unodb::key_view, std::uint64_t>;

// Values

constexpr auto value1 = std::array<std::byte, 1>{};
constexpr auto value10 = std::array<std::byte, 10>{};
constexpr auto value100 = std::array<std::byte, 100>{};
constexpr auto value1000 = std::array<std::byte, 1000>{};
constexpr auto value10000 = std::array<std::byte, 10000>{};

inline constexpr std::array<unodb::value_view, 5> values = {
    unodb::value_view{value1}, unodb::value_view{value10},
    unodb::value_view{value100}, unodb::value_view{value1000},
    unodb::value_view{value10000}};

// PRNG

[[nodiscard]] inline auto& get_prng() {
  static std::random_device rd;
  static std::mt19937 gen{rd()};
  return gen;
}

// Inserts

namespace detail {

template <class Db>
void do_insert_key_ignore_dups(Db& instance, std::uint64_t k,
                               unodb::value_view v) {
  // Args to ::benchmark::DoNoOptimize cannot be const, thus silence MSVC static
  // analyzer on that
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  auto result = instance.insert(k, v);
  ::benchmark::DoNotOptimize(result);
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
}

template <class Db>
void do_insert_key(Db& instance, std::uint64_t k, unodb::value_view v) {
  // Args to ::benchmark::DoNoOptimize cannot be const, thus silence MSVC static
  // analyzer on that
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  auto result = instance.insert(k, v);
#ifndef NDEBUG
  if (!result) {
    std::cerr << "Failed to insert ";
    ::unodb::detail::dump_key(std::cerr, k);
    std::cerr << "\nCurrent tree:";
    instance.dump(std::cerr);
    UNODB_DETAIL_ASSERT(result);
  }
#endif
  ::benchmark::DoNotOptimize(result);
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
}

}  // namespace detail

template <class Db>
void insert_key_ignore_dups(Db& instance, std::uint64_t k,
                            unodb::value_view v) {
  detail::do_insert_key_ignore_dups(instance, k, v);
}

template <>
inline void insert_key_ignore_dups(
    unodb::olc_db<std::uint64_t, unodb::value_view>& instance, std::uint64_t k,
    unodb::value_view v) {
  const quiescent_state_on_scope_exit qsbr_after_get{};
  detail::do_insert_key_ignore_dups(instance, k, v);
}

template <class Db>
void insert_key(Db& instance, std::uint64_t k, unodb::value_view v) {
  detail::do_insert_key(instance, k, v);
}

template <>
inline void insert_key(
    unodb::olc_db<std::uint64_t, unodb::value_view>& instance, std::uint64_t k,
    unodb::value_view v) {
  const quiescent_state_on_scope_exit qsbr_after_get{};
  detail::do_insert_key(instance, k, v);
}

// Deletes

namespace detail {

template <class Db>
void do_delete_key_if_exists(Db& instance, std::uint64_t k) {
  // Args to ::benchmark::DoNoOptimize cannot be const, thus silence MSVC static
  // analyzer on that
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  auto result = instance.remove(k);
  ::benchmark::DoNotOptimize(result);
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
}

template <class Db>
void do_delete_key(Db& instance, std::uint64_t k) {
  // Args to ::benchmark::DoNoOptimize cannot be const, thus silence MSVC static
  // analyzer on that
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  auto result = instance.remove(k);
#ifndef NDEBUG
  if (!result) {
    std::cerr << "Failed to delete existing ";
    ::unodb::detail::dump_key(std::cerr, k);
    std::cerr << "\nTree:";
    instance.dump(std::cerr);
    UNODB_DETAIL_ASSERT(result);
  }
#endif
  ::benchmark::DoNotOptimize(result);
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
}

}  // namespace detail

template <class Db>
void delete_key_if_exists(Db& instance, std::uint64_t k) {
  detail::do_delete_key_if_exists(instance, k);
}

template <>
inline void delete_key_if_exists(
    unodb::olc_db<std::uint64_t, unodb::value_view>& instance,
    std::uint64_t k) {
  const quiescent_state_on_scope_exit qsbr_after_get{};
  detail::do_delete_key_if_exists(instance, k);
}

template <class Db>
void delete_key(Db& instance, std::uint64_t k) {
  detail::do_delete_key(instance, k);
}

template <>
inline void delete_key(
    unodb::olc_db<std::uint64_t, unodb::value_view>& instance,
    std::uint64_t k) {
  const quiescent_state_on_scope_exit qsbr_after_get{};
  detail::do_delete_key(instance, k);
}

// Gets

namespace detail {

template <class Db>
void do_get_key(const Db& instance, std::uint64_t k) {
  // Args to ::benchmark::DoNoOptimize cannot be const, thus silence MSVC static
  // analyzer on that
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  auto result = instance.get(k);
  ::benchmark::DoNotOptimize(result);
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
}

template <class Db>
void do_get_existing_key(const Db& instance, std::uint64_t k) {
  // Args to ::benchmark::DoNoOptimize cannot be const, thus silence MSVC static
  // analyzer on that
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  auto result = instance.get(k);

#ifndef NDEBUG
  if (!Db::key_found(result)) {
    std::cerr << "Failed to get existing ";
    ::unodb::detail::dump_key(std::cerr, k);
    std::cerr << "\nTree:";
    instance.dump(std::cerr);
    UNODB_DETAIL_CRASH();
  }
#endif
  ::benchmark::DoNotOptimize(result);
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
}

}  // namespace detail

template <class Db>
void get_key(const Db& instance, std::uint64_t k) {
  detail::do_get_key(instance, k);
}

template <>
inline void get_key(
    const unodb::olc_db<std::uint64_t, unodb::value_view>& instance,
    std::uint64_t k) {
  const quiescent_state_on_scope_exit qsbr_after_get{};
  detail::do_get_key(instance, k);
}

template <class Db>
void get_existing_key(const Db& instance, std::uint64_t k) {
  detail::do_get_existing_key(instance, k);
}

template <>
inline void get_existing_key(
    const unodb::olc_db<std::uint64_t, unodb::value_view>& instance,
    std::uint64_t k) {
  const quiescent_state_on_scope_exit qsbr_after_get{};
  detail::do_get_existing_key(instance, k);
}

// Teardown

template <class Db>
void destroy_tree(Db& instance, ::benchmark::State& state);

extern template void destroy_tree<unodb::db<std::uint64_t, unodb::value_view>>(
    unodb::db<std::uint64_t, unodb ::value_view>&, ::benchmark::State&);
extern template void
destroy_tree<unodb::mutex_db<std::uint64_t, unodb::value_view>>(
    unodb::mutex_db<std::uint64_t, unodb::value_view>&, ::benchmark::State&);
extern template void
destroy_tree<unodb::olc_db<std::uint64_t, unodb::value_view>>(
    unodb::olc_db<std::uint64_t, unodb::value_view>&, ::benchmark::State&);

extern template void
destroy_tree<unodb::db<unodb::key_view, unodb::value_view>>(
    unodb::db<unodb::key_view, unodb::value_view>&, ::benchmark::State&);
extern template void
destroy_tree<unodb::olc_db<unodb::key_view, unodb::value_view>>(
    unodb::olc_db<unodb::key_view, unodb::value_view>&, ::benchmark::State&);

extern template void destroy_tree<unodb::db<unodb::key_view, std::uint64_t>>(
    unodb::db<unodb::key_view, std::uint64_t>&, ::benchmark::State&);
extern template void
destroy_tree<unodb::olc_db<unodb::key_view, std::uint64_t>>(
    unodb::olc_db<unodb::key_view, std::uint64_t>&, ::benchmark::State&);

// ===================================================================
// key_view benchmark utilities
// ===================================================================

/// Pre-generated key set with stable storage for benchmarking.
///
/// Keys are stored in a contiguous byte buffer; key_view instances
/// point into it.  The buffer lifetime matches the key_view_set
/// lifetime, so key_views remain valid as long as the set exists.
///
/// Supports parameterization on value type.
/// for benchmarking Value=uint64_t (tuple identifier use case).
class key_view_set {
 public:
  /// G1: 9-byte compound keys (tag + uint64).
  ///
  /// All keys share the same tag byte → 8 shared prefix bytes →
  /// 1-level chain I4 at the root.  The uint64 component varies
  /// sequentially (0..n-1), producing unique dispatch bytes.
  static key_view_set compound(std::uint8_t tag, std::size_t n) {
    key_view_set ks;
    ks.key_len_ = 9;
    ks.buf_.resize(n * 9);
    ks.views_.reserve(n);
    unodb::key_encoder enc;
    for (std::size_t i = 0; i < n; ++i) {
      const auto kv = enc.reset()
                          .encode(tag)
                          .encode(static_cast<std::uint64_t>(i))
                          .get_key_view();
      UNODB_DETAIL_DISABLE_MSVC_WARNING(26459)
      UNODB_DETAIL_DISABLE_MSVC_WARNING(26481)
      std::copy(kv.begin(), kv.end(), ks.buf_.data() + i * 9);
      ks.views_.emplace_back(ks.buf_.data() + i * 9, 9);
      UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
      UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
    }
    return ks;
  }

  /// G2: 18-byte deep compound keys (tag + uint64 + mid + uint64).
  ///
  /// All keys share tag + fixed uint64 → 16 shared prefix bytes →
  /// 2-level chain.  Exercises multi-level chain insert/remove and
  /// the Step 2 loop in the atomic chain cut algorithm.
  static key_view_set deep_compound(std::uint8_t tag, std::size_t n) {
    key_view_set ks;
    ks.key_len_ = 18;
    ks.buf_.resize(n * 18);
    ks.views_.reserve(n);
    unodb::key_encoder enc;
    for (std::size_t i = 0; i < n; ++i) {
      const auto kv = enc.reset()
                          .encode(tag)
                          .encode(std::uint64_t{0x4242424242424242ULL})
                          .encode(static_cast<std::uint8_t>(i & 0xFF))
                          .encode(static_cast<std::uint64_t>(i))
                          .get_key_view();
      std::ranges::copy(kv, ks.buf_.data() + i * 18);
      ks.views_.emplace_back(ks.buf_.data() + i * 18, 18);
    }
    return ks;
  }

  /// G4: Multi-tag compound keys — independent chain subtrees.
  ///
  /// Uses @p tag_count distinct first bytes, assigned round-robin.
  /// The root inode fans out to tag_count independent chain subtrees.
  /// Simulates real-world key distribution where keys have diverse
  /// prefixes (e.g., different column values in a secondary index).
  static key_view_set multi_tag(std::uint8_t tag_count, std::size_t n) {
    key_view_set ks;
    ks.key_len_ = 9;
    ks.buf_.resize(n * 9);
    ks.views_.reserve(n);
    unodb::key_encoder enc;
    for (std::size_t i = 0; i < n; ++i) {
      const auto tag = static_cast<std::uint8_t>(1 + (i % tag_count));
      auto kv = enc.reset()
                    .encode(tag)
                    .encode(std::uint64_t{i / tag_count})
                    .get_key_view();
      std::copy(kv.begin(), kv.end(), ks.buf_.data() + i * 9);
      ks.views_.emplace_back(ks.buf_.data() + i * 9, 9);
    }
    return ks;
  }

  /// G6: Fixed-length keys with maximum chain depth.
  ///
  /// Produces keys of exactly @p key_len bytes: tag(1) + pad(key_len-2)
  /// + variant(1).  All keys within a tag group share (key_len-1) bytes,
  /// producing chain depth = (key_len - 2) / 8.  Four tag groups create
  /// a root I4.  Used in the key length sweep to characterize
  /// per-chain-level cost.
  ///
  /// @param key_len Key length in bytes (must be >= 2).
  /// @param n Number of keys (must be <= 256 * 4 = 1024).
  static key_view_set chain_depth(std::size_t key_len, std::size_t n) {
    UNODB_DETAIL_ASSERT(key_len >= 2);
    key_view_set ks;
    ks.key_len_ = key_len;
    ks.buf_.resize(n * key_len);
    ks.views_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      auto* const dst = ks.buf_.data() + i * key_len;
      // tag byte: rotate through 4 tags to create a root I4.
      dst[0] = static_cast<std::byte>(1 + (i / 256) % 4);
      // pad bytes: all 0x42 (shared prefix).
      for (std::size_t j = 1; j + 1 < key_len; ++j) dst[j] = std::byte{0x42};
      // variant byte: unique within each tag group.
      dst[key_len - 1] = static_cast<std::byte>(i & 0xFF);
      ks.views_.emplace_back(dst, key_len);
    }
    return ks;
  }

  /// G5: Dense sequential keys — no chains.
  ///
  /// encode(uint64) only, producing 8-byte keys with no shared prefix.
  /// Baseline for isolating key_view encoding overhead vs u64 keys.
  static key_view_set dense_sequential(std::size_t n) {
    key_view_set ks;
    ks.key_len_ = 8;
    ks.buf_.resize(n * 8);
    ks.views_.reserve(n);
    unodb::key_encoder enc;
    for (std::size_t i = 0; i < n; ++i) {
      auto kv =
          enc.reset().encode(static_cast<std::uint64_t>(i)).get_key_view();
      std::copy(kv.begin(), kv.end(), ks.buf_.data() + i * 8);
      ks.views_.emplace_back(ks.buf_.data() + i * 8, 8);
    }
    return ks;
  }

  [[nodiscard]] const std::vector<unodb::key_view>& keys() const noexcept {
    return views_;
  }
  [[nodiscard]] std::size_t size() const noexcept { return views_.size(); }
  [[nodiscard]] unodb::key_view operator[](std::size_t i) const noexcept {
    return views_[i];
  }

 private:
  std::vector<std::byte> buf_;
  std::vector<unodb::key_view> views_;
  std::size_t key_len_{0};
};

}  // namespace unodb::benchmark

#endif  // UNODB_DETAIL_MICRO_BENCHMARK_UTILS_HPP
