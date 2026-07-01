// Copyright 2026 UnoDB contributors

// Example: bulk_load with sequential and parallel execution policies.

#include "global.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "art.hpp"
#include "art_common.hpp"
#include "mutex_art.hpp"
#include "portability_execution.hpp"

namespace {

constexpr std::size_t key_count = 100'000;

/// Build sorted key-value pairs. The caller is responsible for
/// pre-encoding and pre-sorting keys before calling bulk_load.
[[nodiscard]] auto make_sorted_data() {
  constexpr std::array<std::byte, 8> val{};
  const auto value = unodb::value_view{val};
  std::vector<std::pair<std::uint64_t, unodb::value_view>> kv;
  kv.reserve(key_count);
  for (std::size_t i = 0; i < key_count; ++i)
    kv.emplace_back(static_cast<std::uint64_t>(i), value);
  return kv;
}

}  // namespace

int main() {
  const auto data = make_sorted_data();

  // ─── Sequential bulk_load (default) ────────────────────────────────────────
  {
    unodb::db<std::uint64_t, unodb::value_view> tree;
    tree.bulk_load(data.begin(), data.end());  // default: std::execution::seq
    std::cerr << "Sequential bulk_load: " << key_count << " keys loaded\n";
    std::cerr << "  get(42) found: " << tree.get(42).has_value() << '\n';
    tree.clear();
  }

  // ─── Parallel bulk_load ────────────────────────────────────────────────────
  // std::execution::par enables concurrent subtree construction.
  // The implementation partitions at the root level and builds each
  // subtree on a separate thread. Safe for all tree modes because
  // bulk_load operates on an unpublished tree (no concurrent readers).
  {
    unodb::db<std::uint64_t, unodb::value_view> tree;
    tree.bulk_load(std::execution::par, data.begin(), data.end());
    std::cerr << "Parallel bulk_load: " << key_count << " keys loaded\n";
    std::cerr << "  get(99999) found: " << tree.get(99999).has_value() << '\n';
    tree.clear();
  }

  // ─── mutex_db: same API ────────────────────────────────────────────────────
  {
    unodb::mutex_db<std::uint64_t, unodb::value_view> tree;
    tree.bulk_load(std::execution::par, data.begin(), data.end());
    std::cerr << "mutex_db parallel bulk_load: " << key_count
              << " keys loaded\n";
    tree.clear();
  }

  std::cerr << "Done.\n";
}
