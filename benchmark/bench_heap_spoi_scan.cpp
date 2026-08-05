// bench_heap_spoi_scan.cpp — Heap-backed secondary index scan benchmark.
//
// Exercises the new olc_db<key_view, uint64_t, HeapType> pattern with 14.5M
// SPOI tuples.  The heap is allocated on 2MB huge pages to simulate the
// physical layout of REL_VP in the p8 design.
//
// NOT intended for check-in to unodb — experimental performance exploration.
//
// Build:
//   cd build-heap-release
//   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++ \
//     -DTESTS=OFF -DBENCHMARKS=ON -DSTATS=OFF
//   make -j$(nproc) bench_heap_spoi_scan
//
// Run:
//   ./benchmark/bench_heap_spoi_scan --benchmark_filter=".*"

#include "global.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <sys/mman.h>

#include <benchmark/benchmark.h>

#include "art_common.hpp"
#include "olc_art.hpp"
#include "qsbr.hpp"

UNODB_DETAIL_DISABLE_MSVC_WARNING(4189)

namespace {

// ============================================================================
// Huge-page backed tuple heap (simulates REL_VP physical layout)
// ============================================================================

/// A flat array of 32-byte tuples backed by 2MB huge pages.
/// Satisfies the TupleHeap concept.
class HugePageHeap {
 public:
  explicit HugePageHeap(std::size_t tuple_count)
      : count_{tuple_count},
        byte_size_{tuple_count * kTupleSize} {
    // Allocate with 2MB huge pages (MAP_HUGETLB).
    // Falls back to regular pages if huge pages are unavailable.
    data_ = static_cast<std::byte*>(mmap(
        nullptr, byte_size_, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT),
        -1, 0));
    if (data_ == MAP_FAILED) {
      // Fallback: regular pages
      std::cerr << "  [WARN] 2MB huge pages unavailable, using regular pages\n";
      data_ = static_cast<std::byte*>(mmap(
          nullptr, byte_size_, PROT_READ | PROT_WRITE,
          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
      if (data_ == MAP_FAILED) {
        std::cerr << "  [FATAL] mmap failed\n";
        std::abort();
      }
      huge_pages_ = false;
    } else {
      huge_pages_ = true;
    }
  }

  ~HugePageHeap() {
    if (data_ != nullptr && data_ != MAP_FAILED) {
      munmap(data_, byte_size_);
    }
  }

  HugePageHeap(const HugePageHeap&) = delete;
  HugePageHeap& operator=(const HugePageHeap&) = delete;

  /// Load sorted 32-byte keys from a binary file.
  void load_from_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      std::cerr << "Cannot open " << path << "\n";
      std::abort();
    }
    f.read(reinterpret_cast<char*>(data_),
           static_cast<std::streamsize>(byte_size_));
    const auto bytes_read = static_cast<std::size_t>(f.gcount());
    const auto loaded = bytes_read / kTupleSize;
    if (loaded < count_) {
      std::cerr << "  [WARN] Loaded only " << loaded << " of " << count_
                << " tuples\n";
      count_ = loaded;
    }
  }

  /// TupleHeap concept: extract_key(tuple_id, buf) -> key_view.
  [[nodiscard]] unodb::key_view extract_key(
      std::uint64_t tuple_id,
      [[maybe_unused]] unodb::key_encoder& buf) const noexcept {
    return {data_ + tuple_id * kTupleSize, kTupleSize};
  }

  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  [[nodiscard]] bool using_huge_pages() const noexcept { return huge_pages_; }

  [[nodiscard]] unodb::key_view key_at(std::size_t i) const noexcept {
    return {data_ + i * kTupleSize, kTupleSize};
  }

  static constexpr std::size_t kTupleSize = 32;

 private:
  std::byte* data_{nullptr};
  std::size_t count_;
  std::size_t byte_size_;
  bool huge_pages_{false};
};

// Verify concept satisfaction.
static_assert(unodb::TupleHeap<HugePageHeap, std::uint64_t>);

// ============================================================================
// Min-distinct-prefix computation (same as POC benchmark)
// ============================================================================

std::uint8_t common_prefix_len(const std::byte* a, const std::byte* b,
                               std::uint8_t max_len) noexcept {
  std::uint8_t i = 0;
  while (i < max_len && a[i] == b[i]) ++i;
  return i;
}

struct prefix_set {
  std::vector<std::byte> buf;
  std::vector<std::uint32_t> off;
  std::vector<std::uint8_t> len;

  [[nodiscard]] unodb::key_view prefix(std::size_t i) const noexcept {
    return {buf.data() + off[i], len[i]};
  }
};

prefix_set compute_min_prefixes(const HugePageHeap& heap) {
  const auto n = heap.count();
  constexpr std::uint8_t key_len = 32;

  prefix_set ps;
  ps.off.reserve(n);
  ps.len.reserve(n);
  ps.buf.reserve(n * 5);

  std::uint32_t offset = 0;
  for (std::size_t i = 0; i < n; ++i) {
    auto cur = heap.key_at(i);

    std::uint8_t need_left = 1;
    if (i > 0) {
      auto prev = heap.key_at(i - 1);
      need_left = static_cast<std::uint8_t>(
          common_prefix_len(cur.data(), prev.data(), key_len) + 1);
    }

    std::uint8_t need_right = 1;
    if (i + 1 < n) {
      auto next = heap.key_at(i + 1);
      need_right = static_cast<std::uint8_t>(
          common_prefix_len(cur.data(), next.data(), key_len) + 1);
    }

    const auto plen = std::max(need_left, need_right);
    ps.off.push_back(offset);
    ps.len.push_back(plen);
    ps.buf.insert(ps.buf.end(), cur.data(), cur.data() + plen);
    offset += plen;
  }

  return ps;
}

// ============================================================================
// Global dataset (loaded once)
// ============================================================================

struct dataset {
  std::unique_ptr<HugePageHeap> heap;
  prefix_set prefixes;
  bool ready{false};
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
dataset g_ds;

constexpr std::size_t kMaxTuples = 14'500'000;

dataset& get_dataset() {
  if (!g_ds.ready) {
    std::cerr << "=== Heap SPOI Scan Benchmark ===\n";
    std::cerr << "Loading " << kMaxTuples << " tuples from /tmp/spatial-art-keys.bin...\n";

    g_ds.heap = std::make_unique<HugePageHeap>(kMaxTuples);
    g_ds.heap->load_from_file("/tmp/spatial-art-keys.bin");

    std::cerr << "  Loaded " << g_ds.heap->count() << " tuples ("
              << (g_ds.heap->count() * 32 / (1024 * 1024)) << " MB)\n";
    std::cerr << "  Huge pages: "
              << (g_ds.heap->using_huge_pages() ? "YES (2MB)" : "NO (fallback)")
              << "\n";

    std::cerr << "Computing min-distinct-prefixes...\n";
    g_ds.prefixes = compute_min_prefixes(*g_ds.heap);

    double avg_len = static_cast<double>(g_ds.prefixes.buf.size()) /
                     static_cast<double>(g_ds.heap->count());
    std::cerr << "  Avg prefix len: " << avg_len << " bytes\n";
    std::cerr << "  Prefix storage: "
              << (g_ds.prefixes.buf.size() / (1024 * 1024)) << " MB\n";

    g_ds.ready = true;
  }
  return g_ds;
}

// ============================================================================
// Tree types
// ============================================================================

// The EXISTING pattern: external min-prefix, tree stores short keys.
using existing_db_t = unodb::olc_db<unodb::key_view, std::uint64_t>;

// The NEW pattern: heap-backed, tree uses the same min-prefix keys.
using heap_db_t = unodb::olc_db<unodb::key_view, std::uint64_t, HugePageHeap>;

// ============================================================================
// Benchmark helpers
// ============================================================================

template <class Db>
struct tree_builder;

// Specialization for existing (void PolicyTag) tree.
template <>
struct tree_builder<existing_db_t> {
  static std::unique_ptr<existing_db_t> build(const dataset& ds, std::size_t n) {
    auto db = std::make_unique<existing_db_t>();
    for (std::size_t i = 0; i < n; ++i) {
      std::ignore = db->insert(ds.prefixes.prefix(i),
                               static_cast<std::uint64_t>(i));
    }
    return db;
  }
};

// Specialization for heap-backed tree.
template <>
struct tree_builder<heap_db_t> {
  static std::unique_ptr<heap_db_t> build(const dataset& ds, std::size_t n) {
    auto db = std::make_unique<heap_db_t>(*ds.heap);
    for (std::size_t i = 0; i < n; ++i) {
      std::ignore = db->insert(ds.prefixes.prefix(i),
                               static_cast<std::uint64_t>(i));
    }
    return db;
  }
};

// ============================================================================
// Benchmarks
// ============================================================================

// Full scan — just iterate tuple_ids (primary throughput metric).
template <class Db>
void BM_full_scan(benchmark::State& state) {
  auto& ds = get_dataset();
  const auto n = std::min(static_cast<std::size_t>(state.range(0)),
                          ds.heap->count());

  auto db = tree_builder<Db>::build(ds, n);

  for (const auto _ : state) {
    std::size_t count = 0;
    db->scan([&count](auto /*visitor*/) noexcept {
      ++count;
      return false;
    });
    benchmark::DoNotOptimize(count);
    assert(count == n);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// Full scan with key recovery from heap (simulates real consumer).
template <class Db>
void BM_scan_with_recovery(benchmark::State& state) {
  auto& ds = get_dataset();
  const auto n = std::min(static_cast<std::size_t>(state.range(0)),
                          ds.heap->count());

  auto db = tree_builder<Db>::build(ds, n);

  for (const auto _ : state) {
    std::size_t count = 0;
    std::uint64_t checksum = 0;
    db->scan([&count, &checksum, &ds](auto visitor) noexcept {
      const auto tuple_id = visitor.get_value();
      auto full_key = ds.heap->key_at(tuple_id);
      checksum += static_cast<std::uint64_t>(full_key[0]);
      ++count;
      return false;
    });
    benchmark::DoNotOptimize(count);
    benchmark::DoNotOptimize(checksum);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// 20% range scan from midpoint.
template <class Db>
void BM_range_scan(benchmark::State& state) {
  auto& ds = get_dataset();
  const auto n = std::min(static_cast<std::size_t>(state.range(0)),
                          ds.heap->count());

  auto db = tree_builder<Db>::build(ds, n);

  const auto start_idx = n * 4 / 10;
  const auto scan_limit = n / 5;

  for (const auto _ : state) {
    std::size_t count = 0;
    db->scan_from(ds.prefixes.prefix(start_idx),
                  [&count, scan_limit](auto /*v*/) noexcept {
                    ++count;
                    return count >= scan_limit;
                  });
    benchmark::DoNotOptimize(count);
  }

  state.SetItemsProcessed(state.iterations() *
                           static_cast<std::int64_t>(scan_limit));
}

// Insert benchmark (tree build time).
template <class Db>
void BM_insert(benchmark::State& state) {
  auto& ds = get_dataset();
  const auto n = std::min(static_cast<std::size_t>(state.range(0)),
                          ds.heap->count());

  for (const auto _ : state) {
    auto db = tree_builder<Db>::build(ds, n);
    benchmark::DoNotOptimize(db);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// --- Registration ---

void spoi_sizes(benchmark::internal::Benchmark* b) {
  for (auto n : {1'000'000, 4'000'000, 14'500'000}) b->Arg(n);
}

// Existing pattern (external prefix, void PolicyTag)
BENCHMARK(BM_full_scan<existing_db_t>)
    ->Apply(spoi_sizes)->Unit(benchmark::kMillisecond)
    ->Name("FullScan/existing");
BENCHMARK(BM_scan_with_recovery<existing_db_t>)
    ->Apply(spoi_sizes)->Unit(benchmark::kMillisecond)
    ->Name("ScanRecover/existing");
BENCHMARK(BM_range_scan<existing_db_t>)
    ->Apply(spoi_sizes)->Unit(benchmark::kMillisecond)
    ->Name("RangeScan20pct/existing");
BENCHMARK(BM_insert<existing_db_t>)
    ->Apply(spoi_sizes)->Unit(benchmark::kMillisecond)
    ->Name("Insert/existing");

// New pattern (heap-backed, PolicyTag = HugePageHeap)
BENCHMARK(BM_full_scan<heap_db_t>)
    ->Apply(spoi_sizes)->Unit(benchmark::kMillisecond)
    ->Name("FullScan/heap");
BENCHMARK(BM_scan_with_recovery<heap_db_t>)
    ->Apply(spoi_sizes)->Unit(benchmark::kMillisecond)
    ->Name("ScanRecover/heap");
BENCHMARK(BM_range_scan<heap_db_t>)
    ->Apply(spoi_sizes)->Unit(benchmark::kMillisecond)
    ->Name("RangeScan20pct/heap");
BENCHMARK(BM_insert<heap_db_t>)
    ->Apply(spoi_sizes)->Unit(benchmark::kMillisecond)
    ->Name("Insert/heap");

}  // namespace

BENCHMARK_MAIN();
