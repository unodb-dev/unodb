// Copyright 2026 UnoDB contributors

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "art.hpp"
#include "art_common.hpp"
#include "micro_benchmark_utils.hpp"
#include "mutex_art.hpp"
#include "olc_art.hpp"
#include "portability_execution.hpp"

UNODB_DETAIL_DISABLE_MSVC_WARNING(26445)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)

namespace {

using db = unodb::benchmark::db;
using mutex_db = unodb::benchmark::mutex_db;
using olc_db = unodb::benchmark::olc_db;

using kv_db = unodb::benchmark::kv_db;
using kv_mutex_db = unodb::mutex_db<unodb::key_view, unodb::value_view>;
using kv_olc_db = unodb::benchmark::kv_olc_db;

constexpr auto val8 = std::array<std::byte, 8>{};
const auto value = unodb::value_view{val8};

// ===================================================================
// Helpers
// ===================================================================

/// Generate sorted sequential keys [0..n).
[[nodiscard]] auto sequential_keys(std::int64_t n) {
  std::vector<std::pair<std::uint64_t, unodb::value_view>> kv;
  kv.reserve(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i)
    kv.emplace_back(static_cast<std::uint64_t>(i), value);
  return kv;
}

/// Generate sorted random unique keys.
[[nodiscard]] auto random_keys(std::int64_t n) {
  std::mt19937_64 rng(42);  // NOLINT(cert-msc32-c,cert-msc51-cpp)
  std::vector<std::uint64_t> keys(static_cast<std::size_t>(n));
  std::generate(keys.begin(), keys.end(), rng);
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  std::vector<std::pair<std::uint64_t, unodb::value_view>> kv;
  kv.reserve(keys.size());
  for (auto k : keys) kv.emplace_back(k, value);
  return kv;
}

// ===================================================================
// BM01: bulk_load sequential uint64 keys (all tree modes)
// ===================================================================

template <class Db>
void BM_BulkLoad_seq(benchmark::State& state) {
  const auto kv = sequential_keys(state.range(0));
  for (const auto _ : state) {
    Db test_db;
    test_db.bulk_load(kv.begin(), kv.end());
    benchmark::DoNotOptimize(test_db);
    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(kv.size()));
}

template <class Db>
void BM_Insert_seq(benchmark::State& state) {
  const auto kv = sequential_keys(state.range(0));
  for (const auto _ : state) {
    Db test_db;
    for (const auto& [k, v] : kv) static_cast<void>(test_db.insert(k, v));
    benchmark::DoNotOptimize(test_db);
    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(kv.size()));
}

// ===================================================================
// BM02: bulk_load random uint64 keys (all tree modes)
// ===================================================================

template <class Db>
void BM_BulkLoad_rand(benchmark::State& state) {
  const auto kv = random_keys(state.range(0));
  for (const auto _ : state) {
    Db test_db;
    test_db.bulk_load(kv.begin(), kv.end());
    benchmark::DoNotOptimize(test_db);
    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(kv.size()));
}

template <class Db>
void BM_Insert_rand(benchmark::State& state) {
  const auto kv = random_keys(state.range(0));
  for (const auto _ : state) {
    Db test_db;
    for (const auto& [k, v] : kv) static_cast<void>(test_db.insert(k, v));
    benchmark::DoNotOptimize(test_db);
    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(kv.size()));
}

// ===================================================================
// BM03: bulk_load key_view compound keys (all key_view tree modes)
// ===================================================================

template <class Db>
void BM_BulkLoad_kv(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  auto keys = unodb::benchmark::key_view_set::compound(0x01, n);
  std::vector<std::pair<unodb::key_view, unodb::value_view>> kv;
  kv.reserve(n);
  for (std::size_t i = 0; i < n; ++i) kv.emplace_back(keys[i], value);

  for (const auto _ : state) {
    Db test_db;
    test_db.bulk_load(kv.begin(), kv.end());
    benchmark::DoNotOptimize(test_db);
    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <class Db>
void BM_Insert_kv(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  auto keys = unodb::benchmark::key_view_set::compound(0x01, n);
  std::vector<std::pair<unodb::key_view, unodb::value_view>> kv;
  kv.reserve(n);
  for (std::size_t i = 0; i < n; ++i) kv.emplace_back(keys[i], value);

  for (const auto _ : state) {
    Db test_db;
    for (const auto& [k, v] : kv) static_cast<void>(test_db.insert(k, v));
    benchmark::DoNotOptimize(test_db);
    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// ===================================================================
// BM04: bulk_load parallel scaling (olc_db)
// ===================================================================

void BM_BulkLoad_par_db(benchmark::State& state) {
  const auto kv = random_keys(state.range(0));
  for (const auto _ : state) {
    db test_db;
    test_db.bulk_load(std::execution::par, kv.begin(), kv.end());
    benchmark::DoNotOptimize(test_db);
    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(kv.size()));
}

void BM_BulkLoad_par_mutex(benchmark::State& state) {
  const auto kv = random_keys(state.range(0));
  for (const auto _ : state) {
    mutex_db test_db;
    test_db.bulk_load(std::execution::par, kv.begin(), kv.end());
    benchmark::DoNotOptimize(test_db);
    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(kv.size()));
}

void BM_BulkLoad_par_olc(benchmark::State& state) {
  const auto kv = random_keys(state.range(0));
  for (const auto _ : state) {
    olc_db test_db;
    test_db.bulk_load(std::execution::par, kv.begin(), kv.end());
    benchmark::DoNotOptimize(test_db);
    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(kv.size()));
}

// ===================================================================
// BM05: bulk_load scaling (N varies, all tree modes)
// ===================================================================

template <class Db>
void BM_BulkLoad_scaling(benchmark::State& state) {
  const auto kv = random_keys(state.range(0));
  for (const auto _ : state) {
    Db test_db;
    test_db.bulk_load(kv.begin(), kv.end());
    benchmark::DoNotOptimize(test_db);
    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(kv.size()));
}

// ===================================================================
// BM07: memory comparison (requires STATS build)
// ===================================================================

#ifdef UNODB_DETAIL_WITH_STATS

template <class Db>
void BM_BulkLoad_memory(benchmark::State& state) {
  const auto kv = random_keys(state.range(0));
  std::size_t bulk_mem = 0;
  std::size_t insert_mem = 0;

  for (const auto _ : state) {
    {
      Db bulk_db;
      bulk_db.bulk_load(kv.begin(), kv.end());
      bulk_mem = bulk_db.get_current_memory_use();
      benchmark::DoNotOptimize(bulk_mem);
      state.PauseTiming();
      bulk_db.clear();
      state.ResumeTiming();
    }
    {
      state.PauseTiming();
      Db insert_db;
      for (const auto& [k, v] : kv) static_cast<void>(insert_db.insert(k, v));
      insert_mem = insert_db.get_current_memory_use();
      insert_db.clear();
      state.ResumeTiming();
    }
  }
  state.counters["bulk_bytes"] = benchmark::Counter(
      static_cast<double>(bulk_mem), benchmark::Counter::kDefaults,
      benchmark::Counter::OneK::kIs1024);
  state.counters["insert_bytes"] = benchmark::Counter(
      static_cast<double>(insert_mem), benchmark::Counter::kDefaults,
      benchmark::Counter::OneK::kIs1024);
}

#endif  // UNODB_DETAIL_WITH_STATS

// ===================================================================
// BM08: first-scan penalty after bulk_load (all tree modes)
// ===================================================================

template <class Db>
void BM_BulkLoad_first_scan(benchmark::State& state) {
  const auto kv = random_keys(state.range(0));
  for (const auto _ : state) {
    state.PauseTiming();
    Db test_db;
    test_db.bulk_load(kv.begin(), kv.end());
    state.ResumeTiming();

    std::size_t count = 0;
    test_db.scan([&count](const auto& /*visitor*/) {
      ++count;
      return false;  // continue
    });
    benchmark::DoNotOptimize(count);

    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(kv.size()));
}

template <class Db>
void BM_Insert_first_scan(benchmark::State& state) {
  const auto kv = random_keys(state.range(0));
  for (const auto _ : state) {
    state.PauseTiming();
    Db test_db;
    for (const auto& [k, v] : kv) static_cast<void>(test_db.insert(k, v));
    state.ResumeTiming();

    std::size_t count = 0;
    test_db.scan([&count](const auto& /*visitor*/) {
      ++count;
      return false;  // continue
    });
    benchmark::DoNotOptimize(count);

    state.PauseTiming();
    test_db.clear();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<std::int64_t>(kv.size()));
}

}  // namespace

// ===================================================================
// Registration — all tree modes
// ===================================================================

UNODB_START_BENCHMARKS()

// BM01: Sequential bulk_load vs insert (db, mutex_db, olc_db)
BENCHMARK_TEMPLATE(BM_BulkLoad_seq, db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_seq, mutex_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_seq, olc_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_seq, db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_seq, mutex_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_seq, olc_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);

// BM02: Random bulk_load vs insert (db, mutex_db, olc_db)
BENCHMARK_TEMPLATE(BM_BulkLoad_rand, db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_rand, mutex_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_rand, olc_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_rand, db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_rand, mutex_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_rand, olc_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);

// BM03: key_view bulk_load vs insert (kv_db, kv_mutex_db, kv_olc_db)
BENCHMARK_TEMPLATE(BM_BulkLoad_kv, kv_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_kv, kv_mutex_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_kv, kv_olc_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_kv, kv_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_kv, kv_mutex_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_kv, kv_olc_db)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond);

// BM04: Parallel execution (all tree modes, par vs seq)
BENCHMARK(BM_BulkLoad_par_db)->Arg(1 << 20)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BulkLoad_par_mutex)->Arg(1 << 20)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BulkLoad_par_olc)->Arg(1 << 20)->Unit(benchmark::kMillisecond);

// BM05: Size scaling (db, mutex_db, olc_db)
BENCHMARK_TEMPLATE(BM_BulkLoad_scaling, db)
    ->Arg(1 << 10)
    ->Arg(1 << 14)
    ->Arg(1 << 17)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_scaling, mutex_db)
    ->Arg(1 << 10)
    ->Arg(1 << 14)
    ->Arg(1 << 17)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_scaling, olc_db)
    ->Arg(1 << 10)
    ->Arg(1 << 14)
    ->Arg(1 << 17)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);

// BM07: Memory comparison (stats build only; db, mutex_db, olc_db)
#ifdef UNODB_DETAIL_WITH_STATS
BENCHMARK_TEMPLATE(BM_BulkLoad_memory, db)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_memory, mutex_db)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_memory, olc_db)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);
#endif

// BM08: First-scan penalty (db, mutex_db, olc_db)
BENCHMARK_TEMPLATE(BM_BulkLoad_first_scan, db)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_first_scan, mutex_db)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_BulkLoad_first_scan, olc_db)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_first_scan, db)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_first_scan, mutex_db)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Insert_first_scan, olc_db)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMillisecond);

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

UNODB_BENCHMARK_MAIN();
