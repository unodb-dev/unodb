// Copyright 2026 UnoDB contributors

// Should be the first include
#include "global.hpp"

#include <cstddef>
#include <cstdint>

#include <benchmark/benchmark.h>

#include "micro_benchmark_utils.hpp"
#include "qsbr.hpp"

/// @file
/// key_view micro-benchmarks for ART tree operations.

// Google Benchmark uses `for (auto _ : state)` where _ is intentionally unused.
UNODB_DETAIL_DISABLE_MSVC_WARNING(4189)
UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
UNODB_DETAIL_DISABLE_GCC_WARNING("-Wuseless-cast")
///
/// ## Benchmark groups
///
/// **Core chain benchmarks** (8-byte values):
/// Exercise insert/get/remove/scan across four key generators at
/// multiple tree sizes (1K, 16K, 262K).
///
/// **kv_vs_u64 comparison** (100-byte values):
/// Direct comparison against the u64 dense_insert benchmark.  Uses
/// dense sequential 8-byte encoded keys (no chains) and 100-byte
/// values to match the u64 benchmark's value size.  Timing structure
/// (PauseTiming/destroy_tree) is identical to the u64 benchmark.
///
/// **Key length sweep** (8-byte values, 1024 keys):
/// Varies key length from 8 to 256 bytes, producing chain depths
/// from 0 to 31.  Characterizes the per-chain-level cost.
///
/// ## Key generators
///
/// - **G1 compound** (9 bytes): tag(1) + uint64.  Same tag for all
///   keys → 1-level chain.  The basic chain workload.
/// - **G2 deep** (18 bytes): tag(1) + uint64 + mid(1) + uint64.
///   Same tag + fixed uint64 → 2-level chain.  Exercises multi-level
///   chain insert/remove and the Step 2 loop in the atomic chain cut.
/// - **G4 multi_tag** (9 bytes): 8 distinct first bytes, round-robin.
///   Root inode is I16 with 8 independent chain subtrees.  Simulates
///   real-world key distribution where keys have diverse prefixes
///   (e.g., different column values in a secondary index).
/// - **G5 dense** (8 bytes): encode(uint64) only, no chains.
///   Baseline for isolating key_view encoding overhead vs u64.
/// - **G6 chain_depth** (variable length): fixed-length keys with
///   maximum shared prefix.  Chain depth = (key_len - 2) / 8.
///   Used in the key length sweep.

UNODB_START_BENCHMARKS()

using unodb::benchmark::key_view_set;

namespace {

constexpr auto val_bytes = std::array<std::byte, 8>{};
const auto val = unodb::value_view{val_bytes};

constexpr auto val100_bytes = std::array<std::byte, 100>{};
const auto val100 = unodb::value_view{val100_bytes};

// Core benchmarks — timing matches u64 dense_insert exactly:
//   PauseTiming → construct → ClobberMemory → ResumeTiming
//   ... timed work ...
//   PauseTiming → destroy_tree (clear + ClobberMemory + ResumeTiming)

template <class Db>
void chain_insert(benchmark::State& state, key_view_set (*gen)(std::size_t)) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = gen(n);
  for (const auto _ : state) {
    state.PauseTiming();
    Db db;
    benchmark::ClobberMemory();
    state.ResumeTiming();
    for (std::size_t i = 0; i < n; ++i)
      benchmark::DoNotOptimize(db.insert(ks[i], val));
    state.PauseTiming();
    unodb::benchmark::destroy_tree(db, state);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <class Db>
void chain_get(benchmark::State& state, key_view_set (*gen)(std::size_t)) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = gen(n);
  Db db;
  for (std::size_t i = 0; i < n; ++i) std::ignore = db.insert(ks[i], val);
  for (const auto _ : state) {
    for (std::size_t i = 0; i < n; ++i) benchmark::DoNotOptimize(db.get(ks[i]));
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <class Db>
void chain_remove(benchmark::State& state, key_view_set (*gen)(std::size_t)) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = gen(n);
  for (const auto _ : state) {
    state.PauseTiming();
    Db db;
    for (std::size_t i = 0; i < n; ++i) std::ignore = db.insert(ks[i], val);
    benchmark::ClobberMemory();
    state.ResumeTiming();
    for (std::size_t i = 0; i < n; ++i)
      benchmark::DoNotOptimize(db.remove(ks[i]));
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <class Db>
void chain_scan(benchmark::State& state, key_view_set (*gen)(std::size_t)) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = gen(n);
  Db db;
  for (std::size_t i = 0; i < n; ++i) std::ignore = db.insert(ks[i], val);
  for (const auto _ : state) {
    std::size_t count = 0;
    db.scan([&count](auto /*visitor*/) noexcept {
      ++count;
      return false;
    });
    benchmark::DoNotOptimize(count);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// Comparison vs u64: 100-byte values, dense 8B keys.

template <class Db>
void kv_vs_u64_insert(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = key_view_set::dense_sequential(n);
  for (const auto _ : state) {
    state.PauseTiming();
    Db db;
    benchmark::ClobberMemory();
    state.ResumeTiming();
    for (std::size_t i = 0; i < n; ++i)
      benchmark::DoNotOptimize(db.insert(ks[i], val100));
    state.PauseTiming();
    unodb::benchmark::destroy_tree(db, state);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <class Db>
void kv_vs_u64_get(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = key_view_set::dense_sequential(n);
  Db db;
  for (std::size_t i = 0; i < n; ++i) std::ignore = db.insert(ks[i], val100);
  for (const auto _ : state) {
    for (std::size_t i = 0; i < n; ++i) benchmark::DoNotOptimize(db.get(ks[i]));
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <class Db>
void kv_vs_u64_remove(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = key_view_set::dense_sequential(n);
  for (const auto _ : state) {
    state.PauseTiming();
    Db db;
    for (std::size_t i = 0; i < n; ++i) std::ignore = db.insert(ks[i], val100);
    benchmark::ClobberMemory();
    state.ResumeTiming();
    for (std::size_t i = 0; i < n; ++i)
      benchmark::DoNotOptimize(db.remove(ks[i]));
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// Value-in-slot benchmarks: db<key_view, uint64_t>
// Trees with leaf elimination — values packed directly into inode slots.

template <class Db>
void vis_insert(benchmark::State& state, key_view_set (*gen)(std::size_t)) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = gen(n);
  for (const auto _ : state) {
    state.PauseTiming();
    Db db;
    benchmark::ClobberMemory();
    state.ResumeTiming();
    for (std::size_t i = 0; i < n; ++i)
      benchmark::DoNotOptimize(db.insert(ks[i], static_cast<std::uint64_t>(i)));
    state.PauseTiming();
    unodb::benchmark::destroy_tree(db, state);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <class Db>
void vis_get(benchmark::State& state, key_view_set (*gen)(std::size_t)) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = gen(n);
  Db db;
  for (std::size_t i = 0; i < n; ++i)
    std::ignore = db.insert(ks[i], static_cast<std::uint64_t>(i));
  for (const auto _ : state) {
    for (std::size_t i = 0; i < n; ++i) benchmark::DoNotOptimize(db.get(ks[i]));
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <class Db>
void vis_remove(benchmark::State& state, key_view_set (*gen)(std::size_t)) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = gen(n);
  for (const auto _ : state) {
    state.PauseTiming();
    Db db;
    for (std::size_t i = 0; i < n; ++i)
      std::ignore = db.insert(ks[i], static_cast<std::uint64_t>(i));
    benchmark::ClobberMemory();
    state.ResumeTiming();
    for (std::size_t i = 0; i < n; ++i)
      benchmark::DoNotOptimize(db.remove(ks[i]));
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <class Db>
void vis_scan(benchmark::State& state, key_view_set (*gen)(std::size_t)) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ks = gen(n);
  Db db;
  for (std::size_t i = 0; i < n; ++i)
    std::ignore = db.insert(ks[i], static_cast<std::uint64_t>(i));
  for (const auto _ : state) {
    std::size_t count = 0;
    db.scan([&count](auto /*visitor*/) noexcept {
      ++count;
      return false;
    });
    benchmark::DoNotOptimize(count);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// Key generators

key_view_set gen_compound(std::size_t n) {
  return key_view_set::compound(0x42, n);
}
key_view_set gen_deep(std::size_t n) {
  return key_view_set::deep_compound(0x42, n);
}
key_view_set gen_multi_tag(std::size_t n) {
  return key_view_set::multi_tag(8, n);
}
key_view_set gen_dense(std::size_t n) {
  return key_view_set::dense_sequential(n);
}

// Key length sweep generators: chain depth = (key_len - 2) / 8

key_view_set gen_kl8(std::size_t n) { return key_view_set::chain_depth(8, n); }
key_view_set gen_kl16(std::size_t n) {
  return key_view_set::chain_depth(16, n);
}
key_view_set gen_kl32(std::size_t n) {
  return key_view_set::chain_depth(32, n);
}
key_view_set gen_kl64(std::size_t n) {
  return key_view_set::chain_depth(64, n);
}
key_view_set gen_kl128(std::size_t n) {
  return key_view_set::chain_depth(128, n);
}
key_view_set gen_kl256(std::size_t n) {
  return key_view_set::chain_depth(256, n);
}

// Sizes

void kv_sizes(benchmark::internal::Benchmark* b) {
  for (auto n : {1 << 10, 1 << 14, 1 << 18}) b->Arg(n);
}

void u64_sizes(benchmark::internal::Benchmark* b) {
  for (auto n : {1 << 12, 1 << 15, 1 << 18}) b->Arg(n);
}

void kl_sizes(benchmark::internal::Benchmark* b) { b->Arg(1024); }

// Registration: db

using DB = unodb::benchmark::kv_db;

BENCHMARK_CAPTURE(chain_insert<DB>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_insert<DB>, deep, gen_deep)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_insert<DB>, multi_tag, gen_multi_tag)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_insert<DB>, dense, gen_dense)->Apply(kv_sizes);

BENCHMARK_CAPTURE(chain_get<DB>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_get<DB>, deep, gen_deep)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_get<DB>, multi_tag, gen_multi_tag)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_get<DB>, dense, gen_dense)->Apply(kv_sizes);

BENCHMARK_CAPTURE(chain_remove<DB>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_remove<DB>, deep, gen_deep)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_remove<DB>, multi_tag, gen_multi_tag)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_remove<DB>, dense, gen_dense)->Apply(kv_sizes);

BENCHMARK_CAPTURE(chain_scan<DB>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_scan<DB>, dense, gen_dense)->Apply(kv_sizes);

BENCHMARK_CAPTURE(chain_insert<DB>, kl8, gen_kl8)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_insert<DB>, kl16, gen_kl16)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_insert<DB>, kl32, gen_kl32)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_insert<DB>, kl64, gen_kl64)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_insert<DB>, kl128, gen_kl128)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_insert<DB>, kl256, gen_kl256)->Apply(kl_sizes);

BENCHMARK_CAPTURE(chain_get<DB>, kl8, gen_kl8)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_get<DB>, kl16, gen_kl16)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_get<DB>, kl32, gen_kl32)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_get<DB>, kl64, gen_kl64)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_get<DB>, kl128, gen_kl128)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_get<DB>, kl256, gen_kl256)->Apply(kl_sizes);

BENCHMARK_CAPTURE(chain_remove<DB>, kl8, gen_kl8)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_remove<DB>, kl16, gen_kl16)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_remove<DB>, kl32, gen_kl32)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_remove<DB>, kl64, gen_kl64)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_remove<DB>, kl128, gen_kl128)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_remove<DB>, kl256, gen_kl256)->Apply(kl_sizes);

// Registration: olc_db

using OLC = unodb::benchmark::kv_olc_db;

BENCHMARK_CAPTURE(chain_insert<OLC>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_insert<OLC>, dense, gen_dense)->Apply(kv_sizes);

BENCHMARK_CAPTURE(chain_get<OLC>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_get<OLC>, dense, gen_dense)->Apply(kv_sizes);

BENCHMARK_CAPTURE(chain_remove<OLC>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_remove<OLC>, dense, gen_dense)->Apply(kv_sizes);

BENCHMARK_CAPTURE(chain_scan<OLC>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(chain_scan<OLC>, dense, gen_dense)->Apply(kv_sizes);

BENCHMARK_CAPTURE(chain_insert<OLC>, kl16, gen_kl16)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_insert<OLC>, kl64, gen_kl64)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_insert<OLC>, kl256, gen_kl256)->Apply(kl_sizes);

BENCHMARK_CAPTURE(chain_get<OLC>, kl16, gen_kl16)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_get<OLC>, kl64, gen_kl64)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_get<OLC>, kl256, gen_kl256)->Apply(kl_sizes);

BENCHMARK_CAPTURE(chain_remove<OLC>, kl16, gen_kl16)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_remove<OLC>, kl64, gen_kl64)->Apply(kl_sizes);
BENCHMARK_CAPTURE(chain_remove<OLC>, kl256, gen_kl256)->Apply(kl_sizes);

// Registration: kv vs u64 comparison (100B values)

BENCHMARK(kv_vs_u64_insert<DB>)->Apply(u64_sizes);
BENCHMARK(kv_vs_u64_get<DB>)->Apply(u64_sizes);
BENCHMARK(kv_vs_u64_remove<DB>)->Apply(u64_sizes);
BENCHMARK(kv_vs_u64_insert<OLC>)->Apply(u64_sizes);

// Registration: value-in-slot (VIS)

using VIS = unodb::benchmark::kv_u64_db;
using VIS_OLC = unodb::benchmark::kv_u64_olc_db;

BENCHMARK_CAPTURE(vis_insert<VIS>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_insert<VIS>, dense, gen_dense)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_get<VIS>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_get<VIS>, dense, gen_dense)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_remove<VIS>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_remove<VIS>, dense, gen_dense)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_scan<VIS>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_scan<VIS>, dense, gen_dense)->Apply(kv_sizes);

BENCHMARK_CAPTURE(vis_insert<VIS_OLC>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_insert<VIS_OLC>, dense, gen_dense)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_get<VIS_OLC>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_get<VIS_OLC>, dense, gen_dense)->Apply(kv_sizes);
// OLC vis_remove omitted: single-threaded remove benchmark not meaningful for
// OLC (no contention to measure).
BENCHMARK_CAPTURE(vis_scan<VIS_OLC>, compound, gen_compound)->Apply(kv_sizes);
BENCHMARK_CAPTURE(vis_scan<VIS_OLC>, dense, gen_dense)->Apply(kv_sizes);

}  // namespace

UNODB_BENCHMARK_MAIN();

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_GCC_WARNINGS()
