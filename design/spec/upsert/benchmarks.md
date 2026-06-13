> Benchmark plan for `upsert` (#847): 6 scenarios measuring throughput, contention, latency, and backoff strategies.

## Scope

New file: `benchmark/micro_benchmark_olc_upsert.cpp`. Uses existing
`micro_benchmark_concurrency.hpp` infrastructure with `concurrency_ranges`
(1–32 threads). [source: test-design]

---

## Benchmarks

| ID | Name | Setup | Workload | Source |
|----|------|-------|----------|--------|
| B1 | Upsert Update — Disjoint Keys | Pre-insert 2M keys (disjoint ranges/thread) | Each thread upserts with XOR lambda | test-design B1 |
| B2 | Upsert Update — Shared Hot Keys | Pre-insert 64 hot keys | All threads upsert-update same keys (increment) | test-design B2 |
| B3 | Upsert Erase — Shared Hot Keys | Pre-insert 64 hot keys | Half threads erase, half update; re-insert erased keys each iteration | test-design B3 |
| B4 | Upsert Insert Path — Disjoint | Empty tree | Each thread upserts unique keys | test-design B4 |
| B5 | Upsert Mixed — Random Operations | Pre-insert 1M keys | Random key + weighted action (50% update, 30% insert, 15% keep, 5% erase) | test-design B5 |
| B6 | Backoff Strategy Comparison | Same as B2 | Variants: (a) spin_wait_loop_body, (b) no backoff, (c) exponential, (d) randomized | test-design B6 |

---

## Counters per Benchmark

| ID | Counters |
|----|----------|
| B1 | ops/sec, items processed |
| B2 | ops/sec, OLC restart count (STATS), restarts/op ratio, p50/p95/p99 latency (ns) |
| B3 | ops/sec, erase retry count, erase success rate, p50/p95/p99 latency (ns) |
| B4 | ops/sec, ratio vs plain `insert()` |
| B5 | ops/sec by thread count, action distribution |
| B6 | ops/sec, p99 latency, restart count |

[source: test-design, §11.4]

---

## Latency Measurement

B2 and B3 MUST include p50/p95/p99 latency. [source: §11.4]

Implementation: thread-local ring buffer of `rdtsc` timestamps. Post-processed
into percentiles after benchmark loop completes. Requires invariant-TSC CPU
(checked at runtime) and thread pinning for cross-run comparability. [source: §11.4]

---

## Thread Counts

All benchmarks vary thread count: 1, 2, 4, 8, 16, 32 (or hardware max).
[source: test-design]

---

## Comparison Baselines

| Benchmark | Baseline |
|-----------|----------|
| B4 | Plain `insert()` on same workload — measures upsert overhead on insert path |
| B6 | Variant (a) is the default; others are compile-time alternatives for evaluation |

---

## Stats Counters (STATS build only)

| Counter | Used in |
|---------|---------|
| `upsert_erase_retry_count` | B3 (erase retry count) |
| `upsert_erase_retry_threshold_exceeded` | B3 (fires if retries > 64 per op) |
| OLC restart count (existing) | B2, B6 |

[source: C#3, C#5]

---

## Infrastructure

- File: `benchmark/micro_benchmark_olc_upsert.cpp`
- Header: `micro_benchmark_concurrency.hpp` (existing)
- Build: added to `benchmark/CMakeLists.txt`
- Key type: `uint64_t` (all benchmarks use fixed-size keys for reproducibility)
- Value type: `uint64_t` (VIS path exercised)

---

## Verification

- [ ] All 6 benchmarks compile and run on supported compilers
- [ ] Latency percentiles reported for B2 and B3
- [ ] B4 ratio vs plain insert is ≤ 1.05× (insert path overhead target)
- [ ] B6 produces comparable data across all 4 backoff variants
- [ ] STATS counters fire correctly in B3

## Deferred Items

- `key_view` benchmarks (variable-length key overhead — future)
- `value_view` benchmarks (explicit leaf path — future)
- NUMA-aware thread pinning (future infrastructure)
