# Upsert (#847) — Test & Microbenchmark Design

## Test Suite

Tests are numbered continuing from the design doc (§7 tests 1-16, §9.3 tests 17-22).
Test 23 added for the erase action per Round 2.

### Unit Tests (all db types: `db`, `mutex_db`, `olc_db`)

| # | Test | Key assertion |
|---|------|---------------|
| 1 | Insert path — key absent | Returns `true`, value inserted, lambda not called |
| 2 | Keep — key present | Returns `false`, value unchanged |
| 3 | Update — key present | Returns `false`, value mutated, verified via `get()` |
| 4 | Update idempotency | Two upserts with same update lambda → second sees first's result |
| 5 | Erase — key present | Returns `false`, key removed, `get()` returns empty |
| 6 | Mixed operations | Insert N keys, upsert each with conditional update |
| 14 | Root leaf | Single-entry tree, upsert on that key (all 3 actions) |
| 15 | Empty tree | Upsert → takes insert path |
| 16 | After `clear()` | Upsert → takes insert path |

### Type Coverage

| # | Key type | Value type | Storage mode | Notes |
|---|----------|-----------|--------------|-------|
| 11 | `uint64_t` | `uint64_t` | Fixed key, VIS (packed value) | `can_eliminate_leaf = true` |
| 12 | `key_view` | `uint64_t` | Variable key, VIS | `can_eliminate_leaf = true` |
| 13 | `key_view` | `value_view` | Variable key, explicit leaf | `can_eliminate_leaf = false`, update blocked (`CANNOT_HAPPEN` in `if constexpr` dead path) |
| 13b | `uint64_t` | `value_view` | Fixed key, explicit leaf | Same constraint |

For types 13/13b: the `update` action is unreachable for `value_view` types (guarded by
`if constexpr`; hits `CANNOT_HAPPEN()` if somehow reached). Test `keep` and `erase` only.

### Concurrency Tests (olc_db only)

| # | Test | Scenario | Verification |
|---|------|----------|--------------|
| 7 | Upsert + get | N threads upsert overlapping keys, readers do `get()` | No crashes, values consistent |
| 8 | Upsert + upsert (disjoint) | Two threads, non-overlapping key ranges | Both succeed, all keys present |
| 9 | OLC restart coverage | Sync points force write_guard upgrade failure | Operation retries and succeeds |
| 10 | Upsert(update) + remove | One thread updates, another removes same key | Final state consistent (key present OR absent) |
| 17 | **CAS same key (increment)** | N threads: `upsert(K, 0, [](auto& v){ v+=1; return update; })` | Final value == N (no lost updates) |
| 18 | **CAS during node growth** | T1 upserts on I4 node, T2 inserts triggering I4→I16 | T1 restarts after growth, succeeds |
| 19 | **CAS on key being removed** | T1 pauses at duplicate detection, T2 removes K, T1 resumes | T1 takes insert path (key was removed) |
| 20 | **CAS + concurrent scan** | T1 upserts(update), T2 scans | Scan sees old or new value, never torn |
| 21 | **Random ops stress** | Add `upsert` as operation in random_op_thread | No crashes under sustained mixed load |
| 22 | **Idempotency under contention** | Lambda counts invocations (thread-local) | Value reflects exactly N successful updates, lambda called ≥ N times |
| 23 | **Erase CAS retry** | T1 upserts(erase) on K, T2 concurrently updates K | T1 retries until version matches, then erases. Or: T2's update wins and T1's lambda re-invoked on new value, may return different action |

### Erase-Specific Tests

| # | Test | Scenario | Verification |
|---|------|----------|--------------|
| 23a | Erase triggers shrink | Upsert-erase on a key in a min-size inode | Inode shrinks correctly (I16→I4) |
| 23b | Erase triggers chain cut | Upsert-erase on a key under a chain (key_view) | Chain cut executes, tree well-formed |
| 23c | Erase root leaf | Single-entry tree, upsert-erase | Tree becomes empty |
| 23d | Erase VIS value | Packed value in inode slot, upsert-erase | Slot cleared, bitmask updated |
| 23e | Erase + re-insert | Upsert-erase K, then upsert K (insert path) | Key re-inserted with new value |
| 23f | Concurrent erase × erase | Two threads upsert-erase same key | Exactly one succeeds, other retries (finds key absent → insert path). Post: key present with insert value, tree size == 1. No ASAN errors. |
| 23g | Erase after concurrent remove (sync point) | T1 upserts(erase) on K, pauses after lambda returns erase. T2 removes K. T1 resumes. | T1's try_remove finds key absent → version mismatch → restart → UObserveAbsent → takes insert path. Final: key present with T1's insert value. Verified via sync_after_erase_lambda_returns. |

### OOM Tests (GCC debug builds)

| # | Test | Scenario |
|---|------|----------|
| OOM-1 | Upsert insert path OOM | Allocation failure during leaf/inode creation → exception, tree unchanged |
| OOM-2 | Upsert erase shrink OOM | Allocation failure during smaller-inode creation → operation fails gracefully |

---

## Microbenchmark Design

### Goals

1. Measure upsert throughput under varying contention levels
2. Quantify the cost of the erase retry loop (version validation failures)
3. Inform the backoff strategy decision (connection to #635)
4. Compare upsert vs separate `get` + `insert`/`remove` sequences

### Benchmark Scenarios

All benchmarks use the existing `concurrent_benchmark` infrastructure
(`micro_benchmark_concurrency.hpp`) with `concurrency_ranges` (1-32 threads).

#### B1: Upsert Update — Disjoint Keys (Baseline)

Each thread upserts on its own key range. No contention. Measures raw
upsert overhead vs plain insert.

```
Setup: Pre-insert 2M keys (disjoint ranges per thread)
Work:  Each thread: upsert(key_i, val, [](auto& v){ v ^= 0xFF; return update; })
```

**Counters:** ops/sec, items processed

#### B2: Upsert Update — Shared Hot Keys (Contention)

All threads upsert on the SAME small set of keys. Measures OLC restart rate.

```
Setup: Pre-insert 64 hot keys
Work:  Each thread: upsert(hot_keys[i % 64], val, [](auto& v){ v += 1; return update; })
Vary:  Thread count: 1, 2, 4, 8, 16, 32
```

**Counters:** ops/sec, OLC restart count (via STATS build), restarts/op ratio, p50/p95/p99 latency (ns, via thread-local rdtsc sampling)

#### B3: Upsert Erase — Shared Hot Keys (Erase Retry)

Measures the cost of the version-validated erase retry loop.

```
Setup: Pre-insert 64 hot keys
Work:  Half threads: upsert(key, val, [](auto&){ return erase; })
       Half threads: upsert(key, val, [](auto& v){ v += 1; return update; })
       (Erasers and updaters contend — erasers must retry when updaters win)
Post:  Re-insert erased keys each iteration to sustain contention
```

**Counters:** ops/sec, erase retry count, erase success rate, p50/p95/p99 latency (ns, via thread-local rdtsc sampling)

#### B4: Upsert Insert Path — Disjoint (Insert Overhead)

Measures the overhead of the upsert code path vs plain `insert` when
the key is absent (insert path taken).

```
Setup: Empty tree
Work:  Each thread: upsert(unique_key_i, val, lambda_never_called)
Compare: Same workload with plain insert()
```

**Counters:** ops/sec, ratio vs plain insert

#### B5: Upsert Mixed — Random Operations

Realistic mixed workload: 50% update, 30% insert, 15% keep, 5% erase.

```
Setup: Pre-insert 1M keys
Work:  Each thread picks random key, random action weighted as above
```

**Counters:** ops/sec by thread count, action distribution

#### B6: Backoff Strategy Comparison (ref #635)

Same as B2 (hot keys, high contention) but with different backoff strategies:

```
Variants:
  (a) spin_wait_loop_body() — current (_mm_pause / yield)
  (b) No backoff (tight retry)
  (c) Exponential backoff (1, 2, 4, 8 pauses)
  (d) Randomized backoff
```

**Counters:** ops/sec, p99 latency (if measurable), restart count

This benchmark directly informs the #635 decision. Run after implementation
is complete; results determine whether to change the backoff strategy.

---

## Implementation Notes

- Test file: `test/test_art_upsert.cpp` (new file, all db types via typed test)
- Benchmark file: `benchmark/micro_benchmark_olc_upsert.cpp` (new file)
- Both use existing infrastructure — no new test/benchmark framework needed
- Sync points for tests 9, 18, 19: use existing `sync.hpp` / `thread_sync.hpp` mechanism (debug builds only)
- OOM tests: use existing `test_art_oom.cpp` pattern (inject allocation failures)
