# Adversarial Performance Review: CAS (#847) and Bulk Loader (#636)

**Reviewer:** Systems performance / cache optimization perspective
**Date:** 2026-04-25
**Status:** Review

---

## 1. CAS Hot Path Overhead

### 1.1 Template Policy Tag: Code Size Impact

The design proposes templating `try_insert` with a policy tag
(`insert_only_tag` / `insert_or_resolve_tag`) and using `if constexpr` to
select behavior at the duplicate-detection sites.

**Verdict: Acceptable, but watch the instantiation count.**

`try_insert` is already a large function (~250 lines of non-trivial control
flow with multiple OLC lock acquisitions). Each `if constexpr` branch is
eliminated at compile time, so the generated code for `insert_only_tag`
should be identical to today's `try_insert` — zero regression on the insert
path. The `insert_or_resolve_tag` instantiation adds the lambda invocation
and lock-upgrade logic at the two duplicate sites, but this code is on the
cold path for insert-heavy workloads.

**Risk:** `try_insert` is already instantiated for every `<Key, Value>` pair.
Adding the policy tag doubles the instantiation count. For a library with
N key-value combinations, that's 2N copies of a ~250-line function in the
binary. On a typical build with 3–4 instantiations (uint64/uint64,
key_view/uint64, key_view/value_view, etc.), this adds ~3–4 extra copies.
At ~2–4 KB per instantiation (estimated from the function complexity), that's
8–16 KB of additional .text. Not a problem for server workloads, but worth
noting for embedded use cases.

**Recommendation:** Verify with `nm --print-size` or `bloaty` after
implementation. If code size is a concern, consider a runtime bool parameter
with `[[likely]]`/`[[unlikely]]` annotations instead of the template tag —
the branch predictor will learn the pattern quickly since the call site is
monomorphic.

### 1.2 Lambda Inlining

The design passes a user-supplied lambda `FN fn` through the template chain:
`insert_or_resolve` → `try_insert_impl<insert_or_resolve_tag, FN>`.

**Verdict: The lambda WILL inline — but only if the call chain is shallow
enough for the inliner.**

The lambda is invoked inside `try_insert_impl`, which is a large function.
Modern compilers (GCC 12+, Clang 17+) will inline small lambdas into their
call site regardless of the enclosing function's size, because the lambda's
`operator()` is a tiny function and the call is monomorphic (the type is
known at compile time). The concern is not the lambda itself but the
surrounding code: if `try_insert_impl` is too large for the compiler to
inline into the retry loop, the lambda invocation adds one extra call frame.

**Recommendation:** Check the generated assembly for the
`insert_or_resolve<uint64_t, uint64_t>` instantiation. The lambda body
should appear inline at the duplicate-detection site with no `call`
instruction. If not, mark the lambda invocation site with
`UNODB_DETAIL_FORCE_INLINE` or restructure the duplicate-handling into a
separate `[[gnu::always_inline]]` helper.

### 1.3 Unnecessary Copying: Unpack → Lambda → Repack for VIS

The VIS path in the design does:

```cpp
auto existing_value = art_policy::unpack_value(child_in_parent->load());
const auto action = fn(existing_value);
if (action == upsert_action::update) {
    *child_in_parent = art_policy::pack_value(existing_value);
}
```

This is three `memcpy` operations:
1. `unpack_value`: `node_ptr` → XOR → `memcpy` into `Value v`
2. Lambda modifies `v` in place
3. `pack_value`: `memcpy` from `v` → XOR → `memcpy` into `node_ptr`

For `Value = uint64_t`, each `memcpy` is 8 bytes — the compiler will
optimize these to register moves. The XOR is a single instruction. **Total
overhead: ~3 instructions for the round-trip.** This is negligible.

**However**, the design passes `existing_value` by mutable reference to the
lambda, then reads it back after the lambda returns. This means the compiler
cannot keep `existing_value` in a register across the lambda call — it must
be spilled to the stack so the lambda can take its address. For a `uint64_t`
value, this is one extra store + load pair (~2 cycles on modern x86).

**Recommendation:** This is fine. The alternative (passing by value and
returning the new value) changes the lambda signature and is less ergonomic.
The 2-cycle overhead is dwarfed by the lock-upgrade cost (~10–50 cycles for
the atomic CAS in `write_guard`).

### 1.4 The Keyed-Leaf Path: Same Analysis

```cpp
auto existing_value = leaf->template get_value<Value>();
const auto action = fn(existing_value);
if (action == upsert_action::update) {
    leaf->template set_value<Value>(existing_value);
}
```

`get_value` does a `memcpy` from `data + key_size` into a local. `set_value`
does a `memcpy` back. Same analysis as VIS: ~3 instructions, dominated by
the lock-upgrade cost. **No concern.**

---

## 2. Branch Prediction

### 2.1 The Duplicate-Key Path Is Cold for Insert, Hot for CAS

The current code marks the duplicate-detection branch as `UNODB_DETAIL_UNLIKELY`:

```cpp
if (UNODB_DETAIL_UNLIKELY(k.cmp(existing_key) == 0)) {
    return false;  // duplicate
}
```

For plain `insert()`, this is correct — duplicates are rare. But for
`insert_or_resolve()`, the **entire point** is to hit the duplicate path.
In a CAS-heavy workload (e.g., "insert if absent, update if present"), the
duplicate path is taken on the majority of calls.

**The design does not address this.** The `UNODB_DETAIL_UNLIKELY` hint will
cause the branch predictor to mispredict on every CAS call that hits an
existing key. On modern x86, a branch misprediction costs ~15–20 cycles.
For a tight CAS loop, this adds ~15–20 cycles per operation.

**Recommendation:** The template policy tag provides a natural fix:

```cpp
if constexpr (std::is_same_v<InsertPolicy, insert_or_resolve_tag>) {
    if (k.cmp(existing_key) == 0) {  // no UNLIKELY hint
        // invoke lambda
    }
} else {
    if (UNODB_DETAIL_UNLIKELY(k.cmp(existing_key) == 0)) {
        return false;
    }
}
```

This preserves the `UNLIKELY` hint for plain insert (where duplicates are
rare) and removes it for CAS (where duplicates are expected). The `if
constexpr` ensures zero cost — the wrong branch is eliminated at compile
time.

**Severity: Medium.** 15–20 cycles per CAS operation is measurable in
microbenchmarks but may be masked by the lock-upgrade cost in OLC. Still,
it's a free fix — do it.

### 2.2 VIS Duplicate Path: Same Issue

```cpp
if (inode->is_value_in_slot(node_type, ci_chk)) {
    return false;  // duplicate
}
```

The current code doesn't have an explicit `UNLIKELY` here, but the compiler
may still lay out the code with the fall-through (non-duplicate) path as the
hot path. For CAS, the `is_value_in_slot` check is the hot path.

**Recommendation:** Same fix — use the policy tag to control branch layout.

### 2.3 The `check()` Calls After Value Read

The design adds validation checks between the value read and the lambda:

```cpp
auto existing_value = leaf->template get_value<Value>();
if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) return {};
if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check())) return {};
const auto action = fn(existing_value);
```

These `check()` calls are necessary for correctness (the value could have
been modified between the read and the lambda invocation). But they add two
atomic loads (version tag reads) on the hot path. Each atomic load is ~1–5
cycles depending on cache state.

**Verdict: Necessary and correct.** The alternative (checking only after the
lambda) would allow the lambda to see a torn value, which is worse. The
2–10 cycle overhead is acceptable.

---

## 3. Bulk Loader: Phase 2 Bottom-Up Complexity Claims

### 3.1 "O(N) vs O(N × key_length)" — Is This Accurate?

The design claims bottom-up construction eliminates root-to-leaf traversals,
reducing complexity from O(N × key_length) to O(N).

**This is misleading.** The bottom-up builder still processes every byte of
every key:

- `common_prefix(keys, depth)` scans all keys in the group to find the
  shared prefix. For N keys with average key length L, this is O(N × L)
  total work across all recursive calls.
- `group_by_byte(keys, depth + prefix_length)` examines one byte per key
  per level. Across all levels, this is O(N × L / (avg_prefix_length + 1)).

The correct complexity is **O(N × L)** for both approaches. What changes is
the **constant factor**:

| Operation | Repeated Insert | Bottom-Up |
|-----------|----------------|-----------|
| Key byte comparisons | ~2× per byte (prefix check + dispatch) | ~1× per byte |
| Pointer chasing | O(L) cache misses per insert | Sequential scan |
| Inode allocation | O(N) temporary + O(N) final | O(N) final only |
| Lock overhead (OLC) | O(N × L) atomic ops | Zero (exclusive access) |

The real win is **cache behavior** (sequential scan vs pointer chasing) and
**allocation reduction** (no temporary inodes), not asymptotic complexity.

**Recommendation:** Fix the complexity claim in the design doc. The speedup
is real but comes from constant-factor improvements, not asymptotic ones.

### 3.2 Memory Allocation Pattern: Are We Thrashing the Allocator?

The bottom-up builder allocates nodes in a depth-first recursive pattern:

```
build_subtree(keys[0..255], depth=0)
  build_subtree(keys[0..3], depth=1)     → allocate leaf, leaf, leaf, leaf
                                          → allocate inode4
  build_subtree(keys[4..19], depth=1)    → allocate 16 leaves
                                          → allocate inode16
  ...
  → allocate inode256 (root)
```

This allocation pattern is **bottom-up**: leaves first, then their parent
inode, then the grandparent, etc. The allocator sees a stream of
small-then-large allocations.

**Concern: Heap fragmentation.** The default allocator (`allocate_aligned`
from `heap.hpp`) delegates to `std::aligned_alloc` / `_aligned_malloc`.
For a bulk load of 10M keys:
- ~10M leaf allocations (variable size, typically 16–64 bytes each)
- ~40K–400K inode allocations (64–2048 bytes each)

This is 10M+ allocations in rapid succession. The system allocator (glibc
malloc, jemalloc, tcmalloc) handles this fine — they're designed for
allocation-heavy workloads. But the allocation pattern is **not sequential
in address space** — leaves and inodes are interleaved, so the resulting
tree has poor spatial locality.

**Contrast with repeated sorted insert:** The sorted-insert approach
allocates nodes in traversal order (root → leaf), which means parent inodes
are allocated before their children. This gives slightly better spatial
locality for top-down traversals (the common access pattern).

**Recommendation:** For Phase 2, consider a **bump allocator** (arena) for
the bulk load. Allocate a single large block, bump-allocate nodes from it.
This gives:
1. O(1) allocation cost (no malloc overhead)
2. Perfect spatial locality (nodes are contiguous in memory)
3. Easy cleanup on failure (free the entire arena)

The tree's pluggable `allocator_type` already supports this — the caller
can supply a bump allocator via the `alloc` callback. Document this as a
recommended pattern for bulk load.

### 3.3 The `common_prefix` Scan Is Redundant for Sorted Input

The bottom-up builder calls `common_prefix(keys, depth)` to find the shared
prefix of all keys in a group. For sorted input, the shared prefix is
determined by the **first and last** keys in the group — all intermediate
keys share at least that prefix (by the sorted order invariant).

The design's pseudocode implies scanning all keys:

```
prefix = common_prefix(keys, depth)
```

**Recommendation:** Optimize to `common_prefix(keys[0], keys[N-1], depth)`.
This reduces the prefix computation from O(N × prefix_length) to
O(prefix_length) per group. For groups with long shared prefixes (common
with key_view keys), this is a significant constant-factor improvement.

---

## 4. Bulk Loader Phase 3: Parallelism Analysis

### 4.1 Expected Speedup

The design claims 10–30× speedup for Phase 3 (parallel bottom-up). Let's
check this.

**Amdahl's Law analysis:**

The parallel phase is the subtree construction. The serial phases are:
1. Sorting (if not pre-sorted): O(N log N) — dominates for large N
2. Partitioning by first byte: O(N) — single pass
3. Assembling the root: O(256) — trivial

If the input is pre-sorted (the common case per the API contract), the
serial fraction is partitioning + assembly ≈ O(N) + O(256). The parallel
fraction is subtree construction ≈ O(N × L / P) where P is the thread
count.

For P = 16 threads and L = 8 (uint64 keys):
- Serial: O(N) for partitioning
- Parallel: O(N × 8 / 16) = O(N/2)
- Speedup ≈ N / (N + N/2) ≈ 1.5× ... wait, that's wrong.

The issue is that partitioning is O(N) and construction is also O(N × L).
For L = 8, construction is 8N work. With P = 16 threads, parallel
construction is 8N/16 = N/2. Total = N (partition) + N/2 (parallel build)
= 1.5N. Sequential would be N + 8N = 9N. Speedup = 9N / 1.5N = **6×**.

For L = 32 (key_view with 32-byte keys) and P = 16:
- Sequential: N + 32N = 33N
- Parallel: N + 32N/16 = N + 2N = 3N
- Speedup = 33N / 3N = **11×**

**The 10–30× claim is optimistic for uint64 keys but plausible for long
key_view keys with high thread counts.** For uint64 keys with 8–16 threads,
expect 4–8×.

**Recommendation:** Temper the speedup claims. Provide separate estimates
for uint64 (short keys) and key_view (long keys). The partitioning overhead
is non-trivial and limits speedup for short keys.

### 4.2 Partitioning Overhead

Partitioning by first byte requires a single pass over the sorted array to
find bucket boundaries. For pre-sorted input, this is a binary search for
each of the 256 possible first bytes, or a single linear scan with 256
boundary markers.

**The linear scan is O(N) with excellent cache behavior** (sequential read
of the key array). The binary search approach is O(256 × log N) which is
faster for large N but has worse cache behavior (random access into the key
array).

**Recommendation:** Use the linear scan. For 10M keys, the scan touches
each key once — this is ~80 MB for uint64 keys, which takes ~20 ms at
memory bandwidth. The binary search saves time but the cache misses negate
the benefit.

### 4.3 NUMA Effects

The design does not mention NUMA. For multi-socket systems:

- **Partitioning** allocates the key array on one NUMA node. Threads on
  other nodes will access it via remote memory (2–3× latency penalty).
- **Subtree construction** allocates nodes via the tree's allocator, which
  uses the system allocator. On Linux with the default first-touch policy,
  nodes are allocated on the NUMA node of the thread that first touches
  them. This means each subtree's nodes are local to the building thread.
- **After construction**, queries traverse from the root (on one NUMA node)
  into subtrees (on various NUMA nodes). The first few levels of traversal
  are fast (root is hot in cache), but deeper traversals may cross NUMA
  boundaries.

**Severity: Low for most deployments** (single-socket is the common case
for in-memory databases). For multi-socket, the NUMA-aware fix is to use
`mbind()` or `numa_alloc_onnode()` to place the root and first-level inodes
on a shared/interleaved NUMA policy. This is out of scope for Phase 3 but
should be documented as a known limitation.

### 4.4 Load Imbalance

Partitioning by first byte creates up to 256 buckets, but the distribution
is rarely uniform:
- For uint64 keys in [0, N), the first byte is the MSB of the big-endian
  encoding. For N < 2^56, only one bucket is non-empty.
- For random uint64 keys, all 256 buckets are roughly equal.
- For string keys (key_view), the distribution depends on the key
  distribution. ASCII keys cluster in bytes 0x20–0x7E.

**For sequential uint64 keys, Phase 3 parallelism is useless** — all keys
land in one bucket. The design should detect this and fall back to Phase 2
(or partition at a deeper level).

**Recommendation:** Add a heuristic: if the largest bucket contains >50% of
keys, re-partition at depth 1 (second byte) within that bucket. This adds
one level of recursion to the partitioning but ensures reasonable load
balance for skewed distributions.

---

## 5. Memory Layout: `set_value<Value>` and Cache Line Bouncing

### 5.1 The Problem

In OLC, leaves can be read-shared across threads. Multiple readers can
concurrently call `get_value<Value>()` on the same leaf, which reads from
`data + key_size`. This is fine — concurrent reads don't conflict.

`set_value<Value>()` writes to `data + key_size` via `memcpy`. In the CAS
design, this happens under a `write_guard` (the leaf's optimistic lock is
held in write mode). The write guard prevents concurrent readers from
validating their read (they'll see the version bump and restart).

**But the cache line containing the value is shared across all readers.**
When the writer modifies the value, the cache line is invalidated on all
other cores (MESI protocol: Shared → Invalid). Any concurrent reader that
has the leaf's cache line in its L1/L2 will take a cache miss on the next
access.

**Severity: Low.** This is the standard cost of any write in a
read-mostly concurrent data structure. The OLC protocol already causes a
version-tag write on every mutation (the `write_guard` constructor bumps
the version), which invalidates the lock's cache line. The value write is
on the same or adjacent cache line, so it doesn't add a new invalidation —
the cache line was already being invalidated by the lock.

**However**, there's a subtlety: the leaf's optimistic lock (`olc_node_header::m_lock`) is at offset 0 of the leaf. The value is at offset `sizeof(Header) + key_size`. For small keys (e.g., 8-byte uint64), the lock and value are on the **same 64-byte cache line**. For larger keys (e.g., 32-byte key_view), they may be on **different cache lines**. In the latter case, `set_value` causes an **additional** cache line invalidation beyond what the lock write already causes.

**Recommendation:** For Phase 1, this is acceptable. If profiling shows
cache line bouncing as a bottleneck (unlikely for typical workloads), the
fix is to ensure the lock and value are on the same cache line by padding
the leaf header. But this increases leaf size, which hurts cache utilization
for read-heavy workloads. **Don't optimize this without profiling data.**

### 5.2 VIS Path: No Cache Line Bouncing Concern

For the VIS path, the value is packed into the inode's `children[]` array.
The `write_guard` is on the inode's lock (same cache line as the children
array for small inodes). Writing the packed value doesn't cause additional
invalidations beyond the lock write. **No concern.**

---

## 6. Benchmark Methodology

### 6.1 CAS Benchmarks (Required)

The CAS design must be validated with these benchmarks:

**B1: CAS insert-only (no duplicates)**
- Insert N unique keys via `insert_or_resolve` with a no-op lambda.
- Compare against plain `insert()`.
- **Expected:** ≤2% overhead (the `if constexpr` branch is eliminated; the
  only cost is the extra template instantiation's icache pressure).
- **Purpose:** Regression test — CAS must not slow down the insert path.

**B2: CAS update-only (all duplicates)**
- Pre-populate tree with N keys. Then call `insert_or_resolve` on all N
  keys with `action::update`.
- Measure throughput (ops/sec) and compare against `get()` + `remove()` +
  `insert()` (the current workaround).
- **Expected:** 3–10× faster than the workaround (one traversal vs three).

**B3: CAS mixed (50% insert, 50% update)**
- Zipfian key distribution: 50% of calls hit existing keys, 50% are new.
- Measure throughput and compare against plain insert + separate update.

**B4: CAS contention (OLC)**
- N threads doing `insert_or_resolve` on overlapping key ranges.
- Measure throughput scaling with thread count (1, 2, 4, 8, 16).
- **Purpose:** Verify that the lock-upgrade path doesn't serialize under
  contention.

**B5: CAS VIS vs keyed-leaf**
- Compare CAS performance for `<key_view, uint64_t>` (VIS path) vs
  `<uint64_t, uint64_t>` (keyed-leaf path).
- **Purpose:** Verify the VIS unpack→lambda→repack overhead is negligible.

### 6.2 Bulk Loader Benchmarks (Required)

**B6: Phase 1 vs random insert**
- Insert 1M, 10M, 100M keys via `bulk_load` (sorted) vs `insert` (random).
- Measure wall time, cache misses (perf stat), and allocator calls.
- **Expected:** 2–5× speedup for Phase 1.

**B7: Phase 2 vs Phase 1**
- Same key counts, compare bottom-up vs sorted-insert.
- Measure wall time, peak memory (bottom-up should use ~50% less peak
  memory due to no temporary inodes).

**B8: Phase 3 scaling**
- 10M keys, vary thread count (1, 2, 4, 8, 16, 32).
- Plot throughput vs threads. Identify the saturation point.
- **Purpose:** Validate the parallelism claims and identify NUMA/contention
  bottlenecks.

**B9: Key distribution sensitivity**
- Compare bulk load performance for:
  - Sequential uint64 keys (worst case for Phase 3 partitioning)
  - Random uint64 keys (best case for Phase 3 partitioning)
  - Skewed key_view keys (ASCII strings with common prefixes)
- **Purpose:** Validate the load-balance heuristic.

**B10: Post-load query performance**
- After bulk load, measure `get()` throughput for random lookups.
- Compare trees built by Phase 1 vs Phase 2 vs Phase 3.
- **Purpose:** Verify that the bottom-up builder produces trees with
  equivalent or better query performance (spatial locality of nodes).

### 6.3 Instrumentation

All benchmarks should report:
- Wall time (Google Benchmark)
- `perf stat` counters: instructions, cycles, L1-dcache-load-misses,
  LLC-load-misses, branch-misses
- Allocator statistics: total allocations, total bytes, peak RSS
- For OLC: restart count (how many OLC retries occurred)

---

## 7. Summary of Findings

| Finding | Severity | Action |
|---------|----------|--------|
| Template tag doubles `try_insert` instantiations | Low | Monitor with `bloaty`; acceptable for now |
| Lambda will inline for small bodies | Info | Verify with `objdump` after implementation |
| VIS unpack→lambda→repack: ~3 extra instructions | Negligible | No action needed |
| `UNODB_DETAIL_UNLIKELY` on duplicate path hurts CAS | **Medium** | Remove hint for `insert_or_resolve_tag` instantiation |
| Bulk loader O(N) claim is misleading | Low | Fix doc: it's O(N × L) with better constants |
| Bottom-up allocator thrashing | Low | Document bump-allocator pattern; defer arena to Phase 3 |
| `common_prefix` scans all keys unnecessarily | **Medium** | Optimize to first/last comparison for sorted input |
| Phase 3 speedup claims are optimistic for short keys | Low | Temper claims; provide per-key-type estimates |
| Phase 3 load imbalance for sequential uint64 | **Medium** | Add re-partitioning heuristic for skewed buckets |
| NUMA effects undocumented | Low | Document as known limitation |
| `set_value` cache line bouncing in OLC | Low | No action; lock write already invalidates the line |
| Missing benchmark plan | **High** | Implement B1–B10 before merging either design |
