# Adversarial Concurrency Review: CAS/Upsert (#847) and Bulk Loader (#636)

**Reviewer:** Kiro (concurrency & lock-free data structures)
**Date:** 2026-04-25
**Scope:** Lock protocol correctness, linearizability, progress guarantees, memory ordering
**Sources reviewed:**
- `design/cas-insert-or-resolve-847.md` (from branch `design/cas-847`)
- `design/bulk-load-636.md`
- `olc_art.hpp` — `try_insert`, `add_or_choose_subtree`, lock handover protocol
- `optimistic_lock.hpp` — `read_critical_section`, `write_guard`, `try_upgrade_to_write_lock`, version word layout

---

## Part 1: CAS / `insert_or_resolve` (#847)

### 1.1 CRITICAL — Lambda Observes Stale Data on Upgrade Failure (ABA Window)

**Location:** §4.5.2, keyed-leaf path, and §4.5.3, VIS path.

The design proposes:

```
1. Read existing value into local copy (under RCS)
2. check() the RCS
3. Invoke lambda on the local copy
4. Upgrade RCS → write_guard
5. If upgrade fails → return {} (restart)
6. Write back the lambda's result
```

**The bug:** Between step 2 (`check()`) and step 4 (`write_guard` upgrade), another writer can:
- Acquire the write lock
- Modify the value (or even delete and re-insert the key)
- Release the write lock

The version counter advances by +2 on each write-unlock (`write_unlock` stores `old + 2`). The upgrade CAS at step 4 will correctly *fail* because the version no longer matches. So the write-back is prevented — **good**.

**However**, the lambda at step 3 has already executed on stale data. If the lambda has **side effects** (logging, incrementing an external counter, sending a message, updating a secondary index), those side effects are **not rolled back** on restart. The design document does not mention this.

**Severity:** Medium-High. The design says `FN = upsert_action(value_type& existing)` — if the lambda is pure (only mutates `existing` and returns an action), the side effects are harmless because the local copy is discarded on restart. But the API contract does not enforce purity. A user who writes:

```cpp
tree.insert_or_resolve(key, val, [&](auto& existing) {
    audit_log.append("saw value: ", existing);  // side effect on stale data
    existing += delta;
    return upsert_action::update;
});
```

will get spurious audit log entries with stale values.

**Recommendation:** Document explicitly that the lambda may be called multiple times with stale values on OLC restart. Alternatively, restructure to call the lambda *after* the write_guard is acquired (read the value under write lock). This is what libcuckoo does — the lambda runs under the bucket lock.

### 1.2 CRITICAL — Missing Validation After `find_child` in VIS Path

**Location:** §4.5.3, VIS path.

The design proposes reading the packed value via `art_policy::unpack_value(child_in_parent->load())` and then calling `check()` on `node_critical_section`. But look at the actual `try_insert` code flow:

```cpp
const auto add_result{inode->template add_or_choose_subtree<...>(...)};
if (!add_result) return {};
auto* const child_in_parent = *add_result;
if (child_in_parent == nullptr) return true;  // inserted

// VIS check:
const auto [ci_chk, _] = inode->find_child(node_type, remaining_key[0]);
if (inode->is_value_in_slot(node_type, ci_chk)) {
    // DUPLICATE — this is where insert_or_resolve would invoke the lambda
```

The `find_child` and `is_value_in_slot` calls happen **after** `add_or_choose_subtree` returns, which may have already released `parent_critical_section` (in the non-full, non-chain path, `add_or_choose_subtree` calls `parent_critical_section.try_read_unlock()`). So at the VIS duplicate detection site, we may only hold `node_critical_section`.

The design's §4.5.3 pseudocode shows:
```cpp
if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check())) return {};
const auto action = fn(existing_value);
```

But it does **not** re-validate after `unpack_value`. The `child_in_parent->load()` is a `std::memory_order_relaxed` load (from `in_critical_section::load()`). Between the `add_or_choose_subtree` return and the `unpack_value` call, the node could have been modified by a concurrent writer. The `check()` call after `unpack_value` catches this — **but only if the check is placed correctly**.

**The actual risk:** The design shows `check()` before `fn()`, which is correct for detecting staleness. But the `unpack_value` itself could read a partially-updated value if the relaxed load races with a concurrent write. The OLC protocol relies on the `check()` *after* all reads to validate consistency. The design does place the check after the read, so this is **correct but fragile** — any reordering of the check before the read would break it.

**Recommendation:** Add a comment in the implementation noting that the `check()` must come strictly after all reads from the node, per the OLC read protocol (Boehm 2012 seqlock rules). Consider using `std::atomic_thread_fence(std::memory_order_acquire)` before the check, as the existing `check()` implementation does.

### 1.3 ABA Problem — Value Changes Between Read and Write-Guard

**Location:** Both keyed-leaf and VIS paths.

The OLC version counter is 62 bits wide (bits 2–63 of the lock word). Each write-unlock increments by 2 (clearing the lock bit and advancing the version). A true ABA wrap-around requires 2^62 write operations on the same node — **not a practical concern**. The version space is large enough that ABA is not a real risk here.

**Verdict:** No ABA problem. The version counter is effectively monotonic.

### 1.4 Linearizability Analysis

**Question:** Is `insert_or_resolve` linearizable?

**For the insert path (key absent):** The linearization point is the same as `try_insert` — the atomic store to `*node_in_parent` under the write_guard. This is correct.

**For the `keep` path (key present, no mutation):** The linearization point is the successful `check()` / `try_read_unlock()` that validates the read. The operation observes a consistent snapshot. This is linearizable.

**For the `update` path (key present, value mutated):** The proposed linearization point is the `set_value` / `pack_value` store under the write_guard. But there's a subtlety:

1. The value is read under RCS (optimistic read).
2. The lambda transforms the value.
3. The write_guard is acquired.
4. The transformed value is written back.

Between steps 1 and 3, the value could have been changed by another `insert_or_resolve(update)`. The write_guard upgrade will fail in that case (version mismatch), causing a restart. On restart, the value is re-read and the lambda re-invoked. So the final write is always based on a value that was current at the time the write_guard was acquired.

**But wait** — the write_guard upgrade only checks that the *version* hasn't changed. It does not re-read the value. The value written back is the lambda's transformation of the value read at step 1. If the version hasn't changed (no concurrent writer touched this node), the value is still current. If the version has changed, the upgrade fails and we restart.

**This is linearizable.** The linearization point for `update` is the successful `try_upgrade_to_write_lock` CAS. At that instant, the version matches, meaning no writer has intervened since the read, so the read value is still current, and the write_guard prevents any concurrent writer until the unlock.

**However**, this is a **read-modify-write** that is NOT atomic with respect to other `insert_or_resolve(update)` operations on the same key if they happen to read the same version. Two threads could:
1. Both read version V, value X
2. Both compute lambda(X) → Y₁, Y₂
3. Only one succeeds the CAS upgrade (the other restarts)

This is correct — the CAS serializes them. But it means `insert_or_resolve` is **not wait-free** for the update path; it's lock-free (the loser restarts).

### 1.5 Progress Guarantees — Livelock Risk

**Location:** `insert_internal` retry loop:

```cpp
while (true) {
    result = try_insert(insert_key, v, cached_leaf);
    if (result) break;
}
```

This is an unbounded retry loop with no backoff. Under high contention on the same node, multiple threads can repeatedly fail their write_guard upgrades and restart indefinitely.

**Is this a livelock?** Strictly, no — the OLC protocol guarantees that at least one writer succeeds per version increment (the CAS is strong, not spurious). So the system makes progress globally. But individual threads can starve.

**For `insert_or_resolve`:** The contention window is wider than for plain `insert` because the lambda execution time is added between the read and the upgrade attempt. A slow lambda increases the probability of version mismatch at upgrade time.

**Recommendation:** Consider exponential backoff in the retry loop, or at minimum document that lambda execution time directly impacts contention. The existing `spin_wait_loop_body()` (a single `_mm_pause`) is insufficient for high-contention scenarios.

### 1.6 The `erase` Deferral Is Correct but Creates a Composability Gap

**Location:** §4.5.4.

The design defers `erase` to Phase 2 because the lock state at the duplicate-detection point (two read locks) doesn't match what `remove_or_choose_subtree` expects. This is correct — the remove path needs write locks on parent, node, and potentially grandparent for shrinking.

**But:** The Phase 1 workaround ("caller does a separate `remove()` call") is **not linearizable** as a compound operation. Between `insert_or_resolve` returning `false` (key exists, lambda said erase) and the caller's `remove()`, another thread can:
- Insert a new value for the same key
- Read the value that was supposed to be erased

This is a known limitation, but it should be documented as a **linearizability gap** in Phase 1.

### 1.7 Lambda Exception Safety — Subtle Correctness

**Location:** §8, Open Question 2.

The design says: "the value is unmodified (lambda operates on a local copy; write-back only happens after lambda returns successfully)."

This is correct for the `update` path — the lambda mutates a stack-local copy, and `set_value` is only called after the lambda returns. If the lambda throws, the local copy is destroyed, and the tree value is untouched.

**But:** If the lambda throws *after* the write_guard has been acquired (which it can't in the proposed design — the lambda runs before the upgrade), this would be fine. In the proposed design, the lambda runs before the upgrade, so an exception simply causes the function to unwind without modifying the tree. **Correct.**

**Edge case:** If the lambda throws, the retry loop in `insert_internal` will re-invoke `try_insert`, which will re-read the value and re-invoke the lambda. If the lambda always throws for this value, the retry loop becomes infinite. This should be documented.

### 1.8 Template Deduplication Strategy — Correctness Risk

**Location:** §4.7, tag-dispatch approach.

The proposal to use `if constexpr` with `insert_or_resolve_tag` inside `try_insert_impl` is sound, but the `try_insert` function is already ~200 lines of intricate lock protocol code. Adding conditional branches at the duplicate-detection sites increases the risk of accidentally breaking the lock protocol for the plain `insert` path during maintenance.

**Recommendation:** Consider keeping `try_insert` and `try_insert_or_resolve` as separate functions that share helper functions for the traversal, rather than merging them with `if constexpr`. The code duplication cost is lower than the maintenance risk.

---

## Part 2: Bulk Loader (#636)

### 2.1 Phase 3 "Zero Synchronization" Claim — Partially True

**Location:** §3.4.

The design claims: "Each of the up to 256 subtrees is built independently with zero shared state. The only synchronization point is assembling the root inode after all subtrees complete."

**Analysis of the claim:**

For `db` (single-threaded): The parallel build requires a thread-safe allocator or per-thread arenas. The design acknowledges this in §3.5: "the parallel build must use a thread-safe allocator or build with per-thread arenas that are merged." So the "zero synchronization" claim is **false for `db`** — the allocator is shared state.

For `olc_db`: The design says "the existing QSBR allocator is already thread-safe." Let me check — QSBR is for deferred reclamation, not allocation. The allocator itself (`art_allocator.hpp`) uses the tree's `allocator_type`. Whether this is thread-safe depends on the allocator. The default allocator (`std::allocator`) is thread-safe for allocation (the C++ standard guarantees this), but the **stats counters** are shared state:

```cpp
++node_counts[as_i<T>];
current_memory_use += sizeof(T);
++growing_inode_counts[internal_as_i<T>];
```

These are `std::atomic` in `olc_db` but plain integers in `db`. The design acknowledges this in §3.5: "accumulate per-thread stats and merge after join." So the "zero synchronization" claim should be qualified: **zero synchronization on tree structure**, but stats and allocation require coordination.

### 2.2 Root Assembly Step — Missing Atomicity Analysis

**Location:** §3.4, root assembly.

```
root = allocate_inode(count_non_null(subtrees))
for byte in 0..255:
    if subtrees[byte] != null:
        root.add_child(byte, subtrees[byte])
```

For `olc_db`, the root must be published atomically. The design says: "the tree is published atomically by setting the root pointer." But the root inode is being populated *before* it's published. If a concurrent reader somehow obtains a pointer to the root before it's fully populated (e.g., through a stale pointer from a previous tree), they could see a partially-constructed inode.

**Is this possible?** The design requires an empty tree and exclusive access during the build. If these preconditions hold, there are no concurrent readers, so the root assembly is safe. **But the preconditions are not enforced at runtime in release builds** (only debug asserts). A caller who violates the precondition gets undefined behavior with no diagnostic.

**Recommendation:** Add a runtime check (not just assert) that the tree is empty, or use a flag/mutex to prevent concurrent access during bulk load.

### 2.3 Phase 2 Bottom-Up Construction — Stack Depth Concern

**Location:** §3.3, Open Question 4.

The design calculates: "recursion depth ≤ max_key_length / (key_prefix_capacity + 1) ≈ max_key_length / 8. For 256-byte keys, that's ~32 levels — safe."

**But:** For `key_view` keys, the maximum key length is `std::numeric_limits<key_size_type>::max()` which is `UINT32_MAX` (4 GB). The recursion depth would be ~536 million levels. This is a stack overflow.

**In practice:** Keys this long are unlikely, but the API accepts them. The design should either:
1. Cap the recursion depth and switch to iterative for deep trees, or
2. Document a maximum key length for bulk load, or
3. Use iterative construction with an explicit stack from the start.

### 2.4 Phase 1 Sorted Insert — OLC Lock Contention

**Location:** §3.2, §3.5.

Phase 1 for `olc_db` calls `insert()` in a loop. Each insert acquires write locks as usual. The design says "No change needed."

**Concern:** If `olc_db::bulk_load` is called while other threads are reading the tree (which is allowed — the precondition is only that the tree starts empty, not that no readers exist), the sorted insertion pattern creates a **hot path** on the rightmost frontier. Every insert touches the same rightmost inode chain, creating sustained contention on those nodes' optimistic locks.

This is worse than random insertion for OLC because random insertion spreads contention across the tree, while sorted insertion concentrates it.

**Recommendation:** Document that Phase 1 `olc_db::bulk_load` should be called without concurrent readers for best performance, or consider acquiring a tree-wide write lock for the duration.

### 2.5 Memory Ordering in Bottom-Up Construction

**Location:** §3.3, Phase 2.

The bottom-up builder allocates inodes and wires children. For `db`, no ordering concerns. For `olc_db`, the design says "built without locks (no concurrent readers during construction)."

**But:** The root publication step — `root.store(new_root)` — must use `std::memory_order_release` to ensure all the child wiring is visible to subsequent readers. The existing `root` is an `in_critical_section<olc_node_ptr>`, which uses `std::memory_order_relaxed` for stores.

**This is a bug.** If the root is stored with relaxed ordering, a reader on another core could see the new root pointer but not the fully-constructed subtree (on weakly-ordered architectures like ARM). The reader would follow the root pointer into partially-visible memory.

**Mitigation:** The first reader will acquire an RCS on the root, which calls `try_read_lock()` → `load_acquire()`. The acquire on the reader side pairs with... what? The root store was relaxed. There's no release on the writer side to pair with.

**Wait** — let me re-examine. The `in_critical_section::store()` uses `memory_order_relaxed`. But in the normal `try_insert` path, the root is written under a `write_guard`, and the `write_guard` destructor calls `write_unlock()`, which does `version.store(new_version, memory_order_release)`. This release on the *lock's version word* is what the reader's `load_acquire` on the *lock's version word* pairs with.

For the bulk loader's root publication, there is no write_guard (the tree was built without locks). So the release fence is missing.

**Fix:** The bulk loader must either:
1. Acquire the root_pointer_lock write_guard before storing the root (which provides the release fence), or
2. Use `std::atomic_thread_fence(std::memory_order_release)` before the root store, or
3. Use `root.store(new_root, std::memory_order_release)` — but `in_critical_section` doesn't expose this.

**Severity:** High on ARM/POWER. No-op on x86 (TSO makes relaxed stores visible in order). But the code must be correct on all architectures.

### 2.6 QSBR Epoch Management During Parallel Build

**Location:** §3.4, Phase 3.

The design says: "For `olc_db`, the parallel build phase uses thread-local allocators (no QSBR needed during construction since no concurrent readers exist yet)."

**Concern:** QSBR requires every thread that accesses the tree to register with the QSBR system and periodically announce quiescent states. If the parallel build threads allocate nodes through the tree's allocator but don't register with QSBR, and then a subsequent `remove()` defers deallocation of those nodes via QSBR, the deferred deallocation will wait for all registered threads to pass through a quiescent state. If the build threads have exited without deregistering, QSBR may hang.

**This depends on the QSBR implementation.** If threads auto-deregister on exit, this is fine. If not, it's a resource leak or deadlock.

**Recommendation:** Ensure build threads either (a) don't register with QSBR at all (if they only allocate, never defer-deallocate), or (b) properly deregister after the build completes.

---

## Part 3: Cross-Cutting Concerns

### 3.1 `in_critical_section` Uses `memory_order_relaxed` Everywhere

The `in_critical_section<T>` wrapper uses `memory_order_relaxed` for both `load()` and `store()`. This is correct **only** when all accesses are protected by the OLC protocol (the version check provides the necessary ordering). But it means any code path that accesses `in_critical_section` data **without** a subsequent version check has no ordering guarantees.

The bulk loader's root publication (§2.5 above) is one such path. Any future code that reads/writes `in_critical_section` data outside the OLC protocol must add explicit fences.

### 3.2 The `check()` Method's Fence

```cpp
bool check(version_type locked_version) const noexcept {
#ifndef UNODB_DETAIL_THREAD_SANITIZER
    std::atomic_thread_fence(std::memory_order_acquire);
#endif
    const auto result{locked_version == version.load_relaxed()};
```

The acquire fence before the relaxed load is the Boehm seqlock pattern. This is correct — the fence orders all preceding reads (the data reads) before the version check. But note:

1. The fence is **disabled under ThreadSanitizer** (`#ifndef UNODB_DETAIL_THREAD_SANITIZER`). This means TSan may not detect ordering bugs that the fence would mask. This is intentional (TSan doesn't understand standalone fences well), but it means TSan testing has a blind spot here.

2. The `load_relaxed()` after the fence is correct on x86 (where all loads are acquire) but relies on the fence for correctness on ARM. If the compiler reorders the fence below the load (which it shouldn't — `atomic_thread_fence` is a compiler barrier), the check would be unsound.

### 3.3 `write_unlock` Advances Version by +2

```cpp
void write_unlock() noexcept {
    const auto old_lock_word = load_relaxed();
    const auto new_lock_word = old_lock_word.get() + 2;
    version.store(new_lock_word, std::memory_order_release);
}
```

This clears the lock bit (bit 1) and advances the version (bits 2+) by 1. The `+2` is because the version occupies bits 2–63, so incrementing the raw word by 2 increments the version field by 1. This is correct.

The `memory_order_release` on the store ensures all writes under the write_guard are visible before the version change. Readers who subsequently load the version with `load_acquire` (in `try_read_lock`) will see all the writes. **Correct.**

---

## Summary of Findings

| # | Component | Severity | Issue |
|---|-----------|----------|-------|
| 1.1 | CAS `update` | **Medium-High** | Lambda may execute on stale data with side effects on OLC restart. Must document or restructure. |
| 1.2 | CAS VIS path | **Low-Medium** | Fragile ordering between `unpack_value` and `check()`. Correct as written but needs defensive comments. |
| 1.4 | CAS linearizability | **Info** | Linearizable. Linearization point is the successful `try_upgrade_to_write_lock` CAS for `update`, or the `check()`/`try_read_unlock()` for `keep`. |
| 1.5 | CAS progress | **Medium** | Unbounded retry loop with no backoff. Lambda execution time widens the contention window. |
| 1.6 | CAS `erase` Phase 1 | **Medium** | Separate `remove()` call is not linearizable as a compound operation. Document the gap. |
| 1.8 | CAS code structure | **Low** | `if constexpr` in 200-line lock protocol function is a maintenance risk. |
| 2.1 | Bulk Phase 3 | **Low** | "Zero synchronization" claim is overstated — allocator and stats are shared. |
| 2.2 | Bulk root assembly | **Low-Medium** | Preconditions not enforced at runtime. UB on violation with no diagnostic. |
| 2.3 | Bulk stack depth | **Medium** | Recursion depth unbounded for long `key_view` keys. |
| 2.5 | Bulk memory ordering | **High** | Root publication via `in_critical_section::store()` uses relaxed ordering. Missing release fence for the bulk-built subtree. Broken on ARM. |
| 2.6 | Bulk QSBR | **Medium** | Parallel build threads may interact incorrectly with QSBR lifecycle. |
| 3.1 | Cross-cutting | **Info** | `in_critical_section` relaxed ordering is safe only under OLC protocol. |
| 3.2 | Cross-cutting | **Info** | TSan disables the acquire fence in `check()`, creating a testing blind spot. |

### Top 3 Action Items

1. **Fix bulk loader root publication ordering** (§2.5). Use a write_guard on `root_pointer_lock` when publishing the root, or add an explicit release fence. This is a correctness bug on non-x86.

2. **Document lambda re-execution semantics** (§1.1). The `insert_or_resolve` API must state that the lambda may be called multiple times, and that only the final invocation's result is committed. Side-effecting lambdas must be idempotent or externally guarded.

3. **Add backoff to the retry loop** (§1.5). At minimum, use exponential backoff with a cap. The current `spin_wait_loop_body()` (single `_mm_pause`) is insufficient for the wider contention window that `insert_or_resolve` creates.
