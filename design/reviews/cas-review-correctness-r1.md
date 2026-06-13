# Adversarial Correctness Review: CAS `insert_or_resolve` (#847) and Bulk Loader (#636)

**Reviewer:** Formal verification & testing expert (adversarial)
**Date:** 2026-04-25
**Documents reviewed:**
- `design/cas-insert-or-resolve-847.md` (branch `design/cas-847`)
- `design/bulk-load-636.md`
- `test/db_test_utils.hpp`, `test/test_art_concurrency.cpp`, `test/test_art_key_view_full_chain.cpp`

---

## Part A: CAS / `insert_or_resolve` Design Review

### A.1 Missing Test Scenarios (Beyond the 16 Listed)

The design lists 16 test scenarios (§7). The following are **not covered** and represent real correctness risks:

#### A.1.1 Concurrent `insert_or_resolve` on the Same Key (CRITICAL)

Two threads call `insert_or_resolve(K, V, fn)` for the same key K simultaneously. Both reach the duplicate-detection site. Both read the existing value, both invoke their lambdas, both attempt to upgrade to write_guard. Only one can succeed; the other must restart. But after restart, the value has changed — the restarting thread's lambda sees the *updated* value from the winner, not the original.

**Why this matters:** The design's §4.5.2 shows the lambda operating on a *local copy* (`auto existing_value = leaf->get_value<Value>()`). After restart, the value is re-read, so the lambda runs again on the new value. This is correct **only if the lambda is idempotent or the caller tolerates re-execution**. The design does not document this re-execution guarantee or warn callers.

**Required test:**
```
T1: insert_or_resolve(K, V, [](auto& v) { v += 1; return update; })
T2: insert_or_resolve(K, V, [](auto& v) { v += 1; return update; })
// Final value must be original + 2, not original + 1
```

#### A.1.2 `insert_or_resolve` During Node Growth (I4→I16, etc.)

Thread T1 calls `insert_or_resolve` on key K (exists, in an I4 with 4 children). Thread T2 inserts a new key, triggering I4→I16 growth. T1's `node_critical_section` was taken on the old I4. After growth, the old I4 is obsoleted. T1's version check must fail and trigger restart.

**Why this matters:** The design's §4.5.3 (VIS path) upgrades `node_critical_section` to `write_guard` on the inode. If the inode was replaced by growth between the read and the upgrade, the version check catches it. But the design doesn't explicitly test this interleaving.

**Required test:** Use `sync_before_insert_grow_guard` (already exists for insert) to pause T2's growth, then have T1 attempt `insert_or_resolve(update)` on a key in the same inode.

#### A.1.3 `insert_or_resolve` on a Key Being Concurrently Removed

T1 calls `insert_or_resolve(K, V, fn)` — reaches the duplicate site, reads the value. T2 calls `remove(K)` — acquires write guards, removes the leaf. T1 attempts write_guard upgrade — the leaf's version has changed (or the leaf is freed). T1 must restart. On restart, the key is absent, so T1 takes the insert path.

**Why this matters:** After T2's remove, the leaf may be QSBR-deferred. T1 must not dereference a freed leaf. The OLC version check on the leaf should catch this, but the design doesn't analyze the QSBR interaction.

**Required test:**
```
Pre: insert(K, V)
T1: insert_or_resolve(K, V2, fn)  // paused at duplicate site
T2: remove(K)                      // completes
T1: resumes, upgrade fails, restarts → inserts K with V2
Final: get(K) == V2
```

#### A.1.4 VIS→Keyed-Leaf Transition During `insert_or_resolve`

When `can_eliminate_leaf` is true, values are packed into inode slots. The design's §4.5.3 handles VIS update by repacking. But what if the lambda's update produces a value that no longer fits in a slot (e.g., value size changes)? The design acknowledges this in §4.6 Open Question 1 but doesn't specify behavior.

**Risk:** If `Value = value_view` and the lambda changes the value size, `pack_value` will silently corrupt or the `set_value` will write past the leaf boundary.

**Required constraint:** `insert_or_resolve` with `update` action MUST assert at compile time or runtime that the value type is fixed-size, OR document that `value_view` update is UB if size changes.

#### A.1.5 `insert_or_resolve` with `erase` on Single-Entry Tree (Root is Leaf)

The design defers `erase` to Phase 2 but doesn't analyze the root-is-leaf case. When the tree has exactly one entry and the root is a leaf (not an inode), the erase path must set root to nullptr and decrement all counters. The existing `remove` handles this, but the "positioned remove" helper proposed for Phase 2 must also handle it.

#### A.1.6 `insert_or_resolve` After `clear()`

The design's scenario 16 mentions `insert_or_resolve` after `clear()`, but doesn't test the interaction with QSBR. After `clear()`, deferred deallocations may still be pending. An `insert_or_resolve` that takes the insert path must not interfere with pending QSBR reclamation.

#### A.1.7 Lambda Exception During OLC Write Guard

The design's §4.5.2 shows the lambda executing *before* the write guard upgrade. This is correct — the lambda operates on a local copy. But what if the lambda throws? The local copy is discarded, read locks are still held (as `read_critical_section` objects on the stack). Their destructors must handle the unwind correctly. The design should explicitly state that lambda exceptions are safe because no write has occurred.

**However:** If the lambda is invoked *after* write guard acquisition (which the code in §4.5.2 does NOT do — the lambda runs before upgrade), this is fine. But the code comment says "Invoke the lambda" before "Upgrade leaf lock to write." Verify this ordering is preserved in implementation.

---

### A.2 Invariant Violations

#### A.2.1 Node Counts After `update`

The `update` action modifies a value in-place. It must NOT change:
- `node_counts[]` — no nodes created or destroyed
- `current_memory_use` — leaf size unchanged (for fixed-size values)
- `growing_inode_counts[]` — no growth
- `shrinking_inode_counts[]` — no shrinkage
- `key_prefix_splits` — no prefix changes

**Test requirement:** After `insert_or_resolve(update)`, assert all stats are identical to before the call. The existing `tree_verifier` infrastructure tracks these (see `db_test_utils.hpp` lines with `UNODB_DETAIL_WITH_STATS`).

#### A.2.2 VIS Value Bitmask Consistency After `update`

For VIS trees (`can_eliminate_leaf` = true), values are packed into inode child slots with a bitmask tracking which slots contain values vs. child pointers. The `update` action repacks the value via `pack_value(existing_value)` and stores it back. This must NOT change the bitmask — the slot was already marked as a value.

**Risk:** If the implementation clears and re-sets the bitmask bit during update, a concurrent reader could see the slot as "not a value" momentarily. For `olc_db`, the write guard prevents concurrent reads, so this is safe. But for `db` (no locks), the bitmask must not be touched during update.

**Test requirement:** Insert N keys into a VIS tree, `insert_or_resolve(update)` on one, then verify `is_value_in_slot` returns true for all original VIS slots.

#### A.2.3 Memory Accounting for `erase` (Phase 2)

When `erase` removes a leaf, it must:
1. Decrement `node_counts[LEAF]` (for non-VIS) or clear the VIS bitmask bit
2. Decrement `current_memory_use` by the leaf's allocation size
3. Potentially trigger inode shrinkage (I16→I4, etc.) with corresponding stats updates
4. For `olc_db`, defer deallocation via QSBR

The design's Phase 2 "positioned remove" must replicate all of this. Missing any counter update will cause `tree_verifier::assert_empty()` to fail after removing all keys.

---

### A.3 Edge Cases

#### A.3.1 Empty `key_view`

The existing code rejects empty `key_view` with `std::length_error` (tested in `test_art_key_view_full_chain.cpp::EmptyKeyRejected`). `insert_or_resolve` must also reject empty keys on the insert path. On the resolve path, an empty key can never match (no key in the tree is empty), so the lambda is never invoked.

**Test:** `insert_or_resolve(empty_key, v, fn)` → `std::length_error`.

#### A.3.2 Maximum-Length Keys

Keys up to `UINT32_MAX` bytes are accepted (longer throws `std::length_error`). `insert_or_resolve` on a max-length key that already exists must traverse the full chain depth without stack overflow. The chain depth is bounded by `max_key_length / (key_prefix_capacity + 1)` ≈ `UINT32_MAX / 8` ≈ 500M levels. This WILL overflow the stack.

**Risk:** This is an existing limitation of the tree, not specific to `insert_or_resolve`. But the design should document it.

#### A.3.3 Keys Differing Only in the Last Byte

These trigger the dispatch-byte collision path (chain creation). `insert_or_resolve` on such a key must correctly traverse the chain to the bottom inode and find the leaf. The existing `CompoundKeysIdenticalExceptLastByte` test covers insert; an analogous test is needed for `insert_or_resolve`.

#### A.3.4 Values at VIS Eligibility Boundary

VIS is enabled when `sizeof(Value) <= sizeof(uint64_t)`. For `Value = uint64_t` (exactly 8 bytes), values are packed. For `Value = value_view`, VIS is disabled (variable size). The `insert_or_resolve(update)` path diverges:
- VIS: repack into inode slot (§4.5.3)
- Non-VIS: `set_value` on the leaf (§4.5.2)

**Test:** Explicitly test both paths with `<key_view, uint64_t>` (VIS) and `<key_view, value_view>` (non-VIS).

---

### A.4 GenMC / Model Checking Feasibility

#### A.4.1 Can the OLC CAS Protocol Be Verified with GenMC?

**Short answer: Partially, with significant effort.**

GenMC is a stateless model checker for C/C++ programs under weak memory models (RC11, IMM). It explores all thread interleavings and detects data races, assertion violations, and liveness issues.

**What can be modeled:**
- The `optimistic_lock` (version counter + locked bit) is a single `std::atomic<uint64_t>`. GenMC handles atomics natively.
- The read-check-upgrade protocol (`read_critical_section` → `check()` → `write_guard`) is a finite-state protocol amenable to model checking.
- A small tree (2-3 keys, 1 inode) with 2-3 threads doing `insert_or_resolve` + `get` + `remove` is feasible.

**What cannot be modeled (practically):**
- QSBR epoch management — GenMC would need to model the epoch counter, per-thread state, and deferred deallocation. This is possible but dramatically increases state space.
- Full tree traversal — GenMC explores all interleavings. A tree with N nodes and T threads has O(N^T) interleavings per operation. Keeping N ≤ 3 and T ≤ 3 is essential.
- Memory allocation — GenMC's default allocator is deterministic. Custom allocators (like unodb's `art_allocator`) need adaptation.

#### A.4.2 Proposed GenMC Test Harness

```cpp
// genmc_insert_or_resolve.cpp — model checking harness
#include <atomic>
#include <cassert>
#include <pthread.h>

// Simplified optimistic_lock (mirrors unodb::optimistic_lock)
struct opt_lock {
    std::atomic<uint64_t> version{0};  // even = unlocked, odd = locked

    uint64_t read_begin() { return version.load(std::memory_order_acquire); }
    bool check(uint64_t v) { return version.load(std::memory_order_acquire) == v; }
    bool try_upgrade(uint64_t v) {
        return version.compare_exchange_strong(v, v | 1, std::memory_order_acq_rel);
    }
    void write_unlock(uint64_t v) {
        version.store(v + 2, std::memory_order_release);  // bump version
    }
};

// Simplified leaf with value
struct leaf {
    opt_lock lock;
    uint64_t value;
};

// Simplified inode with one child slot
struct inode {
    opt_lock lock;
    leaf* child;
};

inode root;
leaf the_leaf = {{}, 42};

// insert_or_resolve(update) — simplified
bool insert_or_resolve_update(uint64_t new_val) {
    while (true) {
        auto pv = root.lock.read_begin();
        if (pv & 1) continue;  // locked, retry

        leaf* l = root.child;
        auto lv = l->lock.read_begin();
        if (lv & 1) continue;

        uint64_t old_val = l->value;  // read under optimistic lock
        if (!root.lock.check(pv)) continue;
        if (!l->lock.check(lv)) continue;

        // Lambda: mutate local copy
        uint64_t updated = old_val + new_val;

        // Upgrade leaf lock
        if (!l->lock.try_upgrade(lv)) continue;
        // Release parent read lock (just check)
        if (!root.lock.check(pv)) {
            l->lock.write_unlock(lv);  // release without writing
            continue;
        }
        l->value = updated;
        l->lock.write_unlock(lv);
        return false;  // key existed
    }
}

// Concurrent reader
uint64_t read_value() {
    while (true) {
        auto pv = root.lock.read_begin();
        if (pv & 1) continue;
        leaf* l = root.child;
        auto lv = l->lock.read_begin();
        if (lv & 1) continue;
        uint64_t v = l->value;
        if (l->lock.check(lv) && root.lock.check(pv)) return v;
    }
}

// Thread functions
void* writer1(void*) { insert_or_resolve_update(1); return nullptr; }
void* writer2(void*) { insert_or_resolve_update(2); return nullptr; }
void* reader(void*)  { uint64_t v = read_value(); assert(v >= 42); return nullptr; }

int main() {
    root.child = &the_leaf;
    pthread_t t1, t2, t3;
    pthread_create(&t1, nullptr, writer1, nullptr);
    pthread_create(&t2, nullptr, writer2, nullptr);
    pthread_create(&t3, nullptr, reader, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    pthread_join(t3, nullptr);
    // Final value: 42+1+2=45, or 42+2+1=45, or 42+(1+2)=45, etc.
    // Due to re-reads on restart, both increments are applied.
    assert(the_leaf.value == 45);
    return 0;
}
```

**Limitations:** This verifies the lock protocol only, not the full tree traversal. A more complete model would need to include inode child lookup and node growth, which would require 500+ lines of model code.

---

### A.5 Regression Risk

#### A.5.1 Tests That May Break When `insert_or_resolve` Is Added

1. **`test_art_oom.cpp`** — OOM tests are sensitive to allocation counts. Adding `insert_or_resolve` as a template method in `art.hpp` / `olc_art.hpp` may change template instantiation patterns, perturbing allocation counts. The OOM tests use hardcoded `fail_limit` values.

2. **`db_test_utils.hpp::tree_verifier`** — The verifier's `insert_internal` asserts `UNODB_ASSERT_TRUE(test_db.insert(k, v))`. If `insert_or_resolve` is implemented by cloning `try_insert` with a template parameter (§4.7), any bug in the shared code path affects both `insert` and `insert_or_resolve`.

3. **`test_art_concurrency.cpp`** — The `random_op_thread` function uses a 4-way switch (insert/remove/get/scan). Adding `insert_or_resolve` as a 5th operation changes the thread assignment modulo, which could alter coverage of existing operations. This is a test design issue, not a correctness issue.

4. **Stats assertions** — Many tests assert exact `node_counts` and `growing_inode_counts`. If the template-dispatch approach (§4.7) changes code generation, inlining decisions may differ, potentially affecting allocation patterns in debug vs. release builds.

#### A.5.2 API Surface Compatibility

The `insert_or_resolve` return type is `bool` (true = inserted). This is compatible with `insert`'s return type. However, the caller cannot distinguish between `keep` and `update` from the return value alone. If a future API change enriches the return type (e.g., `std::pair<bool, upsert_action>`), all call sites must be updated.

**Recommendation:** Return `bool` for Phase 1 (matches issue spec). Add a `insert_or_resolve_result` struct in Phase 2 if needed.

---

## Part B: Bulk Loader Design Review

### B.1 Invariant Preservation After `bulk_load`

#### B.1.1 Tree Invariants That Must Hold

After `bulk_load`, the tree must pass **every invariant check** that a sequentially-inserted tree passes:

| Invariant | Sequential Insert | Bulk Load (Phase 1) | Bulk Load (Phase 2) |
|-----------|------------------|---------------------|---------------------|
| `get(K)` returns correct value for all K | ✓ (by construction) | ✓ (delegates to `insert`) | **Must verify** |
| `scan()` visits all keys in order | ✓ | ✓ | **Must verify** |
| `node_counts[]` accurate | ✓ | ✓ | **RISK: bottom-up builder must count** |
| `current_memory_use` accurate | ✓ | ✓ | **RISK: direct allocation bypasses tracking** |
| `growing_inode_counts[]` accurate | ✓ | ✓ | **RISK: no growth events in bottom-up** |
| `key_prefix_splits` accurate | ✓ | ✓ | **RISK: no splits in bottom-up** |
| Inode prefix matches key bytes | ✓ | ✓ | **Must verify** |
| VIS bitmask consistent with slot contents | ✓ | ✓ | **Must verify** |
| Chain I4 nodes have exactly 1 child | ✓ | ✓ | **Must verify** |
| No inode has 0 children | ✓ | ✓ | **Must verify** |

#### B.1.2 Node Count Accuracy (Phase 2 Risk)

The bottom-up builder (§3.3) allocates inodes directly at their final size. This means:
- `growing_inode_counts` should reflect one event per inode at its final type (not the I4→I16→I48→I256 growth chain)
- `node_counts` must be incremented for every inode AND every leaf (or VIS slot)
- `current_memory_use` must include the exact allocation size of each node

**The design says** (§3.5): "After allocating an inode of type T: `++node_counts[as_i<T>]; current_memory_use += sizeof(T); ++growing_inode_counts[internal_as_i<T>];`"

**Problem:** `sizeof(T)` is wrong for memory accounting. The actual allocation size includes alignment padding and may differ from `sizeof`. The existing `insert` path uses `allocate_aligned` which returns the actual size. The bottom-up builder must use the same allocation path.

**Test requirement:** After `bulk_load(N keys)`, verify:
```cpp
assert(db.get_node_count<LEAF>() == N);  // for non-VIS
assert(db.get_current_memory_use() > 0);
// Insert the same N keys sequentially into a second tree, compare stats:
assert(bulk_tree.get_current_memory_use() == sequential_tree.get_current_memory_use());
```

Wait — this assertion is **wrong** for Phase 2. The bottom-up builder allocates inodes at their final size, while sequential insert creates temporary smaller inodes that are freed. The final `current_memory_use` should be the same (same final tree structure), but `growing_inode_counts` will differ.

**Corrected test:** Compare `node_counts` and `current_memory_use` (should match). Accept that `growing_inode_counts` and `shrinking_inode_counts` will differ.

#### B.1.3 VIS Bitmask in Bottom-Up Builder

For VIS trees, the bottom-up builder must set the value bitmask correctly when packing values into inode slots. The existing `add_or_choose_subtree` sets the bitmask during insert. The bottom-up builder's `add_child(byte, child)` must also set the bitmask if `child` is a packed value.

**Risk:** If the builder uses a generic `add_child` that doesn't distinguish between inode children and packed values, the bitmask will be wrong. Every subsequent `get`, `remove`, and `scan` will misidentify slots.

**Test requirement:** Bulk-load a VIS tree, then verify every key via `get()` and a full `scan()`. The existing `check_present_values()` in `tree_verifier` does both.

---

### B.2 Error Recovery (Phase 2/3)

#### B.2.1 Allocation Failure Mid-Build

The design says (§3.5): "On `std::bad_alloc`, free all nodes allocated so far. Use RAII guards."

**Analysis of the recursive builder:**

```
build_subtree(keys[], depth):
    ...
    inode = allocate_inode(n)   // CAN THROW
    for (byte, subkeys) in groups:
        child = build_subtree(subkeys, ...)  // CAN THROW
        inode.add_child(byte, child)
    return inode
```

If `allocate_inode` throws at level L, all subtrees built at levels > L are already returned as `node_ptr` values. Who owns them?

**Scenario:** Builder has constructed 100 subtrees. The 101st `allocate_inode` throws. The 100 subtrees are owned by... nothing. They leak.

**Required design:** Each level must use a guard (like `subtree_guard` in `art_internal_impl.hpp`) that owns the partially-built inode and its children. On exception, the guard's destructor recursively frees the subtree.

```cpp
auto inode = allocate_inode(n);
subtree_guard guard{inode};  // owns inode
for (byte, subkeys) in groups:
    auto child = build_subtree(subkeys, depth + prefix_length + 1);
    inode.add_child(byte, child);  // transfers ownership to inode
guard.release();  // success — caller takes ownership
return inode;
```

**But:** The children added via `add_child` are now owned by the inode. If a *later* `build_subtree` call throws, the guard destroys the inode, which must recursively destroy its children. Does the inode destructor do this? In unodb, inode destruction is handled by `delete_subtree` (recursive). The guard must call `delete_subtree`, not just `dealloc`.

**Test requirement:** Use `allocation_failure_injector` (from `test_heap.hpp`) to fail at every possible allocation point during `bulk_load`. After each failure:
1. The tree must be empty (no partial state)
2. `current_memory_use` must be 0
3. No memory leaks (ASAN/valgrind clean)

#### B.2.2 Tree State After Failed `bulk_load`

The design's precondition is "tree MUST be empty." If `bulk_load` fails mid-build, the tree must remain empty. The root pointer must not be set until the entire tree is successfully constructed.

**Phase 1 (sorted insert):** Each `insert()` call is atomic. If the Nth insert fails, the tree contains N-1 keys. This violates the "all or nothing" expectation.

**Problem:** Phase 1's simple loop does NOT provide transactional semantics. A failed `bulk_load` leaves a partially-populated tree.

**Options:**
1. Document that Phase 1 `bulk_load` is not atomic — caller must `clear()` on failure
2. Build into a temporary tree, swap on success (adds memory overhead)
3. Accept partial state (pragmatic but surprising)

**Recommendation:** Option 1 for Phase 1. Phase 2's bottom-up builder naturally provides atomicity (root is set only after full construction).

#### B.2.3 Retry Safety

After a failed `bulk_load`, can the caller retry?

- **Phase 1:** If the tree is partially populated, retrying `bulk_load` violates the empty-tree precondition. Caller must `clear()` first.
- **Phase 2:** If the builder uses RAII guards correctly, the tree remains empty after failure. Retry is safe.

**Test requirement:** `bulk_load` fails → `clear()` → `bulk_load` succeeds → verify correctness.

---

### B.3 Bulk Loader Edge Cases

#### B.3.1 Single-Key Bulk Load

`bulk_load({(K, V)})` — one key. The tree should have a single leaf as root (no inodes). Phase 1 handles this (single `insert`). Phase 2's `build_subtree` must handle `len(keys) == 1` → `make_leaf`.

#### B.3.2 All Keys Share Maximum Prefix

N keys that share `key_prefix_capacity` (7) bytes of prefix. The bottom-up builder must create chain I4 nodes. For `key_view` keys with 100-byte shared prefix, this means 100/8 ≈ 12 chain levels. The builder's recursion depth is bounded by key length, not key count.

#### B.3.3 Duplicate Keys in Input

The design says "no duplicate keys" is a precondition, asserted in debug builds. But what happens in release builds if the caller violates this?

- **Phase 1:** `insert()` returns `false` for the duplicate. The count returned by `bulk_load` will be less than `distance(first, last)`. No corruption, but the caller may be confused.
- **Phase 2:** The bottom-up builder groups by byte. Two identical keys end up in the same group at every level, eventually reaching `len(keys) == 2` with identical keys. The builder creates a leaf for one and... what? It can't create two leaves with the same key. This is **undefined behavior** in Phase 2.

**Recommendation:** Phase 2 must detect duplicates during grouping (adjacent identical keys in sorted input) and either skip or assert.

#### B.3.4 Empty Input

`bulk_load(first, first)` — zero keys. The tree should remain empty. `bulk_load` should return 0.

#### B.3.5 Unsorted Input (Release Build)

If the caller passes unsorted input in a release build (precondition violated), Phase 1 will insert keys in the given order. The tree will be correct but suboptimal (no sorted-order benefit). Phase 2's bottom-up builder will produce a **corrupt tree** — grouping by byte assumes sorted order. Keys in the wrong group will be unreachable.

**Recommendation:** Phase 2 should validate sorted order even in release builds (O(N) check — just compare adjacent keys). The cost is negligible compared to the build.

---

### B.4 Bulk Loader + `olc_db` Interaction

#### B.4.1 Phase 1: OLC Lock Overhead

Phase 1 calls `insert()` in a loop. For `olc_db`, each insert acquires and releases OLC locks. With no concurrent readers (empty tree, exclusive access), this is pure overhead. The locks will never contend, but the atomic operations (version reads, CAS for write guard) add ~10-20ns per insert.

**Optimization opportunity:** Add a `bulk_insert_unlocked` path for `olc_db` that bypasses OLC when the caller guarantees exclusive access. This is a Phase 2/3 concern.

#### B.4.2 Phase 2/3: Atomic Root Publication

The design says: "The root is published with a single atomic store." For `olc_db`, the root pointer is `std::atomic<node_ptr>`. A single `store(release)` is sufficient to publish the tree to concurrent readers.

**But:** Before publication, no QSBR threads should be accessing the tree. The design requires "exclusive access during the build." This means:
1. No concurrent `get`, `insert`, `remove`, or `scan` during `bulk_load`
2. After `bulk_load` returns, the tree is visible to all threads

**Risk:** If a reader thread is in a quiescent state during `bulk_load` and resumes after publication, it will see the new tree. This is correct. But if a reader is mid-traversal when `bulk_load` starts (violating the exclusive access precondition), it may see a partially-constructed tree.

**Test requirement:** Verify that `bulk_load` on `olc_db` with no concurrent access produces a correct tree. Verify that concurrent access during `bulk_load` is documented as UB.

---

## Part C: Cross-Cutting Concerns

### C.1 Test Infrastructure Gaps

#### C.1.1 No `tree_verifier` Support for `insert_or_resolve`

The `tree_verifier` class in `db_test_utils.hpp` has `insert`, `remove`, `try_insert`, `try_remove`, `try_get`, and `check_present_values`. It does NOT have `insert_or_resolve` or `try_insert_or_resolve`. Adding these requires:

1. A `resolve` method that updates the verifier's ground-truth `values` map
2. Handling the three actions: `keep` (no map change), `update` (update map value), `erase` (remove from map)
3. Stats verification after each action

#### C.1.2 No Concurrency Test for `insert_or_resolve`

The existing `ARTConcurrencyTest` uses a 4-way operation switch. Adding `insert_or_resolve` requires:
1. A 5th operation in `random_op_thread` and `key_range_op_thread`
2. A dedicated `insert_or_resolve` thread function for targeted testing
3. Sync-point tests (like CT1-CT4) for `insert_or_resolve` + concurrent mutation

#### C.1.3 No Bulk Load Test Infrastructure

The test suite has no `bulk_load` tests. Required:
1. Correctness: `bulk_load` N keys → `check_present_values()` → `scan()` verification
2. Stats: compare `node_counts` and `memory_use` with sequential insert
3. OOM: `allocation_failure_injector` at every allocation point
4. Edge cases: 0 keys, 1 key, duplicate keys, unsorted keys (debug assert)
5. Type coverage: all 9 db types (`u64_db`, `key_view_db`, `key_view_u64val_db`, × 3 concurrency modes)

### C.2 Interaction Between `insert_or_resolve` and `bulk_load`

If a tree is bulk-loaded and then `insert_or_resolve` is called on an existing key, the tree structure produced by bulk load must be compatible with the `insert_or_resolve` traversal. Specifically:

- Phase 1 bulk load produces the same tree structure as sequential insert → no issue
- Phase 2 bulk load produces a different structure (pre-sized inodes, no growth history) → the traversal is the same (prefix matching + child lookup), so `insert_or_resolve` should work. But the inode types may differ (e.g., bulk load creates I16 directly where sequential insert would have created I4 then grown to I16). The `insert_or_resolve` code must not assume any particular inode type at any level.

**Test requirement:** `bulk_load` N keys → `insert_or_resolve(update)` on existing key → verify value changed.

---

## Part D: Summary of Required Actions

### D.1 Critical (Must Fix Before Merge)

| # | Issue | Design | Risk |
|---|-------|--------|------|
| 1 | Lambda re-execution on OLC restart not documented | CAS §4.5.2 | Silent data corruption if lambda is not idempotent |
| 2 | `value_view` update may change size → UB | CAS §4.6 Q1 | Memory corruption |
| 3 | Phase 1 `bulk_load` is not atomic on failure | Bulk §3.2 | Partially-populated tree surprises caller |
| 4 | Phase 2 bottom-up builder leaks on allocation failure | Bulk §3.3 | Memory leak |

### D.2 High Priority (Should Fix Before Merge)

| # | Issue | Design | Risk |
|---|-------|--------|------|
| 5 | No test for concurrent `insert_or_resolve` on same key | CAS §7 | Missed increment bug |
| 6 | No test for `insert_or_resolve` during node growth | CAS §7 | OLC restart not exercised |
| 7 | No test for `insert_or_resolve` on key being removed | CAS §7 | Use-after-free via QSBR |
| 8 | Phase 2 bulk loader doesn't detect duplicates | Bulk §3.3 | Corrupt tree in release builds |
| 9 | `growing_inode_counts` semantics differ between bulk and sequential | Bulk §3.5 | Stats comparison tests fail |

### D.3 Medium Priority (Should Address)

| # | Issue | Design | Risk |
|---|-------|--------|------|
| 10 | No GenMC model for OLC CAS protocol | CAS — | Unverified lock protocol |
| 11 | No `tree_verifier` support for `insert_or_resolve` | Test infra | Manual test maintenance burden |
| 12 | Phase 2 should validate sorted order in release builds | Bulk §3.3 | Silent corruption |
| 13 | `bulk_load` + `insert_or_resolve` interaction untested | Cross-cutting | Structural assumption violation |
| 14 | OOM tests for `insert_or_resolve` not planned | CAS §7 | Exception safety unverified |
| 15 | VIS bitmask consistency after `update` not tested | CAS §4.5.3 | Bitmask desync |
