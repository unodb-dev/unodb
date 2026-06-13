> Test plan for `upsert` (#847): 39 test cases covering unit, concurrency, type coverage, erase, contract verification, and OOM.

## Scope

New file: `test/test_art_upsert.cpp`. All `db` types via typed test (GTest
`TYPED_TEST_SUITE`). Concurrency tests target `olc_db` only. OOM tests follow
existing `test_art_oom.cpp` pattern.

---

## Coverage Target

Patch coverage ≥ 98.63%. [source: AGENTS.md]

---

## Unit Tests — Basic Semantics

| ID | Name | Pre-condition | Action | Post-condition | Source |
|----|------|---------------|--------|----------------|--------|
| 1 | insert_path_key_absent | Empty or key not present | `upsert(k, v, fn)` | Returns `true`, `get(k) == v`, lambda not called | §7 #1 |
| 2 | keep_key_present | Key k present with value v0 | `upsert(k, v1, [](auto&){return keep;})` | Returns `false`, `get(k) == v0` | §7 #2 |
| 3 | update_key_present | Key k present with v0 | `upsert(k, v1, [](auto& x){x=42; return update;})` | Returns `false`, `get(k) == 42` | §7 #3 |
| 4 | update_idempotency | Key k present with v0 | Two upserts: `fn=[](auto& x){x+=10; return update;}` | Returns `false` both times. `get(k) == v0+20` | §7 #4 |
| 5 | erase_key_present | Key k present | `upsert(k, v, [](auto&){return erase;})` | Returns `false`, `get(k)` empty | §7 #5 |
| 6 | mixed_operations | Insert keys 0..99 with value=key*2 | Upsert each: if v<100 update to v+1, else keep | `get(i) == i*2+1` for i<50, `get(i) == i*2` for i≥50 | §7 #6 |
| 14 | root_leaf_all_actions | Single-entry tree (k=1, v=10) | Upsert keep: `get(1)==10`. Upsert update(v=20): `get(1)==20`. Upsert erase: `get(1)` empty, tree empty. | Per-action state verified | §7 #14 |
| 15 | empty_tree | Empty tree | `upsert(k, v, fn)` | Insert path, returns `true` | §7 #15 |
| 16 | after_clear | Tree cleared | `upsert(k, v, fn)` | Insert path, returns `true` | §7 #16 |

---

## Type Coverage

| ID | Key | Value | can_eliminate_leaf | Actions tested | Source |
|----|-----|-------|--------------------|----------------|--------|
| 11 | `uint64_t` | `uint64_t` | true (VIS) | keep, update, erase | §7 #11 |
| 12 | `key_view` | `uint64_t` | true (VIS) | keep, update, erase | §7 #12 |
| 13 | `key_view` | `value_view` | false | keep, erase only | §7 #13 |
| 13b | `uint64_t` | `value_view` | false | keep, erase only | test-design #13b |

For value_view types: verify `update` action triggers `CANNOT_HAPPEN` (death test or
assertion-failure test in debug builds). [source: A#5]

---

## Concurrency Tests (olc_db only)

| ID | Name | Setup | Threads | Verified property | Source |
|----|------|-------|---------|-------------------|--------|
| 7 | upsert_plus_get | N keys | Writers + readers | No crashes. All `get(k)` return a value from the set of written values. Tree size == N. | §7 #7 |
| 8 | upsert_disjoint | Empty | 2 threads, disjoint ranges | All keys present. `get(k) == expected` for each range. Tree size == sum of ranges. | §7 #8 |
| 9 | olc_restart | Sync point forces upgrade failure | 2 threads | `get(k) == final_value` (lambda's update applied exactly once). Tree size unchanged. | §7 #9 |
| 10 | upsert_vs_remove | 1 key | Updater + remover | `get(k)` returns updated value OR empty. Tree size == 0 or 1. | §7 #10 |
| 17 | cas_increment | 1 key, N threads | All increment | Final value == N | §9.3 #17 |
| 18 | cas_during_growth | I4 node | T1 upserts, T2 triggers I4→I16 | Both keys present. T1's value == lambda result. Tree well-formed (node counts correct). | §9.3 #18 |
| 19 | cas_key_removed | 1 key | T1 pauses at dup, T2 removes | T1 inserts, `get(k) == v` | §9.3 #19 |
| 20 | cas_plus_scan | 1 key | Updater + scanner | Scan sees old or new, never torn | §9.3 #20 |
| 21 | random_ops_stress | 1M keys | Mixed ops, sustained | No crashes. Tree size ≥ 0. All `get(k)` return valid value or empty. No ASAN/TSan errors. | §9.3 #21 |
| 22 | idempotency_under_contention | Hot keys | N threads | Value == N updates, lambda called ≥ N | §9.3 #22 |

---

## Erase-Specific Tests

| ID | Name | Setup | Verified property | Source |
|----|------|-------|-------------------|--------|
| 23 | erase_cas_retry | 1 key, 2 threads | Version mismatch → retry → erase or re-invoke | §10.6 #23 |
| 23a | erase_triggers_shrink | Min-size inode | I16→I4 shrink executes | test-design #23a |
| 23b | erase_triggers_chain_cut | key_view chain | Chain cut, tree well-formed | test-design #23b |
| 23c | erase_root_leaf | Single entry | Tree becomes empty | test-design #23c |
| 23d | erase_vis_value | Packed value | Slot cleared, bitmask updated | test-design #23d |
| 23e | erase_then_reinsert | Key erased | Re-upsert inserts with new value | test-design #23e |
| 23f | concurrent_erase_x_erase | 1 key, 2 threads | Exactly one erases, other inserts. Post: key present, tree size==1, no ASAN. Sync point: `sync_after_erase_lambda_returns`. | test-design #23f, Cr#2, Cr#9 |
| 23g | erase_after_concurrent_remove | 1 key, sync point | T1 erase paused, T2 removes, T1 resumes → insert path. Final: key present with T1's insert value. | §11.3, test-design #23g |

---

## OOM Tests

| ID | Name | Injection point | Expected | Source |
|----|------|-----------------|----------|--------|
| OOM-1 | insert_path_oom | Leaf/inode allocation | `std::bad_alloc` thrown. Tree unchanged. `get(k)` returns pre-existing value or empty. | test-design OOM-1 |
| OOM-2 | erase_shrink_oom | Smaller-inode allocation | `std::bad_alloc` thrown. Key still present (`get(k)` returns value). Node counts unchanged. Subsequent erase (without OOM) succeeds. | test-design OOM-2 |

---

## Contract Verification Tests

| ID | Name | Setup | Verified property | Source |
|----|------|-------|-------------------|--------|
| C1 | static_assert_rejects_bad_lambda | Compile test | Lambda returning `int` or `void` fails `static_assert`. Negative compilation test. | api.md constraint |
| C2 | lambda_sees_different_values | 1 key, sync point forces restart after concurrent update | Lambda's second invocation receives a different value than the first. Thread-local log captures both. | api.md Cr#3 |
| C3 | mutations_discarded_on_erase | 1 key with v=10 | `upsert(k, 0, [](auto& x){x=99; return erase;})`. Post: key absent. If re-inserted, value is NOT 99. | api.md A#2 |
| C4 | throwing_lambda_tree_unchanged | 1 key with v=10 | Lambda throws `std::runtime_error`. Post: `get(k) == 10`. Tree size == 1. No leak (ASAN). | api.md exception safety |
| C5 | stats_counter_increments | 1 key, force erase retry via sync point | `upsert_erase_retry_count` increments. If retries > 64, `threshold_exceeded` fires. (STATS build only.) | api.md C#3, C#5 |
| C6 | parent_rcs_fail_after_commit | 1 key, sync point: fail parent RCS after write_guard acquired | Value is committed (write persists). Operation returns `false` (not nullopt/restart). `get(k) == updated_value`. | protocol.md §10.1 |

---

## Sync Points

Tests 9, 18, 19, 23f, 23g use existing `sync.hpp` / `thread_sync.hpp` mechanism
(debug builds only). [source: test-design]

Test 23f requires sync point `sync_after_erase_lambda_returns`. [source: Cr#9]

---

## Verification

- [ ] All 39 tests pass on GCC 13-15, Clang 17-21, MSVC 17.12+
- [ ] ASAN/TSan/UBSan clean on concurrency tests
- [ ] Patch coverage ≥ 98.63%
- [ ] Death test for value_view + update → CANNOT_HAPPEN

## Deferred Items

- Fuzz testing (future)
- Property-based testing with random lambda behaviors (future)
