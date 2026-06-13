> Internal lock protocol for `upsert` (#847): try_upsert, try_upsert_erase, version validation, retry logic.

## Scope

Private implementation inside `olc_db` (and simpler direct paths in `db`/`mutex_db`).
Covers lock sequences, state transitions, CAS semantics, and TLA+-verified invariants.

---

## Private Signatures (olc_db)

```cpp
template <typename FN>
[[nodiscard]] try_update_result_type try_upsert(
    art_key_type k, value_type v, FN fn,
    olc_db_leaf_unique_ptr_type& cached_leaf);

[[nodiscard]] try_update_result_type try_upsert_erase(
    art_key_type k, version_tag_type captured_ver);
```
[source: §11.5, §11.1]

Return `std::nullopt` → restart outer loop. Return `bool` → committed result.

---

## db / mutex_db Path

`db::upsert` — direct implementation at duplicate-detection site (no retry loop).
`mutex_db::upsert` — delegates to `db` internal under `std::lock_guard{mutex}`.
[source: §4.3, §4.4]

---

## Three Storage Modes

| Mode | Where value lives | Version source |
|------|-------------------|----------------|
| Explicit leaf (keyed) | `leaf->data + key_size` | Leaf's `optimistic_lock` |
| Value-in-slot (VIS/packed) | Child pointer in inode | Containing inode's `optimistic_lock` |
| Root leaf | Root pointer | `root_pointer_lock` |

[source: §11.1]

---

## Keyed-Leaf Update Protocol

```
1. Read value → local copy under node RCS
2. Check parent_critical_section, node_critical_section
3. Invoke lambda(local_copy) → action
4. action == keep: release locks, return false
5. action == update:
   a. Upgrade node RCS → write_guard (CAS validates version)
   b. Upgrade FAILS: spin_wait_loop_body(), return nullopt (restart)
   c. Best-effort: std::ignore = parent_critical_section.try_read_unlock()
   d. leaf->set_value<Value>(local_copy)
   e. return false  ← COMMITTED, parent RCS irrelevant
6. action == erase: → erase protocol (below)
```
[source: §10.1 MUST-FIX]

**CRITICAL:** After write_guard acquired and value written, parent RCS failure
does NOT trigger restart. The write is committed. [source: §10.1]

---

## VIS Update Protocol

```
1. Unpack: art_policy::unpack_value(child_in_parent->load()) → local
2. Check node_critical_section
3. Invoke lambda(local) → action
4. action == keep: release locks, return false
5. action == update:
   a. Upgrade node RCS → write_guard on inode
   b. Upgrade FAILS: spin_wait_loop_body(), return nullopt
   c. Best-effort: std::ignore = parent_critical_section.try_read_unlock()
   d. *child_in_parent = art_policy::pack_value(local)
   e. return false  ← COMMITTED
6. action == erase: → erase protocol (below)
```
[source: §10.1 MUST-FIX]

---

## Erase Protocol (CAS with captured version)

```
1. Lambda returns erase → capture current version_tag_type
2. Release all RCS (exit critical sections)
3. Call try_upsert_erase(key, captured_ver):
   a. Traverse top-down (same as try_remove) with RCS at each level
   b. At target node, read current version M
   c. CAS check: M ≠ captured_ver → return nullopt (restart)
   d. M == captured_ver → proceed with remove protocol:
      - Upgrade parent RCS → write_guard
      - Upgrade node RCS → write_guard
      - Execute removal (unlink, shrink, chain-cut as needed)
   e. return false
4. If try_upsert_erase returns nullopt → restart entire upsert from top
```
[source: §11.1]

### Key-Absent on Re-traverse

Re-traversal finds key absent → return `nullopt` → outer loop restarts →
key absent → insert path → return `true`. [source: §11.2]

### Version Semantics

Any mutation to the value advances the protecting lock's version counter.
The captured version transitively validates that the value the lambda observed
has not been modified. [source: §11.1]

---

## Spin/Backoff

`spin_wait_loop_body()` called at write_guard upgrade failure point only.
[source: §10.3]

---

## Branch Prediction

No `UNODB_DETAIL_UNLIKELY` on duplicate-detection branch in upsert path
(duplicates are expected). Use `if constexpr` with policy tag to select hint.
[source: §9.4, §4.7]

---

## Template Dispatch

Tag-dispatch (`insert_only_tag` / `upsert_tag`) with `if constexpr` at
duplicate-detection site to avoid code duplication with existing insert path.
[source: §4.7]

---

## TLA+ Verified Invariants (OLCUpsertErase.tla, 3592 states)

| Invariant | Meaning |
|-----------|---------|
| CASSafety | Write-locked for erase ⇒ value == observed value |
| MutualExclusion | Upserter + writer never both hold write lock |
| VersionConsistency | Version odd ⟺ exactly one process holds write lock |
| NoEraseOfEmpty | Never erase an already-empty slot |
| UpsertProgress | Under weak fairness, upserter eventually reaches "done" |

[source: TLA+ spec, §11.2]

---

## Verification

- [ ] `try_upsert` returns `nullopt` on RCS/upgrade failure (unit test with sync points)
- [ ] `try_upsert_erase` rejects stale version (version mismatch → restart)
- [ ] Parent RCS failure after write_guard does NOT discard committed write
- [ ] All five TLA+ invariants hold (model checker output in repo)

## Deferred Items

- Exponential/randomized backoff variants (benchmark B6 evaluates, not shipped in Phase 1)
- Batch upsert (future)
