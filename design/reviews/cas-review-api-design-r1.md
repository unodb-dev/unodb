# Adversarial API Design Review — CAS/Upsert (#847) & Bulk Load (#636)

**Reviewer:** Kiro (API design & C++ template metaprogramming)
**Date:** 2026-04-25
**Status:** Review

---

## Part I — `insert_or_resolve` (CAS Design)

### 1. Naming: `insert_or_resolve` Is Wrong

**Severity: High**

"Resolve" implies conflict resolution between two competing values. That is
not what this API does. The lambda does not receive the *proposed* value and
the *existing* value and choose between them — it receives only the existing
value and decides what to do with it. The proposed value `v` is silently
discarded on the existing-key path.

The name also breaks grep-ability. Nobody searching for "upsert" or "put"
will find `insert_or_resolve`.

**Alternatives ranked:**

| Name | Precedent | Accuracy | Discoverability |
|------|-----------|----------|-----------------|
| `upsert` | libcuckoo `uprase_fn`, RocksDB `Merge`, Redis `SET` | Good — universally understood | Excellent |
| `insert_or_update` | Java `ConcurrentHashMap.merge` | Precise | Good |
| `insert_or_resolve` | None | Misleading — "resolve" suggests arbitration | Poor |
| `put` | Java `Map.put`, DynamoDB | Ambiguous — usually means unconditional overwrite | Fair |
| `insert_or_visit` | Boost.Unordered `insert_or_visit` | Accurate for the `keep` case, misleading for `update`/`erase` | Fair |

**Recommendation:** Use `upsert`. It is the industry-standard term for
insert-or-update. If the `erase` action makes "upsert" feel incomplete,
consider `uprase` (libcuckoo's term) or keep `upsert` and document that the
lambda can request erasure.

### 2. Lambda Signature: Missing the Key

**Severity: High**

The proposed signature is:

```cpp
FN = upsert_action(value_type& existing)
```

This is insufficient. The lambda cannot see *which* key it is resolving.
Consider a common use case — conditional update based on key properties:

```cpp
db.insert_or_resolve(key, val, [](value_type& existing) {
    // Which key is this? I need to know to decide whether to update.
    // I can't capture `key` because the lambda might be called in a
    // retry loop (olc_db) with a different key if there's a hash
    // collision... wait, there are no hash collisions in ART. But
    // the caller still has to capture the key manually.
    return upsert_action::update;
});
```

The caller *can* capture the key via the lambda closure, but this is
error-prone and forces an unnecessary capture. libcuckoo's `uprase_fn`
passes `(mapped_type&, UpsertContext)` — but libcuckoo's keys are hashed,
so the key is implicit. In an ART, the key is the traversal path, and the
lambda may legitimately need it.

**Recommendation:** Provide two overloads or make the key available:

```cpp
// Overload 1: key-blind (common case)
template <typename FN>  // FN = upsert_action(value_type&)
bool upsert(Key k, value_type v, FN fn);

// Overload 2: key-aware
template <typename FN>  // FN = upsert_action(const Key&, value_type&)
bool upsert(Key k, value_type v, FN fn);
```

Use SFINAE/concepts to disambiguate:

```cpp
template <typename FN>
  requires std::invocable<FN, value_type&>
bool upsert(Key k, value_type v, FN fn);

template <typename FN>
  requires std::invocable<FN, const Key&, value_type&>
bool upsert(Key k, value_type v, FN fn);
```

This is a zero-cost abstraction — the key-blind overload simply wraps the
lambda.

### 3. Lambda Receives Mutable Ref for All Actions — Const-Correctness Violation

**Severity: Medium**

The lambda signature `upsert_action(value_type&)` gives the lambda a
mutable reference even when it returns `keep` or `erase`. This is a
const-correctness violation:

- For `keep`: the lambda should receive `const value_type&` — it is only
  inspecting the value to decide whether to keep it.
- For `erase`: the lambda should receive `const value_type&` — it is
  inspecting the value to decide whether to erase it.
- For `update`: the lambda needs `value_type&` — it is modifying in place.

But the lambda must declare its return type *before* it knows which action
it will take, so the signature must accommodate the most permissive case
(`value_type&`). This is a fundamental tension.

**Mitigation:** Document that the lambda SHOULD NOT mutate the value if it
returns `keep` or `erase`. The implementation copies the value to a local
before invoking the lambda, so mutations are harmless for `keep`/`erase`
(they are simply discarded). But this is a footgun — a user who mutates and
returns `keep` will be surprised that the mutation is lost.

**Alternative design (rejected for complexity):** Use a visitor pattern
where the lambda receives a proxy object:

```cpp
struct upsert_context {
    const value_type& value() const;
    void update(const value_type& new_val);
    void erase();
    void keep();
};
```

This is cleaner but heavier. Not recommended for a low-level index library.

### 4. Return Type Is Too Narrow

**Severity: Medium**

`insert_or_resolve` returns `bool` — `true` if inserted, `false` if key
existed. The caller cannot distinguish between:

- Key existed, lambda returned `keep` (value unchanged)
- Key existed, lambda returned `update` (value changed)
- Key existed, lambda returned `erase` (entry removed)

The caller *does* know the action because they returned it from the lambda,
but in practice the lambda is often a stateless function object, and the
caller may want to know the outcome without threading state through the
closure.

**Recommendation:** Return a richer type:

```cpp
enum class upsert_result { inserted, kept, updated, erased };
```

Or at minimum, return `std::pair<bool, upsert_action>` where the bool
indicates insertion and the action indicates what happened on the
existing-key path (meaningless when bool is true).

This also future-proofs the API for returning the old value (e.g.,
`std::optional<value_type>` for the previous value on update/erase).

### 5. Template Constraints Are Missing

**Severity: High**

The design shows:

```cpp
template <typename FN>
[[nodiscard]] bool insert_or_resolve(Key k, value_type v, FN fn);
```

There is no constraint on `FN`. If the user passes a lambda with the wrong
signature, they will get an incomprehensible error deep inside the
implementation. The existing `scan` methods have the same problem, but that
is not an excuse to perpetuate it.

**Recommendation:** Add a concept:

```cpp
template <typename FN, typename V>
concept upsert_fn = std::invocable<FN, V&> &&
    std::same_as<std::invoke_result_t<FN, V&>, upsert_action>;

template <upsert_fn<value_type> FN>
[[nodiscard]] bool upsert(Key k, value_type v, FN fn);
```

If C++20 concepts are not available in the project's minimum standard, use
`static_assert` inside the method body:

```cpp
template <typename FN>
bool upsert(Key k, value_type v, FN fn) {
    static_assert(std::is_invocable_r_v<upsert_action, FN, value_type&>,
        "upsert lambda must have signature: upsert_action(value_type&)");
    ...
}
```

### 6. `noexcept` Specification Is Inconsistent

**Severity: Medium**

Existing API:

| Method | `noexcept`? | Throws? |
|--------|-------------|---------|
| `get()` | Yes (`noexcept`) | No |
| `insert()` | No | Yes (`std::bad_alloc`, `std::length_error`) |
| `remove()` | No | No (but not marked `noexcept`) |
| `clear()` | Yes (`noexcept`) | No |

The design does not specify `noexcept` for `insert_or_resolve`. It should
follow `insert`'s convention (not `noexcept`) because:

1. It may allocate (on the insert path).
2. The user's lambda may throw.

**But:** If the lambda throws, what is the exception guarantee? The design
says "lambda operates on a local copy; write-back only happens after lambda
returns successfully." This gives **strong exception safety** for the
`update` path — if the lambda throws, the tree is unmodified. Good.

However, this is not documented in the design's API section. It should be.

**Also:** `remove()` is not `noexcept` but never throws. It should be
`noexcept` (separate issue, but `insert_or_resolve` should not repeat this
inconsistency).

### 7. Value Semantics: `value_view` Update Is a Time Bomb

**Severity: High**

The design acknowledges this:

> For `value_view` values, `set_value` requires the new value to have the
> same size as the existing one (the leaf is immutable-size).

But the API signature `upsert_action(value_type&)` where `value_type =
value_view` gives the lambda a `value_view&` — a *span reference*. The
lambda can change what the span points to, but it cannot change the size of
the data stored in the leaf. If the lambda assigns a `value_view` of
different length, the `set_value` call will silently write the wrong number
of bytes (buffer overflow or truncation).

**This is undefined behavior waiting to happen.**

**Recommendation for Phase 1:** When `Value = value_view`, either:

1. **Disable `update` at compile time** via `static_assert` or SFINAE.
2. **Assert same-size at runtime** in `set_value` (debug) and document the
   restriction.
3. **Change the lambda signature** for `value_view` to receive a
   `std::span<std::byte>` (mutable span of fixed size) instead of
   `value_view&`, making the size constraint explicit.

Option 2 is the minimum viable approach. Option 3 is the correct long-term
design.

### 8. VIS Unpack-Mutate-Repack: Move Semantics Ignored

**Severity: Low (for now)**

The design says:

> For VIS types: unpack to local, pass to lambda, repack.

This works because VIS values are `trivially_copyable` (enforced by
`static_assert` in `get_value`). But the design does not discuss what
happens if `Value` is not trivially copyable. Currently, `get_value` has:

```cpp
static_assert(std::is_trivially_copyable_v<Value>);
```

So this is safe *today*. But if the `Value` constraint is ever relaxed
(e.g., to support `std::string`), the unpack-mutate-repack pattern will
silently do the wrong thing (memcpy of non-trivially-copyable type = UB).

**Recommendation:** Add a comment in the design noting this constraint
explicitly, and ensure `set_value` has the same `static_assert`.

### 9. `erase_fn` Is Redundant

**Severity: Low**

The proposed `erase_fn(Key k, FN fn)` where `FN = bool(const value_type&)`
is equivalent to:

```cpp
db.upsert(k, dummy_value, [&](value_type& existing) {
    return fn(existing) ? upsert_action::erase : upsert_action::keep;
});
```

The only difference is that `erase_fn` does not require a `dummy_value`
argument. But this can be addressed by providing a `visit` or `find_fn`
method instead:

```cpp
template <typename FN>
bool visit(Key k, FN fn);  // fn(const value_type&) → void
```

**Recommendation:** Defer `erase_fn` to Phase 2 as planned. Consider
whether a more general `visit` (read-only lambda on existing key) subsumes
both `erase_fn` and the `keep` path of `upsert`.

### 10. Phase 1 Erase Behavior: Silent No-Op Is Dangerous

**Severity: Medium**

The design says for Phase 1:

> `erase` action: assert-fail or return `keep` with a diagnostic.

Neither option is acceptable:

- **Assert-fail** crashes the program in debug builds and is UB in release.
- **Return `keep`** silently ignores the user's explicit request to erase.

**Recommendation:** In Phase 1, if the lambda returns `erase`:

1. `static_assert(false)` at compile time if possible (not possible with
   runtime enum).
2. `throw std::logic_error("upsert erase action not yet implemented")`.
3. Or simply don't include `erase` in the `upsert_action` enum until Phase
   2. Add it when the implementation supports it.

Option 3 is cleanest — ship `enum class upsert_action { keep, update };`
in Phase 1, extend to `{ keep, update, erase }` in Phase 2.

### 11. OLC Lock Protocol: Read-Lock Check After Lambda Is Racy

**Severity: High (correctness)**

In the OLC `update` path (§4.5.2), the design shows:

```cpp
auto existing_value = leaf->template get_value<Value>();
if (UNODB_DETAIL_UNLIKELY(!parent_critical_section.check())) return {};
if (UNODB_DETAIL_UNLIKELY(!node_critical_section.check())) return {};

const auto action = fn(existing_value);  // lambda runs here

if (action == upsert_action::update) {
    optimistic_lock::write_guard leaf_guard{
        std::move(node_critical_section)};
    ...
```

The problem: between `get_value` and the `check()` calls, the leaf could
have been concurrently modified. The `check()` validates that the version
hasn't changed, which is correct. But between the `check()` and the
`write_guard` upgrade, another writer could have modified the leaf. The
`write_guard` upgrade will detect this (it checks the version again), so
correctness is maintained — but the lambda will have operated on a **stale
copy** of the value.

This is actually fine for the "copy to local, mutate, write back" pattern
because the write_guard upgrade will fail and the operation will restart.
But it means the lambda may be called **multiple times** for the same
logical operation. The design does not document this.

**Recommendation:** Document explicitly:

> **The lambda may be invoked more than once** for the same key due to OLC
> restarts. The lambda MUST be idempotent or at minimum side-effect-free
> (other than mutating the passed value reference). Lambdas that perform
> I/O, increment counters, or acquire external locks are unsafe.

This is a critical API contract that must be in the public documentation.

### 12. Code Duplication Strategy: Tag Dispatch Is Insufficient

**Severity: Medium (maintainability)**

The design proposes tag dispatch to avoid duplicating `try_insert`:

```cpp
template <typename InsertPolicy, typename... FnArgs>
try_update_result_type try_insert_impl(..., FnArgs&&... fn_args);
```

This is the right idea but the wrong mechanism. The duplicate-detection
sites in `try_insert` are deeply nested inside `while(true)` loops with
complex lock state. A tag dispatch at the duplicate site means the entire
function body is duplicated in the template instantiation, with only a few
lines differing.

**Better approach:** Extract the duplicate-detection action into a
**policy functor** passed as a template parameter:

```cpp
struct plain_insert_policy {
    template <typename... Args>
    static bool on_duplicate(Args&&...) { return false; }
};

struct upsert_policy {
    template <typename FN>
    static bool on_duplicate(leaf_type* leaf, FN& fn, ...) {
        auto val = leaf->get_value<value_type>();
        auto action = fn(val);
        if (action == upsert_action::update) {
            leaf->set_value(val);
        }
        return false;
    }
};
```

This keeps the traversal logic in one place and makes the policy
substitution explicit and testable.

---

## Part II — Bulk Load (#636)

### 13. `bulk_load` Preconditions Are Unenforceable

**Severity: High**

The design specifies:

> Precondition: [first, last) MUST be sorted by key in ascending order
> with no duplicate keys.

And:

> These preconditions are asserted in debug builds but not checked in
> release builds (the caller is responsible).

This is a correctness time bomb. Violating the sorted-order precondition
produces a **silently corrupt tree** — not a crash, not an exception, just
wrong results from subsequent queries. This is the worst possible failure
mode.

**Recommendation:** For Phase 1 (sorted sequential insert), checking the
precondition is O(N) — the same complexity as the insert loop itself. The
overhead is negligible:

```cpp
template <typename InputIt>
std::size_t bulk_load(InputIt first, InputIt last) {
    UNODB_DETAIL_ASSERT(empty());
    std::size_t count = 0;
    Key prev{};
    bool have_prev = false;
    for (auto it = first; it != last; ++it) {
        if (have_prev) {
            UNODB_DETAIL_ASSERT(prev < it->first);  // sorted, no dups
        }
        insert(it->first, it->second);
        prev = it->first;
        have_prev = true;
        ++count;
    }
    return count;
}
```

For Phase 2 (bottom-up), the sorted-order check is naturally embedded in
the `common_prefix` / `group_by_byte` logic — an out-of-order key will
produce an incorrect grouping that can be detected.

**Stronger recommendation:** Provide two entry points:

```cpp
// Checked: validates preconditions, throws on violation
std::size_t bulk_load(InputIt first, InputIt last);

// Unchecked: UB on precondition violation (for performance-critical paths)
std::size_t bulk_load_unchecked(InputIt first, InputIt last);
```

### 14. `bulk_load` Return Type Should Not Be `std::size_t`

**Severity: Low**

If the precondition is "no duplicates" and the input is trusted, the return
value is always `distance(first, last)` — the caller already knows this.
Returning it is redundant.

If the precondition is relaxed to allow duplicates (future phase), then the
return value becomes meaningful (number of *unique* keys inserted).

**Recommendation:** Return `void` for Phase 1 (precondition: no duplicates).
Change to `std::size_t` if/when duplicate handling is added.

### 15. Iterator Pair API Is Insufficient

**Severity: Medium**

The proposed API:

```cpp
template <typename InputIt>
std::size_t bulk_load(InputIt first, InputIt last);
```

Requires `InputIt::value_type` to be `std::pair<Key, value_type>` (or
similar). But the design does not specify this constraint. What if the
iterator yields `std::tuple<Key, value_type>`? What about
`std::pair<const Key, value_type>` (the standard map value type)?

**Recommendation:** Specify the constraint explicitly:

```cpp
template <typename InputIt>
  requires requires(InputIt it) {
      { it->first } -> std::convertible_to<Key>;
      { it->second } -> std::convertible_to<value_type>;
  }
std::size_t bulk_load(InputIt first, InputIt last);
```

Or use a projection:

```cpp
template <typename InputIt, typename KeyProj, typename ValProj>
std::size_t bulk_load(InputIt first, InputIt last,
                      KeyProj key_proj, ValProj val_proj);
```

### 16. Error Handling: Allocation Failure Mid-Build

**Severity: High**

The design mentions exception safety briefly:

> On `std::bad_alloc`, free all nodes allocated so far.

But for Phase 1 (sorted sequential insert), this is already handled by the
existing `insert` method — each insert is independent, and on failure the
tree contains all previously-inserted entries. This is the **basic exception
guarantee** (tree is in a valid state, but partially loaded).

For Phase 2 (bottom-up), the situation is worse. The recursive builder
allocates nodes bottom-up. If allocation fails at depth 3 of a 10-level
tree, the partially-built subtree must be freed. The design mentions RAII
guards but does not specify the guarantee:

- **Basic guarantee:** Tree is empty (all partial work freed). This is the
  only sensible guarantee for bottom-up construction, since the tree is
  being built from scratch.
- **Strong guarantee:** Tree is unchanged (empty in, empty out). Same as
  basic for an initially-empty tree.

**Recommendation:** Specify explicitly: "On exception, the tree remains
empty. All partially-allocated nodes are freed." This is the strong
guarantee, and it is achievable because the tree starts empty.

### 17. `olc_db::bulk_load` Concurrency Semantics Are Underspecified

**Severity: High**

The design says:

> Phase 1 (sorted insert): Each insert acquires write locks as usual.

This means `olc_db::bulk_load` in Phase 1 is just a loop of `insert()`
calls. Concurrent readers will see a partially-loaded tree. Is this
acceptable? The design does not say.

For Phase 2/3:

> The tree is built without locks (no concurrent readers during
> construction). The root is published with a single atomic store.

This requires **exclusive access** — no concurrent readers or writers. But
`olc_db` is specifically designed for concurrent access. Requiring exclusive
access for bulk load defeats the purpose.

**Recommendation:** Be explicit about the concurrency contract:

- Phase 1: "Concurrent readers may observe a partially-loaded tree.
  Concurrent writers are safe (OLC protocol). This is the same guarantee as
  calling `insert()` in a loop."
- Phase 2/3: "Requires exclusive access. The caller must ensure no
  concurrent readers or writers exist. Violating this is undefined behavior."

Consider whether Phase 2/3 should only be available on `db` (not `olc_db`),
with `olc_db` always using Phase 1.

### 18. `bulk_load` on Non-Empty Tree: The Design Punts

**Severity: Medium**

The design says:

> Empty-tree precondition. Merging into an existing tree is substantially
> more complex (Phase 2).

This is reasonable for Phase 1, but the precondition should be enforced at
runtime (not just asserted):

```cpp
if (!empty()) {
    throw std::logic_error("bulk_load requires an empty tree");
}
```

An assert-only check means a release build will silently corrupt the tree
if called on a non-empty tree.

### 19. Phase 2 Bottom-Up: `create_with_children` Breaks Encapsulation

**Severity: Medium**

The design proposes new inode factory methods:

```cpp
static auto create_with_children(
    db_type& db, key_prefix_snapshot prefix,
    std::span<std::pair<std::byte, node_ptr>> children);
```

This bypasses the existing growth machinery (`add_or_choose_subtree`,
`add_to_nonfull`). It means:

1. Two code paths for inode construction — one for incremental insert, one
   for bulk load. Both must be kept in sync.
2. The `create_with_children` factory must correctly handle VIS (value-in-slot)
   packing, chain node creation for key_view, and stats tracking.
3. Any future change to inode layout must update both paths.

**Recommendation:** Accept this duplication as the cost of performance, but
mitigate it:

- Add comprehensive property-based tests that verify bulk-loaded trees are
  structurally identical to incrementally-built trees (same node types, same
  prefix compression, same stats).
- Consider making `create_with_children` call the same low-level primitives
  as `add_to_nonfull` (just without the growth check).

### 20. Missing: `bulk_load` for `mutex_db`

**Severity: Low**

The design mentions `mutex_db` delegates to `db::bulk_insert`, but the
locking strategy is not specified. Options:

1. Hold the mutex for the entire bulk load (simple, blocks all readers).
2. Release and re-acquire the mutex periodically (allows reader progress,
   but readers see partial state).
3. Build into the inner `db_` without the mutex, then publish (requires
   exclusive access).

**Recommendation:** Option 1 for Phase 1 (simplest, consistent with
existing `insert` behavior). Document that bulk load holds the lock for the
entire duration.

---

## Part III — Cross-Cutting Concerns

### 21. `[[nodiscard]]` Consistency

**Severity: Low**

Existing API consistently uses `[[nodiscard]]` on `get()`, `insert()`,
`remove()`. Both designs should follow this convention:

- `insert_or_resolve` / `upsert`: `[[nodiscard]]` ✓ (design shows this)
- `bulk_load`: Should be `[[nodiscard]]` if returning count, not needed if
  returning `void`.
- `erase_fn`: `[[nodiscard]]` ✓

### 22. `gnu::pure` / `gnu::const` Attributes

**Severity: Low**

`get()` is marked `[[gnu::pure]]`. `insert_or_resolve` should NOT be
`[[gnu::pure]]` (it modifies the tree). The design does not show this
attribute, which is correct by omission, but it should be explicitly noted
for consistency with the existing codebase's attribute discipline.

### 23. Both Designs Ignore `allocator_type`

**Severity: Low**

Both `db` and `olc_db` accept a custom `allocator_type`. Neither design
discusses how the new operations interact with custom allocators. For
`insert_or_resolve`, this is a non-issue (it uses the existing insert/remove
machinery). For `bulk_load` Phase 2/3, the bottom-up builder must use the
tree's allocator for all node allocations — the design mentions this but
should be more explicit about the Phase 3 parallel case (per-thread
allocators vs shared allocator).

---

## Summary of Findings

### CAS/Upsert — Must Fix Before Implementation

| # | Issue | Severity | Recommendation |
|---|-------|----------|----------------|
| 1 | Name `insert_or_resolve` is misleading | High | Rename to `upsert` |
| 5 | No template constraints on `FN` | High | Add concept or `static_assert` |
| 7 | `value_view` update is UB-prone | High | Disable or assert same-size |
| 10 | Phase 1 `erase` behavior undefined | Medium | Ship enum without `erase` in Phase 1 |
| 11 | Lambda may be called multiple times (OLC) | High | Document idempotency requirement |

### CAS/Upsert — Should Fix

| # | Issue | Severity | Recommendation |
|---|-------|----------|----------------|
| 2 | Lambda cannot see the key | High | Add key-aware overload |
| 4 | Return type too narrow | Medium | Return `upsert_result` enum |
| 6 | `noexcept` and exception safety undocumented | Medium | Document strong guarantee |
| 12 | Code duplication strategy | Medium | Use policy functor, not tag dispatch |

### Bulk Load — Must Fix Before Implementation

| # | Issue | Severity | Recommendation |
|---|-------|----------|----------------|
| 13 | Sorted-order precondition unenforceable | High | Check in release builds or provide checked/unchecked variants |
| 16 | Exception safety unspecified | High | Specify strong guarantee (tree remains empty) |
| 17 | `olc_db` concurrency semantics unspecified | High | Document exclusive-access requirement for Phase 2/3 |

### Bulk Load — Should Fix

| # | Issue | Severity | Recommendation |
|---|-------|----------|----------------|
| 15 | Iterator value_type constraint unspecified | Medium | Add concept or document requirement |
| 18 | Non-empty tree precondition assert-only | Medium | Throw in release builds |
| 19 | `create_with_children` duplication risk | Medium | Property-based tests for structural equivalence |
