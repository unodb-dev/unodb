> Public API surface for `upsert` (#847): types, signatures, constraints, contracts.

## Scope

Public declarations added to `art_common.hpp`, `art.hpp`, `mutex_art.hpp`, and
`olc_art.hpp`. Covers the user-facing contract only; internal protocol is in
`protocol.md`.

---

## Types

```cpp
// art_common.hpp
enum class upsert_action { keep, update, erase };  // [source: §11.6]
```

---

## Signatures

All three `db` classes expose the same public template: [source: §4.1, §11.5]

```cpp
template <typename FN>
[[nodiscard]] bool upsert(Key k, value_type v, FN fn);
```

- Not `noexcept` — insert path allocates; lambda may throw. [source: §9.7]
- `[[nodiscard]]` per project convention. [source: AGENTS.md]

### Constraint (all db classes)

```cpp
static_assert(std::is_invocable_r_v<upsert_action, FN, value_type&>,
    "upsert lambda must be callable as upsert_action(value_type&)");
```
[source: §11.6, D-71d95041]

Single overload only — key-blind lambda. No key-aware overload. [source: D-71d95041]

---

## Return Value

| Condition | Return | Lambda called? |
|-----------|--------|----------------|
| Key absent — inserted | `true` | No |
| Key present — lambda invoked | `false` | Yes |

[source: §4.1]

---

## Lambda Contract

| Rule | Detail |
|------|--------|
| Idempotency | Lambda MUST be idempotent w.r.t. external side effects. [source: §9.1] |
| Re-invocation | MAY be called multiple times per `upsert` call (OLC restarts). [source: §9.1] |
| Different values | MAY receive different input values across re-executions. [source: §10.2, Cr#3] |
| No captured mutable state | MUST NOT use captured mutable state to skip work. [source: §10.2] |
| Different actions | MAY return a different action each time; implementation honors the most recent. [source: §11.7] |
| Mutations discarded on erase | If lambda returns `erase`, any in-place mutations to the value reference are discarded. [source: A#2] |

---

## value_view + update Handling

```cpp
if constexpr (std::is_same_v<value_type, value_view>) {
    UNODB_DETAIL_CANNOT_HAPPEN();  // runtime assertion, not static_assert on template
}
```
[source: A#5, §9.2]

Defense-in-depth `static_assert` inside `set_value`: [source: §10.4]

```cpp
static_assert(!std::is_same_v<Value, value_view>,
    "set_value cannot be used with value_view — leaf size is fixed");
```

---

## set_value (leaf mutation primitive)

```cpp
// basic_leaf<Key, Header>
template <typename Value>
constexpr void set_value(const Value& v) noexcept;
// Requires: std::is_trivially_copyable_v<Value>
```
[source: §4.2]

---

## Exception Safety

If the lambda throws, the tree is unchanged. Lambda operates on a local copy;
write-back occurs only after lambda returns successfully. [source: §9.7]

---

## Stats Counters (STATS build)

| Counter | Fires when |
|---------|-----------|
| `upsert_erase_retry_count` | Each erase re-traversal attempt. [source: C#3] |
| `upsert_erase_retry_threshold_exceeded` | Retry count > 64 for a single operation. [source: C#5] |

Eraser starvation: under sustained concurrent mutation of the same key, the erase
path may retry indefinitely. The stats counters above expose this condition.
[source: C#3]

---

## Verification

- [ ] `static_assert` fires for non-conforming lambda (compile-time test)
- [ ] `value_view` + `update` hits `CANNOT_HAPPEN` (runtime test, not compile error)
- [ ] Return value matches insert/existed semantics across all db types
- [ ] Stats counters increment in STATS build

## Deferred Items

- Key-aware lambda overload (D-71d95041 defers)
- `erase_fn` companion API (§4.6 — Phase 2)
- Richer return type beyond `bool` (§9.7 Q3 — future phase)
