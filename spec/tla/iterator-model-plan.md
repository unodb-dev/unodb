# TLA+ Iterator Verification Model — Plan

**Epic:** S-e80a92f0 (TLA+ Iterator Verification Model)
**Parent:** S-19f88e0e (E4: TLA+ formal verification)

## Completed Phases

### Phase 0: Base Forward Scan (T-d693d16a) ✅

- `OLCIterator.tla` — parametric `ConcurrencyEnabled` (db vs olc_db)
- Forward scan with seek-restart on version check failure
- Properties: OrderPreservation, NoRepeat, Completeness
- 96K states (OLC mode), 14 states (sequential)

### Phase 1: Structural Remove + Havoc (T-78f6239d) ✅

- `OLCIteratorRemove.tla` — writer removes L2, collapses inode A
- Havoc model: stale read through obsolete node returns any leaf
- Correct code: 299 states, all properties pass
- Bug injection catches all three classes:
  - `skip_backtrack` → Completeness violated (7 states)
  - `restart_from_beginning` → NoRepeat violated (12 states)
  - `skip_check` + havoc → OrderPreservation violated (6 states)
- Canary (L2AlwaysVisited) proves structural interleaving

## Remaining Phases (Revised)

### Phase 3: VIS TOCTOU (HIGH priority) — NEXT

**Problem:** Packed leaves (value-in-slot) have no version lock. The parent
inode's version protects them. A TOCTOU window exists between check1 (before
`is_value_in_slot`) and check2 (after `get_child`). If check2 is missing,
a concurrent writer can mutate the slot mid-read.

**Geometry:** Single inode P with 3 slots:
- Slot 1: regular leaf L1
- Slot 2: VIS value V2 (mutable — writer can convert to regular child L2')
- Slot 3: regular leaf L3

**Iterator state machine:**
```
start → snap_ver → at_L1 → check1 → read_slot2 → check2 → deliver → at_L3 → done
                                                     ↓ (fail)
                                                   restart
```

**Writer:** Lock P → mutate slot2 (VIS↔regular, value changes) → unlock P

**Properties:**
- `NoGarbage`: delivered slot2 value ∈ {"V2", "L2prime"}
- `OrderPreservation`: delivered values in slot order
- `VISConsistency`: value delivered matches actual slot state at time of successful check

**Bug injection:** `skip_check2` — havoc on stale read → `NoGarbage` violated

**Expected:** ~500–2000 states. <1s.

**Files:** `OLCIteratorVIS.tla`, `OLCIteratorVIS.cfg`, `OLCIteratorVIS_Bug.cfg`

### Phase 5: Root Pointer Mutation + Emptying (MEDIUM priority)

**Problem:** Root pointer can change (tree grows/shrinks). Iterator takes
`root_pointer_lock`, loads root, then takes node lock. If root changes between
load and node lock, the iterator follows a stale root. Also: tree can become
empty (root→null) during active scan.

**Geometry:** Root pointer variable (mutable). Tree starts with Root→[L1, L2].
Writer can: delete all keys (root→null), or replace root (root→NewRoot).

**Properties:**
- `SafeTermination`: if tree empties, iterator reaches done (not crash)
- `NoStaleRoot`: iterator never descends into a freed root node

**Expected:** ~1000–5000 states.

### Phase 2: Structural Insert (LOW priority — subsumed by havoc)

The havoc model already proves skip_check causes violations for any stale
read. Phase 2 would add a concrete insert scenario (node split, child
migration) but doesn't catch new bug classes. Value is in proving the
correct code's SeekPath handles post-insert topology.

**Demoted.** Do only if a specific insert-related bug is suspected.

### Phase 6: Two Concurrent Writers (LOW priority)

Two writers don't interact with the iterator differently than one — the
iterator only sees version changes. Two writers interacting with each other
is covered by existing OLC chain cut models (Stages 5-10 in the suite).

**Demoted.** Do only if multi-writer iterator bugs are suspected.

### Phase 4: Reverse Scan — DROPPED

Forward and reverse scan are structurally symmetric (`next→prior`,
`left_most→right_most`). If the forward model verifies the backtrack pattern,
the reverse direction holds by symmetry. Bugs in reverse-only would be
implementation typos — not catchable by TLA+.

## Design Decisions

1. **Havoc for stale reads** — non-deterministic value on unchecked stale read.
   Sound (overapproximation), catches all ordering violations, no need to model
   specific memory reuse patterns.

2. **Decomposed verification** — Phase 0-1 proves traversal algorithm correct.
   Phase 3 proves VIS access protocol correct. Compose for full correctness.

3. **Single parameterized spec** — `ConcurrencyEnabled` flag covers both db
   (sequential) and olc_db (concurrent) in one file.

4. **Bug injection validation** — every spec includes a buggy config that must
   fail. If the buggy config passes, the model is too weak.

## Task Mapping

| Phase | Task | Priority | Status |
|-------|------|----------|--------|
| 0 | T-d693d16a | — | done |
| 1 | T-78f6239d | — | done |
| 3 | T-61829475 | high | done |
| 5 | T-1d41fb6a | medium | done |
| — | T-ef029958 | medium | done |
