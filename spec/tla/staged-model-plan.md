# TLA+ Staged Model Plan: OLC ART Tree Maintenance

## Goal

High confidence in the correctness of the unodb OLC ART tree maintenance
algorithm under concurrent insert, remove, and scan operations. Specifically:

1. No lost updates (insert/remove linearizable)
2. No use-after-free (QSBR + version obsolescence)
3. No value-as-pointer dereference (VIS bitmask correctness)
4. No missed or phantom entries in scans (iterator consistency)
5. Chain cut maintains well-formedness invariant
6. No deadlock (lock ordering is always bottom-up or parent→child)

## Resource Constraints (HARD LIMITS)

| Resource | Limit |
|----------|-------|
| JVM heap | 4 GB (`-Xmx4g`) |
| Disk (metadir) | 10 GB max |
| Workers | 4 (`-workers 4`) |
| Wall time | 5 min per stage |
| Cleanup | `rm -rf /tmp/tlc-run` after EVERY run |

**Invocation template:**
```bash
cd .unodb-work/tla && \
  java -Xmx4g -XX:+UseParallelGC \
    -cp ~/tools/tla/tla2tools.jar tlc2.TLC \
    MODULE.tla -config CONFIG.cfg \
    -workers 4 -metadir /tmp/tlc-run \
    -noGenerateSpecTE 2>&1; \
  rm -rf /tmp/tlc-run
```

---

## Critical Interleavings Identified

From the research, these are the hazardous concurrent scenarios:

### H1: Insert × Scan (VIS slot interpretation)
- Writer inserts value=0 into inode256 slot (pack(0)=NULL, sets bitmask)
- Reader scans slots using `children[i] != nullptr` (misses the entry)
- **Property violated:** scan completeness

### H2: Insert × Remove (chain membership)
- Remove's downward pass reads I4 with count=1 (chain candidate)
- Concurrent insert adds a second child to that I4
- Remove's upward locking phase revalidates — must detect count change
- **Property violated:** chain membership precondition correctness

### H3: Remove × Scan (node obsolescence)
- Iterator holds cached version for a node
- Remove obsoletes that node (unlock_and_obsolete)
- Iterator tries to rehydrate — must detect obsolescence and restart
- **Property violated:** iterator safety (use-after-free)

### H4: Remove × Remove (chain cut race)
- Two removes target keys in the same chain
- Both identify overlapping chain segments
- Upward locking must serialize — second remove sees version change
- **Property violated:** well-formedness (double-cut)

### H5: Insert × Insert (node growth race)
- Two inserts target the same full node
- Both pre-allocate larger nodes
- Only one succeeds at upgrade; other restarts
- **Property violated:** lost update

### H6: Insert chain × Remove chain (VIS bitmask race)
- Insert does: add_to_nonfull(pack_value) → set_value_bit → overwrite with chain_ptr → clear_value_bit
- Concurrent scan reads between set_value_bit and clear_value_bit
- OLC version must cover the entire sequence
- **Property violated:** transient inconsistency visible to reader

### H7: Remove shrink × Scan descent
- Remove shrinks I48→I16, replaces pointer in grandparent
- Scanner descending through grandparent reads old pointer, descends into obsoleted I48
- Scanner must detect version change on grandparent before acting
- **Property violated:** stale pointer dereference

---

## Staged Model Design

### Abstraction Principles

1. **Model the protocol, not the data structure.** We don't need to model
   actual key bytes or prefix compression. We model: nodes with slots,
   version counters, and the lock protocol.

2. **One hazard per stage.** Each stage targets one or two specific
   interleavings. This keeps state space small and failures diagnosable.

3. **Minimal domain.** Use the smallest constants that exercise the hazard:
   typically 1-2 nodes, 2-3 slots, 1-2 values.

4. **Bound version counters.** Version is modular — only even/odd and
   same/different matter. Cap at 6 (3 full cycles).

5. **Bound process counts.** 1 writer + 1 reader per stage (except H4/H5
   which need 2 writers).

---

### Stage 1: VIS Slot Interpretation (Sequential)
**File:** `Inode256VIS.tla` — DONE ✓
**Hazard:** H1 (foundation)
**What:** Single inode256, insert/remove values and pointers, verify scan predicates
**Domain:** N=4 slots, Values={0,1,2}, Pointers={-1,-2,-3,-4}
**Result:** 4,096 states, <1s. Correct predicate verified, buggy predicate violated.

---

### Stage 2: OLC Single-Slot Protocol
**File:** `OLCSlot.tla`
**Hazard:** H1 + H6 (concurrent VIS modification)
**What:** 1 writer + 1 reader on a single slot. Writer modifies (slot, bitmask)
under version lock. Reader reads both, validates, interprets.
**Variables (7 scalars):**
- Shared: `slot`, `bitmask`, `version`
- Writer: `wpc` (idle|lock|write_slot|write_bitmask|unlock)
- Reader: `rpc` (idle|read_slot|read_bitmask|validate|interpret|done), `rslot`, `rbitmask`

**Domain:** slot ∈ {-1, 0, 1}, version ∈ 0..6
**State constraint:** `version <= 6`
**Expected:** ~5K states, <5s
**Properties:**
- `NoDereferenceValue`: reader in "done" never has value in deref set
- `SnapshotConsistency`: if validate passes, (rslot, rbitmask) is a committed pair
- `WriterProtocol`: shared state only modified when version is odd

---

### Stage 3: OLC Lock Coupling (Insert Path)
**File:** `OLCInsert.tla`
**Hazard:** H5 (two inserts racing for same slot)
**What:** 2 writers + 1 node with N=3 slots. Each writer: read version → find
empty slot → upgrade → write. Tests that only one succeeds.
**Variables (~10):**
- Shared: `slots[1..3]`, `version`, `count`
- Writer1: `w1pc`, `w1_target`
- Writer2: `w2pc`, `w2_target`

**Domain:** slots ∈ {0, 1} (empty/full), version ∈ 0..6
**Expected:** ~50K states, <30s
**Properties:**
- `NoLostUpdate`: if both target same slot, exactly one succeeds
- `CountConsistency`: count always equals number of non-empty slots
- `MutualExclusion`: at most one writer holds write lock at a time

---

### Stage 4: Chain Membership Under Concurrent Insert
**File:** `OLCChainMembership.tla`
**Hazard:** H2 (insert invalidates chain precondition)
**What:** 1 remover evaluating chain membership on an I4 node, 1 inserter
adding a child to that same I4. Models the revalidation pattern.
**Variables (~10):**
- Shared: `count` (1 or 2), `version`
- Remover: `rpc` (read_count|validate|upgrade|locked|done|restart), `r_cached_count`, `r_cached_ver`
- Inserter: `ipc` (idle|lock|insert|unlock)

**Domain:** count ∈ {1, 2}, version ∈ 0..6
**Expected:** ~2K states, <5s
**Properties:**
- `ChainSafety`: remover never proceeds with cut when count=2
- `ProgressUnderContention`: remover eventually either cuts or restarts
- `NoFalseChain`: if remover locks and proceeds, count was 1 at lock time

---

### Stage 5: Chain Cut with Concurrent Insert (Cases A/B/C)
**File:** `OLCChainCut.tla`
**Hazard:** H2 + H4 (cut_point_parent evaluation under mutation)
**What:** A 3-node chain (parent → chain_node → leaf). Remover does upward
locking + CPP evaluation. Inserter can add children to parent or chain_node.
**Variables (~14):**
- Shared: `parent_count`, `chain_count`, `parent_ver`, `chain_ver`, `parent_child` (points to chain or null)
- Remover: `rpc` (lock_chain|eval_cpp|cut|done|restart), `cut_level`, `holds_chain_lock`
- Inserter: `ipc`, `i_target` (parent or chain)

**Domain:** counts ∈ {1,2,3}, version ∈ 0..6
**Expected:** ~100K states, <60s
**Properties:**
- `WellFormedness`: after cut, no node has count < 1 (for I4 in key_view mode)
- `NoCutWithStaleView`: remover never cuts based on outdated count
- `CaseACorrect`: if parent has 2+ children including chain, cut proceeds
- `CaseBCorrect`: if parent became single-child, chain extends upward
- `CaseCCorrect`: if child pointer gone, remover restarts

---

### Stage 6: Iterator × Concurrent Remove (Obsolescence)
**File:** `OLCIterRemove.tla`
**Hazard:** H3 + H7 (node obsolescence during scan)
**What:** 1 iterator descending/advancing, 1 remover that can obsolete a node.
Tests that iterator detects obsolescence and restarts.
**Variables (~12):**
- Shared: `nodes[1..2]` each with `{version, obsolete, child}`
- Iterator: `ipc`, `i_stack` (cached versions), `i_current`
- Remover: `rpc`, `r_target`

**Domain:** 2 nodes, version ∈ 0..6
**Expected:** ~50K states, <30s
**Properties:**
- `NoUseAfterObsolete`: iterator never acts on data from an obsoleted node
- `RestartOnObsolete`: if rehydrate finds obsolete, iterator restarts
- `ProgressAfterRestart`: iterator eventually re-seeks and advances

---

### Stage 7: Insert Chain × Scan (VIS Transient State)
**File:** `OLCInsertChainVIS.tla`
**Hazard:** H6 (set_value_bit → overwrite → clear_value_bit sequence)
**What:** 1 writer performing the chain-insert protocol (4 steps under one
write lock), 1 reader scanning. Verifies reader never sees the transient
state where bitmask=TRUE but slot holds a chain pointer.
**Variables (~10):**
- Shared: `slot`, `bitmask`, `version`
- Writer: `wpc` (lock|set_bit|write_chain|clear_bit|unlock)
- Reader: `rpc`, `rslot`, `rbitmask`, `rver`

**Domain:** slot ∈ {0, "chain_ptr", "packed_val"}, version ∈ 0..6
**Expected:** ~3K states, <5s
**Properties:**
- `NoTransientVisible`: reader in "done" never sees (bitmask=TRUE, slot=chain_ptr)
- `ProtocolCovers`: all 4 writer steps are under one version lock

---

### Stage 8: Remove × Remove (Double Cut Prevention)
**File:** `OLCDoubleCut.tla`
**Hazard:** H4 (two removes targeting same chain)
**What:** 2 removers, 1 chain of 2 nodes. Both try to lock the chain bottom-up.
Only one can succeed at upgrading; the other sees version change.
**Variables (~12):**
- Shared: `chain_ver`, `chain_obsolete`, `parent_ver`, `parent_child`
- Remover1: `r1pc`, `r1_holds_lock`
- Remover2: `r2pc`, `r2_holds_lock`

**Domain:** version ∈ 0..8 (need more cycles for 2 writers)
**Expected:** ~20K states, <15s
**Properties:**
- `AtMostOneCut`: at most one remover reaches point-of-no-return
- `LoserRestarts`: the other remover detects conflict and restarts
- `NoDoubleFree`: a node is obsoleted at most once

---

## Verification Matrix

| Stage | Hazards | Slots | Procs | Est. States | Time | Key Property |
|-------|---------|-------|-------|-------------|------|--------------|
| 1 ✓ | H1 | 4 | 0 | 4K | <1s | Scan completeness |
| 2 | H1,H6 | 1 | 1W+1R | ~5K | <5s | No value-as-ptr |
| 3 | H5 | 3 | 2W | ~50K | <30s | No lost update |
| 4 | H2 | 1 | 1W+1R | ~2K | <5s | Chain membership |
| 5 | H2,H4 | — | 1W+1R | ~100K | <60s | Cut correctness |
| 6 | H3,H7 | — | 1W+1R | ~50K | <30s | No use-after-obsolete |
| 7 | H6 | 1 | 1W+1R | ~3K | <5s | No transient visible |
| 8 | H4 | — | 2W | ~20K | <15s | At-most-one cut |

**Total estimated time:** <4 minutes for all 8 stages.

---

## Success Criteria

All 8 stages pass TLC with 0 errors. This gives us:
- **Insert correctness:** Stages 3, 7 (mutual exclusion, VIS protocol)
- **Remove correctness:** Stages 4, 5, 8 (chain membership, cut, double-cut)
- **Scan correctness:** Stages 1, 2, 6 (slot interpretation, OLC protocol, obsolescence)
- **Cross-operation:** Stages 4, 5, 6, 7 (insert×remove, remove×scan, insert×scan)

After all stages pass, we implement the code fix with confidence that:
1. The bitmask-only predicate is correct (Stage 1)
2. OLC prevents partial observation (Stage 2)
3. The chain cut algorithm handles concurrent mutations (Stages 4, 5)
4. Iterators correctly detect and recover from concurrent changes (Stage 6)

---

## File Organization

```
.unodb-work/tla/
├── staged-model-plan.md          # THIS FILE
├── Inode256VIS.tla               # Stage 1 (DONE)
├── Inode256VIS.cfg
├── Inode256VIS_BugDemo.cfg
├── OLCSlot.tla                   # Stage 2
├── OLCSlot.cfg
├── OLCInsert.tla                 # Stage 3
├── OLCInsert.cfg
├── OLCChainMembership.tla        # Stage 4
├── OLCChainMembership.cfg
├── OLCChainCut.tla               # Stage 5
├── OLCChainCut.cfg
├── OLCIterRemove.tla             # Stage 6
├── OLCIterRemove.cfg
├── OLCInsertChainVIS.tla         # Stage 7
├── OLCInsertChainVIS.cfg
├── OLCDoubleCut.tla              # Stage 8
├── OLCDoubleCut.cfg
└── run-all-stages.sh             # Orchestrator script
```

---

## Incremental Expansion

After all stages pass at minimal domain, we can selectively expand:
- Stage 5: Add a second inserter (H2+H5 combined) — expect ~500K states
- Stage 6: Add 2 nodes → 3 nodes (deeper tree) — expect ~200K states
- Stage 3: Increase to N=4 slots — expect ~200K states

Only expand if the base passes AND we have specific doubt about a larger case.
Never expand all stages simultaneously.
