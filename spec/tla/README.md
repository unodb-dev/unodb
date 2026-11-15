# TLA+ Verification Suite — OLC ART

Formal models verifying the concurrency protocol for the OLC
(Optimistic Lock Coupling) ART tree.

## Stages

| Stage | Module             | Property                             |
| ----- | ------------------ | ------------------------------------ |
| 1     | Inode256VIS        | Sequential VIS scan correctness      |
| 2     | OLCSlot            | Single-slot read/write protocol      |
| 3     | OLCInsert          | Two writers race on same slot        |
| 4     | OLCChainMembership | Chain membership revalidation        |
| 5     | OLCChainCut        | Chain cut Cases A/B/C                |
| 6     | OLCIterRemove      | Iterator vs remove (obsolescence)    |
| 7     | OLCInsertChainVIS  | Transient VIS during insert          |
| 8     | OLCDoubleCut       | Two concurrent removes               |
| 9     | OLCChainMultiLevel | Multi-level chain cuts               |
| 10    | OLCChainCutFull    | Root shrink and cascade              |
| 11    | OLCKeyViewChain    | VIS with shared prefix (key_view)    |
| 12    | ARTTreeMaintenance | Sequential tree structure invariants |

## Running

```bash
bash spec/tla/run-all-stages.sh
```

Requirements: Java (for TLC), `~/tools/tla/tla2tools.jar`.
Each stage uses 4 GB heap, 4 workers, 5-minute timeout.

## Key Invariants

1. No lost updates (insert/remove linearizable)
2. No use-after-free (QSBR + version obsolescence)
3. No value-as-pointer dereference (VIS bitmask correctness)
4. No missed or phantom entries in scans (iterator consistency)
5. Chain cut maintains tree well-formedness
6. No deadlock: a writer that holds write locks blocks only on a lock above
   them (a chain cut's cut-point parent or the root pointer) or on the leaf it
   removes; it takes every other write lock by a version-checked upgrade, for
   an I4 collapse's survivor after `try_read_lock_no_wait()`, and restarts if
   that fails. Not model-checked.

## Stage 12: Tree Structure Model

Stage 12 verifies sequential correctness of tree maintenance:

- Well-formedness: no single-child I4 unless prefix overflow blocks collapse
- Key preservation: every inserted key not yet removed is reachable
- No orphans: every node is reachable from root
- Correct chain identification: longest single-child I4 sequence above leaf
- Correct collapse: an I4 collapses when the prefix fits
