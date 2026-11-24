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
| 13    | OLCPrefixRewrite   | Descent vs in-place prefix rewrite   |
| 14    | OLCChainCutVsSplit | Chain cut vs split of a node it uses |
| 15    | OLCIterReseek      | Iterator re-seek after failed step   |
| H10   | OLCRemoveDescent   | Value bit vs pointer in remove       |

Stage 13 models a descent at an inode's old depth while an I4 collapse or a
prefix split rewrites that inode's key prefix in place.  It checks the
survivor lock, the single-load prefix read, the check for a key that ends
inside a prefix, an upsert's parent recheck, and the chain remove's check
of a chain run's parent when it hands that parent's lock over; each has a
`OLCPrefixRewrite_BugDemo_*.cfg` that drops it and violates an invariant.

Stage 14 models `try_chain_cut` while an insert splits the prefix of a node
the cut relies on, which moves that node without obsoleting it and installs
an I4 above it.  It checks that Step 3 re-reads a moved node only with its
link to the node below, and that the root pointer is locked only while it
points to `stk[0]`, for a root chain cut and a shrinking root cut-point
parent.  Its three bug demonstrations each lose the inserted key.

Stage 15 models `next()` after a failed step for key_view keys without a
heap.  A step writes bytes into `keybuf_` before checks that can fail, and
the bug demonstration re-seeks to that `keybuf_`, as the code did before it
saved a copy, and returns an earlier key or skips a stable one.

H10 models a remove descent that reads a child's value bit after its check,
against a concurrent remove or an insert that rewrites the slot and its bit
in separate stores.  `OLCRemoveDescent_BugDemo.cfg` drops the re-check after
the bit read, as `try_remove_key_view`'s chain-I4 branch did, and violates
`NoDereferenceNonPointer`.

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
