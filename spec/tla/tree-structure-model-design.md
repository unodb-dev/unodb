# Stage 12: ART key_view Tree Structure Model — Design Parameters

## Overview

This document captures the design decisions for a TLA+ model of the ART (Adaptive Radix Tree) `key_view` algorithm. The model verifies structural invariants of insert and remove operations, focusing on chain nodes and the `try_collapse_i4` algorithm.

This is Stage 12 in the unodb verification suite. Stages 2–11 cover concurrency; this stage verifies sequential correctness of tree structure maintenance.

---

## Context

The ART `key_view` algorithm stores variable-length keys in the inode path (prefix bytes + dispatch bytes). Each inode level consumes up to 7 prefix bytes + 1 dispatch byte = 8 bytes. Keys that share a prefix create a shared path; where they diverge, a branching node is created.

**Chain nodes** are I4 inodes with exactly 1 child — they exist solely to encode key bytes in the path. The chain cut algorithm removes chains atomically when a leaf is deleted.

**try_collapse_i4** merges a single-child I4 with its child when the combined prefix fits:

```
parent_prefix + 1 (dispatch) + child_prefix ≤ 7
```

---

## Verification Goals

| # | Invariant | Description |
|---|-----------|-------------|
| 1 | Well-formedness | No single-child I4 exists unless prefix overflow blocks collapse OR child is a keyless VIS entry |
| 2 | Key preservation | Every inserted key not yet removed is reachable from root |
| 3 | No orphans | Every node in the tree is reachable from root |
| 4 | Correct chain identification | Chain membership identifies the longest sequence of single-child I4s above a leaf |
| 5 | Correct collapse | `try_collapse_i4` fires exactly when prefix fits AND child is not keyless |

---

## Abstraction Choices

### Abstracted away (not modeled)

- Concurrency (sequential model — covered by Stages 2–11)
- Actual byte values (use abstract key segments)
- Inode type transitions (I4→I16→I48→I256) — model only I4 for chains, generic "inode" for branching
- Memory allocation/deallocation

### Must capture

- Key length variety (keys of 1, 2, 3 segments; each segment = 8 bytes of real key)
- Prefix capacity (7 bytes per node = modeled as capacity of 1 abstract unit per node)
- Shared prefix divergence (two keys sharing 0, 1, or 2 segments then diverging)
- Chain depth (0, 1, 2 levels)
- Collapse decision (prefix fits vs overflow)
- VIS entries (value-in-slot: leaf value stored directly in parent's child slot)

---

## State Space Design

Target: **<5000 distinct states, <30s TLC runtime**.

| Parameter | Choice | Rationale |
|-----------|--------|-----------|
| Keys | 5 abstract keys | Exercises all paths including prefix overflow |
| Key lengths | 1–5 segments | Exercises no-chain through multi-segment chains |
| Key alphabet | 2 symbols {1, 2} | Minimum for branching without explosion |
| Prefix capacity | 2 | Exercises both collapse-success (combined ≤ 2) and collapse-blocked (combined > 2) |
| Tree representation | `node_id → {prefix, children, is_vis}` | Minimal record per node |
| Operations | `Insert(key)`, `Remove(key)` non-deterministic | Full insert/remove interleaving |
| StateConstraint | nxt ≤ 8 (max 7 nodes) | Keeps state space at ~5000 |

**Actual TLC results:** 5,035 distinct states, depth 14, 1 second.

**Key set:** `{<<1>>, <<1,2>>, <<1,2,1>>, <<2,1>>, <<1,2,1,2,1>>}`
- `<<1>>`: length 1, VIS at branching node
- `<<1,2>>`: length 2, shares prefix with `<<1>>`
- `<<1,2,1>>`: length 3, extends `<<1,2>>`
- `<<2,1>>`: length 2, diverges at root from `<<1,...>>`
- `<<1,2,1,2,1>>`: length 5, creates prefix overflow scenario

---

## Key Variety Dimensions

Derived from test survey of the C++ implementation:

| Dimension | Values modeled | Exercises |
|-----------|---------------|-----------|
| Key length | {1, 2, 3} segments | no-chain, 1-chain, 2-chain |
| Shared prefix | {0, 1, 2} segments | immediate diverge, partial share, deep share |
| Prefix capacity | 1 unit per node | collapse fits vs overflow |
| Chain depth | {0, 1, 2} | no-chain, single-chain, multi-chain |
| VIS vs leaf | both | collapse blocked by keyless child |
| Collapse decision | fits / overflow | `try_collapse_i4` success / failure |

---

## Invariant Definitions

```
WellFormed ==
  ∀ node ∈ tree:
    node.child_count = 1 =>
      \/ node.prefix_len + 1 + child.prefix_len > CAPACITY  \* overflow
      \/ child is VIS entry                                   \* keyless, cannot promote

KeyPreservation ==
  ∀ key ∈ inserted_keys \ removed_keys:
    Reachable(root, key)

NoOrphans ==
  ∀ node ∈ tree:
    node = root \/ ∃ parent ∈ tree: node ∈ parent.children

CorrectChainId ==
  ∀ leaf reachable from root:
    chain_above(leaf) = longest sequence of single-child I4s ending at leaf

CollapseCorrect ==
  After every remove that leaves a single-child I4:
    IF prefix_fits ∧ child_is_not_keyless THEN collapse happened
    ELSE single-child I4 persists
```

---

## Operations Modeled

### Insert(key)

1. Traverse tree following key segments
2. At divergence point: split existing node's prefix, create new branching node
3. Build chain for remaining key segments below divergence
4. Place leaf (or VIS entry if key is fully consumed at an existing node)

### Remove(key)

1. Find leaf/VIS entry
2. Identify chain above leaf (longest sequence of single-child I4s)
3. Cut chain: remove all chain nodes + leaf atomically
4. Update cut_point_parent (remove child entry)
5. If cut_point_parent becomes single-child I4: `try_collapse_i4`

### try_collapse_i4(node)

1. Check: node has exactly 1 child
2. Check: child is not a VIS/keyless entry
3. Check: `node.prefix_len + 1 + child.prefix_len ≤ CAPACITY`
4. If all pass: merge node into child (prepend prefix + dispatch to child's prefix)
5. Replace node with child in parent

---

## Expected Algorithmic Paths Exercised

With 3–4 keys of lengths {1, 2, 3}:

| Path | Trigger |
|------|---------|
| Insert into empty tree | First key |
| Insert creating chain | Second key shares prefix |
| Insert splitting chain | Third key diverges mid-chain |
| Remove with no chain | Short key, direct child of branching node |
| Remove with chain cut | Long key, chain above leaf |
| Remove triggering collapse | cut_point_parent becomes single-child, prefix fits |
| Remove — collapse blocked (overflow) | Combined prefix > CAPACITY |
| Remove — collapse blocked (VIS child) | Child is keyless VIS entry |
| Insert/remove VIS entry | Key fully consumed at branching node |

---

## Design Notes

- **PrefixCapacity = 2**: Exercises both collapse-success (combined ≤ 2) and collapse-blocked (combined > 2). With capacity=1, collapse would almost never fire; with capacity=3+, overflow would be unreachable with short keys.
- **Root handling**: Root is excluded from WellFormed — a single-child root is valid (no parent to merge into).
- **VIS entries**: Modeled as `is_vis` flag on nodes. A node with `is_vis=TRUE` and children has 2+ "occupants" and is not a chain candidate.
- **Collapse during insert**: When a prefix split shortens a node's prefix, the node may become newly collapsible. The model performs collapse eagerly after every structural change.

---

## Verified Coverage (from expert review)

| Algorithmic Path | Exercised? | Example Scenario |
|-----------------|------------|-----------------|
| Prefix overflow persistence | ✓ | Insert <<1,2,1,2,1>> then <<1>> |
| Collapse after remove | ✓ | Insert <<1>>, <<1,2,1>>, <<1,2>>, Remove <<1,2>> |
| Collapse after insert (prefix split) | ✓ | Insert <<1,2,1>> then <<2,1>> |
| VIS + child coexistence | ✓ | Insert <<1,2>> then <<1,2,1>> |
| Key-as-prefix-of-another | ✓ | <<1>> and <<1,2>> in either order |
| Chain split (diverge mid-chain) | ✓ | Insert <<1,2,1,2,1>> then <<1>> |
| Collapse blocked by overflow | ✓ | Node with prefix=<<>> + child prefix=<<2,1>> = 3 > 2 |
| Multi-level CleanUp recursion | ⚠️ Not exercised | Collapse prevents prerequisite chains from forming |

**Gap analysis**: Multi-level CleanUp is not reachable because collapse eagerly merges single-child non-VIS nodes, preventing the prerequisite chain structure. This is correct behavior — the same happens in the real algorithm. Multi-level chain cut under concurrency is covered by Stages 5, 9, and 10.
