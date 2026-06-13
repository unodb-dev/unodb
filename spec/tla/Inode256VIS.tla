--------------------------- MODULE Inode256VIS ---------------------------
\* TLA+ specification for inode256 value-in-slot (VIS) correctness.
\*
\* Models a single inode256 with N slots. Each slot is either:
\*   - Empty (no child, no value)
\*   - Pointer (holds a child node reference)
\*   - Value (holds a packed user value, bitmask bit set)
\*
\* Key invariant: every scan for "occupied" slots must find exactly
\* the union of pointer-slots and value-slots. With bitmask-only
\* encoding, pack(0) = 0 = NULL, so `slot[i] # 0` is INSUFFICIENT.
\*
\* Domain: N=4 slots, values in {0,1,2}, pointers in {-1,-2,-3,-4}.
\* NULL = 0. Pointers are negative (always non-zero). Values are >= 0.

EXTENDS Integers, FiniteSets

CONSTANTS
    N,          \* Number of slots (4 for model checking)
    Values      \* Set of user values (e.g., {0, 1, 2})

VARIABLES
    slot,       \* slot[i] \in Int (0=NULL, <0=pointer, >=0=packed value)
    bitmask,    \* bitmask[i] \in BOOLEAN (TRUE = value-in-slot)
    hasPtr      \* hasPtr[i] \in BOOLEAN (TRUE = slot holds a pointer)

NULL == 0
Pointers == {-1, -2, -3, -4}
Pack(v) == v    \* Identity encoding. pack(0) = 0 = NULL.
Slots == 1..N
vars == <<slot, bitmask, hasPtr>>

-----------------------------------------------------------------------------
\* PREDICATES

\* Ground truth: a slot is occupied if it holds a pointer OR a value.
Occupied(i) == hasPtr[i] \/ bitmask[i]

OccupiedSet == {i \in Slots : Occupied(i)}

\* BUGGY: nullptr check only. Misses value=0 slots.
BuggyOccupied(i) == slot[i] # NULL

BuggyOccupiedSet == {i \in Slots : BuggyOccupied(i)}

\* CORRECT: nullptr OR bitmask.
CorrectOccupied(i) == slot[i] # NULL \/ bitmask[i]

CorrectOccupiedSet == {i \in Slots : CorrectOccupied(i)}

-----------------------------------------------------------------------------
\* INITIAL STATE

Init ==
    /\ slot = [i \in Slots |-> NULL]
    /\ bitmask = [i \in Slots |-> FALSE]
    /\ hasPtr = [i \in Slots |-> FALSE]

-----------------------------------------------------------------------------
\* OPERATIONS

InsertValue(i, v) ==
    /\ ~Occupied(i)
    /\ slot' = [slot EXCEPT ![i] = Pack(v)]
    /\ bitmask' = [bitmask EXCEPT ![i] = TRUE]
    /\ hasPtr' = [hasPtr EXCEPT ![i] = FALSE]

InsertPointer(i, p) ==
    /\ ~Occupied(i)
    /\ p \in Pointers
    /\ slot' = [slot EXCEPT ![i] = p]
    /\ bitmask' = [bitmask EXCEPT ![i] = FALSE]
    /\ hasPtr' = [hasPtr EXCEPT ![i] = TRUE]

Remove(i) ==
    /\ Occupied(i)
    /\ slot' = [slot EXCEPT ![i] = NULL]
    /\ bitmask' = [bitmask EXCEPT ![i] = FALSE]
    /\ hasPtr' = [hasPtr EXCEPT ![i] = FALSE]

-----------------------------------------------------------------------------
\* NEXT STATE

Step ==
    \/ \E i \in Slots, v \in Values : InsertValue(i, v)
    \/ \E i \in Slots, p \in Pointers : InsertPointer(i, p)
    \/ \E i \in Slots : Remove(i)

Spec == Init /\ [][Step]_vars

-----------------------------------------------------------------------------
\* INVARIANTS

\* The correct check (slot#NULL \/ bitmask) matches ground truth.
CorrectScanInvariant == CorrectOccupiedSet = OccupiedSet

\* The buggy check (slot#NULL only) should be VIOLATED by TLC.
BuggyScanInvariant == BuggyOccupiedSet = OccupiedSet

\* Structural consistency of the state.
BitmaskConsistency ==
    \A i \in Slots :
        /\ (bitmask[i] => ~hasPtr[i])
        /\ (hasPtr[i] => ~bitmask[i])
        /\ (hasPtr[i] => slot[i] \in Pointers)
        /\ (~Occupied(i) => slot[i] = NULL)

\* Iterator safety: value slots are never pointers.
IteratorSafety ==
    \A i \in Slots : (bitmask[i] => ~hasPtr[i])

=============================================================================
