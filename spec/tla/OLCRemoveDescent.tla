--------------------------- MODULE OLCRemoveDescent -----------------------
\* H10: Remove descent slot-content safety.
\*
\* Parameterized by VIS (can_eliminate_leaf):
\*   VIS=FALSE: slots hold PTR or NULL only. No bitmask.
\*   VIS=TRUE:  slots hold PTR, VIS_ZERO, or NULL. Bitmask distinguishes.
\*
\* Models one Reader (remove traversal) and one Writer (concurrent remove).
\*
\* Reader protocol (remove_or_choose_subtree):
\*   1. find_child: read key_index → if mapped, return slot address
\*   2. load: read slot content (atomic)
\*   3. validate: check parent version unchanged
\*   4. dispatch: if VIS ∧ bitmask → VIS path; else → dereference as pointer
\*
\* Writer protocol:
\*   1. lock (version goes odd)
\*   2. clear: set slot=NULL, bitmask=FALSE, key_index=FALSE
\*   3. unlock (version goes even, incremented)
\*
\* Safety: reader never dereferences unless r_child = "PTR".

EXTENDS Integers

CONSTANTS
    MaxVersion,
    VIS         \* TRUE or FALSE — models can_eliminate_leaf

VARIABLES
    \* Shared state (parent inode slot)
    key_index,  \* TRUE=find_child would succeed, FALSE=key absent
    slot,       \* "PTR" | "VIS_ZERO" | "NULL" (VIS_ZERO only when VIS=TRUE)
    bitmask,    \* TRUE=slot is packed value (only meaningful when VIS=TRUE)
    version,    \* even=unlocked, odd=write-locked

    \* Reader state
    rpc,
    r_ver,      \* captured version
    r_found,    \* find_child result
    r_child,    \* loaded slot content
    r_bitmask,  \* loaded bitmask

    \* Writer state
    wpc

vars == <<key_index, slot, bitmask, version, rpc, r_ver, r_found, r_child, r_bitmask, wpc>>

-----------------------------------------------------------------------------
\* Initial states depend on VIS
InitSlots == IF VIS THEN {"PTR", "VIS_ZERO"} ELSE {"PTR"}

Init ==
    /\ key_index = TRUE
    /\ slot \in InitSlots
    /\ bitmask = (VIS /\ slot = "VIS_ZERO")
    /\ version = 0
    /\ rpc = "idle"
    /\ r_ver = 0
    /\ r_found = FALSE
    /\ r_child = "NULL"
    /\ r_bitmask = FALSE
    /\ wpc = "idle"

-----------------------------------------------------------------------------
\* READER

RFind ==
    /\ rpc = "idle"
    /\ version % 2 = 0
    /\ r_ver' = version
    /\ r_found' = key_index
    /\ rpc' = "check_find"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_child, r_bitmask, wpc>>

RCheckFind ==
    /\ rpc = "check_find"
    /\ IF r_found
       THEN rpc' = "load"
       ELSE rpc' = "done"       \* key not found
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child, r_bitmask, wpc>>

RLoad ==
    /\ rpc = "load"
    /\ r_child' = slot
    /\ rpc' = "validate"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_bitmask, wpc>>

RValidate ==
    /\ rpc = "validate"
    /\ IF version = r_ver /\ version % 2 = 0
       THEN rpc' = "dispatch"
       ELSE rpc' = "restart"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child, r_bitmask, wpc>>

RDispatch ==
    /\ rpc = "dispatch"
    /\ IF VIS
       THEN \* Read bitmask, branch on it
            /\ r_bitmask' = bitmask
            /\ IF bitmask
               THEN rpc' = "vis_remove"
               ELSE rpc' = "check_child"
       ELSE \* No VIS — no unvalidated read, safe to dereference
            /\ r_bitmask' = FALSE
            /\ rpc' = "deref"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child, wpc>>

\* Guard: re-validate version before dereferencing child as pointer.
\* Models: if (!node_critical_section.check()) return {};
\* If version changed since first check, the bitmask may be stale.
RCheckChild ==
    /\ rpc = "check_child"
    /\ IF version = r_ver /\ version % 2 = 0
       THEN rpc' = "deref"
       ELSE rpc' = "restart"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child, r_bitmask, wpc>>

RDeref ==
    /\ rpc = "deref"
    /\ rpc' = "done"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child, r_bitmask, wpc>>

RVisRemove ==
    /\ rpc = "vis_remove"
    /\ rpc' = "done"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child, r_bitmask, wpc>>

RRestart ==
    /\ rpc = "restart"
    /\ rpc' = "idle"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child, r_bitmask, wpc>>

RDone ==
    /\ rpc = "done"
    /\ UNCHANGED vars

-----------------------------------------------------------------------------
\* WRITER

WLock ==
    /\ wpc = "idle"
    /\ version % 2 = 0
    /\ version <= MaxVersion
    /\ version' = version + 1
    /\ wpc' = "clear"
    /\ UNCHANGED <<key_index, slot, bitmask, rpc, r_ver, r_found, r_child, r_bitmask>>

WClear ==
    /\ wpc = "clear"
    /\ slot' = "NULL"
    /\ bitmask' = FALSE
    /\ key_index' = FALSE
    /\ wpc' = "unlock"
    /\ UNCHANGED <<version, rpc, r_ver, r_found, r_child, r_bitmask>>

WUnlock ==
    /\ wpc = "unlock"
    /\ version' = version + 1
    /\ wpc' = "idle"
    /\ UNCHANGED <<key_index, slot, bitmask, rpc, r_ver, r_found, r_child, r_bitmask>>

-----------------------------------------------------------------------------
Step ==
    \/ RFind \/ RCheckFind \/ RLoad \/ RValidate \/ RDispatch
    \/ RCheckChild \/ RDeref \/ RVisRemove \/ RRestart \/ RDone
    \/ WLock \/ WClear \/ WUnlock

Spec == Init /\ [][Step]_vars

StateConstraint == version <= MaxVersion

-----------------------------------------------------------------------------
\* SAFETY

\* Reader only dereferences when it loaded a real pointer.
NoDereferenceNonPointer ==
    rpc = "deref" => r_child = "PTR"

\* VIS path only taken when slot held a packed value.
VisPathCorrectness ==
    rpc = "vis_remove" => r_child = "VIS_ZERO"

TypeOK ==
    /\ key_index \in {TRUE, FALSE}
    /\ slot \in {"PTR", "VIS_ZERO", "NULL"}
    /\ bitmask \in {TRUE, FALSE}
    /\ version \in 0..MaxVersion+2
    /\ rpc \in {"idle", "check_find", "load", "validate", "dispatch",
                "check_child", "deref", "vis_remove", "done", "restart"}
    /\ r_ver \in 0..MaxVersion+2
    /\ r_found \in {TRUE, FALSE}
    /\ r_child \in {"PTR", "VIS_ZERO", "NULL"}
    /\ r_bitmask \in {TRUE, FALSE}
    /\ wpc \in {"idle", "lock", "clear", "unlock"}

=============================================================================
