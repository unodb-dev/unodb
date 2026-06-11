--------------------------- MODULE OLCSlot --------------------------------
\* Stage 2: OLC protocol correctness for a single slot.
\*
\* One writer inserts/removes values (including 0) and pointers.
\* One reader reads (slot, bitmask), validates version, interprets.
\*
\* Verifies: reader never acts on inconsistent (slot, bitmask) pair,
\* and never dereferences a packed value as a pointer.

EXTENDS Integers, FiniteSets

CONSTANTS
    MaxVersion  \* Bound on version counter (e.g., 6)

VARIABLES
    \* Shared state (the single slot in the inode)
    slot,       \* 0=NULL, 1=nonzero_value, -1=pointer
    bitmask,    \* TRUE if slot holds a packed value
    version,    \* Even=unlocked, Odd=write-locked

    \* Writer state
    wpc,        \* idle|choose|lock|write_slot|write_bitmask|unlock
    wval,       \* What writer will store (-1, 0, or 1)
    wbm,        \* What bitmask writer will set

    \* Reader state
    rpc,        \* idle|read_slot|read_bitmask|validate|interpret|done
    rver,       \* Version captured at start
    rslot,      \* Local copy of slot
    rbitmask    \* Local copy of bitmask

vars == <<slot, bitmask, version, wpc, wval, wbm, rpc, rver, rslot, rbitmask>>

-----------------------------------------------------------------------------
\* INITIAL STATE

Init ==
    /\ slot = 0
    /\ bitmask = FALSE
    /\ version = 0
    /\ wpc = "idle"
    /\ wval = 0
    /\ wbm = FALSE
    /\ rpc = "idle"
    /\ rver = 0
    /\ rslot = 0
    /\ rbitmask = FALSE

-----------------------------------------------------------------------------
\* WRITER ACTIONS

\* Writer chooses to insert a value (0 or 1) or pointer (-1), or remove
WriterChoose ==
    /\ wpc = "idle"
    /\ version <= MaxVersion  \* bound version growth
    /\ \E v \in {-1, 0, 1}, b \in {TRUE, FALSE} :
        \* Valid operations:
        \*   (0, FALSE) = remove/clear slot
        \*   (0, TRUE)  = insert value 0 (the critical case!)
        \*   (1, TRUE)  = insert value 1 (nonzero value)
        \*   (-1, FALSE) = insert pointer
        /\ (v > 0 => b = TRUE)        \* positive values MUST have bitmask
        /\ (v = -1 => b = FALSE)      \* pointer never has bitmask
        /\ (v = 0 /\ b = FALSE => TRUE) \* remove is always valid
        /\ wval' = v
        /\ wbm' = b
    /\ wpc' = "lock"
    /\ UNCHANGED <<slot, bitmask, version, rpc, rver, rslot, rbitmask>>

\* Writer acquires lock
WriterLock ==
    /\ wpc = "lock"
    /\ version % 2 = 0         \* must be unlocked
    /\ version' = version + 1  \* odd = locked
    /\ wpc' = "write_slot"
    /\ UNCHANGED <<slot, bitmask, wval, wbm, rpc, rver, rslot, rbitmask>>

\* Writer stores slot value
WriterWriteSlot ==
    /\ wpc = "write_slot"
    /\ slot' = wval
    /\ wpc' = "write_bitmask"
    /\ UNCHANGED <<bitmask, version, wval, wbm, rpc, rver, rslot, rbitmask>>

\* Writer stores bitmask
WriterWriteBitmask ==
    /\ wpc = "write_bitmask"
    /\ bitmask' = wbm
    /\ wpc' = "unlock"
    /\ UNCHANGED <<slot, version, wval, wbm, rpc, rver, rslot, rbitmask>>

\* Writer releases lock
WriterUnlock ==
    /\ wpc = "unlock"
    /\ version' = version + 1  \* even = unlocked
    /\ wpc' = "idle"
    /\ UNCHANGED <<slot, bitmask, wval, wbm, rpc, rver, rslot, rbitmask>>

-----------------------------------------------------------------------------
\* READER ACTIONS

\* Reader starts: capture version
ReaderStart ==
    /\ rpc = "idle"
    /\ rver' = version
    /\ rpc' = "read_slot"
    /\ UNCHANGED <<slot, bitmask, version, wpc, wval, wbm, rslot, rbitmask>>

\* Reader reads slot
ReaderReadSlot ==
    /\ rpc = "read_slot"
    /\ rslot' = slot
    /\ rpc' = "read_bitmask"
    /\ UNCHANGED <<slot, bitmask, version, wpc, wval, wbm, rver, rbitmask>>

\* Reader reads bitmask
ReaderReadBitmask ==
    /\ rpc = "read_bitmask"
    /\ rbitmask' = bitmask
    /\ rpc' = "validate"
    /\ UNCHANGED <<slot, bitmask, version, wpc, wval, wbm, rver, rslot>>

\* Reader validates: version must be even AND unchanged
ReaderValidateOK ==
    /\ rpc = "validate"
    /\ version = rver
    /\ version % 2 = 0
    /\ rpc' = "interpret"
    /\ UNCHANGED <<slot, bitmask, version, wpc, wval, wbm, rver, rslot, rbitmask>>

\* Reader validates: FAILS → discard and retry
ReaderValidateFail ==
    /\ rpc = "validate"
    /\ (version # rver \/ version % 2 = 1)
    /\ rpc' = "idle"
    /\ UNCHANGED <<slot, bitmask, version, wpc, wval, wbm, rver, rslot, rbitmask>>

\* Reader interprets using CORRECT predicate
ReaderInterpret ==
    /\ rpc = "interpret"
    /\ rpc' = "done"
    /\ UNCHANGED <<slot, bitmask, version, wpc, wval, wbm, rver, rslot, rbitmask>>

\* Reader resets to idle
ReaderReset ==
    /\ rpc = "done"
    /\ rpc' = "idle"
    /\ UNCHANGED <<slot, bitmask, version, wpc, wval, wbm, rver, rslot, rbitmask>>

-----------------------------------------------------------------------------
\* NEXT STATE

Step ==
    \/ WriterChoose
    \/ WriterLock
    \/ WriterWriteSlot
    \/ WriterWriteBitmask
    \/ WriterUnlock
    \/ ReaderStart
    \/ ReaderReadSlot
    \/ ReaderReadBitmask
    \/ ReaderValidateOK
    \/ ReaderValidateFail
    \/ ReaderInterpret
    \/ ReaderReset

Spec == Init /\ [][Step]_vars

-----------------------------------------------------------------------------
\* INVARIANTS

\* If reader successfully validated and is interpreting/done,
\* its local (rslot, rbitmask) is a consistent committed pair.
\* Specifically: if rbitmask=TRUE, the slot holds a value (not a pointer).
\* If rbitmask=FALSE and rslot#0, it's a pointer.
SnapshotConsistency ==
    rpc \in {"interpret", "done"} =>
        /\ (rbitmask = TRUE => rslot >= 0)   \* value slots hold values (>=0)
        /\ (rslot = -1 => rbitmask = FALSE)  \* pointer slots have bitmask=FALSE

\* Reader never dereferences a value as a pointer.
\* A dereference would happen if: rslot#0 AND rbitmask=FALSE (thinks it's a ptr)
\* but actually it's a value. With correct protocol, this can't happen.
NoDereferenceValue ==
    rpc \in {"interpret", "done"} =>
        ~(rslot > 0 /\ rbitmask = FALSE)
        \* rslot>0 means nonzero value; if bitmask=FALSE reader would think it's a ptr

\* Writer only modifies shared state when version is odd (locked)
WriterProtocol ==
    wpc \in {"write_slot", "write_bitmask"} => version % 2 = 1

\* Shared state consistency: when unlocked, slot and bitmask agree
UnlockedConsistency ==
    version % 2 = 0 =>
        /\ (bitmask = TRUE => slot >= 0)
        /\ (slot = -1 => bitmask = FALSE)

=============================================================================
