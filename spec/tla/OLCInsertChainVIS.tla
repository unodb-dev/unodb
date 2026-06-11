--------------------------- MODULE OLCInsertChainVIS ----------------------
\* Stage 7: Insert chain protocol with transient VIS bitmask state.
\*
\* When inserting a chain into a non-full node, the writer does:
\*   1. add_to_nonfull(pack_value) → sets value_bit (slot looks like VIS)
\*   2. overwrite slot with chain pointer
\*   3. clear_value_bit (slot is now a pointer)
\*
\* All 3 steps happen under ONE write lock. A concurrent reader that
\* validates successfully must never see the transient state.
\*
\* Verifies: reader never sees (bitmask=TRUE, slot=chain_ptr).

EXTENDS Integers

CONSTANTS
    MaxVersion

VARIABLES
    \* Shared state
    slot,       \* 0=empty, 1=packed_value, -1=chain_ptr
    bitmask,    \* TRUE if slot holds a value
    version,    \* Even=unlocked, Odd=locked

    \* Writer state (chain insert protocol)
    wpc,        \* idle|lock|set_bit|write_chain|clear_bit|unlock|done

    \* Reader state
    rpc,        \* idle|read_slot|read_bitmask|validate|done
    rver,       \* Cached version
    rslot,      \* Local slot copy
    rbitmask    \* Local bitmask copy

vars == <<slot, bitmask, version, wpc, rpc, rver, rslot, rbitmask>>

-----------------------------------------------------------------------------
Init ==
    /\ slot = 0
    /\ bitmask = FALSE
    /\ version = 0
    /\ wpc = "idle"
    /\ rpc = "idle"
    /\ rver = 0
    /\ rslot = 0
    /\ rbitmask = FALSE

-----------------------------------------------------------------------------
\* WRITER: chain insert protocol (4 steps under one lock)

WriterLock ==
    /\ wpc = "idle"
    /\ version % 2 = 0
    /\ version <= MaxVersion
    /\ slot = 0              \* slot must be empty
    /\ version' = version + 1
    /\ wpc' = "set_bit"
    /\ UNCHANGED <<slot, bitmask, rpc, rver, rslot, rbitmask>>

WriterSetBit ==
    /\ wpc = "set_bit"
    /\ slot' = 1             \* store packed value temporarily
    /\ bitmask' = TRUE       \* mark as VIS
    /\ wpc' = "write_chain"
    /\ UNCHANGED <<version, rpc, rver, rslot, rbitmask>>

WriterWriteChain ==
    /\ wpc = "write_chain"
    /\ slot' = -1            \* overwrite with chain pointer
    /\ wpc' = "clear_bit"
    /\ UNCHANGED <<bitmask, version, rpc, rver, rslot, rbitmask>>

WriterClearBit ==
    /\ wpc = "clear_bit"
    /\ bitmask' = FALSE      \* slot is now a pointer, not a value
    /\ wpc' = "unlock"
    /\ UNCHANGED <<slot, version, rpc, rver, rslot, rbitmask>>

WriterUnlock ==
    /\ wpc = "unlock"
    /\ version' = version + 1
    /\ wpc' = "done"
    /\ UNCHANGED <<slot, bitmask, rpc, rver, rslot, rbitmask>>

-----------------------------------------------------------------------------
\* READER: reads slot + bitmask, validates, interprets

ReaderStart ==
    /\ rpc = "idle"
    /\ rver' = version
    /\ rpc' = "read_slot"
    /\ UNCHANGED <<slot, bitmask, version, wpc, rslot, rbitmask>>

ReaderReadSlot ==
    /\ rpc = "read_slot"
    /\ rslot' = slot
    /\ rpc' = "read_bitmask"
    /\ UNCHANGED <<slot, bitmask, version, wpc, rver, rbitmask>>

ReaderReadBitmask ==
    /\ rpc = "read_bitmask"
    /\ rbitmask' = bitmask
    /\ rpc' = "validate"
    /\ UNCHANGED <<slot, bitmask, version, wpc, rver, rslot>>

ReaderValidateOK ==
    /\ rpc = "validate"
    /\ version = rver
    /\ version % 2 = 0
    /\ rpc' = "done"
    /\ UNCHANGED <<slot, bitmask, version, wpc, rver, rslot, rbitmask>>

ReaderValidateFail ==
    /\ rpc = "validate"
    /\ (version # rver \/ version % 2 = 1)
    /\ rpc' = "idle"
    /\ UNCHANGED <<slot, bitmask, version, wpc, rver, rslot, rbitmask>>

ReaderReset ==
    /\ rpc = "done"
    /\ rpc' = "idle"
    /\ UNCHANGED <<slot, bitmask, version, wpc, rver, rslot, rbitmask>>

-----------------------------------------------------------------------------
Step ==
    \/ WriterLock \/ WriterSetBit \/ WriterWriteChain
    \/ WriterClearBit \/ WriterUnlock
    \/ ReaderStart \/ ReaderReadSlot \/ ReaderReadBitmask
    \/ ReaderValidateOK \/ ReaderValidateFail \/ ReaderReset

Spec == Init /\ [][Step]_vars

StateConstraint == version <= MaxVersion

-----------------------------------------------------------------------------
\* INVARIANTS

\* CRITICAL: Reader never sees transient state (bitmask=TRUE, slot=chain_ptr)
NoTransientVisible ==
    rpc = "done" => ~(rbitmask = TRUE /\ rslot = -1)

\* Reader's validated snapshot is consistent
SnapshotConsistency ==
    rpc = "done" =>
        /\ (rbitmask = TRUE => rslot >= 0)   \* value slots hold values
        /\ (rslot = -1 => rbitmask = FALSE)  \* chain ptr has no bitmask

=============================================================================
