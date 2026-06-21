--------------------------- MODULE OLCIteratorVIS ----------------------------
(* Phase 3: VIS (Value-In-Slot) TOCTOU model.

   Micro-model of the critical section where an iterator reads a packed
   leaf (VIS) from a parent inode. The packed leaf has NO version lock —
   the parent's version protects it.

   Geometry: One inode P with 3 child slots:
     Slot 1: regular leaf L1
     Slot 2: VIS value V2 (mutable — writer can change it)
     Slot 3: regular leaf L3

   Forward scan: L1, V2, L3.

   The iterator:
     1. Arrives at P with version snapshot (i_snap = p_ver)
     2. Delivers L1
     3. check1: validate p_ver == i_snap (before reading slot 2)
     4. Read slot 2: observe is_value_in_slot and get_child
     5. check2: validate p_ver == i_snap (after reading slot 2)
     6. Deliver slot 2 value
     7. Deliver L3, done

   The writer:
     1. Lock P (p_ver becomes odd)
     2. Mutate slot 2 (change VIS to different value)
     3. Unlock P (p_ver becomes next even)

   TOCTOU: Between check1 and check2, the writer can lock+mutate+unlock.
   check2 catches this (version changed). Bug: skip check2 → stale/garbage read.
*)

EXTENDS Naturals, Sequences

CONSTANTS
    MaxVersion,
    BugMode       \* "none" | "skip_check2"

VARIABLES
    p_ver,            \* Parent inode version (even=unlocked, odd=locked)
    slot2_value,      \* Current value in slot 2: "V2" | "V2prime"
    ipc,              \* Iterator PC
    i_snap,           \* Iterator's version snapshot of P
    i_read_val,       \* What iterator read from slot 2 (may be stale)
    visited,          \* Delivered values
    restarts,         \* Restart count
    wpc               \* Writer PC: idle | locked | mutated | done

vars == <<p_ver, slot2_value, ipc, i_snap, i_read_val, visited, restarts, wpc>>

\* --- Helpers ---

CheckOK == p_ver = i_snap /\ (p_ver % 2 = 0)
CanLock == p_ver % 2 = 0

\* All possible values that could appear in slot 2 (including garbage on havoc)
SlotValues == {"V2", "V2prime"}
AllValues == {"L1", "L3", "V2", "V2prime", "garbage"}

\* --- Initial state ---
Init ==
    /\ p_ver = 0
    /\ slot2_value = "V2"
    /\ ipc = "start"
    /\ i_snap = 0
    /\ i_read_val = "V2"
    /\ visited = <<>>
    /\ restarts = 0
    /\ wpc = "idle"

\* --- Iterator actions ---

\* Take snapshot of parent version, deliver L1
IterStart == ipc = "start" /\
    IF ~CanLock THEN
        /\ restarts' = restarts + 1
        /\ UNCHANGED <<p_ver, slot2_value, ipc, i_snap, i_read_val, visited, wpc>>
    ELSE
        /\ i_snap' = p_ver
        /\ visited' = <<"L1">>
        /\ ipc' = "check1"
        /\ UNCHANGED <<p_ver, slot2_value, i_read_val, restarts, wpc>>

\* Check1: validate parent version before reading VIS slot
IterCheck1 == ipc = "check1" /\
    IF ~CheckOK THEN
        \* Version changed — restart
        /\ ipc' = "start"
        /\ visited' = <<>>
        /\ restarts' = restarts + 1
        /\ UNCHANGED <<p_ver, slot2_value, i_snap, i_read_val, wpc>>
    ELSE
        /\ ipc' = "read_slot2"
        /\ UNCHANGED <<p_ver, slot2_value, i_snap, i_read_val, visited, restarts, wpc>>

\* Read slot 2: observe the current value
\* (This is the TOCTOU-vulnerable read — happens BETWEEN check1 and check2)
IterReadSlot2 == ipc = "read_slot2" /\
    /\ i_read_val' = slot2_value  \* reads CURRENT value (may differ from snapshot time)
    /\ ipc' = "check2"
    /\ UNCHANGED <<p_ver, slot2_value, i_snap, visited, restarts, wpc>>

\* Check2: validate parent version AFTER reading VIS slot
IterCheck2 == ipc = "check2" /\
    IF BugMode = "skip_check2" THEN
        \* BUG: skip the check — deliver whatever was read
        \* If version changed, the read was stale. Model with havoc.
        IF ~CheckOK THEN
            \* Stale read — havoc: could deliver any value
            \E v \in AllValues :
                /\ visited' = Append(visited, v)
                /\ ipc' = "deliver_L3"
                /\ UNCHANGED <<p_ver, slot2_value, i_snap, i_read_val, restarts, wpc>>
        ELSE
            \* Version still matches — read is valid even without check
            /\ visited' = Append(visited, i_read_val)
            /\ ipc' = "deliver_L3"
            /\ UNCHANGED <<p_ver, slot2_value, i_snap, i_read_val, restarts, wpc>>
    ELSE
        \* Correct code: validate version
        IF ~CheckOK THEN
            /\ ipc' = "start"
            /\ visited' = <<>>
            /\ restarts' = restarts + 1
            /\ UNCHANGED <<p_ver, slot2_value, i_snap, i_read_val, wpc>>
        ELSE
            \* Version matches — read is valid, deliver it
            /\ visited' = Append(visited, i_read_val)
            /\ ipc' = "deliver_L3"
            /\ UNCHANGED <<p_ver, slot2_value, i_snap, i_read_val, restarts, wpc>>

\* Deliver L3 and finish
IterDeliverL3 == ipc = "deliver_L3" /\
    /\ visited' = Append(visited, "L3")
    /\ ipc' = "done"
    /\ UNCHANGED <<p_ver, slot2_value, i_snap, i_read_val, restarts, wpc>>

IterDone == ipc = "done" /\ UNCHANGED vars

\* --- Writer actions ---

WriterLock == wpc = "idle" /\
    /\ p_ver % 2 = 0
    /\ p_ver + 1 <= MaxVersion
    /\ p_ver' = p_ver + 1
    /\ wpc' = "locked"
    /\ UNCHANGED <<slot2_value, ipc, i_snap, i_read_val, visited, restarts>>

WriterMutate == wpc = "locked" /\
    /\ slot2_value' = "V2prime"
    /\ wpc' = "mutated"
    /\ UNCHANGED <<p_ver, ipc, i_snap, i_read_val, visited, restarts>>

WriterUnlock == wpc = "mutated" /\
    /\ p_ver + 1 <= MaxVersion
    /\ p_ver' = p_ver + 1
    /\ wpc' = "done"
    /\ UNCHANGED <<slot2_value, ipc, i_snap, i_read_val, visited, restarts>>

WriterDone == wpc = "done" /\ UNCHANGED vars

\* --- Specification ---

Next ==
    \/ IterStart
    \/ IterCheck1
    \/ IterReadSlot2
    \/ IterCheck2
    \/ IterDeliverL3
    \/ IterDone
    \/ WriterLock
    \/ WriterMutate
    \/ WriterUnlock
    \/ WriterDone

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* --- State Constraint ---
StateConstraint == restarts <= 3

\* --- Invariants ---

TypeOK ==
    /\ p_ver \in 0..MaxVersion
    /\ slot2_value \in SlotValues
    /\ ipc \in {"start", "check1", "read_slot2", "check2", "deliver_L3", "done"}
    /\ i_snap \in 0..MaxVersion
    /\ i_read_val \in AllValues
    /\ visited \in Seq(AllValues)
    /\ restarts \in Nat
    /\ wpc \in {"idle", "locked", "mutated", "done"}

\* NoGarbage: the delivered slot2 value must be a real slot value, not garbage
NoGarbage ==
    Len(visited) >= 2 => visited[2] \in SlotValues

\* OrderPreservation: L1 is always first, L3 is always last
OrderPreservation ==
    /\ (Len(visited) >= 1 => visited[1] = "L1")
    /\ (Len(visited) >= 3 => visited[3] = "L3")

\* Completeness: when done, exactly 3 values delivered
Completeness ==
    ipc = "done" => Len(visited) = 3

\* VISConsistency: the slot2 value delivered is one that actually existed
\* in slot2 at some point (either "V2" or "V2prime")
VISConsistency ==
    Len(visited) >= 2 => visited[2] \in {"V2", "V2prime"}

\* Canary: slot2 is never mutated (should FAIL — proves writer interleaves)
SlotNeverMutated ==
    slot2_value = "V2"

=============================================================================
