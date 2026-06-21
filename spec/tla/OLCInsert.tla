--------------------------- MODULE OLCInsert ---------------------------
(* Stage 3: Two writers racing for the same node — mutual exclusion via OLC.
   Models the version-based optimistic lock protocol where writers must
   acquire exclusive access (odd version) before modifying node state.

   Fix C2: WWrite split into WWriteSlot + WWriteCount to model non-atomic
   count maintenance. CountConsistency only checked when version is even
   (unlocked state). *)

EXTENDS Naturals, FiniteSets

CONSTANTS NumSlots

Slots == 1..NumSlots

VARIABLES version, slots, count,
          w1pc, w1target,
          w2pc, w2target

vars == <<version, slots, count, w1pc, w1target, w2pc, w2target>>

TypeOK ==
    /\ version \in Nat
    /\ slots \in [Slots -> {0, 1}]
    /\ count \in 0..NumSlots
    /\ w1pc \in {"idle", "locked", "write_slot", "write_count", "unlock"}
    /\ w2pc \in {"idle", "locked", "write_slot", "write_count", "unlock"}
    /\ w1target \in Slots
    /\ w2target \in Slots

Init ==
    /\ version = 0
    /\ slots = [i \in Slots |-> 0]
    /\ count = 0
    /\ w1pc = "idle"
    /\ w2pc = "idle"
    /\ w1target = 1
    /\ w2target = 1

(* --- Writer 1 actions --- *)

W1Lock ==
    /\ w1pc = "idle"
    /\ version % 2 = 0          \* not already write-locked
    /\ \E t \in Slots : slots[t] = 0 /\ w1target' = t
    /\ version' = version + 1   \* odd = locked
    /\ w1pc' = "locked"
    /\ UNCHANGED <<slots, count, w2pc, w2target>>

W1WriteSlot ==
    /\ w1pc = "locked"
    /\ slots' = [slots EXCEPT ![w1target] = 1]
    /\ w1pc' = "write_slot"
    /\ UNCHANGED <<version, count, w1target, w2pc, w2target>>

W1WriteCount ==
    /\ w1pc = "write_slot"
    /\ count' = count + 1
    /\ w1pc' = "write_count"
    /\ UNCHANGED <<version, slots, w1target, w2pc, w2target>>

W1Unlock ==
    /\ w1pc = "write_count"
    /\ version' = version + 1   \* even = unlocked
    /\ w1pc' = "idle"
    /\ UNCHANGED <<slots, count, w1target, w2pc, w2target>>

(* --- Writer 2 actions --- *)

W2Lock ==
    /\ w2pc = "idle"
    /\ version % 2 = 0
    /\ \E t \in Slots : slots[t] = 0 /\ w2target' = t
    /\ version' = version + 1
    /\ w2pc' = "locked"
    /\ UNCHANGED <<slots, count, w1pc, w1target>>

W2WriteSlot ==
    /\ w2pc = "locked"
    /\ slots' = [slots EXCEPT ![w2target] = 1]
    /\ w2pc' = "write_slot"
    /\ UNCHANGED <<version, count, w1pc, w1target, w2target>>

W2WriteCount ==
    /\ w2pc = "write_slot"
    /\ count' = count + 1
    /\ w2pc' = "write_count"
    /\ UNCHANGED <<version, slots, w1pc, w1target, w2target>>

W2Unlock ==
    /\ w2pc = "write_count"
    /\ version' = version + 1
    /\ w2pc' = "idle"
    /\ UNCHANGED <<slots, count, w1pc, w1target, w2target>>

(* --- Step and Spec --- *)

Step ==
    \/ W1Lock \/ W1WriteSlot \/ W1WriteCount \/ W1Unlock
    \/ W2Lock \/ W2WriteSlot \/ W2WriteCount \/ W2Unlock

Spec == Init /\ [][Step]_vars /\ WF_vars(Step)

(* --- Invariants --- *)

MutualExclusion ==
    ~(w1pc \in {"locked", "write_slot", "write_count"} /\
      w2pc \in {"locked", "write_slot", "write_count"})

CountConsistency ==
    version % 2 = 0 => count = Cardinality({i \in Slots : slots[i] = 1})

=========================================================================
