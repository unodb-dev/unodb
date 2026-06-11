-------------------------- MODULE OLCDoubleCut -------------------------
(* Stage 8: Two removes on the same chain — at-most-one-cut.
   Models two removers competing to cut a single chain node. Only one
   should succeed in marking it obsolete.

   Fix M3: Added RDone actions so removers cycle back to idle, making
   AtMostOneCut non-trivial. Replaced NoDoubleFree with NoDoubleObsolete
   which verifies: once obsoleted, no one else can lock it. *)

EXTENDS Naturals

VARIABLES r1pc, r2pc, c_locked, c_obsolete, cut_count

vars == <<r1pc, r2pc, c_locked, c_obsolete, cut_count>>

TypeOK ==
    /\ r1pc \in {"idle", "locked", "cut", "done"}
    /\ r2pc \in {"idle", "locked", "cut", "done"}
    /\ c_locked \in BOOLEAN
    /\ c_obsolete \in BOOLEAN
    /\ cut_count \in Nat

Init ==
    /\ r1pc = "idle"
    /\ r2pc = "idle"
    /\ c_locked = FALSE
    /\ c_obsolete = FALSE
    /\ cut_count = 0

(* --- Remover 1 actions --- *)

R1Lock ==
    /\ r1pc = "idle"
    /\ ~c_locked
    /\ ~c_obsolete
    /\ c_locked' = TRUE
    /\ r1pc' = "locked"
    /\ UNCHANGED <<r2pc, c_obsolete, cut_count>>

R1Cut ==
    /\ r1pc = "locked"
    /\ c_obsolete' = TRUE
    /\ cut_count' = cut_count + 1
    /\ r1pc' = "cut"
    /\ UNCHANGED <<r2pc, c_locked>>

R1Unlock ==
    /\ r1pc = "cut"
    /\ c_locked' = FALSE
    /\ r1pc' = "done"
    /\ UNCHANGED <<r2pc, c_obsolete, cut_count>>

R1Done ==
    /\ r1pc = "done"
    /\ r1pc' = "idle"
    /\ UNCHANGED <<r2pc, c_locked, c_obsolete, cut_count>>

(* --- Remover 2 actions --- *)

R2Lock ==
    /\ r2pc = "idle"
    /\ ~c_locked
    /\ ~c_obsolete
    /\ c_locked' = TRUE
    /\ r2pc' = "locked"
    /\ UNCHANGED <<r1pc, c_obsolete, cut_count>>

R2Cut ==
    /\ r2pc = "locked"
    /\ c_obsolete' = TRUE
    /\ cut_count' = cut_count + 1
    /\ r2pc' = "cut"
    /\ UNCHANGED <<r1pc, c_locked>>

R2Unlock ==
    /\ r2pc = "cut"
    /\ c_locked' = FALSE
    /\ r2pc' = "done"
    /\ UNCHANGED <<r1pc, c_obsolete, cut_count>>

R2Done ==
    /\ r2pc = "done"
    /\ r2pc' = "idle"
    /\ UNCHANGED <<r1pc, c_locked, c_obsolete, cut_count>>

(* --- Step and Spec --- *)

Step ==
    \/ R1Lock \/ R1Cut \/ R1Unlock \/ R1Done
    \/ R2Lock \/ R2Cut \/ R2Unlock \/ R2Done

Spec == Init /\ [][Step]_vars /\ WF_vars(Step)

(* --- Invariants --- *)

MutualExclusion ==
    ~(r1pc \in {"locked", "cut"} /\ r2pc \in {"locked", "cut"})

AtMostOneCut == cut_count <= 1

NoDoubleObsolete ==
    /\ (r1pc = "locked" => ~c_obsolete)
    /\ (r2pc = "locked" => ~c_obsolete)

=========================================================================
