------------------------ MODULE OLCUpsertErase_BugDemo -----------------------
(* Stage 9 BUG DEMO: Same as OLCUpsertErase but the upserter skips version
   validation — it always proceeds to erase regardless of whether the value
   changed. This MUST violate CASSafety, proving the check is necessary. *)

EXTENDS Naturals

CONSTANTS Values, Empty

ASSUME Empty \notin Values

VARIABLES value, version, upc, u_observed, u_obs_ver, wpc, w_new_val

vars == <<value, version, upc, u_observed, u_obs_ver, wpc, w_new_val>>

TypeOK ==
    /\ value \in Values \cup {Empty}
    /\ version \in Nat
    /\ upc \in {"idle", "observed", "released", "retraversing",
                "write_locked", "erased", "done"}
    /\ u_observed \in Values \cup {Empty}
    /\ u_obs_ver \in Nat
    /\ wpc \in {"idle", "locked", "written", "done"}
    /\ w_new_val \in Values

Init ==
    /\ value \in Values
    /\ version = 0
    /\ upc = "idle"
    /\ u_observed = Empty
    /\ u_obs_ver = 0
    /\ wpc = "idle"
    /\ w_new_val \in Values

(* --- Upserter (BUG: no version check) --- *)

UObserve ==
    /\ upc = "idle"
    /\ version % 2 = 0
    /\ value # Empty
    /\ u_observed' = value
    /\ u_obs_ver' = version
    /\ upc' = "observed"
    /\ UNCHANGED <<value, version, wpc, w_new_val>>

URelease ==
    /\ upc = "observed"
    /\ upc' = "released"
    /\ UNCHANGED <<value, version, u_observed, u_obs_ver, wpc, w_new_val>>

URetraverse ==
    /\ upc = "released"
    /\ upc' = "retraversing"
    /\ UNCHANGED <<value, version, u_observed, u_obs_ver, wpc, w_new_val>>

(* BUG: Always acquires write lock without checking version match. *)
UValidate_BUGGY ==
    /\ upc = "retraversing"
    /\ version % 2 = 0
    /\ version' = version + 1
    /\ upc' = "write_locked"
    /\ UNCHANGED <<value, u_observed, u_obs_ver, wpc, w_new_val>>

UErase ==
    /\ upc = "write_locked"
    /\ value' = Empty
    /\ upc' = "erased"
    /\ UNCHANGED <<version, u_observed, u_obs_ver, wpc, w_new_val>>

UUnlock ==
    /\ upc = "erased"
    /\ version' = version + 1
    /\ upc' = "done"
    /\ UNCHANGED <<value, u_observed, u_obs_ver, wpc, w_new_val>>

UDone ==
    /\ upc = "done"
    /\ upc' = "idle"
    /\ UNCHANGED <<value, version, u_observed, u_obs_ver, wpc, w_new_val>>

(* --- Concurrent writer (same as correct spec) --- *)

WLock ==
    /\ wpc = "idle"
    /\ version % 2 = 0
    /\ version' = version + 1
    /\ wpc' = "locked"
    /\ UNCHANGED <<value, upc, u_observed, u_obs_ver, w_new_val>>

WWrite ==
    /\ wpc = "locked"
    /\ value' = w_new_val
    /\ wpc' = "written"
    /\ UNCHANGED <<version, upc, u_observed, u_obs_ver, w_new_val>>

WUnlock ==
    /\ wpc = "written"
    /\ version' = version + 1
    /\ wpc' = "done"
    /\ UNCHANGED <<value, upc, u_observed, u_obs_ver, w_new_val>>

WDone ==
    /\ wpc = "done"
    /\ w_new_val' \in Values
    /\ wpc' = "idle"
    /\ UNCHANGED <<value, version, upc, u_observed, u_obs_ver>>

Step ==
    \/ UObserve \/ URelease \/ URetraverse \/ UValidate_BUGGY
    \/ UErase \/ UUnlock \/ UDone
    \/ WLock \/ WWrite \/ WUnlock \/ WDone

Spec == Init /\ [][Step]_vars /\ WF_vars(Step)

(* This MUST be violated — proving the version check is necessary. *)
CASSafety ==
    upc = "write_locked" => value = u_observed

StateConstraint == version <= 10

=========================================================================
