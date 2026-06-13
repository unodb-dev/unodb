------------------------- MODULE OLCUpsertErase -------------------------
(* Stage 9: Upsert-erase CAS protocol — version-validated positioned erase.

   Models the critical interleaving: an upserter finds a duplicate key,
   the lambda observes the current value and returns "erase". The upserter
   must release its optimistic read locks, re-traverse, and remove the
   value — but ONLY if the value hasn't changed since the lambda observed it.

   Hazard H9: Between releasing RCS and re-acquiring write locks, a
   concurrent writer can change the value OR remove the key entirely.
   The version counter must detect this and force the upserter to retry.

   Key CAS invariant: if the upserter successfully erases, the value at
   the moment of erasure == the value the lambda observed.

   Processes:
   - 1 upserter: observe → lambda(erase) → release → retraverse → validate → remove|retry
   - 1 concurrent writer: can change the value, remove the key, or re-insert

   Round 3 extension: WErase action models concurrent removal (key absent
   on re-traverse). Upserter detects key-absent and restarts (takes insert
   path on next iteration since key is gone).
*)

EXTENDS Naturals

CONSTANTS Values, Empty
  (* Values = {V1, V2} — the domain of possible values at this key.
     Empty — sentinel indicating the key has been erased. Must not be in Values. *)

ASSUME Empty \notin Values

VARIABLES
    (* --- Shared node state --- *)
    value,          \* current value at the key (or Empty if erased)
    version,        \* OLC version counter (even=unlocked, odd=write-locked)

    (* --- Upserter state --- *)
    upc,            \* upserter program counter
    u_observed,     \* value the lambda saw
    u_obs_ver,      \* version when the lambda observed the value

    (* --- Concurrent writer state --- *)
    wpc,            \* writer program counter
    w_new_val       \* value the writer will install

vars == <<value, version, upc, u_observed, u_obs_ver, wpc, w_new_val>>

TypeOK ==
    /\ value \in Values \cup {Empty}
    /\ version \in Nat
    /\ upc \in {"idle", "observed", "released", "retraversing",
                "validate", "write_locked", "erased", "done",
                "key_absent", "insert_unlock"}
    /\ u_observed \in Values \cup {Empty}
    /\ u_obs_ver \in Nat
    /\ wpc \in {"idle", "locked", "written", "erased_w", "done"}
    /\ w_new_val \in Values

Init ==
    /\ value \in Values          \* key exists with some value
    /\ version = 0
    /\ upc = "idle"
    /\ u_observed = Empty
    /\ u_obs_ver = 0
    /\ wpc = "idle"
    /\ w_new_val \in Values

(* ================================================================== *)
(* --- Upserter actions ---                                           *)
(* ================================================================== *)

(* Step 1: Upserter finds the key and observes value under RCS.
   Precondition: node is not write-locked (version is even) and key exists. *)
UObserve ==
    /\ upc = "idle"
    /\ version % 2 = 0
    /\ value # Empty             \* key must exist to observe
    /\ u_observed' = value
    /\ u_obs_ver' = version
    /\ upc' = "observed"
    /\ UNCHANGED <<value, version, wpc, w_new_val>>

(* Step 1b: Upserter finds key ABSENT — takes insert path.
   This happens on retry after a concurrent erase removed the key. *)
UObserveAbsent ==
    /\ upc = "idle"
    /\ version % 2 = 0
    /\ value = Empty             \* key is gone
    /\ upc' = "key_absent"
    /\ UNCHANGED <<value, version, u_observed, u_obs_ver, wpc, w_new_val>>

(* Step 1c: Key absent → upserter takes insert path (inserts default value).
   Models the insert path of upsert when key is not found.
   Lock, write, unlock — simplified to two steps. *)
UInsertOnAbsent ==
    /\ upc = "key_absent"
    /\ version % 2 = 0
    /\ version' = version + 1   \* acquire write lock
    /\ value' = w_new_val       \* insert some value (stand-in for the upsert's v param)
    /\ upc' = "insert_unlock"
    /\ UNCHANGED <<u_observed, u_obs_ver, wpc, w_new_val>>

(* Step 1d: Unlock after insert. *)
UInsertUnlock ==
    /\ upc = "insert_unlock"
    /\ version' = version + 1   \* even = unlocked
    /\ upc' = "done"
    /\ UNCHANGED <<value, u_observed, u_obs_ver, wpc, w_new_val>>

(* Step 2: Lambda returns "erase". Upserter releases RCS. *)
URelease ==
    /\ upc = "observed"
    /\ upc' = "released"
    /\ UNCHANGED <<value, version, u_observed, u_obs_ver, wpc, w_new_val>>

(* Step 3: Re-traverse from root. Arrives at the node. *)
URetraverse ==
    /\ upc = "released"
    /\ upc' = "retraversing"
    /\ UNCHANGED <<value, version, u_observed, u_obs_ver, wpc, w_new_val>>

(* Step 4: Validate version. Key still exists and version matches → proceed.
   Version mismatch → retry from idle. Key absent → retry from idle. *)
UValidate ==
    /\ upc = "retraversing"
    /\ version % 2 = 0          \* can acquire write lock
    /\ value # Empty             \* key still exists
    /\ IF u_obs_ver = version
       THEN                      \* Version matches — value unchanged, proceed
            /\ version' = version + 1   \* acquire write lock (odd)
            /\ upc' = "write_locked"
       ELSE                      \* Version mismatch — value may have changed, retry
            /\ upc' = "idle"
            /\ UNCHANGED version
    /\ UNCHANGED <<value, u_observed, u_obs_ver, wpc, w_new_val>>

(* Step 4b: Upserter arrives but node is write-locked. Restart. *)
URestartOnLocked ==
    /\ upc = "retraversing"
    /\ version % 2 = 1          \* write-locked by someone else
    /\ upc' = "idle"
    /\ UNCHANGED <<value, version, u_observed, u_obs_ver, wpc, w_new_val>>

(* Step 4c: Upserter arrives but key is ABSENT (concurrent erase).
   Restart from idle — next iteration will take insert path. *)
UKeyGone ==
    /\ upc = "retraversing"
    /\ version % 2 = 0
    /\ value = Empty             \* key was removed by concurrent writer
    /\ upc' = "idle"            \* restart — will hit UObserveAbsent next
    /\ UNCHANGED <<value, version, u_observed, u_obs_ver, wpc, w_new_val>>

(* Step 5: Erase the value (under write lock). *)
UErase ==
    /\ upc = "write_locked"
    /\ value' = Empty
    /\ upc' = "erased"
    /\ UNCHANGED <<version, u_observed, u_obs_ver, wpc, w_new_val>>

(* Step 6: Unlock. *)
UUnlock ==
    /\ upc = "erased"
    /\ version' = version + 1   \* even = unlocked
    /\ upc' = "done"
    /\ UNCHANGED <<value, u_observed, u_obs_ver, wpc, w_new_val>>

(* Step 7: Upserter cycles back to idle. *)
UDone ==
    /\ upc = "done"
    /\ upc' = "idle"
    /\ UNCHANGED <<value, version, u_observed, u_obs_ver, wpc, w_new_val>>

(* ================================================================== *)
(* --- Concurrent writer actions ---                                  *)
(*                                                                    *)
(* Models any concurrent mutation: update, remove, or re-insert.      *)
(* ================================================================== *)

(* Writer acquires write lock. *)
WLock ==
    /\ wpc = "idle"
    /\ version % 2 = 0
    /\ version' = version + 1
    /\ wpc' = "locked"
    /\ UNCHANGED <<value, upc, u_observed, u_obs_ver, w_new_val>>

(* Writer changes the value (update or re-insert). *)
WWrite ==
    /\ wpc = "locked"
    /\ value' = w_new_val
    /\ wpc' = "written"
    /\ UNCHANGED <<version, upc, u_observed, u_obs_ver, w_new_val>>

(* Writer ERASES the key (concurrent remove). *)
WErase ==
    /\ wpc = "locked"
    /\ value' = Empty
    /\ wpc' = "erased_w"
    /\ UNCHANGED <<version, upc, u_observed, u_obs_ver, w_new_val>>

(* Writer unlocks after write. *)
WUnlockWrite ==
    /\ wpc = "written"
    /\ version' = version + 1
    /\ wpc' = "done"
    /\ UNCHANGED <<value, upc, u_observed, u_obs_ver, w_new_val>>

(* Writer unlocks after erase. *)
WUnlockErase ==
    /\ wpc = "erased_w"
    /\ version' = version + 1
    /\ wpc' = "done"
    /\ UNCHANGED <<value, upc, u_observed, u_obs_ver, w_new_val>>

(* Writer cycles back. *)
WDone ==
    /\ wpc = "done"
    /\ w_new_val' \in Values
    /\ wpc' = "idle"
    /\ UNCHANGED <<value, version, upc, u_observed, u_obs_ver>>

(* ================================================================== *)
(* --- Specification ---                                              *)
(* ================================================================== *)

Step ==
    \/ UObserve \/ UObserveAbsent \/ UInsertOnAbsent \/ UInsertUnlock
    \/ URelease \/ URetraverse \/ UValidate
    \/ URestartOnLocked \/ UKeyGone \/ UErase \/ UUnlock \/ UDone
    \/ WLock \/ WWrite \/ WErase \/ WUnlockWrite \/ WUnlockErase \/ WDone

Spec == Init /\ [][Step]_vars /\ WF_vars(Step)

(* ================================================================== *)
(* --- Safety Properties ---                                          *)
(* ================================================================== *)

(* CAS SAFETY: If the upserter holds write lock (about to erase),
   the current value MUST be the value the lambda observed. *)
CASSafety ==
    upc = "write_locked" => value = u_observed

(* MUTUAL EXCLUSION: Upserter and writer never both hold write lock. *)
MutualExclusion ==
    ~(upc \in {"write_locked", "erased", "insert_unlock"} /\ wpc \in {"locked", "written", "erased_w"})

(* VERSION CONSISTENCY: version is odd iff exactly one process holds
   the write lock. *)
VersionConsistency ==
    version % 2 = 1 <=>
        ( upc \in {"write_locked", "erased", "insert_unlock"}
        \/ wpc \in {"locked", "written", "erased_w"} )

(* NO ERASE OF EMPTY: upserter never erases an already-empty slot. *)
NoEraseOfEmpty ==
    upc = "write_locked" => value # Empty

(* ================================================================== *)
(* --- Liveness Properties ---                                        *)
(* ================================================================== *)

(* PROGRESS: Under weak fairness, the upserter eventually reaches "done". *)
UpsertProgress == <>(upc = "done")

(* State constraint to keep version counter bounded for model checking. *)
StateConstraint == version <= 10

=========================================================================
