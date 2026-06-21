-------------------------- MODULE OLCIteratorRoot ----------------------------
(* Phase 5: Root pointer mutation and tree emptying.

   Models the two-level locking protocol at the root:
     Level 1: root_pointer_lock (protects the root pointer variable)
     Level 2: node version lock (protects the root node's contents)

   The iterator must:
     1. Take root_pointer_lock snapshot
     2. Load root pointer
     3. If null → return end()
     4. CHECK root_pointer_lock (validates loaded pointer is still current)
     5. Take node lock on loaded root
     6. Release root_pointer_lock
     7. Act on node

   The writer can:
     - Set root = null (remove last key → tree empty)
     - Replace root (tree structural change)

   TOCTOU: between step 2 and step 4, writer can change root.
   Bug: skip step 4 → iterator uses stale/freed root pointer.
*)

EXTENDS Naturals, Sequences

CONSTANTS
    MaxVersion,
    BugMode       \* "none" | "skip_rp_check"

\* Possible root values
RootValues == {"L", "N", "null"}

\* "Freed" represents memory that was reclaimed — accessing it is UB
AllNodes == {"L", "N", "null", "freed"}

VARIABLES
    rp_ver,       \* root_pointer_lock version (even=unlocked, odd=locked)
    root,         \* current root pointer: "L" | "N" | "null"
    
    ipc,          \* iterator PC
    i_rp_snap,    \* snapshot of rp_ver
    i_loaded,     \* what iterator loaded from root
    visited,      \* delivered results
    restarts,
    
    wpc           \* writer PC

vars == <<rp_ver, root, ipc, i_rp_snap, i_loaded, visited, restarts, wpc>>

\* --- Helpers ---
RPCheckOK == rp_ver = i_rp_snap /\ (rp_ver % 2 = 0)
RPCanLock == rp_ver % 2 = 0

\* --- Initial state: tree has root = "L" (a single leaf) ---
Init ==
    /\ rp_ver = 0
    /\ root = "L"
    /\ ipc = "start"
    /\ i_rp_snap = 0
    /\ i_loaded = "null"
    /\ visited = <<>>
    /\ restarts = 0
    /\ wpc = "idle"

\* --- Iterator ---

\* Step 1: take root_pointer_lock snapshot
IterStart == ipc = "start" /\
    IF ~RPCanLock THEN
        /\ restarts' = restarts + 1
        /\ UNCHANGED <<rp_ver, root, ipc, i_rp_snap, i_loaded, visited, wpc>>
    ELSE
        /\ i_rp_snap' = rp_ver
        /\ ipc' = "load_root"
        /\ UNCHANGED <<rp_ver, root, i_loaded, visited, restarts, wpc>>

\* Step 2: load root pointer (reads CURRENT value — may become stale)
IterLoadRoot == ipc = "load_root" /\
    /\ i_loaded' = root
    /\ ipc' = "check_null"
    /\ UNCHANGED <<rp_ver, root, i_rp_snap, visited, restarts, wpc>>

\* Step 3: if null, return end()
IterCheckNull == ipc = "check_null" /\
    IF i_loaded = "null" THEN
        /\ ipc' = "done"
        /\ UNCHANGED <<rp_ver, root, i_rp_snap, i_loaded, visited, restarts, wpc>>
    ELSE
        /\ ipc' = "check_rp"
        /\ UNCHANGED <<rp_ver, root, i_rp_snap, i_loaded, visited, restarts, wpc>>

\* Step 4: CHECK root_pointer_lock — validates loaded root is still current
IterCheckRP == ipc = "check_rp" /\
    IF BugMode = "skip_rp_check" THEN
        \* BUG: skip the check — proceed with potentially stale root
        IF ~RPCheckOK THEN
            \* Version changed — loaded root is stale. Havoc: could be anything
            \E node \in AllNodes :
                /\ i_loaded' = node
                /\ ipc' = "act"
                /\ UNCHANGED <<rp_ver, root, i_rp_snap, visited, restarts, wpc>>
        ELSE
            \* Version matches — loaded root is still valid
            /\ ipc' = "act"
            /\ UNCHANGED <<rp_ver, root, i_rp_snap, i_loaded, visited, restarts, wpc>>
    ELSE
        \* Correct code: validate
        IF ~RPCheckOK THEN
            /\ ipc' = "start"
            /\ restarts' = restarts + 1
            /\ UNCHANGED <<rp_ver, root, i_rp_snap, i_loaded, visited, wpc>>
        ELSE
            /\ ipc' = "act"
            /\ UNCHANGED <<rp_ver, root, i_rp_snap, i_loaded, visited, restarts, wpc>>

\* Step 5-7: act on the loaded root (take node lock, traverse)
IterAct == ipc = "act" /\
    /\ visited' = Append(visited, i_loaded)
    /\ ipc' = "done"
    /\ UNCHANGED <<rp_ver, root, i_rp_snap, i_loaded, restarts, wpc>>

IterDone == ipc = "done" /\ UNCHANGED vars

\* --- Writer: empty the tree (set root = null) ---

WriterLockRP == wpc = "idle" /\
    /\ rp_ver % 2 = 0
    /\ rp_ver + 1 <= MaxVersion
    /\ rp_ver' = rp_ver + 1
    /\ wpc' = "locked"
    /\ UNCHANGED <<root, ipc, i_rp_snap, i_loaded, visited, restarts>>

WriterEmpty == wpc = "locked" /\
    /\ root' = "null"
    /\ wpc' = "mutated"
    /\ UNCHANGED <<rp_ver, ipc, i_rp_snap, i_loaded, visited, restarts>>

WriterUnlockRP == wpc = "mutated" /\
    /\ rp_ver + 1 <= MaxVersion
    /\ rp_ver' = rp_ver + 1
    /\ wpc' = "done"
    /\ UNCHANGED <<root, ipc, i_rp_snap, i_loaded, visited, restarts>>

WriterDone == wpc = "done" /\ UNCHANGED vars

\* --- Specification ---

Next ==
    \/ IterStart
    \/ IterLoadRoot
    \/ IterCheckNull
    \/ IterCheckRP
    \/ IterAct
    \/ IterDone
    \/ WriterLockRP
    \/ WriterEmpty
    \/ WriterUnlockRP
    \/ WriterDone

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* --- State Constraint ---
StateConstraint == restarts <= 3

\* --- Invariants ---

TypeOK ==
    /\ rp_ver \in 0..MaxVersion
    /\ root \in RootValues
    /\ ipc \in {"start", "load_root", "check_null", "check_rp", "act", "done"}
    /\ i_rp_snap \in 0..MaxVersion
    /\ i_loaded \in AllNodes
    /\ visited \in Seq(AllNodes)
    /\ restarts \in Nat
    /\ wpc \in {"idle", "locked", "mutated", "done"}

\* NoStaleRoot: if the iterator acts, it acts on a VALID root (one that is
\* currently the root OR was the root at the time of a successful check).
\* Concretely: visited must never contain "freed" or "null" (acting on null = crash)
NoActOnInvalid ==
    \A i \in 1..Len(visited) : visited[i] \in {"L", "N"}

\* SafeEmpty: if tree is empty when iterator finishes, visited is empty
\* (iterator correctly detected null and returned end())
\* Note: if tree was non-empty when iterator started, it's fine to visit it
\* even if it was emptied later (snapshot consistency).

\* The key safety property: iterator never dereferences a stale root pointer
\* that points to freed memory. In our model, "freed" = garbage value.
NoGarbage ==
    \A i \in 1..Len(visited) : visited[i] /= "freed"

\* Canary: root is never emptied (should fail — proves writer interleaves)
RootNeverNull ==
    root /= "null"

=============================================================================
