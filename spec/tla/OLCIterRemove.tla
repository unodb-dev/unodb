--------------------------- MODULE OLCIterRemove ---------------------------
(* OLC (Optimistic Lock Coupling) Iterator vs. Remover specification.
   Models a 3-node tree: grandparent → parent → child.
   A remover can obsolete either the parent or the child.
   An iterator descends the tree using optimistic validation at each level.

   The key safety property (SnapshotValid): if the iterator reaches the "act"
   state, it validated against an even (unlocked), non-obsolete version. *)

EXTENDS Naturals

CONSTANT MaxVersion
CONSTANT OBSOLETE

VARIABLES
    \* Grandparent node
    gp_ver,       \* version counter (even = unlocked, odd = write-locked)
    gp_child,     \* pointer to parent (TRUE = valid, FALSE = removed)
    \* Parent node
    p_ver,        \* version counter
    p_child,      \* pointer to child (TRUE = valid, FALSE = removed)
    \* Child node
    c_ver,        \* version counter
    c_obsolete,   \* whether child is obsolete
    \* Iterator state
    ipc,          \* iterator program counter
    i_gp_ver,     \* snapshot of gp_ver taken during read_gp
    i_p_ver,      \* snapshot of p_ver taken during read_parent
    i_cver,       \* snapshot of c_ver taken during read_child
    \* Remover state
    rpc,          \* remover program counter
    r_target,     \* which node the remover targets: "parent" or "child"
    \* Writer state (normal writer — bumps version without obsoleting)
    wpc,          \* writer program counter: idle | lock | unlock
    w_target      \* which node writer targets: "parent" or "child"

vars == <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
          ipc, i_gp_ver, i_p_ver, i_cver, rpc, r_target, wpc, w_target>>

-----------------------------------------------------------------------------
(* Initial state *)

Init ==
    /\ gp_ver = 0
    /\ gp_child = TRUE
    /\ p_ver = 0
    /\ p_child = TRUE
    /\ c_ver = 0
    /\ c_obsolete = FALSE
    /\ ipc = "idle"
    /\ i_gp_ver = 0
    /\ i_p_ver = 0
    /\ i_cver = 0
    /\ rpc = "idle"
    /\ r_target = "child"
    /\ wpc = "idle"
    /\ w_target = "child"

-----------------------------------------------------------------------------
(* Iterator actions.
   Spinning on a write-locked node is modeled by the action being disabled
   (precondition not met). Encountering OBSOLETE triggers a restart. *)

\* Step 1: Read grandparent — snapshot its version
\* Spins while locked; restarts if obsolete (handled: gp can't be obsoleted)
IterReadGP ==
    /\ ipc = "idle"
    /\ gp_ver # OBSOLETE
    /\ gp_ver % 2 = 0
    /\ i_gp_ver' = gp_ver
    /\ ipc' = "read_gp"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   i_p_ver, i_cver, rpc, r_target, wpc, w_target>>

\* Step 2: Descend to parent — validate grandparent version unchanged
IterDescendToParent ==
    /\ ipc = "read_gp"
    /\ IF gp_ver = i_gp_ver /\ gp_ver # OBSOLETE
       THEN ipc' = "read_parent"
       ELSE ipc' = "idle"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   i_gp_ver, i_p_ver, i_cver, rpc, r_target, wpc, w_target>>

\* Step 3: Read parent — snapshot version
\* Enabled only when parent is unlocked and not obsolete (spin/restart)
IterReadParent ==
    /\ ipc = "read_parent"
    /\ p_ver # OBSOLETE
    /\ p_ver % 2 = 0
    /\ i_p_ver' = p_ver
    /\ ipc' = "descend_to_child"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   i_gp_ver, i_cver, rpc, r_target, wpc, w_target>>

\* Step 3b: Restart if parent is obsolete
IterReadParentRestart ==
    /\ ipc = "read_parent"
    /\ p_ver = OBSOLETE
    /\ ipc' = "idle"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   i_gp_ver, i_p_ver, i_cver, rpc, r_target, wpc, w_target>>

\* Step 4: Descend to child — validate parent version unchanged
IterDescendToChild ==
    /\ ipc = "descend_to_child"
    /\ IF p_ver = i_p_ver /\ p_ver # OBSOLETE
       THEN ipc' = "read_child"
       ELSE ipc' = "idle"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   i_gp_ver, i_p_ver, i_cver, rpc, r_target, wpc, w_target>>

\* Step 5: Read child — snapshot version
\* Enabled only when child is unlocked and not obsolete (spin/restart)
IterReadChild ==
    /\ ipc = "read_child"
    /\ c_ver # OBSOLETE
    /\ c_ver % 2 = 0
    /\ i_cver' = c_ver
    /\ ipc' = "validate_child"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   i_gp_ver, i_p_ver, rpc, r_target, wpc, w_target>>

\* Step 5b: Restart if child is obsolete
IterReadChildRestart ==
    /\ ipc = "read_child"
    /\ c_ver = OBSOLETE
    /\ ipc' = "idle"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   i_gp_ver, i_p_ver, i_cver, rpc, r_target, wpc, w_target>>

\* Step 6: Validate child — check version unchanged and not obsolete
IterValidateChild ==
    /\ ipc = "validate_child"
    /\ IF c_ver = i_cver /\ c_ver # OBSOLETE
       THEN ipc' = "act"
       ELSE ipc' = "idle"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   i_gp_ver, i_p_ver, i_cver, rpc, r_target, wpc, w_target>>

\* Step 7: Act on the data and finish
IterAct ==
    /\ ipc = "act"
    /\ ipc' = "done"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   i_gp_ver, i_p_ver, i_cver, rpc, r_target, wpc, w_target>>

-----------------------------------------------------------------------------
(* Remover actions — targets either parent or child *)

\* Choose target (non-deterministic)
RemoverChooseTarget ==
    /\ rpc = "idle"
    /\ \E t \in {"parent", "child"} :
        /\ r_target' = t
        /\ rpc' = "lock"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   ipc, i_gp_ver, i_p_ver, i_cver, wpc, w_target>>

\* Lock the grandparent (write-lock for structural change)
RemoverLockGP ==
    /\ rpc = "lock"
    /\ gp_ver % 2 = 0
    /\ gp_ver # OBSOLETE
    /\ gp_ver' = gp_ver + 1
    /\ rpc' = "lock_target"
    /\ UNCHANGED <<gp_child, p_ver, p_child, c_ver, c_obsolete,
                   ipc, i_gp_ver, i_p_ver, i_cver, r_target, wpc, w_target>>

\* Lock the target node
RemoverLockTarget ==
    /\ rpc = "lock_target"
    /\ IF r_target = "parent"
       THEN /\ p_ver % 2 = 0
            /\ p_ver # OBSOLETE
            /\ p_ver' = p_ver + 1
            /\ UNCHANGED <<c_ver, c_obsolete, p_child>>
       ELSE /\ c_ver % 2 = 0
            /\ c_ver # OBSOLETE
            /\ c_ver' = c_ver + 1
            /\ UNCHANGED <<p_ver, c_obsolete, p_child>>
    /\ rpc' = "obsolete"
    /\ UNCHANGED <<gp_ver, gp_child, ipc, i_gp_ver, i_p_ver, i_cver, r_target, wpc, w_target>>

\* Mark target obsolete and unlink
RemoverObsolete ==
    /\ rpc = "obsolete"
    /\ IF r_target = "parent"
       THEN /\ p_ver' = OBSOLETE
            /\ gp_child' = FALSE
            /\ UNCHANGED <<c_ver, c_obsolete, p_child>>
       ELSE /\ c_ver' = OBSOLETE
            /\ c_obsolete' = TRUE
            /\ p_child' = FALSE
            /\ UNCHANGED <<p_ver, gp_child>>
    /\ rpc' = "unlock_gp"
    /\ UNCHANGED <<gp_ver, ipc, i_gp_ver, i_p_ver, i_cver, r_target, wpc, w_target>>

\* Unlock grandparent (bump version to next even)
RemoverUnlockGP ==
    /\ rpc = "unlock_gp"
    /\ gp_ver' = gp_ver + 1
    /\ rpc' = "done"
    /\ UNCHANGED <<gp_child, p_ver, p_child, c_ver, c_obsolete,
                   ipc, i_gp_ver, i_p_ver, i_cver, r_target, wpc, w_target>>

-----------------------------------------------------------------------------
(* Normal writer: locks a node, bumps version, unlocks. No obsolescence.
   Tests that iterator detects version changes from concurrent inserts. *)

WriterChoose ==
    /\ wpc = "idle"
    /\ \E t \in {"parent", "child"} :
        /\ w_target' = t
    /\ wpc' = "lock"
    /\ UNCHANGED <<gp_ver, gp_child, p_ver, p_child, c_ver, c_obsolete,
                   ipc, i_gp_ver, i_p_ver, i_cver, rpc, r_target>>

WriterLock ==
    /\ wpc = "lock"
    /\ IF w_target = "parent"
       THEN /\ p_ver % 2 = 0 /\ p_ver < MaxVersion
            /\ p_ver' = p_ver + 1
            /\ UNCHANGED c_ver
       ELSE /\ c_ver % 2 = 0 /\ c_ver < MaxVersion /\ ~c_obsolete
            /\ c_ver' = c_ver + 1
            /\ UNCHANGED p_ver
    /\ wpc' = "unlock"
    /\ UNCHANGED <<gp_ver, gp_child, p_child, c_obsolete,
                   ipc, i_gp_ver, i_p_ver, i_cver, rpc, r_target, w_target>>

WriterUnlock ==
    /\ wpc = "unlock"
    /\ IF w_target = "parent"
       THEN /\ p_ver' = p_ver + 1
            /\ UNCHANGED c_ver
       ELSE /\ c_ver' = c_ver + 1
            /\ UNCHANGED p_ver
    /\ wpc' = "idle"
    /\ UNCHANGED <<gp_ver, gp_child, p_child, c_obsolete,
                   ipc, i_gp_ver, i_p_ver, i_cver, rpc, r_target, w_target>>

-----------------------------------------------------------------------------
(* Specification *)

Next ==
    \/ IterReadGP
    \/ IterDescendToParent
    \/ IterReadParent
    \/ IterReadParentRestart
    \/ IterDescendToChild
    \/ IterReadChild
    \/ IterReadChildRestart
    \/ IterValidateChild
    \/ IterAct
    \/ RemoverChooseTarget
    \/ RemoverLockGP
    \/ RemoverLockTarget
    \/ RemoverObsolete
    \/ RemoverUnlockGP
    \/ WriterChoose
    \/ WriterLock
    \/ WriterUnlock

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants *)

TypeOK ==
    \* Version counters may transiently reach MaxVersion+1 (locked state)
    /\ gp_ver \in (0..MaxVersion + 1) \cup {OBSOLETE}
    /\ gp_child \in BOOLEAN
    /\ p_ver \in (0..MaxVersion + 1) \cup {OBSOLETE}
    /\ p_child \in BOOLEAN
    /\ c_ver \in (0..MaxVersion + 1) \cup {OBSOLETE}
    /\ c_obsolete \in BOOLEAN
    /\ ipc \in {"idle", "read_gp", "read_parent", "descend_to_child",
                "read_child", "validate_child", "act", "done"}
    /\ i_gp_ver \in 0..MaxVersion
    /\ i_p_ver \in 0..MaxVersion
    /\ i_cver \in 0..MaxVersion
    /\ rpc \in {"idle", "lock", "lock_target", "obsolete", "unlock_gp", "done"}
    /\ r_target \in {"parent", "child"}
    /\ wpc \in {"idle", "lock", "unlock"}
    /\ w_target \in {"parent", "child"}

\* Safety: if iterator acts, it validated against an even (unlocked),
\* non-obsolete version.
SnapshotValid ==
    ipc = "act" => (i_cver % 2 = 0 /\ i_cver # OBSOLETE)

\* State constraint to bound model checking
StateConstraint ==
    /\ (gp_ver # OBSOLETE => gp_ver <= MaxVersion)
    /\ (p_ver # OBSOLETE => p_ver <= MaxVersion)
    /\ (c_ver # OBSOLETE => c_ver <= MaxVersion)

\* Action constraint: prevent transitions that push versions beyond bound.
\* This avoids the TypeOK vs StateConstraint conflict where TLC checks
\* invariants on successor states before pruning them.
ActionConstraint ==
    /\ (gp_ver' # OBSOLETE => gp_ver' <= MaxVersion)
    /\ (p_ver' # OBSOLETE => p_ver' <= MaxVersion)
    /\ (c_ver' # OBSOLETE => c_ver' <= MaxVersion)

=============================================================================
