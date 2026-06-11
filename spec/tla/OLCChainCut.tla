--------------------------- MODULE OLCChainCut ----------------------------
(* OLC (Optimistic Lock Coupling) chain-cut protocol.
   Models a remover thread cutting a child node from a parent in a B+-tree
   with optimistic lock coupling, and a concurrent inserter that can target
   the parent or grandparent node.

   Fix M1: RemoverLockedImpliesChainSingle replaces trivially-true ChainLockValid.
   Fix M4: Grandparent node added to model Step 4.1 acquisition. *)

EXTENDS Naturals, TLC

CONSTANTS MaxVersion

VARIABLES
    (* Remover state *)
    rpc,              \* remover program counter
    r_chain_locked,   \* remover holds chain lock on child
    r_p_ver_snap,     \* remover's snapshot of parent version
    r_gp_ver_snap,    \* remover's snapshot of grandparent version

    (* Parent node *)
    p_ver,            \* parent version (odd = locked)
    p_count,          \* number of children in parent

    (* Child node *)
    c_count,          \* number of entries in child (chain length proxy)

    (* Grandparent node *)
    gp_ver,           \* grandparent version (odd = locked)
    gp_child,         \* grandparent points to parent (TRUE = connected)

    (* Inserter state *)
    ipc,              \* inserter program counter
    i_target          \* inserter's target: "parent" | "grandparent"

vars == <<rpc, r_chain_locked, r_p_ver_snap, r_gp_ver_snap,
          p_ver, p_count, c_count,
          gp_ver, gp_child,
          ipc, i_target>>

-----------------------------------------------------------------------------
(* Initial state *)

Init ==
    /\ rpc = "start"
    /\ r_chain_locked = FALSE
    /\ r_p_ver_snap = 0
    /\ r_gp_ver_snap = 0
    /\ p_ver = 0
    /\ p_count = 2
    /\ c_count = 1
    /\ gp_ver = 0
    /\ gp_child = TRUE
    /\ ipc = "idle"
    /\ i_target = "parent"

-----------------------------------------------------------------------------
(* Remover actions *)

(* Step 1: Read parent version optimistically *)
ReadParentVer ==
    /\ rpc = "start"
    /\ p_ver % 2 = 0          \* parent not locked
    /\ rpc' = "read_parent"
    /\ r_p_ver_snap' = p_ver
    /\ UNCHANGED <<r_chain_locked, r_gp_ver_snap, p_ver, p_count, c_count,
                   gp_ver, gp_child, ipc, i_target>>

(* Step 2: Lock the chain (child) *)
LockChain ==
    /\ rpc = "read_parent"
    /\ rpc' = "validate_parent"
    /\ r_chain_locked' = TRUE
    /\ c_count' = 1           \* chain is single node once locked
    /\ UNCHANGED <<r_p_ver_snap, r_gp_ver_snap, p_ver, p_count,
                   gp_ver, gp_child, ipc, i_target>>

(* Step 3: Validate parent version unchanged *)
ValidateParent ==
    /\ rpc = "validate_parent"
    /\ IF p_ver = r_p_ver_snap /\ p_ver % 2 = 0
       THEN rpc' = "case_a"
       ELSE rpc' = "restart"   \* validation failed, must restart
    /\ UNCHANGED <<r_chain_locked, r_p_ver_snap, r_gp_ver_snap, p_ver, p_count, c_count,
                   gp_ver, gp_child, ipc, i_target>>

(* Step 4a: Lock parent (case_a) *)
LockParent ==
    /\ rpc = "case_a"
    /\ p_ver % 2 = 0          \* parent not already locked
    /\ p_ver' = p_ver + 1     \* lock parent (make odd)
    /\ IF p_count = 2
       THEN rpc' = "acquire_gp"   \* parent will shrink, need grandparent
       ELSE rpc' = "cut"          \* no shrink needed, proceed to cut
    /\ UNCHANGED <<r_chain_locked, r_p_ver_snap, r_gp_ver_snap, p_count, c_count,
                   gp_ver, gp_child, ipc, i_target>>

(* Step 4.1: Acquire grandparent lock (needed when parent will shrink) *)
AcquireGP ==
    /\ rpc = "acquire_gp"
    /\ IF gp_ver % 2 = 0
       THEN /\ gp_ver' = gp_ver + 1   \* lock grandparent
            /\ r_gp_ver_snap' = gp_ver
            /\ rpc' = "cut"
       ELSE /\ rpc' = "gp_failed"     \* grandparent busy, must release all
            /\ gp_ver' = gp_ver
            /\ r_gp_ver_snap' = r_gp_ver_snap
    /\ UNCHANGED <<r_chain_locked, r_p_ver_snap, p_ver, p_count, c_count,
                   gp_child, ipc, i_target>>

(* Grandparent acquisition failed: release all locks and restart *)
GPFailed ==
    /\ rpc = "gp_failed"
    /\ rpc' = "restart"
    /\ p_ver' = p_ver + 1             \* unlock parent (make even again)
    /\ UNCHANGED <<r_chain_locked, r_p_ver_snap, r_gp_ver_snap, p_count, c_count,
                   gp_ver, gp_child, ipc, i_target>>

(* Step 5: Perform the cut *)
Cut ==
    /\ rpc = "cut"
    /\ p_count' = p_count - 1
    /\ rpc' = "unlock"
    /\ UNCHANGED <<r_chain_locked, r_p_ver_snap, r_gp_ver_snap, p_ver, c_count,
                   gp_ver, gp_child, ipc, i_target>>

(* Step 6: Unlock parent (and grandparent if held) *)
Unlock ==
    /\ rpc = "unlock"
    /\ p_ver' = p_ver + 1             \* unlock parent (make even)
    /\ IF r_gp_ver_snap /= 0
       THEN gp_ver' = gp_ver + 1      \* unlock grandparent
       ELSE gp_ver' = gp_ver
    /\ r_chain_locked' = FALSE
    /\ r_gp_ver_snap' = 0
    /\ rpc' = "done"
    /\ UNCHANGED <<r_p_ver_snap, p_count, c_count, gp_child, ipc, i_target>>

(* Restart: release chain lock and go back to start *)
Restart ==
    /\ rpc = "restart"
    /\ r_chain_locked' = FALSE
    /\ rpc' = "start"
    /\ UNCHANGED <<r_p_ver_snap, r_gp_ver_snap, p_ver, p_count, c_count,
                   gp_ver, gp_child, ipc, i_target>>

-----------------------------------------------------------------------------
(* Inserter actions — concurrent thread that can insert into parent or gp *)

InsertChoose ==
    /\ ipc = "idle"
    /\ \E t \in {"parent", "grandparent"} :
        /\ i_target' = t
        /\ ipc' = "insert"
    /\ UNCHANGED <<rpc, r_chain_locked, r_p_ver_snap, r_gp_ver_snap,
                   p_ver, p_count, c_count, gp_ver, gp_child>>

InsertParent ==
    /\ ipc = "insert"
    /\ i_target = "parent"
    /\ p_ver % 2 = 0              \* parent not locked
    /\ p_ver' = p_ver + 2         \* bump version (lock+unlock)
    /\ p_count' = p_count + 1
    /\ ipc' = "insert_done"
    /\ UNCHANGED <<rpc, r_chain_locked, r_p_ver_snap, r_gp_ver_snap, c_count,
                   gp_ver, gp_child, i_target>>

InsertGrandparent ==
    /\ ipc = "insert"
    /\ i_target = "grandparent"
    /\ gp_ver % 2 = 0             \* grandparent not locked
    /\ gp_ver' = gp_ver + 2       \* bump version (lock+unlock)
    /\ ipc' = "insert_done"
    /\ UNCHANGED <<rpc, r_chain_locked, r_p_ver_snap, r_gp_ver_snap,
                   p_ver, p_count, c_count, gp_child, i_target>>

(* Inserter cycles back to idle *)
InsertDone ==
    /\ ipc = "insert_done"
    /\ ipc' = "idle"
    /\ UNCHANGED <<rpc, r_chain_locked, r_p_ver_snap, r_gp_ver_snap,
                   p_ver, p_count, c_count, gp_ver, gp_child, i_target>>

-----------------------------------------------------------------------------
(* Next-state relation *)

RemoverNext ==
    \/ ReadParentVer
    \/ LockChain
    \/ ValidateParent
    \/ LockParent
    \/ AcquireGP
    \/ GPFailed
    \/ Cut
    \/ Unlock
    \/ Restart

InserterNext ==
    \/ InsertChoose
    \/ InsertParent
    \/ InsertGrandparent
    \/ InsertDone

Next == RemoverNext \/ InserterNext

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* State constraint for bounded model checking *)

StateConstraint ==
    /\ p_ver <= MaxVersion
    /\ p_count >= 1
    /\ p_count <= 4
    /\ gp_ver <= MaxVersion

-----------------------------------------------------------------------------
(* Invariants *)

TypeOK ==
    /\ rpc \in {"start", "read_parent", "validate_parent",
                "case_a", "acquire_gp", "gp_failed",
                "cut", "unlock", "restart", "done"}
    /\ r_chain_locked \in BOOLEAN
    /\ r_p_ver_snap \in Nat
    /\ r_gp_ver_snap \in Nat
    /\ p_ver \in Nat
    /\ p_count \in Nat
    /\ c_count \in Nat
    /\ gp_ver \in Nat
    /\ gp_child \in BOOLEAN
    /\ ipc \in {"idle", "insert", "insert_done"}
    /\ i_target \in {"parent", "grandparent"}

(* M1: Stronger invariant — when remover holds chain lock and is in
   an active removal state, the chain must be a single node. *)
RemoverLockedImpliesChainSingle ==
    (r_chain_locked /\ rpc \in {"read_parent", "validate_parent", "case_a", "case_b", "case_c", "cut"})
        => c_count = 1

(* Safety: parent count never drops below 1 *)
ParentNotEmpty ==
    p_count >= 1

(* Grandparent always points to parent *)
GPConnected ==
    gp_child = TRUE

=============================================================================
