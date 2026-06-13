--------------------------- MODULE OLCChainMultiLevel ---------------------
\* Stage 9: Multi-level chain cut with concurrent insert.
\*
\* Topology: parent → chain0 → chain1 → (leaf, implicit)
\* Parent has 2+ children (one is chain0).
\* chain0 has 1 child (chain1). chain1 has 1 child (leaf).
\*
\* Remover: locks chain1, then chain0 (bottom-up), then evaluates parent.
\* Inserter: can add a child to chain0 OR chain1, breaking membership.
\*
\* Key scenarios exercised:
\* - cut_level moves DOWN: insert breaks chain0 after chain1 locked
\* - cut_level stays: both chain nodes lock successfully
\* - Case A on parent: normal cut
\* - Case C on parent: child pointer gone (concurrent remove of chain0)
\*
\* Verifies: cut only proceeds with correct locks and valid state.

EXTENDS Integers

CONSTANTS MaxVersion

VARIABLES
    \* Parent node
    p_count,    \* 2 or 3 (chain0 + sibling, optionally + inserted)
    p_has_c0,   \* parent points to chain0
    p_ver,

    \* Chain node 0 (top of chain, child of parent)
    c0_count,   \* 1 or 2
    c0_ver,

    \* Chain node 1 (bottom of chain, child of chain0)
    c1_count,   \* 1 (always — leaf is implicit)
    c1_ver,

    \* Remover state
    rpc,        \* idle|lock_c1|lock_c0|read_parent|validate_parent|
                \* upgrade_parent|cut|done|restart
    r_cut_level,\* 0=cut from chain0, 1=cut from chain1 only
    r_c1_locked,\* holds write lock on chain1
    r_c0_locked,\* holds write lock on chain0
    r_pver,     \* cached parent version
    r_p_has_c0, \* cached: parent has chain0?

    \* Inserter state
    ipc,        \* idle|lock|write|unlock|done
    i_target,   \* "c0" or "c1"
    i_ver       \* cached version of target for lock

vars == <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
          rpc, r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
          ipc, i_target, i_ver>>

-----------------------------------------------------------------------------
Init ==
    /\ p_count = 2 /\ p_has_c0 = TRUE /\ p_ver = 0
    /\ c0_count = 1 /\ c0_ver = 0
    /\ c1_count = 1 /\ c1_ver = 0
    /\ rpc = "idle" /\ r_cut_level = 0
    /\ r_c1_locked = FALSE /\ r_c0_locked = FALSE
    /\ r_pver = 0 /\ r_p_has_c0 = FALSE
    /\ ipc = "idle" /\ i_target = "c0" /\ i_ver = 0

-----------------------------------------------------------------------------
\* REMOVER: bottom-up chain locking, then CPP evaluation

\* Step 2a: Lock chain1 (bottom of chain)
RemoverLockC1 ==
    /\ rpc = "idle"
    /\ c1_ver % 2 = 0 /\ c1_count = 1  \* chain membership precondition
    /\ c1_ver' = c1_ver + 1             \* lock
    /\ r_c1_locked' = TRUE
    /\ r_cut_level' = 1                 \* initially cut from chain1
    /\ rpc' = "lock_c0"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count,
                   r_c0_locked, r_pver, r_p_has_c0, ipc, i_target, i_ver>>

\* Step 2b: Try to lock chain0 (revalidation with precondition)
RemoverLockC0_OK ==
    /\ rpc = "lock_c0"
    /\ c0_ver % 2 = 0 /\ c0_count = 1  \* precondition holds
    /\ c0_ver' = c0_ver + 1             \* lock
    /\ r_c0_locked' = TRUE
    /\ r_cut_level' = 0                 \* cut from chain0 (full chain)
    /\ rpc' = "read_parent"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c1_count, c1_ver,
                   r_c1_locked, r_pver, r_p_has_c0, ipc, i_target, i_ver>>

\* Step 2b: chain0 precondition FAILS (count != 1 or version odd)
\* cut_level stays at 1 — chain0 becomes the cut_point_parent
RemoverLockC0_Fail ==
    /\ rpc = "lock_c0"
    /\ (c0_ver % 2 = 1 \/ c0_count # 1)  \* precondition fails
    /\ r_cut_level' = 1                   \* cut only chain1
    /\ rpc' = "read_c0_as_cpp"            \* chain0 is now CPP
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
                   r_c1_locked, r_c0_locked, r_pver, r_p_has_c0, ipc, i_target, i_ver>>

\* Step 3 (when chain0 is CPP): read chain0 state
RemoverReadC0AsCPP ==
    /\ rpc = "read_c0_as_cpp"
    /\ r_pver' = c0_ver                  \* reuse r_pver for c0's version
    /\ rpc' = "validate_c0_cpp"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
                   r_cut_level, r_c1_locked, r_c0_locked, r_p_has_c0, ipc, i_target, i_ver>>

\* Validate chain0 as CPP and upgrade
RemoverValidateC0CPP ==
    /\ rpc = "validate_c0_cpp"
    /\ IF c0_ver = r_pver /\ c0_ver % 2 = 0
       THEN /\ c0_ver' = c0_ver + 1     \* lock chain0 as CPP
            /\ r_c0_locked' = TRUE
            /\ rpc' = "cut"
       ELSE /\ rpc' = "restart"          \* version changed
            /\ UNCHANGED <<c0_ver, r_c0_locked>>
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c1_count, c1_ver,
                   r_cut_level, r_c1_locked, r_pver, r_p_has_c0, ipc, i_target, i_ver>>

\* Step 3 (when parent is CPP): read parent state
RemoverReadParent ==
    /\ rpc = "read_parent"
    /\ r_pver' = p_ver
    /\ r_p_has_c0' = p_has_c0
    /\ rpc' = "validate_parent"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
                   r_cut_level, r_c1_locked, r_c0_locked, ipc, i_target, i_ver>>

\* Validate parent and upgrade
RemoverValidateParent ==
    /\ rpc = "validate_parent"
    /\ IF p_ver = r_pver /\ p_ver % 2 = 0
       THEN IF ~r_p_has_c0
            THEN rpc' = "restart"        \* Case C
            ELSE rpc' = "upgrade_parent"  \* Case A
       ELSE rpc' = "read_parent"         \* stale
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
                   r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
                   ipc, i_target, i_ver>>

RemoverUpgradeParent ==
    /\ rpc = "upgrade_parent"
    /\ IF p_ver = r_pver /\ p_ver % 2 = 0
       THEN /\ p_ver' = p_ver + 1
            /\ rpc' = "cut"
       ELSE /\ rpc' = "read_parent"
            /\ UNCHANGED p_ver
    /\ UNCHANGED <<p_count, p_has_c0, c0_count, c0_ver, c1_count, c1_ver,
                   r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
                   ipc, i_target, i_ver>>

\* Step 4: Cut (point of no return)
RemoverCut ==
    /\ rpc = "cut"
    /\ IF r_cut_level = 0
       THEN \* Full chain cut: remove chain0 from parent
            /\ p_has_c0' = FALSE
            /\ p_count' = p_count - 1
            /\ p_ver' = p_ver + 1        \* unlock parent
            /\ c0_ver' = c0_ver + 1      \* obsolete chain0
            /\ c1_ver' = c1_ver + 1      \* obsolete chain1
            /\ UNCHANGED <<c0_count, c1_count>>
       ELSE \* Partial cut: remove chain1 from chain0
            /\ c0_count' = c0_count - 1
            /\ c0_ver' = c0_ver + 1      \* unlock chain0 (CPP)
            /\ c1_ver' = c1_ver + 1      \* obsolete chain1
            /\ UNCHANGED <<p_count, p_has_c0, p_ver, c1_count>>
    /\ r_c0_locked' = FALSE
    /\ r_c1_locked' = FALSE
    /\ rpc' = "done"
    /\ UNCHANGED <<r_cut_level, r_pver, r_p_has_c0, ipc, i_target, i_ver>>

RemoverRestart ==
    /\ rpc = "restart"
    /\ IF r_c1_locked THEN c1_ver' = c1_ver + 1 ELSE UNCHANGED c1_ver
    /\ IF r_c0_locked THEN c0_ver' = c0_ver + 1 ELSE UNCHANGED c0_ver
    /\ r_c1_locked' = FALSE /\ r_c0_locked' = FALSE
    /\ rpc' = "idle"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c1_count,
                   r_cut_level, r_pver, r_p_has_c0, ipc, i_target, i_ver>>

-----------------------------------------------------------------------------
\* INSERTER: adds a child to chain0 or chain1 (breaking chain membership)

InsertChoose ==
    /\ ipc = "idle"
    /\ \E t \in {"c0", "c1"} :
        /\ (t = "c0" => c0_count < 2)
        /\ (t = "c1" => c1_count < 2)
        /\ i_target' = t
    /\ ipc' = "lock"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
                   rpc, r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
                   i_ver>>

InsertLock ==
    /\ ipc = "lock"
    /\ IF i_target = "c0"
       THEN /\ c0_ver % 2 = 0 /\ c0_ver <= MaxVersion
            /\ c0_ver' = c0_ver + 1
            /\ UNCHANGED c1_ver
       ELSE /\ c1_ver % 2 = 0 /\ c1_ver <= MaxVersion
            /\ c1_ver' = c1_ver + 1
            /\ UNCHANGED c0_ver
    /\ ipc' = "write"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c1_count,
                   rpc, r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
                   i_target, i_ver>>

InsertWrite ==
    /\ ipc = "write"
    /\ IF i_target = "c0"
       THEN /\ c0_count' = c0_count + 1
            /\ UNCHANGED c1_count
       ELSE /\ c1_count' = c1_count + 1
            /\ UNCHANGED c0_count
    /\ ipc' = "unlock"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_ver, c1_ver,
                   rpc, r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
                   i_target, i_ver>>

InsertUnlock ==
    /\ ipc = "unlock"
    /\ IF i_target = "c0"
       THEN /\ c0_ver' = c0_ver + 1
            /\ UNCHANGED c1_ver
       ELSE /\ c1_ver' = c1_ver + 1
            /\ UNCHANGED c0_ver
    /\ ipc' = "done"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c1_count,
                   rpc, r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
                   i_target, i_ver>>

InsertDone ==
    /\ ipc = "done"
    /\ ipc' = "idle"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
                   rpc, r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
                   i_target, i_ver>>

-----------------------------------------------------------------------------
Step ==
    \/ RemoverLockC1 \/ RemoverLockC0_OK \/ RemoverLockC0_Fail
    \/ RemoverReadC0AsCPP \/ RemoverValidateC0CPP
    \/ RemoverReadParent \/ RemoverValidateParent \/ RemoverUpgradeParent
    \/ RemoverCut \/ RemoverRestart
    \/ InsertChoose \/ InsertLock \/ InsertWrite \/ InsertUnlock \/ InsertDone

Spec == Init /\ [][Step]_vars
StateConstraint == p_ver <= MaxVersion /\ c0_ver <= MaxVersion /\ c1_ver <= MaxVersion

-----------------------------------------------------------------------------
\* INVARIANTS

\* Cut only when holding correct locks and state is valid
CutSafety ==
    rpc = "cut" =>
        /\ r_c1_locked                   \* always hold chain1 lock
        /\ (r_cut_level = 0 => r_c0_locked)  \* full cut needs chain0 lock
        /\ (r_cut_level = 0 => p_has_c0)     \* parent still has chain0

\* Chain membership: if remover holds chain lock, count must be 1
ChainLockConsistency ==
    /\ (r_c1_locked /\ c1_ver % 2 = 1 => c1_count = 1)
    /\ (r_c0_locked /\ c0_ver % 2 = 1 /\ r_cut_level = 0 => c0_count = 1)

\* Well-formedness: counts never go below valid range
WellFormedness ==
    /\ p_count >= 1
    /\ c0_count >= 0
    /\ c1_count >= 0

\* Partial cut correctness: if cut_level=1, chain0 is CPP (locked by remover)
PartialCutValid ==
    (rpc = "cut" /\ r_cut_level = 1) => r_c0_locked

=============================================================================
