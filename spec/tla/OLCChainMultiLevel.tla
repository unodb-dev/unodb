--------------------------- MODULE OLCChainMultiLevel ---------------------
\* Stage 9: Multi-level chain cut with concurrent insert.
\*
\* Topology: parent → chain0 → chain1 → (leaf, implicit)
\* Parent has 2+ children (one is chain0).
\* chain0 has 1 child (chain1). chain1 has 1 child (leaf).
\*
\* Remover: locks chain1, then chain0 (bottom-up), then evaluates parent.
\* Inserter: can add a child to chain0 OR chain1, breaking membership.  No
\* other remove runs, and no insert replaces a child link, so the chain links
\* the code checks in Step 3 always hold here; Stage 14
\* (OLCChainCutVsSplit) models a prefix split that replaces one.  c0_c1_idx tracks chain1's
\* slot among chain0's children: the cut_level-moves-DOWN insert below moves
\* chain1 to the second slot whenever the new key byte sorts before chain1's.
\* A partial cut must cut at chain1's slot as of the version at which the
\* remover locks chain0 (CutAtChainIndex).  With CutAtDescentIndex, as in
\* OLCChainMultiLevel_BugDemo.cfg, it cuts at the slot the descent read, as
\* the code did before it looked the slot up again under that lock, and
\* violates CutAtChainIndex.  No insert targets parent, so the full cut's
\* slot there cannot shift.
\*
\* Key scenarios exercised:
\* - cut_level moves DOWN: insert breaks chain0 after chain1 locked
\* - cut_level stays: both chain nodes lock successfully
\* - Case A on parent: normal cut
\*
\* Verifies: cut only proceeds with correct locks and valid state.

EXTENDS Integers

CONSTANTS
    MaxVersion,
    CutAtDescentIndex  \* TRUE: cut at the descent's slot (the pre-fix code)

VARIABLES
    \* Parent node
    p_count,    \* 2 or 3 (chain0 + sibling, optionally + inserted)
    p_has_c0,   \* parent points to chain0
    p_ver,

    \* Chain node 0 (top of chain, child of parent)
    c0_count,   \* 1 or 2
    c0_ver,
    c0_c1_idx,  \* chain1's slot among chain0's children

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
    r_cut_idx,  \* chain1's slot in chain0 that a partial cut removes

    \* Inserter state
    ipc,        \* idle|lock|write|unlock|done
    i_target,   \* "c0" or "c1"
    i_ver       \* cached version of target for lock

vars == <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c0_c1_idx, c1_count,
          c1_ver, rpc, r_cut_level, r_c1_locked, r_c0_locked, r_pver,
          r_p_has_c0, r_cut_idx, ipc, i_target, i_ver>>

-----------------------------------------------------------------------------
Init ==
    /\ p_count = 2 /\ p_has_c0 = TRUE /\ p_ver = 0
    /\ c0_count = 1 /\ c0_ver = 0 /\ c0_c1_idx = 0
    /\ c1_count = 1 /\ c1_ver = 0
    /\ rpc = "idle" /\ r_cut_level = 0
    /\ r_c1_locked = FALSE /\ r_c0_locked = FALSE
    /\ r_pver = 0 /\ r_p_has_c0 = FALSE /\ r_cut_idx = 0
    /\ ipc = "idle" /\ i_target = "c0" /\ i_ver = 0

-----------------------------------------------------------------------------
\* REMOVER: bottom-up chain locking, then CPP evaluation

\* Lock chain1 (bottom of chain), which the code holds from the descent.
RemoverLockC1 ==
    /\ rpc = "idle"
    /\ c1_ver % 2 = 0 /\ c1_count = 1  \* chain membership precondition
    /\ c1_ver' = c1_ver + 1             \* lock
    /\ r_c1_locked' = TRUE
    /\ r_cut_level' = 1                 \* initially cut from chain1
    /\ r_cut_idx' = c0_c1_idx           \* the slot the descent read
    /\ rpc' = "lock_c0"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count,
                   r_c0_locked, r_pver, r_p_has_c0, ipc, i_target, i_ver,
                   c0_c1_idx>>

\* Lock chain0 as a chain member.  The code does this in Step 2 only at
\* chain0's descent version; once that has moved, Step 3 Case B does it at
\* the current version if chain0 is still a one-child I4 leading to chain1.
\* With no descent versions or child links here, both reduce to this action.
RemoverLockC0_OK ==
    /\ rpc = "lock_c0"
    /\ c0_ver % 2 = 0 /\ c0_count = 1  \* precondition holds
    /\ c0_ver' = c0_ver + 1             \* lock
    /\ r_c0_locked' = TRUE
    /\ r_cut_level' = 0                 \* cut from chain0 (full chain)
    /\ rpc' = "read_parent"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c1_count, c1_ver,
                   r_c1_locked, r_pver, r_p_has_c0, ipc, i_target, i_ver,
                   c0_c1_idx, r_cut_idx>>

\* chain0 is not a lockable one-child I4 (count != 1 or version odd):
\* Step 3 takes it as the cut_point_parent, so cut_level stays at 1.
RemoverLockC0_Fail ==
    /\ rpc = "lock_c0"
    /\ (c0_ver % 2 = 1 \/ c0_count # 1)  \* precondition fails
    /\ r_cut_level' = 1                   \* cut only chain1
    /\ rpc' = "read_c0_as_cpp"            \* chain0 is now CPP
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
                   r_c1_locked, r_c0_locked, r_pver, r_p_has_c0, ipc, i_target,
                   i_ver, c0_c1_idx, r_cut_idx>>

\* Step 3 (when chain0 is CPP): read chain0 state
RemoverReadC0AsCPP ==
    /\ rpc = "read_c0_as_cpp"
    /\ r_pver' = c0_ver                  \* reuse r_pver for c0's version
    /\ r_cut_idx' = IF CutAtDescentIndex THEN r_cut_idx ELSE c0_c1_idx
    /\ rpc' = "validate_c0_cpp"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
                   r_cut_level, r_c1_locked, r_c0_locked, r_p_has_c0, ipc,
                   i_target, i_ver, c0_c1_idx>>

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
                   r_cut_level, r_c1_locked, r_pver, r_p_has_c0, ipc, i_target,
                   i_ver, c0_c1_idx, r_cut_idx>>

\* Step 3 (when parent is CPP): read parent state
RemoverReadParent ==
    /\ rpc = "read_parent"
    /\ r_pver' = p_ver
    /\ r_p_has_c0' = p_has_c0
    /\ rpc' = "validate_parent"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
                   r_cut_level, r_c1_locked, r_c0_locked, ipc, i_target, i_ver,
                   c0_c1_idx, r_cut_idx>>

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
                   ipc, i_target, i_ver, c0_c1_idx, r_cut_idx>>

RemoverUpgradeParent ==
    /\ rpc = "upgrade_parent"
    /\ IF p_ver = r_pver /\ p_ver % 2 = 0
       THEN /\ p_ver' = p_ver + 1
            /\ rpc' = "cut"
       ELSE /\ rpc' = "read_parent"
            /\ UNCHANGED p_ver
    /\ UNCHANGED <<p_count, p_has_c0, c0_count, c0_ver, c1_count, c1_ver,
                   r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
                   ipc, i_target, i_ver, c0_c1_idx, r_cut_idx>>

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
    /\ UNCHANGED <<r_cut_level, r_pver, r_p_has_c0, ipc, i_target, i_ver,
                   c0_c1_idx, r_cut_idx>>

RemoverRestart ==
    /\ rpc = "restart"
    /\ IF r_c1_locked THEN c1_ver' = c1_ver + 1 ELSE UNCHANGED c1_ver
    /\ IF r_c0_locked THEN c0_ver' = c0_ver + 1 ELSE UNCHANGED c0_ver
    /\ r_c1_locked' = FALSE /\ r_c0_locked' = FALSE
    /\ rpc' = "idle"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c1_count,
                   r_cut_level, r_pver, r_p_has_c0, ipc, i_target, i_ver,
                   c0_c1_idx, r_cut_idx>>

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
                   i_ver, c0_c1_idx, r_cut_idx>>

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
                   i_target, i_ver, c0_c1_idx, r_cut_idx>>

InsertWrite ==
    /\ ipc = "write"
    /\ IF i_target = "c0"
       THEN /\ c0_count' = c0_count + 1
            \* The new key byte sorts before or after chain1's.
            /\ c0_c1_idx' \in {c0_c1_idx, c0_c1_idx + 1}
            /\ UNCHANGED c1_count
       ELSE /\ c1_count' = c1_count + 1
            /\ UNCHANGED <<c0_count, c0_c1_idx>>
    /\ ipc' = "unlock"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_ver, c1_ver,
                   rpc, r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
                   i_target, i_ver, r_cut_idx>>

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
                   i_target, i_ver, c0_c1_idx, r_cut_idx>>

InsertDone ==
    /\ ipc = "done"
    /\ ipc' = "idle"
    /\ UNCHANGED <<p_count, p_has_c0, p_ver, c0_count, c0_ver, c1_count, c1_ver,
                   rpc, r_cut_level, r_c1_locked, r_c0_locked, r_pver, r_p_has_c0,
                   i_target, i_ver, c0_c1_idx, r_cut_idx>>

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

\* A partial cut removes chain1, not a sibling an insert moved into its slot.
CutAtChainIndex ==
    (rpc = "cut" /\ r_cut_level = 1) => r_cut_idx = c0_c1_idx

=============================================================================
