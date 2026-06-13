--------------------------- MODULE OLCKeyViewChain ------------------------
\* Stage 11: Variable-length keys with shared prefix and VIS.
\*
\* Models the critical scenario where two keys share a prefix:
\*   Key A (short): value stored as VIS in shared_node
\*   Key B (long): path continues through shared_node → chain_B → leaf_B
\*
\* Tree structure:
\*   parent → shared_node(count depends on VIS + child)
\*              ├── VIS entry (key A's value, bitmask-tracked)
\*              └── chain_B → leaf_B (key B's exclusive path)
\*
\* The critical invariant: chain identification for key B must NOT
\* include shared_node when it has a VIS entry (count should be 2).
\* If VIS is invisible to count (the bug), shared_node appears to have
\* count=1, and removing key B would incorrectly cut shared_node,
\* LOSING key A's value.
\*
\* This stage verifies:
\* 1. Correct count (VIS-aware): chain identification stops at shared_node
\* 2. Buggy count (VIS-invisible): chain identification incorrectly
\*    includes shared_node → data loss
\* 3. Concurrent insert/remove of key A while key B is being removed

EXTENDS Integers

CONSTANTS MaxVersion, OBSOLETE

VARIABLES
    \* Parent node (above shared_node)
    p_ver,
    p_count,        \* >= 2 (shared_node + at least one sibling)

    \* Shared node (the node where keys A and B diverge)
    s_ver,
    s_has_vis,      \* TRUE if key A's VIS entry is present
    s_has_child_b,  \* TRUE if pointer to chain_B is present
    \* Correct count: s_has_vis + s_has_child_b (+ other children)
    \* Buggy count: only counts non-null slots (misses VIS with value=0)

    \* Chain B node (key B's exclusive chain below shared_node)
    cb_ver,

    \* Remover B: removing key B (chain cut through chain_B)
    rbpc,           \* idle|lock_cb|eval_shared|validate_shared|
                    \* upgrade_shared|eval_parent|cut|done|restart
    rb_cb_locked,
    rb_s_locked,    \* TRUE if shared_node locked as chain member (BUG path)
    rb_cached_ver,
    rb_chain_includes_shared,  \* TRUE if chain identification included shared_node

    \* Inserter/Remover A: can insert or remove key A's VIS entry
    apc,            \* idle|lock_s|insert_vis|remove_vis|unlock|done
    a_op            \* "insert" or "remove"

vars == <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
          rbpc, rb_cb_locked, rb_s_locked, rb_cached_ver,
          rb_chain_includes_shared,
          apc, a_op>>

\* The CORRECT count of shared_node's children
CorrectCount == (IF s_has_vis THEN 1 ELSE 0) + (IF s_has_child_b THEN 1 ELSE 0)

\* BUGGY count: only counts non-null slots. If VIS value=0, pack(0)=NULL,
\* so the VIS entry is invisible. This is the bug we're fixing.
\* Model this by assuming VIS entry has value=0 (worst case).
BuggyCount == (IF s_has_child_b THEN 1 ELSE 0)  \* VIS always invisible

\* Chain membership precondition (correct version)
IsChainMember == CorrectCount = 1

\* BUGGY chain membership (for bug demonstration)
BuggyIsChainMember == BuggyCount = 1

-----------------------------------------------------------------------------
Init ==
    /\ p_ver = 0 /\ p_count = 2
    /\ s_ver = 0 /\ s_has_vis = TRUE /\ s_has_child_b = TRUE
    /\ cb_ver = 0
    /\ rbpc = "idle"
    /\ rb_cb_locked = FALSE /\ rb_s_locked = FALSE
    /\ rb_cached_ver = 0 /\ rb_chain_includes_shared = FALSE
    /\ apc = "idle" /\ a_op = "remove"

-----------------------------------------------------------------------------
\* REMOVER B: removing key B via chain cut

\* Lock chain_B (bottom of key B's exclusive chain)
RBLockCB ==
    /\ rbpc = "idle"
    /\ cb_ver % 2 = 0 /\ cb_ver # OBSOLETE
    /\ s_has_child_b = TRUE   \* chain_B still exists
    /\ cb_ver' = cb_ver + 1
    /\ rb_cb_locked' = TRUE
    /\ rbpc' = "eval_shared"
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b,
                   rb_s_locked, rb_cached_ver, rb_chain_includes_shared,
                   apc, a_op>>

\* Evaluate shared_node for chain membership
RBEvalShared ==
    /\ rbpc = "eval_shared"
    /\ rb_cached_ver' = s_ver
    /\ rbpc' = "validate_shared"
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_chain_includes_shared,
                   apc, a_op>>

\* Validate and check chain membership
RBValidateShared ==
    /\ rbpc = "validate_shared"
    /\ IF s_ver = rb_cached_ver /\ s_ver % 2 = 0
       THEN \* Version valid — check chain membership using CORRECT count
            IF IsChainMember
            THEN \* shared_node IS a chain member (count=1)
                 \* This means only chain_B exists, no VIS → safe to include
                 /\ rb_chain_includes_shared' = TRUE
                 /\ rbpc' = "upgrade_shared"
            ELSE \* shared_node is NOT a chain member (count>=2)
                 \* VIS entry present → shared_node is the CPP boundary
                 /\ rb_chain_includes_shared' = FALSE
                 /\ rbpc' = "eval_parent_as_cpp"
       ELSE \* Version changed — re-read
            /\ rbpc' = "eval_shared"
            /\ UNCHANGED rb_chain_includes_shared
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_cached_ver, apc, a_op>>

\* Upgrade shared_node to write (chain member path)
RBUpgradeShared ==
    /\ rbpc = "upgrade_shared"
    /\ IF s_ver = rb_cached_ver /\ s_ver % 2 = 0
       THEN /\ s_ver' = s_ver + 1
            /\ rb_s_locked' = TRUE
            /\ rbpc' = "eval_parent_as_cpp"
       ELSE /\ rbpc' = "eval_shared"  \* retry
            /\ UNCHANGED <<s_ver, rb_s_locked>>
    /\ UNCHANGED <<p_ver, p_count, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_cached_ver, rb_chain_includes_shared,
                   apc, a_op>>

\* Evaluate parent as CPP (shared_node is either chain member or CPP itself)
RBEvalParentAsCPP ==
    /\ rbpc = "eval_parent_as_cpp"
    /\ rb_cached_ver' = p_ver
    /\ rbpc' = "validate_parent"
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_chain_includes_shared,
                   apc, a_op>>

RBValidateParent ==
    /\ rbpc = "validate_parent"
    /\ IF p_ver = rb_cached_ver /\ p_ver % 2 = 0
       THEN /\ p_ver' = p_ver + 1   \* lock parent as CPP
            /\ rbpc' = "cut"
       ELSE /\ rbpc' = "eval_parent_as_cpp"
            /\ UNCHANGED p_ver
    /\ UNCHANGED <<p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_cached_ver,
                   rb_chain_includes_shared, apc, a_op>>

\* CUT
RBCut ==
    /\ rbpc = "cut"
    /\ IF rb_chain_includes_shared
       THEN \* Cut includes shared_node: remove shared_node from parent
            /\ p_count' = p_count - 1
            /\ p_ver' = p_ver + 1        \* unlock parent
            /\ s_ver' = OBSOLETE         \* obsolete shared_node
            /\ cb_ver' = OBSOLETE        \* obsolete chain_B
            /\ s_has_child_b' = FALSE
            /\ UNCHANGED s_has_vis       \* VIS entry lost if it existed!
       ELSE \* Cut only chain_B: remove chain_B from shared_node
            /\ s_has_child_b' = FALSE
            /\ IF rb_s_locked
               THEN s_ver' = s_ver + 1   \* unlock shared (was locked as CPP? no...)
               ELSE s_ver' = s_ver        \* shared wasn't locked
            /\ p_ver' = p_ver + 1        \* unlock parent (CPP)
            /\ cb_ver' = OBSOLETE
            /\ UNCHANGED <<p_count, s_has_vis>>
    /\ rb_cb_locked' = FALSE /\ rb_s_locked' = FALSE
    /\ rbpc' = "done"
    /\ UNCHANGED <<rb_cached_ver, rb_chain_includes_shared, apc, a_op>>

RBRestart ==
    /\ rbpc = "restart"
    /\ cb_ver' = IF rb_cb_locked THEN cb_ver + 1 ELSE cb_ver
    /\ s_ver' = IF rb_s_locked THEN s_ver + 1 ELSE s_ver
    /\ rb_cb_locked' = FALSE /\ rb_s_locked' = FALSE
    /\ rbpc' = "idle"
    /\ UNCHANGED <<p_ver, p_count, s_has_vis, s_has_child_b,
                   rb_cached_ver, rb_chain_includes_shared, apc, a_op>>

RBDone ==
    /\ rbpc = "done"
    /\ rbpc' = "idle"
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_cached_ver,
                   rb_chain_includes_shared, apc, a_op>>

-----------------------------------------------------------------------------
\* ACTOR A: insert or remove key A's VIS entry in shared_node

AChoose ==
    /\ apc = "idle"
    /\ \E op \in {"insert", "remove"} :
        /\ (op = "insert" => ~s_has_vis)   \* can only insert if not present
        /\ (op = "remove" => s_has_vis)    \* can only remove if present
        /\ a_op' = op
    /\ apc' = "lock_s"
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rbpc, rb_cb_locked, rb_s_locked, rb_cached_ver,
                   rb_chain_includes_shared>>

ALockS ==
    /\ apc = "lock_s"
    /\ s_ver % 2 = 0 /\ s_ver # OBSOLETE /\ s_ver <= MaxVersion
    /\ s_ver' = s_ver + 1
    /\ apc' = "write"
    /\ UNCHANGED <<p_ver, p_count, s_has_vis, s_has_child_b, cb_ver,
                   rbpc, rb_cb_locked, rb_s_locked, rb_cached_ver,
                   rb_chain_includes_shared, a_op>>

AWrite ==
    /\ apc = "write"
    /\ IF a_op = "insert"
       THEN s_has_vis' = TRUE
       ELSE s_has_vis' = FALSE
    /\ apc' = "unlock"
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_child_b, cb_ver,
                   rbpc, rb_cb_locked, rb_s_locked, rb_cached_ver,
                   rb_chain_includes_shared, a_op>>

AUnlock ==
    /\ apc = "unlock"
    /\ s_ver' = s_ver + 1
    /\ apc' = "idle"
    /\ UNCHANGED <<p_ver, p_count, s_has_vis, s_has_child_b, cb_ver,
                   rbpc, rb_cb_locked, rb_s_locked, rb_cached_ver,
                   rb_chain_includes_shared, a_op>>

-----------------------------------------------------------------------------
Step ==
    \/ RBLockCB \/ RBEvalShared \/ RBValidateShared \/ RBUpgradeShared
    \/ RBEvalParentAsCPP \/ RBValidateParent \/ RBCut \/ RBRestart \/ RBDone
    \/ AChoose \/ ALockS \/ AWrite \/ AUnlock

Spec == Init /\ [][Step]_vars

StateConstraint ==
    /\ p_ver <= MaxVersion
    /\ (s_ver # OBSOLETE => s_ver <= MaxVersion)
    /\ (cb_ver # OBSOLETE => cb_ver <= MaxVersion)

-----------------------------------------------------------------------------
\* INVARIANTS

\* CRITICAL: If chain cut includes shared_node, then at CUT TIME
\* shared_node must NOT have a VIS entry. Otherwise key A's value is lost.
NoDataLoss ==
    (rbpc = "cut" /\ rb_chain_includes_shared) => ~s_has_vis

\* Chain membership was correctly evaluated: if shared_node is included
\* AND we reach cut, the VIS entry must have been absent when we validated.
\* (The OLC version check ensures our read was consistent.)
\* NOTE: In this model, this reduces to the same predicate as NoDataLoss
\* because the single-point-in-time abstraction conflates validation and cut.
\* A richer model with separate validation/cut steps would distinguish them.
ChainMembershipCorrect ==
    (rbpc = "cut" /\ rb_chain_includes_shared) => ~s_has_vis

\* Well-formedness: parent count stays valid
WellFormedness == p_count >= 1

-----------------------------------------------------------------------------
\* BUGGY VARIANT: Uses BuggyIsChainMember (VIS invisible to count)

BuggyRBValidateShared ==
    /\ rbpc = "validate_shared"
    /\ IF s_ver = rb_cached_ver /\ s_ver % 2 = 0
       THEN IF BuggyIsChainMember    \* <-- THE BUG: ignores VIS entry
            THEN /\ rb_chain_includes_shared' = TRUE
                 /\ rbpc' = "upgrade_shared"
            ELSE /\ rb_chain_includes_shared' = FALSE
                 /\ rbpc' = "eval_parent_as_cpp"
       ELSE /\ rbpc' = "eval_shared"
            /\ UNCHANGED rb_chain_includes_shared
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_cached_ver, apc, a_op>>

BuggyStep ==
    \/ RBLockCB \/ RBEvalShared \/ BuggyRBValidateShared \/ RBUpgradeShared
    \/ RBEvalParentAsCPP \/ RBValidateParent \/ RBCut \/ RBRestart \/ RBDone
    \/ AChoose \/ ALockS \/ AWrite \/ AUnlock

BuggySpec == Init /\ [][BuggyStep]_vars

=============================================================================
