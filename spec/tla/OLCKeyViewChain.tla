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
\* 3. Concurrent insert/remove of key A while key B is being removed.
\*    shared_node's children are booleans, not ordered, so this model cannot
\*    tell a cut of chain_B from a cut at an index that key A's insert
\*    shifted before B's last read of shared_node.  B then locks shared_node
\*    at the version it read, so no later write can shift the index before
\*    the cut.

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
    rb_s_locked,    \* TRUE while B holds shared_node's write lock
    rb_s_cached_ver,  \* shared_node's version at B's last read of it
    rb_p_cached_ver,  \* parent's version at B's last read of it
    rb_chain_includes_shared,  \* TRUE if chain identification included shared_node

    \* Inserter/Remover A: can insert or remove key A's VIS entry
    apc,            \* idle|lock_s|insert_vis|remove_vis|unlock|done
    a_op            \* "insert" or "remove"

vars == <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
          rbpc, rb_cb_locked, rb_s_locked, rb_s_cached_ver, rb_p_cached_ver,
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
    /\ rb_s_cached_ver = 0 /\ rb_p_cached_ver = 0
    /\ rb_chain_includes_shared = FALSE
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
                   rb_s_locked, rb_s_cached_ver, rb_p_cached_ver,
                   rb_chain_includes_shared, apc, a_op>>

\* Evaluate shared_node for chain membership
RBEvalShared ==
    /\ rbpc = "eval_shared"
    /\ rb_s_cached_ver' = s_ver
    /\ rbpc' = "validate_shared"
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_p_cached_ver,
                   rb_chain_includes_shared, apc, a_op>>

\* Validate and check chain membership.  A chain member (count=1: only
\* chain_B, no VIS) is safe to include in the cut; otherwise key A's VIS entry
\* makes shared_node the cut-point parent, which the cut writes.  Either way,
\* lock it next.
RBValidateShared ==
    /\ rbpc = "validate_shared"
    /\ IF s_ver = rb_s_cached_ver /\ s_ver % 2 = 0
       THEN \* Version valid — check chain membership using CORRECT count
            /\ rb_chain_includes_shared' = IsChainMember
            /\ rbpc' = "upgrade_shared"
       ELSE \* Version changed — re-read
            /\ rbpc' = "eval_shared"
            /\ UNCHANGED rb_chain_includes_shared
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_s_cached_ver, rb_p_cached_ver,
                   apc, a_op>>

\* Upgrade shared_node to write at the version B validated, as a chain member
\* or as the cut-point parent
RBUpgradeShared ==
    /\ rbpc = "upgrade_shared"
    /\ IF s_ver = rb_s_cached_ver /\ s_ver % 2 = 0
       THEN /\ s_ver' = s_ver + 1
            /\ rb_s_locked' = TRUE
            /\ rbpc' = "eval_parent"
       ELSE /\ rbpc' = "eval_shared"  \* retry
            /\ UNCHANGED <<s_ver, rb_s_locked>>
    /\ UNCHANGED <<p_ver, p_count, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_cached_ver, rb_p_cached_ver,
                   rb_chain_includes_shared, apc, a_op>>

\* Evaluate the parent: the cut-point parent if shared_node is a chain member,
\* else the grandparent that Step 4.1 locks because the two-child shared_node
\* needs shrinking; try_collapse_i4 then keeps shared_node, whose remaining
\* entry is key A's packed value
RBEvalParent ==
    /\ rbpc = "eval_parent"
    /\ rb_p_cached_ver' = p_ver
    /\ rbpc' = "validate_parent"
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_s_cached_ver,
                   rb_chain_includes_shared, apc, a_op>>

\* Only B writes p_ver, so the ELSE branch never fires.  Its re-read is Step
\* 3's for a cut-point parent; for the grandparent that Step 4.1 locks, the
\* code instead restarts the remove if the version moved since the descent.
RBValidateParent ==
    /\ rbpc = "validate_parent"
    /\ IF p_ver = rb_p_cached_ver /\ p_ver % 2 = 0
       THEN /\ p_ver' = p_ver + 1   \* lock parent
            /\ rbpc' = "cut"
       ELSE /\ rbpc' = "eval_parent"
            /\ UNCHANGED p_ver
    /\ UNCHANGED <<p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_s_cached_ver, rb_p_cached_ver,
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
            /\ s_ver' = s_ver + 1        \* unlock cut-point parent
            /\ p_ver' = p_ver + 1        \* unlock parent (grandparent)
            /\ cb_ver' = OBSOLETE
            /\ UNCHANGED <<p_count, s_has_vis>>
    /\ rb_cb_locked' = FALSE /\ rb_s_locked' = FALSE
    /\ rbpc' = "done"
    /\ UNCHANGED <<rb_s_cached_ver, rb_p_cached_ver, rb_chain_includes_shared,
                   apc, a_op>>

RBRestart ==
    /\ rbpc = "restart"
    /\ cb_ver' = IF rb_cb_locked THEN cb_ver + 1 ELSE cb_ver
    /\ s_ver' = IF rb_s_locked THEN s_ver + 1 ELSE s_ver
    /\ rb_cb_locked' = FALSE /\ rb_s_locked' = FALSE
    /\ rbpc' = "idle"
    /\ UNCHANGED <<p_ver, p_count, s_has_vis, s_has_child_b,
                   rb_s_cached_ver, rb_p_cached_ver, rb_chain_includes_shared,
                   apc, a_op>>

RBDone ==
    /\ rbpc = "done"
    /\ rbpc' = "idle"
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_s_cached_ver, rb_p_cached_ver,
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
                   rbpc, rb_cb_locked, rb_s_locked, rb_s_cached_ver,
                   rb_p_cached_ver, rb_chain_includes_shared>>

ALockS ==
    /\ apc = "lock_s"
    /\ s_ver % 2 = 0 /\ s_ver # OBSOLETE /\ s_ver <= MaxVersion
    /\ s_ver' = s_ver + 1
    /\ apc' = "write"
    /\ UNCHANGED <<p_ver, p_count, s_has_vis, s_has_child_b, cb_ver,
                   rbpc, rb_cb_locked, rb_s_locked, rb_s_cached_ver,
                   rb_p_cached_ver, rb_chain_includes_shared, a_op>>

AWrite ==
    /\ apc = "write"
    /\ IF a_op = "insert"
       THEN s_has_vis' = TRUE
       ELSE s_has_vis' = FALSE
    /\ apc' = "unlock"
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_child_b, cb_ver,
                   rbpc, rb_cb_locked, rb_s_locked, rb_s_cached_ver,
                   rb_p_cached_ver, rb_chain_includes_shared, a_op>>

AUnlock ==
    /\ apc = "unlock"
    /\ s_ver' = s_ver + 1
    /\ apc' = "idle"
    /\ UNCHANGED <<p_ver, p_count, s_has_vis, s_has_child_b, cb_ver,
                   rbpc, rb_cb_locked, rb_s_locked, rb_s_cached_ver,
                   rb_p_cached_ver, rb_chain_includes_shared, a_op>>

-----------------------------------------------------------------------------
Step ==
    \/ RBLockCB \/ RBEvalShared \/ RBValidateShared \/ RBUpgradeShared
    \/ RBEvalParent \/ RBValidateParent \/ RBCut \/ RBRestart \/ RBDone
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

\* The membership decision B validated still holds when it cuts.
ChainMembershipCorrect ==
    rbpc = "cut" => (rb_chain_includes_shared = IsChainMember)

\* B cuts with shared_node write-locked at the version it last read, on both
\* paths, so key A's insert or remove cannot land between that read and the
\* cut.
SharedLockedAtCut ==
    rbpc = "cut" => (rb_s_locked /\ s_ver = rb_s_cached_ver + 1)

\* Well-formedness: parent count stays valid
WellFormedness == p_count >= 1

-----------------------------------------------------------------------------
\* BUGGY VARIANT: Uses BuggyIsChainMember (VIS invisible to count)

BuggyRBValidateShared ==
    /\ rbpc = "validate_shared"
    /\ IF s_ver = rb_s_cached_ver /\ s_ver % 2 = 0
       THEN \* <-- THE BUG: BuggyIsChainMember ignores the VIS entry
            /\ rb_chain_includes_shared' = BuggyIsChainMember
            /\ rbpc' = "upgrade_shared"
       ELSE /\ rbpc' = "eval_shared"
            /\ UNCHANGED rb_chain_includes_shared
    /\ UNCHANGED <<p_ver, p_count, s_ver, s_has_vis, s_has_child_b, cb_ver,
                   rb_cb_locked, rb_s_locked, rb_s_cached_ver, rb_p_cached_ver,
                   apc, a_op>>

BuggyStep ==
    \/ RBLockCB \/ RBEvalShared \/ BuggyRBValidateShared \/ RBUpgradeShared
    \/ RBEvalParent \/ RBValidateParent \/ RBCut \/ RBRestart \/ RBDone
    \/ AChoose \/ ALockS \/ AWrite \/ AUnlock

BuggySpec == Init /\ [][BuggyStep]_vars

=============================================================================
