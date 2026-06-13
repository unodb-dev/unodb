--------------------------- MODULE OLCChainCutFull ------------------------
\* Stage 10: Full chain cut algorithm covering all critical paths.
\*
\* Topology: root_ptr → gp → parent → chain → (leaf, implicit)
\*   - root_ptr: separate lock (not a node), can be CPP if chain extends to gp
\*   - gp: can become single-child (Case B extends chain to include parent)
\*   - parent: the initial CPP candidate (2+ children)
\*   - chain: single-child I4 (the chain being cut)
\*
\* Paths exercised:
\*   P1: Normal cut — parent is CPP, no shrink needed
\*   P2: Parent needs shrink — acquire gp as grandparent, collapse
\*   P3: Case B — parent became single-child, chain extends up
\*   P4: Cascading Case B — gp also single-child, cut_level reaches 0 (root)
\*   P5: Root as CPP — set root=nullptr (tree emptied)
\*   P6: Grandparent acquisition fails — restart
\*   P7: Case C — child pointer gone, restart
\*
\* Concurrent inserter can add children to parent or gp (breaking Case B,
\* preventing shrink, etc.)

EXTENDS Integers

CONSTANTS MaxVersion, OBSOLETE

VARIABLES
    \* Root pointer (separate lock, not a node)
    root_ver,       \* root pointer lock version
    root_child,     \* root points to gp (TRUE/FALSE, FALSE = tree empty)

    \* Grandparent node
    gp_count,       \* 1..3
    gp_child,       \* gp points to parent (TRUE)
    gp_ver,

    \* Parent node (initial CPP candidate)
    p_count,        \* 1..3
    p_child,        \* parent points to chain (TRUE)
    p_ver,

    \* Chain node
    c_ver,

    \* Remover state
    rpc,
    r_cut_level,    \* 0=root is CPP, 1=gp is CPP, 2=parent is CPP, 3=chain-only
    r_chain_locked,
    r_parent_locked,
    r_gp_locked,
    r_cpp_locked,   \* TRUE when CPP write guard held
    r_needs_shrink, \* TRUE if CPP needs shrink after cut
    r_cached_ver,   \* cached version for current CPP evaluation

    \* Inserter state
    ipc,
    i_target        \* "parent" or "gp"

vars == <<root_ver, root_child, gp_count, gp_child, gp_ver,
          p_count, p_child, p_ver, c_ver,
          rpc, r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
          r_cpp_locked, r_needs_shrink, r_cached_ver,
          ipc, i_target>>

UNCHANGED_inserter == UNCHANGED <<ipc, i_target>>
UNCHANGED_shared == UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                                p_count, p_child, p_ver, c_ver>>

-----------------------------------------------------------------------------
Init ==
    /\ root_ver = 0 /\ root_child = TRUE
    /\ gp_count \in {1, 2} /\ gp_child = TRUE /\ gp_ver = 0
    /\ p_count \in {1, 2} /\ p_child = TRUE /\ p_ver = 0
    /\ c_ver = 0
    /\ rpc = "idle"
    /\ r_cut_level = 3 /\ r_chain_locked = FALSE
    /\ r_parent_locked = FALSE /\ r_gp_locked = FALSE
    /\ r_cpp_locked = FALSE /\ r_needs_shrink = FALSE /\ r_cached_ver = 0
    /\ ipc = "idle" /\ i_target = "parent"

-----------------------------------------------------------------------------
\* REMOVER

\* Step 2: Lock chain node
RLockChain ==
    /\ rpc = "idle"
    /\ c_ver % 2 = 0
    /\ c_ver # OBSOLETE       \* not already obsoleted
    /\ p_child = TRUE         \* chain still connected (not already cut)
    /\ c_ver' = c_ver + 1
    /\ r_chain_locked' = TRUE
    /\ r_cut_level' = 3       \* start: only chain locked
    /\ rpc' = "eval_parent"
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, p_ver,
                   r_parent_locked, r_gp_locked, r_cpp_locked,
                   r_needs_shrink, r_cached_ver, ipc, i_target>>

\* Step 3: Evaluate parent as CPP candidate
REvalParent ==
    /\ rpc = "eval_parent"
    /\ r_cached_ver' = p_ver
    /\ rpc' = "validate_parent"
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, p_ver, c_ver,
                   r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_cpp_locked, r_needs_shrink, ipc, i_target>>

RValidateParent ==
    /\ rpc = "validate_parent"
    /\ IF p_ver = r_cached_ver /\ p_ver % 2 = 0
       THEN IF ~p_child
            THEN rpc' = "restart"                    \* Case C
            ELSE IF p_count = 1
            THEN rpc' = "case_b_parent"              \* Case B: parent is single-child
            ELSE rpc' = "upgrade_parent"             \* Case A: parent is CPP
       ELSE rpc' = "eval_parent"                     \* stale, re-read
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, p_ver, c_ver,
                   r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_cpp_locked, r_needs_shrink, r_cached_ver, ipc, i_target>>

\* Case A: lock parent as CPP
RUpgradeParent ==
    /\ rpc = "upgrade_parent"
    /\ IF p_ver = r_cached_ver /\ p_ver % 2 = 0
       THEN /\ p_ver' = p_ver + 1
            /\ r_cpp_locked' = TRUE
            /\ r_cut_level' = 2          \* parent is CPP
            /\ rpc' = "check_shrink"
       ELSE /\ rpc' = "eval_parent"
            /\ UNCHANGED <<p_ver, r_cpp_locked, r_cut_level>>
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, c_ver,
                   r_chain_locked, r_parent_locked, r_gp_locked,
                   r_needs_shrink, r_cached_ver, ipc, i_target>>

\* Case B: parent became single-child → lock it as chain member, eval gp
RCaseBParent ==
    /\ rpc = "case_b_parent"
    /\ IF p_ver = r_cached_ver /\ p_ver % 2 = 0
       THEN /\ p_ver' = p_ver + 1       \* lock parent as chain member
            /\ r_parent_locked' = TRUE
            /\ rpc' = "eval_gp"
       ELSE /\ rpc' = "eval_parent"     \* retry
            /\ UNCHANGED <<p_ver, r_parent_locked>>
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, c_ver,
                   r_cut_level, r_chain_locked, r_gp_locked,
                   r_cpp_locked, r_needs_shrink, r_cached_ver, ipc, i_target>>

\* Evaluate gp as CPP (after Case B on parent)
REvalGP ==
    /\ rpc = "eval_gp"
    /\ r_cached_ver' = gp_ver
    /\ rpc' = "validate_gp"
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, p_ver, c_ver,
                   r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_cpp_locked, r_needs_shrink, ipc, i_target>>

RValidateGP ==
    /\ rpc = "validate_gp"
    /\ IF gp_ver = r_cached_ver /\ gp_ver % 2 = 0
       THEN IF ~gp_child
            THEN rpc' = "restart"                    \* Case C at gp level
            ELSE IF gp_count = 1
            THEN rpc' = "case_b_gp"                  \* Cascading Case B!
            ELSE rpc' = "upgrade_gp"                 \* Case A at gp level
       ELSE rpc' = "eval_gp"
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, p_ver, c_ver,
                   r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_cpp_locked, r_needs_shrink, r_cached_ver, ipc, i_target>>

\* Case A at gp: lock gp as CPP
RUpgradeGP ==
    /\ rpc = "upgrade_gp"
    /\ IF gp_ver = r_cached_ver /\ gp_ver % 2 = 0
       THEN /\ gp_ver' = gp_ver + 1
            /\ r_cpp_locked' = TRUE
            /\ r_cut_level' = 1          \* gp is CPP
            /\ rpc' = "check_shrink"
       ELSE /\ rpc' = "eval_gp"
            /\ UNCHANGED <<gp_ver, r_cpp_locked, r_cut_level>>
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child,
                   p_count, p_child, p_ver, c_ver,
                   r_chain_locked, r_parent_locked, r_gp_locked,
                   r_needs_shrink, r_cached_ver, ipc, i_target>>

\* Cascading Case B: gp also single-child → lock gp, go to root
RCaseBGP ==
    /\ rpc = "case_b_gp"
    /\ IF gp_ver = r_cached_ver /\ gp_ver % 2 = 0
       THEN /\ gp_ver' = gp_ver + 1     \* lock gp as chain member
            /\ r_gp_locked' = TRUE
            /\ rpc' = "acquire_root"     \* root is now CPP
       ELSE /\ rpc' = "eval_gp"
            /\ UNCHANGED <<gp_ver, r_gp_locked>>
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child,
                   p_count, p_child, p_ver, c_ver,
                   r_cut_level, r_chain_locked, r_parent_locked,
                   r_cpp_locked, r_needs_shrink, r_cached_ver, ipc, i_target>>

\* Root as CPP (cut_level = 0)
RAcquireRoot ==
    /\ rpc = "acquire_root"
    /\ IF root_ver % 2 = 0
       THEN /\ root_ver' = root_ver + 1
            /\ r_cpp_locked' = TRUE
            /\ r_cut_level' = 0
            /\ rpc' = "cut"              \* root CPP → always cut (no shrink check)
       ELSE /\ rpc' = "restart"
            /\ UNCHANGED <<root_ver, r_cpp_locked, r_cut_level>>
    /\ UNCHANGED <<root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, p_ver, c_ver,
                   r_chain_locked, r_parent_locked, r_gp_locked,
                   r_needs_shrink, r_cached_ver, ipc, i_target>>

\* Check if CPP needs shrink (only for non-root CPP)
RCheckShrink ==
    /\ rpc = "check_shrink"
    /\ IF r_cut_level = 2
       THEN r_needs_shrink' = (p_count <= 2)   \* I4 with 2 children → collapse after cut
       ELSE r_needs_shrink' = (gp_count <= 2)  \* gp is CPP
    /\ IF r_needs_shrink'
       THEN rpc' = "acquire_gp_for_shrink"
       ELSE rpc' = "cut"
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, p_ver, c_ver,
                   r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_cpp_locked, r_cached_ver, ipc, i_target>>

\* Acquire grandparent for shrink (can fail → restart)
\* When CPP is parent (cut_level=2), grandparent is gp node.
\* When CPP is gp (cut_level=1), grandparent is root pointer.
RAcquireGPForShrink_Parent ==
    /\ rpc = "acquire_gp_for_shrink"
    /\ r_cut_level = 2
    /\ IF gp_ver % 2 = 0 /\ gp_ver <= MaxVersion
       THEN /\ gp_ver' = gp_ver + 1
            /\ r_gp_locked' = TRUE
            /\ rpc' = "cut"
       ELSE /\ rpc' = "restart"
            /\ UNCHANGED <<gp_ver, r_gp_locked>>
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child,
                   p_count, p_child, p_ver, c_ver,
                   r_cut_level, r_chain_locked, r_parent_locked,
                   r_cpp_locked, r_needs_shrink, r_cached_ver, ipc, i_target>>

RAcquireGPForShrink_GP ==
    /\ rpc = "acquire_gp_for_shrink"
    /\ r_cut_level = 1
    /\ IF root_ver % 2 = 0
       THEN /\ root_ver' = root_ver + 1
            /\ r_cpp_locked' = TRUE
            /\ rpc' = "cut"
       ELSE /\ rpc' = "restart"
            /\ UNCHANGED <<root_ver, r_cpp_locked>>
    /\ UNCHANGED <<root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, p_ver, c_ver,
                   r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_needs_shrink, r_cached_ver, ipc, i_target>>

\* CUT — point of no return
RCut ==
    /\ rpc = "cut"
    /\ CASE r_cut_level = 0 ->           \* Root is CPP: empty tree
            /\ root_child' = FALSE
            /\ root_ver' = root_ver + 1  \* unlock root
            /\ gp_ver' = OBSOLETE        \* obsolete gp
            /\ p_ver' = OBSOLETE         \* obsolete parent
            /\ c_ver' = OBSOLETE         \* obsolete chain
            /\ UNCHANGED <<gp_count, gp_child, p_count, p_child>>
         [] r_cut_level = 1 ->           \* GP is CPP: remove parent from gp
            /\ gp_child' = FALSE
            /\ gp_count' = gp_count - 1
            /\ gp_ver' = gp_ver + 1     \* unlock gp
            /\ root_ver' = IF r_needs_shrink THEN root_ver + 1 ELSE root_ver
            /\ p_ver' = OBSOLETE         \* obsolete parent
            /\ c_ver' = OBSOLETE         \* obsolete chain
            /\ UNCHANGED <<root_child, p_count, p_child>>
         [] r_cut_level = 2 ->           \* Parent is CPP: remove chain from parent
            /\ p_child' = FALSE
            /\ p_count' = p_count - 1
            /\ p_ver' = p_ver + 1        \* unlock parent
            /\ c_ver' = OBSOLETE         \* obsolete chain
            /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver>>
         [] r_cut_level = 3 ->           \* Chain-only (no additional chain on stack)
            /\ c_ver' = OBSOLETE         \* obsolete chain
            /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                           p_count, p_child, p_ver>>
    /\ r_chain_locked' = FALSE /\ r_parent_locked' = FALSE
    /\ r_gp_locked' = FALSE /\ r_cpp_locked' = FALSE
    /\ rpc' = "done"
    /\ UNCHANGED <<r_cut_level, r_needs_shrink, r_cached_ver, ipc, i_target>>

\* Restart: release all held locks
RRestart ==
    /\ rpc = "restart"
    /\ c_ver' = IF r_chain_locked THEN c_ver + 1 ELSE c_ver
    /\ p_ver' = IF r_parent_locked THEN p_ver + 1 ELSE p_ver
    /\ gp_ver' = IF r_gp_locked THEN gp_ver + 1 ELSE gp_ver
    /\ root_ver' = IF r_cpp_locked /\ r_cut_level = 0
                      THEN root_ver + 1
                   ELSE IF r_needs_shrink /\ r_cut_level = 1
                      THEN root_ver + 1
                   ELSE root_ver
    /\ r_chain_locked' = FALSE /\ r_parent_locked' = FALSE
    /\ r_gp_locked' = FALSE /\ r_cpp_locked' = FALSE
    /\ rpc' = "idle"
    /\ UNCHANGED <<root_child, gp_count, gp_child, p_count, p_child,
                   r_cut_level, r_needs_shrink, r_cached_ver, ipc, i_target>>

RDone ==
    /\ rpc = "done"
    /\ rpc' = "idle"
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, p_ver, c_ver,
                   r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_cpp_locked, r_needs_shrink, r_cached_ver, ipc, i_target>>

-----------------------------------------------------------------------------
\* INSERTER: can add children to parent or gp (prevents Case B, prevents shrink)

IChoose ==
    /\ ipc = "idle"
    /\ \E t \in {"parent", "gp"} :
        /\ (t = "parent" => p_count < 3)
        /\ (t = "gp" => gp_count < 3)
        /\ i_target' = t
    /\ ipc' = "lock"
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, gp_ver,
                   p_count, p_child, p_ver, c_ver,
                   rpc, r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_cpp_locked, r_needs_shrink, r_cached_ver>>

ILock ==
    /\ ipc = "lock"
    /\ IF i_target = "parent"
       THEN /\ p_ver % 2 = 0 /\ p_ver <= MaxVersion
            /\ p_ver' = p_ver + 1
            /\ UNCHANGED gp_ver
       ELSE /\ gp_ver % 2 = 0 /\ gp_ver <= MaxVersion
            /\ gp_ver' = gp_ver + 1
            /\ UNCHANGED p_ver
    /\ ipc' = "write"
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, p_count, p_child, c_ver,
                   rpc, r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_cpp_locked, r_needs_shrink, r_cached_ver, i_target>>

IWrite ==
    /\ ipc = "write"
    /\ IF i_target = "parent"
       THEN /\ p_count' = p_count + 1 /\ UNCHANGED gp_count
       ELSE /\ gp_count' = gp_count + 1 /\ UNCHANGED p_count
    /\ ipc' = "unlock"
    /\ UNCHANGED <<root_ver, root_child, gp_child, gp_ver, p_child, p_ver, c_ver,
                   rpc, r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_cpp_locked, r_needs_shrink, r_cached_ver, i_target>>

IUnlock ==
    /\ ipc = "unlock"
    /\ IF i_target = "parent"
       THEN /\ p_ver' = p_ver + 1 /\ UNCHANGED gp_ver
       ELSE /\ gp_ver' = gp_ver + 1 /\ UNCHANGED p_ver
    /\ ipc' = "idle"
    /\ UNCHANGED <<root_ver, root_child, gp_count, gp_child, p_count, p_child, c_ver,
                   rpc, r_cut_level, r_chain_locked, r_parent_locked, r_gp_locked,
                   r_cpp_locked, r_needs_shrink, r_cached_ver, i_target>>

-----------------------------------------------------------------------------
Step ==
    \/ RLockChain \/ REvalParent \/ RValidateParent \/ RUpgradeParent
    \/ RCaseBParent \/ REvalGP \/ RValidateGP \/ RUpgradeGP
    \/ RCaseBGP \/ RAcquireRoot
    \/ RCheckShrink \/ RAcquireGPForShrink_Parent \/ RAcquireGPForShrink_GP
    \/ RCut \/ RRestart \/ RDone
    \/ IChoose \/ ILock \/ IWrite \/ IUnlock

Spec == Init /\ [][Step]_vars

StateConstraint ==
    /\ root_ver <= MaxVersion
    /\ (gp_ver # OBSOLETE => gp_ver <= MaxVersion)
    /\ (p_ver # OBSOLETE => p_ver <= MaxVersion)
    /\ (c_ver # OBSOLETE => c_ver <= MaxVersion)

-----------------------------------------------------------------------------
\* INVARIANTS

\* Cut only when appropriate locks held and state valid
CutSafety ==
    rpc = "cut" =>
        /\ r_chain_locked
        /\ (r_cut_level = 0 => r_cpp_locked)       \* root locked
        /\ (r_cut_level = 1 => r_cpp_locked)       \* gp locked as CPP
        /\ (r_cut_level = 2 => r_cpp_locked)       \* parent locked as CPP
        /\ (r_cut_level <= 1 => r_parent_locked)   \* parent locked as chain member
        /\ (r_cut_level = 0 => r_gp_locked)        \* gp locked as chain member

\* Well-formedness: counts stay valid
WellFormedness ==
    /\ gp_count >= 1
    /\ p_count >= 1

\* Root emptied only when cut_level=0 AND tree still has content
RootEmptyOnlyWhenFullCut ==
    (rpc = "cut" /\ r_cut_level = 0) => (root_child = TRUE /\ gp_child = TRUE /\ p_child = TRUE)

\* Shrink only attempted when CPP will be undersized
ShrinkCorrectness ==
    rpc = "acquire_gp_for_shrink" =>
        \/ (r_cut_level = 2 /\ p_count <= 2)
        \/ (r_cut_level = 1 /\ gp_count <= 2)

=============================================================================
