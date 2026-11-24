-------------------------- MODULE OLCChainCutVsSplit -----------------------
\* Stage 14: a chain cut against a prefix split of a node the cut relies on.
\*
\* try_chain_cut (olc_art.hpp) locks a chain bottom-up, finds the cut-point
\* parent, and writes the one link that disconnects the chain.  A prefix
\* split moves the version of the node it splits without obsoleting it, and
\* installs a new I4 above that node.  So the cut can find a node it relies on
\* live at a new version, with a link above or below it rewritten.  Stages 5,
\* 9 and 10 state that these links always hold.  This stage models the two
\* checks that make the cut restart instead (commit "Check every link a chain
\* cut relies on"):
\*   LinkCheck: Step 3 re-reads a node whose version moved and requires that
\*      it still leads to the node below it (`pcp->load() !=
\*      stk[cut_level].node`).  The pre-fix code had a Step 2 fallback that
\*      relocked a moved chain node at its current version if it was still a
\*      one-child I4, without that check.  LinkCheck = FALSE models that
\*      fallback.
\*   RootCheck: lock_root_pointer_to_stk0() write-locks the root pointer only
\*      while it still points to stk[0], in Step 3 at cut_level 0 and in Step
\*      4.1 when a root cut-point parent shrinks.  The pre-fix code took the
\*      root pointer lock fresh and wrote the root unchecked.
\*
\* Stack, top-down: R = stk[0] (the root), C1 = stk[1], C2 = stk[2].  C2 is
\* the chain node whose read lock the remove upgraded before calling
\* try_chain_cut (its parent_guard), with the chain below C2 and the leaf
\* already write-locked; that part is abstract.  C1 is a one-child I4 chain
\* node.  Shape:
\*   "root_chain":  R is a one-child I4 too.  The whole stack is the chain,
\*                  cut_level reaches 0, and Step 4.2 sets root = nullptr.
\*   "root_shrink": R is a min-size inode with other children: the cut-point
\*                  parent.  Step 4.1 locks the root pointer, and Step 4.2
\*                  replaces R in it with R shrunk by the chain's entry.
\*   "inner_shrink": as root_shrink, but R is below the root, under an inode
\*                  G = stk[-1].  The variables root and rp_ver then stand
\*                  for G's slot for R and G's version.  Step 4.1 locks G by
\*                  upgrading it at its descent version (rehydrate_read_lock
\*                  and check()), so a split of R, which moves G, restarts the
\*                  cut without any content check: the case that shows why
\*                  the root pointer, which has no descent version, needed
\*                  RootCheck.
\*
\* One insert, of a key K that splits a prefix:
\*   "split_root": K diverges inside R's prefix.  The insert upgrades the
\*                 root pointer (G in inner_shrink) and R, installs N
\*                 (children R and K) in it, and shortens R's prefix.
\*   "split_c1":   K diverges inside C1's prefix.  It upgrades R and C1,
\*                 points R's slot for C1 at Z (children C1 and K), and
\*                 shortens C1's prefix.
\* The insert takes its locks by upgrading the read locks of its descent, so
\* an upgrade fails, and the insert restarts, if the remove has moved or
\* holds that lock.  It then retries from a fresh descent, at most until the
\* version bound.  If the cut has obsoleted a node it needs, its retry inserts
\* K into the tree the cut left, where K stays reachable ("fresh").
\*
\* The remove's restart ends the behavior: a fresh remove descent is correct
\* by the other stages.
\*
\* Safety: NoLostInsert.  Step 4.2 computes what its write discards: root =
\* nullptr discards whatever the root pointer leads to; replacing R in the
\* root pointer discards N if the root pointed to N; removing R's entry for
\* the chain discards Z if R's slot pointed to Z.  The insert's key K must not
\* be discarded.
\*
\* OLCChainCutVsSplit.cfg, OLCChainCutVsSplit_Shrink.cfg and
\* OLCChainCutVsSplit_InnerShrink.cfg pass with both checks; InnerShrink
\* passes without RootCheck too.  OLCChainCutVsSplit_BugDemo_PreFix.cfg turns
\* both checks off, as in the pre-fix code.  OLCChainCutVsSplit_BugDemo_NoRootCheck.cfg (root_chain) and
\* OLCChainCutVsSplit_BugDemo_NoRootCheckShrink.cfg (root_shrink) drop
\* RootCheck, and OLCChainCutVsSplit_BugDemo_NoLinkCheck.cfg drops LinkCheck;
\* each SHOULD violate NoLostInsert.

EXTENDS Integers

CONSTANTS
    Shape,        \* "root_chain", "root_shrink" or "inner_shrink"
    Inserts,      \* subset of {"split_root", "split_c1"}
    LinkCheck,    \* re-read moved nodes only with the link check (Step 3)
    RootCheck,    \* lock the root pointer only while it points to R
    MaxVersion

OBSOLETE == -1

VARIABLES
    \* Shared tree
    root,       \* "R" | "N" | "R_shrunk" | "null"
    rp_ver,     \* root pointer lock version
    r_ver,      \* R's version (OBSOLETE once cut out)
    c1_ver,     \* C1's version
    r_slot,     \* R's slot for C1's key byte: "C1" | "Z"

    \* Remove (try_chain_cut)
    rpc,
    r_held_c1,  \* holds C1's write lock
    r_held_r,   \* holds R's write lock
    r_held_rp,  \* holds the root pointer's write lock
    r_read,     \* version of the node Step 3 read-locked
    cut_level,  \* 2: C2 is the top of what is locked; 1: C1; 0: R

    \* Insert
    ipc,
    i_kind,     \* "split_root" | "split_c1"
    i_v1,       \* descent version of the first node it upgrades
    i_v2,       \* descent version of the second
    key_at,     \* where K is: "none" | "N" | "Z" | "fresh"
    lost        \* Step 4.2 discarded K

vars == <<root, rp_ver, r_ver, c1_ver, r_slot, rpc, r_held_c1, r_held_r,
          r_held_rp, r_read, cut_level, ipc, i_kind, i_v1, i_v2, key_at, lost>>

tree == <<root, rp_ver, r_ver, c1_ver, r_slot, key_at, lost>>
rem  == <<rpc, r_held_c1, r_held_r, r_held_rp, r_read, cut_level>>
ins  == <<ipc, i_kind, i_v1, i_v2>>

Even(v) == v >= 0 /\ v % 2 = 0

\* The remove's descent read R and C1 at version 0.
DescentVer == 0

\* R is a one-child I4 in root_chain, and stays one after either split: a
\* split of R installs N above it, a split of C1 replaces R's only child.
ROneChildI4 == Shape = "root_chain"

-----------------------------------------------------------------------------
Init ==
    /\ root = "R" /\ rp_ver = 0 /\ r_ver = 0 /\ c1_ver = 0 /\ r_slot = "C1"
    /\ rpc = "step2_c1" /\ r_held_c1 = FALSE /\ r_held_r = FALSE
    /\ r_held_rp = FALSE /\ r_read = 0 /\ cut_level = 2
    /\ ipc = "idle" /\ i_kind \in Inserts /\ i_v1 = 0 /\ i_v2 = 0
    /\ key_at = "none" /\ lost = FALSE

-----------------------------------------------------------------------------
\* REMOVE

\* Step 2: upgrade a chain node at its descent version.  A node whose version
\* moved fails, and it and the nodes above it are left to Step 3.  With
\* LinkCheck = FALSE, the pre-fix fallback instead re-reads it at its current
\* version and locks it if it is still a one-child I4, without checking its
\* child.
Step2Lock(ver, held, next, isOneChildI4) ==
    IF ver = DescentVer
    THEN /\ ver' = ver + 1 /\ held' = TRUE
         /\ rpc' = next /\ cut_level' = cut_level - 1
    ELSE IF ~LinkCheck /\ Even(ver) /\ isOneChildI4
    THEN /\ ver' = ver + 1 /\ held' = TRUE
         /\ rpc' = next /\ cut_level' = cut_level - 1
    ELSE /\ rpc' = "step3" /\ UNCHANGED <<ver, held, cut_level>>

\* C1 is a one-child I4 chain node whatever the insert does to it.
RStep2C1 ==
    /\ rpc = "step2_c1"
    /\ Step2Lock(c1_ver, r_held_c1,
                 IF Shape = "root_chain" THEN "step2_r" ELSE "step3", TRUE)
    /\ UNCHANGED <<root, rp_ver, r_ver, r_slot, key_at, lost, r_held_r,
                   r_held_rp, r_read>>
    /\ UNCHANGED ins

\* Step 2 stops at the chain top; R is in the chain only in root_chain.  A
\* Step 2 that has left C1 to Step 3 does not reach R (its loop breaks).
RStep2R ==
    /\ rpc = "step2_r"
    /\ Step2Lock(r_ver, r_held_r, "step3", ROneChildI4)
    /\ UNCHANGED <<root, rp_ver, c1_ver, r_slot, key_at, lost, r_held_c1,
                   r_held_rp, r_read>>
    /\ UNCHANGED ins

\* Step 3 at cut_level 0, or Step 4.1 for a shrinking root cut-point parent:
\* read-lock the root pointer, check it with RootCheck, upgrade.  For
\* inner_shrink, Step 4.1 upgrades G at its descent version instead.
RLockRootPointer ==
    /\ rpc \in {"step3", "gp_root"}
    /\ (rpc = "step3" => cut_level = 0)
    /\ Shape = "inner_shrink" \/ Even(rp_ver)
    /\ IF IF Shape = "inner_shrink" THEN rp_ver # DescentVer
          ELSE RootCheck /\ root # "R"
       THEN /\ rpc' = "restart" /\ UNCHANGED <<rp_ver, r_held_rp>>
       ELSE /\ rp_ver' = rp_ver + 1 /\ r_held_rp' = TRUE /\ rpc' = "cut"
    /\ UNCHANGED <<root, r_ver, c1_ver, r_slot, key_at, lost, r_held_c1,
                   r_held_r, r_read, cut_level>>
    /\ UNCHANGED ins

\* Step 3 at cut_level > 0: read-lock stk[cut_level - 1] at its descent
\* version or, if that moved, at its current version, and check that its slot
\* still leads to stk[cut_level].  C1's slot for C2 never changes here.
RStep3Read ==
    /\ rpc = "step3" /\ cut_level > 0
    /\ LET v == IF cut_level = 2 THEN c1_ver ELSE r_ver
           linkOK == cut_level = 2 \/ r_slot = "C1"
       IN /\ Even(v)
          /\ r_read' = v
          /\ rpc' = IF linkOK THEN "step3_lock" ELSE "restart"
    /\ UNCHANGED <<root, rp_ver, r_ver, c1_ver, r_slot, key_at, lost,
                   r_held_c1, r_held_r, r_held_rp, cut_level>>
    /\ UNCHANGED ins

\* Upgrade it.  A one-child I4 extends the chain (Case B); otherwise it is
\* the cut-point parent, and a shrinking one needs the root pointer next.  A
\* failed upgrade re-reads (the code's `continue`).
RStep3Lock ==
    /\ rpc = "step3_lock"
    /\ IF cut_level = 2
       THEN IF c1_ver = r_read
            THEN /\ c1_ver' = c1_ver + 1 /\ r_held_c1' = TRUE
                 /\ cut_level' = 1
                 /\ rpc' = "step3"
                 /\ UNCHANGED <<r_ver, r_held_r>>
            ELSE /\ rpc' = "step3"
                 /\ UNCHANGED <<c1_ver, r_held_c1, r_ver, r_held_r, cut_level>>
       ELSE IF r_ver = r_read
            THEN /\ r_ver' = r_ver + 1 /\ r_held_r' = TRUE
                 /\ IF ROneChildI4
                    THEN /\ cut_level' = 0 /\ rpc' = "step3"
                    ELSE /\ rpc' = "gp_root" /\ UNCHANGED cut_level
                 /\ UNCHANGED <<c1_ver, r_held_c1>>
            ELSE /\ rpc' = "step3"
                 /\ UNCHANGED <<c1_ver, r_held_c1, r_ver, r_held_r, cut_level>>
    /\ UNCHANGED <<root, rp_ver, r_slot, key_at, lost, r_held_rp, r_read>>
    /\ UNCHANGED ins

\* Step 4.2: the point of no return.
RCut ==
    /\ rpc = "cut"
    /\ IF Shape = "root_chain"
       THEN \* root = nullptr: everything the root pointer led to goes.
            /\ lost' = (key_at \in {"N", "Z"})
            /\ root' = "null"
            /\ r_ver' = OBSOLETE
       ELSE \* Replace R in the root pointer with R shrunk by its chain entry.
            /\ lost' = \/ key_at = "N"
                       \/ (key_at = "Z" /\ r_slot = "Z")
            /\ root' = "R_shrunk"
            /\ r_ver' = OBSOLETE
    /\ c1_ver' = OBSOLETE
    /\ rp_ver' = rp_ver + 1
    /\ r_held_c1' = FALSE /\ r_held_r' = FALSE /\ r_held_rp' = FALSE
    /\ rpc' = "done"
    /\ UNCHANGED <<r_slot, key_at, r_read, cut_level>>
    /\ UNCHANGED ins

\* Restart: release every lock held, which moves those versions.
RRestart ==
    /\ rpc = "restart"
    /\ c1_ver' = IF r_held_c1 THEN c1_ver + 1 ELSE c1_ver
    /\ r_ver' = IF r_held_r THEN r_ver + 1 ELSE r_ver
    /\ rp_ver' = IF r_held_rp THEN rp_ver + 1 ELSE rp_ver
    /\ r_held_c1' = FALSE /\ r_held_r' = FALSE /\ r_held_rp' = FALSE
    /\ rpc' = "restarted"
    /\ UNCHANGED <<root, r_slot, key_at, lost, r_read, cut_level>>
    /\ UNCHANGED ins

-----------------------------------------------------------------------------
\* INSERT

\* The two locks the insert upgrades, top-down.
I1Ver == IF i_kind = "split_root" THEN rp_ver ELSE r_ver
I2Ver == IF i_kind = "split_root" THEN r_ver ELSE c1_ver

\* Descent: read both nodes at unlocked versions.  An obsolete node means the
\* cut already ran: the insert's retry finds the tree the cut left and adds K
\* there.
IDescend ==
    /\ ipc = "idle" /\ key_at = "none"
    /\ IF I1Ver = OBSOLETE \/ I2Ver = OBSOLETE \/ root # "R"
       THEN /\ key_at' = "fresh" /\ ipc' = "done"
            /\ UNCHANGED <<i_v1, i_v2>>
       ELSE /\ Even(I1Ver) /\ Even(I2Ver)
            /\ i_v1' = I1Ver /\ i_v2' = I2Ver
            /\ ipc' = "up1"
            /\ UNCHANGED key_at
    /\ UNCHANGED <<root, rp_ver, r_ver, c1_ver, r_slot, lost, i_kind>>
    /\ UNCHANGED rem

\* Upgrade the first lock; on failure retry the descent.
IUp1 ==
    /\ ipc = "up1"
    /\ IF I1Ver = i_v1
       THEN /\ IF i_kind = "split_root"
               THEN /\ rp_ver' = rp_ver + 1 /\ UNCHANGED r_ver
               ELSE /\ r_ver' = r_ver + 1 /\ UNCHANGED rp_ver
            /\ ipc' = "up2"
       ELSE /\ ipc' = "idle" /\ UNCHANGED <<rp_ver, r_ver>>
    /\ UNCHANGED <<root, c1_ver, r_slot, key_at, lost, i_kind, i_v1, i_v2>>
    /\ UNCHANGED rem

\* Upgrade the second lock and split; on failure release the first, which
\* moves its version, and retry.
IUp2 ==
    /\ ipc = "up2"
    /\ IF I2Ver = i_v2
       THEN \* Split, then unlock both: each version moves by 2 in all.
            /\ IF i_kind = "split_root"
               THEN /\ root' = "N" /\ key_at' = "N"
                    /\ rp_ver' = rp_ver + 1 /\ r_ver' = r_ver + 2
                    /\ UNCHANGED <<r_slot, c1_ver>>
               ELSE /\ r_slot' = "Z" /\ key_at' = "Z"
                    /\ r_ver' = r_ver + 1 /\ c1_ver' = c1_ver + 2
                    /\ UNCHANGED <<root, rp_ver>>
            /\ ipc' = "done"
       ELSE /\ IF i_kind = "split_root"
               THEN /\ rp_ver' = rp_ver + 1 /\ UNCHANGED r_ver
               ELSE /\ r_ver' = r_ver + 1 /\ UNCHANGED rp_ver
            /\ ipc' = "idle"
            /\ UNCHANGED <<root, r_slot, key_at, c1_ver>>
    /\ UNCHANGED <<lost, i_kind, i_v1, i_v2>>
    /\ UNCHANGED rem

-----------------------------------------------------------------------------
Terminated ==
    /\ rpc \in {"done", "restarted"}
    /\ ipc = "done"
    /\ UNCHANGED vars

Step ==
    \/ RStep2C1 \/ RStep2R \/ RLockRootPointer \/ RStep3Read \/ RStep3Lock
    \/ RCut \/ RRestart
    \/ IDescend \/ IUp1 \/ IUp2
    \/ Terminated

Spec == Init /\ [][Step]_vars

StateConstraint ==
    /\ rp_ver <= MaxVersion /\ r_ver <= MaxVersion /\ c1_ver <= MaxVersion

-----------------------------------------------------------------------------
\* SAFETY

\* Reachability guard, which OLCChainCutVsSplit*_Reach.cfg must violate: a
\* split completes and the cut restarts instead of losing its key.
ReachRestartAfterSplit == ~(rpc = "restarted" /\ key_at \in {"N", "Z"})

NoLostInsert == ~lost

\* The cut never writes without the locks Step 4.2 needs.
CutHoldsLocks == rpc = "cut" => (r_held_c1 /\ r_held_r /\ r_held_rp)

TypeOK ==
    /\ root \in {"R", "N", "R_shrunk", "null"}
    /\ r_slot \in {"C1", "Z"}
    /\ key_at \in {"none", "N", "Z", "fresh"}
    /\ rpc \in {"step2_c1", "step2_r", "step3", "step3_lock", "gp_root",
                "cut", "restart", "restarted", "done"}
    /\ ipc \in {"idle", "up1", "up2", "done"}
    /\ cut_level \in 0..2

=============================================================================
