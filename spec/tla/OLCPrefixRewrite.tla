--------------------------- MODULE OLCPrefixRewrite -----------------------
\* Stage 13: an OLC descent against an in-place rewrite of an inode's key
\* prefix.
\*
\* Two writers rewrite the key prefix word of an inode S that stays live, so
\* the descent's read lock on S is not obsoleted:
\*   "collapse": a remove leaves S as the only child of its parent I4 P.  It
\*      write-locks P, prepends P's prefix and dispatch byte to S's prefix
\*      (the prefix gets longer, word "long"), links S where P was, and
\*      obsoletes P.  olc_db::collapse_i4() and olc_inode_4::leave_last_child().
\*      With SurvivorLock it also write-locks S first
\*      (try_write_lock_i4_survivor(), "Lock the surviving child when an OLC
\*      I4 collapses"); before that commit it did not.
\*   "split": an insert whose key diverges inside S's prefix.  It write-locks
\*      P, which is S's parent, and S, puts a new I4 between them and cuts S's
\*      prefix (word "short").  try_insert / try_upsert, the
\*      "shared_prefix_length < key_prefix_length" arms.
\*
\* One descent, for a key stored below S, reached S through P, so it is at S's old depth and the only
\* prefix word that is correct there is "orig".  ReaderKind orders its steps
\* as one of the code's descents does:
\*   "get":    read-lock S, then validate P (try_get, try_remove_key_view,
\*             the iterator's traversals and try_seek)
\*   "upsert": validate P, then read-lock S (try_upsert: the
\*             sync_in_upsert_before_node_lock point comes after the parent's
\*             check).  With ParentRecheck it validates P again before it uses
\*             what it read at S ("Check the parent before an OLC upsert's
\*             value-in-slot lambda", and must_restart_on_key_ending_in_prefix()).
\*             The model applies the recheck to every use.  In the code the
\*             other arms either upgrade P's read lock to a write lock, which
\*             validates it, or reach a leaf whose key they compare.
\*             try_insert orders its steps the same way.
\*   "remove_chain": validate P, then read-lock S, as try_remove_key_view
\*             does when S is the first chain node of a run: P's last check
\*             is remove_or_choose_subtree's, or the root pointer's, before
\*             S is read-locked.  With HandOverCheck it validates P again
\*             when it hands P's read lock over to S, before it acts on what
\*             it read at S ("Check a chain run's parent before the key_view
\*             remove descends past it"); before that commit it dropped P's
\*             read lock unchecked, and only S was validated from then on.
\* Then it reads S's prefix length and bytes: in one load with SingleLoad
\* (match_key_prefix(), "Read each OLC descent's key prefix in one load"), or
\* in two before it.
\*
\* A descent at the old depth that reads a rewritten prefix compares the
\* wrong bytes.  The comparison can mismatch (the descent would report the
\* key absent), match the wrong bytes (it would go on at the wrong depth), or,
\* for a "long" prefix only, match up to the key's end, since the key is
\* shorter than the lengthened prefix.  Without KeyEndCheck
\* (key_matches_prefix_to_end(), "Check for a key that ends inside an OLC
\* inode's key prefix") the last case then indexes the key past its end.
\*
\* The bytes compared are abstract: which word was read, and which outcome
\* the comparison had, not the bytes themselves.  A restarted descent starts
\* over from above P; once the writer has finished it reaches S at S's new
\* depth, where the rewritten word is the correct one, so r_expect follows the
\* word.  The writer runs once.
\*
\* Safety:
\*   NoTornPrefix:       the length and bytes compared come from one word
\*   NoKeyOverrun:       the descent never indexes the key past its end
\*   NoStaleDepthResult: a descent that finishes used the prefix word that is
\*                       correct at its depth
\*
\* OLCPrefixRewrite.cfg (get), OLCPrefixRewrite_Upsert.cfg (upsert) and
\* OLCPrefixRewrite_RemoveChain.cfg (remove_chain) pass with every fix.  Each
\* of the five OLCPrefixRewrite_BugDemo_*.cfg drops one fix and SHOULD violate
\* the invariant it names.

EXTENDS Integers

CONSTANTS
    ReaderKind,     \* "get" or "upsert"
    Writers,        \* subset of {"collapse", "split"}
    SurvivorLock,   \* collapse write-locks S
    SingleLoad,     \* prefix length and bytes from one load
    KeyEndCheck,    \* check for a key that ends inside the prefix
    ParentRecheck,  \* upsert validates P again before using S's contents
    HandOverCheck   \* remove_chain validates P when it hands P's lock to S

OBSOLETE == -1

VARIABLES
    p_ver,      \* P's version: even unlocked, odd locked, OBSOLETE
    s_ver,      \* S's version: even unlocked, odd locked
    word,       \* S's prefix word: "orig", "long" (collapse), "short" (split)

    wpc,        \* writer pc
    w_kind,     \* "collapse" or "split"
    w_s_locked, \* the writer holds S's write lock

    rpc,        \* reader pc
    r_p_ver,    \* P's version at the descent's read of P
    r_s_ver,    \* S's version at the descent's read lock of S
    r_len_w,    \* word the prefix length was loaded from
    r_bytes_w,  \* word the prefix bytes were loaded from
    r_expect,   \* the word that is correct at the descent's depth
    r_outcome,  \* "match" | "mismatch" | "wrong_match" | "key_end"
    r_restarted \* the descent restarted and reached S from its new parent

vars == <<p_ver, s_ver, word, wpc, w_kind, w_s_locked, rpc, r_p_ver, r_s_ver,
          r_len_w, r_bytes_w, r_expect, r_outcome, r_restarted>>

wvars == <<wpc, w_kind, w_s_locked>>
rvars == <<rpc, r_p_ver, r_s_ver, r_len_w, r_bytes_w, r_expect, r_outcome,
           r_restarted>>

Even(v) == v >= 0 /\ v % 2 = 0

-----------------------------------------------------------------------------
Init ==
    /\ p_ver = 0 /\ s_ver = 0 /\ word = "orig"
    /\ wpc = "idle" /\ w_kind \in Writers /\ w_s_locked = FALSE
    \* The descent has read-locked P and found S's slot in it.
    /\ rpc = "at_p" /\ r_p_ver = 0 /\ r_s_ver = 0
    /\ r_len_w = "orig" /\ r_bytes_w = "orig" /\ r_expect = "orig"
    /\ r_outcome = "match" /\ r_restarted = FALSE

-----------------------------------------------------------------------------
\* WRITER

WLockP ==
    /\ wpc = "idle"
    /\ Even(p_ver)
    /\ p_ver' = p_ver + 1
    /\ wpc' = "lock_s"
    /\ UNCHANGED <<s_ver, word, w_kind, w_s_locked>>
    /\ UNCHANGED rvars

\* A split always locks S.  A collapse locks it only with SurvivorLock, and
\* then without waiting: try_read_lock_no_wait() on a locked S restarts the
\* collapse, which here only waits, as no other writer holds S.
WLockS ==
    /\ wpc = "lock_s"
    /\ IF w_kind = "split" \/ SurvivorLock
       THEN /\ Even(s_ver)
            /\ s_ver' = s_ver + 1
            /\ w_s_locked' = TRUE
       ELSE UNCHANGED <<s_ver, w_s_locked>>
    /\ wpc' = "rewrite"
    /\ UNCHANGED <<p_ver, word, w_kind>>
    /\ UNCHANGED rvars

\* The prefix rewrite: key_prefix::prepend() or the split's cut, one store
\* of the prefix word.
WRewrite ==
    /\ wpc = "rewrite"
    /\ word' = IF w_kind = "collapse" THEN "long" ELSE "short"
    /\ wpc' = "unlock"
    /\ UNCHANGED <<p_ver, s_ver, w_kind, w_s_locked>>
    /\ UNCHANGED rvars

\* A collapse obsoletes P; a split unlocks it, which moves its version.
WUnlock ==
    /\ wpc = "unlock"
    /\ p_ver' = IF w_kind = "collapse" THEN OBSOLETE ELSE p_ver + 1
    /\ s_ver' = IF w_s_locked THEN s_ver + 1 ELSE s_ver
    /\ w_s_locked' = FALSE
    /\ wpc' = "done"
    /\ UNCHANGED <<word, w_kind>>
    /\ UNCHANGED rvars

-----------------------------------------------------------------------------
\* READER

\* Read-lock S: try_read_lock() spins while S is write-locked.
LockS(next) ==
    /\ Even(s_ver)
    /\ r_s_ver' = s_ver
    /\ rpc' = next

\* Validate S's parent at the version the descent read it at.  After a
\* restart that parent is the grandparent (collapse) or the new I4 (split),
\* which the writer, running once, no longer changes.
PValid == r_restarted \/ p_ver = r_p_ver

RGetLockS ==
    /\ rpc = "at_p" /\ ReaderKind = "get"
    /\ LockS("validate_p")
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_len_w, r_bytes_w, r_expect,
                   r_outcome>>
    /\ UNCHANGED <<wvars, r_restarted>>

RGetValidateP ==
    /\ rpc = "validate_p"
    /\ rpc' = IF PValid THEN "read_len" ELSE "restart"
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_s_ver, r_len_w, r_bytes_w,
                   r_expect, r_outcome>>
    /\ UNCHANGED <<wvars, r_restarted>>

RUpsertValidateP ==
    /\ rpc = "at_p" /\ ReaderKind \in {"upsert", "remove_chain"}
    /\ rpc' = IF PValid THEN "lock_s" ELSE "restart"
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_s_ver, r_len_w, r_bytes_w,
                   r_expect, r_outcome>>
    /\ UNCHANGED <<wvars, r_restarted>>

RUpsertLockS ==
    /\ rpc = "lock_s"
    /\ LockS("read_len")
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_len_w, r_bytes_w, r_expect,
                   r_outcome>>
    /\ UNCHANGED <<wvars, r_restarted>>

\* get_snapshot() loads the length and the bytes together; key_prefix's
\* length() and get_shared_length() load them separately.
RReadLen ==
    /\ rpc = "read_len"
    /\ r_len_w' = word
    /\ IF SingleLoad
       THEN /\ r_bytes_w' = word
            /\ rpc' = "compare"
       ELSE /\ rpc' = "read_bytes"
            /\ UNCHANGED r_bytes_w
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_s_ver, r_expect, r_outcome>>
    /\ UNCHANGED <<wvars, r_restarted>>

RReadBytes ==
    /\ rpc = "read_bytes"
    /\ r_bytes_w' = word
    /\ rpc' = "compare"
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_s_ver, r_len_w, r_expect,
                   r_outcome>>
    /\ UNCHANGED <<wvars, r_restarted>>

\* Comparing the key with the prefix.  With the correct word the key matches;
\* with a rewritten one any outcome its length allows is possible.
Outcomes(w) ==
    IF w = r_expect THEN {"match"}
    ELSE IF w = "long" THEN {"mismatch", "wrong_match", "key_end"}
    ELSE {"mismatch", "wrong_match"}

RCompare ==
    /\ rpc = "compare"
    /\ \E o \in Outcomes(r_bytes_w) :
         /\ r_outcome' = o
         /\ rpc' = IF o = "key_end" /\ ~KeyEndCheck THEN "index_past_end"
                   ELSE "use"
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_s_ver, r_len_w, r_bytes_w,
                   r_expect>>
    /\ UNCHANGED <<wvars, r_restarted>>

\* The key ended inside the prefix and the descent indexed it there anyway.
RIndexPastEnd ==
    /\ rpc = "index_past_end"
    /\ rpc' = "finished"
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_s_ver, r_len_w, r_bytes_w,
                   r_expect, r_outcome>>
    /\ UNCHANGED <<wvars, r_restarted>>

\* Before the descent acts on what it read at S (returns, continues, or hands
\* it to an upsert lambda), it validates S.  An upsert with ParentRecheck,
\* and a chain remove with HandOverCheck, validate P too.
RUse ==
    /\ rpc = "use"
    /\ LET ok == /\ s_ver = r_s_ver
                 /\ (ReaderKind = "upsert" /\ ParentRecheck) => PValid
                 /\ (ReaderKind = "remove_chain" /\ HandOverCheck) => PValid
       IN rpc' = IF ok THEN "finished" ELSE "restart"
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_s_ver, r_len_w, r_bytes_w,
                   r_expect, r_outcome>>
    /\ UNCHANGED <<wvars, r_restarted>>

\* Start over from above P.  try_read_lock() on the root path waits for the
\* writer's locks, so the restart lands after the writer finishes, at S's
\* depth after the rewrite.
RRestart ==
    /\ rpc = "restart"
    /\ wpc = "done"
    /\ rpc' = "at_p_new"
    /\ r_expect' = word
    /\ r_restarted' = TRUE
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_s_ver, r_len_w, r_bytes_w,
                   r_outcome>>
    /\ UNCHANGED wvars

\* At S's new depth with no concurrent writer: read-lock S and read its
\* prefix again.
RRestartAtS ==
    /\ rpc = "at_p_new"
    /\ LockS("read_len")
    /\ UNCHANGED <<p_ver, s_ver, word, r_p_ver, r_len_w, r_bytes_w, r_expect,
                   r_outcome>>
    /\ UNCHANGED <<wvars, r_restarted>>

\* Both done: the behavior ends.
Terminated ==
    /\ rpc = "finished" /\ wpc = "done"
    /\ UNCHANGED vars

-----------------------------------------------------------------------------
Step ==
    \/ Terminated
    \/ WLockP \/ WLockS \/ WRewrite \/ WUnlock
    \/ RGetLockS \/ RGetValidateP \/ RUpsertValidateP \/ RUpsertLockS
    \/ RReadLen \/ RReadBytes \/ RCompare \/ RIndexPastEnd \/ RUse
    \/ RRestart \/ RRestartAtS

Spec == Init /\ [][Step]_vars

-----------------------------------------------------------------------------
\* SAFETY

NoTornPrefix == rpc = "compare" => r_len_w = r_bytes_w

NoKeyOverrun == rpc # "index_past_end"

NoStaleDepthResult ==
    rpc = "finished" => (r_bytes_w = r_expect /\ r_outcome = "match")

TypeOK ==
    /\ p_ver \in {OBSOLETE} \cup 0..3
    /\ s_ver \in 0..3
    /\ word \in {"orig", "long", "short"}
    /\ wpc \in {"idle", "lock_s", "rewrite", "unlock", "done"}
    /\ w_kind \in {"collapse", "split"}
    /\ ReaderKind \in {"get", "upsert", "remove_chain"}
    /\ rpc \in {"at_p", "validate_p", "lock_s", "read_len", "read_bytes",
                "compare", "index_past_end", "use", "restart", "at_p_new",
                "finished"}
    /\ r_outcome \in {"match", "mismatch", "wrong_match", "key_end"}

=============================================================================
