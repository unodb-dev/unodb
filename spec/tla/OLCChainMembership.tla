--------------------------- MODULE OLCChainMembership ---------------------
\* Stage 4: Chain membership precondition under concurrent insert.
\*
\* One remover evaluates whether an I4 node is a chain member (count==1).
\* One inserter can add a child to that same I4 concurrently.
\* The remover uses OLC revalidation: read count → validate version → act.
\*
\* Verifies: remover never proceeds with cut when count is actually 2.

EXTENDS Integers

CONSTANTS
    MaxVersion

VARIABLES
    \* Shared I4 node state
    count,      \* 1 or 2 children
    version,    \* Even=unlocked, Odd=locked

    \* Remover state (evaluating chain membership)
    rpc,        \* idle|read_count|validate|upgrade|locked|done|restart
    r_count,    \* Cached count value
    r_ver,      \* Cached version

    \* Inserter state
    ipc         \* idle|lock|insert|unlock|done

vars == <<count, version, rpc, r_count, r_ver, ipc>>

-----------------------------------------------------------------------------
Init ==
    /\ count = 1        \* starts as chain member (single child)
    /\ version = 0
    /\ rpc = "idle"
    /\ r_count = 0
    /\ r_ver = 0
    /\ ipc = "idle"

-----------------------------------------------------------------------------
\* INSERTER: adds a child (count 1→2)

InsertLock ==
    /\ ipc = "idle"
    /\ version % 2 = 0
    /\ version <= MaxVersion
    /\ count < 2           \* only insert if room
    /\ version' = version + 1
    /\ ipc' = "lock"
    /\ UNCHANGED <<count, rpc, r_count, r_ver>>

InsertWrite ==
    /\ ipc = "lock"
    /\ count' = count + 1
    /\ ipc' = "unlock"
    /\ UNCHANGED <<version, rpc, r_count, r_ver>>

InsertUnlock ==
    /\ ipc = "unlock"
    /\ version' = version + 1
    /\ ipc' = "done"
    /\ UNCHANGED <<count, rpc, r_count, r_ver>>

-----------------------------------------------------------------------------
\* REMOVER: evaluates chain membership via OLC

RemoverStart ==
    /\ rpc = "idle"
    /\ rpc' = "read_count"
    /\ r_ver' = version       \* capture version
    /\ UNCHANGED <<count, version, r_count, ipc>>

RemoverReadCount ==
    /\ rpc = "read_count"
    /\ r_count' = count       \* read shared count
    /\ rpc' = "validate"
    /\ UNCHANGED <<count, version, r_ver, ipc>>

RemoverValidateOK ==
    /\ rpc = "validate"
    /\ version = r_ver        \* unchanged
    /\ version % 2 = 0       \* not locked
    /\ IF r_count = 1
       THEN rpc' = "upgrade"  \* chain member, try to lock
       ELSE rpc' = "not_chain" \* not a chain member
    /\ UNCHANGED <<count, version, r_count, r_ver, ipc>>

RemoverValidateFail ==
    /\ rpc = "validate"
    /\ (version # r_ver \/ version % 2 = 1)
    /\ rpc' = "restart"
    /\ UNCHANGED <<count, version, r_count, r_ver, ipc>>

RemoverUpgrade ==
    /\ rpc = "upgrade"
    /\ IF version = r_ver /\ version % 2 = 0
       THEN /\ version' = version + 1
            /\ rpc' = "locked"
       ELSE /\ rpc' = "restart"
            /\ UNCHANGED version
    /\ UNCHANGED <<count, r_count, r_ver, ipc>>

RemoverDone ==
    /\ rpc = "locked"
    /\ rpc' = "done"
    /\ version' = version + 1  \* unlock
    /\ UNCHANGED <<count, r_count, r_ver, ipc>>

RemoverNotChain ==
    /\ rpc = "not_chain"
    /\ rpc' = "done"
    /\ UNCHANGED <<count, version, r_count, r_ver, ipc>>

RemoverRestart ==
    /\ rpc = "restart"
    /\ rpc' = "idle"
    /\ UNCHANGED <<count, version, r_count, r_ver, ipc>>

-----------------------------------------------------------------------------
Step ==
    \/ InsertLock \/ InsertWrite \/ InsertUnlock
    \/ RemoverStart \/ RemoverReadCount
    \/ RemoverValidateOK \/ RemoverValidateFail
    \/ RemoverUpgrade \/ RemoverDone \/ RemoverNotChain \/ RemoverRestart

Spec == Init /\ [][Step]_vars

-----------------------------------------------------------------------------
\* INVARIANTS

\* CRITICAL: Remover never holds lock (proceeds with cut) when count=2.
\* If remover is in "locked" state, the actual count MUST be 1.
ChainSafety ==
    rpc = "locked" => count = 1

\* If remover decided "not_chain", it read count#1 under valid version.
\* (This is a weaker check — mainly for debugging.)
NotChainCorrect ==
    rpc = "not_chain" => r_count # 1

=============================================================================
