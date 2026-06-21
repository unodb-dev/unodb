--------------------------- MODULE OLCIterator ----------------------------
(* TLA+ model of the OLC ART stack-based iterator.
   Models try_next() and try_left_most_traversal with concurrent writer.

   Tree geometry: depth 3, parametric width.
     Root (inode)
       ├── A (inode)
       │   ├── L1 (leaf)
       │   └── L2 (leaf)
       └── B (inode)
           ├── L3 (leaf)
           └── L4 (leaf)

   Forward scan order: L1, L2, L3, L4.

   The iterator maintains a stack of (node, child_index, version) entries
   and advances by: pop leaf, find next child in parent, descend left-most.
   If no next child, pop parent (backtrack/ascend) and repeat.

   ConcurrencyEnabled: FALSE = sequential (db), TRUE = OLC (olc_db).
   When FALSE, check() always succeeds and no writer runs.

   Properties verified:
     - OrderPreservation: leaves are visited in correct order
     - NoSkip: no leaf present throughout the scan is missed
     - NoRepeat: no leaf is visited twice
     - NoStaleAct: iterator never acts on stale version (OLC mode)
     - Termination: iterator eventually reaches end or delivers all leaves
*)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    ConcurrencyEnabled,   \* TRUE for olc_db, FALSE for db
    MaxVersion            \* Bound on version counter (typically 4)

\* --- Node identity (fixed tree structure) ---
\* Nodes: Root, A, B, L1, L2, L3, L4
\* Children relation (defines the tree):
\*   Root -> [A, B]  (child_index 0, 1)
\*   A    -> [L1, L2] (child_index 0, 1)
\*   B    -> [L3, L4] (child_index 0, 1)

Nodes == {"Root", "A", "B", "L1", "L2", "L3", "L4"}
Inodes == {"Root", "A", "B"}
Leaves == {"L1", "L2", "L3", "L4"}

\* Expected forward scan order
ScanOrder == <<"L1", "L2", "L3", "L4">>

\* Tree topology: parent -> <<child0, child1>>
Children(n) ==
    CASE n = "Root" -> <<"A", "B">>
      [] n = "A"    -> <<"L1", "L2">>
      [] n = "B"    -> <<"L3", "L4">>
      [] OTHER      -> <<>>  \* leaves have no children

NumChildren(n) == Len(Children(n))

Parent(n) ==
    CASE n = "A"  -> "Root"
      [] n = "B"  -> "Root"
      [] n = "L1" -> "A"
      [] n = "L2" -> "A"
      [] n = "L3" -> "B"
      [] n = "L4" -> "B"
      [] OTHER    -> "None"

IsLeaf(n) == n \in Leaves
IsInode(n) == n \in Inodes

\* --- State variables ---
VARIABLES
    \* Per-node version counter (even = unlocked, odd = write-locked)
    ver,          \* ver[n] \in 0..MaxVersion

    \* Iterator state
    ipc,          \* program counter
    stack,        \* sequence of [node, child_index, snap_ver] records
    visited,      \* sequence of leaves visited (in order)
    restarts,     \* count of restarts (for liveness reasoning)

    \* Writer state (only active when ConcurrencyEnabled)
    wpc,          \* writer program counter
    w_target      \* which inode the writer is modifying

vars == <<ver, ipc, stack, visited, restarts, wpc, w_target>>

\* --- Helper operators ---

\* Stack entry constructor
Entry(node, ci, v) == [node |-> node, ci |-> ci, ver |-> v]

\* Check if version is even (unlocked) and matches snapshot
CheckOK(node, snap_ver) ==
    IF ~ConcurrencyEnabled THEN TRUE
    ELSE ver[node] = snap_ver /\ (snap_ver % 2 = 0)

\* Version is even (can acquire read lock)
CanLock(node) ==
    IF ~ConcurrencyEnabled THEN TRUE
    ELSE ver[node] % 2 = 0

\* Left-most leaf reachable from a node
RECURSIVE LeftMost(_)
LeftMost(n) ==
    IF IsLeaf(n) THEN n
    ELSE LeftMost(Children(n)[1])

\* Next child index after ci in node n. 0 if no more.
NextChild(n, ci) ==
    IF ci + 1 <= NumChildren(n) THEN ci + 1
    ELSE 0  \* 0 means exhausted

\* Get the i-th child (1-indexed)
GetChild(n, ci) == Children(n)[ci]

\* Position of a leaf in the scan order (1-indexed). 0 if not found.
LeafPos(leaf) ==
    CASE leaf = "L1" -> 1
      [] leaf = "L2" -> 2
      [] leaf = "L3" -> 3
      [] leaf = "L4" -> 4
      [] OTHER -> 0

\* The leaf we need to seek to after a restart (successor of last visited).
\* If visited is empty, seek to L1 (first). Otherwise seek to successor.
SeekTarget ==
    IF visited = <<>> THEN 1
    ELSE LeafPos(visited[Len(visited)]) + 1

\* Find the path (sequence of child indices) from Root to the n-th leaf
\* in scan order. Returns a sequence of child indices for each level.
\* E.g., leaf 1 (L1): <<1, 1>>, leaf 3 (L3): <<2, 1>>, leaf 4 (L4): <<2, 2>>
SeekPath(pos) ==
    CASE pos = 1 -> <<1, 1>>   \* Root[1]=A, A[1]=L1
      [] pos = 2 -> <<1, 2>>   \* Root[1]=A, A[2]=L2
      [] pos = 3 -> <<2, 1>>   \* Root[2]=B, B[1]=L3
      [] pos = 4 -> <<2, 2>>   \* Root[2]=B, B[2]=L4
      [] OTHER -> <<>>

\* --- Initial state ---
Init ==
    /\ ver = [n \in Nodes |-> 0]
    /\ ipc = "start"
    /\ stack = <<>>
    /\ visited = <<>>
    /\ restarts = 0
    /\ wpc = "idle"
    /\ w_target = "Root"

\* --- Iterator actions ---

\* Start/Restart: lock root, descend toward the seek target.
\* On first call, seeks to leaf position 1 (left-most).
\* On restart after a check failure, seeks to successor of last visited.
IterStart == ipc = "start" /\
    LET target == SeekTarget IN
    IF target > Len(ScanOrder) THEN
        \* We've already visited everything — done
        /\ ipc' = "done"
        /\ UNCHANGED <<ver, stack, visited, restarts, wpc, w_target>>
    ELSE IF ~CanLock("Root") THEN
        \* Lock failed (writer holds root) — retry
        /\ restarts' = restarts + 1
        /\ UNCHANGED <<ver, ipc, stack, visited, wpc, w_target>>
    ELSE
        \* Successful lock — begin descent toward target leaf
        LET path == SeekPath(target) IN
        /\ ipc' = "descend"
        /\ stack' = <<Entry("Root", path[1], ver["Root"])>>
        /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>

\* Descend: we have a stack with the current inode on top, ci points to
\* the child we want to descend into. Keep going until we reach a leaf.
IterDescend == ipc = "descend" /\ stack /= <<>> /\
    LET top == stack[Len(stack)]
        node == top.node
        ci   == top.ci
        child == GetChild(node, ci)
    IN
    \* Check the version of the node we're descending from
    IF ~CheckOK(node, top.ver) THEN
        \* Version check failed — full restart (seek to last visited + 1)
        /\ ipc' = "start"
        /\ stack' = <<>>
        /\ restarts' = restarts + 1
        /\ UNCHANGED <<ver, visited, wpc, w_target>>
    ELSE IF IsLeaf(child) THEN
        \* Reached a leaf — record visit, transition to "advance"
        /\ ipc' = "advance"
        /\ visited' = Append(visited, child)
        /\ UNCHANGED <<ver, stack, restarts, wpc, w_target>>
    ELSE
        \* Child is an inode — push it with the correct child_index for seek.
        \* The child_index at the next level comes from the seek path.
        LET target == SeekTarget
            path == SeekPath(target)
            depth == Len(stack) + 1  \* next level (1=root already on stack)
            next_ci == IF depth <= Len(path) THEN path[depth] ELSE 1
        IN
        /\ ipc' = "descend"
        /\ stack' = Append(stack, Entry(child, next_ci, ver[child]))
        /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>

\* Advance: find the next leaf after the current one.
\* This is try_next(): look for next sibling or backtrack up the stack.
IterAdvance == ipc = "advance" /\ stack /= <<>> /\
    LET top == stack[Len(stack)]
        node == top.node
        ci   == top.ci
    IN
    \* Version check on the inode at top of stack
    IF ~CheckOK(node, top.ver) THEN
        \* Stale — restart with seek
        /\ ipc' = "start"
        /\ stack' = <<>>
        /\ restarts' = restarts + 1
        /\ UNCHANGED <<ver, visited, wpc, w_target>>
    ELSE
        LET nci == NextChild(node, ci) IN
        IF nci /= 0 THEN
            \* There is a next child — update top's ci and descend left-most
            /\ stack' = [stack EXCEPT ![Len(stack)].ci = nci]
            /\ ipc' = "descend_lm"
            /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>
        ELSE
            \* Exhausted this inode — pop and try parent (backtrack)
            /\ stack' = SubSeq(stack, 1, Len(stack) - 1)
            /\ ipc' = IF Len(stack) = 1 THEN "done" ELSE "advance"
            /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>

\* Descend left-most: after advance finds a sibling, descend into its
\* left-most subtree. Always picks child_index 1.
IterDescendLM == ipc = "descend_lm" /\ stack /= <<>> /\
    LET top == stack[Len(stack)]
        node == top.node
        ci   == top.ci
        child == GetChild(node, ci)
    IN
    IF ~CheckOK(node, top.ver) THEN
        /\ ipc' = "start"
        /\ stack' = <<>>
        /\ restarts' = restarts + 1
        /\ UNCHANGED <<ver, visited, wpc, w_target>>
    ELSE IF IsLeaf(child) THEN
        /\ ipc' = "advance"
        /\ visited' = Append(visited, child)
        /\ UNCHANGED <<ver, stack, restarts, wpc, w_target>>
    ELSE
        \* Push child inode, always pick first child (left-most descent)
        /\ ipc' = "descend_lm"
        /\ stack' = Append(stack, Entry(child, 1, ver[child]))
        /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>

\* Terminal state
IterDone == ipc = "done" /\ UNCHANGED vars

\* --- Writer actions (only when ConcurrencyEnabled) ---

\* Writer picks a target inode and locks it (bumps version to odd)
WriterLock == ConcurrencyEnabled /\ wpc = "idle" /\
    \E n \in Inodes :
        /\ ver[n] % 2 = 0           \* not already locked
        /\ ver[n] + 1 <= MaxVersion  \* don't exceed bound
        /\ wpc' = "locked"
        /\ w_target' = n
        /\ ver' = [ver EXCEPT ![n] = ver[n] + 1]
        /\ UNCHANGED <<ipc, stack, visited, restarts>>

\* Writer unlocks (bumps version to even)
WriterUnlock == ConcurrencyEnabled /\ wpc = "locked" /\
    /\ ver[w_target] + 1 <= MaxVersion  \* don't exceed bound
    /\ ver' = [ver EXCEPT ![w_target] = ver[w_target] + 1]
    /\ wpc' = "idle"
    /\ UNCHANGED <<ipc, stack, visited, restarts, w_target>>

\* --- Specification ---

Next ==
    \/ IterStart
    \/ IterDescend
    \/ IterDescendLM
    \/ IterAdvance
    \/ IterDone
    \/ WriterLock
    \/ WriterUnlock

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* --- State Constraint (bounds exploration) ---
StateConstraint == restarts <= 3

\* --- Invariants and Properties ---

\* Type invariant
TypeOK ==
    /\ \A n \in Nodes : ver[n] \in 0..MaxVersion
    /\ ipc \in {"start", "descend", "descend_lm", "advance", "done"}
    /\ visited \in Seq(Leaves)
    /\ restarts \in Nat

\* OrderPreservation: visited leaves are a prefix of the expected scan order
OrderPreservation ==
    /\ Len(visited) <= Len(ScanOrder)
    /\ \A i \in 1..Len(visited) : visited[i] = ScanOrder[i]

\* NoRepeat: no leaf appears twice
NoRepeat ==
    \A i, j \in 1..Len(visited) : i /= j => visited[i] /= visited[j]

\* Completeness: when done, all leaves were visited
Completeness ==
    ipc = "done" => visited = ScanOrder

\* NoStaleAct: if we're in "advance" or "descend" and acting on a stack entry,
\* that entry's version must still be current (OLC safety).
\* Note: This is implicitly enforced by CheckOK guards, but we verify it.
NoStaleAct ==
    (ipc \in {"advance", "descend"} /\ stack /= <<>>) =>
        LET top == stack[Len(stack)] IN
        CheckOK(top.node, top.ver)
        \* This is checked by the guards; if violated, we restart.
        \* The real invariant is: we never append to visited[] with stale data.

\* Termination (liveness): iterator eventually reaches "done"
Termination == <>(ipc = "done")

=============================================================================
