--------------------------- MODULE OLCIterator_Bug850 -------------------------
(* Bug injection: simulate the #850 backtrack skip bug.

   The real bug: after exhausting a node's children and popping it,
   the iterator failed to continue ascending — it stopped and reported
   end() prematurely, skipping all leaves in sibling subtrees.

   We inject this by changing IterAdvance: when a node is exhausted,
   instead of popping and continuing to try the parent, go directly
   to "done". This should violate Completeness.

   Also tests: if IterAdvance skips the version check, the model
   should catch NoStaleAct or OrderPreservation violations.
*)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    ConcurrencyEnabled,
    MaxVersion,
    BugMode          \* "skip_backtrack" | "skip_check" | "none"

Nodes == {"Root", "A", "B", "L1", "L2", "L3", "L4"}
Inodes == {"Root", "A", "B"}
Leaves == {"L1", "L2", "L3", "L4"}

ScanOrder == <<"L1", "L2", "L3", "L4">>

Children(n) ==
    CASE n = "Root" -> <<"A", "B">>
      [] n = "A"    -> <<"L1", "L2">>
      [] n = "B"    -> <<"L3", "L4">>
      [] OTHER      -> <<>>

NumChildren(n) == Len(Children(n))

IsLeaf(n) == n \in Leaves
IsInode(n) == n \in Inodes

VARIABLES ver, ipc, stack, visited, restarts, wpc, w_target

vars == <<ver, ipc, stack, visited, restarts, wpc, w_target>>

Entry(node, ci, v) == [node |-> node, ci |-> ci, ver |-> v]

CheckOK(node, snap_ver) ==
    IF ~ConcurrencyEnabled THEN TRUE
    ELSE ver[node] = snap_ver /\ (snap_ver % 2 = 0)

CanLock(node) ==
    IF ~ConcurrencyEnabled THEN TRUE
    ELSE ver[node] % 2 = 0

NextChild(n, ci) ==
    IF ci + 1 <= NumChildren(n) THEN ci + 1
    ELSE 0

GetChild(n, ci) == Children(n)[ci]

LeafPos(leaf) ==
    CASE leaf = "L1" -> 1
      [] leaf = "L2" -> 2
      [] leaf = "L3" -> 3
      [] leaf = "L4" -> 4
      [] OTHER -> 0

SeekTarget ==
    IF visited = <<>> THEN 1
    ELSE LeafPos(visited[Len(visited)]) + 1

SeekPath(pos) ==
    CASE pos = 1 -> <<1, 1>>
      [] pos = 2 -> <<1, 2>>
      [] pos = 3 -> <<2, 1>>
      [] pos = 4 -> <<2, 2>>
      [] OTHER -> <<>>

Init ==
    /\ ver = [n \in Nodes |-> 0]
    /\ ipc = "start"
    /\ stack = <<>>
    /\ visited = <<>>
    /\ restarts = 0
    /\ wpc = "idle"
    /\ w_target = "Root"

\* --- Iterator ---

IterStart == ipc = "start" /\
    LET target == IF BugMode = "restart_from_beginning" THEN 1
                  ELSE SeekTarget
    IN
    IF target > Len(ScanOrder) THEN
        /\ ipc' = "done"
        /\ UNCHANGED <<ver, stack, visited, restarts, wpc, w_target>>
    ELSE IF ~CanLock("Root") THEN
        /\ restarts' = restarts + 1
        /\ UNCHANGED <<ver, ipc, stack, visited, wpc, w_target>>
    ELSE
        LET path == SeekPath(target) IN
        /\ ipc' = "descend"
        /\ stack' = <<Entry("Root", path[1], ver["Root"])>>
        /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>

IterDescend == ipc = "descend" /\ stack /= <<>> /\
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
        LET target == SeekTarget
            path == SeekPath(target)
            depth == Len(stack) + 1
            next_ci == IF depth <= Len(path) THEN path[depth] ELSE 1
        IN
        /\ ipc' = "descend"
        /\ stack' = Append(stack, Entry(child, next_ci, ver[child]))
        /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>

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
        /\ ipc' = "descend_lm"
        /\ stack' = Append(stack, Entry(child, 1, ver[child]))
        /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>

\* BUG INJECTION: IterAdvance
IterAdvance == ipc = "advance" /\ stack /= <<>> /\
    LET top == stack[Len(stack)]
        node == top.node
        ci   == top.ci
    IN
    \* BUG: skip_check — don't validate version before acting
    IF BugMode /= "skip_check" /\ ~CheckOK(node, top.ver) THEN
        /\ ipc' = "start"
        /\ stack' = <<>>
        /\ restarts' = restarts + 1
        /\ UNCHANGED <<ver, visited, wpc, w_target>>
    ELSE
        LET nci == NextChild(node, ci) IN
        IF nci /= 0 THEN
            /\ stack' = [stack EXCEPT ![Len(stack)].ci = nci]
            /\ ipc' = "descend_lm"
            /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>
        ELSE
            \* BUG: skip_backtrack — stop instead of ascending to parent
            IF BugMode = "skip_backtrack" THEN
                /\ ipc' = "done"
                /\ stack' = <<>>
                /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>
            ELSE
                /\ stack' = SubSeq(stack, 1, Len(stack) - 1)
                /\ ipc' = IF Len(stack) = 1 THEN "done" ELSE "advance"
                /\ UNCHANGED <<ver, visited, restarts, wpc, w_target>>

IterDone == ipc = "done" /\ UNCHANGED vars

\* --- Writer ---

WriterLock == ConcurrencyEnabled /\ wpc = "idle" /\
    \E n \in Inodes :
        /\ ver[n] % 2 = 0
        /\ ver[n] + 1 <= MaxVersion
        /\ wpc' = "locked"
        /\ w_target' = n
        /\ ver' = [ver EXCEPT ![n] = ver[n] + 1]
        /\ UNCHANGED <<ipc, stack, visited, restarts>>

WriterUnlock == ConcurrencyEnabled /\ wpc = "locked" /\
    /\ ver[w_target] + 1 <= MaxVersion
    /\ ver' = [ver EXCEPT ![w_target] = ver[w_target] + 1]
    /\ wpc' = "idle"
    /\ UNCHANGED <<ipc, stack, visited, restarts, w_target>>

Next ==
    \/ IterStart
    \/ IterDescend
    \/ IterDescendLM
    \/ IterAdvance
    \/ IterDone
    \/ WriterLock
    \/ WriterUnlock

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* --- Constraints and Properties ---

StateConstraint == restarts <= 3

TypeOK ==
    /\ \A n \in Nodes : ver[n] \in 0..MaxVersion
    /\ ipc \in {"start", "descend", "descend_lm", "advance", "done"}
    /\ visited \in Seq(Leaves)
    /\ restarts \in Nat

OrderPreservation ==
    /\ Len(visited) <= Len(ScanOrder)
    /\ \A i \in 1..Len(visited) : visited[i] = ScanOrder[i]

NoRepeat ==
    \A i, j \in 1..Len(visited) : i /= j => visited[i] /= visited[j]

Completeness ==
    ipc = "done" => visited = ScanOrder

=============================================================================
