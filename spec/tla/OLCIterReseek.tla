---------------------------- MODULE OLCIterReseek -------------------------
\* Stage 15: an OLC iterator's next() re-seeking after a failed step.
\*
\* For key_view keys without a heap, the iterator's current key is its own
\* keybuf_.  try_next() pops and pushes keybuf_ bytes as it walks to the next
\* entry, before checks that can still fail: it pops the bytes of every level
\* it leaves, pushes the next child's dispatch byte, and the left-most descent
\* below pushes each level's bytes (olc_art.hpp, iterator::try_next() and
\* try_left_most_traversal()).  A byte read before a failing check may come
\* from a node a writer is rewriting, so it can be any value.  next() then
\* re-seeks to "the current key" and steps again.  Before "Re-seek a failed
\* OLC scan step to a copy of the current key" that key was a view of
\* keybuf_, so it re-sought to whatever the failed step left there.  With
\* SaveCopy it re-seeks to a copy saved before the step (save_reseek_key()).
\*
\* Tree: keys are two bytes.  The root dispatches on byte 1 to one inner node
\* per byte-1 value, which dispatches on byte 2 to a leaf.  Stable keys are
\* present throughout.  One writer inserts or removes one volatile key V once.
\* It write-locks V's inner node, or the root if V is the only key with its
\* first byte, since then the root gains or loses a child.
\*
\* One next() from cur0.  Its step:
\*   inner: read the next byte after cur's in cur's inner node, check the
\*          node.  Found: push it, check again (get_child), done.  Not found:
\*          pop to the root.
\*   root:  check the root, read the next first byte, push it, then descend
\*          left-most: read-lock that inner node, read its smallest byte,
\*          push it, check, done.
\* A failed check sends next() to re-seek: the first key >= the re-seek key.
\* If that key is the re-seek key itself, next() steps again from it,
\* otherwise it returns it.  The re-seek itself is atomic here: try_seek's
\* own descent is covered by Stage 13.
\*
\* Safety, for the key e that next() returns (END at the end):
\*   Monotonic:     e is after cur0
\*   NoSkipStable:  e is not after any stable key that is after cur0
\*
\* Not modeled: heap mode, where the re-seek key comes from the heap, and a
\* step that grows keybuf_ past its buffer, which left the old view pointing
\* into freed memory (a memory-safety defect, not an ordering one).
\*
\* OLCIterReseek.cfg passes with SaveCopy.  OLCIterReseek_BugDemo.cfg drops
\* it and SHOULD violate Monotonic or NoSkipStable.

EXTENDS Integers, Sequences, FiniteSets

CONSTANT SaveCopy  \* re-seek to a copy of the current key

\* V = <<0, 1>> shares cur's inner node; V = <<2, 0>> shares the inner node
\* that the left-most descent enters; V = <<1, 0>> is the only key with its
\* first byte, so its insert or remove writes the root.
Bytes == {0, 1, 2}
Stable == {<<0, 0>>, <<0, 2>>, <<2, 1>>}
Volatile == {<<0, 1>>, <<1, 0>>, <<2, 0>>}

NONE == -1
ROOT == -2   \* w_node value for the root
END == <<3>>  \* end(): sorts after every key

VARIABLES
    present,    \* set of present keys
    rver,       \* root version
    nver,       \* [Bytes -> version] of each inner node
    V,          \* the volatile key
    wpc,        \* "idle" | "write" | "unlock" | "done"
    w_node,     \* ROOT or the first byte of V's inner node

    cur0,       \* where next() started
    cur,        \* the key the step starts from
    ipc,
    it_rv,      \* root version the iterator's stack holds
    it_nv,      \* cur's inner node version the stack holds
    lm_v,       \* version of the inner node the left-most descent locked
    rb,         \* byte the step read
    nb1,        \* next first byte the step read at the root
    keybuf,     \* keybuf_
    saved,      \* the copy save_reseek_key() made
    e           \* the key next() returned

vars == <<present, rver, nver, V, wpc, w_node, cur0, cur, ipc, it_rv, it_nv,
          lm_v, rb, nb1, keybuf, saved, e>>
wv == <<present, rver, nver, V, wpc, w_node>>

Even(v) == v % 2 = 0

\* Lexicographic order; a one-byte prefix sorts before its extensions.
Lt(x, y) ==
    \/ x[1] < y[1]
    \/ x[1] = y[1] /\ Len(x) < Len(y)
    \/ x[1] = y[1] /\ Len(x) = 2 /\ Len(y) = 2 /\ x[2] < y[2]

Min(S) == CHOOSE k \in S : \A j \in S : ~Lt(j, k)

InNode(a) == {k \in present : k[1] = a}
NextByte(a, b) ==
    LET S == {k[2] : k \in {j \in InNode(a) : j[2] > b}}
    IN IF S = {} THEN NONE ELSE CHOOSE x \in S : \A y \in S : x <= y
NextFirst(a) ==
    LET S == {k[1] : k \in {j \in present : j[1] > a}}
    IN IF S = {} THEN NONE ELSE CHOOSE x \in S : \A y \in S : x <= y
MinByte(a) == CHOOSE x \in {k[2] : k \in InNode(a)} :
                  \A y \in {k[2] : k \in InNode(a)} : x <= y

\* A read of a node a writer holds locked can return any byte.
Torn(a) == w_node = a /\ ~Even(nver[a])
RootTorn == w_node = ROOT /\ ~Even(rver)

-----------------------------------------------------------------------------
Init ==
    /\ V \in Volatile
    /\ present \in {Stable, Stable \cup {V}}
    /\ rver = 0 /\ nver = [a \in Bytes |-> 0]
    /\ wpc = "idle"
    /\ w_node = IF \E k \in Stable : k[1] = V[1] THEN V[1] ELSE ROOT
    /\ cur0 \in Stable /\ cur = cur0
    /\ ipc = "save" /\ it_rv = 0 /\ it_nv = 0 /\ lm_v = 0
    /\ rb = NONE /\ nb1 = NONE
    /\ keybuf = cur0 /\ saved = cur0 /\ e = <<-1>>

-----------------------------------------------------------------------------
\* WRITER: lock, toggle V, unlock.

WLock ==
    /\ wpc = "idle"
    /\ IF w_node = ROOT THEN rver' = rver + 1 /\ UNCHANGED nver
       ELSE nver' = [nver EXCEPT ![w_node] = @ + 1] /\ UNCHANGED rver
    /\ wpc' = "write"
    /\ UNCHANGED <<present, V, w_node>>

WWrite ==
    /\ wpc = "write"
    /\ present' = IF V \in present THEN present \ {V} ELSE present \cup {V}
    /\ wpc' = "unlock"
    /\ UNCHANGED <<rver, nver, V, w_node>>

WUnlock ==
    /\ wpc = "unlock"
    /\ IF w_node = ROOT THEN rver' = rver + 1 /\ UNCHANGED nver
       ELSE nver' = [nver EXCEPT ![w_node] = @ + 1] /\ UNCHANGED rver
    /\ wpc' = "done"
    /\ UNCHANGED <<present, V, w_node>>

Writer == (WLock \/ WWrite \/ WUnlock) /\ UNCHANGED <<cur0, cur, ipc, it_rv,
              it_nv, lm_v, rb, nb1, keybuf, saved, e>>

-----------------------------------------------------------------------------
\* ITERATOR

Fail == ipc' = "reseek"

ISave ==
    /\ ipc = "save"
    /\ saved' = keybuf
    /\ ipc' = "inner_read"
    /\ UNCHANGED <<cur0, cur, it_rv, it_nv, lm_v, rb, nb1, keybuf, e>>

\* inode->next(): read the next byte after cur's in cur's inner node.
IInnerRead ==
    /\ ipc = "inner_read"
    /\ IF Torn(cur[1]) THEN rb' \in Bytes \cup {NONE}
       ELSE rb' = NextByte(cur[1], cur[2])
    /\ ipc' = "inner_check"
    /\ UNCHANGED <<cur0, cur, it_rv, it_nv, lm_v, nb1, keybuf, saved, e>>

\* check(); then pop() and either push(e2) or pop to the root.
IInnerCheck ==
    /\ ipc = "inner_check"
    /\ IF nver[cur[1]] # it_nv
       THEN Fail /\ UNCHANGED keybuf
       ELSE IF rb = NONE
       THEN keybuf' = <<cur[1]>> /\ ipc' = "root_check"
       ELSE keybuf' = <<cur[1], rb>> /\ ipc' = "inner_check2"
    /\ UNCHANGED <<cur0, cur, it_rv, it_nv, lm_v, rb, nb1, saved, e>>

\* get_child() and its check(), then push_leaf: done.
IInnerCheck2 ==
    /\ ipc = "inner_check2"
    /\ IF nver[cur[1]] # it_nv THEN Fail /\ UNCHANGED e
       ELSE e' = <<cur[1], rb>> /\ ipc' = "done"
    /\ UNCHANGED <<cur0, cur, it_rv, it_nv, lm_v, rb, nb1, keybuf, saved>>

\* At the root: rehydrate and check it, read its next child, pop and push.
IRootCheck ==
    /\ ipc = "root_check"
    /\ IF rver # it_rv THEN Fail /\ UNCHANGED <<nb1, keybuf, e>>
       ELSE /\ nb1' \in IF RootTorn THEN Bytes ELSE {NextFirst(cur[1])}
            /\ IF nb1' = NONE
               THEN e' = END /\ ipc' = "done" /\ UNCHANGED keybuf
               ELSE keybuf' = <<nb1'>> /\ ipc' = "root_check2"
                    /\ UNCHANGED e
    /\ UNCHANGED <<cur0, cur, it_rv, it_nv, lm_v, rb, saved>>

\* get_child() and its check() at the root.
IRootCheck2 ==
    /\ ipc = "root_check2"
    /\ IF rver # it_rv THEN Fail ELSE ipc' = "lm_lock"
    /\ UNCHANGED <<cur0, cur, it_rv, it_nv, lm_v, rb, nb1, keybuf, saved, e>>

\* try_left_most_traversal(): read-lock the inner node (waits while locked).
ILmLock ==
    /\ ipc = "lm_lock"
    /\ Even(nver[nb1])
    /\ lm_v' = nver[nb1]
    /\ ipc' = "lm_read"
    /\ UNCHANGED <<cur0, cur, it_rv, it_nv, rb, nb1, keybuf, saved, e>>

\* begin(), push its byte, check().  An empty node is a node a remove has
\* emptied, which moved its version.
ILmRead ==
    /\ ipc = "lm_read"
    /\ IF Torn(nb1) THEN rb' \in Bytes
       ELSE rb' = IF InNode(nb1) = {} THEN NONE ELSE MinByte(nb1)
    /\ keybuf' = IF rb' = NONE THEN keybuf ELSE <<nb1, rb'>>
    /\ ipc' = "lm_check"
    /\ UNCHANGED <<cur0, cur, it_rv, it_nv, lm_v, nb1, saved, e>>

ILmCheck ==
    /\ ipc = "lm_check"
    /\ IF nver[nb1] # lm_v \/ rb = NONE THEN Fail /\ UNCHANGED e
       ELSE e' = <<nb1, rb>> /\ ipc' = "done"
    /\ UNCHANGED <<cur0, cur, it_rv, it_nv, lm_v, rb, nb1, keybuf, saved>>

\* try_seek(akey, match, true): waits out the writer, then finds the first
\* key >= akey.  A match steps again from it, from a fresh stack.
IReseek ==
    /\ ipc = "reseek"
    /\ wpc \in {"idle", "done"}
    /\ LET akey == IF SaveCopy THEN saved ELSE keybuf
           ge == {k \in present : ~Lt(k, akey)}
       IN IF ge = {} THEN e' = END /\ ipc' = "done"
                          /\ UNCHANGED <<cur, it_rv, it_nv, keybuf, saved>>
          ELSE LET f == Min(ge) IN
               IF f = akey
               THEN /\ cur' = f /\ keybuf' = f /\ saved' = f
                    /\ it_rv' = rver /\ it_nv' = nver[f[1]]
                    /\ ipc' = "inner_read" /\ UNCHANGED e
               ELSE /\ e' = f /\ ipc' = "done"
                    /\ UNCHANGED <<cur, it_rv, it_nv, keybuf, saved>>
    /\ UNCHANGED <<cur0, lm_v, rb, nb1>>

\* Both finished.  A next() that finished first waits for the writer.
IDone == ipc = "done" /\ wpc = "done" /\ UNCHANGED vars

Iterator ==
    (ISave \/ IInnerRead \/ IInnerCheck \/ IInnerCheck2 \/ IRootCheck
     \/ IRootCheck2 \/ ILmLock \/ ILmRead \/ ILmCheck \/ IReseek)
    /\ UNCHANGED wv

Next == Writer \/ Iterator \/ IDone

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* SAFETY

Monotonic == ipc = "done" => (e = END \/ Lt(cur0, e))

NoSkipStable ==
    ipc = "done" =>
        \A s \in Stable : Lt(cur0, s) => (e # END /\ ~Lt(s, e))

=============================================================================
