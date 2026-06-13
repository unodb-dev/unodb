--------------------------- MODULE ARTTreeMaintenance ---------------------------
(***************************************************************************)
(* Stage 12: ART Tree Structure Maintenance                                *)
(*                                                                         *)
(* Verifies sequential correctness of insert/remove on an ART tree,        *)
(* focusing on chain nodes and try_collapse_i4.                            *)
(*                                                                         *)
(* Key encoding: path from root spells the key. Each node contributes      *)
(* prefix bytes, then 1 dispatch byte selects a child. Terminal node's     *)
(* prefix consumes final bytes; is_vis=TRUE marks key presence.            *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    PrefixCapacity   \* Max prefix bytes per node (2)

\* Fixed key set exercising all variety dimensions
Keys == {<<1>>, <<1, 2>>, <<1, 2, 1>>, <<2, 1>>, <<1, 2, 1, 2, 1>>}

VARIABLES
    nodes,           \* node_id :> [prefix, children, is_vis]
    root,            \* node_id of root, or 0 if empty
    present,         \* Set of keys currently stored
    nxt              \* Next fresh node ID

vars == <<nodes, root, present, nxt>>

NULL == 0
Alphabet == {1, 2}
EmptyFcn == [x \in {} |-> x]

(***************************************************************************)
(* KEY REACHABILITY                                                        *)
(***************************************************************************)

RECURSIVE KeyAt(_, _, _)
KeyAt(nid, key, pos) ==
    IF nid = NULL \/ nid \notin DOMAIN nodes THEN FALSE
    ELSE
      LET n == nodes[nid]
          plen == Len(n.prefix)
      IN
      IF plen > Len(key) - pos + 1 THEN FALSE
      ELSE IF plen > 0 /\ SubSeq(key, pos, pos + plen - 1) /= n.prefix THEN FALSE
      ELSE
        LET ap == pos + plen IN
        IF ap > Len(key) THEN n.is_vis
        ELSE
            LET d == key[ap] IN
            IF d \notin DOMAIN n.children THEN FALSE
            ELSE KeyAt(n.children[d], key, ap + 1)

(***************************************************************************)
(* REACHABILITY                                                            *)
(***************************************************************************)

RECURSIVE Reachable(_)
Reachable(nid) ==
    IF nid = NULL \/ nid \notin DOMAIN nodes THEN {}
    ELSE LET n == nodes[nid] IN
         {nid} \union UNION {Reachable(n.children[b]) : b \in DOMAIN n.children}

(***************************************************************************)
(* INSERT — Build Chain                                                    *)
(***************************************************************************)

RECURSIVE MkChain(_, _, _)
MkChain(key, pos, base) ==
    LET remaining == Len(key) - pos + 1 IN
    IF remaining <= 0 THEN
        <<(base :> [prefix |-> <<>>, children |-> EmptyFcn, is_vis |-> TRUE]),
          base, base + 1>>
    ELSE IF remaining <= PrefixCapacity THEN
        <<(base :> [prefix |-> SubSeq(key, pos, Len(key)),
                    children |-> EmptyFcn, is_vis |-> TRUE]),
          base, base + 1>>
    ELSE
        LET plen == IF remaining - 1 <= PrefixCapacity
                    THEN remaining - 1 ELSE PrefixCapacity
            pfx == SubSeq(key, pos, pos + plen - 1)
            d == key[pos + plen]
            sub == MkChain(key, pos + plen + 1, base + 1)
        IN <<(base :> [prefix |-> pfx,
                       children |-> (d :> sub[2]),
                       is_vis |-> FALSE]) @@ sub[1],
            base, sub[3]>>

(***************************************************************************)
(* INSERT — Recursive Core                                                 *)
(***************************************************************************)

\* Returns <<new_nodes, new_root, new_nxt, replacement_id>>
RECURSIVE DoInsert(_, _, _, _, _, _)
DoInsert(ns, rt, key, pos, nid, nx) ==
    LET n == ns[nid]
        plen == Len(n.prefix)
        key_rem == Len(key) - pos + 1
        match_max == IF plen < key_rem THEN plen ELSE key_rem
        RECURSIVE PfxMatch(_)
        PfxMatch(i) ==
            IF i > match_max THEN match_max
            ELSE IF key[pos + i - 1] /= n.prefix[i] THEN i - 1
            ELSE PfxMatch(i + 1)
        mlen == IF plen = 0 THEN 0 ELSE PfxMatch(1)
    IN
    IF mlen < plen THEN
        \* PREFIX SPLIT
        LET sb == n.prefix[mlen + 1]
            osuf == IF mlen + 2 > plen THEN <<>>
                    ELSE SubSeq(n.prefix, mlen + 2, plen)
            bpfx == IF mlen = 0 THEN <<>>
                    ELSE SubSeq(n.prefix, 1, mlen)
            kpos == pos + mlen
            ns2 == [ns EXCEPT ![nid].prefix = osuf]
            \* Check if nid (with new prefix osuf) is collapsible
            nid_collapsible ==
                /\ ~n.is_vis
                /\ Cardinality(DOMAIN n.children) = 1
                /\ LET cd == CHOOSE b \in DOMAIN n.children : TRUE
                       cid == n.children[cd]
                   IN /\ cid \in DOMAIN ns
                      /\ Len(osuf) + 1 + Len(ns[cid].prefix) <= PrefixCapacity
        IN
        IF kpos > Len(key) THEN
            LET bid == nx
                bn == [prefix |-> bpfx, children |-> (sb :> nid), is_vis |-> TRUE]
                raw == <<(bid :> bn) @@ ns2, IF nid = rt THEN bid ELSE rt, nx + 1, bid>>
            IN IF nid_collapsible THEN
                 LET cd == CHOOSE b \in DOMAIN n.children : TRUE
                     cid == n.children[cd]
                     new_pfx == osuf \o <<cd>> \o ns[cid].prefix
                     ns_c == [nd \in (DOMAIN raw[1] \ {nid}) |->
                               IF nd = cid
                               THEN [raw[1][cid] EXCEPT !.prefix = new_pfx]
                               ELSE raw[1][nd]]
                     ns_d == [ns_c EXCEPT ![bid].children = (sb :> cid)]
                 IN <<ns_d, raw[2], raw[3], raw[4]>>
               ELSE raw
        ELSE
            LET kd == key[kpos]
                ch == MkChain(key, kpos + 1, nx + 1)
                bid == nx
                bc == (sb :> nid) @@ (kd :> ch[2])
                bn == [prefix |-> bpfx, children |-> bc, is_vis |-> FALSE]
                raw == <<(bid :> bn) @@ ch[1] @@ ns2,
                         IF nid = rt THEN bid ELSE rt, ch[3], bid>>
            IN IF nid_collapsible THEN
                 LET cd == CHOOSE b \in DOMAIN n.children : TRUE
                     cid == n.children[cd]
                     new_pfx == osuf \o <<cd>> \o ns[cid].prefix
                     ns_c == [nd \in (DOMAIN raw[1] \ {nid}) |->
                               IF nd = cid
                               THEN [raw[1][cid] EXCEPT !.prefix = new_pfx]
                               ELSE raw[1][nd]]
                     ns_d == [ns_c EXCEPT ![bid].children[sb] = cid]
                 IN <<ns_d, raw[2], raw[3], raw[4]>>
               ELSE raw
    ELSE
        \* FULL PREFIX MATCH
        LET ap == pos + plen IN
        IF ap > Len(key) THEN
            <<[ns EXCEPT ![nid].is_vis = TRUE], rt, nx, nid>>
        ELSE
            LET d == key[ap] IN
            IF d \in DOMAIN n.children THEN
                LET cid == n.children[d]
                    sub == DoInsert(ns, rt, key, ap + 1, cid, nx)
                    ns1 == IF sub[4] /= cid
                           THEN [sub[1] EXCEPT ![nid].children[d] = sub[4]]
                           ELSE sub[1]
                    \* Check if nid is now collapsible
                    n1 == ns1[nid]
                    single == Cardinality(DOMAIN n1.children) = 1
                              /\ ~n1.is_vis /\ nid /= sub[2]
                IN
                IF single THEN
                    LET cd2 == CHOOSE b \in DOMAIN n1.children : TRUE
                        cid2 == n1.children[cd2]
                        combined == Len(n1.prefix) + 1 + Len(ns1[cid2].prefix)
                    IN
                    IF combined <= PrefixCapacity THEN
                        LET new_pfx == n1.prefix \o <<cd2>> \o ns1[cid2].prefix
                            ns2 == [nd \in (DOMAIN ns1 \ {nid}) |->
                                      IF nd = cid2
                                      THEN [ns1[cid2] EXCEPT !.prefix = new_pfx]
                                      ELSE ns1[nd]]
                        IN <<ns2, sub[2], sub[3], cid2>>
                    ELSE <<ns1, sub[2], sub[3], nid>>
                ELSE <<ns1, sub[2], sub[3], nid>>
            ELSE
                LET ch == MkChain(key, ap + 1, nx)
                    ns2 == [ns EXCEPT ![nid].children = n.children @@ (d :> ch[2])]
                IN <<ch[1] @@ ns2, rt, ch[3], nid>>

(***************************************************************************)
(* INSERT — Top-level Action                                               *)
(***************************************************************************)

Insert(key) ==
    /\ key \notin present
    /\ IF root = NULL THEN
           LET ch == MkChain(key, 1, nxt)
           IN /\ nodes' = ch[1]
              /\ root' = ch[2]
              /\ present' = present \union {key}
              /\ nxt' = ch[3]
       ELSE
           LET r == DoInsert(nodes, root, key, 1, root, nxt)
           IN /\ nodes' = r[1]
              /\ root' = r[2]
              /\ present' = present \union {key}
              /\ nxt' = r[3]

(***************************************************************************)
(* REMOVE — Path Tracing                                                   *)
(***************************************************************************)

\* Returns <<..., <<nid, dispatch>>, ...>>. dispatch=0 means VIS at nid.
RECURSIVE TraceFrom(_, _, _, _)
TraceFrom(ns, nid, key, pos) ==
    IF nid \notin DOMAIN ns THEN <<>>
    ELSE
      LET n == ns[nid]
          plen == Len(n.prefix)
      IN
      IF plen > Len(key) - pos + 1 THEN <<>>
      ELSE IF plen > 0 /\ SubSeq(key, pos, pos + plen - 1) /= n.prefix THEN <<>>
      ELSE
        LET ap == pos + plen IN
        IF ap > Len(key) THEN
            IF n.is_vis THEN <<<<nid, 0>>>> ELSE <<>>
        ELSE
            LET d == key[ap] IN
            IF d \notin DOMAIN n.children THEN <<>>
            ELSE
                LET cid == n.children[d]
                    rest == TraceFrom(ns, cid, key, ap + 1)
                IN
                IF rest /= <<>> THEN <<<<nid, d>>>> \o rest
                ELSE
                    \* Check if child is terminal leaf
                    IF cid \in DOMAIN ns
                       /\ ns[cid].is_vis
                       /\ Len(ns[cid].prefix) = 0
                       /\ ap + 1 > Len(key)
                    THEN <<<<nid, d>>>>
                    ELSE <<>>

(***************************************************************************)
(* REMOVE — Collapse                                                       *)
(***************************************************************************)

CollapseOp(ns, rt, nid, parent_id, d_to_nid) ==
    IF nid = NULL \/ nid \notin DOMAIN ns THEN <<ns, rt>>
    ELSE IF Cardinality(DOMAIN ns[nid].children) /= 1 THEN <<ns, rt>>
    ELSE IF ns[nid].is_vis THEN <<ns, rt>>  \* Node has VIS + 1 child = 2 occupants, not single-child
    ELSE IF nid = rt THEN <<ns, rt>>
    ELSE
        LET n == ns[nid]
            cd == CHOOSE b \in DOMAIN n.children : TRUE
            cid == n.children[cd]
        IN
        IF cid \notin DOMAIN ns THEN <<ns, rt>>
        ELSE
            LET child == ns[cid]
                combined == Len(n.prefix) + 1 + Len(child.prefix)
            IN
            IF combined > PrefixCapacity THEN <<ns, rt>>
            ELSE
                LET new_pfx == n.prefix \o <<cd>> \o child.prefix
                    new_vis == child.is_vis
                    upd == [child EXCEPT !.prefix = new_pfx]
                    ns2 == [nd \in (DOMAIN ns \ {nid}) |->
                              IF nd = cid THEN upd ELSE ns[nd]]
                    ns3 == IF parent_id = NULL THEN ns2
                           ELSE [ns2 EXCEPT
                                   ![parent_id].children[d_to_nid] = cid]
                IN <<ns3, IF nid = rt THEN cid ELSE rt>>

(***************************************************************************)
(* REMOVE — Cleanup (remove empty non-VIS nodes up the path)               *)
(***************************************************************************)

\* After removing a child, clean up: if node is now childless and non-VIS,
\* remove it from its parent too. Then try collapse on the resulting parent.
\* path = full path, idx = index of node to check, ns = current node map
RECURSIVE CleanUp(_, _, _, _)
CleanUp(ns, rt, path, idx) ==
    IF idx < 1 THEN <<ns, rt>>
    ELSE
        LET nid == path[idx][1] IN
        IF nid \notin DOMAIN ns THEN <<ns, rt>>
        ELSE
            LET n == ns[nid] IN
            IF DOMAIN n.children = {} /\ ~n.is_vis THEN
                \* Dead node — remove it
                IF nid = rt THEN
                    \* Dead root — tree becomes empty
                    <<[nd \in (DOMAIN ns \ {nid}) |-> ns[nd]], NULL>>
                ELSE IF idx = 1 THEN
                    \* Root-level entry with no parent in path
                    <<[nd \in (DOMAIN ns \ {nid}) |-> ns[nd]], NULL>>
                ELSE
                    LET pid == path[idx - 1][1]
                        pd == path[idx - 1][2]
                        ns2 == [nd \in (DOMAIN ns \ {nid}) |-> ns[nd]]
                        ns3 == [ns2 EXCEPT ![pid].children =
                                  [b \in (DOMAIN ns2[pid].children \ {pd}) |->
                                      ns2[pid].children[b]]]
                    IN CleanUp(ns3, rt, path, idx - 1)
            ELSE
                \* Node is alive — try collapse
                LET gpid == IF idx >= 2 THEN path[idx - 1][1] ELSE NULL
                    gpd == IF idx >= 2 THEN path[idx - 1][2] ELSE 0
                IN CollapseOp(ns, rt, nid, gpid, gpd)

(***************************************************************************)
(* REMOVE — Top-level Action                                               *)
(***************************************************************************)

Remove(key) ==
    /\ key \in present
    /\ root /= NULL
    /\ LET path == TraceFrom(nodes, root, key, 1) IN
       /\ path /= <<>>
       /\ LET last == path[Len(path)]
              last_nid == last[1]
              last_d == last[2]
          IN
          IF last_d = 0 THEN
              \* Key is VIS at last_nid
              LET n == nodes[last_nid] IN
              IF DOMAIN n.children = {} THEN
                  \* Leaf VIS — remove node
                  IF last_nid = root THEN
                      /\ nodes' = EmptyFcn
                      /\ root' = NULL
                      /\ present' = present \ {key}
                      /\ nxt' = nxt
                  ELSE
                      LET pe == path[Len(path) - 1]
                          pid == pe[1]
                          pd == pe[2]
                          ns2 == [nd \in (DOMAIN nodes \ {last_nid}) |-> nodes[nd]]
                          ns3 == [ns2 EXCEPT ![pid].children =
                                    [b \in (DOMAIN ns2[pid].children \ {pd}) |->
                                        ns2[pid].children[b]]]
                          cl == CleanUp(ns3, root, path, Len(path) - 1)
                      IN
                      /\ nodes' = cl[1]
                      /\ root' = cl[2]
                      /\ present' = present \ {key}
                      /\ nxt' = nxt
              ELSE
                  \* Internal VIS — clear flag, then clean up if needed
                  LET ns2 == [nodes EXCEPT ![last_nid].is_vis = FALSE] IN
                  IF Cardinality(DOMAIN n.children) = 1 THEN
                      \* Became single-child non-VIS — try collapse
                      LET gpid == IF Len(path) >= 2
                                  THEN path[Len(path) - 1][1] ELSE NULL
                          gpd == IF Len(path) >= 2
                                 THEN path[Len(path) - 1][2] ELSE 0
                          cl == CollapseOp(ns2, root, last_nid, gpid, gpd)
                      IN
                      /\ nodes' = cl[1]
                      /\ root' = cl[2]
                      /\ present' = present \ {key}
                      /\ nxt' = nxt
                  ELSE
                      /\ nodes' = ns2
                      /\ root' = root
                      /\ present' = present \ {key}
                      /\ nxt' = nxt
          ELSE
              \* Key ends at leaf child via dispatch last_d
              LET leaf_id == nodes[last_nid].children[last_d]
                  ns2 == [nd \in (DOMAIN nodes \ {leaf_id}) |-> nodes[nd]]
                  ns3 == [ns2 EXCEPT ![last_nid].children =
                            [b \in (DOMAIN ns2[last_nid].children \ {last_d}) |->
                                ns2[last_nid].children[b]]]
                  cl == CleanUp(ns3, root, path, Len(path))
              IN
              /\ nodes' = cl[1]
              /\ root' = cl[2]
              /\ present' = present \ {key}
              /\ nxt' = nxt

(***************************************************************************)
(* SPECIFICATION                                                           *)
(***************************************************************************)

Init ==
    /\ nodes = EmptyFcn
    /\ root = NULL
    /\ present = {}
    /\ nxt = 1

Next ==
    \/ \E key \in Keys : Insert(key)
    \/ \E key \in Keys : Remove(key)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* INVARIANTS                                                              *)
(***************************************************************************)

NoOrphans ==
    IF root = NULL THEN DOMAIN nodes = {}
    ELSE DOMAIN nodes = Reachable(root)

KeyPreservation ==
    \A key \in present : root /= NULL /\ KeyAt(root, key, 1)

WellFormed ==
    \A nid \in DOMAIN nodes :
        (Cardinality(DOMAIN nodes[nid].children) = 1 /\ nid /= root
         /\ ~nodes[nid].is_vis) =>
            LET cd == CHOOSE b \in DOMAIN nodes[nid].children : TRUE
                cid == nodes[nid].children[cd]
            IN
            \/ cid \notin DOMAIN nodes
            \/ Len(nodes[nid].prefix) + 1 + Len(nodes[cid].prefix) > PrefixCapacity

TypeOK ==
    /\ root \in (DOMAIN nodes) \union {NULL}
    /\ present \subseteq Keys
    /\ nxt \in Nat \ {0}
    /\ \A nid \in DOMAIN nodes :
        /\ \A i \in 1..Len(nodes[nid].prefix) :
            nodes[nid].prefix[i] \in Alphabet
        /\ nodes[nid].is_vis \in BOOLEAN
        /\ \A b \in DOMAIN nodes[nid].children :
            /\ b \in Alphabet
            /\ nodes[nid].children[b] \in DOMAIN nodes

StateConstraint == nxt <= 7

(***************************************************************************)
(* BUGGY INSERT — No ancestor collapse on unwind                           *)
(*                                                                         *)
(* Models the real implementation's iterative insert which does NOT check   *)
(* whether ancestors became collapsible after a prefix split deeper in     *)
(* the tree.                                                               *)
(***************************************************************************)

RECURSIVE BuggyDoInsert(_, _, _, _, _, _)
BuggyDoInsert(ns, rt, key, pos, nid, nx) ==
    LET n == ns[nid]
        plen == Len(n.prefix)
        key_rem == Len(key) - pos + 1
        match_max == IF plen < key_rem THEN plen ELSE key_rem
        RECURSIVE PfxMatch(_)
        PfxMatch(i) ==
            IF i > match_max THEN match_max
            ELSE IF key[pos + i - 1] /= n.prefix[i] THEN i - 1
            ELSE PfxMatch(i + 1)
        mlen == IF plen = 0 THEN 0 ELSE PfxMatch(1)
    IN
    IF mlen < plen THEN
        \* PREFIX SPLIT — identical to correct version
        LET sb == n.prefix[mlen + 1]
            osuf == IF mlen + 2 > plen THEN <<>>
                    ELSE SubSeq(n.prefix, mlen + 2, plen)
            bpfx == IF mlen = 0 THEN <<>>
                    ELSE SubSeq(n.prefix, 1, mlen)
            kpos == pos + mlen
            ns2 == [ns EXCEPT ![nid].prefix = osuf]
            nid_collapsible ==
                /\ ~n.is_vis
                /\ Cardinality(DOMAIN n.children) = 1
                /\ LET cd == CHOOSE b \in DOMAIN n.children : TRUE
                       cid == n.children[cd]
                   IN /\ cid \in DOMAIN ns
                      /\ Len(osuf) + 1 + Len(ns[cid].prefix) <= PrefixCapacity
        IN
        IF kpos > Len(key) THEN
            LET bid == nx
                bn == [prefix |-> bpfx, children |-> (sb :> nid), is_vis |-> TRUE]
                raw == <<(bid :> bn) @@ ns2, IF nid = rt THEN bid ELSE rt, nx + 1, bid>>
            IN IF nid_collapsible THEN
                 LET cd == CHOOSE b \in DOMAIN n.children : TRUE
                     cid == n.children[cd]
                     new_pfx == osuf \o <<cd>> \o ns[cid].prefix
                     ns_c == [nd \in (DOMAIN raw[1] \ {nid}) |->
                               IF nd = cid
                               THEN [raw[1][cid] EXCEPT !.prefix = new_pfx]
                               ELSE raw[1][nd]]
                     ns_d == [ns_c EXCEPT ![bid].children = (sb :> cid)]
                 IN <<ns_d, raw[2], raw[3], raw[4]>>
               ELSE raw
        ELSE
            LET kd == key[kpos]
                ch == MkChain(key, kpos + 1, nx + 1)
                bid == nx
                bc == (sb :> nid) @@ (kd :> ch[2])
                bn == [prefix |-> bpfx, children |-> bc, is_vis |-> FALSE]
                raw == <<(bid :> bn) @@ ch[1] @@ ns2,
                         IF nid = rt THEN bid ELSE rt, ch[3], bid>>
            IN IF nid_collapsible THEN
                 LET cd == CHOOSE b \in DOMAIN n.children : TRUE
                     cid == n.children[cd]
                     new_pfx == osuf \o <<cd>> \o ns[cid].prefix
                     ns_c == [nd \in (DOMAIN raw[1] \ {nid}) |->
                               IF nd = cid
                               THEN [raw[1][cid] EXCEPT !.prefix = new_pfx]
                               ELSE raw[1][nd]]
                     ns_d == [ns_c EXCEPT ![bid].children[sb] = cid]
                 IN <<ns_d, raw[2], raw[3], raw[4]>>
               ELSE raw
    ELSE
        \* FULL PREFIX MATCH
        LET ap == pos + plen IN
        IF ap > Len(key) THEN
            <<[ns EXCEPT ![nid].is_vis = TRUE], rt, nx, nid>>
        ELSE
            LET d == key[ap] IN
            IF d \in DOMAIN n.children THEN
                \* BUG: No ancestor collapse check after recursive return
                LET cid == n.children[d]
                    sub == BuggyDoInsert(ns, rt, key, ap + 1, cid, nx)
                    ns1 == IF sub[4] /= cid
                           THEN [sub[1] EXCEPT ![nid].children[d] = sub[4]]
                           ELSE sub[1]
                IN <<ns1, sub[2], sub[3], nid>>
            ELSE
                LET ch == MkChain(key, ap + 1, nx)
                    ns2 == [ns EXCEPT ![nid].children = n.children @@ (d :> ch[2])]
                IN <<ch[1] @@ ns2, rt, ch[3], nid>>

BuggyInsert(key) ==
    /\ key \notin present
    /\ IF root = NULL THEN
           LET ch == MkChain(key, 1, nxt)
           IN /\ nodes' = ch[1]
              /\ root' = ch[2]
              /\ present' = present \union {key}
              /\ nxt' = ch[3]
       ELSE
           LET r == BuggyDoInsert(nodes, root, key, 1, root, nxt)
           IN /\ nodes' = r[1]
              /\ root' = r[2]
              /\ present' = present \union {key}
              /\ nxt' = r[3]

BuggyNext ==
    \/ \E key \in Keys : BuggyInsert(key)
    \/ \E key \in Keys : Remove(key)

BuggySpec == Init /\ [][BuggyNext]_vars

================================================================================
