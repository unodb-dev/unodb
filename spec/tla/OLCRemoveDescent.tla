--------------------------- MODULE OLCRemoveDescent -----------------------
\* H10: Remove descent slot-content safety.
\*
\* Parameterized by VIS (can_eliminate_leaf):
\*   VIS=FALSE: slots hold PTR or NULL only. No bitmask.
\*   VIS=TRUE:  slots hold PTR, VIS_ZERO, or NULL. Bitmask distinguishes.
\*
\* Models one Reader (a remove descent) and one Writer that changes the
\* reader's slot in place under the node's write lock.
\*
\* Reader protocol.  The same steps model both remove descents that read a
\* child's value bit: remove_or_choose_subtree, and try_remove_key_view's
\* chain-I4 branch (olc_art.hpp, "Chain I4 detection").
\*   1. find_child: read key_index; if mapped, return the slot address
\*   2. load: read the slot's content (one atomic load)
\*   3. validate: check() the node's version
\*   4. dispatch: read the value bit.  With VIS and the bit set, take the
\*      packed-value path; otherwise go on to treat the content as a pointer
\*   5. check_child: check() the node again (ReCheck), because the bit was
\*      read after step 3's check.  remove_or_choose_subtree has always done
\*      this; the chain-I4 branch gained it in "Check an OLC chain I4 again
\*      after reading its child's value bit"
\*   6. deref: read-lock the child as a node
\*   On the packed-value path, both descents upgrade the node's read lock to
\*   a write lock before removing (vis_upgrade), which fails if the version
\*   moved, so a bit read after step 3 cannot make them remove the wrong
\*   slot.
\*
\* Writer protocol, one of WriterKinds:
\*   "clear":  a remove of the reader's key.  Sets slot=NULL, bitmask=FALSE
\*             and key_index=FALSE.
\*   "insert": an insert into the same node that shifts the reader's slot.
\*             basic_inode_4/16's add (art_internal_impl.hpp, both
\*             add_to_nonfull overloads, at the
\*             "Shift existing value bits to match shifted children" lines)
\*             first moves the children, then calls bitmask_base::insert_at(),
\*             which shifts the bits and leaves the new position's bit
\*             clear, and sets it with set_value_bit() only if the new child
\*             is a packed value.  So the slot's content and its bit change
\*             in separate stores, and in between the bit can be clear for a
\*             slot that holds, or held, a packed value.  key_index stays
\*             TRUE: the reader's key byte is still mapped.
\*
\* Safety: the reader never dereferences unless r_child = "PTR".
\*
\* OLCRemoveDescent_BugDemo.cfg sets ReCheck = FALSE, the chain-I4 branch
\* before that commit, and violates NoDereferenceNonPointer.

EXTENDS Integers

CONSTANTS
    MaxVersion,
    VIS,         \* TRUE or FALSE — models can_eliminate_leaf
    ReCheck,     \* TRUE: check() again after reading the value bit (step 5)
    WriterKinds  \* subset of {"clear", "insert"}

VARIABLES
    \* Shared state (parent inode slot)
    key_index,  \* TRUE=find_child would succeed, FALSE=key absent
    slot,       \* "PTR" | "VIS_ZERO" | "NULL" (VIS_ZERO only when VIS=TRUE)
    bitmask,    \* TRUE=slot is packed value (only meaningful when VIS=TRUE)
    version,    \* even=unlocked, odd=write-locked

    \* Reader state
    rpc,
    r_ver,      \* captured version
    r_found,    \* find_child result
    r_child,    \* loaded slot content
    r_bitmask,  \* loaded bitmask

    \* Writer state
    wpc,
    w_kind,     \* "clear" or "insert"
    w_new,      \* insert: the child it leaves in the reader's slot
    w_mid_bit   \* insert: the bit insert_at() shifts into the reader's slot

vars == <<key_index, slot, bitmask, version, rpc, r_ver, r_found, r_child,
          r_bitmask, wpc, w_kind, w_new, w_mid_bit>>

wvars == <<wpc, w_kind, w_new, w_mid_bit>>

-----------------------------------------------------------------------------
\* Initial states depend on VIS
SlotKinds == IF VIS THEN {"PTR", "VIS_ZERO"} ELSE {"PTR"}

Init ==
    /\ key_index = TRUE
    /\ slot \in SlotKinds
    /\ bitmask = (VIS /\ slot = "VIS_ZERO")
    /\ version = 0
    /\ rpc = "idle"
    /\ r_ver = 0
    /\ r_found = FALSE
    /\ r_child = "NULL"
    /\ r_bitmask = FALSE
    /\ wpc = "idle"
    /\ w_kind \in WriterKinds
    /\ w_new \in SlotKinds
    \* Without VIS there is no bitmask: the bit stays FALSE.
    /\ w_mid_bit \in IF VIS THEN {TRUE, FALSE} ELSE {FALSE}

-----------------------------------------------------------------------------
\* READER

RFind ==
    /\ rpc = "idle"
    /\ version % 2 = 0
    /\ r_ver' = version
    /\ r_found' = key_index
    /\ rpc' = "check_find"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_child, r_bitmask>>
    /\ UNCHANGED wvars

RCheckFind ==
    /\ rpc = "check_find"
    /\ IF r_found
       THEN rpc' = "load"
       ELSE rpc' = "done"       \* key not found
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child,
                   r_bitmask>>
    /\ UNCHANGED wvars

RLoad ==
    /\ rpc = "load"
    /\ r_child' = slot
    /\ rpc' = "validate"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_bitmask>>
    /\ UNCHANGED wvars

RValidate ==
    /\ rpc = "validate"
    /\ IF version = r_ver /\ version % 2 = 0
       THEN rpc' = "dispatch"
       ELSE rpc' = "restart"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child,
                   r_bitmask>>
    /\ UNCHANGED wvars

RDispatch ==
    /\ rpc = "dispatch"
    /\ IF VIS
       THEN \* Read bitmask, branch on it
            /\ r_bitmask' = bitmask
            /\ IF bitmask
               THEN rpc' = "vis_remove"
               ELSE rpc' = IF ReCheck THEN "check_child" ELSE "deref"
       ELSE \* No VIS — no unvalidated read, safe to dereference
            /\ r_bitmask' = FALSE
            /\ rpc' = "deref"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child>>
    /\ UNCHANGED wvars

\* Guard: re-validate version before dereferencing child as pointer.
\* Models: if (!node_critical_section.check()) return {};
\* If version changed since first check, the bitmask may be stale.
RCheckChild ==
    /\ rpc = "check_child"
    /\ IF version = r_ver /\ version % 2 = 0
       THEN rpc' = "deref"
       ELSE rpc' = "restart"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child,
                   r_bitmask>>
    /\ UNCHANGED wvars

RDeref ==
    /\ rpc = "deref"
    /\ rpc' = "done"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child,
                   r_bitmask>>
    /\ UNCHANGED wvars

\* write_guard{std::move(node_critical_section)}: the upgrade succeeds only
\* at the version step 1 read.
RVisRemove ==
    /\ rpc = "vis_remove"
    /\ rpc' = IF version = r_ver /\ version % 2 = 0
              THEN "vis_removed" ELSE "restart"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child,
                   r_bitmask>>
    /\ UNCHANGED wvars

RVisDone ==
    /\ rpc = "vis_removed"
    /\ rpc' = "done"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child,
                   r_bitmask>>
    /\ UNCHANGED wvars

RRestart ==
    /\ rpc = "restart"
    /\ rpc' = "idle"
    /\ UNCHANGED <<key_index, slot, bitmask, version, r_ver, r_found, r_child,
                   r_bitmask>>
    /\ UNCHANGED wvars

RDone ==
    /\ rpc = "done"
    /\ UNCHANGED vars

-----------------------------------------------------------------------------
\* WRITER

WLock ==
    /\ wpc = "idle"
    /\ version % 2 = 0
    /\ version <= MaxVersion
    /\ version' = version + 1
    /\ wpc' = IF w_kind = "clear" THEN "clear" ELSE "ins_children"
    /\ UNCHANGED <<key_index, slot, bitmask, rpc, r_ver, r_found, r_child,
                   r_bitmask, w_kind, w_new, w_mid_bit>>

WClear ==
    /\ wpc = "clear"
    /\ slot' = "NULL"
    /\ bitmask' = FALSE
    /\ key_index' = FALSE
    /\ wpc' = "unlock"
    /\ UNCHANGED <<version, rpc, r_ver, r_found, r_child, r_bitmask,
                   w_kind, w_new, w_mid_bit>>

\* Insert, store 1: the children move, leaving w_new in the reader's slot.
WInsChildren ==
    /\ wpc = "ins_children"
    /\ slot' = w_new
    /\ wpc' = "ins_shift_bits"
    /\ UNCHANGED <<key_index, bitmask, version, rpc, r_ver, r_found, r_child,
                   r_bitmask, w_kind, w_new, w_mid_bit>>

\* Insert, store 2: bitmask_base::insert_at() shifts the bits.
WInsShiftBits ==
    /\ wpc = "ins_shift_bits"
    /\ bitmask' = w_mid_bit
    /\ wpc' = "ins_set_bit"
    /\ UNCHANGED <<key_index, slot, version, rpc, r_ver, r_found, r_child,
                   r_bitmask, w_kind, w_new, w_mid_bit>>

\* Insert, store 3: the bit now matches the slot's content.
WInsSetBit ==
    /\ wpc = "ins_set_bit"
    /\ bitmask' = (VIS /\ w_new = "VIS_ZERO")
    /\ wpc' = "unlock"
    /\ UNCHANGED <<key_index, slot, version, rpc, r_ver, r_found, r_child,
                   r_bitmask, w_kind, w_new, w_mid_bit>>

WUnlock ==
    /\ wpc = "unlock"
    /\ version' = version + 1
    /\ wpc' = "idle"
    /\ UNCHANGED <<key_index, slot, bitmask, rpc, r_ver, r_found, r_child,
                   r_bitmask, w_kind, w_new, w_mid_bit>>

-----------------------------------------------------------------------------
Step ==
    \/ RFind \/ RCheckFind \/ RLoad \/ RValidate \/ RDispatch
    \/ RCheckChild \/ RDeref \/ RVisRemove \/ RVisDone \/ RRestart \/ RDone
    \/ WLock \/ WClear \/ WInsChildren \/ WInsShiftBits \/ WInsSetBit
    \/ WUnlock

Spec == Init /\ [][Step]_vars

StateConstraint == version <= MaxVersion

-----------------------------------------------------------------------------
\* SAFETY

\* Reader only dereferences when it loaded a real pointer.
NoDereferenceNonPointer ==
    rpc = "deref" => r_child = "PTR"

\* The packed-value path removes only a slot that held a packed value.
VisPathCorrectness ==
    rpc = "vis_removed" => r_child = "VIS_ZERO"

TypeOK ==
    /\ key_index \in {TRUE, FALSE}
    /\ slot \in {"PTR", "VIS_ZERO", "NULL"}
    /\ bitmask \in {TRUE, FALSE}
    /\ version \in 0..MaxVersion+2
    /\ rpc \in {"idle", "check_find", "load", "validate", "dispatch",
                "check_child", "deref", "vis_remove", "vis_removed", "done",
                "restart"}
    /\ r_ver \in 0..MaxVersion+2
    /\ r_found \in {TRUE, FALSE}
    /\ r_child \in {"PTR", "VIS_ZERO", "NULL"}
    /\ r_bitmask \in {TRUE, FALSE}
    /\ wpc \in {"idle", "clear", "ins_children", "ins_shift_bits",
                "ins_set_bit", "unlock"}
    /\ w_kind \in {"clear", "insert"}
    /\ w_new \in {"PTR", "VIS_ZERO"}
    /\ w_mid_bit \in {TRUE, FALSE}

=============================================================================
