#!/bin/bash
# Run all TLA+ iterator model specs and verify expected results.
# Each spec has: correct config (must PASS), bug config (must FAIL), canary config (must FAIL).
cd "$(dirname "$0")"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS=0; FAIL=0; ERRORS=0

run_check() {
  local desc="$1" tla="$2" cfg="$3" expect="$4"
  printf "  %-60s " "$desc"
  rm -rf /tmp/tlc-iter-val
  local output
  output=$(timeout 60 java -Xmx4g -XX:+UseParallelGC \
    -cp ~/tools/tla/tla2tools.jar tlc2.TLC \
    "$tla" -config "$cfg" -workers 4 \
    -metadir /tmp/tlc-iter-val -noGenerateSpecTE 2>&1) || true
  rm -rf /tmp/tlc-iter-val
  local found_error=false
  if echo "$output" | grep -q "is violated\|Error:.*Invariant\|Error:.*Property"; then
    found_error=true
  fi

  if [ "$expect" = "pass" ]; then
    if [ "$found_error" = "false" ] && echo "$output" | grep -q "No error has been found"; then
      local states
      states=$(echo "$output" | grep "distinct states found" | grep -oP "\d+ distinct" | tail -1)
      echo -e "${GREEN}PASS${NC} ($states)"
      PASS=$((PASS + 1))
    else
      echo -e "${RED}UNEXPECTED FAIL${NC}"
      echo "$output" | grep -E "Error:|violated" | head -3
      ERRORS=$((ERRORS + 1))
    fi
  elif [ "$expect" = "fail" ]; then
    if [ "$found_error" = "true" ]; then
      local violated
      violated=$(echo "$output" | grep "is violated" | sed 's/.*Invariant //' | sed 's/ is.*//')
      echo -e "${GREEN}CAUGHT${NC} ($violated)"
      PASS=$((PASS + 1))
    else
      echo -e "${RED}MISSED — should have failed${NC}"
      ERRORS=$((ERRORS + 1))
    fi
  fi
}

echo "=== TLA+ Iterator Model Validation Suite ==="
echo

echo -e "${YELLOW}Phase 0: Base Forward Scan (OLCIterator)${NC}"
run_check "Sequential (db) — correct"          OLCIterator.tla OLCIterator_Seq.cfg      pass
run_check "OLC (olc_db) — correct"             OLCIterator.tla OLCIterator_OLC.cfg      pass

echo
echo -e "${YELLOW}Phase 0: Bug Injection (OLCIterator_Bug850)${NC}"
run_check "skip_backtrack — catches Completeness"    OLCIterator_Bug850.tla OLCIterator_Bug850_Backtrack.cfg  fail
run_check "restart_from_beginning — catches NoRepeat" OLCIterator_Bug850.tla OLCIterator_Bug850_NoSeek.cfg    fail

echo
echo -e "${YELLOW}Phase 1: Structural Remove (OLCIteratorRemove)${NC}"
run_check "Correct code — all invariants"       OLCIteratorRemove.tla OLCIteratorRemove_OLC.cfg       pass
run_check "Canary — L2AlwaysVisited fails"      OLCIteratorRemove.tla OLCIteratorRemove_Canary.cfg    fail
run_check "skip_check + havoc — OrderPreservation" OLCIteratorRemove.tla OLCIteratorRemove_SkipCheck.cfg fail

echo
echo -e "${YELLOW}Phase 3: VIS TOCTOU (OLCIteratorVIS)${NC}"
run_check "Correct code — all invariants"       OLCIteratorVIS.tla OLCIteratorVIS.cfg          pass
run_check "Canary — SlotNeverMutated fails"     OLCIteratorVIS.tla OLCIteratorVIS_Canary.cfg   fail
run_check "skip_check2 — NoGarbage violated"    OLCIteratorVIS.tla OLCIteratorVIS_Bug.cfg      fail

echo
echo -e "${YELLOW}Phase 5: Root Pointer (OLCIteratorRoot)${NC}"
run_check "Correct code — all invariants"       OLCIteratorRoot.tla OLCIteratorRoot.cfg         pass
run_check "Canary — RootNeverNull fails"        OLCIteratorRoot.tla OLCIteratorRoot_Canary.cfg  fail
run_check "skip_rp_check — NoActOnInvalid"      OLCIteratorRoot.tla OLCIteratorRoot_Bug.cfg     fail

echo
echo "────────────────────────────────────────────────────────"
echo -e "Results: ${GREEN}$PASS passed${NC}, ${RED}$ERRORS errors${NC}"
if [ $ERRORS -gt 0 ]; then
  echo -e "${RED}VALIDATION FAILED${NC}"
  exit 1
else
  echo -e "${GREEN}All checks passed!${NC}"
fi
