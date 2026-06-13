#!/bin/bash
# Run all TLA+ model checking stages sequentially.
# Resource limits: 4GB heap, 4 workers, 5min timeout per stage.
set -e
cd "$(dirname "$0")"

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
FAIL=0

run_stage() {
  local name="$1" tla="$2" cfg="$3"
  printf "  %-50s " "$name"
  rm -rf /tmp/tlc-run
  local output
  output=$(timeout 300 java -Xmx4g -XX:+UseParallelGC \
    -cp ~/tools/tla/tla2tools.jar tlc2.TLC \
    "$tla" -config "$cfg" -workers 4 \
    -metadir /tmp/tlc-run -noGenerateSpecTE 2>&1) || true
  rm -rf /tmp/tlc-run
  if echo "$output" | grep -q "No error has been found"; then
    local states
    states=$(echo "$output" | grep "distinct states found" | grep -oP "\\d+ distinct" | tail -1)
    echo -e "${GREEN}PASS${NC} ($states)"
  else
    echo -e "${RED}FAIL${NC}"
    echo "$output" | grep -A5 "Error:" | head -10
    FAIL=1
  fi
}

echo "=== TLA+ OLC ART Verification Suite ==="
echo
run_stage "Stage 1: Inode256VIS (sequential scan)"      Inode256VIS.tla      Inode256VIS.cfg
run_stage "Stage 2: OLCSlot (single-slot protocol)"     OLCSlot.tla          OLCSlot.cfg
run_stage "Stage 3: OLCInsert (two writers race)"       OLCInsert.tla        OLCInsert.cfg
run_stage "Stage 4: OLCChainMembership (revalidation)"  OLCChainMembership.tla OLCChainMembership.cfg
run_stage "Stage 5: OLCChainCut (Cases A/B/C)"          OLCChainCut.tla      OLCChainCut.cfg
run_stage "Stage 6: OLCIterRemove (obsolescence)"       OLCIterRemove.tla    OLCIterRemove.cfg
run_stage "Stage 7: OLCInsertChainVIS (transient VIS)"  OLCInsertChainVIS.tla OLCInsertChainVIS.cfg
run_stage "Stage 8: OLCDoubleCut (two removes)"         OLCDoubleCut.tla     OLCDoubleCut.cfg
run_stage "Stage 9: OLCChainMultiLevel (multi-level chain)"  OLCChainMultiLevel.tla OLCChainMultiLevel.cfg
run_stage "Stage 10: OLCChainCutFull (root/shrink/cascade)" OLCChainCutFull.tla OLCChainCutFull.cfg
run_stage "Stage 11: OLCKeyViewChain (VIS + shared prefix)"  OLCKeyViewChain.tla OLCKeyViewChain.cfg
run_stage "Stage 12: ARTTreeMaintenance (sequential tree)"   ARTTreeMaintenance.tla ARTTreeMaintenance.cfg
echo
if [ $FAIL -eq 0 ]; then
  echo -e "${GREEN}All stages passed!${NC}"
else
  echo -e "${RED}Some stages failed.${NC}"
  exit 1
fi
