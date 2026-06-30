/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "UnoDB", "index.html", [
    [ "unodb", "index.html#autotoc_md130", [
      [ "Introduction", "index.html#autotoc_md131", null ],
      [ "Requirements", "index.html#autotoc_md132", [
        [ "Build dependencies", "index.html#autotoc_md133", null ],
        [ "Optional vendored dependencies, bundled as Git submodules", "index.html#autotoc_md134", null ]
      ] ],
      [ "Building", "index.html#autotoc_md135", null ],
      [ "Platform-Specific Notes", "index.html#autotoc_md136", [
        [ "Ubuntu 22.04", "index.html#autotoc_md137", null ],
        [ "Amazon Linux 2023", "index.html#autotoc_md138", null ],
        [ "Amazon Linux 2", "index.html#autotoc_md139", null ]
      ] ],
      [ "Usage", "index.html#autotoc_md140", null ],
      [ "Technical Details", "index.html#autotoc_md141", [
        [ "Adaptive Radix Tree", "index.html#autotoc_md142", null ],
        [ "Sequential Lock", "index.html#autotoc_md143", null ],
        [ "Quiescent State-Based Reclamation (QSBR)", "index.html#autotoc_md144", null ]
      ] ],
      [ "Related Projects", "index.html#autotoc_md145", null ],
      [ "Contributing", "index.html#autotoc_md146", null ],
      [ "Literature", "index.html#autotoc_md147", null ]
    ] ],
    [ "Agent Bootstrap — unodb", "md_AGENTS.html", [
      [ "Scope", "md_AGENTS.html#autotoc_md1", null ],
      [ "Repo Orientation", "md_AGENTS.html#autotoc_md2", null ],
      [ "Graph Orientation", "md_AGENTS.html#autotoc_md3", null ],
      [ "Active Branch Stack", "md_AGENTS.html#autotoc_md4", null ],
      [ "CI Process", "md_AGENTS.html#autotoc_md5", [
        [ "Before pushing (MANDATORY — no exceptions)", "md_AGENTS.html#autotoc_md6", null ],
        [ "Fork CI (MSVC + coverage) — run BEFORE pushing to upstream", "md_AGENTS.html#autotoc_md7", null ],
        [ "Upstream CI — poll after pushing", "md_AGENTS.html#autotoc_md8", null ],
        [ "Formatting", "md_AGENTS.html#autotoc_md9", null ]
      ] ],
      [ "Git Rules", "md_AGENTS.html#autotoc_md10", null ],
      [ "Subagent Workspace Rule", "md_AGENTS.html#autotoc_md11", null ],
      [ "Push Workflow (complete sequence)", "md_AGENTS.html#autotoc_md12", null ],
      [ "The Old Monolith (937685c4)", "md_AGENTS.html#autotoc_md13", null ],
      [ "Merge Plan", "md_AGENTS.html#autotoc_md14", null ],
      [ "Known Bugs", "md_AGENTS.html#autotoc_md15", null ],
      [ "Local Tooling Gap (discovered 2026-06-10)", "md_AGENTS.html#autotoc_md16", null ],
      [ "Code Quality Rules (from CONTRIBUTING.md)", "md_AGENTS.html#autotoc_md17", null ],
      [ "Resume State (2026-06-27T14:03 UTC)", "md_AGENTS.html#autotoc_md18", [
        [ "Current Work", "md_AGENTS.html#autotoc_md19", null ],
        [ "Branch State", "md_AGENTS.html#autotoc_md20", null ],
        [ "On Restart", "md_AGENTS.html#autotoc_md21", null ]
      ] ],
      [ "Emacs", "md_AGENTS.html#autotoc_md22", null ],
      [ "Build Rules", "md_AGENTS.html#autotoc_md23", [
        [ "TSan and OLC Fields", "md_AGENTS.html#autotoc_md24", null ]
      ] ]
    ] ],
    [ "CONTRIBUTING", "md_CONTRIBUTING.html", [
      [ "Contributing to UnoDB", "md_CONTRIBUTING.html#autotoc_md27", [
        [ "Optional development dependencies", "md_CONTRIBUTING.html#autotoc_md28", null ],
        [ "General workflow", "md_CONTRIBUTING.html#autotoc_md29", null ],
        [ "Development CMake options", "md_CONTRIBUTING.html#autotoc_md30", null ],
        [ "Code organization", "md_CONTRIBUTING.html#autotoc_md31", null ],
        [ "Code style guide", "md_CONTRIBUTING.html#autotoc_md32", null ],
        [ "Documentation style guide", "md_CONTRIBUTING.html#autotoc_md33", null ],
        [ "Linting and static analysis", "md_CONTRIBUTING.html#autotoc_md34", null ],
        [ "Testing", "md_CONTRIBUTING.html#autotoc_md35", null ],
        [ "Fuzzing", "md_CONTRIBUTING.html#autotoc_md36", null ],
        [ "Commit messages", "md_CONTRIBUTING.html#autotoc_md37", null ],
        [ "Pull Requests", "md_CONTRIBUTING.html#autotoc_md38", null ],
        [ "Benchmarking", "md_CONTRIBUTING.html#autotoc_md39", null ],
        [ "License", "md_CONTRIBUTING.html#autotoc_md40", null ]
      ] ]
    ] ],
    [ "CAS / Upsert API Design — <tt>insert_or_resolve</tt> (Issue #847)", "md_design_2cas-insert-or-resolve-847.html", [
      [ "1. Prior Art: libcuckoo", "md_design_2cas-insert-or-resolve-847.html#autotoc_md42", [
        [ "API Surface", "md_design_2cas-insert-or-resolve-847.html#autotoc_md43", null ],
        [ "<tt>uprase_fn</tt> Protocol", "md_design_2cas-insert-or-resolve-847.html#autotoc_md44", null ],
        [ "Thread Safety Guarantees", "md_design_2cas-insert-or-resolve-847.html#autotoc_md45", null ],
        [ "Error Handling", "md_design_2cas-insert-or-resolve-847.html#autotoc_md46", null ],
        [ "Key Design Insight", "md_design_2cas-insert-or-resolve-847.html#autotoc_md47", null ]
      ] ],
      [ "2. Issue #847 Spec Summary", "md_design_2cas-insert-or-resolve-847.html#autotoc_md48", null ],
      [ "3. Current Insert Implementation Analysis", "md_design_2cas-insert-or-resolve-847.html#autotoc_md49", [
        [ "3.1 <tt>db</tt> (non-concurrent, <tt>art.hpp</tt>)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md50", null ],
        [ "3.2 <tt>mutex_db</tt> (<tt>mutex_art.hpp</tt>)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md51", null ],
        [ "3.3 <tt>olc_db</tt> (<tt>olc_art.hpp</tt>)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md52", null ],
        [ "3.4 Key Types: <tt>get_value<Value></tt> and the missing <tt>set_value<Value></tt>", "md_design_2cas-insert-or-resolve-847.html#autotoc_md53", null ],
        [ "3.5 <tt>art_policy</tt> — VIS pack/unpack", "md_design_2cas-insert-or-resolve-847.html#autotoc_md54", null ]
      ] ],
      [ "4. Proposed Design", "md_design_2cas-insert-or-resolve-847.html#autotoc_md55", [
        [ "4.1 Public API", "md_design_2cas-insert-or-resolve-847.html#autotoc_md56", null ],
        [ "4.2 <tt>set_value<Value></tt> on Leaf", "md_design_2cas-insert-or-resolve-847.html#autotoc_md57", null ],
        [ "4.3 Implementation: <tt>db</tt> (non-concurrent)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md58", null ],
        [ "4.4 Implementation: <tt>mutex_db</tt>", "md_design_2cas-insert-or-resolve-847.html#autotoc_md59", null ],
        [ "4.5 Implementation: <tt>olc_db</tt> — The Hard Case", "md_design_2cas-insert-or-resolve-847.html#autotoc_md60", [
          [ "4.5.1 <tt>keep</tt> Action", "md_design_2cas-insert-or-resolve-847.html#autotoc_md61", null ],
          [ "4.5.2 <tt>update</tt> Action — Keyed-Leaf Path", "md_design_2cas-insert-or-resolve-847.html#autotoc_md62", null ],
          [ "4.5.3 <tt>update</tt> Action — VIS Path", "md_design_2cas-insert-or-resolve-847.html#autotoc_md63", null ],
          [ "4.5.4 <tt>erase</tt> Action — Deferred to Phase 2", "md_design_2cas-insert-or-resolve-847.html#autotoc_md64", null ]
        ] ],
        [ "4.6 <tt>erase_fn</tt> Companion API", "md_design_2cas-insert-or-resolve-847.html#autotoc_md65", null ],
        [ "4.7 Internal Plumbing", "md_design_2cas-insert-or-resolve-847.html#autotoc_md66", [
          [ "New Internal Methods", "md_design_2cas-insert-or-resolve-847.html#autotoc_md67", null ],
          [ "Template Approach: Avoid Code Duplication", "md_design_2cas-insert-or-resolve-847.html#autotoc_md68", null ]
        ] ]
      ] ],
      [ "5. Phasing", "md_design_2cas-insert-or-resolve-847.html#autotoc_md69", [
        [ "Phase 1: <tt>keep</tt> + <tt>update</tt> only", "md_design_2cas-insert-or-resolve-847.html#autotoc_md70", null ],
        [ "Phase 2: <tt>erase</tt> action + <tt>erase_fn</tt>", "md_design_2cas-insert-or-resolve-847.html#autotoc_md71", null ]
      ] ],
      [ "6. Lock Analysis Summary", "md_design_2cas-insert-or-resolve-847.html#autotoc_md72", null ],
      [ "7. Test Strategy", "md_design_2cas-insert-or-resolve-847.html#autotoc_md73", [
        [ "Unit Tests", "md_design_2cas-insert-or-resolve-847.html#autotoc_md74", null ],
        [ "Concurrency Tests (olc_db)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md75", null ],
        [ "Type Coverage", "md_design_2cas-insert-or-resolve-847.html#autotoc_md76", null ],
        [ "Edge Cases", "md_design_2cas-insert-or-resolve-847.html#autotoc_md77", null ]
      ] ],
      [ "8. Open Questions", "md_design_2cas-insert-or-resolve-847.html#autotoc_md78", null ],
      [ "9. Round 1 Findings — Resolutions", "md_design_2cas-insert-or-resolve-847.html#autotoc_md79", [
        [ "9.1 CRITICAL: Lambda Re-execution on OLC Restart (Correctness A.1.1, Concurrency §1.1)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md80", null ],
        [ "9.2 CRITICAL: <tt>value_view</tt> Update Size Constraint (Correctness A.1.4)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md81", null ],
        [ "9.3 HIGH: Test Plan Gaps — Concurrent CAS Scenarios (Correctness A.1.1–A.1.3)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md82", null ],
        [ "9.4 HIGH: Branch Prediction Hint for CAS Path (Performance §2.1)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md83", null ],
        [ "9.5 MEDIUM: API Naming — <tt>insert_or_resolve</tt> vs <tt>upsert</tt> (API §1)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md84", null ],
        [ "9.6 MEDIUM: Lambda Signature — Key Availability (API §2)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md85", null ],
        [ "9.7 Resolved Open Questions (§8 Updates)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md86", null ]
      ] ],
      [ "10. Round 2 Findings — Resolutions", "md_design_2cas-insert-or-resolve-847.html#autotoc_md87", [
        [ "10.1 MUST-FIX: Double-Apply Bug — Parent RCS After Committed Write (§4.5.2, §4.5.3)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md88", null ],
        [ "10.2 SHOULD-FIX: Strengthen Idempotency Contract (§9.1 addendum)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md89", null ],
        [ "10.3 SHOULD-FIX: Contention Mitigation — spin_wait at Upgrade Failure", "md_design_2cas-insert-or-resolve-847.html#autotoc_md90", null ],
        [ "10.4 SHOULD-FIX: set_value Defense-in-Depth", "md_design_2cas-insert-or-resolve-847.html#autotoc_md91", null ],
        [ "10.5 SHOULD-FIX: Lambda Constraint — Return Type Check", "md_design_2cas-insert-or-resolve-847.html#autotoc_md92", null ],
        [ "10.6 Test Plan Corrections", "md_design_2cas-insert-or-resolve-847.html#autotoc_md93", null ]
      ] ],
      [ "11. Round 3 Findings — Resolutions", "md_design_2cas-insert-or-resolve-847.html#autotoc_md94", [
        [ "11.1 MUST-FIX: Explicit Version Validation in Erase Protocol (Concurrency #8)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md95", null ],
        [ "11.2 MUST-FIX: TLA+ Model Extended for Key-Absent Case (Correctness #1)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md96", null ],
        [ "11.3 MUST-FIX: Test 23g — Erase After Concurrent Remove (Correctness #7)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md97", null ],
        [ "11.4 MUST-FIX: Benchmark B2/B3 Tail Latency (Performance #5)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md98", null ],
        [ "11.5 MUST-FIX: <tt>try_upsert</tt> Private Method Signature (API #1)", "md_design_2cas-insert-or-resolve-847.html#autotoc_md99", null ],
        [ "11.6 Naming: Rename to <tt>upsert</tt> / <tt>try_upsert</tt>", "md_design_2cas-insert-or-resolve-847.html#autotoc_md100", null ],
        [ "11.7 NEEDS-SME Resolutions", "md_design_2cas-insert-or-resolve-847.html#autotoc_md101", null ]
      ] ]
    ] ],
    [ "Upsert (#847) — Test & Microbenchmark Design", "md_design_2upsert-test-benchmark-design.html", [
      [ "Test Suite", "md_design_2upsert-test-benchmark-design.html#autotoc_md103", [
        [ "Unit Tests (all db types: <tt>db</tt>, <tt>mutex_db</tt>, <tt>olc_db</tt>)", "md_design_2upsert-test-benchmark-design.html#autotoc_md104", null ],
        [ "Type Coverage", "md_design_2upsert-test-benchmark-design.html#autotoc_md105", null ],
        [ "Concurrency Tests (olc_db only)", "md_design_2upsert-test-benchmark-design.html#autotoc_md106", null ],
        [ "Erase-Specific Tests", "md_design_2upsert-test-benchmark-design.html#autotoc_md107", null ],
        [ "OOM Tests (GCC debug builds)", "md_design_2upsert-test-benchmark-design.html#autotoc_md108", null ]
      ] ],
      [ "Microbenchmark Design", "md_design_2upsert-test-benchmark-design.html#autotoc_md110", [
        [ "Goals", "md_design_2upsert-test-benchmark-design.html#autotoc_md111", null ],
        [ "Benchmark Scenarios", "md_design_2upsert-test-benchmark-design.html#autotoc_md112", [
          [ "B1: Upsert Update — Disjoint Keys (Baseline)", "md_design_2upsert-test-benchmark-design.html#autotoc_md113", null ],
          [ "B2: Upsert Update — Shared Hotkeys (Contention)", "md_design_2upsert-test-benchmark-design.html#autotoc_md114", null ],
          [ "B3: Upsert Erase — Shared Hotkeys (Erase Retry)", "md_design_2upsert-test-benchmark-design.html#autotoc_md115", null ],
          [ "B4: Upsert Insert Path — Disjoint (Insert Overhead)", "md_design_2upsert-test-benchmark-design.html#autotoc_md116", null ],
          [ "B5: Upsert Mixed — Random Operations", "md_design_2upsert-test-benchmark-design.html#autotoc_md117", null ],
          [ "B6: Backoff Strategy Comparison (ref #635)", "md_design_2upsert-test-benchmark-design.html#autotoc_md118", null ]
        ] ]
      ] ],
      [ "Implementation Notes", "md_design_2upsert-test-benchmark-design.html#autotoc_md120", null ]
    ] ],
    [ "Topics", "topics.html", "topics" ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", null ],
        [ "Typedefs", "functions_type.html", null ],
        [ "Related Symbols", "functions_rela.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Functions", "globals_func.html", null ],
        [ "Variables", "globals_vars.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"annotated.html",
"classunodb_1_1db.html#ae96ae9741ef57bdec0060d3b43f2fd36",
"classunodb_1_1detail_1_1basic__inode__256.html#a69ee45c77017182499a59843ebe0ada0",
"classunodb_1_1detail_1_1basic__inode__impl.html#a697a7cbb45d3f7c687712e19e8dcefd7",
"classunodb_1_1detail_1_1key__buffer.html#a1afd3af1a568531767801a0a18707abf",
"classunodb_1_1key__encoder.html#a255c2ad351167e9f3b9d4aa8c0b338d6",
"classunodb_1_1optimistic__lock_1_1atomic__version__type.html#a58e25f698739d17ed135f6940e31674d",
"classunodb_1_1qsbr__per__thread.html#aa27231a20f8d382edb8f6257b208c15c",
"group__internal.html#ga143793f205c69f573831b99b3dee1b9c",
"namespaceanonymous__namespace_02test__art__iter_8cpp_03.html",
"qsbr_8hpp.html",
"structunodb_1_1detail_1_1value__bitmask__field_3_01Enabled_00_01std_1_1array_3_01T_00_01N_01_4_00_01CritSec_01_4.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';