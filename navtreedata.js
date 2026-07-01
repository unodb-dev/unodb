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
    [ "unodb", "index.html#autotoc_md51", [
      [ "Introduction", "index.html#autotoc_md52", null ],
      [ "Requirements", "index.html#autotoc_md53", [
        [ "Build dependencies", "index.html#autotoc_md54", null ],
        [ "Optional vendored dependencies, bundled as Git submodules", "index.html#autotoc_md55", null ]
      ] ],
      [ "Building", "index.html#autotoc_md56", null ],
      [ "Platform-Specific Notes", "index.html#autotoc_md57", [
        [ "Ubuntu 22.04", "index.html#autotoc_md58", null ],
        [ "Amazon Linux 2023", "index.html#autotoc_md59", null ],
        [ "Amazon Linux 2", "index.html#autotoc_md60", null ]
      ] ],
      [ "Usage", "index.html#autotoc_md61", null ],
      [ "Technical Details", "index.html#autotoc_md62", [
        [ "Adaptive Radix Tree", "index.html#autotoc_md63", null ],
        [ "Sequential Lock", "index.html#autotoc_md64", null ],
        [ "Quiescent State-Based Reclamation (QSBR)", "index.html#autotoc_md65", null ]
      ] ],
      [ "Related Projects", "index.html#autotoc_md66", null ],
      [ "Contributing", "index.html#autotoc_md67", null ],
      [ "Literature", "index.html#autotoc_md68", null ]
    ] ],
    [ "Agent Bootstrap — unodb", "md_AGENTS.html", [
      [ "Scope", "md_AGENTS.html#autotoc_md1", null ],
      [ "Repository Orientation", "md_AGENTS.html#autotoc_md2", null ],
      [ "CI Process", "md_AGENTS.html#autotoc_md3", [
        [ "Before pushing (MANDATORY — no exceptions)", "md_AGENTS.html#autotoc_md4", null ],
        [ "Fork CI (MSVC + coverage) — run BEFORE pushing to upstream", "md_AGENTS.html#autotoc_md5", null ],
        [ "Upstream CI — poll after pushing", "md_AGENTS.html#autotoc_md6", null ],
        [ "Formatting", "md_AGENTS.html#autotoc_md7", null ]
      ] ],
      [ "Git Rules", "md_AGENTS.html#autotoc_md8", null ],
      [ "Push Workflow (complete sequence)", "md_AGENTS.html#autotoc_md9", null ],
      [ "Code Quality Rules (from CONTRIBUTING.md)", "md_AGENTS.html#autotoc_md10", null ],
      [ "Build Rules", "md_AGENTS.html#autotoc_md11", [
        [ "TSan and OLC Fields", "md_AGENTS.html#autotoc_md12", null ]
      ] ]
    ] ],
    [ "Agent Bootstrap — unodb", "md_CLAUDE.html", [
      [ "Scope", "md_CLAUDE.html#autotoc_md16", null ],
      [ "Repository Orientation", "md_CLAUDE.html#autotoc_md17", null ],
      [ "CI Process", "md_CLAUDE.html#autotoc_md18", [
        [ "Before pushing (MANDATORY — no exceptions)", "md_CLAUDE.html#autotoc_md19", null ],
        [ "Fork CI (MSVC + coverage) — run BEFORE pushing to upstream", "md_CLAUDE.html#autotoc_md20", null ],
        [ "Upstream CI — poll after pushing", "md_CLAUDE.html#autotoc_md21", null ],
        [ "Formatting", "md_CLAUDE.html#autotoc_md22", null ]
      ] ],
      [ "Git Rules", "md_CLAUDE.html#autotoc_md23", null ],
      [ "Push Workflow (complete sequence)", "md_CLAUDE.html#autotoc_md24", null ],
      [ "Code Quality Rules (from CONTRIBUTING.md)", "md_CLAUDE.html#autotoc_md25", null ],
      [ "Build Rules", "md_CLAUDE.html#autotoc_md26", [
        [ "TSan and OLC Fields", "md_CLAUDE.html#autotoc_md27", null ]
      ] ]
    ] ],
    [ "CONTRIBUTING", "md_CONTRIBUTING.html", [
      [ "Contributing to UnoDB", "md_CONTRIBUTING.html#autotoc_md28", [
        [ "Optional development dependencies", "md_CONTRIBUTING.html#autotoc_md29", null ],
        [ "General workflow", "md_CONTRIBUTING.html#autotoc_md30", null ],
        [ "Development CMake options", "md_CONTRIBUTING.html#autotoc_md31", null ],
        [ "Code organization", "md_CONTRIBUTING.html#autotoc_md32", null ],
        [ "Code style guide", "md_CONTRIBUTING.html#autotoc_md33", null ],
        [ "Documentation style guide", "md_CONTRIBUTING.html#autotoc_md34", null ],
        [ "Linting and static analysis", "md_CONTRIBUTING.html#autotoc_md35", null ],
        [ "Testing", "md_CONTRIBUTING.html#autotoc_md36", null ],
        [ "Fuzzing", "md_CONTRIBUTING.html#autotoc_md37", null ],
        [ "Commit messages", "md_CONTRIBUTING.html#autotoc_md38", null ],
        [ "Pull Requests", "md_CONTRIBUTING.html#autotoc_md39", null ],
        [ "Benchmarking", "md_CONTRIBUTING.html#autotoc_md40", null ],
        [ "License", "md_CONTRIBUTING.html#autotoc_md41", null ]
      ] ]
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
"namespaceanonymous__namespace_02test__key__encode__decode_8cpp_03.html#aa43163d747ae59e7046fc4349b03d30e",
"structunodb_1_1detail_1_1basic__art__policy.html#a2e809e4fd7111083184ce1b5081b6a6e",
"structunodb_1_1quiescent__state__on__scope__exit.html#a9104260779823aab63c3b3cd7d4961ab"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';