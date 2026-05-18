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
    [ "unodb", "index.html#autotoc_md25", [
      [ "Introduction", "index.html#autotoc_md26", null ],
      [ "Requirements", "index.html#autotoc_md27", [
        [ "Build dependencies", "index.html#autotoc_md28", null ],
        [ "Optional vendored dependencies, bundled as Git submodules", "index.html#autotoc_md29", null ]
      ] ],
      [ "Building", "index.html#autotoc_md30", null ],
      [ "Platform-Specific Notes", "index.html#autotoc_md31", [
        [ "Ubuntu 22.04", "index.html#autotoc_md32", null ],
        [ "Amazon Linux 2023", "index.html#autotoc_md33", null ],
        [ "Amazon Linux 2", "index.html#autotoc_md34", null ]
      ] ],
      [ "Usage", "index.html#autotoc_md35", null ],
      [ "Technical Details", "index.html#autotoc_md36", [
        [ "Adaptive Radix Tree", "index.html#autotoc_md37", null ],
        [ "Sequential Lock", "index.html#autotoc_md38", null ],
        [ "Quiescent State-Based Reclamation (QSBR)", "index.html#autotoc_md39", null ]
      ] ],
      [ "Related Projects", "index.html#autotoc_md40", null ],
      [ "Contributing", "index.html#autotoc_md41", null ],
      [ "Literature", "index.html#autotoc_md42", null ]
    ] ],
    [ "CONTRIBUTING", "md_CONTRIBUTING.html", [
      [ "Contributing to UnoDB", "md_CONTRIBUTING.html#autotoc_md2", [
        [ "Optional development dependencies", "md_CONTRIBUTING.html#autotoc_md3", null ],
        [ "General workflow", "md_CONTRIBUTING.html#autotoc_md4", null ],
        [ "Development CMake options", "md_CONTRIBUTING.html#autotoc_md5", null ],
        [ "Code organization", "md_CONTRIBUTING.html#autotoc_md6", null ],
        [ "Code style guide", "md_CONTRIBUTING.html#autotoc_md7", null ],
        [ "Documentation style guide", "md_CONTRIBUTING.html#autotoc_md8", null ],
        [ "Linting and static analysis", "md_CONTRIBUTING.html#autotoc_md9", null ],
        [ "Testing", "md_CONTRIBUTING.html#autotoc_md10", null ],
        [ "Fuzzing", "md_CONTRIBUTING.html#autotoc_md11", null ],
        [ "Commit messages", "md_CONTRIBUTING.html#autotoc_md12", null ],
        [ "Pull Requests", "md_CONTRIBUTING.html#autotoc_md13", null ],
        [ "Benchmarking", "md_CONTRIBUTING.html#autotoc_md14", null ],
        [ "License", "md_CONTRIBUTING.html#autotoc_md15", null ]
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
"classunodb_1_1db_1_1iterator.html#a4839e9e424e83bb670e2bef82c71255a",
"classunodb_1_1detail_1_1basic__inode__256.html#aada67e1b1be140b695c809ff1ebbeb44",
"classunodb_1_1detail_1_1basic__inode__impl.html#aa0833ddfc1e689707d1c487e2f34767c",
"classunodb_1_1detail_1_1key__buffer.html#afd1654d8668d711357379120f4b94dcc",
"classunodb_1_1key__encoder.html#ac8d4a2e16914a3296cb0666ff0620e72",
"classunodb_1_1optimistic__lock_1_1write__guard.html",
"classunodb_1_1qsbr__ptr.html#a56fbbdb445e458db43b61bf8ea734cc3",
"group__test-internals.html#ga32c0ae2e848a4564ebc04be693c7af15",
"namespaceunodb_1_1detail.html#aa11fc0dc02ee0b061504cd50856ea7dd",
"structunodb_1_1detail_1_1dealloc__vector__list__node.html#adb97856268b00e592400068336a02c37",
"unionunodb_1_1detail_1_1key__prefix.html#a6051ae53979f60d48223a6d98f650f0d"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';